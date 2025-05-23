import { config } from "dotenv";
config();
import { createRetrieverTool } from "langchain/tools/retriever";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { START, END, Annotation, StateGraph } from "@langchain/langgraph";
import { pull } from "langchain/hub";
import { z } from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createInterface } from "readline";

// Define the state graph
const GraphState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
        reducer: (x, y) => x.concat(y),
        default: () => [],
    })
});

// Define multiple URLs for knowledge base
const urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://wyag.thb.lt/"
];

// Load and process documents
const docs = await Promise.all(
    urls.map((url) => new CheerioWebBaseLoader(url, {
        selector: "p",
        timeout: 600000
    }).load())
);
const docsList = docs.flat();

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
});
const docSplits = await textSplitter.splitDocuments(docsList);

// Initialize vector store with embeddings
const vectorStore = await MemoryVectorStore.fromDocuments(
    docSplits,
    new GoogleGenerativeAIEmbeddings()
);

const retriever = vectorStore.asRetriever();

// Create retriever tool
const tool = createRetrieverTool(retriever, {
    name: "retrieve_information",
    description: "Search and retrieve relevant information from the knowledge base.",
});
const tools = [tool];

const toolNode = new ToolNode<typeof GraphState.State>(tools);


/**
 * Decides whether the agent should retrieve more information or end the process.
 * @param {typeof GraphState.State} state - The current state of the agent.
 * @returns {string} - A decision to either "continue" the retrieval process or "end" it.
 */
function shouldRetrieve(state: typeof GraphState.State): string {
    const { messages } = state;
    const lastMessage = messages[messages.length - 1];

    if ("tool_calls" in lastMessage && Array.isArray(lastMessage.tool_calls) && lastMessage.tool_calls.length) {
        return "retrieve";
    }
    process.stdout.write('Bot: ');
    process.stdout.write(String(lastMessage.content));
    return END;
}

/**
 * Determines whether the Agent should continue based on the relevance of retrieved documents.
 * @param {typeof GraphState.State} state - The current state of the agent.
 * @returns {Promise<Partial<typeof GraphState.State>>} - The updated state.
 */
async function gradeDocuments(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
    const { messages } = state;
    const tool = {
        name: "give_relevance_score",
        description: "Give a relevance score to the retrieved documents.",
        schema: z.object({
            binaryScore: z.string().describe("Relevance score 'yes' or 'no'"),
        })
    }

    const prompt = ChatPromptTemplate.fromTemplate(
        `You are a grader assessing relevance of retrieved docs to a user question.
    Here are the retrieved docs:
    \n ------- \n
    {context} 
    \n ------- \n
    Here is the user question: {question}
    If the content of the docs are relevant to the users question, score them as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the docs are relevant to the question.
    Yes: The docs are relevant to the question.
    No: The docs are not relevant to the question.`,
    );

    const model = new ChatGoogleGenerativeAI({
        model: "gemini-2.0-flash",
        temperature: 0,
        apiKey: process.env.GOOGLE_API_KEY,
    }).bindTools([tool], {
        tool_choice: tool.name,
    });

    const chain = prompt.pipe(model);

    const lastMessage = messages[messages.length - 1];

    const score = await chain.invoke({
        question: messages[0].content as string,
        context: lastMessage.content as string,
    });

    return {
        messages: [score]
    };
}

/**
 * Check the relevance of the previous LLM tool call.
 * @param {typeof GraphState.State} state - The current state of the agent.
 * @returns {string} - A directive to either "yes" or "no" based on the relevance of the documents.
 */
function checkRelevance(state: typeof GraphState.State): string {
    const { messages } = state;
    const lastMessage = messages[messages.length - 1];
    if (!("tool_calls" in lastMessage)) {
        throw new Error("The 'checkRelevance' node requires the most recent message to contain tool calls.")
    }
    const toolCalls = (lastMessage as AIMessage).tool_calls;
    if (!toolCalls || !toolCalls.length) {
        throw new Error("Last message was not a function message");
    }

    if (toolCalls[0].args.binaryScore === "yes") {
        return "yes";
    }
    return "no";
}

/**
 * Invokes the agent model to generate a response based on the current state.
 * @param {typeof GraphState.State} state - The current state of the agent.
 * @returns {Promise<Partial<typeof GraphState.State>>} - The updated state.
 */
async function agent(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
    const { messages } = state;
    // Find the AIMessage which contains the `give_relevance_score` tool call, and remove it.
    const filteredMessages = messages.filter((message) => {
        if ("tool_calls" in message && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
            return message.tool_calls[0].name !== "give_relevance_score";
        }
        return true;
    });

    const model = new ChatGoogleGenerativeAI({
        model: "gemini-2.0-flash",
        temperature: 0,
        streaming: true,
        apiKey: process.env.GOOGLE_API_KEY,
    }).bindTools(tools);

    const response = await model.invoke(filteredMessages);
    return {
        messages: [response],
    };
}

/**
 * Transform the query to produce a better question.
 * @param {typeof GraphState.State} state - The current state of the agent.
 * @returns {Promise<Partial<typeof GraphState.State>>} - The updated state.
 */
async function rewrite(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
    const { messages } = state;
    const question = messages[0].content as string;
    const prompt = ChatPromptTemplate.fromTemplate(
        `Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question:`,
    );

    const model = new ChatGoogleGenerativeAI({
        model: "gemini-2.0-flash",
        temperature: 0,
        streaming: true,
        apiKey: process.env.GOOGLE_API_KEY,
    });
    const response = await prompt.pipe(model).invoke({ question });
    return {
        messages: [response],
    };
}


async function generate(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
    const { messages } = state;
    const question = messages[0].content as string;
    const lastToolMessage = messages.slice().reverse().find((msg) => msg.getType() === "tool");

    if (!lastToolMessage) {
        throw new Error("No tool message found in the conversation history");
    }

    const docs = lastToolMessage.content as string;
    const prompt = await pull<ChatPromptTemplate>("rlm/rag-prompt");

    const llm = new ChatGoogleGenerativeAI({
        model: "gemini-2.0-flash",
        temperature: 0,
        apiKey: process.env.GOOGLE_API_KEY,
        streaming: true,
    });

    const ragChain = prompt.pipe(llm);

    const response = await ragChain.invoke({
        context: docs,
        question,
    });

    process.stdout.write('\nBot: ');
    process.stdout.write(String(response.content) + '\n\n');

    return {
        messages: [response],
    };
}

// Define the graph
const workflow = new StateGraph(GraphState)
    .addNode("agent", agent)
    .addNode("retrieve", toolNode)
    .addNode("gradeDocuments", gradeDocuments)
    .addNode("rewrite", rewrite)
    .addNode("generate", generate);

workflow.addEdge(START, "agent");

// Decide whether to retrieve or not
workflow.addConditionalEdges(
    "agent",
    shouldRetrieve,
);

workflow.addEdge("retrieve", "gradeDocuments");

workflow.addConditionalEdges(
    "gradeDocuments",
    checkRelevance,
    {
        yes: "generate",
        no: "rewrite", // placeholder
    },
);

workflow.addEdge("generate", END);
workflow.addEdge("rewrite", "agent");

// Compile graph
const app = workflow.compile();

const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
});

// Store the chat history
let chatHistory: BaseMessage[] = [];

const chatWithAgent = async (newMessage: string) => {
    // Add the user's message to the chat history
    chatHistory.push(new HumanMessage(newMessage));

    const inputs = {
        messages: chatHistory,
    };

    // Run the graph and get the output
    const stream = await app.stream(inputs);
    for await (const output of stream) {
        for (const [key, value] of Object.entries(output)) {
            const lastMsg = output[key].messages[output[key].messages.length - 1];

            // Add all messages from the node to the chat history
            chatHistory.push(lastMsg);
        }
    }
}

async function startChat() {
    console.log('Chatbot started! Type "exit" to quit.');
    const askQuestion = async () => {
        rl.question("You: ", async (input) => {
            if (input.toLowerCase() === "exit") {
                console.log("Goodbye!");
                rl.close();
                return;
            }
            await chatWithAgent(input);
            askQuestion();
        });
    };
    askQuestion();
}

startChat();