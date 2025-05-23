# RAG POC with LangGraph

This project demonstrates a Retrieval-Augmented Generation (RAG) implementation using LangGraph to orchestrate interactions between an agent, a retriever, a document grader, a query rewriter, and a generator. It leverages Lilian Weng's blog posts for content.

## Key Components

- **LangGraph:** Orchestrates the workflow.
- **Google Gemini API:** Powers the LLM and embeddings.
- **CheerioWebBaseLoader:** Loads content from web URLs.
- **RecursiveCharacterTextSplitter:** Splits documents into chunks.
- **MemoryVectorStore:** Stores and retrieves document embeddings.

## Workflow

1.  **Agent:** Receives the initial query.
2.  **Retriever:** Fetches relevant documents.
3.  **Document Grader:** Assesses document relevance.
4.  **Rewriter:** Reformulates the query if needed.
5.  **Generator:** Generates the final answer.

## Setup

1.  **Install dependencies:** `npm install`
2.  **Configure environment variables:** Create a `.env` file and set `GOOGLE_API_KEY`.
3.  **Run the application:** `npm start`
