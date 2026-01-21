from app.chat.llms.chatopenai import build_llm
from app.chat.memories.sql_memory import get_sql_session_history
from app.chat.models import ChatArgs
from app.chat.vector_stores.pinecone import build_retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Build a RAG (Retrieval-Augmented Generation) chain with conversation history
#
# Flow when invoked:
# 1. Loads past messages from SQL database for the given conversation_id
# 2. Retrieves relevant PDF chunks from Pinecone based on the user's question
# 3. Builds a prompt with: system instructions + retrieved context + conversation history + current question
# 4. Sends the complete prompt to ChatGPT
# 5. Returns the response as a string
# 6. Automatically saves both the question and response to the database
#
# Usage:
#   chain.invoke(
#       {"input": "What is this about?"},
#       config={"configurable": {"session_id": conversation_id}}
#   )
def build_chat(chat_args: ChatArgs):
    retriever = build_retriever(chat_args)
    llm = build_llm(chat_args)

    # Create a RAG prompt with message history
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the following context to answer the user's question:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # Build the RAG chain: retrieve context -> format prompt -> LLM -> parse output
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["input"])), # retrieve relevant docs from pinecone and format as text
            "input": lambda x: x["input"], # pass through user input
            "history": lambda x: x.get("history", []), # load past messages from SQL memory
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap the RAG chain with message history
    chain_with_history = RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=get_sql_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    return chain_with_history

"""
in the end our prompt looks like this:
System: You are a helpful assistant. Use the following context to answer the user's question:
System: You are a helpful assistant. Use the following context...

chunk 1. <text from relevant PDF in pinecone>

chunk 2. <more text from relevant PDF in pinecone>

chunk 3. <more text from relevant PDF in pinecone>

History: [past messages]
Human: What is X?
"""
