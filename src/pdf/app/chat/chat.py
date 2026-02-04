from app.chat.llms.chatopenai import llm_registry
from app.chat.memories.memory_registry import memory_registry
from app.chat.models import ChatArgs
from app.chat.score import get_random_component_by_score
from app.chat.vector_stores.pinecone import retriever_registry
from app.web.api import get_conversation_components, set_conversation_components
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


def select_component(component_registry, chat_args: ChatArgs, component_type: str):
    components = get_conversation_components(chat_args.conversation_id)
    previous_component = components[component_type]

    if previous_component:
        component = component_registry[previous_component](chat_args)
        return previous_component, component
    else:
        random_component_name = get_random_component_by_score(
            component_registry, component_type
        )
        component = component_registry[random_component_name](chat_args)
        return random_component_name, component


def build_chat(chat_args: ChatArgs):
    retriever_name, retriever = select_component(
        retriever_registry, chat_args, "retriever"
    )

    llm_name, llm = select_component(llm_registry, chat_args, "llm")

    memory_name, memory = select_component(memory_registry, chat_args, "memory")

    print(f"Using LLM: {llm_name}, Retriever: {retriever_name}, Memory: {memory_name}")

    # Print message history for debugging
    try:
        history = memory(str(chat_args.conversation_id))
        messages = history.messages
        print(f"\n=== Message History (Conversation {chat_args.conversation_id}) ===")
        for i, msg in enumerate(messages):
            print(f"{i + 1}. [{msg.type}]: {msg.content}")
        print(f"=== Total messages: {len(messages)} ===\n")
    except Exception as e:
        print(f"Could not load history: {e}")

    set_conversation_components(
        conversation_id=chat_args.conversation_id,
        llm=llm_name,
        retriever=retriever_name,
        memory=memory_name,
    )

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
            "context": lambda x: format_docs(
                retriever.invoke(x["input"])
            ),  # retrieve relevant docs from pinecone and format as text
            "input": lambda x: x["input"],  # pass through user input
            "history": lambda x: x.get(
                "history", []
            ),  # load past messages from SQL memory
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap the RAG chain with message history
    chain_with_history = RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=memory,
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
