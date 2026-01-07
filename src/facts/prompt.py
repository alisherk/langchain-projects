from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.pdf.app.settings import get_settings
from src.facts.filter_retriever import FilterRetriever

settings = get_settings()

embeddings = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
db = Chroma(persist_directory="src/facts/chroma_db", embedding_function=embeddings)

retrieval = FilterRetriever(db=db, embeddings=embeddings)

llm = ChatOpenAI(
    model_name=settings.model, 
    openai_api_key=settings.openai_api_key, 
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
"""
Use the following context to answer the question.
Context:
{context}

Question: 
{input}

Answer:
""")

# Create retrieval chain using LCEL (LangChain Expression Language)
retrieval_chain = (
    {"context": retrieval, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = retrieval_chain.invoke("What is an interesting fact about the English language?")

print("Result:", result)

##this can also be done with postgres like so:
'''
from langchain_postgres import PostgresVectorStore
from langchain_postgres.vectorstore import HierarchicalFakeSplitter
from langchain_openai import OpenAIEmbeddings

# Setup
connection_string = "postgresql://user:password@localhost/dbname"
embeddings = OpenAIEmbeddings()

# Create vector store
vector_store = PostgresVectorStore(
    connection_string=connection_string,
    embedding_function=embeddings,
    collection_name="my_facts"
)

# Add documents
vector_store.add_documents(documents)

# Retrieve
retriever = vector_store.as_retriever()
'''