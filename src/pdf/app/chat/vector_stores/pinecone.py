from app.chat.embeddings.openai import embeddings
from app.chat.models import ChatArgs
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from app.settings import get_settings

settings = get_settings()

pc = Pinecone(api_key=settings.pinecone_api_key, environment=settings.pinecone_env_name)

index = pc.Index(settings.pinecone_index_name)

vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

def build_retriever(chat_args: ChatArgs):
    search_kwargs = {"filter": { "pdf_id": chat_args.pdf_id }}

    return vectorstore.as_retriever(search_kwargs=search_kwargs)