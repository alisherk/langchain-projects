from app.chat.embeddings.openai import embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from app.settings import get_settings

settings = get_settings()

pc = Pinecone(api_key=settings.pinecone_api_key, environment=settings.pinecone_env_name)

index = pc.Index(settings.pinecone_index_name)

vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
