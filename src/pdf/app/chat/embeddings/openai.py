from app.settings import get_settings
from langchain_openai import OpenAIEmbeddings

settings = get_settings()

embeddings = OpenAIEmbeddings(
    openai_api_key=settings.openai_api_key,
    model=settings.text_embedding_model,
    dimensions=512,  # Match Pinecone index dimension
)
