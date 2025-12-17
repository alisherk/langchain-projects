from langchain_openai import OpenAIEmbeddings

from src.settings import get_settings

settings = get_settings()

embeddings = OpenAIEmbeddings(
    openai_api_key=settings.openai_api_key
)