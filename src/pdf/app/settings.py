from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    text_embedding_model: str = "text-embedding-3-small"
    model: str = "gpt-4o"
    secret_key: str = "123"
    sqlalchemy_database_uri: str = "sqlite:///sqlite.db"
    upload_url: str = "https://prod-upload-langchain.fly.dev"
    redis_url: str = ""
    pinecone_api_key: str = ""
    pinecone_env_name: str = ""
    pinecone_index_name: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
