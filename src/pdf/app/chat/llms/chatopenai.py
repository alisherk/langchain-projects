from app.settings import get_settings
from langchain_openai import ChatOpenAI
from app.chat.models import ChatArgs

settings = get_settings()

def build_llm(chat_args: ChatArgs) -> ChatOpenAI:
    return ChatOpenAI(
        model_name=settings.model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
        max_retries=3,
        streaming=chat_args.streaming,
    )
