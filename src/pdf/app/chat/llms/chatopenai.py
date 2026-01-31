from app.settings import get_settings
from langchain_openai import ChatOpenAI
from app.chat.models import ChatArgs
from functools import partial

settings = get_settings()

def build_llm(chat_args: ChatArgs, model_name: str) -> ChatOpenAI:
    return ChatOpenAI(
        model_name=model_name,
        openai_api_key=settings.openai_api_key,
        temperature=0,
        max_retries=3,
        streaming=chat_args.streaming,
    )

llm_registry = {
    "gpt-3.5-turbo": partial(build_llm, model_name="gpt-3.5-turbo"),
    "gpt-4o": partial(build_llm, model_name="gpt-4o"),
    "gpt-5": partial(build_llm, model_name="gpt-5"),
}