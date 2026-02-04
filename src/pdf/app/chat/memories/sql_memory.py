from typing import List

from app.web.api import add_message_to_conversation, get_messages_by_conversation_id
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage


class SQLMessageHistory(BaseChatMessageHistory):
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id

    def add_message(self, message) -> None:
        return add_message_to_conversation(
            self.conversation_id, message.type, message.content
        )

    @property
    def messages(self) -> List[BaseMessage]:
        return get_messages_by_conversation_id(self.conversation_id)

    def clear(self) -> None:
        pass


def build_sql_memory(chat_args):
    """Returns SQL session history function for RunnableWithMessageHistory."""

    def get_session_history(conversation_id: str) -> SQLMessageHistory:
        return SQLMessageHistory(conversation_id=conversation_id)

    return get_session_history


# SQL memory registry
sql_memory_registry = {
    "sql_memory": build_sql_memory,
}
