from functools import partial
from typing import Dict, List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

# Global store to persist window memories across requests
_window_memory_store: Dict[str, "WindowMessageHistory"] = {}


class WindowMessageHistory(BaseChatMessageHistory):
    """In-memory message history that keeps only the last k messages."""

    def __init__(self, conversation_id: str, k: int = 5):
        self.conversation_id = conversation_id
        self.k = k
        self._messages: List[BaseMessage] = []

    @property
    def messages(self) -> List[BaseMessage]:
        """Return the last k messages."""
        return self._messages[-self.k :]

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the store."""
        self._messages.append(message)

    def clear(self) -> None:
        """Clear all messages."""
        self._messages = []


def build_window_memory(chat_args, k: int = 5):
    """Returns window history function that persists across requests."""

    def history_func(conversation_id: str) -> WindowMessageHistory:
        # Check if we already have a history for this conversation
        key = f"{conversation_id}_{k}"
        if key not in _window_memory_store:
            _window_memory_store[key] = WindowMessageHistory(
                conversation_id=conversation_id, k=k
            )
        return _window_memory_store[key]

    return history_func


window_memory_registry = {
    "window_memory": partial(build_window_memory, k=5),  # Default with 5 messages
}