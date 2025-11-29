import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from src.settings import get_settings


class SimpleChatHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, history_dir: str = "chat_histories"):
        self.session_id = session_id
        self.history_dir = history_dir
        self.messages = []
        self.history_dir = os.path.join(
            f"src/tchat/{history_dir}", f"{session_id}.json"
        )

        os.makedirs(os.path.dirname(self.history_dir), exist_ok=True)

        self.load_messages()

    def load_messages(self):
        if os.path.exists(self.history_dir):
            with open(self.history_dir, "r") as f:
                try:
                    for msg in json.load(f):
                        if msg["type"] == "human":
                            msg = HumanMessage(content=msg["content"])
                            self.messages.append(msg)
                        elif msg["type"] == "ai":
                            msg = AIMessage(content=msg["content"])
                            self.messages.append(msg)
                except Exception:
                    print("Error loading message history, starting fresh.")

    def save_messages(self) -> None:
        try:
            data = []
            for msg in self.messages:
                if isinstance(msg, HumanMessage):
                    data.append({"type": "human", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    data.append({"type": "ai", "content": msg.content})
            with open(self.history_dir, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving message history: {e}")

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        self.save_messages()

    def clear(self) -> None:
        self.messages = []


settings = get_settings()

# Create chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("placeholder", "{history}"),
        ("human", "{content}"),
    ]
)

chat = ChatOpenAI(
    model_name=settings.model,
    openai_api_key=settings.openai_api_key,
    temperature=0,
    max_retries=3,
)

# Create chain using pipe operator
chain = prompt | chat

# Store message history
store = {}


def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = SimpleChatHistory(session_id)
    return store[session_id]


# Wrap chain with history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="content",
    history_messages_key="history",
    append_output_messages=True,
)

session_id = "user_session"
while True:
    content = input(">> ")
    response = chain_with_history.invoke(
        {"content": content},
        config={"configurable": {"session_id": session_id}},
    )
    print(response.content)
