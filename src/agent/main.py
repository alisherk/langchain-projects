from typing import Any, Dict

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from src.agent.sql import list_tables, run_sqlite_query, write_file
from src.settings import get_settings


# Custom callback handler for detailed output
class CustomCallbackHandler(BaseCallbackHandler):
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        tool_name = serialized.get("name", "Unknown")
        print(f"\n🔧 TOOL CALLED: {tool_name}")
        print(
            f"📥 Input: {input_str[:200]}..."
            if len(input_str) > 200
            else f"📥 Input: {input_str}"
        )

    def on_tool_end(self, output: str, **kwargs) -> None:
        # Convert output to string if it's not already
        output_str = str(output) if output else ""
        print(
            f"📤 Output: {output_str[:200]}..."
            if len(output_str) > 200
            else f"📤 Output: {output_str}"
        )
        print("-" * 80)

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        print(f"❌ Tool Error: {error}")

    def on_llm_start(self, serialized: Dict[str, Any], prompts: list, **kwargs) -> None:
        print("\n🤖 LLM THINKING...")

    def on_llm_end(self, response, **kwargs) -> None:
        print("✅ LLM Response received")
        print(response)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        print("\n⛓️  CHAIN STARTED")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        print("⛓️  CHAIN COMPLETED\n")

    def on_agent_action(self, action, **kwargs) -> None:
        print(f"\n🎯 AGENT ACTION: {action.tool}")
        print(
            f"💭 Reasoning: {action.log[:150]}..."
            if len(action.log) > 150
            else f"💭 Reasoning: {action.log}"
        )


settings = get_settings()

# 1. Create the language model
model = ChatOpenAI(
    model_name=settings.model,
    openai_api_key=settings.openai_api_key,
    temperature=0,
    max_retries=3,
    request_timeout=60,
)

# 2. Create the list of tools
tools = [list_tables, run_sqlite_query, write_file]

# 3. Create the agent with a simple system prompt string
system_prompt = """You are an AI that has access to a SQLite database. You have access to the following tools: list_tables, run_sqlite_query, and write_file.

Database Schema:
- users: id, name, email
- addresses: id, user_id, street, city
- products: id, name, price
- carts: id, user_id
- orders: id, user_id, total
- order_products: id, order_id, product_id, amount

Important: The products table uses 'id' as primary key, and order_products uses 'amount' for quantity."""

# 4. Create the agent
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
)

# 5. Define a function to ask questions to the agent
conversation_history = []


def ask(question: str) -> str:
    conversation_history.append({"role": "user", "content": question})
    print(f"\n{'=' * 80}")
    print(f"❓ QUESTION: {question}")
    print(f"{'=' * 80}")

    result = agent.invoke(
        {"messages": conversation_history},
        config={"callbacks": [CustomCallbackHandler()]},
    )
    conversation_history.append(
        {"role": "assistant", "content": result["messages"][-1].content}
    )

    print(f"\n{'=' * 80}")
    print("💬 FINAL ANSWER:")
    print(f"{'=' * 80}\n")

    return result["messages"][-1].content


# 6. Now you can use it simply:
task1 = ask(
    "Summarize top 5 most popular products and create a html report and save as a file report.html at src/agent path."
)
print(task1)

task2 = ask(
    "inspect our report.html file, and show us the first product in the report."
)
print(task2)
