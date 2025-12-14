import langchain
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.agent.sql import run_sqlite_query, list_tables
from src.settings import get_settings

langchain.verbose = True
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
tools = [list_tables, run_sqlite_query]


# 3. Create the agent with a simple system prompt string
system_prompt = "You are an AI that has access to a SQLite database. You have access to the following tools: list_tables and run_sqlite_query."

# 4. Create the agent
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
)

# 5. Define a function to ask questions to the agent
def ask(question: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


# 6. Now you can use it simply:
answer = ask("Summarize top 5 most popular products and create a html report and save as a file report.html.")
print(answer)
