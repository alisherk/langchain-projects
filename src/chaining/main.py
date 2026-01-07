from langchain_core.prompts import PromptTemplate
from openai import OpenAI

from src.pdf.app.settings import get_settings

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)

# Chain 1: Create a function
code_prompt = PromptTemplate(
    input_variables=["function"],
    template="Write a Python function called {function}",
)

# Chain 2: Generate test case
test_prompt = PromptTemplate(
    input_variables=["code"],
    template="Write a test case for the following code:\n{code}",
)

# Execute Chain 1
formatted_code = code_prompt.format(function="add_numbers")
response_code = client.chat.completions.create(
    model=settings.model,
    messages=[{"role": "user", "content": formatted_code}],
    max_tokens=200,
)

generated_code = response_code.choices[0].message.content
print("Generated Code:\n", generated_code)

# Execute Chain 2
formatted_test = test_prompt.format(code=generated_code)
response_test = client.chat.completions.create(
    model=settings.model,
    messages=[{"role": "user", "content": formatted_test}],
    max_tokens=200,
)
generated_test = response_test.choices[0].message.content
print("Generated Test Case:\n", generated_test)
