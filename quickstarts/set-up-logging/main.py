import os
import json
import random

from openai import OpenAI
from dotenv import load_dotenv


# Load API keys from .env file
load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_KEY"))


def calculator(operation: str, num1: int, num2: int) -> str:
    """Do arithmetic operations on two numbers."""
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2
    else:
        raise NotImplementedError("Invalid operation")


# Needed for LLM function calling
CALCULATOR_JSON_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Do arithmetic operations on two numbers.",
        "parameters": {
            "type": "object",
            "required": ["operation", "num1", "num2"],
            "properties": {
                "operation": {"type": "string"},
                "num1": {"type": "integer"},
                "num2": {"type": "integer"},
            },
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def pick_random_number():
    """Pick a random number between 1 and 100."""
    return random.randint(1, 100)


# Needed for LLM function calling
PICK_RANDOM_NUMBER_JSON_SCHEMA = {
    "type": "function",
    "function": {
        "name": "pick_random_number",
        "description": "Pick a random number between 1 and 100.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


TOPICS = ["math", "science"]
TONE = "groovy 80s surfer dude"
LLM_HYPERPARAMETERS = {
    "temperature": 0.7,
    "max_tokens": 200,
    "top_p": 1,
    "stop": "\n\n\n",
    "presence_penalty": 0.5,
    "frequency_penalty": 0.5,
    "seed": 42,
}
PROMPT_TEMPLATE = (
    "You are a helpful assistant knowledgeable on the "
    "following topics: {topics}. When you reply you "
    "should use the following tone of voice: {tone}"
)


def call_agent(messages: list[str]) -> str:
    output = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=[
            CALCULATOR_JSON_SCHEMA,
            PICK_RANDOM_NUMBER_JSON_SCHEMA,
        ],
        **LLM_HYPERPARAMETERS,
    )

    # Check if tool calls are present in the output
    if output.choices[0].message.tool_calls:
        for tool_call in output.choices[0].message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "calculator":
                result = calculator(**arguments)

            elif tool_call.function.name == "pick_random_number":
                result = pick_random_number(**arguments)

            else:
                raise NotImplementedError("Invalid tool call")

            return f"[TOOL CALL] {result}"

    # Otherwise, return the LLM response
    return output.choices[0].message.content


if __name__ == "__main__":
    messages = [
        {
            "role": "system",
            "content": PROMPT_TEMPLATE.format(
                topics=" ".join(TOPICS),
                tone=TONE,
            ),
        },
    ]
    while True:
        user_input = input("You: ")
        input_output = [user_input]
        if user_input == "exit":
            # Exit the demo
            break
        messages.append({"role": "user", "content": user_input})
        response = call_agent(messages=messages)
        messages.append({"role": "assistant", "content": str(response)})
        print(f"Agent: {response}")
