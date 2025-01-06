"""This script demonstrates instrumenting a simple conversational agent with function calling.

The example uses the Humanloop SDK to declare Files in code.

Type 'exit' to end the conversation.
"""

import json
import os
import random

from dotenv import load_dotenv
from humanloop import Humanloop
from openai import OpenAI

load_dotenv()


DIRECTORY = "SDK/Surfer Agent"
if DIRECTORY_PREFIX := os.getenv("DIRECTORY_PREFIX"):
    DIRECTORY = f"{DIRECTORY_PREFIX}/{DIRECTORY}"


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
    "You are a helpful assistant knowledgeable on the following topics: {topics}. "
    "When you reply you should use the following tone of voice: {tone}"
)

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

humanloop = Humanloop(api_key=os.getenv("HUMANLOOP_KEY"))


@humanloop.tool(path=f"{DIRECTORY}/Calculator")
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


@humanloop.tool(path=f"{DIRECTORY}/Random Number")
def pick_random_number():
    """Pick a random number between 1 and 100."""
    return random.randint(1, 100)


@humanloop.prompt(
    path=f"{DIRECTORY}/Agent Prompt",
    template=PROMPT_TEMPLATE,
    tools=[
        pick_random_number.json_schema,
        calculator.json_schema,
    ],
)
def call_agent(messages: list[str]) -> str:
    output = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        # Use .json_schema property on decorated functions to easily access
        # the definition for function calls
        tools=[
            {
                "type": "function",
                "function": calculator.json_schema,
            },
            {
                "type": "function",
                "function": pick_random_number.json_schema,
            },
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

            return f"[TOOL CALL: {tool_call.function.name}] {result}"

    return output.choices[0].message.content


@humanloop.flow(path=f"{DIRECTORY}/Agent Flow")
def agent_chat_workflow():
    messages = [
        {
            "role": "system",
            "content": PROMPT_TEMPLATE.format(
                topics=" ".join(TOPICS),
                tone=TONE,
            ),
        },
    ]
    input_output_pairs = []
    while True:
        user_input = input("You: ")
        input_output = [user_input]
        if user_input == "exit":
            break
        messages.append({"role": "user", "content": user_input})
        response = call_agent(messages=messages)
        messages.append({"role": "assistant", "content": str(response)})
        input_output.append(str(response))
        print(f"Agent: {response}")
        input_output_pairs.append(input_output)
    return json.dumps(input_output_pairs)


if __name__ == "__main__":
    agent_chat_workflow()
