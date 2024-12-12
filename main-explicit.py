import os
import json
import datetime

from humanloop import Humanloop
from openai import OpenAI

openai = OpenAI(api_key=os.getenv("OPENAI_KEY"))
humanloop = Humanloop(
    api_key=os.getenv("HUMANLOOP_KEY"),
    base_url="http://0.0.0.0:80/v5",
)


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
        return "Invalid operation"


TOOL_JSON_SCHEMA = {
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
}


def call_model(trace_id: str, messages: list[str]) -> str:
    prompt_start_time = datetime.datetime.now()
    output = openai.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        tools=[
            {
                "type": "function",
                "function": TOOL_JSON_SCHEMA,
            }
        ],
        temperature=0.7,
    )
    prompt_log_id = humanloop.prompts.log(
        path="Logging Quickstart/QA Prompt",
        prompt={
            "model": "gpt-4o",
            "tools": [TOOL_JSON_SCHEMA],
            "temperature": 0.7,
        },
        output=output.choices[0].message.content,
        trace_parent_id=trace_id,
        start_time=prompt_start_time,
        end_time=datetime.datetime.now(),
    ).id

    # Check if model asked for a tool call
    if output.choices[0].message.tool_calls:
        for tool_call in output.choices[0].message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "calculator":
                tool_start_time = datetime.datetime.now()
                result = calculator(**arguments)
                humanloop.tools.log(
                    path="Logging Quickstart/Calculator",
                    tool={
                        "name": "calculator",
                        "description": "Do arithmetic operations on two numbers.",
                        "function": TOOL_JSON_SCHEMA,
                    },
                    inputs=arguments,
                    output=result,
                    trace_parent_id=prompt_log_id,
                    start_time=tool_start_time,
                    end_time=datetime.datetime.now(),
                )
                return f"[TOOL CALL] {result}"

    # Otherwise, return the LLM response
    return output.choices[0].message.content


def conversation():
    trace_id = humanloop.flows.log(
        path="Logging Quickstart/QA Agent",
        flow={
            "attributes": {},
        },
    ).id
    messages = [
        {
            "role": "system",
            "content": "You are a a groovy 80s surfer dude "
            "helping with math and science.",
        },
    ]
    while True:
        user_input = input("You: ")
        if user_input == "exit":
            break
        messages.append({"role": "user", "content": user_input})
        response = call_model(trace_id=trace_id, messages=messages)
        messages.append({"role": "assistant", "content": response})
        print(f"Agent: {response}")

    humanloop.flows.update_log(
        log_id=trace_id,
        output="",
        trace_status="complete",
    )


if __name__ == "__main__":
    conversation()
