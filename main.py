import os
from humanloop import Humanloop
from openai import OpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage as Message
import wikipedia
import json


openai = OpenAI(
    api_key=os.getenv("OPENAI_KEY"),
)
humanloop = Humanloop(
    api_key=os.getenv("HUMANLOOP_KEY"),
    base_url="http://0.0.0.0:80/v5",
)


@humanloop.tool(path="QA Agent/Search Wikipedia")
def search_wikipedia(query: str) -> dict:
    """Search Wikipedia to get up-to-date information for a query."""
    try:
        page = wikipedia.page(query)
        return {
            "title": page.title,
            "content": page.content,
            "url": page.url,
        }
    except Exception as _:
        return {
            "title": "",
            "content": "No results found",
            "url": "",
        }


@humanloop.prompt(path="QA Agent/Prompt")
def call_model(messages: list[Message]) -> Message:
    """Calls the model with the given messages"""
    system_message = {
        "role": "system",
        "content": (
            "You are an assistant that help to answer user questions. You should "
            "leverage wikipedia to answer questions so that the information is up to date. "
            "If the response from Wikipedia does not seem relevant, rephrase the question "
            "and call the tool again. Then finally respond to the user."
            "Formulate the response so that it is easy to understand for a 5 year old."
        ),
    }
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[system_message] + messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_wikipedia",
                    "description": "Search the internet to get up to date answers for a query.",
                    "parameters": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                },
            }
        ],
    )
    return response.choices[0].message.to_dict(exclude_unset=False)


@humanloop.flow(path="QA Agent/Agent")
def call_agent(query: str) -> str:
    """Calls the main agent loop and returns the final result"""
    messages = [{"role": "user", "content": query}]
    # Retry for a relevant response 3 times at most
    for _ in range(3):
        response = call_model(messages)
        messages.append(response)
        if response["tool_calls"]:
            # Call wikipedia to get up-to-date information
            for tool_call in response["tool_calls"]:
                source = search_wikipedia(
                    **json.loads(tool_call["function"]["arguments"])
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(source),
                        "tool_call_id": tool_call["id"],
                    }
                )
        else:
            # Respond to the user
            return response["content"]


if __name__ == "__main__":

    def easy_to_understand(log):
        return len(log["output"]) < 100

    # Read the evaluation dataset
    with open("dataset.jsonl", "r") as fp:
        dataset = [json.loads(line) for line in fp]

    humanloop.evaluations.run(
        name="QA Agent Answer Comprehensiveness",
        file={
            "path": "QA Agent/Agent",
            "callable": call_agent,
        },
        evaluators=[
            {
                "path": "QA Agent/Comprehension",
                "callable": easy_to_understand,
                "args_type": "target_free",
                "return_type": "boolean",
            }
        ],
        dataset={
            "path": "QA Agent/Children Questions",
            "datapoints": dataset,
        },
        workers=8,
    )
