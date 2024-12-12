import json
import os

from humanloop import Humanloop
from openai import OpenAI

import wikipedia


openai = OpenAI(api_key=os.getenv("OPENAI_KEY"))
humanloop = Humanloop(api_key=os.getenv("HUMANLOOP_KEY"))


def search_wikipedia(query: str) -> dict:
    """Search the internet to get data sources for a query."""

    try:
        # Let Wikipedia suggest a relevant page
        page = wikipedia.page(query)
        return {
            "title": page.title,
            "content": page.content,
            "url": page.url,
        }
    except Exception:
        return {
            "title": "",
            "content": "No results found",
            "url": "",
        }


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_wikipedia",
        "description": "Search the internet to get data sources for a query.",
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


def agent(messages: list[dict], tool_call: bool) -> str:
    args = {
        "messages": messages,
        "model": "gpt-4o",
        "temperature": 0,
    }
    if tool_call:
        args["tools"] = [TOOL_SCHEMA]
    response = openai.chat.completions.create(**args)
    return {
        "content": response.choices[0].message.content,
        "tool_calls": [
            tool_call.to_dict()
            for tool_call in response.choices[0].message.tool_calls or []
        ],
    }


def workflow(question: str) -> str:
    source = None
    messages = [
        {
            "role": "system",
            "content": (
                "You must find a good source to answer a "
                "question using the provided Wikipedia tool."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    # Maximum 3 attempts to find a relevant source
    for _ in range(3):
        response = agent(messages, tool_call=True)

        if "FINISHED" in (response["content"] or "") or not response["tool_calls"]:
            # Model chose a source to answer the question
            break

        tool_call = response["tool_calls"][0]
        arguments = json.loads(tool_call["function"]["arguments"])
        source = search_wikipedia(**arguments)
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": f"Found a source called {source['title']}",
                },
                {
                    "role": "user",
                    "content": (
                        "Is this a relevant source? Output 'FINISHED' "
                        "or rephrase the question to essential "
                        "terms and call the tool again to get a "
                        "new information source."
                    ),
                },
            ]
        )

    source = source or {
        "title": "",
        "content": "No relevant source found",
        "url": "",
    }
    answer = agent(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a summarizer. Answer "
                    "the question based on this source: "
                    f"{source['content']}"
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        tool_call=False,
    )["content"]
    return f"{answer}\n\nSource: {source['url']}"


if __name__ == "__main__":
    humanloop.evaluators.upsert(
        path="QA Agent/Comprehension",
        spec={
            "arguments_type": "target_free",
            "return_type": "number",
            "evaluator_type": "llm",
            "prompt": {
                "model": "gpt-4o",
                "endpoint": "complete",
                "template": (
                    "You must decide if an explanation is simple "
                    "enough to be understood by a 5-year old. "
                    "A better explanation is shorter and uses less jargon "
                    "Rate the answer from 1 to 10, where 10 is the best.\n"
                    "\n<Question>\n{{log.inputs.question}}\n</Question>\n\n"
                    "\n<Answer>\n{{log.output}}</Answer>\n\n"
                    "First provide your rationale, then on a newline, output your judgment."
                ),
                "provider": "openai",
                "temperature": 0,
            },
        },
    )

    # Read the evaluation dataset
    with open("dataset.jsonl", "r") as fp:
        dataset = [json.loads(line) for line in fp]
    humanloop.evaluations.run(
        name="QA Agent Answer Comprehensiveness",
        file={
            "path": "QA Agent/Workflow",
            "callable": workflow,
        },
        evaluators=[
            {
                "path": "QA Agent/Comprehension",
            }
        ],
        dataset={
            "path": "QA Agent/Children Comprehension",
            "datapoints": dataset,
        },
        workers=8,
    )
