"""
This script demonstrates how to build an agentic RAG pipeline using Humanloop and OpenAI's API.

The model can choose to call the retrieval tool, or instead ask for clarification from the user.

This leverages tool calling: https://platform.openai.com/docs/guides/function-calling

It assumes that Prompts and Tools are managed in the client's code and logging and evals are done via Humanloop.
"""

import json
from dotenv import load_dotenv
import os
from chromadb import chromadb
from openai import OpenAI
import pandas as pd
from datetime import datetime
from humanloop import Humanloop

# Load .env file that contains API keys
load_dotenv()

# Init clients
hl = Humanloop(api_key=os.getenv("HUMANLOOP_KEY"))
chroma = chromadb.Client()
openai = OpenAI(api_key=os.getenv("OPENAI_KEY"))

# Init the knowledge base - use whatever vector DB you have
print("Populating knowledge base")
collection = chroma.get_or_create_collection(name="MedQA")
knowledge_base = pd.read_parquet("../../assets/sources/textbooks.parquet")
knowledge_base = knowledge_base.sample(10, random_state=42)
collection.add(
    documents=knowledge_base["contents"].to_list(),
    ids=knowledge_base["id"].to_list(),
)
print("Knowledge base populated")

APP_NAME = "agentic-rag-demo"

# Define all the parameters for your pipeline
MODEL = "gpt-4o-mini"
# Define your Prompt template. It contains variables for answer options. It also contains instructions for tool use.
PROMPT_TEMPLATE = [
    {
        "role": "system",
        "content": """Answer the question provided by the user factually. And reference the options for the answer:

Options:
- {{option_A}}
- {{option_B}}
- {{option_C}}
- {{option_D}}
- {{option_E}}

---
If the question is clearly stated, you can use the `retrieve_knowledge` tool to get more context before answering.
If you need more information, first call the `get_clarification` tool to ask the user for more details, before then calling the `retrieve_knowledge` tool.

When you are happy with the context, call the `answer_question` tool to provide the answer to the user, where you should provide your reasoning and a clear citation.
""",
    }
]

# Define the tool schemas for the model
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": "Looks up relevant context in a knowledge base for a given query.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "The query to retrieve knowledge to help answer."},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_clarification",
            "description": "Asks the user clarifying information.",
            "parameters": {
                "type": "object",
                "required": ["question"],
                "properties": {
                    "question": {"type": "string", "description": "Details of the clarification you would like from the user in order to help answer the question."},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "provide_answer",
            "description": "Submits the answer to the user.",
            "parameters": {
                "type": "object",
                "required": ["answer", "reasoning", "citation"],
                "properties": {
                    "answer": {"type": "string", "description": "The final synthesized answer."},
                    "reasoning": {"type": "string", "description": "The reasoning for your answer."},
                    "citation": {"type": "string", "description": "The citation to the knowledge base entry you used when answering."},
                },
                "additionalProperties": False,
            },
        },
    }
]


# Implement the mock tools associated to the tool schemas
def retrieve_knowledge(query: str) -> str:
    """Retrieve most relevant document from the vector db (Chroma) for the question."""
    print("RETRIEVER:\n", query)
    response = collection.query(query_texts=[query], n_results=1)
    retrieved_doc = response["documents"][0][0]
    return retrieved_doc


def get_clarification(question: str) -> str:
    """Ask the user for clarifying information."""
    # TODO: Implement this with your own logic for sending and receiving messages from the user
    print("AI:\n", question)
    return input("User:\n")


def provide_answer(answer: str, reasoning: str, citation: str) -> str:
    """Provide the answer to the user."""
    # TODO: Implement this with your own logic for sending the final response to the user
    print(
        f"\n\nAnswer: {answer}\nReasoning: {reasoning}\nCitation: {citation}"
    )


def call_model(inputs: dict[str, str], messages: list[dict], trace_id: str) -> dict:
    """Calls the model with the provided inputs and messages."""

    # Populate the Prompt template
    start_time = datetime.now()
    populated_template = hl.prompts.populate_template(
        template=PROMPT_TEMPLATE,
        inputs=inputs
    )
    # Call OpenAI to get response - note you can also instead manage your prompts on Humanloop and use our proxy:
    # https://humanloop.com/docs/v5/guides/prompts/call-prompt
    chat_completion = openai.chat.completions.create(
        model=MODEL,
        messages=populated_template + messages,
        # include the tool schemas so the model can make a call
        tools=TOOL_SCHEMAS,
    )
    # Log to Humanloop
    hl.prompts.log(
        path=f"{APP_NAME}/MedQA answer",
        prompt={
            "model": MODEL,
            "template": PROMPT_TEMPLATE,
            "tools": [tool["function"] for tool in TOOL_SCHEMAS],
        },
        inputs=inputs,
        output=chat_completion.choices[0].message.content,
        output_message=chat_completion.choices[0].message,
        trace_parent_id=trace_id,
        start_time=start_time,
        end_time=datetime.now()
    )

    return chat_completion.choices[0].message.to_dict(exclude_unset=False)


def ask_question(question: str, **inputs):
    """Ask a question and get an answer using a simple RAG pipeline.

    param: inputs: the question and options needed by the Prompt template above

    Return: the message history with the latest being the model response
    """

    # Open up the trace on Humanloop
    trace = hl.flows.log(
        path=f"{APP_NAME}/Medqa flow",
        flow={
            # Optionally define attributes that uniquely determine your app so HL can version to Flow
            "attributes": {
                "prompt": {
                    "template": PROMPT_TEMPLATE,
                    "model": MODEL,
                },
                "tools": TOOL_SCHEMAS,
            }
        },
        inputs=inputs,
        start_time=datetime.now(),
    )
    # Initialize the messages with the question
    messages = [{"role": "user", "content": question}]
    tool_name_to_schema = {tool["function"]["name"]: tool for tool in TOOL_SCHEMAS}

    # Do the main agent loop
    steps = 0
    finished = False
    while not finished:
        steps += 1
        response = call_model(inputs=inputs, messages=messages, trace_id=trace.id)
        messages.append(response)
        # Process the tool calls in your system
        if response["tool_calls"]:
            for tool_call in response["tool_calls"]:
                tool_args = json.loads(tool_call["function"]["arguments"])
                tool_name = tool_call["function"]["name"]
                start_time = datetime.now()
                if tool_name == "retrieve_knowledge":
                    tool_output = retrieve_knowledge(**tool_args)
                elif tool_name == "get_clarification":
                    tool_output = get_clarification(**tool_args)
                elif tool_name == "provide_answer":
                    provide_answer(**tool_args)
                    # Session is over...
                    finished = True
                    break
                else:
                    raise ValueError(f"Unknown tool call: {tool_call}")
                    # Add the tool output to the messages to send back to the model
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(tool_output),
                        "tool_call_id": tool_call["id"],
                    }
                )
                # Log the tool call to Humanloop
                hl.tools.log(
                    path=f"{APP_NAME}/Knowledge base",
                    tool={"function": tool_name_to_schema[tool_call["function"]["name"]]["function"]},
                    output=tool_output,
                    trace_parent_id=trace.id,
                    start_time=start_time,
                    end_time=datetime.now()
                )

        if steps > 10:
            raise ValueError("Too many steps in the conversation.")

    # Close the trace on Humanloop so any monitoring evaluators will be run
    hl.flows.update_log(
        log_id=trace.id,
        output=json.dumps(tool_args),
        trace_status="complete",
    )


if __name__ == "__main__":

    datapoint = {
        "inputs": {
            "question": "A 46-year-old man is brought to the emergency department for evaluation of altered mental status. He was found on the floor in front of his apartment. He is somnolent but responsive when aroused. His pulse is 64\/min, respiratory rate is 15\/min, and blood pressure is 120\/75 mm Hg. On physical examination, an alcoholic smell and slurred speech are noted. Neurological exam shows diminished deep tendon reflexes bilaterally and an ataxic gait. His pupils are normal. Blood alcohol concentration is 0.04%. An ECG shows no abnormalities. Which of the following is the most likely cause of this patient's symptoms?",
            "option_A": "Hypoglycemia",
            "option_B": "Cerebral ischemia",
            "option_C": "Ethanol intoxication",
            "option_D": "Cannabis intoxication",
            "option_E": "Benzodiazepine intoxication\n\""},
        "target": {"output": "Benzodiazepine intoxication\n\""}
    }

    ask_question(**datapoint["inputs"])
