import os
from dotenv import load_dotenv
from chromadb import chromadb
from openai import OpenAI
from humanloop import Humanloop

from exact_match import exact_match as exact_match_evaluator
from levenshtein import compare_log_and_target as levenshtein_evaluator
from constants import PROMPT_TEMPLATE
from utilities import (
    compute_workspace_path,
    populate_knowledge_base,
    read_evaluation_dataset,
)


# SETUP
load_dotenv()
MODEL = os.getenv("MODEL")

# INIT CLIENTS
hl = Humanloop(api_key=os.getenv("HUMANLOOP_KEY"))
openai = OpenAI(api_key=os.getenv("OPENAI_KEY"))
chroma = chromadb.Client()


# Populate the knowledge base for the RAG application
populate_knowledge_base(chroma=chroma)


def retrieve_knowledge(query: str) -> str:
    """Retrieve most relevant document from the vector db (Chroma) for the question."""
    collection = chroma.get_or_create_collection(name="MedQA")
    response = collection.query(query_texts=[query], n_results=1)
    retrieved_doc = response["documents"][0][0]
    return retrieved_doc


def call_model(**inputs: dict[str, str]) -> dict:
    """Calls the model with the provided inputs and messages."""

    # Populate the Prompt template
    messages = hl.prompts.populate_template(template=PROMPT_TEMPLATE, inputs=inputs)
    # Call OpenAI to get response - note you can also instead manage your prompts on Humanloop and use our proxy:
    # https://humanloop.com/docs/v5/guides/prompts/call-prompt
    chat_completion = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    return chat_completion.choices[0].message.content


def ask_question(**inputs) -> str:
    """Ask a question and get an answer using a simple RAG pipeline"""
    # Retrieve context
    retrieved_data = retrieve_knowledge(inputs["question"])
    inputs = {**inputs, "retrieved_data": retrieved_data}
    # Call LLM
    output = call_model(**inputs)
    return output


if __name__ == "__main__":
    output = ask_question(
        **{
            "question": "A 34-year-old male suffers from inherited hemophilia A. He and his wife have three unaffected daughters. What is the probability that the second daughter is a carrier of the disease?",
            "option_A": "0%",
            "option_B": "25%",
            "option_C": "50%",
            "option_D": "75%",
            "option_E": "100%",
        }
    )

    hl.evaluations.run(
        name="Initial experiments",
        # What is being Evaluated
        file={
            "path": compute_workspace_path("MedQA flow"),
            "callable": ask_question,
            # Bump the version when you make changes to the pipeline
            "version": {
                "version": "0.1.0",
                "description": "Initial version of the agentic RAG pipeline.",
                "template": PROMPT_TEMPLATE,
                "model": MODEL,
            },
        },
        # Can also specify a path to the Dataset or provide a Dataset ID
        dataset={
            "path": compute_workspace_path("MedQA test"),
            "datapoints": read_evaluation_dataset(),
        },
        # Replace with your own Evaluators
        evaluators=[
            {
                "path": compute_workspace_path("Exact match"),
                "callable": exact_match_evaluator,
                "args_type": "target_required",
                "return_type": "boolean",
            },
            {
                "path": compute_workspace_path("Levenshtein"),
                "callable": levenshtein_evaluator,
                "args_type": "target_required",
                "return_type": "number",
            },
        ],
    )
