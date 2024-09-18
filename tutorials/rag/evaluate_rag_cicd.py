"""
The goal of this script is to demonstrate how a CI/CD process can leverage
Humanloop Evaluations to check the performance of your AI application. We assume your
Prompt and Tool definitions are managed in code (alternatively they can be managed
on Humanloop, which lies outside the scope of this script).

This script assumes that you have a Flow, a Dataset and Evaluators already
managed on Humanloop. You can follow the evaluate-rag-flow.py notebook to
set these up.

We will use the `ask_question` pipeline as a simple AI application we wish to evaluate.
The pipeline takes a question as input, retrieves relevant information from a Chroma
vectordb and then uses OpenAI to generate an answer.
"""

import argparse
import os
from dotenv import load_dotenv
import inspect
import uuid
import pandas as pd
from chromadb import chromadb
from openai import OpenAI
from humanloop import Humanloop
from cicd.utils import (
    populate_template,
    check_evaluation_threshold,
    check_evaluation_improvement,
)
from typing import Callable
from datetime import datetime
from tqdm.contrib.concurrent import thread_map
import time

# Init clients and setup Vector db
load_dotenv()
chroma = chromadb.Client()
openai = OpenAI(api_key=os.getenv("OPENAI_KEY"))
humanloop = Humanloop(api_key=os.getenv("HUMANLOOP_KEY"), base_url="http://0.0.0.0/v5")
collection = chroma.get_or_create_collection(name="MedQA")
knowledge_base = pd.read_parquet("../../assets/sources/textbooks.parquet")
knowledge_base = knowledge_base.sample(10, random_state=42)
collection.add(
    documents=knowledge_base["contents"].to_list(),
    ids=knowledge_base["id"].to_list(),
)

# Define your Prompt details in code
model = "gpt-4-turbo"
temperature = 0
template = [
    {
        "role": "system",
        "content": """Answer the following question. Always choose A if you don't know.

Question: {{question}}

Options:
- {{option_A}}
- {{option_B}}
- {{option_C}}
- {{option_D}}
- {{option_E}}

---

Here is some retrieved information that might be helpful.
Retrieved data:
{{retrieved_data}}

---

Give you answer in 3 sections using the following format. Do not include the quotes or the brackets. Do include the "---" separators.
```
<chosen option verbatim>
---
<clear explanation of why the option is correct and why the other options are incorrect. keep it ELI5.>
---
<quote relevant information snippets from the retrieved data verbatim. every line here should be directly copied from the retrieved data>
```
""",
    }
]


# --- An existing implementation of a RAG pipeline


def retrieval_tool(question: str) -> str:
    """Retrieve most relevant document from the vector db (Chroma) given the question."""
    response = collection.query(query_texts=[question], n_results=1)
    retrieved_doc = response["documents"][0][0]
    return retrieved_doc


def ask_question(
    inputs: dict[str, str],
) -> str:
    """Ask a question and get an answer using a simple RAG pipeline"""

    # Retrieve context
    retrieved_data = retrieval_tool(inputs["question"])
    # Populate the Prompt template
    inputs = {**inputs, "retrieved_data": retrieved_data}
    messages = populate_template(template, inputs)

    # Call OpenAI to get a response
    chat_completion = openai.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    answer = chat_completion.choices[0].message.content

    return answer


# --- An entry point for CI for leveraging Humanloop Evaluations


def run_evaluation(
    pipeline: Callable,
    flow_id: str,
    evaluation_id: str,
    attributes: dict,
    max_workers: int = 5,
) -> bool:
    """Run your pipeline for a Humanloop Evaluation.

    Args:
        pipeline: The pipeline function to evaluate.
        evaluation_id: The ID of the Evaluation to append with this run.
        attributes: The attributes that uniquely identify your pipeline version.
        max_workers: The number of workers to use for parallel processing.

    Returns:
        bool: The overall check for the CI. Customise the checks to your needs.

    This function will print the progress of the Evaluation and the final results.
    """

    # Pull down your dataset
    evaluation = humanloop.evaluations.get(id=evaluation_id)
    dataset = humanloop.datasets.get(
        id=evaluation.dataset.id,
        include_datapoints=True,
    )

    # Define the function to execute your pipeline in parallel and Log to Humanloop
    def process_datapoint(datapoint):
        start_time = datetime.now()
        try:
            output = pipeline(
                inputs=datapoint.inputs,
            )
            _ = humanloop.flows.log(
                trace_id=uuid.uuid4().hex,
                id=flow_id,
                flow={"attributes": attributes},
                inputs=datapoint.inputs,
                output=output,
                source_datapoint_id=datapoint.id,
                evaluation_id=evaluation.id,
                trace_status="complete",
                start_time=start_time,
                end_time=datetime.now(),
            )

        except Exception as error:
            # If it fails, still post the log to Humanloop
            _ = humanloop.flows.log(
                trace_id=uuid.uuid4().hex,
                id=flow_id,
                flow={"attributes": attributes},
                inputs=datapoint.inputs,
                error=str(error),
                source_datapoint_id=datapoint.id,
                evaluation_id=evaluation.id,
                trace_status="complete",
                start_time=start_time,
                end_time=datetime.now(),
            )

    # Execute your pipeline and send the logs to Humanloop in parallel
    thread_map(
        process_datapoint,
        dataset.datapoints,
        max_workers=max_workers,
        desc="Processing Datapoints",
    )

    # Wait for the Evaluation to complete then print the results
    complete = False
    stats = None
    while not complete:
        stats = humanloop.evaluations.get_stats(id=evaluation.id)
        print(stats.progress)
        complete = stats.status == "completed"
        if not complete:
            time.sleep(10)

    # Print Evaluation results
    print(stats.report)

    evaluation = humanloop.evaluations.get(id=evaluation.id)
    print("Navigate to Evaluation: ", evaluation.url)

    # Run checks to determine whether to pass or fail the CI - customise checks to your needs
    check_threshold = check_evaluation_threshold(
        evaluation=evaluation,
        stats=stats,
        evaluator_path="evals_demo/reasoning",
        threshold=0.5,
    )
    check_improvement = check_evaluation_improvement(
        evaluation=evaluation,
        stats=stats,
        evaluator_path="evals_demo/reasoning",
    )
    overall_check = check_threshold and check_improvement

    return overall_check


if __name__ == "__main__":
    # Command line arguments for linking to an existing Evaluation
    parser = argparse.ArgumentParser(
        description="Run evaluation with specified ID and evaluator path."
    )
    parser.add_argument(
        "--evaluation_id",
        type=str,
        help="Evaluation ID for the run. If not specified, a new one will be created.",
    )
    args = parser.parse_args()
    evaluation_id = args.evaluation_id

    # These attributes should represent the configuration of your pipeline
    attributes = {
        "prompt": {
            "template": template,
            "model": model,
            "temperature": temperature,
        },
        "function": {
            "name": "retrieval_tool_v3",
            "description": "Retrieval tool for MedQA.",
        },
        "source_code": inspect.getsource(retrieval_tool),
    }

    # Create or get a Flow version
    flow = humanloop.flows.upsert(
        path="evals_demo/answer-flow",
        attributes=attributes,
    )

    # If an Evaluation does not already exist, create one
    if not evaluation_id:
        evaluation = humanloop.evaluations.create(
            name="MedQA CICD",
            # NB: you can use `path`or `id` for references on Humanloop
            file={"id": flow.id},
            # Assume Evaluators and Datasets already exist
            dataset={"path": "evals_demo/medqa-test"},
            evaluators=[
                {"path": "evals_demo/exact_match"},
                {"path": "evals_demo/levenshtein"},
                {"path": "evals_demo/reasoning"},
            ],
        )
        print("New Evaluation created: ", evaluation.id)
        evaluation_id = evaluation.id

    # Trigger an Evaluation run
    check = run_evaluation(
        pipeline=ask_question,
        flow_id=flow.id,
        evaluation_id=evaluation_id,
        # attributes specify what version of the pipeline is being evaluated
        attributes=attributes,
    )

    # Use this check to pass or fail your CI...
