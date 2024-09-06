""" Evaluate RAG pipeline `ask_question` using Humanloop, with Prompt stored in code."""

import argparse
import os
from dotenv import load_dotenv
import inspect
import uuid
import pandas as pd
from chromadb import chromadb
from openai import OpenAI
from humanloop import Humanloop
from cicd.cicd import check_evaluation_results, create_new_evaluation, run_evaluation
from cicd.utils import populate_template


# Init clients and setup Vector db
load_dotenv()
chroma = chromadb.Client()
openai = OpenAI(api_key=os.getenv("OPENAI_KEY"))
humanloop = Humanloop(
    api_key=os.getenv("HUMANLOOP_KEY"), base_url="https://neostaging.humanloop.ml/v5"
)
collection = chroma.get_or_create_collection(name="MedQA")
knowledge_base = pd.read_parquet("../../assets/sources/textbooks.parquet")
knowledge_base = knowledge_base.sample(10, random_state=42)
collection.add(
    documents=knowledge_base["contents"].to_list(),
    ids=knowledge_base["id"].to_list(),
)

# Define Prompt in code
model = "gpt-4o-mini"
temperature = 0
template = [
    {
        "role": "system",
        "content": """Answer the following question factually as best you can. Guess if the context is clearly wrong.

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


def retrieval_tool(question: str) -> str:
    """Retrieve most relevant document from the vector db (Chroma) for the question."""
    response = collection.query(query_texts=[question], n_results=1)
    retrieved_doc = response["documents"][0][0]
    return retrieved_doc


def ask_question(
    inputs: dict[str, str],
    datapoint_id: str | None = None,
    evaluation_id: str | None = None,
) -> str:
    """Ask a question and get an answer using a simple RAG pipeline"""

    # Retrieve context
    retrieved_data = retrieval_tool(inputs["question"])

    # Log the context and retriever details to your Humanloop Tool
    session_id = uuid.uuid4().hex
    humanloop.tools.log(
        path="evals_demo/medqa-retrieval",
        tool={
            "function": {
                "name": "retrieval_tool",
                "description": "Retrieval tool for MedQA.",
            },
            "source_code": inspect.getsource(retrieval_tool),
        },
        output=retrieved_data,
        session_id=session_id,
    )

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

    # Log the response and Prompt details to your Humanloop Prompt
    humanloop.prompts.log(
        path="evals_demo/medqa-answer",
        prompt={
            "model": model,
            "temperature": temperature,
            "template": template,
        },
        inputs=inputs,
        output=chat_completion.choices[0].message.content,
        output_message=chat_completion.choices[0].message,
        session_id=session_id,
        # NB: arguments to link to Evaluation and Dataset
        source_datapoint_id=datapoint_id,
        evaluation_id=evaluation_id,
    )

    return answer


if __name__ == "__main__":
    # Command line arguments for Evaluation ID and Evaluator Path
    parser = argparse.ArgumentParser(
        description="Run evaluation with specified ID and evaluator path."
    )
    parser.add_argument(
        "--evaluation_id",
        type=str,
        help="Evaluation ID for the run. If not specified, a new one will be created.",
    )

    args = parser.parse_args()

    # Determine Evaluation ID - destination for CI runs.
    EVALUATION_ID = (
        args.evaluation_id if args.evaluation_id else create_new_evaluation()
    )

    # Trigger another run
    run_evaluation(pipeline=ask_question, evaluation_id=EVALUATION_ID)

    # Display and check the results
    stats, overall_check = check_evaluation_results(
        evaluation_id=EVALUATION_ID,
        evaluator="evals_demo/reasoning"
    )

