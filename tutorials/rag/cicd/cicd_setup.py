"""Set up evaluate_rag_cicd.py.

Uploads the Dataset and Evaluators.
Note that while the Evaluators are similar to the example Evaluators created
automatically when you signed up, these Evaluators contain an additional step
processing the model's output.
"""

from contextlib import contextmanager
import os
import pandas as pd
from pathlib import Path
from humanloop import Humanloop, UnprocessableEntityError

# Create a Humanloop client
humanloop = Humanloop(
    api_key=os.getenv("HUMANLOOP_KEY"), base_url=os.getenv("HUMANLOOP_BASE_URL")
)

assets_folder = Path(__file__).parents[3].resolve() / "assets"


def upload_dataset():
    df = pd.read_json(assets_folder / "datapoints.jsonl", lines=True)

    datapoints = [row.to_dict() for _i, row in df.iterrows()][0:20]
    with ignore_already_committed():
        humanloop.datasets.upsert(
            path="evals_demo/medqa-small",
            datapoints=datapoints,
            commit_message=f"Added {len(datapoints)} datapoints from MedQA test dataset.",
        )


def upload_evaluators():
    # Upload Code Evaluators
    for evaluator_name, return_type in [
        ("exact_match", "boolean"),
        ("levenshtein", "number"),
    ]:
        with open(assets_folder / f"evaluators/{evaluator_name}.py", "r") as f:
            code = f.read()

        with ignore_already_committed():
            humanloop.evaluators.upsert(
                path=f"evals_demo/{evaluator_name}",
                spec={
                    "evaluator_type": "python",
                    "arguments_type": "target_required",
                    "return_type": return_type,
                    "code": code,
                },
                commit_message=f"New version from {evaluator_name}.py",
            )

    # Upload an LLM Evaluator
    with ignore_already_committed():
        humanloop.evaluators.upsert(
            path="evals_demo/reasoning",
            spec={
                "evaluator_type": "llm",
                "arguments_type": "target_free",
                "return_type": "boolean",
                "prompt": {
                    "model": "gpt-4o",
                    "endpoint": "complete",
                    "temperature": 0,
                    "template": 'An answer is shown below. The answer contains 3 sections, separated by "---". The first section is the final answer. The second section is an explanation. The third section is a citation.\n\nEvaluate if the final answer follows from the citation and the reasoning in the explanation section. Give a brief explanation/discussion. Do not make your judgment based on factuality, but purely based on the logic presented.\nOn a new line, give a final verdict of "True" or "False".\n\nAnswer:\n{{log.output}}',
                },
            },
            commit_message="Initial reasoning evaluator.",
        )


@contextmanager
def ignore_already_committed():
    """Context manager to ignore the error where a version has already been committed."""
    try:
        yield
    except UnprocessableEntityError as e:
        try:
            if "already been committed" in e.body.detail["description"]:
                return
        except Exception:
            pass
        raise e


if __name__ == "__main__":
    upload_dataset()
    upload_evaluators()
    print("Datasets and Evaluators uploaded.")
