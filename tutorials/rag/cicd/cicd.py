""" CICD entry point methods."""
import os
from dotenv import load_dotenv

from tqdm.contrib.concurrent import thread_map
from humanloop import Humanloop, EvaluationStats

from .utils import (
    print_evaluation_progress,
    print_evaluation_results,
    check_evaluation_threshold,
    check_evaluation_improvement,
)
import time
from typing import Callable

load_dotenv()

humanloop = Humanloop(
    api_key=os.getenv("HUMANLOOP_KEY"), base_url="https://neostaging.humanloop.ml/v5"
)


def run_evaluation(pipeline: Callable, evaluation_id: str, max_workers: int = 5):
    """Run a variation of your Pipeline over the Dataset for a specific evaluation to populate results in parallel."""

    # Pull down your dataset
    evaluation = humanloop.evaluations.get(id=evaluation_id)
    retrieved_dataset = humanloop.datasets.get(
        id=evaluation.dataset.id,
        include_datapoints=True,
    )

    # Define the function to be executed in parallel
    def process_datapoint(datapoint):
        pipeline(
            inputs=datapoint.inputs,
            datapoint_id=datapoint.id,
            evaluation_id=evaluation.id,
        )

    # Use thread_map to parallelize and display a progress bar
    thread_map(
        process_datapoint,
        retrieved_dataset.datapoints,
        max_workers=max_workers,
        desc="Processing Datapoints",
    )

    evaluation = humanloop.evaluations.get(id=evaluation.id)
    print("Navigate to Evaluation: ", evaluation.url)


def check_evaluation_results(
    evaluation_id: str,
    evaluator: str
) -> tuple[EvaluationStats, bool]:
    """Check the results of an Evaluation."""

    # Wait for Evaluation to complete
    complete = False
    evaluation = None
    stats = None
    while not complete:
        evaluation = humanloop.evaluations.get(id=evaluation_id)
        stats = humanloop.evaluations.get_stats(id=evaluation_id)
        print_evaluation_progress(evaluation=evaluation, stats=stats)
        complete = evaluation.status == "completed"
        if not complete:
            time.sleep(10)

    # Print Evaluation results
    print_evaluation_results(evaluation=evaluation, stats=stats)

    # Run checks
    check_threshold = check_evaluation_threshold(
        evaluation=evaluation,
        stats=stats,
        evaluator_path=evaluator,
        threshold=0.5,
    )
    check_improvement = check_evaluation_improvement(
        evaluation=evaluation,
        stats=stats,
        evaluator_path=evaluator,
    )
    overall_check = check_threshold and check_improvement

    return stats, overall_check


def create_new_evaluation() -> str:
    """Create a new Evaluation ID as a destination for the CICD."""
    evaluation = humanloop.evaluations.create(
        # NB: you can also use the `id` to reference Datasets and Evaluators
        dataset={"path": "evals_demo/medqa-test"},
        evaluators=[
            {"path": "evals_demo/exact_match"},
            {"path": "evals_demo/levenshtein"},
            {"path": "evals_demo/reasoning"},
        ],
    )
    print("New Evaluation created: ", evaluation.id)
    return evaluation.id