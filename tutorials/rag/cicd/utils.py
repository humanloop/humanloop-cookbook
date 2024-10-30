"""Util methods for formatting and checking Evaluation results."""

from humanloop import EvaluationResponse, EvaluationStats, RunStatsResponse
from humanloop import BooleanEvaluatorStatsResponse as BooleanStats
from humanloop import NumericEvaluatorStatsResponse as NumericStats

# ANSI escape codes for colors
YELLOW = "\033[93m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def populate_template(template: list, inputs: dict[str, str]) -> list:
    """Populate a template with input variables."""
    messages = []
    for i, template_message in enumerate(template):
        content = template_message["content"]
        for key, value in inputs.items():
            content = content.replace("{{" + key + "}}", value)
        message = {**template_message, "content": content}
        messages.append(message)
    return messages


def get_score_from_evaluator_stat(stat: NumericStats | BooleanStats) -> float | None:
    """Get the score from an Evaluator Stat."""
    score = None
    match stat:
        case BooleanStats():
            if stat.total_logs:
                score = round(stat.num_true / stat.total_logs, 2)
        case NumericStats():
            score = round(stat.mean, 2)
        case _:
            raise ValueError("Invalid Evaluator Stat type.")

    return score


def get_evaluator_stats_by_path(
    stat: RunStatsResponse, evaluation: EvaluationResponse
) -> dict[str, NumericStats | BooleanStats]:
    """Get the Evaluator stats by path."""
    evaluators_by_id = {
        evaluator.version.version_id: evaluator for evaluator in evaluation.evaluators
    }
    evaluator_stats_by_path = {
        evaluators_by_id[
            evaluator_stat.evaluator_version_id
        ].version.path: evaluator_stat
        for evaluator_stat in stat.evaluator_stats
    }
    return evaluator_stats_by_path


def check_evaluation_threshold(
    evaluation: EvaluationResponse,
    stats: EvaluationStats,
    evaluator_path: str,
    threshold: float,
) -> bool:
    """Checks if the latest version has an average Evaluator result above a threshold."""

    latest_run_stats = stats.run_stats[0]
    evaluator_stats_by_path = get_evaluator_stats_by_path(
        stat=latest_run_stats, evaluation=evaluation
    )

    if evaluator_path in evaluator_stats_by_path:
        evaluator_stat = evaluator_stats_by_path[evaluator_path]
        score = get_score_from_evaluator_stat(stat=evaluator_stat)
        if score is None:
            raise ValueError(f"Score not found for evaluator {evaluator_path}.")
        if score >= threshold:
            print(
                f"{GREEN}✅ Latest eval [{score}] above threshold [{threshold}] for evaluator {evaluator_path}.{RESET}"
            )
            return True
        else:
            print(
                f"{RED}❌ Latest score [{score}] below the threshold [{threshold}] for evaluator {evaluator_path}.{RESET}"
            )
            return False
    else:
        raise ValueError(f"Evaluator {evaluator_path} not found in the stats.")


def check_evaluation_improvement(
    evaluation: EvaluationResponse, evaluator_path: str, stats: EvaluationStats
) -> bool:
    """Check the latest version has improved across for a specific Evaluator."""

    latest_run_stats = stats.run_stats[0]
    latest_evaluator_stats_by_path = get_evaluator_stats_by_path(
        stat=latest_run_stats, evaluation=evaluation
    )
    if len(stats.run_stats) == 1:
        print(f"{YELLOW}⚠️ No previous versions to compare with.{RESET}")
        return True

    previous_evaluator_stats_by_path = get_evaluator_stats_by_path(
        stat=stats.run_stats[-2], evaluation=evaluation
    )
    if (
        evaluator_path in latest_evaluator_stats_by_path
        and evaluator_path in previous_evaluator_stats_by_path
    ):
        latest_evaluator_stat = latest_evaluator_stats_by_path[evaluator_path]
        previous_evaluator_stat = previous_evaluator_stats_by_path[evaluator_path]
        latest_score = get_score_from_evaluator_stat(stat=latest_evaluator_stat)
        previous_score = get_score_from_evaluator_stat(stat=previous_evaluator_stat)
        diff = round(latest_score - previous_score, 2)
        if diff >= 0:
            print(
                f"{GREEN}✅ Improvement of [{diff}] for evaluator {evaluator_path}{RESET}"
            )
            return True
        else:
            print(
                f"{RED}❌ Regression of [{diff}] for evaluator {evaluator_path}{RESET}"
            )
            return False
    else:
        raise ValueError(f"Evaluator {evaluator_path} not found in the stats.")
