""" Util methods for formatting and checking Evaluation results."""

from prettytable import PrettyTable
from datetime import datetime
from humanloop import EvaluationResponse, EvaluationStats, VersionStats
from humanloop import BooleanEvaluatorVersionStats as BooleanStats
from humanloop import NumericEvaluatorVersionStats as NumericStats

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
    match stat:
        case BooleanStats():
            score = round(stat.num_true / stat.total_logs, 2)
        case NumericStats():
            score = round(stat.mean, 2)
        case _:
            raise ValueError("Invalid Evaluator Stat type.")
    return score


def print_evaluation_results(
        evaluation: EvaluationResponse, stats: EvaluationStats
) -> None:
    """Pretty prints a table of stats for a given Evaluation."""

    evaluators_by_id = {
        evaluator.version.version_id: evaluator for evaluator in evaluation.evaluators
    }
    versions_by_id = {
        evaluatee.version["version_id"]: evaluatee
        for evaluatee in evaluation.evaluatees
    }
    sorted_stats = sorted(
        stats.version_stats,
        key=lambda x: versions_by_id[x.version_id].version["created_at"]
    )
    version_ids = [stat.version_id for stat in sorted_stats]
    evaluator_ids = [evaluator.version.version_id for evaluator in evaluation.evaluators]

    # Table of evaluator X versions
    table = PrettyTable()
    table.field_names = ["Version id"] + version_ids
    table.align["Version id"] = "r"
    table.add_row(["Created", *[versions_by_id[stat.version_id].version["created_at"].split('.')[0] for stat in sorted_stats]])
    table.add_row(["Model", *[versions_by_id[stat.version_id].version["model"] for stat in sorted_stats]], divider=True)
    table.add_row(["Evaluators", *["" for _ in sorted_stats]], divider=True)

    data = {
        evaluator_id: {stat.version_id: None for stat in sorted_stats}
        for evaluator_id in evaluator_ids
    }
    for stat in sorted_stats:
        version = versions_by_id[stat.version_id]
        for evaluator_stat in stat.evaluator_version_stats:
            evaluator = evaluators_by_id[evaluator_stat.evaluator_version_id]
            score = get_score_from_evaluator_stat(stat=evaluator_stat)
            data[evaluator.version.version_id][version.version["version_id"]] = score

    for evaluator_id in evaluator_ids:
        evaluator = evaluators_by_id[evaluator_id]
        row = [evaluator.version.path]  # Start with the evaluator name/path
        for version_id in version_ids:
            score = data[evaluator_id][version_id]
            row.append(score if score is not None else "N/A")
        table.add_row(row)

    print(f"\n{CYAN}📊 Evaluation Results for {version.version['path']} {RESET}")
    print(table, "\n\n")


def get_sorted_version_stats(
    stats: EvaluationStats,
    evaluation: EvaluationResponse
) -> list[VersionStats]:
    """Sort the VersionStats by created_at."""
    versions_by_id = {
        evaluatee.version["version_id"]: evaluatee
        for evaluatee in evaluation.evaluatees
    }
    sorted_stats = sorted(
        stats.version_stats,
        key=lambda x: versions_by_id[x.version_id].version["created_at"]
    )
    if len(sorted_stats) == 0:
        raise ValueError("No VersionStats found.")

    return sorted_stats


def get_evaluator_stats_by_path(
    stat: VersionStats, evaluation: EvaluationResponse
) -> dict[str, NumericStats | BooleanStats]:
    """Get the Evaluator stats by path."""
    evaluators_by_id = {
        evaluator.version.version_id: evaluator for evaluator in evaluation.evaluators
    }
    evaluator_stats_by_path = {
        evaluators_by_id[evaluator_stat.evaluator_version_id].version.path: evaluator_stat
        for evaluator_stat in stat.evaluator_version_stats
    }
    return evaluator_stats_by_path


def check_evaluation_threshold(
    evaluation: EvaluationResponse,
    stats: EvaluationStats,
    evaluator_path: str,
    threshold: float,
) -> bool:
    """Checks if the latest version has an average Evaluator result above a threshold."""

    version_stats = get_sorted_version_stats(stats=stats, evaluation=evaluation)
    evaluator_stats_by_path = get_evaluator_stats_by_path(
        stat=version_stats[-1],
        evaluation=evaluation
    )

    if evaluator_path in evaluator_stats_by_path:
        evaluator_stat = evaluator_stats_by_path[evaluator_path]
        score = get_score_from_evaluator_stat(stat=evaluator_stat)
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
    evaluation: EvaluationResponse,
    evaluator_path: str,
    stats: EvaluationStats
) -> bool:
    """Check the latest version has improved across for a specific Evaluator."""

    version_stats = get_sorted_version_stats(stats=stats, evaluation=evaluation)
    latest_evaluator_stats_by_path = get_evaluator_stats_by_path(
        stat=version_stats[-1],
        evaluation=evaluation
    )
    if len(version_stats) == 1:
        print(
            f"{YELLOW}⚠️ No previous versions to compare with.{RESET}"
        )
        return True

    previous_evaluator_stats_by_path = get_evaluator_stats_by_path(
        stat=version_stats[-2],
        evaluation=evaluation
    )
    if evaluator_path in latest_evaluator_stats_by_path and evaluator_path in previous_evaluator_stats_by_path:
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


def print_evaluation_progress(
        evaluation: EvaluationResponse,
        stats: EvaluationStats
) -> None:
    """Prints the progress of an Evaluation Logs and Evaluators"""

    num_evaluators = len(evaluation.evaluators)
    num_versions = len(evaluation.evaluatees)
    print(f"\n{CYAN}⏳ Evaluation Progress{RESET}")
    print(f"Total Logs: {stats.overall_stats.total_logs}/{num_versions*stats.overall_stats.num_datapoints}")
    print(f"Total Judgments: {stats.overall_stats.total_evaluator_logs}/{num_evaluators*num_versions*stats.overall_stats.num_datapoints}\n\n")