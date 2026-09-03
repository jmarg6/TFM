import logging
import random
from typing import Callable

from utils.algorithm_utils import (
    HistoryEntry,
    Mask,
    Metrics,
    extract_fitness,
)
from utils.config import Algorithm, Metric
from utils.selection_utils import (
    create_random_mask,
    validate_search_subset_size,
)

# ==============================================================================
# LOGGER
# ==============================================================================

logger = logging.getLogger(__name__)


# ==============================================================================
# RANDOM SEARCH
# ==============================================================================

def random_search(
    fitness_func: Callable[[Mask], Metrics],
    total_instances: int,
    keep_percentage: float = 0.5,
    max_evaluations: int = 100,
    target_metric: Metric = Metric.ACCURACY,
    rng: random.Random | None = None,
) -> tuple[Mask, float, list[HistoryEntry], list[float], int]:
    """
    Perform Random Search for fixed-size instance selection.

    At each evaluation, an independent random subset is generated and
    evaluated. The best solution found over the complete evaluation budget is
    returned.

    Random Search acts as the baseline optimization method and therefore
    always consumes the complete fitness-evaluation budget.

    Args:
        fitness_func:
            Function receiving a binary selection mask and returning the
            evaluation metrics of the selected subset.

        total_instances:
            Total number of candidate training instances.

        keep_percentage:
            Fraction of training instances retained in every candidate
            solution.

        max_evaluations:
            Exact number of candidate subsets to evaluate.

        target_metric:
            Metric used as the optimization objective.

        rng:
            Optional pseudo-random number generator.

    Returns:
        A tuple containing:

        - Best selection mask found.
        - Best fitness value obtained.
        - Complete evaluation history.
        - Running best-fitness history.
        - Total number of fitness evaluations performed.
    """
    if total_instances <= 0:
        raise ValueError(
            f"'total_instances' must be greater than 0, "
            f"but received {total_instances}."
        )

    if not 0.0 < keep_percentage < 1.0:
        raise ValueError(
            f"'keep_percentage' must be in the interval (0, 1), "
            f"but received {keep_percentage}. "
            "The 100% dataset must be evaluated separately as the "
            "full-dataset baseline."
        )

    if max_evaluations <= 0:
        raise ValueError(
            f"'max_evaluations' must be greater than 0, "
            f"but received {max_evaluations}."
        )
    
    validate_search_subset_size(
        total_instances=total_instances,
        keep_percentage=keep_percentage,
    )

    algorithm_id = Algorithm.RANDOM_SEARCH.value
    metric_name = target_metric.value

    best_fitness = float("-inf")
    best_solution: Mask = {}

    best_solution_id: int | None = None
    best_found_at_evaluation: int | None = None

    fitness_history: list[HistoryEntry] = []
    best_fitness_history: list[float] = []

    logger.info(
        "Starting %s | target_metric=%s | keep_percentage=%.4f | "
        "max_evaluations=%d",
        algorithm_id,
        metric_name,
        keep_percentage,
        max_evaluations,
    )

    for evaluation in range(
        1,
        max_evaluations + 1,
    ):
        solution_id = evaluation

        current_solution = create_random_mask(
            total_instances=total_instances,
            keep_percentage=keep_percentage,
            rng=rng,
        )

        metrics_result = dict(
            fitness_func(current_solution)
        )

        current_fitness = extract_fitness(
            metrics=metrics_result,
            metric_name=metric_name,
        )

        selected_instances = sum(
            current_solution.values()
        )

        final_percentage = (
            selected_instances / total_instances
        )

        improved_best = (
            current_fitness > best_fitness
        )

        if improved_best:
            best_fitness = current_fitness
            best_solution = current_solution.copy()
            best_solution_id = solution_id
            best_found_at_evaluation = evaluation

            logger.info(
                "[%s] New best at evaluation %d/%d | %s=%.6f",
                algorithm_id,
                evaluation,
                max_evaluations,
                metric_name,
                best_fitness,
            )

        history_entry: HistoryEntry = dict(
            metrics_result
        )

        history_entry.update(
            {
                "Algorithm": algorithm_id,
                "Evaluation": evaluation,
                "Solution ID": solution_id,
                "Candidate Type": "random",
                "Initial Percentage": keep_percentage,
                "Final Percentage": final_percentage,
                "Selected Instances": selected_instances,
                "Target Metric": metric_name,
                "Fitness": current_fitness,
                "Best Fitness": best_fitness,
                "Improved Best": improved_best,
                "Best Solution ID": best_solution_id,
                "Best Found At Evaluation":
                    best_found_at_evaluation,
            }
        )

        fitness_history.append(
            history_entry
        )

        best_fitness_history.append(
            best_fitness
        )

    evaluations_done = len(
        fitness_history
    )

    logger.info(
        "[%s] Finished | evaluations=%d | best_%s=%.6f",
        algorithm_id,
        evaluations_done,
        metric_name,
        best_fitness,
    )

    return (
        best_solution,
        best_fitness,
        fitness_history,
        best_fitness_history,
        evaluations_done,
    )