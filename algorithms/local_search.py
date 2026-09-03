import logging
import random
from typing import Callable

from utils.algorithm_utils import (
    HistoryEntry,
    Mask,
    Metrics,
    extract_fitness,
    hamming_distance,
    validate_mask,
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
# NEIGHBOR GENERATION
# ==============================================================================

def generate_neighbor(
    mask: Mask,
    num_swaps: int = 1,
    rng: random.Random | None = None,
) -> Mask:
    """
    Generate a fixed-size neighboring solution using swap moves.

    A neighbor is created by removing `num_swaps` currently selected instances
    and replacing them with the same number of currently unselected instances.

    The total number of selected instances is therefore preserved exactly.

    Args:
        mask:
            Original binary selection mask.

        num_swaps:
            Exact number of selected/unselected instance pairs to exchange.

        rng:
            Optional pseudo-random number generator.

    Returns:
        A new neighboring mask.

    Raises:
        ValueError:
            If the mask is invalid or the requested number of swaps cannot be
            performed.
    """
    validate_mask(mask)

    if num_swaps <= 0:
        raise ValueError(
            f"'num_swaps' must be greater than 0, "
            f"but received {num_swaps}."
        )

    generator = (
        rng
        if rng is not None
        else random
    )

    selected_indices = [
        index
        for index, selected in mask.items()
        if selected == 1
    ]

    unselected_indices = [
        index
        for index, selected in mask.items()
        if selected == 0
    ]

    max_feasible_swaps = min(
        len(selected_indices),
        len(unselected_indices),
    )

    if max_feasible_swaps == 0:
        raise ValueError(
            "A neighbor cannot be generated because the mask does not contain "
            "both selected and unselected instances."
        )

    if num_swaps > max_feasible_swaps:
        raise ValueError(
            f"'num_swaps'={num_swaps} exceeds the maximum feasible number "
            f"of swaps ({max_feasible_swaps})."
        )

    indices_to_remove = generator.sample(
        selected_indices,
        k=num_swaps,
    )

    indices_to_add = generator.sample(
        unselected_indices,
        k=num_swaps,
    )

    neighbor = mask.copy()

    for index in indices_to_remove:
        neighbor[index] = 0

    for index in indices_to_add:
        neighbor[index] = 1

    return neighbor


# ==============================================================================
# LOCAL SEARCH
# ==============================================================================

def local_search(
    fitness_func: Callable[[Mask], Metrics],
    total_instances: int,
    keep_percentage: float = 0.1,
    max_evaluations: int = 100,
    num_swaps: int = 1,
    target_metric: Metric = Metric.ACCURACY,
    rng: random.Random | None = None,
) -> tuple[Mask, float, list[HistoryEntry], list[float], int]:
    """
    Perform fixed-size randomized Local Search for instance selection.

    The algorithm starts from one random solution and repeatedly evaluates one
    randomly generated neighboring solution.

    Non-worsening neighbors are accepted, allowing the search to move across
    fitness plateaus.

    The complete evaluation budget is always consumed.

    Args:
        fitness_func:
            Function receiving a binary selection mask and returning its
            evaluation metrics.

        total_instances:
            Total number of candidate training instances.

        keep_percentage:
            Fraction of instances retained in every solution.

        max_evaluations:
            Exact number of fitness evaluations, including the initial
            solution.

        num_swaps:
            Number of selected/unselected instance pairs exchanged when
            generating each neighbor.

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

    if num_swaps <= 0:
        raise ValueError(
            f"'num_swaps' must be greater than 0, "
            f"but received {num_swaps}."
        )

    subset_size = validate_search_subset_size(
        total_instances=total_instances,
        keep_percentage=keep_percentage,
    )

    max_feasible_swaps = min(
        subset_size,
        total_instances - subset_size,
    )

    if num_swaps > max_feasible_swaps:
        raise ValueError(
            f"'num_swaps'={num_swaps} exceeds the maximum feasible number "
            f"of swaps ({max_feasible_swaps})."
        )

    algorithm_id = Algorithm.LOCAL_SEARCH.value
    metric_name = target_metric.value

    logger.info(
        "Starting %s | target_metric=%s | keep_percentage=%.4f | "
        "max_evaluations=%d | num_swaps=%d",
        algorithm_id,
        metric_name,
        keep_percentage,
        max_evaluations,
        num_swaps,
    )

    # --------------------------------------------------------------------------
    # Initial solution
    # --------------------------------------------------------------------------

    current_solution = create_random_mask(
        total_instances=total_instances,
        keep_percentage=keep_percentage,
        rng=rng,
    )

    initial_metrics = dict(
        fitness_func(current_solution)
    )

    current_fitness = extract_fitness(
        metrics=initial_metrics,
        metric_name=metric_name,
    )

    current_solution_id = 1

    best_solution = current_solution.copy()
    best_fitness = current_fitness
    best_solution_id = current_solution_id
    best_found_at_evaluation = 1

    selected_instances = sum(
        current_solution.values()
    )

    selected_percentage = (
        selected_instances / total_instances
    )

    initial_history_entry: HistoryEntry = dict(
        initial_metrics
    )

    initial_history_entry.update(
        {
            "Algorithm": algorithm_id,
            "Evaluation": 1,
            "Solution ID": current_solution_id,
            "Candidate Type": "initial",
            "Initial Percentage": keep_percentage,
            "Final Percentage": selected_percentage,
            "Selected Instances": selected_instances,
            "Target Metric": metric_name,
            "Fitness": current_fitness,
            "Current Fitness Before": None,
            "Current Fitness": current_fitness,
            "Best Fitness": best_fitness,
            "Accepted": True,
            "Improved Best": True,
            "Neighbor Swaps": 0,
            "Hamming Distance": 0,
            "Current Solution ID Before": None,
            "Current Solution ID After": current_solution_id,
            "Best Solution ID": best_solution_id,
            "Best Found At Evaluation":
                best_found_at_evaluation,
        }
    )

    fitness_history: list[HistoryEntry] = [
        initial_history_entry
    ]

    best_fitness_history: list[float] = [
        best_fitness
    ]

    # --------------------------------------------------------------------------
    # Neighborhood search
    # --------------------------------------------------------------------------

    for evaluation in range(
        2,
        max_evaluations + 1,
    ):
        candidate_solution_id = evaluation

        current_fitness_before = (
            current_fitness
        )

        current_solution_id_before = (
            current_solution_id
        )

        candidate_solution = generate_neighbor(
            mask=current_solution,
            num_swaps=num_swaps,
            rng=rng,
        )

        candidate_hamming_distance = hamming_distance(
            current_solution,
            candidate_solution,
        )

        candidate_metrics = dict(
            fitness_func(candidate_solution)
        )

        candidate_fitness = extract_fitness(
            metrics=candidate_metrics,
            metric_name=metric_name,
        )

        accepted = (
            candidate_fitness >= current_fitness
        )

        if accepted:
            current_solution = (
                candidate_solution.copy()
            )

            current_fitness = (
                candidate_fitness
            )

            current_solution_id = (
                candidate_solution_id
            )

        improved_best = (
            candidate_fitness > best_fitness
        )

        if improved_best:
            best_solution = (
                candidate_solution.copy()
            )

            best_fitness = (
                candidate_fitness
            )

            best_solution_id = (
                candidate_solution_id
            )

            best_found_at_evaluation = (
                evaluation
            )

            logger.info(
                "[%s] New best at evaluation %d/%d | %s=%.6f",
                algorithm_id,
                evaluation,
                max_evaluations,
                metric_name,
                best_fitness,
            )

        selected_instances = sum(
            candidate_solution.values()
        )

        selected_percentage = (
            selected_instances / total_instances
        )

        history_entry: HistoryEntry = dict(
            candidate_metrics
        )

        history_entry.update(
            {
                "Algorithm": algorithm_id,
                "Evaluation": evaluation,
                "Solution ID": candidate_solution_id,
                "Candidate Type": "neighbor",
                "Initial Percentage": keep_percentage,
                "Final Percentage": selected_percentage,
                "Selected Instances": selected_instances,
                "Target Metric": metric_name,
                "Fitness": candidate_fitness,
                "Current Fitness Before":
                    current_fitness_before,
                "Current Fitness":
                    current_fitness,
                "Best Fitness":
                    best_fitness,
                "Accepted":
                    accepted,
                "Improved Best":
                    improved_best,
                "Neighbor Swaps":
                    num_swaps,
                "Hamming Distance":
                    candidate_hamming_distance,
                "Current Solution ID Before":
                    current_solution_id_before,
                "Current Solution ID After":
                    current_solution_id,
                "Best Solution ID":
                    best_solution_id,
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