import logging
import random
from typing import Callable

from algorithms.genetic import (
    fixed_size_crossover,
    tournament_selection,
)
from algorithms.local_search import generate_neighbor
from utils.algorithm_utils import (
    HistoryEntry,
    Mask,
    Metrics,
    extract_fitness,
    hamming_distance,
)
from utils.config import Algorithm, Metric
from utils.selection_utils import (
    create_random_mask,
    mutate_mask,
    validate_search_subset_size,
)


# ==============================================================================
# LOGGER
# ==============================================================================

logger = logging.getLogger(__name__)


# ==============================================================================
# MEMETIC ALGORITHM
# ==============================================================================

def memetic_algorithm(
    fitness_func: Callable[[Mask], Metrics],
    total_instances: int,
    population_size: int = 10,
    keep_percentage: float = 0.1,
    max_evaluations: int = 100,
    tournament_size: int = 3,
    mutation_probability: float = 0.1,
    mutation_fraction: float = 0.05,
    local_search_probability: float = 0.2,
    local_search_steps: int = 5,
    local_search_num_swaps: int = 1,
    target_metric: Metric = Metric.ACCURACY,
    rng: random.Random | None = None,
) -> tuple[Mask, float, list[HistoryEntry], list[float], int]:
    """
    Perform a fixed-size Memetic Algorithm for instance selection.

    The algorithm combines the same evolutionary operators used by the Genetic
    Algorithm with a probabilistic Local Search refinement phase.

    Every call to `fitness_func` consumes one unit from the same shared global
    evaluation budget.

    Args:
        fitness_func:
            Function receiving a binary selection mask and returning its
            evaluation metrics.

        total_instances:
            Total number of candidate training instances.

        population_size:
            Number of individuals in each complete population.

        keep_percentage:
            Fraction of selected instances maintained by every solution.

        max_evaluations:
            Exact maximum number of fitness evaluations.

        tournament_size:
            Number of individuals participating in each tournament.

        mutation_probability:
            Probability of applying mutation to each offspring.

        mutation_fraction:
            Fraction of selected instances exchanged during mutation.

        local_search_probability:
            Probability of activating Local Search for an evaluated offspring.

        local_search_steps:
            Maximum number of additional neighbors evaluated during one Local
            Search phase.

        local_search_num_swaps:
            Number of fixed-size swaps used to generate each Local Search
            neighbor.

        target_metric:
            Metric used as the optimization objective.

        rng:
            Optional pseudo-random number generator.

    Returns:
        A tuple containing the best solution, best fitness, complete evaluation
        history, running best-fitness history, and number of evaluations.
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

    if population_size < 2:
        raise ValueError(
            f"'population_size' must be at least 2, "
            f"but received {population_size}."
        )

    if max_evaluations < population_size:
        raise ValueError(
            "'max_evaluations' must be greater than or equal to "
            "'population_size'."
        )

    if tournament_size <= 0:
        raise ValueError(
            f"'tournament_size' must be greater than 0, "
            f"but received {tournament_size}."
        )

    if tournament_size > population_size - 1:
        raise ValueError(
            f"'tournament_size' must be less than or equal to "
            f"population_size - 1 ({population_size - 1})."
        )

    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError(
            f"'mutation_probability' must be in the interval [0, 1], "
            f"but received {mutation_probability}."
        )

    if not 0.0 < mutation_fraction <= 1.0:
        raise ValueError(
            f"'mutation_fraction' must be in the interval (0, 1], "
            f"but received {mutation_fraction}."
        )

    if not 0.0 <= local_search_probability <= 1.0:
        raise ValueError(
            f"'local_search_probability' must be in the interval [0, 1], "
            f"but received {local_search_probability}."
        )

    if local_search_steps < 0:
        raise ValueError(
            f"'local_search_steps' must be greater than or equal to 0, "
            f"but received {local_search_steps}."
        )

    if (
        local_search_probability > 0.0
        and local_search_steps == 0
    ):
        raise ValueError(
            "'local_search_steps' must be greater than 0 when "
            "'local_search_probability' is greater than 0."
        )

    if local_search_num_swaps <= 0:
        raise ValueError(
            f"'local_search_num_swaps' must be greater than 0, "
            f"but received {local_search_num_swaps}."
        )

    subset_size = validate_search_subset_size(
        total_instances=total_instances,
        keep_percentage=keep_percentage,
    )

    max_feasible_local_swaps = min(
        subset_size,
        total_instances - subset_size,
    )

    if (
        local_search_num_swaps
        > max_feasible_local_swaps
    ):
        raise ValueError(
            f"'local_search_num_swaps'={local_search_num_swaps} exceeds "
            f"the maximum feasible number of swaps "
            f"({max_feasible_local_swaps})."
        )

    algorithm_id = Algorithm.MEMETIC.value
    metric_name = target_metric.value

    generator = (
        rng
        if rng is not None
        else random
    )

    population: list[Mask] = []
    population_ids: list[int] = []
    fitness_values: list[float] = []

    fitness_history: list[HistoryEntry] = []
    best_fitness_history: list[float] = []

    best_solution: Mask = {}
    best_fitness = float("-inf")
    best_solution_id: int | None = None
    best_found_at_evaluation: int | None = None

    evaluations_done = 0
    generation = 0

    logger.info(
        "Starting %s | target_metric=%s | keep_percentage=%.4f | "
        "population_size=%d | max_evaluations=%d",
        algorithm_id,
        metric_name,
        keep_percentage,
        population_size,
        max_evaluations,
    )

    # --------------------------------------------------------------------------
    # Initial population
    # --------------------------------------------------------------------------

    for population_slot in range(
        1,
        population_size + 1,
    ):
        individual = create_random_mask(
            total_instances=total_instances,
            keep_percentage=keep_percentage,
            rng=rng,
        )

        metrics_result = dict(
            fitness_func(individual)
        )

        fitness = extract_fitness(
            metrics=metrics_result,
            metric_name=metric_name,
        )

        evaluations_done += 1
        solution_id = evaluations_done

        population.append(
            individual.copy()
        )

        population_ids.append(
            solution_id
        )

        fitness_values.append(
            fitness
        )

        improved_best = (
            fitness > best_fitness
        )

        if improved_best:
            best_solution = individual.copy()
            best_fitness = fitness
            best_solution_id = solution_id
            best_found_at_evaluation = evaluations_done

        selected_instances = sum(
            individual.values()
        )

        history_entry: HistoryEntry = dict(
            metrics_result
        )

        history_entry.update(
            {
                "Algorithm": algorithm_id,
                "Evaluation": evaluations_done,
                "Solution ID": solution_id,
                "Generation": generation,
                "Population Slot": population_slot,
                "Candidate Type": "initial",
                "Initial Percentage": keep_percentage,
                "Final Percentage":
                    selected_instances / total_instances,
                "Selected Instances": selected_instances,
                "Target Metric": metric_name,
                "Fitness": fitness,
                "Best Fitness": best_fitness,
                "Improved Best": improved_best,
                "Best Solution ID": best_solution_id,
                "Best Found At Evaluation":
                    best_found_at_evaluation,
                "Parent 1 Index": None,
                "Parent 2 Index": None,
                "Parent 1 ID": None,
                "Parent 2 ID": None,
                "Parent 1 Fitness": None,
                "Parent 2 Fitness": None,
                "Parent Hamming Distance": None,
                "Offspring Number": None,
                "Crossover Hamming Parent 1": None,
                "Crossover Hamming Parent 2": None,
                "Mutation Applied": False,
                "Mutation Swaps": 0,
                "Mutation Hamming Distance": 0,
                "Local Search Applied": False,
                "Local Search Origin Evaluation": None,
                "Local Search Origin Solution ID": None,
                "Local Search Step": 0,
                "Local Search Accepted": None,
                "Local Fitness Before": None,
                "Local Fitness After": None,
                "Local Search Hamming Distance": 0,
                "Local Current Solution ID Before": None,
                "Local Current Solution ID After": None,
            }
        )

        fitness_history.append(
            history_entry
        )

        best_fitness_history.append(
            best_fitness
        )

    # --------------------------------------------------------------------------
    # Memetic evolutionary process
    # --------------------------------------------------------------------------

    while evaluations_done < max_evaluations:
        generation += 1

        elite_index = max(
            range(len(population)),
            key=lambda index: fitness_values[index],
        )

        new_population: list[Mask] = [
            population[elite_index].copy()
        ]

        new_population_ids: list[int] = [
            population_ids[elite_index]
        ]

        new_fitness_values: list[float] = [
            fitness_values[elite_index]
        ]

        while (
            len(new_population) < population_size
            and evaluations_done < max_evaluations
        ):
            parent1_index = tournament_selection(
                fitness_values=fitness_values,
                tournament_size=tournament_size,
                rng=rng,
            )

            parent2_index = tournament_selection(
                fitness_values=fitness_values,
                tournament_size=tournament_size,
                rng=rng,
                excluded_index=parent1_index,
            )

            parent1 = population[parent1_index]
            parent2 = population[parent2_index]

            parent1_id = population_ids[parent1_index]
            parent2_id = population_ids[parent2_index]

            parent1_fitness = fitness_values[parent1_index]
            parent2_fitness = fitness_values[parent2_index]

            parent_hamming = hamming_distance(
                parent1,
                parent2,
            )

            child1, child2 = fixed_size_crossover(
                parent1=parent1,
                parent2=parent2,
                rng=rng,
            )

            for offspring_number, crossover_child in enumerate(
                (child1, child2),
                start=1,
            ):
                if (
                    len(new_population) >= population_size
                    or evaluations_done >= max_evaluations
                ):
                    break

                population_slot = (
                    len(new_population) + 1
                )

                child_before_mutation = (
                    crossover_child.copy()
                )

                child = mutate_mask(
                    mask=crossover_child,
                    mutation_probability=mutation_probability,
                    mutation_fraction=mutation_fraction,
                    rng=rng,
                )

                mutation_hamming = hamming_distance(
                    child_before_mutation,
                    child,
                )

                # --------------------------------------------------------------
                # Evaluate offspring
                # --------------------------------------------------------------

                child_metrics = dict(
                    fitness_func(child)
                )

                child_fitness = extract_fitness(
                    metrics=child_metrics,
                    metric_name=metric_name,
                )

                evaluations_done += 1
                child_solution_id = evaluations_done

                improved_best = (
                    child_fitness > best_fitness
                )

                if improved_best:
                    best_solution = child.copy()
                    best_fitness = child_fitness
                    best_solution_id = child_solution_id
                    best_found_at_evaluation = evaluations_done

                # Local Search is considered applied only when there is enough
                # global budget to evaluate at least one additional neighbor.
                local_search_applied = (
                    local_search_steps > 0
                    and evaluations_done < max_evaluations
                    and generator.random()
                    < local_search_probability
                )

                selected_instances = sum(
                    child.values()
                )

                offspring_history: HistoryEntry = dict(
                    child_metrics
                )

                offspring_history.update(
                    {
                        "Algorithm": algorithm_id,
                        "Evaluation": evaluations_done,
                        "Solution ID": child_solution_id,
                        "Generation": generation,
                        "Population Slot": population_slot,
                        "Candidate Type": "offspring",
                        "Initial Percentage":
                            keep_percentage,
                        "Final Percentage":
                            selected_instances
                            / total_instances,
                        "Selected Instances":
                            selected_instances,
                        "Target Metric":
                            metric_name,
                        "Fitness":
                            child_fitness,
                        "Best Fitness":
                            best_fitness,
                        "Improved Best":
                            improved_best,
                        "Best Solution ID":
                            best_solution_id,
                        "Best Found At Evaluation":
                            best_found_at_evaluation,
                        "Parent 1 Index":
                            parent1_index + 1,
                        "Parent 2 Index":
                            parent2_index + 1,
                        "Parent 1 ID":
                            parent1_id,
                        "Parent 2 ID":
                            parent2_id,
                        "Parent 1 Fitness":
                            parent1_fitness,
                        "Parent 2 Fitness":
                            parent2_fitness,
                        "Parent Hamming Distance":
                            parent_hamming,
                        "Offspring Number":
                            offspring_number,
                        "Crossover Hamming Parent 1":
                            hamming_distance(
                                parent1,
                                child_before_mutation,
                            ),
                        "Crossover Hamming Parent 2":
                            hamming_distance(
                                parent2,
                                child_before_mutation,
                            ),
                        "Mutation Applied":
                            mutation_hamming > 0,
                        "Mutation Swaps":
                            mutation_hamming // 2,
                        "Mutation Hamming Distance":
                            mutation_hamming,
                        "Local Search Applied":
                            local_search_applied,
                        "Local Search Origin Evaluation":
                            (
                                evaluations_done
                                if local_search_applied
                                else None
                            ),
                        "Local Search Origin Solution ID":
                            (
                                child_solution_id
                                if local_search_applied
                                else None
                            ),
                        "Local Search Step": 0,
                        "Local Search Accepted": None,
                        "Local Fitness Before": None,
                        "Local Fitness After":
                            child_fitness,
                        "Local Search Hamming Distance": 0,
                        "Local Current Solution ID Before":
                            None,
                        "Local Current Solution ID After":
                            child_solution_id,
                    }
                )

                fitness_history.append(
                    offspring_history
                )

                best_fitness_history.append(
                    best_fitness
                )

                # Current refined state.
                refined_solution = child.copy()
                refined_fitness = child_fitness
                refined_solution_id = child_solution_id

                # --------------------------------------------------------------
                # Optional Local Search
                # --------------------------------------------------------------

                if local_search_applied:
                    origin_evaluation = (
                        child_solution_id
                    )

                    origin_solution_id = (
                        child_solution_id
                    )

                    for local_step in range(
                        1,
                        local_search_steps + 1,
                    ):
                        if evaluations_done >= max_evaluations:
                            break

                        local_fitness_before = (
                            refined_fitness
                        )

                        local_solution_id_before = (
                            refined_solution_id
                        )

                        neighbor = generate_neighbor(
                            mask=refined_solution,
                            num_swaps=
                                local_search_num_swaps,
                            rng=rng,
                        )

                        local_hamming = hamming_distance(
                            refined_solution,
                            neighbor,
                        )

                        neighbor_metrics = dict(
                            fitness_func(neighbor)
                        )

                        neighbor_fitness = extract_fitness(
                            metrics=neighbor_metrics,
                            metric_name=metric_name,
                        )

                        evaluations_done += 1
                        neighbor_solution_id = (
                            evaluations_done
                        )

                        local_accepted = (
                            neighbor_fitness
                            >= refined_fitness
                        )

                        if local_accepted:
                            refined_solution = (
                                neighbor.copy()
                            )

                            refined_fitness = (
                                neighbor_fitness
                            )

                            refined_solution_id = (
                                neighbor_solution_id
                            )

                        improved_best = (
                            neighbor_fitness
                            > best_fitness
                        )

                        if improved_best:
                            best_solution = (
                                neighbor.copy()
                            )

                            best_fitness = (
                                neighbor_fitness
                            )

                            best_solution_id = (
                                neighbor_solution_id
                            )

                            best_found_at_evaluation = (
                                evaluations_done
                            )

                            logger.info(
                                "[%s] New best during Local Search at "
                                "evaluation %d/%d | generation=%d | "
                                "%s=%.6f",
                                algorithm_id,
                                evaluations_done,
                                max_evaluations,
                                generation,
                                metric_name,
                                best_fitness,
                            )

                        selected_instances = sum(
                            neighbor.values()
                        )

                        local_history: HistoryEntry = dict(
                            neighbor_metrics
                        )

                        local_history.update(
                            {
                                "Algorithm":
                                    algorithm_id,
                                "Evaluation":
                                    evaluations_done,
                                "Solution ID":
                                    neighbor_solution_id,
                                "Generation":
                                    generation,
                                "Population Slot":
                                    population_slot,
                                "Candidate Type":
                                    "local_neighbor",
                                "Initial Percentage":
                                    keep_percentage,
                                "Final Percentage":
                                    selected_instances
                                    / total_instances,
                                "Selected Instances":
                                    selected_instances,
                                "Target Metric":
                                    metric_name,
                                "Fitness":
                                    neighbor_fitness,
                                "Best Fitness":
                                    best_fitness,
                                "Improved Best":
                                    improved_best,
                                "Best Solution ID":
                                    best_solution_id,
                                "Best Found At Evaluation":
                                    best_found_at_evaluation,
                                "Parent 1 Index":
                                    parent1_index + 1,
                                "Parent 2 Index":
                                    parent2_index + 1,
                                "Parent 1 ID":
                                    parent1_id,
                                "Parent 2 ID":
                                    parent2_id,
                                "Parent 1 Fitness":
                                    parent1_fitness,
                                "Parent 2 Fitness":
                                    parent2_fitness,
                                "Parent Hamming Distance":
                                    parent_hamming,
                                "Offspring Number":
                                    offspring_number,
                                "Crossover Hamming Parent 1":
                                    hamming_distance(
                                        parent1,
                                        child_before_mutation,
                                    ),
                                "Crossover Hamming Parent 2":
                                    hamming_distance(
                                        parent2,
                                        child_before_mutation,
                                    ),
                                "Mutation Applied":
                                    mutation_hamming > 0,
                                "Mutation Swaps":
                                    mutation_hamming // 2,
                                "Mutation Hamming Distance":
                                    mutation_hamming,
                                "Local Search Applied":
                                    True,
                                "Local Search Origin Evaluation":
                                    origin_evaluation,
                                "Local Search Origin Solution ID":
                                    origin_solution_id,
                                "Local Search Step":
                                    local_step,
                                "Local Search Accepted":
                                    local_accepted,
                                "Local Fitness Before":
                                    local_fitness_before,
                                "Local Fitness After":
                                    refined_fitness,
                                "Local Search Hamming Distance":
                                    local_hamming,
                                "Local Current Solution ID Before":
                                    local_solution_id_before,
                                "Local Current Solution ID After":
                                    refined_solution_id,
                            }
                        )

                        fitness_history.append(
                            local_history
                        )

                        best_fitness_history.append(
                            best_fitness
                        )

                new_population.append(
                    refined_solution.copy()
                )

                new_population_ids.append(
                    refined_solution_id
                )

                new_fitness_values.append(
                    refined_fitness
                )

        population = new_population
        population_ids = new_population_ids
        fitness_values = new_fitness_values

    logger.info(
        "[%s] Finished | evaluations=%d | generations=%d | "
        "best_%s=%.6f",
        algorithm_id,
        evaluations_done,
        generation,
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