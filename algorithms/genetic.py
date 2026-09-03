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
    validate_same_index_space,
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
# CROSSOVER
# ==============================================================================

def fixed_size_crossover(
    parent1: Mask,
    parent2: Mask,
    rng: random.Random | None = None,
) -> tuple[Mask, Mask]:
    """
    Generate two offspring using cardinality-preserving set-based crossover.

    Instances selected by both parents are inherited by both offspring.

    The selected instances on which the parents disagree are randomly divided
    between the two offspring.

    Both offspring preserve exactly the same number of selected instances as
    their parents.

    Args:
        parent1:
            First parent selection mask.

        parent2:
            Second parent selection mask.

        rng:
            Optional pseudo-random number generator.

    Returns:
        Two offspring with the same subset cardinality as the parents.

    Raises:
        ValueError:
            If the parent masks are invalid or incompatible.
    """
    validate_mask(parent1)
    validate_mask(parent2)

    validate_same_index_space(
        mask1=parent1,
        mask2=parent2,
    )

    selected_parent1 = {
        index
        for index, selected in parent1.items()
        if selected == 1
    }

    selected_parent2 = {
        index
        for index, selected in parent2.items()
        if selected == 1
    }

    if len(selected_parent1) != len(selected_parent2):
        raise ValueError(
            "Both parents must select exactly the same number of instances."
        )

    generator = (
        rng
        if rng is not None
        else random
    )

    common_selected = (
        selected_parent1
        & selected_parent2
    )

    # Sorting removes any dependency on the internal iteration order of sets,
    # improving reproducibility across executions and environments.
    differing_selected = sorted(
        selected_parent1
        ^ selected_parent2
    )

    target_subset_size = len(
        selected_parent1
    )

    num_to_select = (
        target_subset_size
        - len(common_selected)
    )

    if num_to_select == 0:
        return (
            parent1.copy(),
            parent2.copy(),
        )

    child1_unique = set(
        generator.sample(
            differing_selected,
            k=num_to_select,
        )
    )

    child2_unique = (
        set(differing_selected)
        - child1_unique
    )

    child1_selected = (
        common_selected
        | child1_unique
    )

    child2_selected = (
        common_selected
        | child2_unique
    )

    child1 = {
        index: int(index in child1_selected)
        for index in parent1
    }

    child2 = {
        index: int(index in child2_selected)
        for index in parent1
    }

    return (
        child1,
        child2,
    )


# ==============================================================================
# TOURNAMENT SELECTION
# ==============================================================================

def tournament_selection(
    fitness_values: list[float],
    tournament_size: int = 3,
    rng: random.Random | None = None,
    excluded_index: int | None = None,
) -> int:
    """
    Select one individual using tournament selection.

    Args:
        fitness_values:
            Fitness value associated with each population individual.

        tournament_size:
            Number of individuals participating in the tournament.

        rng:
            Optional pseudo-random number generator.

        excluded_index:
            Optional population index excluded from the tournament.

    Returns:
        Index of the tournament winner.
    """
    if not fitness_values:
        raise ValueError(
            "'fitness_values' must contain at least one value."
        )

    if tournament_size <= 0:
        raise ValueError(
            f"'tournament_size' must be greater than 0, "
            f"but received {tournament_size}."
        )

    available_indices = [
        index
        for index in range(len(fitness_values))
        if index != excluded_index
    ]

    if tournament_size > len(available_indices):
        raise ValueError(
            f"'tournament_size'={tournament_size} exceeds the number of "
            f"available individuals ({len(available_indices)})."
        )

    generator = (
        rng
        if rng is not None
        else random
    )

    tournament_indices = generator.sample(
        available_indices,
        k=tournament_size,
    )

    return max(
        tournament_indices,
        key=lambda index: fitness_values[index],
    )


# ==============================================================================
# GENETIC ALGORITHM
# ==============================================================================

def genetic_algorithm(
    fitness_func: Callable[[Mask], Metrics],
    total_instances: int,
    population_size: int = 10,
    keep_percentage: float = 0.1,
    max_evaluations: int = 100,
    tournament_size: int = 3,
    mutation_probability: float = 0.1,
    mutation_fraction: float = 0.05,
    target_metric: Metric = Metric.ACCURACY,
    rng: random.Random | None = None,
) -> tuple[Mask, float, list[HistoryEntry], list[float], int]:
    """
    Perform a fixed-size Genetic Algorithm for instance selection.

    The algorithm uses:

    - Random fixed-size initialization.
    - Tournament selection.
    - Cardinality-preserving crossover.
    - Fixed-size swap mutation.
    - One-individual elitism.
    - A shared fixed fitness-evaluation budget.

    Args:
        fitness_func:
            Function receiving a binary selection mask and returning its
            evaluation metrics.

        total_instances:
            Total number of candidate training instances.

        population_size:
            Number of individuals in each complete population.

        keep_percentage:
            Fraction of instances selected by every individual.

        max_evaluations:
            Exact maximum number of fitness evaluations.

        tournament_size:
            Number of individuals participating in each tournament.

        mutation_probability:
            Probability of applying mutation to each offspring.

        mutation_fraction:
            Fraction of selected instances exchanged when mutation occurs.

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

    validate_search_subset_size(
        total_instances=total_instances,
        keep_percentage=keep_percentage,
    )

    algorithm_id = Algorithm.GENETIC.value
    metric_name = target_metric.value

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
            }
        )

        fitness_history.append(
            history_entry
        )

        best_fitness_history.append(
            best_fitness
        )

    # --------------------------------------------------------------------------
    # Evolutionary process
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

                metrics_result = dict(
                    fitness_func(child)
                )

                fitness = extract_fitness(
                    metrics=metrics_result,
                    metric_name=metric_name,
                )

                evaluations_done += 1
                solution_id = evaluations_done

                new_population.append(
                    child.copy()
                )

                new_population_ids.append(
                    solution_id
                )

                new_fitness_values.append(
                    fitness
                )

                improved_best = (
                    fitness > best_fitness
                )

                if improved_best:
                    best_solution = child.copy()
                    best_fitness = fitness
                    best_solution_id = solution_id
                    best_found_at_evaluation = evaluations_done

                    logger.info(
                        "[%s] New best at evaluation %d/%d | "
                        "generation=%d | %s=%.6f",
                        algorithm_id,
                        evaluations_done,
                        max_evaluations,
                        generation,
                        metric_name,
                        best_fitness,
                    )

                selected_instances = sum(
                    child.values()
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
                        "Population Slot":
                            len(new_population),
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
                            fitness,
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
                    }
                )

                fitness_history.append(
                    history_entry
                )

                best_fitness_history.append(
                    best_fitness
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