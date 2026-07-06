import random
from typing import Callable, Tuple, List, Dict

from utils.selection_utils import create_random_mask, mutate_mask
from utils.config import Metric, Algorithm

# Sincronizados con las optimizaciones O(1) de los módulos previos
from algorithms.genetic import tournament_selection, weighted_crossover
from algorithms.local_search import generate_neighbor


def memetic_algorithm(
    fitness_func: Callable[[dict], dict],
    total_instances: int,
    population_size: int = 20,
    keep_percentage: float = 0.1,
    max_evaluations: int = 50,
    patience: int = 10,
    tournament_size: int = 3,
    mutation_rate: float = 0.1,
    local_search_probability: float = 0.2,
    local_search_evaluations: int = 10,
    local_search_neighbor_size: int = 5,
    target_metric: Metric = Metric.ACCURACY,
    adjust_size: bool = False
) -> Tuple[Dict[int, int], float, List[dict], List[float], int]:
    """
    Implements a Memetic Algorithm (MA) for Instance Selection.
    Combines a global Genetic Algorithm with a Local Search phase to refine individuals.

    Args:
        fitness_func: Function that receives a mask and returns the metrics.
        total_instances: Total number of images in the original dataset.
        population_size: Number of individuals in the population.
        keep_percentage: Initial percentage of images to select.
        max_evaluations: Maximum number of overall evaluations allowed.
        patience: Stopping criterion if no improvements are found.
        tournament_size: Size of the tournament for parent selection.
        mutation_rate: Probability of mutating a newly created offspring.
        local_search_probability: Probability of applying local search to an offspring.
        local_search_evaluations: Max evaluations permitted within a single local search.
        local_search_neighbor_size: Number of instances modified to generate a neighbor.
        target_metric: Metric to optimize.
        adjust_size: Flag to determine if masks can dynamically fluctuate in size.

    Returns:
        Tuple containing: (Best Mask, Best Fitness, Metrics History, Best Fitness History, Total Evaluations)
    """
    evaluations_done = 0
    
    # Selección dinámica de la etiqueta del algoritmo según configuración central
    alg_id = Algorithm.FREE_MEMETIC.value if adjust_size else Algorithm.MEMETIC.value

    def local_search_improvement_with_limit(individual: dict, remaining_evaluations: int):
        """Internal helper to apply localized Hill Climbing to an offspring with metadata tracking."""
        nonlocal evaluations_done
        
        current_solution = individual.copy()
        current_fitness_dict = fitness_func(current_solution)
        evaluations_done += 1
        local_evals = 1

        # Enriquecimiento inicial de metadatos locales
        current_pct = sum(current_solution.values()) / total_instances
        base_hist = current_fitness_dict.copy()
        base_hist.update({
            "Algorithm": alg_id,
            "Initial Percentage": keep_percentage,
            "Final Percentage": current_pct,
            "Iteration": evaluations_done
        })

        current_fitness = current_fitness_dict.get(target_metric.value, 0.0)
        best_solution_local = current_solution.copy()
        best_fitness_local = current_fitness
        best_fitness_dict_local = base_hist.copy()
        fitness_history_local = [base_hist]

        max_local_evals = min(local_search_evaluations, remaining_evaluations)

        while local_evals < max_local_evals and evaluations_done < max_evaluations:
            # CORRECCIÓN: Pasamos 'adjust_size' de forma nativa para respetar variantes libres/acotadas
            neighbor = generate_neighbor(current_solution, local_search_neighbor_size, adjust_size)
            neighbor_fitness_dict = fitness_func(neighbor)
            
            evaluations_done += 1
            local_evals += 1
            
            # Inyección de metadatos en la historia profunda del vecindario
            neighbor_pct = sum(neighbor.values()) / total_instances
            hist_entry = neighbor_fitness_dict.copy()
            hist_entry.update({
                "Algorithm": alg_id,
                "Initial Percentage": keep_percentage,
                "Final Percentage": neighbor_pct,
                "Iteration": evaluations_done
            })
            fitness_history_local.append(hist_entry)
            
            neighbor_fitness = neighbor_fitness_dict.get(target_metric.value, 0.0)

            if neighbor_fitness > current_fitness:
                current_solution = neighbor.copy()
                current_fitness = neighbor_fitness

                if current_fitness > best_fitness_local:
                    best_solution_local = current_solution.copy()
                    best_fitness_local = current_fitness
                    best_fitness_dict_local = hist_entry.copy()

        return best_solution_local, best_fitness_local, best_fitness_dict_local, fitness_history_local


    print(f"Starting {alg_id} (Target: {target_metric.value.upper()} | Initial Retention: {keep_percentage*100}%)")

    # 1. Generate and evaluate the initial population
    population = [create_random_mask(total_instances, keep_percentage) for _ in range(population_size)]
    fitness_dicts = []
    fitness_history = []
    
    for idx, ind in enumerate(population):
        if evaluations_done >= max_evaluations:
            break
        evaluations_done += 1
        print(f"--- [{alg_id}] Initial Pop Evaluation {evaluations_done}/{population_size} ---")
        
        f_dict = fitness_func(ind)
        
        # Inyección estructural de metadatos iniciales
        current_pct = sum(ind.values()) / total_instances
        hist_entry = f_dict.copy()
        hist_entry.update({
            "Algorithm": alg_id,
            "Initial Percentage": keep_percentage,
            "Final Percentage": current_pct,
            "Iteration": evaluations_done
        })
        fitness_dicts.append(hist_entry)
        fitness_history.append(hist_entry)
        
    if not fitness_dicts:
        return {}, 0.0, [], [], evaluations_done

    fitness_values = [f_dict[target_metric.value] for f_dict in fitness_dicts]

    # Track best individual globally
    best_fitness_idx = fitness_values.index(max(fitness_values))
    best_individual = population[best_fitness_idx].copy()
    best_fitness = fitness_values[best_fitness_idx]
    best_fitness_dict = fitness_dicts[best_fitness_idx].copy()

    best_fitness_history = [best_fitness]
    evaluations_without_improvement = 0

    # 2. Main Evolutionary Loop
    while evaluations_done < max_evaluations:
        new_population = []
        new_fitness_dicts = []

        # Elitism
        new_population.append(best_individual.copy())
        new_fitness_dicts.append(best_fitness_dict.copy())

        while len(new_population) < population_size and evaluations_done < max_evaluations:
            # Selección veloz O(1) basada en índices enteros
            idx1 = tournament_selection(fitness_values, tournament_size)
            idx2 = tournament_selection(fitness_values, tournament_size)

            parent1 = population[idx1]
            parent2 = population[idx2]
            fitness1 = fitness_values[idx1]
            fitness2 = fitness_values[idx2]
            
            # Crossover
            child1, child2 = weighted_crossover(parent1, parent2, fitness1, fitness2, adjust_size)

            # Mutation
            child1 = mutate_mask(child1, mutation_rate, is_constrained=not adjust_size)
            child2 = mutate_mask(child2, mutation_rate, is_constrained=not adjust_size)

            # Process offspring
            for child in [child1, child2]:
                if len(new_population) < population_size and evaluations_done < max_evaluations:
                    
                    # --- MEMETIC REFINEMENT PHASE ---
                    if random.random() < local_search_probability:
                        print(f"--- [{alg_id}] Launching Local Search Phase at Eval {evaluations_done + 1} ---")
                        (
                            improved_child,
                            child_fitness,
                            child_f_dict,
                            local_fitness_history
                        ) = local_search_improvement_with_limit(
                            child,
                            max_evaluations - evaluations_done
                        )
                        new_population.append(improved_child)
                        new_fitness_dicts.append(child_f_dict)
                        fitness_history.extend(local_fitness_history)
                    
                    # --- STANDARD GENETIC EVALUATION PHASE ---
                    else:
                        evaluations_done += 1
                        print(f"--- [{alg_id}] Evaluation {evaluations_done}/{max_evaluations} ---")
                        child_fitness_dict = fitness_func(child)
                        
                        child_pct = sum(child.values()) / total_instances
                        hist_entry = child_fitness_dict.copy()
                        hist_entry.update({
                            "Algorithm": alg_id,
                            "Initial Percentage": keep_percentage,
                            "Final Percentage": child_pct,
                            "Iteration": evaluations_done
                        })
                        new_population.append(child)
                        new_fitness_dicts.append(hist_entry)
                        fitness_history.append(hist_entry)

        # Update generational states safely
        population = new_population
        fitness_dicts = new_fitness_dicts
        fitness_values = [f_dict[target_metric.value] for f_dict in fitness_dicts]

        # Check for global improvements
        current_best_idx = fitness_values.index(max(fitness_values))
        if fitness_values[current_best_idx] > best_fitness:
            best_individual = population[current_best_idx].copy()
            best_fitness = fitness_values[current_best_idx]
            best_fitness_dict = fitness_dicts[current_best_idx].copy()
            evaluations_without_improvement = 0
            print(f"New global best solution found! {target_metric.value.capitalize()}: {best_fitness:.4f}")
        else:
            evaluations_without_improvement += 1

        best_fitness_history.append(best_fitness)
        print(f"Generation metrics -> Evals: {evaluations_done}/{max_evaluations} | Best Fitness: {best_fitness:.4f} | Stagnation: {evaluations_without_improvement}/{patience}")

        if evaluations_without_improvement >= patience:
            print(f"Search terminated due to stagnation after {evaluations_done} evaluations.")
            break

    print(f"[{alg_id}] Execution Finished. Best {target_metric.value}: {best_fitness:.4f}")
    return best_individual, best_fitness, fitness_history, best_fitness_history, evaluations_done