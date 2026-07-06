import random
from typing import Callable, Tuple, List, Dict

from utils.selection_utils import create_random_mask, mutate_mask
from utils.config import Metric, Algorithm

def weighted_crossover(
    parent1: Dict[int, int], 
    parent2: Dict[int, int], 
    fitness1: float = 1.0, 
    fitness2: float = 1.0, 
    adjust_size: bool = False
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Performs crossover between two parents to generate two offspring.
    
    Args:
        parent1, parent2: The mask dictionaries of the parents.
        fitness1, fitness2: Fitness scores of the parents used to bias trait selection.
        adjust_size: If False, strictly maintains the exact number of selected images.
                     If True, performs a uniform/weighted crossover where size can fluctuate.
                     
    Returns:
        Two new child masks.
    """
    child1 = parent1.copy()
    child2 = parent2.copy()

    if adjust_size:
        # Flexible size: Use a probabilistic approach to inherit traits symmetrically
        total_fitness = fitness1 + fitness2
        prob1 = fitness1 / total_fitness if total_fitness > 0 else 0.5
        
        for key in child1.keys():
            # If parent2 is preferred or wins the roll, swap genes symmetrically
            if random.random() > prob1:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
                
    else:
        # Strict size: Swap exact differences to maintain the same total number of 1s
        selected1 = set(img for img, val in parent1.items() if val == 1)
        selected2 = set(img for img, val in parent2.items() if val == 1)

        only_in_1 = list(selected1 - selected2)
        only_in_2 = list(selected2 - selected1)

        num_swap = min(len(only_in_1), len(only_in_2)) // 2

        if num_swap > 0:
            swap_from_1 = random.sample(only_in_1, num_swap)
            swap_from_2 = random.sample(only_in_2, num_swap)

            # Perform the gene swap
            for img1, img2 in zip(swap_from_1, swap_from_2):
                child1[img1] = 0
                child1[img2] = 1

                child2[img2] = 0
                child2[img1] = 1

    return child1, child2


def tournament_selection(
    fitness_values: List[float], 
    tournament_size: int = 3
) -> int:
    """
    Selects an individual from the population using tournament selection.
    Returns the integer INDEX of the winner to guarantee O(1) performance.
    """
    tournament_indices = random.sample(range(len(fitness_values)), tournament_size)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    
    winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return winner_idx


def genetic_algorithm(
    fitness_func: Callable[[dict], dict],
    total_instances: int,
    population_size: int = 10,
    keep_percentage: float = 0.1,
    max_evaluations: int = 50,
    patience: int = 10,
    tournament_size: int = 3,
    mutation_rate: float = 0.1,
    target_metric: Metric = Metric.ACCURACY,
    adjust_size: bool = False
) -> Tuple[Dict[int, int], float, List[dict], List[float], int]:
    """
    Implements a Genetic Algorithm for instance selection.

    Args:
        fitness_func: Function that receives a mask and returns the metrics.
        total_instances: Total number of images in the original dataset.
        population_size: Number of individuals in the population.
        keep_percentage: Initial percentage of images to select.
        max_evaluations: Maximum number of overall evaluations allowed.
        patience: Stopping criterion if no improvements are found.
        tournament_size: Size of the tournament for parent selection.
        mutation_rate: Probability of mutating a newly created offspring.
        target_metric: Metric to optimize.
        adjust_size: Flag to determine if crossover should allow subset size fluctuation.

    Returns:
        Tuple containing: (Best Mask, Best Fitness, Metrics History, Best Fitness History, Total Evaluations)
    """
    alg_id = Algorithm.FREE_GENETIC_V2.value if adjust_size else Algorithm.GENETIC.value
    print(f"Starting {alg_id} (Target: {target_metric.value.upper()} | Initial Retention: {keep_percentage*100}%)")

    # 1. Generate and evaluate the initial population step-by-step
    initial_pop_size = min(population_size, max_evaluations)
    population = [create_random_mask(total_instances, keep_percentage) for _ in range(initial_pop_size)]
    
    fitness_dicts = []
    fitness_history = []
    evaluations_done = 0

    for ind in population:
        evaluations_done += 1
        print(f"--- [{alg_id}] Initial Pop Evaluation {evaluations_done}/{initial_pop_size} ---")
        f_dict = fitness_func(ind)
        
        # Metadata Injection for tracking
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
        
    fitness_values = [f_dict[target_metric.value] for f_dict in fitness_dicts]

    # Track best individual
    best_fitness_idx = fitness_values.index(max(fitness_values))
    best_individual = population[best_fitness_idx].copy()
    best_fitness = fitness_values[best_fitness_idx]
    
    best_fitness_history = [best_fitness]
    evaluations_without_improvement = 0

    # 2. Main Evolutionary Loop
    while evaluations_done < max_evaluations:
        new_population = []
        new_fitness_dicts = []

        # Elitism: carry over the best individual to the next generation
        new_population.append(best_individual.copy())
        new_fitness_dicts.append(fitness_dicts[best_fitness_idx])

        # Generate new population
        while len(new_population) < population_size and evaluations_done < max_evaluations:
            
            # Selection (Returns indices -> O(1) performance)
            idx1 = tournament_selection(fitness_values, tournament_size)
            idx2 = tournament_selection(fitness_values, tournament_size)
            
            parent1 = population[idx1]
            parent2 = population[idx2]
            fitness1 = fitness_values[idx1]
            fitness2 = fitness_values[idx2]

            # Crossover
            child1, child2 = weighted_crossover(parent1, parent2, fitness1, fitness2, adjust_size)

            # Mutation (Clean linkage to selection_utils mutation rules)
            child1 = mutate_mask(child1, mutation_rate, is_constrained=not adjust_size)
            child2 = mutate_mask(child2, mutation_rate, is_constrained=not adjust_size)

            # Evaluation
            for child in [child1, child2]:
                if len(new_population) < population_size and evaluations_done < max_evaluations:
                    evaluations_done += 1
                    print(f"--- [{alg_id}] Evaluation {evaluations_done}/{max_evaluations} ---")
                    
                    child_fitness_dict = fitness_func(child)
                    
                    # Metadata Injection
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

        # Update population state
        population = new_population
        fitness_dicts = new_fitness_dicts
        fitness_values = [f_dict[target_metric.value] for f_dict in fitness_dicts]

        # Check for global improvements
        current_best_idx = fitness_values.index(max(fitness_values))
        if fitness_values[current_best_idx] > best_fitness:
            best_individual = population[current_best_idx].copy()
            best_fitness = fitness_values[current_best_idx]
            best_fitness_idx = current_best_idx
            evaluations_without_improvement = 0
            print(f"New best solution found. {target_metric.value.capitalize()}: {best_fitness:.4f}")
        else:
            evaluations_without_improvement += 1
            print(f"Generation finished without improvement. Patience: {evaluations_without_improvement}/{patience}")

        best_fitness_history.append(best_fitness)

        # Check stagnation
        if evaluations_without_improvement >= patience:
            print(f"Search terminated due to stagnation after {evaluations_done} evaluations.")
            break

    print(f"[{alg_id}] Finished. Best {target_metric.value}: {best_fitness:.4f}")
    return best_individual, best_fitness, fitness_history, best_fitness_history, evaluations_done