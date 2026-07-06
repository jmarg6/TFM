import random
from typing import Callable, Tuple, List, Dict

from utils.selection_utils import create_random_mask
from utils.config import Metric, Algorithm

def generate_neighbor(mask: Dict[int, int], num_changes: int = 5, adjust_size: bool = False) -> Dict[int, int]:
    """
    Generates a neighbor by modifying the current selection mask using vectorized sampling.

    Args:
        mask: Dictionary with image indices as keys and 0/1 as values.
        num_changes: Number of elements to change/swap to generate the neighbor.
        adjust_size: If True, randomly adds or removes instances (changing the total subset size).
                     If False, swaps instances (strictly maintaining the subset size).

    Returns:
        A new neighbor mask dictionary.
    """
    neighbor = mask.copy()

    selected = [idx for idx, val in neighbor.items() if val == 1]
    unselected = [idx for idx, val in neighbor.items() if val == 0]

    # Ensure we don't try to change more items than available
    modifications = min(num_changes, len(selected), len(unselected))
    if modifications == 0:
        return neighbor

    if adjust_size:
        # Flexible size: sample how many to add and remove in total, avoiding sequential .remove()
        num_to_add = 0
        num_to_remove = 0
        for _ in range(modifications):
            if random.random() < 0.5:
                num_to_add += 1
            else:
                num_to_remove += 1

        # Clip bounds safely to avoid sampling errors
        num_to_add = min(num_to_add, len(unselected))
        num_to_remove = min(num_to_remove, len(selected))

        # Vectorized writes: ultra fast O(1) assignments
        if num_to_add > 0:
            for idx in random.sample(unselected, num_to_add):
                neighbor[idx] = 1
        if num_to_remove > 0:
            for idx in random.sample(selected, num_to_remove):
                neighbor[idx] = 0
    else:
        # Strict size swap: Sample all indices at once, completely bypassing the O(N) list-removal loop
        to_remove = random.sample(selected, modifications)
        to_add = random.sample(unselected, modifications)
        
        for r, a in zip(to_remove, to_add):
            neighbor[r] = 0
            neighbor[a] = 1

    return neighbor


def local_search(
    fitness_func: Callable[[dict], dict],
    total_instances: int,
    keep_percentage: float = 0.1,
    max_evaluations: int = 100,
    patience: int = 20,
    neighbor_size: int = 10,
    target_metric: Metric = Metric.ACCURACY,
    adjust_size: bool = False
) -> Tuple[Dict[int, int], float, List[dict], List[float], int]:
    """
    Implements a Hill Climbing / Local Search algorithm for instance selection.

    Args:
        fitness_func: Function that receives a mask and returns the metrics.
        total_instances: Total number of images in the original dataset.
        keep_percentage: Initial percentage of images to select.
        max_evaluations: Maximum number of overall evaluations allowed.
        patience: Stopping criterion if no improvements are found.
        neighbor_size: Number of modifications made to generate each neighbor.
        target_metric: Metric to optimize.
        adjust_size: Flag to determine if the subset size can fluctuate.

    Returns:
        Tuple containing: (Best Mask, Best Fitness, Metrics History, Best Fitness History, Total Evaluations)
    """
    # Dynamically select algorithm tag from central Enums configuration
    alg_id = Algorithm.FREE_LOCAL_SEARCH.value if adjust_size else Algorithm.LOCAL_SEARCH.value
    print(f"Starting {alg_id} (Target: {target_metric.value.upper()} | Initial Retention: {keep_percentage*100}%)")

    # 1. Generate and evaluate the initial solution
    current_solution = create_random_mask(total_instances, keep_percentage)
    current_fitness_dict = fitness_func(current_solution)
    current_fitness = current_fitness_dict.get(target_metric.value, 0.0)
    evaluations_done = 1

    best_fitness = current_fitness
    best_solution = current_solution.copy()
    
    # Inject initial metadata safely into the history log
    initial_pct = keep_percentage
    current_pct = sum(current_solution.values()) / total_instances
    
    initial_hist = current_fitness_dict.copy()
    initial_hist.update({
        "Algorithm": alg_id,
        "Initial Percentage": initial_pct,
        "Final Percentage": current_pct,
        "Iteration": evaluations_done
    })
    
    fitness_history = [initial_hist]
    best_fitness_history = [best_fitness]

    evaluations_without_improvement = 0

    # 2. Iterative Neighborhood Search
    while evaluations_done < max_evaluations:
        print(f"--- [{alg_id}] Evaluation {evaluations_done + 1}/{max_evaluations} ---")
        
        # Generate and evaluate neighbor
        neighbor = generate_neighbor(current_solution, neighbor_size, adjust_size)
        neighbor_fitness_dict = fitness_func(neighbor)
        neighbor_fitness = neighbor_fitness_dict.get(target_metric.value, 0.0)
        
        evaluations_done += 1

        # Acceptance criterion: Accept if it is equal or better (navigates plateaus)
        if neighbor_fitness >= current_fitness:
            current_solution = neighbor.copy()
            current_fitness = neighbor_fitness
            current_fitness_dict = neighbor_fitness_dict

            # Check for strict global improvement
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = current_solution.copy()
                evaluations_without_improvement = 0
                print(f"New best solution found. {target_metric.value.capitalize()}: {best_fitness:.4f}")
            else:
                evaluations_without_improvement += 1
                print(f"Accepted plateau neighbor. Patience: {evaluations_without_improvement}/{patience}")
        else:
            evaluations_without_improvement += 1
            print(f"Rejected worse neighbor. Patience: {evaluations_without_improvement}/{patience}")

        # Recalculate dynamic database retention allocation mapping
        current_pct = sum(current_solution.values()) / total_instances
        
        # Build clean history entry containing core model metrics and metadata tracking fields
        hist_entry = current_fitness_dict.copy()
        hist_entry.update({
            "Algorithm": alg_id,
            "Initial Percentage": initial_pct,
            "Final Percentage": current_pct,
            "Iteration": evaluations_done
        })
        
        fitness_history.append(hist_entry)
        best_fitness_history.append(best_fitness)

        # Stagnation check
        if evaluations_without_improvement >= patience:
            print(f"Search terminated due to stagnation after {evaluations_done} evaluations.")
            break

    print(f"[{alg_id}] Finished. Best {target_metric.value}: {best_fitness:.4f}")
    return best_solution, best_fitness, fitness_history, best_fitness_history, evaluations_done