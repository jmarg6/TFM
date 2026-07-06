from typing import Callable, Tuple, List
from utils.selection_utils import create_random_mask
from utils.config import Metric, Algorithm

def random_search(
    fitness_func: Callable[[dict], dict],
    total_instances: int,
    keep_percentage: float = 0.5,
    max_evaluations: int = 100,
    patience: int = 20,
    target_metric: Metric = Metric.ACCURACY
) -> Tuple[dict, float, List[dict], List[float], int]:
    """
    Implements a Random Search algorithm for Instance Selection.
    
    Generates independent random subsets and keeps the best one.
    Stops if it reaches 'max_evaluations' or if 'patience' iterations pass without improvement.

    Args:
        fitness_func: Function that receives a mask (dict) and returns the metrics (dict).
        total_instances: Total number of images in the original dataset.
        keep_percentage: Percentage of images to keep (e.g., 0.10 for 10%).
        max_evaluations: Maximum number of evaluations allowed.
        patience: Early stopping criterion (stagnation limit).
        target_metric: Metric to optimize (e.g., ACCURACY or F1).

    Returns:
        Tuple containing: (Best Mask, Best Fitness, Metrics History, Best Fitness History, Total Evaluations)
    """
    evaluations_without_improvement = 0
    evaluations_done = 0
    
    best_fitness = -1.0
    best_solution = {}
    
    fitness_history = []
    best_fitness_history = []
    
    # Text identifier for consistency across logs and charts
    alg_id = Algorithm.RANDOM_SEARCH.value
    print(f"Starting Random Search (Target: {target_metric.value.upper()} | Retention: {keep_percentage*100}%)")

    # Main evaluation loop
    while evaluations_done < max_evaluations:
        print(f"--- [{alg_id}] Evaluation {evaluations_done + 1}/{max_evaluations} ---")
        
        # Generate a completely new random subset (mask)
        current_solution = create_random_mask(total_instances, keep_percentage)
        
        # Evaluate the generated mask using the provided fitness function
        metrics_result = fitness_func(current_solution)
        
        # Defensive lookups avoiding key errors
        current_fitness = metrics_result.get(target_metric.value, 0.0)
        
        # INYECCIÓN DE METADATOS: Enriquecemos el diccionario para alimentar a plot_utils de forma nativa
        metrics_result["Algorithm"] = alg_id
        metrics_result["Initial Percentage"] = keep_percentage
        metrics_result["Final Percentage"] = keep_percentage  # En RS clásico se mantiene idéntico
        metrics_result["Iteration"] = evaluations_done + 1
        
        # Store the enriched metrics dictionary for this iteration
        fitness_history.append(metrics_result)
        evaluations_done += 1

        # Check if the current solution improves the best known fitness
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = current_solution.copy()  # Clonamos defensivamente para evitar mutaciones accidentales
            evaluations_without_improvement = 0
            print(f"New best solution found. {target_metric.value.capitalize()}: {best_fitness:.4f}")
        else:
            evaluations_without_improvement += 1
            print(f"No improvement. Patience: {evaluations_without_improvement}/{patience}")

        # Track the running best fitness score over time
        best_fitness_history.append(best_fitness)

        # Early stopping check based on patience
        if evaluations_without_improvement >= patience:
            print(f"Search terminated due to stagnation after {evaluations_done} evaluations.")
            break

    print(f"[{alg_id}] Finished. Best {target_metric.value}: {best_fitness:.4f}")
    
    return best_solution, best_fitness, fitness_history, best_fitness_history, evaluations_done