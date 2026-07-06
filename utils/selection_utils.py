import random
from typing import Dict

def create_random_mask(total_instances: int, keep_percentage: float = 0.5) -> Dict[int, int]:
    """
    Creates an initial random selection mask to filter dataset instances.
    
    Args:
        total_instances: The total number of images/samples in the dataset.
        keep_percentage: The fraction of instances to retain (e.g., 0.25 for 25%).
        
    Returns:
        A dictionary mapping each instance index to 1 (keep) or 0 (discard).
    """
    # Optimized creation using population sampling directly
    num_to_keep = int(total_instances * keep_percentage)
    selected_indices = set(random.sample(range(total_instances), num_to_keep))
    
    return {i: (1 if i in selected_indices else 0) for i in range(total_instances)}


def mutate_mask(mask: Dict[int, int], mutation_rate: float = 0.1, is_constrained: bool = True) -> Dict[int, int]:
    """
    Applies a mutation operator to a selection mask.
    Supports both constrained size-preserving swaps and unconstrained free bit-flips.
    
    Args:
        mask: The original selection mask dictionary mapping indices to 0 or 1.
        mutation_rate: The probability of applying mutation to an element.
        is_constrained: If True, forces the mask to keep the exact same number of 1s (swapping).
                        If False, allows independent bit flipping (size can grow or shrink).
                       
    Returns:
        A new mutated dictionary mask.
    """
    mutated = mask.copy()

    # --- MODE 1: FREE / UNCONSTRAINED MUTATION (Independent Bit Flipping) ---
    if not is_constrained:
        # Flip each bit independently based on mutation rate
        for idx in mutated.keys():
            if random.random() < mutation_rate:
                mutated[idx] = 1 - mutated[idx]  # Flips 1 to 0, and 0 to 1
        return mutated

    # --- MODE 2: CONSTRAINED MUTATION (Size-Preserving Swap) ---
    # Trigger mutation globally based on mutation rate probability
    if random.random() > mutation_rate:
        return mutated

    selected = [idx for idx, val in mutated.items() if val == 1]
    discarded = [idx for idx, val in mutated.items() if val == 0]

    if not selected or not discarded:
        return mutated

    # Determine how many instances to swap (up to a safe maximum)
    num_swaps = max(1, int(len(selected) * mutation_rate * 0.5))
    num_swaps = min(num_swaps, len(selected), len(discarded))

    # VECTOR OPTIMIZATION: Sample all positions at once instead of looping with .remove()
    swaps_out = random.sample(selected, num_swaps)
    swaps_in = random.sample(discarded, num_swaps)

    # Apply all changes in O(1) dictionary writes
    for to_remove, to_add in zip(swaps_out, swaps_in):
        mutated[to_remove] = 0
        mutated[to_add] = 1

    return mutated