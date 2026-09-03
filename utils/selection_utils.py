import random
from math import floor

from utils.algorithm_utils import Mask, validate_mask


# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================

def _round_half_up(value: float) -> int:
    """
    Round a non-negative floating-point value using half-up rounding.

    Unlike Python's built-in round(), values ending exactly in .5 are always
    rounded upward instead of using bankers' rounding.

    Args:
        value:
            Non-negative floating-point value to round.

    Returns:
        Rounded integer value.
    """
    return floor(value + 0.5)


# ==============================================================================
# SUBSET SIZE
# ==============================================================================

def calculate_subset_size(
    total_instances: int,
    keep_percentage: float,
) -> int:
    """
    Calculate the exact number of instances to retain.

    The requested retention fraction is converted into an integer subset size
    using half-up rounding. At least one instance is retained whenever the
    retention percentage is greater than zero.

    Args:
        total_instances:
            Total number of candidate instances.

        keep_percentage:
            Fraction of instances to retain, expressed in the interval
            (0, 1].

    Returns:
        Exact number of instances that must be selected.

    Raises:
        ValueError:
            If the number of instances or retention percentage is invalid.
    """
    if total_instances <= 0:
        raise ValueError(
            f"'total_instances' must be greater than 0, "
            f"but received {total_instances}."
        )

    if not 0.0 < keep_percentage <= 1.0:
        raise ValueError(
            f"'keep_percentage' must be in the interval (0, 1], "
            f"but received {keep_percentage}."
        )

    requested_size = _round_half_up(
        total_instances * keep_percentage
    )

    return max(
        1,
        min(
            requested_size,
            total_instances,
        ),
    )

def validate_search_subset_size(
    total_instances: int,
    keep_percentage: float,
) -> int:
    """
    Validate and return the subset size used by an optimization algorithm.

    The optimization algorithms require at least one selected instance and at
    least one unselected instance. The full dataset must be evaluated
    separately as the 100% baseline.

    Args:
        total_instances:
            Total number of available instances.

        keep_percentage:
            Requested fraction of instances to retain.

    Returns:
        Exact number of selected instances.

    Raises:
        ValueError:
            If the resulting subset would be empty or would contain the full
            dataset.
    """
    subset_size = calculate_subset_size(
        total_instances=total_instances,
        keep_percentage=keep_percentage,
    )

    if subset_size >= total_instances:
        raise ValueError(
            "The requested retention percentage results in selecting the "
            "entire dataset after rounding. The 100% dataset must be evaluated "
            "separately as the full-dataset baseline."
        )

    return subset_size

# ==============================================================================
# RANDOM MASK GENERATION
# ==============================================================================

def create_random_mask(
    total_instances: int,
    keep_percentage: float = 0.5,
    rng: random.Random | None = None,
) -> Mask:
    """
    Create a random fixed-size binary selection mask.

    Exactly the number of instances determined by `calculate_subset_size()` is
    selected. Every subset of that size has the same probability of being
    generated.

    Args:
        total_instances:
            Total number of candidate instances.

        keep_percentage:
            Fraction of instances to retain.

        rng:
            Optional pseudo-random number generator. Providing an explicitly
            seeded `random.Random` instance is recommended for reproducible
            experiments. If omitted, Python's global random generator is used.

    Returns:
        A dictionary mapping each instance index to:

        - 1 if the instance is selected.
        - 0 if the instance is discarded.
    """
    num_to_keep = calculate_subset_size(
        total_instances=total_instances,
        keep_percentage=keep_percentage,
    )

    generator = (
        rng
        if rng is not None
        else random
    )

    selected_indices = set(
        generator.sample(
            range(total_instances),
            k=num_to_keep,
        )
    )

    return {
        index: int(index in selected_indices)
        for index in range(total_instances)
    }


# ==============================================================================
# FIXED-SIZE MUTATION
# ==============================================================================

def mutate_mask(
    mask: Mask,
    mutation_probability: float = 0.1,
    mutation_fraction: float = 0.05,
    rng: random.Random | None = None,
) -> Mask:
    """
    Apply a fixed-size swap mutation to a binary selection mask.

    The operator preserves the exact number of selected instances.

    When mutation is triggered, a fraction of the currently selected instances
    is replaced by the same number of previously unselected instances.

    Args:
        mask:
            Original binary selection mask.

        mutation_probability:
            Probability of applying the mutation operator to the individual.

        mutation_fraction:
            Fraction of the currently selected subset to replace when mutation
            is applied.

        rng:
            Optional pseudo-random number generator.

    Returns:
        A new mutated mask. The original mask is never modified.

    Raises:
        ValueError:
            If the mask or mutation parameters are invalid.
    """
    validate_mask(mask)

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

    generator = (
        rng
        if rng is not None
        else random
    )

    mutated_mask = mask.copy()

    # Decide whether mutation is applied to this individual.
    if generator.random() >= mutation_probability:
        return mutated_mask

    selected_indices = [
        index
        for index, selected in mutated_mask.items()
        if selected == 1
    ]

    unselected_indices = [
        index
        for index, selected in mutated_mask.items()
        if selected == 0
    ]

    if not selected_indices or not unselected_indices:
        return mutated_mask

    requested_swaps = _round_half_up(
        len(selected_indices) * mutation_fraction
    )

    num_swaps = max(
        1,
        min(
            requested_swaps,
            len(selected_indices),
            len(unselected_indices),
        ),
    )

    indices_to_remove = generator.sample(
        selected_indices,
        k=num_swaps,
    )

    indices_to_add = generator.sample(
        unselected_indices,
        k=num_swaps,
    )

    for index in indices_to_remove:
        mutated_mask[index] = 0

    for index in indices_to_add:
        mutated_mask[index] = 1

    return mutated_mask