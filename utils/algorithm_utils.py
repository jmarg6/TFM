from math import isfinite
from typing import Any, Mapping, TypeAlias


# ==============================================================================
# TYPE ALIASES
# ==============================================================================

Mask: TypeAlias = dict[int, int]
Metrics: TypeAlias = Mapping[str, Any]
HistoryEntry: TypeAlias = dict[str, Any]


# ==============================================================================
# MASK VALIDATION
# ==============================================================================

def validate_mask(mask: Mask) -> None:
    """
    Validate that a selection mask is non-empty and binary.

    Args:
        mask:
            Selection mask to validate.

    Raises:
        ValueError:
            If the mask is empty or contains values different from 0 and 1.
    """
    if not mask:
        raise ValueError(
            "'mask' must contain at least one instance."
        )

    invalid_values = {
        value
        for value in mask.values()
        if value not in (0, 1)
    }

    if invalid_values:
        raise ValueError(
            "'mask' must contain only binary values (0 or 1). "
            f"Invalid values found: {sorted(invalid_values)}."
        )


def validate_same_index_space(
    mask1: Mask,
    mask2: Mask,
) -> None:
    """
    Validate that two masks contain exactly the same instance indices.

    Args:
        mask1:
            First selection mask.

        mask2:
            Second selection mask.

    Raises:
        ValueError:
            If the masks do not contain exactly the same keys.
    """
    if mask1.keys() != mask2.keys():
        raise ValueError(
            "Both masks must contain exactly the same instance indices."
        )


# ==============================================================================
# FITNESS VALIDATION
# ==============================================================================

def extract_fitness(
    metrics: Mapping[str, Any],
    metric_name: str,
) -> float:
    """
    Extract and validate a target fitness value from an evaluation result.

    Args:
        metrics:
            Mapping containing the metrics returned by the fitness function.

        metric_name:
            Name of the metric used as the optimization objective.

    Returns:
        Validated finite fitness value.

    Raises:
        KeyError:
            If the target metric is not present.

        TypeError:
            If the target metric cannot be converted to a floating-point
            value.

        ValueError:
            If the target metric is not finite.
    """
    if metric_name not in metrics:
        raise KeyError(
            f"Target metric '{metric_name}' was not returned by "
            f"'fitness_func'. Available metrics: {sorted(metrics.keys())}"
        )

    try:
        fitness = float(metrics[metric_name])
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Metric '{metric_name}' must be numeric, but received "
            f"{metrics[metric_name]!r}."
        ) from exc

    if not isfinite(fitness):
        raise ValueError(
            f"Metric '{metric_name}' returned a non-finite value: "
            f"{fitness}."
        )

    return fitness


# ==============================================================================
# DISTANCE METRICS
# ==============================================================================

def hamming_distance(
    mask1: Mask,
    mask2: Mask,
) -> int:
    """
    Calculate the Hamming distance between two binary selection masks.

    The Hamming distance corresponds to the number of positions whose binary
    values differ.

    Args:
        mask1:
            First binary selection mask.

        mask2:
            Second binary selection mask.

    Returns:
        Number of positions with different values.

    Raises:
        ValueError:
            If the masks do not contain exactly the same instance indices.
    """
    validate_same_index_space(
        mask1=mask1,
        mask2=mask2,
    )

    return sum(
        mask1[index] != mask2[index]
        for index in mask1
    )