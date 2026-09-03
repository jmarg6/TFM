"""Single stratified random sampling for fixed-cardinality instance selection."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from math import floor
from typing import Sequence

from utils.algorithm_utils import Mask
from utils.selection_utils import calculate_subset_size


def _allocate_stratified_counts(
    *,
    class_sizes: dict[int, int],
    subset_size: int,
) -> dict[int, int]:
    """
    Allocate an exact subset cardinality proportionally across classes.

    Hamilton's largest-remainder method is used. Ties are resolved by the
    integer class label, making the allocation deterministic.
    """
    if not class_sizes:
        raise ValueError("'class_sizes' must contain at least one class.")

    if any(size <= 0 for size in class_sizes.values()):
        raise ValueError("Every class must contain at least one instance.")

    total_instances = sum(class_sizes.values())

    if not 0 < subset_size <= total_instances:
        raise ValueError(
            "'subset_size' must be in the interval "
            f"[1, {total_instances}], but received {subset_size}."
        )

    ideal_counts = {
        class_label: subset_size * class_size / total_instances
        for class_label, class_size in class_sizes.items()
    }

    selected_counts = {
        class_label: min(
            class_sizes[class_label],
            floor(ideal_count),
        )
        for class_label, ideal_count in ideal_counts.items()
    }

    remaining = subset_size - sum(selected_counts.values())

    # Largest fractional remainders receive the remaining positions. Class
    # labels provide a deterministic tie-breaker.
    allocation_order = sorted(
        class_sizes,
        key=lambda class_label: (
            -(
                ideal_counts[class_label]
                - floor(ideal_counts[class_label])
            ),
            class_label,
        ),
    )

    while remaining > 0:
        allocated_in_pass = False

        for class_label in allocation_order:
            if selected_counts[class_label] >= class_sizes[class_label]:
                continue

            selected_counts[class_label] += 1
            remaining -= 1
            allocated_in_pass = True

            if remaining == 0:
                break

        if not allocated_in_pass:
            raise RuntimeError(
                "Unable to allocate the requested stratified subset size."
            )

    if sum(selected_counts.values()) != subset_size:
        raise RuntimeError(
            "The stratified allocation does not match the requested subset "
            "cardinality."
        )

    return selected_counts


def create_stratified_random_mask(
    *,
    targets: Sequence[int],
    keep_percentage: float,
    rng: random.Random | None = None,
) -> tuple[Mask, dict[int, int], dict[int, int]]:
    """
    Generate one class-stratified random subset without validation search.

    Exactly one subset is sampled. The total cardinality is computed with the
    same half-up rounding rule used by the optimization algorithms. Class
    quotas are proportional to the class frequencies and sum exactly to the
    requested total cardinality.

    Args:
        targets:
            Integer class label for every instance in the base-training mask
            index space.

        keep_percentage:
            Fraction of base-training instances to retain.

        rng:
            Optional explicitly seeded pseudo-random number generator.

    Returns:
        A tuple containing the binary mask, selected counts per class, and
        available counts per class.
    """
    normalized_targets = [
        int(target)
        for target in targets
    ]

    if not normalized_targets:
        raise ValueError("'targets' must contain at least one label.")

    total_instances = len(normalized_targets)
    subset_size = calculate_subset_size(
        total_instances=total_instances,
        keep_percentage=keep_percentage,
    )

    positions_by_class: dict[int, list[int]] = defaultdict(list)

    for position, class_label in enumerate(normalized_targets):
        positions_by_class[class_label].append(position)

    class_sizes = dict(
        sorted(
            Counter(normalized_targets).items()
        )
    )

    selected_counts = _allocate_stratified_counts(
        class_sizes=class_sizes,
        subset_size=subset_size,
    )

    generator = rng if rng is not None else random
    selected_positions: set[int] = set()

    for class_label in sorted(positions_by_class):
        class_positions = positions_by_class[class_label]
        class_selected_count = selected_counts[class_label]

        selected_positions.update(
            generator.sample(
                class_positions,
                k=class_selected_count,
            )
        )

    if len(selected_positions) != subset_size:
        raise RuntimeError(
            "The sampled stratified subset does not match the requested "
            "cardinality."
        )

    mask = {
        position: int(position in selected_positions)
        for position in range(total_instances)
    }

    return (
        mask,
        selected_counts,
        class_sizes,
    )
