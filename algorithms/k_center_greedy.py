"""Global k-center greedy selection on frozen feature vectors.

This module implements the standard farthest-first traversal used as the
k-center greedy coreset baseline. Selection is global: class labels are never
used to construct the subset. Labels may be supplied only to report the class
distribution of the selected prefix after selection has finished.

The exact pairwise squared-Euclidean distance matrix is cached as a float32
memory-mapped file. This makes high-retention experiments feasible because the
farthest-first traversal can update nearest-center distances from cached rows
instead of recomputing distances from 4096-dimensional features at every step.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import random
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from utils.algorithm_utils import Mask
from utils.selection_utils import calculate_subset_size


logger = logging.getLogger(__name__)

_CACHE_FORMAT_VERSION = 1
_DISTANCE_DTYPE = np.float32
_DISTANCE_FILENAME = "pairwise_squared_euclidean_float32.dat"
_DISTANCE_METADATA_FILENAME = "distance_cache_metadata.json"

# The same in-memory feature tensor is reused for every percentage and seed in
# one experiment process. Caching its fingerprint avoids hashing up to 1.3 GB
# of feature data repeatedly.
_FEATURE_FINGERPRINT_CACHE: dict[
    tuple[int, tuple[int, ...], str],
    str,
] = {}


def _validate_feature_matrix(
    features: torch.Tensor,
) -> torch.Tensor:
    """Validate and return a contiguous CPU float32 feature matrix."""
    if not isinstance(features, torch.Tensor):
        raise TypeError(
            "'features' must be a torch.Tensor."
        )

    if features.ndim != 2:
        raise ValueError(
            "'features' must have shape [num_instances, feature_dimension], "
            f"but received {tuple(features.shape)}."
        )

    if int(features.shape[0]) <= 1:
        raise ValueError(
            "At least two feature vectors are required by k-center greedy."
        )

    if int(features.shape[1]) <= 0:
        raise ValueError(
            "The feature dimension must be greater than 0."
        )

    if not torch.isfinite(features).all():
        raise ValueError(
            "The feature matrix contains NaN or infinite values."
        )

    return (
        features
        .detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
        .contiguous()
    )


def _validate_targets(
    targets: Sequence[int] | None,
    expected_size: int,
) -> list[int] | None:
    """Validate optional labels used only for post-selection reporting."""
    if targets is None:
        return None

    normalized_targets = [
        int(target)
        for target in targets
    ]

    if len(normalized_targets) != expected_size:
        raise ValueError(
            "Target and feature counts do not match. "
            f"Targets: {len(normalized_targets)}, "
            f"features: {expected_size}."
        )

    return normalized_targets


def _calculate_feature_fingerprint(
    features: torch.Tensor,
    chunk_rows: int = 1024,
) -> str:
    """Calculate a stable SHA-256 fingerprint without duplicating the matrix."""
    cache_key = (
        int(features.data_ptr()),
        tuple(int(value) for value in features.shape),
        str(features.dtype),
    )

    cached = _FEATURE_FINGERPRINT_CACHE.get(
        cache_key
    )

    if cached is not None:
        return cached

    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(
            {
                "shape": list(features.shape),
                "dtype": str(features.dtype),
            },
            sort_keys=True,
        ).encode("utf-8")
    )

    for start in range(
        0,
        int(features.shape[0]),
        chunk_rows,
    ):
        end = min(
            start + chunk_rows,
            int(features.shape[0]),
        )

        chunk = (
            features[start:end]
            .contiguous()
            .numpy()
        )

        hasher.update(
            chunk.tobytes(order="C")
        )

    fingerprint = hasher.hexdigest()
    _FEATURE_FINGERPRINT_CACHE[
        cache_key
    ] = fingerprint

    return fingerprint


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write JSON through a temporary file and atomically replace the target."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f"{path.name}.tmp.{os.getpid()}"
    )

    with temporary_path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            payload,
            output_file,
            indent=2,
            sort_keys=True,
        )

    temporary_path.replace(
        path
    )


def _read_json(
    path: Path,
) -> dict[str, Any]:
    """Read and validate a JSON object."""
    with path.open(
        mode="r",
        encoding="utf-8",
    ) as input_file:
        payload = json.load(
            input_file
        )

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a JSON object in '{path}'."
        )

    return payload


@contextlib.contextmanager
def _exclusive_file_lock(
    lock_path: Path,
) -> Iterator[None]:
    """Acquire an exclusive Linux file lock for one cache-building operation."""
    lock_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with lock_path.open(
        mode="a+",
        encoding="utf-8",
    ) as lock_file:
        fcntl.flock(
            lock_file.fileno(),
            fcntl.LOCK_EX,
        )

        try:
            yield
        finally:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_UN,
            )


def _distance_cache_is_valid(
    *,
    distance_path: Path,
    metadata_path: Path,
    num_instances: int,
    feature_dimension: int,
    feature_fingerprint: str,
) -> bool:
    """Return whether an existing distance cache matches the feature matrix."""
    if not distance_path.is_file() or not metadata_path.is_file():
        return False

    try:
        metadata = _read_json(
            metadata_path
        )
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False

    expected_size_bytes = (
        num_instances
        * num_instances
        * np.dtype(_DISTANCE_DTYPE).itemsize
    )

    return (
        metadata.get("cache_format_version")
        == _CACHE_FORMAT_VERSION
        and int(metadata.get("num_instances", -1))
        == num_instances
        and int(metadata.get("feature_dimension", -1))
        == feature_dimension
        and metadata.get("feature_fingerprint")
        == feature_fingerprint
        and metadata.get("dtype")
        == str(np.dtype(_DISTANCE_DTYPE))
        and metadata.get("distance_metric")
        == "squared_euclidean"
        and metadata.get("feature_normalization")
        == "none"
        and distance_path.stat().st_size
        == expected_size_bytes
    )


def _build_distance_cache(
    *,
    features: torch.Tensor,
    cache_dir: Path,
    feature_fingerprint: str,
    device: torch.device,
    block_size: int,
) -> dict[str, Any]:
    """Build the complete pairwise squared-Euclidean distance matrix."""
    if block_size <= 0:
        raise ValueError(
            f"'block_size' must be greater than 0, but received {block_size}."
        )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for k-center distance construction but is "
            "not available."
        )

    num_instances = int(
        features.shape[0]
    )
    feature_dimension = int(
        features.shape[1]
    )

    expected_size_bytes = (
        num_instances
        * num_instances
        * np.dtype(_DISTANCE_DTYPE).itemsize
    )

    available_bytes = shutil.disk_usage(
        cache_dir
    ).free

    # Keep a modest safety margin for metadata, temporary files, and concurrent
    # filesystem activity.
    if available_bytes < int(expected_size_bytes * 1.10):
        raise OSError(
            "Insufficient free disk space for the global k-center distance "
            f"cache. Required approximately "
            f"{expected_size_bytes / (1024 ** 3):.2f} GiB, available "
            f"{available_bytes / (1024 ** 3):.2f} GiB."
        )

    distance_path = (
        cache_dir
        / _DISTANCE_FILENAME
    )

    metadata_path = (
        cache_dir
        / _DISTANCE_METADATA_FILENAME
    )

    temporary_distance_path = distance_path.with_name(
        f"{distance_path.name}.tmp.{os.getpid()}"
    )

    if temporary_distance_path.exists():
        temporary_distance_path.unlink()

    logger.info(
        "Building global pairwise distance cache | instances=%d | "
        "features=%d | size=%.2f GiB | block_size=%d | device=%s",
        num_instances,
        feature_dimension,
        expected_size_bytes / (1024 ** 3),
        block_size,
        device,
    )

    start_time = time.perf_counter()

    device_features = features.to(
        device=device,
        dtype=torch.float32,
    )

    squared_norms = (
        device_features
        .square()
        .sum(dim=1)
    )

    transposed_features = (
        device_features
        .transpose(0, 1)
        .contiguous()
    )

    distance_memmap = np.memmap(
        temporary_distance_path,
        dtype=_DISTANCE_DTYPE,
        mode="w+",
        shape=(
            num_instances,
            num_instances,
        ),
    )

    num_blocks = (
        num_instances
        + block_size
        - 1
    ) // block_size

    for block_index, start in enumerate(
        range(
            0,
            num_instances,
            block_size,
        ),
        start=1,
    ):
        end = min(
            start + block_size,
            num_instances,
        )

        block = device_features[
            start:end
        ]

        dot_products = (
            block
            @ transposed_features
        )

        squared_distances = (
            squared_norms[start:end, None]
            + squared_norms[None, :]
            - 2.0 * dot_products
        )

        squared_distances.clamp_min_(
            0.0
        )

        local_rows = torch.arange(
            end - start,
            device=device,
        )
        global_columns = torch.arange(
            start,
            end,
            device=device,
        )

        squared_distances[
            local_rows,
            global_columns,
        ] = 0.0

        distance_memmap[
            start:end,
            :,
        ] = (
            squared_distances
            .detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .numpy()
        )

        if (
            block_index == 1
            or block_index % 10 == 0
            or block_index == num_blocks
        ):
            logger.info(
                "Distance cache progress | block=%d/%d | rows=%d/%d",
                block_index,
                num_blocks,
                end,
                num_instances,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    distance_memmap.flush()
    del distance_memmap

    del transposed_features
    del squared_norms
    del device_features

    if device.type == "cuda":
        torch.cuda.empty_cache()

    build_duration = (
        time.perf_counter()
        - start_time
    )

    temporary_distance_path.replace(
        distance_path
    )

    metadata = {
        "cache_format_version": _CACHE_FORMAT_VERSION,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "num_instances": num_instances,
        "feature_dimension": feature_dimension,
        "feature_fingerprint": feature_fingerprint,
        "dtype": str(
            np.dtype(_DISTANCE_DTYPE)
        ),
        "distance_metric": "squared_euclidean",
        "feature_normalization": "none",
        "distance_matrix_shape": [
            num_instances,
            num_instances,
        ],
        "distance_file": str(
            distance_path
        ),
        "distance_file_size_bytes": expected_size_bytes,
        "distance_file_size_gib": (
            expected_size_bytes
            / (1024 ** 3)
        ),
        "block_size": block_size,
        "build_device": str(device),
        "build_device_name": (
            torch.cuda.get_device_name(
                device
            )
            if device.type == "cuda"
            else "cpu"
        ),
        "torch_version": torch.__version__,
        "build_duration_seconds": build_duration,
    }

    _write_json_atomic(
        metadata_path,
        metadata,
    )

    logger.info(
        "Global pairwise distance cache completed | seconds=%.3f | path=%s",
        build_duration,
        distance_path,
    )

    return metadata


def ensure_global_distance_cache(
    *,
    features: torch.Tensor,
    cache_dir: str | Path,
    device: torch.device | str,
    block_size: int = 512,
) -> tuple[Path, dict[str, Any], bool]:
    """Create or reuse the global pairwise distance cache."""
    cpu_features = _validate_feature_matrix(
        features
    )

    resolved_device = torch.device(
        device
    )

    resolved_cache_dir = Path(
        cache_dir
    )
    resolved_cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    num_instances = int(
        cpu_features.shape[0]
    )
    feature_dimension = int(
        cpu_features.shape[1]
    )

    feature_fingerprint = _calculate_feature_fingerprint(
        cpu_features
    )

    distance_path = (
        resolved_cache_dir
        / _DISTANCE_FILENAME
    )

    metadata_path = (
        resolved_cache_dir
        / _DISTANCE_METADATA_FILENAME
    )

    lock_path = (
        resolved_cache_dir
        / "distance_cache.lock"
    )

    with _exclusive_file_lock(
        lock_path
    ):
        if _distance_cache_is_valid(
            distance_path=distance_path,
            metadata_path=metadata_path,
            num_instances=num_instances,
            feature_dimension=feature_dimension,
            feature_fingerprint=feature_fingerprint,
        ):
            metadata = _read_json(
                metadata_path
            )
            return (
                distance_path,
                metadata,
                True,
            )

        # Remove incompatible or incomplete artifacts before rebuilding.
        if distance_path.exists():
            distance_path.unlink()

        if metadata_path.exists():
            metadata_path.unlink()

        metadata = _build_distance_cache(
            features=cpu_features,
            cache_dir=resolved_cache_dir,
            feature_fingerprint=feature_fingerprint,
            device=resolved_device,
            block_size=block_size,
        )

        return (
            distance_path,
            metadata,
            False,
        )


def _order_cache_paths(
    cache_dir: Path,
    search_seed: int,
) -> tuple[Path, Path, Path, Path]:
    """Return order, radius, metadata, and lock paths for one seed."""
    order_path = (
        cache_dir
        / f"farthest_first_order_seed_{search_seed}.npy"
    )

    radius_path = (
        cache_dir
        / f"covering_radius_squared_seed_{search_seed}.npy"
    )

    metadata_path = (
        cache_dir
        / f"farthest_first_order_seed_{search_seed}.json"
    )

    lock_path = (
        cache_dir
        / f"farthest_first_order_seed_{search_seed}.lock"
    )

    return (
        order_path,
        radius_path,
        metadata_path,
        lock_path,
    )


def _order_cache_is_valid(
    *,
    order_path: Path,
    radius_path: Path,
    metadata_path: Path,
    num_instances: int,
    search_seed: int,
    feature_fingerprint: str,
    required_centers: int,
) -> bool:
    """Return whether a cached traversal contains the required prefix."""
    if (
        not order_path.is_file()
        or not radius_path.is_file()
        or not metadata_path.is_file()
    ):
        return False

    try:
        metadata = _read_json(
            metadata_path
        )

        order = np.load(
            order_path,
            mmap_mode="r",
        )

        radii = np.load(
            radius_path,
            mmap_mode="r",
        )
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False

    return (
        metadata.get("cache_format_version")
        == _CACHE_FORMAT_VERSION
        and int(metadata.get("num_instances", -1))
        == num_instances
        and int(metadata.get("search_seed", -1))
        == search_seed
        and metadata.get("feature_fingerprint")
        == feature_fingerprint
        and int(metadata.get("selected_centers", -1))
        >= required_centers
        and order.ndim == 1
        and radii.ndim == 1
        and len(order) >= required_centers
        and len(radii) >= required_centers
    )


def _save_numpy_array_atomic(
    path: Path,
    array: np.ndarray,
) -> None:
    """Save a NumPy array atomically without automatic suffix changes."""
    temporary_path = path.with_name(
        f"{path.name}.tmp.{os.getpid()}"
    )

    with temporary_path.open(
        mode="wb",
    ) as output_file:
        np.save(
            output_file,
            array,
            allow_pickle=False,
        )

    temporary_path.replace(
        path
    )


def _build_farthest_first_order(
    *,
    distance_path: Path,
    distance_metadata: dict[str, Any],
    cache_dir: Path,
    search_seed: int,
    num_centers: int,
) -> dict[str, Any]:
    """Build one seeded global farthest-first traversal."""
    num_instances = int(
        distance_metadata["num_instances"]
    )

    if not 0 < num_centers < num_instances:
        raise ValueError(
            "'num_centers' must be in the interval "
            f"[1, {num_instances - 1}], but received {num_centers}."
        )

    distance_matrix = np.memmap(
        distance_path,
        dtype=_DISTANCE_DTYPE,
        mode="r",
        shape=(
            num_instances,
            num_instances,
        ),
    )

    generator = random.Random(
        search_seed
    )

    first_center = generator.randrange(
        num_instances
    )

    selected_order = np.empty(
        num_centers,
        dtype=np.int32,
    )

    covering_radius_squared = np.empty(
        num_centers,
        dtype=np.float32,
    )

    selected_order[0] = first_center

    minimum_squared_distances = np.array(
        distance_matrix[first_center],
        dtype=np.float32,
        copy=True,
    )

    # A negative marker excludes selected positions permanently because valid
    # squared distances are non-negative and np.minimum preserves the marker.
    minimum_squared_distances[
        first_center
    ] = -1.0

    covering_radius_squared[0] = float(
        minimum_squared_distances.max()
    )

    start_time = time.perf_counter()
    progress_interval = max(
        1,
        num_centers // 20,
    )

    logger.info(
        "Building global farthest-first order | seed=%d | centers=%d | "
        "instances=%d | first_center=%d",
        search_seed,
        num_centers,
        num_instances,
        first_center,
    )

    for order_position in range(
        1,
        num_centers,
    ):
        next_center = int(
            np.argmax(
                minimum_squared_distances
            )
        )

        if minimum_squared_distances[next_center] < 0.0:
            raise RuntimeError(
                "k-center greedy attempted to select an already selected "
                "instance."
            )

        selected_order[
            order_position
        ] = next_center

        np.minimum(
            minimum_squared_distances,
            distance_matrix[next_center],
            out=minimum_squared_distances,
        )

        minimum_squared_distances[
            next_center
        ] = -1.0

        covering_radius_squared[
            order_position
        ] = float(
            minimum_squared_distances.max()
        )

        completed_centers = (
            order_position + 1
        )

        if (
            completed_centers == 2
            or completed_centers % progress_interval == 0
            or completed_centers == num_centers
        ):
            logger.info(
                "Farthest-first progress | seed=%d | centers=%d/%d | "
                "covering_radius=%.6f",
                search_seed,
                completed_centers,
                num_centers,
                sqrt(
                    max(
                        0.0,
                        float(
                            covering_radius_squared[
                                order_position
                            ]
                        ),
                    )
                ),
            )

    build_duration = (
        time.perf_counter()
        - start_time
    )

    order_path, radius_path, metadata_path, _ = _order_cache_paths(
        cache_dir=cache_dir,
        search_seed=search_seed,
    )

    _save_numpy_array_atomic(
        order_path,
        selected_order,
    )

    _save_numpy_array_atomic(
        radius_path,
        covering_radius_squared,
    )

    metadata = {
        "cache_format_version": _CACHE_FORMAT_VERSION,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "num_instances": num_instances,
        "feature_fingerprint": distance_metadata[
            "feature_fingerprint"
        ],
        "search_seed": search_seed,
        "selected_centers": num_centers,
        "first_center_position": first_center,
        "selection_scope": "global",
        "uses_class_labels_for_selection": False,
        "distance_metric": "squared_euclidean",
        "feature_normalization": "none",
        "initial_center_strategy": "seeded_random_global_instance",
        "traversal": "farthest_first",
        "order_file": str(order_path),
        "covering_radius_file": str(radius_path),
        "build_duration_seconds": build_duration,
    }

    _write_json_atomic(
        metadata_path,
        metadata,
    )

    del minimum_squared_distances
    del distance_matrix

    logger.info(
        "Global farthest-first order completed | seed=%d | centers=%d | "
        "seconds=%.3f",
        search_seed,
        num_centers,
        build_duration,
    )

    return metadata


def ensure_global_farthest_first_order(
    *,
    distance_path: Path,
    distance_metadata: dict[str, Any],
    cache_dir: str | Path,
    search_seed: int,
    num_centers: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], bool]:
    """Create or reuse a seeded global farthest-first traversal prefix."""
    if search_seed < 0:
        raise ValueError(
            f"'search_seed' must be non-negative, but received {search_seed}."
        )

    resolved_cache_dir = Path(
        cache_dir
    )
    resolved_cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    num_instances = int(
        distance_metadata["num_instances"]
    )

    feature_fingerprint = str(
        distance_metadata[
            "feature_fingerprint"
        ]
    )

    (
        order_path,
        radius_path,
        metadata_path,
        lock_path,
    ) = _order_cache_paths(
        cache_dir=resolved_cache_dir,
        search_seed=search_seed,
    )

    with _exclusive_file_lock(
        lock_path
    ):
        reused = _order_cache_is_valid(
            order_path=order_path,
            radius_path=radius_path,
            metadata_path=metadata_path,
            num_instances=num_instances,
            search_seed=search_seed,
            feature_fingerprint=feature_fingerprint,
            required_centers=num_centers,
        )

        if not reused:
            for path in (
                order_path,
                radius_path,
                metadata_path,
            ):
                if path.exists():
                    path.unlink()

            _build_farthest_first_order(
                distance_path=distance_path,
                distance_metadata=distance_metadata,
                cache_dir=resolved_cache_dir,
                search_seed=search_seed,
                num_centers=num_centers,
            )

        metadata = _read_json(
            metadata_path
        )

        order = np.load(
            order_path,
            mmap_mode="r",
        )

        radii = np.load(
            radius_path,
            mmap_mode="r",
        )

        return (
            order,
            radii,
            metadata,
            reused,
        )


def create_global_k_center_mask(
    *,
    features: torch.Tensor,
    targets: Sequence[int] | None,
    keep_percentage: float,
    max_keep_percentage: float,
    search_seed: int,
    device: torch.device | str,
    cache_dir: str | Path,
    distance_block_size: int = 512,
) -> tuple[Mask, dict[str, Any]]:
    """
    Create a fixed-size global k-center greedy coreset.

    Selection is performed over all base-training feature vectors. Class labels
    are not used by the traversal. When labels are supplied, they are consulted
    only after selection to report the selected class distribution.

    A traversal is built up to ``max_keep_percentage`` and cached. Therefore,
    all requested retention levels for the same dataset and seed should be
    executed in one process and obtained as nested prefixes of the same order.

    Args:
        features:
            Frozen feature matrix aligned with the base-training mask space.

        targets:
            Optional class labels aligned with the feature rows. These labels
            are used only for post-selection diagnostics.

        keep_percentage:
            Retention fraction represented by the returned mask.

        max_keep_percentage:
            Largest retention fraction requested in the current experiment.
            The cached traversal is constructed once up to this cardinality.

        search_seed:
            Seed controlling the single initial global center.

        device:
            Device used to construct the pairwise distance cache.

        cache_dir:
            Persistent directory used for the distance matrix and traversal
            orders.

        distance_block_size:
            Number of matrix rows generated per GPU block.

    Returns:
        Selection mask and geometric/cache diagnostics.
    """
    cpu_features = _validate_feature_matrix(
        features
    )

    num_instances = int(
        cpu_features.shape[0]
    )

    normalized_targets = _validate_targets(
        targets=targets,
        expected_size=num_instances,
    )

    if not 0.0 < keep_percentage < 1.0:
        raise ValueError(
            "'keep_percentage' must be in the interval (0, 1)."
        )

    if not 0.0 < max_keep_percentage < 1.0:
        raise ValueError(
            "'max_keep_percentage' must be in the interval (0, 1)."
        )

    if max_keep_percentage < keep_percentage:
        raise ValueError(
            "'max_keep_percentage' must be greater than or equal to "
            "'keep_percentage'."
        )

    selected_instances = calculate_subset_size(
        total_instances=num_instances,
        keep_percentage=keep_percentage,
    )

    max_selected_instances = calculate_subset_size(
        total_instances=num_instances,
        keep_percentage=max_keep_percentage,
    )

    if max_selected_instances >= num_instances:
        raise ValueError(
            "The maximum k-center retention selects the complete dataset. "
            "The 100% baseline must remain separate."
        )

    selection_start = time.perf_counter()

    (
        distance_path,
        distance_metadata,
        distance_cache_reused,
    ) = ensure_global_distance_cache(
        features=cpu_features,
        cache_dir=cache_dir,
        device=device,
        block_size=distance_block_size,
    )

    (
        selected_order,
        covering_radii_squared,
        order_metadata,
        order_cache_reused,
    ) = ensure_global_farthest_first_order(
        distance_path=distance_path,
        distance_metadata=distance_metadata,
        cache_dir=cache_dir,
        search_seed=search_seed,
        num_centers=max_selected_instances,
    )

    selected_prefix = np.asarray(
        selected_order[
            :selected_instances
        ],
        dtype=np.int64,
    )

    if len(np.unique(selected_prefix)) != selected_instances:
        raise RuntimeError(
            "The cached k-center order contains duplicate indices."
        )

    if (
        int(selected_prefix.min()) < 0
        or int(selected_prefix.max()) >= num_instances
    ):
        raise RuntimeError(
            "The cached k-center order contains indices outside the mask "
            "space."
        )

    selected_set = set(
        int(position)
        for position in selected_prefix.tolist()
    )

    mask = {
        position: int(position in selected_set)
        for position in range(num_instances)
    }

    available_by_class: dict[int, int] = {}
    selected_by_class: dict[int, int] = {}

    if normalized_targets is not None:
        available_by_class = dict(
            sorted(
                Counter(
                    normalized_targets
                ).items()
            )
        )

        selected_by_class = dict(
            sorted(
                Counter(
                    normalized_targets[position]
                    for position in selected_set
                ).items()
            )
        )

        for class_label in available_by_class:
            selected_by_class.setdefault(
                class_label,
                0,
            )

    covering_radius = sqrt(
        max(
            0.0,
            float(
                covering_radii_squared[
                    selected_instances - 1
                ]
            ),
        )
    )

    diagnostics = {
        "selection_scope": "global",
        "uses_class_labels_for_selection": False,
        "labels_used_only_for_reporting": (
            normalized_targets is not None
        ),
        "distance_metric": "squared_euclidean",
        "feature_normalization": "none",
        "initial_center_strategy": "seeded_random_global_instance",
        "first_center_position": int(
            order_metadata[
                "first_center_position"
            ]
        ),
        "traversal": "farthest_first",
        "requested_keep_percentage": keep_percentage,
        "max_cached_keep_percentage": max_keep_percentage,
        "selected_instances": selected_instances,
        "max_cached_centers": max_selected_instances,
        "covering_radius": covering_radius,
        "distance_cache_path": str(
            distance_path
        ),
        "distance_cache_reused": distance_cache_reused,
        "distance_cache_build_seconds": float(
            distance_metadata.get(
                "build_duration_seconds",
                0.0,
            )
        ),
        "distance_cache_size_gib": float(
            distance_metadata.get(
                "distance_file_size_gib",
                0.0,
            )
        ),
        "order_cache_path": str(
            order_metadata[
                "order_file"
            ]
        ),
        "order_cache_reused": order_cache_reused,
        "order_build_seconds": float(
            order_metadata.get(
                "build_duration_seconds",
                0.0,
            )
        ),
        "mask_materialization_and_cache_seconds": (
            time.perf_counter()
            - selection_start
        ),
        "available_by_class": available_by_class,
        "selected_by_class": selected_by_class,
    }

    return (
        mask,
        diagnostics,
    )
