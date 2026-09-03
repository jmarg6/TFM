from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Callable, Sequence, TypeAlias

import numpy as np
import torch
from PIL import Image
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    TensorDataset,
)
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, ImageFolder

from utils.algorithm_utils import Mask, validate_mask
from utils.config import DatasetName, DatasetSplit


# ==============================================================================
# TYPE ALIASES
# ==============================================================================

DataLoaders: TypeAlias = dict[str, DataLoader]


# ==============================================================================
# CONSTANTS
# ==============================================================================

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


# ==============================================================================
# LOGGER
# ==============================================================================

logger = logging.getLogger(__name__)


# ==============================================================================
# TINY IMAGENET VALIDATION DATASET
# ==============================================================================

class TinyImageNetValidationDataset(Dataset):
    """
    Load the official Tiny ImageNet validation split.

    The official Tiny ImageNet validation directory does not normally follow
    the class-folder structure expected by `torchvision.datasets.ImageFolder`.

    Instead, all validation images are stored inside:

        val/images/

    and their labels are defined in:

        val/val_annotations.txt

    This dataset reads the annotation file and maps the validation labels to
    the same integer class indices used by the training `ImageFolder` dataset.

    Args:
        root:
            Path to the official Tiny ImageNet validation directory.

        class_to_idx:
            Mapping from Tiny ImageNet WordNet IDs to the integer class indices
            used by the training dataset.

        transform:
            Optional image transformation applied to each loaded image.

    Raises:
        FileNotFoundError:
            If the validation image directory or annotation file is missing.

        ValueError:
            If the annotation file contains classes that are not present in
            the training class mapping.
    """

    def __init__(
        self,
        root: str | Path,
        class_to_idx: dict[str, int],
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.class_to_idx = class_to_idx.copy()

        images_dir = (
            self.root
            / "images"
        )

        annotations_path = (
            self.root
            / "val_annotations.txt"
        )

        if not images_dir.is_dir():
            raise FileNotFoundError(
                "Tiny ImageNet validation image directory was not found at "
                f"'{images_dir}'."
            )

        if not annotations_path.is_file():
            raise FileNotFoundError(
                "Tiny ImageNet validation annotations were not found at "
                f"'{annotations_path}'."
            )

        samples: list[
            tuple[Path, int]
        ] = []

        unknown_classes: set[str] = set()

        with annotations_path.open(
            mode="r",
            encoding="utf-8",
        ) as annotation_file:

            for line in annotation_file:
                parts = (
                    line.strip().split("\t")
                )

                if len(parts) < 2:
                    continue

                image_name = parts[0]
                class_id = parts[1]

                if class_id not in self.class_to_idx:
                    unknown_classes.add(
                        class_id
                    )

                    continue

                image_path = (
                    images_dir
                    / image_name
                )

                samples.append(
                    (
                        image_path,
                        self.class_to_idx[class_id],
                    )
                )

        if unknown_classes:
            raise ValueError(
                "Tiny ImageNet validation annotations contain classes that "
                "are missing from the training class mapping: "
                f"{sorted(unknown_classes)}."
            )

        if not samples:
            raise ValueError(
                "No Tiny ImageNet validation samples were found."
            )

        # A deterministic ordering makes the evaluation split independent of
        # the annotation-file implementation details.
        self.samples = sorted(
            samples,
            key=lambda sample: sample[0].name,
        )

        self.targets = [
            target
            for _, target in self.samples
        ]

    def __len__(
        self,
    ) -> int:
        """Return the number of validation samples."""
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int]:
        """
        Load one validation image and its class label.

        Args:
            index:
                Sample position.

        Returns:
            Transformed image and integer class label.
        """
        image_path, target = (
            self.samples[index]
        )

        with Image.open(
            image_path
        ) as image:
            image = (
                image
                .convert("RGB")
            )

            if self.transform is not None:
                image = self.transform(
                    image
                )

        return (
            image,
            target,
        )


# ==============================================================================
# DATASET NAME NORMALIZATION
# ==============================================================================

def normalize_dataset_name(
    dataset_name: DatasetName | str,
) -> DatasetName:
    """
    Normalize a dataset identifier into a validated `DatasetName`.

    Args:
        dataset_name:
            Dataset enum member or string identifier.

    Returns:
        Validated `DatasetName`.

    Raises:
        ValueError:
            If the dataset is unsupported.
    """
    if isinstance(
        dataset_name,
        DatasetName,
    ):
        return dataset_name

    normalized_name = (
        str(dataset_name)
        .strip()
        .upper()
    )

    try:
        return DatasetName(
            normalized_name
        )
    except ValueError as exc:
        allowed = [
            dataset.value
            for dataset in DatasetName
        ]

        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Allowed values: {allowed}."
        ) from exc


# ==============================================================================
# DEFAULT IMAGE TRANSFORM
# ==============================================================================

def build_default_image_transform(
    dataset_name: DatasetName | str,
    image_size: int = 224,
) -> transforms.Compose:
    """
    Build the default deterministic image preprocessing pipeline.

    All datasets are converted to three-channel tensors and normalized using
    ImageNet statistics so that they can be used with ImageNet-pretrained
    models.

    The transform is intentionally deterministic. This is particularly
    important during metaheuristic fitness evaluation, where uncontrolled
    stochastic image augmentation would introduce additional noise into the
    objective function.

    Args:
        dataset_name:
            Dataset identifier.

        image_size:
            Final square image resolution.

    Returns:
        Deterministic torchvision transformation pipeline.

    Raises:
        ValueError:
            If `image_size` is not positive.
    """
    if image_size <= 0:
        raise ValueError(
            f"'image_size' must be greater than 0, "
            f"but received {image_size}."
        )

    dataset = normalize_dataset_name(
        dataset_name
    )

    operations: list[
        Callable
    ] = [
        transforms.Resize(
            (
                image_size,
                image_size,
            )
        )
    ]

    if dataset == DatasetName.MNIST:
        operations.append(
            transforms.Grayscale(
                num_output_channels=3
            )
        )

    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    return transforms.Compose(
        operations
    )


# ==============================================================================
# TARGET EXTRACTION
# ==============================================================================

def extract_dataset_targets(
    dataset: Dataset,
) -> list[int]:
    """
    Extract integer class labels from a supported dataset.

    The helper supports original torchvision datasets, ``Subset`` wrappers,
    and ``TensorDataset`` objects used by the cached-feature pipeline.

    Args:
        dataset:
            Dataset whose labels must be recovered in its current positional
            order.

    Returns:
        List of integer class labels aligned with ``dataset`` positions.

    Raises:
        ValueError:
            If usable labels cannot be recovered or their count is
            inconsistent with the dataset length.
    """
    if isinstance(dataset, Subset):
        parent_targets = extract_dataset_targets(
            dataset.dataset
        )

        normalized_indices = [
            int(index)
            for index in dataset.indices
        ]

        try:
            subset_targets = [
                parent_targets[index]
                for index in normalized_indices
            ]
        except IndexError as exc:
            raise ValueError(
                "The Subset contains an index outside its parent dataset."
            ) from exc

        if len(subset_targets) != len(dataset):
            raise ValueError(
                "The number of extracted Subset targets does not match the "
                "Subset size."
            )

        return subset_targets

    if isinstance(dataset, TensorDataset):
        if len(dataset.tensors) < 2:
            raise ValueError(
                "TensorDataset target extraction requires at least two "
                "tensors: features and labels."
            )

        labels = dataset.tensors[1]
        normalized_labels = normalize_labels(
            labels,
            tensor_name="tensor_dataset_labels",
        ).tolist()

        if len(normalized_labels) != len(dataset):
            raise ValueError(
                "The number of TensorDataset labels does not match the "
                "dataset size."
            )

        return [
            int(target)
            for target in normalized_labels
        ]

    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")

        if isinstance(targets, torch.Tensor):
            targets = (
                targets
                .detach()
                .cpu()
                .reshape(-1)
                .tolist()
            )

        normalized_targets = [
            int(target)
            for target in targets
        ]

        if len(normalized_targets) != len(dataset):
            raise ValueError(
                "The number of extracted targets does not match the dataset "
                f"size. Targets: {len(normalized_targets)}, "
                f"dataset instances: {len(dataset)}."
            )

        return normalized_targets

    if hasattr(dataset, "samples"):
        samples = getattr(dataset, "samples")

        normalized_targets = [
            int(sample[1])
            for sample in samples
        ]

        if len(normalized_targets) != len(dataset):
            raise ValueError(
                "The number of labels recovered from 'samples' does not "
                "match the dataset size."
            )

        return normalized_targets

    raise ValueError(
        "The dataset does not expose usable class labels. Supported sources "
        "include 'targets', 'samples', TensorDataset, and Subset wrappers."
    )


# ==============================================================================
# FEATURE EXTRACTION FROM DATASET WRAPPERS
# ==============================================================================

def extract_dataset_features(
    dataset: Dataset,
) -> torch.Tensor:
    """
    Recover a feature matrix from a cached-feature dataset.

    The helper supports the ``TensorDataset`` and nested ``Subset`` wrappers
    created by ``get_feature_dataloaders``. Returned rows preserve the exact
    positional order of the supplied dataset, which is also the mask index
    space used by the instance-selection algorithms.

    Args:
        dataset:
            Cached-feature dataset whose feature rows must be recovered.

    Returns:
        CPU float tensor with shape ``[num_instances, feature_dimension]``.

    Raises:
        ValueError:
            If the dataset is not backed by cached feature tensors or the
            recovered matrix is inconsistent with the dataset size.
    """
    if isinstance(dataset, Subset):
        parent_features = extract_dataset_features(
            dataset.dataset
        )

        normalized_indices = torch.as_tensor(
            [
                int(index)
                for index in dataset.indices
            ],
            dtype=torch.long,
        )

        if normalized_indices.numel() != len(dataset):
            raise ValueError(
                "The number of Subset indices does not match its size."
            )

        if normalized_indices.numel() == 0:
            raise ValueError(
                "The cached-feature Subset is empty."
            )

        if (
            int(normalized_indices.min().item()) < 0
            or int(normalized_indices.max().item()) >= len(parent_features)
        ):
            raise ValueError(
                "The Subset contains an index outside its parent feature "
                "matrix."
            )

        return (
            parent_features
            .index_select(
                dim=0,
                index=normalized_indices,
            )
            .contiguous()
        )

    if isinstance(dataset, TensorDataset):
        if not dataset.tensors:
            raise ValueError(
                "TensorDataset feature extraction requires a feature tensor."
            )

        features = (
            dataset.tensors[0]
            .detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
        )

        if features.ndim != 2:
            raise ValueError(
                "Cached features must have shape "
                "[num_instances, feature_dimension], but received "
                f"{tuple(features.shape)}."
            )

        if len(features) != len(dataset):
            raise ValueError(
                "The number of cached feature rows does not match the "
                "TensorDataset size."
            )

        return features

    raise ValueError(
        "k-center greedy requires a cached-feature TensorDataset, optionally "
        "wrapped in one or more Subset objects."
    )


# ==============================================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ==============================================================================

def create_stratified_split_indices(
    targets: Sequence[int] | torch.Tensor,
    validation_fraction: float = 0.2,
    split_seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Create deterministic stratified training and validation indices.

    Each class is split independently so that both the training and validation
    partitions preserve class representation.

    The resulting index lists are sorted. Therefore, the selection-mask index
    space is stable and reproducible across raw-image and pre-extracted-feature
    loaders, provided that both representations preserve the same original
    sample ordering.

    Args:
        targets:
            Class label associated with each sample in the original training
            dataset.

        validation_fraction:
            Fraction of each class assigned to the validation partition.

        split_seed:
            Seed controlling which samples from each class are assigned to the
            validation partition.

    Returns:
        A tuple containing:

        - Original dataset indices assigned to the base training partition.
        - Original dataset indices assigned to the validation partition.

    Raises:
        ValueError:
            If the split configuration is invalid or a class contains fewer
            than two samples.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            f"'validation_fraction' must be in the interval (0, 1), "
            f"but received {validation_fraction}."
        )

    targets_tensor = torch.as_tensor(
        targets,
        dtype=torch.long,
    ).reshape(-1)

    if targets_tensor.numel() < 2:
        raise ValueError(
            "At least two samples are required to create training and "
            "validation partitions."
        )

    generator = (
        torch.Generator()
        .manual_seed(split_seed)
    )

    train_indices: list[int] = []
    validation_indices: list[int] = []

    classes = torch.unique(
        targets_tensor,
        sorted=True,
    )

    for class_label in classes:

        class_indices = torch.nonzero(
            targets_tensor == class_label,
            as_tuple=False,
        ).reshape(-1)

        class_size = int(
            class_indices.numel()
        )

        if class_size < 2:
            raise ValueError(
                f"Class {int(class_label)} contains only {class_size} sample. "
                "At least two samples per class are required for a stratified "
                "training/validation split."
            )

        permutation = torch.randperm(
            class_size,
            generator=generator,
        )

        shuffled_indices = (
            class_indices[permutation]
        )

        # Half-up rounding is used to obtain a deterministic integer number of
        # validation samples while preserving at least one sample in both
        # partitions.
        requested_validation_size = int(
            class_size
            * validation_fraction
            + 0.5
        )

        validation_size = max(
            1,
            min(
                requested_validation_size,
                class_size - 1,
            ),
        )

        class_validation_indices = (
            shuffled_indices[
                :validation_size
            ]
            .tolist()
        )

        class_train_indices = (
            shuffled_indices[
                validation_size:
            ]
            .tolist()
        )

        validation_indices.extend(
            class_validation_indices
        )

        train_indices.extend(
            class_train_indices
        )

    # Sorting creates a stable positional index space for the metaheuristics.
    train_indices.sort()
    validation_indices.sort()

    if not train_indices:
        raise ValueError(
            "The training partition is empty."
        )

    if not validation_indices:
        raise ValueError(
            "The validation partition is empty."
        )

    return (
        train_indices,
        validation_indices,
    )


# ==============================================================================
# SELECTION MASK VALIDATION
# ==============================================================================

def get_selected_positions(
    selection_mask: Mask,
    expected_size: int,
) -> list[int]:
    """
    Validate a selection mask and return its selected positions.

    The mask index space must correspond exactly to the base training
    partition:

        0, 1, ..., expected_size - 1

    Missing indices are treated as configuration errors rather than silently
    interpreted as discarded samples.

    Args:
        selection_mask:
            Binary mask produced by an instance-selection algorithm.

        expected_size:
            Exact number of instances in the base training partition.

    Returns:
        Positions whose mask value is equal to 1.

    Raises:
        ValueError:
            If the mask is invalid, has an incompatible index space, or
            selects no instances.
    """
    if expected_size <= 0:
        raise ValueError(
            f"'expected_size' must be greater than 0, "
            f"but received {expected_size}."
        )

    validate_mask(
        selection_mask
    )

    if len(selection_mask) != expected_size:
        raise ValueError(
            "The selection mask size does not match the base training "
            f"partition. Expected {expected_size} entries, "
            f"but received {len(selection_mask)}."
        )

    invalid_keys = [
        key
        for key in selection_mask
        if (
            type(key) is not int
            or key < 0
            or key >= expected_size
        )
    ]

    if invalid_keys:
        raise ValueError(
            "The selection mask must contain exactly the positional indices "
            f"from 0 to {expected_size - 1}. "
            f"Invalid keys found: {invalid_keys[:10]}."
        )

    selected_positions = [
        position
        for position in range(
            expected_size
        )
        if selection_mask[position] == 1
    ]

    if not selected_positions:
        raise ValueError(
            "The selection mask does not select any training instances."
        )

    return selected_positions


# ==============================================================================
# DATA LOADER HELPERS
# ==============================================================================

def resolve_num_workers(
    num_workers: int | None,
) -> int:
    """
    Resolve the number of DataLoader worker processes.

    By default, Windows uses zero worker processes to avoid common notebook and
    multiprocessing issues. Other operating systems use at most four workers.

    Args:
        num_workers:
            Explicit worker count or `None` for automatic selection.

    Returns:
        Non-negative worker count.

    Raises:
        ValueError:
            If an explicit negative worker count is provided.
    """
    if num_workers is not None:
        if num_workers < 0:
            raise ValueError(
                f"'num_workers' must be greater than or equal to 0, "
                f"but received {num_workers}."
            )

        return num_workers

    if os.name == "nt":
        return 0

    return min(
        4,
        os.cpu_count() or 1,
    )


def seed_data_loader_worker(
    worker_id: int,
) -> None:
    """
    Seed the random number generators used inside a DataLoader worker.

    PyTorch automatically assigns each worker a deterministic seed derived from
    the DataLoader generator.

    The worker-specific seed is propagated to:

    - Python's `random` module.
    - NumPy's random number generator.

    This improves reproducibility if current or future dataset transformations
    use either source of randomness.

    Args:
        worker_id:
            Worker identifier supplied automatically by PyTorch.
    """
    del worker_id

    worker_seed = (
        torch.initial_seed()
        % (2**32)
    )

    random.seed(
        worker_seed
    )

    np.random.seed(
        worker_seed
    )


def create_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    """
    Create a reproducible PyTorch DataLoader.

    Args:
        dataset:
            Dataset to load.

        batch_size:
            Number of samples per batch.

        shuffle:
            Whether to shuffle the dataset.

        num_workers:
            Number of worker processes.

        pin_memory:
            Whether to use pinned host memory.

        seed:
            Seed controlling data-order randomization and worker initialization.

    Returns:
        Configured DataLoader.

    Raises:
        ValueError:
            If the batch size is invalid.
    """
    if batch_size <= 0:
        raise ValueError(
            f"'batch_size' must be greater than 0, "
            f"but received {batch_size}."
        )

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_data_loader_worker,
        generator=generator,
        persistent_workers=(
            num_workers > 0
        ),
        drop_last=False,
    )


# ==============================================================================
# RAW IMAGE DATASETS
# ==============================================================================

def get_dataloaders(
    dataset_name: DatasetName | str,
    data_dir: str | Path = "./data",
    selection_mask: Mask | None = None,
    batch_size: int = 32,
    validation_fraction: float = 0.2,
    split_seed: int = 42,
    loader_seed: int = 42,
    image_size: int = 224,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    train_transform: Callable | None = None,
    eval_transform: Callable | None = None,
) -> tuple[DataLoaders, list[str]]:
    """
    Load raw-image training, validation, and test DataLoaders.

    The original training split is divided deterministically and stratified by
    class into:

        - Base training partition.
        - Validation partition.

    Instance-selection masks are applied exclusively to the base training
    partition.

    The original external test split is never modified and must never be used
    by the optimization algorithms.

    Mask semantics:

        A selection mask operates on the positional index space of the base
        training partition:

            0, 1, ..., len(base_train_partition) - 1

        The split is deterministic for a fixed `split_seed`, ensuring that the
        same mask refers to the same training instances across repeated calls.

    Args:
        dataset_name:
            Dataset to load.

        data_dir:
            Root directory containing or receiving the datasets.

        selection_mask:
            Optional binary instance-selection mask applied only to the base
            training partition.

        batch_size:
            Number of samples per batch.

        validation_fraction:
            Fraction of the original training data assigned to validation.

        split_seed:
            Seed controlling the deterministic stratified train/validation
            partition.

        loader_seed:
            Seed controlling training-data shuffling and DataLoader workers.

        image_size:
            Final square image resolution.

        num_workers:
            Number of DataLoader worker processes. If omitted, an
            operating-system-dependent default is used.

        pin_memory:
            Whether DataLoaders use pinned host memory. If omitted, it is
            enabled only when CUDA is available.

        train_transform:
            Optional transformation applied to training images.

            During metaheuristic fitness evaluation, deterministic transforms
            are recommended to reduce objective-function noise.

        eval_transform:
            Optional deterministic transformation applied to validation and
            test images.

    Returns:
        A tuple containing:

        - Dictionary with `train`, `valid`, and `test` DataLoaders.
        - List of class identifiers.

    Raises:
        FileNotFoundError:
            If Tiny ImageNet is not available at the expected location.

        ValueError:
            If the dataset or data-loading configuration is invalid.
    """
    dataset = normalize_dataset_name(
        dataset_name
    )

    data_root = Path(
        data_dir
    )

    data_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_num_workers = resolve_num_workers(
        num_workers
    )

    resolved_pin_memory = (
        torch.cuda.is_available()
        if pin_memory is None
        else pin_memory
    )

    default_transform = build_default_image_transform(
        dataset_name=dataset,
        image_size=image_size,
    )

    if train_transform is None:
        train_transform = (
            default_transform
        )

    if eval_transform is None:
        eval_transform = (
            default_transform
        )

    # --------------------------------------------------------------------------
    # Load dataset objects
    # --------------------------------------------------------------------------

    if dataset == DatasetName.MNIST:

        train_source = MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=train_transform,
        )

        validation_source = MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=eval_transform,
        )

        test_dataset = MNIST(
            root=data_root,
            train=False,
            download=True,
            transform=eval_transform,
        )

        classes = [
            str(class_index)
            for class_index in range(10)
        ]

    elif dataset == DatasetName.CIFAR10:

        train_source = CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=train_transform,
        )

        validation_source = CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=eval_transform,
        )

        test_dataset = CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=eval_transform,
        )

        classes = list(
            train_source.classes
        )

    elif dataset == DatasetName.TINYIMAGENET:

        tiny_imagenet_root = (
            data_root
            / "tiny-imagenet-200"
        )

        train_path = (
            tiny_imagenet_root
            / "train"
        )

        validation_path = (
            tiny_imagenet_root
            / "val"
        )

        if not train_path.is_dir():
            raise FileNotFoundError(
                "Tiny ImageNet training data was not found at "
                f"'{train_path}'."
            )

        if not validation_path.is_dir():
            raise FileNotFoundError(
                "Tiny ImageNet validation data was not found at "
                f"'{validation_path}'."
            )

        train_source = ImageFolder(
            root=train_path,
            transform=train_transform,
        )

        validation_source = ImageFolder(
            root=train_path,
            transform=eval_transform,
        )

        if (
            train_source.class_to_idx
            != validation_source.class_to_idx
        ):
            raise ValueError(
                "Tiny ImageNet training dataset instances produced "
                "inconsistent class mappings."
            )

        # The official labeled Tiny ImageNet validation set is used as the
        # held-out test set. A validation partition for model selection is
        # created exclusively from the original training data.
        test_dataset = (
            TinyImageNetValidationDataset(
                root=validation_path,
                class_to_idx=
                    train_source.class_to_idx,
                transform=eval_transform,
            )
        )

        classes = list(
            train_source.classes
        )

    else:
        # This branch is defensive because `normalize_dataset_name()` already
        # validates all supported values.
        raise ValueError(
            f"Unsupported dataset '{dataset}'."
        )

    # --------------------------------------------------------------------------
    # Create deterministic stratified training and validation partitions
    # --------------------------------------------------------------------------

    targets = extract_dataset_targets(
        train_source
    )

    (
        base_train_indices,
        validation_indices,
    ) = create_stratified_split_indices(
        targets=targets,
        validation_fraction=
            validation_fraction,
        split_seed=split_seed,
    )

    # --------------------------------------------------------------------------
    # Apply the instance-selection mask exclusively to the base training split
    # --------------------------------------------------------------------------

    if selection_mask is None:

        selected_original_indices = (
            base_train_indices
        )

    else:

        selected_positions = (
            get_selected_positions(
                selection_mask=
                    selection_mask,
                expected_size=
                    len(base_train_indices),
            )
        )

        # Convert positions in the metaheuristic search space back into
        # original dataset indices.
        selected_original_indices = [
            base_train_indices[position]
            for position in selected_positions
        ]

    train_dataset = Subset(
        train_source,
        selected_original_indices,
    )

    validation_dataset = Subset(
        validation_source,
        validation_indices,
    )

    # --------------------------------------------------------------------------
    # Build reproducible DataLoaders
    # --------------------------------------------------------------------------

    loaders: DataLoaders = {
        DatasetSplit.TRAIN.value:
            create_data_loader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=
                    resolved_num_workers,
                pin_memory=
                    resolved_pin_memory,
                seed=loader_seed,
            ),

        DatasetSplit.VALIDATION.value:
            create_data_loader(
                dataset=validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=
                    resolved_num_workers,
                pin_memory=
                    resolved_pin_memory,
                seed=loader_seed + 1,
            ),

        DatasetSplit.TEST.value:
            create_data_loader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=
                    resolved_num_workers,
                pin_memory=
                    resolved_pin_memory,
                seed=loader_seed + 2,
            ),
    }

    logger.info(
        "[%s] Raw images loaded | active_train=%d | base_train=%d | "
        "validation=%d | test=%d",
        dataset.value,
        len(train_dataset),
        len(base_train_indices),
        len(validation_dataset),
        len(test_dataset),
    )

    return (
        loaders,
        classes,
    )


# ==============================================================================
# FEATURE TENSOR HELPERS
# ==============================================================================

def load_tensor_file(
    path: str | Path,
) -> torch.Tensor:
    """
    Load one tensor file safely onto CPU memory.

    Args:
        path:
            Path to a serialized PyTorch tensor.

    Returns:
        Loaded tensor.

    Raises:
        FileNotFoundError:
            If the tensor file does not exist.

        TypeError:
            If the loaded object is not a PyTorch tensor.
    """
    tensor_path = Path(
        path
    )

    if not tensor_path.is_file():
        raise FileNotFoundError(
            f"Tensor file was not found at '{tensor_path}'."
        )

    loaded_object = torch.load(
        tensor_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(
        loaded_object,
        torch.Tensor,
    ):
        raise TypeError(
            f"Expected a torch.Tensor in '{tensor_path}', "
            f"but received {type(loaded_object).__name__}."
        )

    return loaded_object


def normalize_labels(
    labels: torch.Tensor,
    tensor_name: str,
) -> torch.Tensor:
    """
    Validate and convert a label tensor into one-dimensional integer labels.

    Args:
        labels:
            Original label tensor.

        tensor_name:
            Human-readable tensor identifier used in error messages.

    Returns:
        One-dimensional `torch.long` label tensor.

    Raises:
        ValueError:
            If labels are empty or contain non-integer floating-point values.
    """
    normalized_labels = (
        labels
        .detach()
        .cpu()
        .reshape(-1)
    )

    if normalized_labels.numel() == 0:
        raise ValueError(
            f"'{tensor_name}' must contain at least one label."
        )

    if normalized_labels.is_floating_point():

        rounded_labels = (
            normalized_labels.round()
        )

        if not torch.allclose(
            normalized_labels,
            rounded_labels,
        ):
            raise ValueError(
                f"'{tensor_name}' contains non-integer floating-point labels."
            )

    return normalized_labels.to(
        dtype=torch.long
    )


# ==============================================================================
# PRE-EXTRACTED FEATURE DATASETS
# ==============================================================================

def get_feature_dataloaders(
    features_dir: str | Path,
    selection_mask: Mask | None = None,
    batch_size: int = 256,
    validation_fraction: float = 0.2,
    split_seed: int = 42,
    loader_seed: int = 42,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> tuple[DataLoaders, int]:
    """
    Load training, validation, and test DataLoaders from feature tensors.

    The original training feature tensor is divided using the exact same
    deterministic stratified splitting logic used by the raw-image loader.

    Therefore, raw images and pre-extracted features share the same
    metaheuristic search space when:

        - The original sample ordering is preserved.
        - The same labels are stored.
        - The same `validation_fraction` is used.
        - The same `split_seed` is used.

    The selection mask is applied exclusively to the base training partition.

    Expected files:

        train_features.pt
        train_labels.pt
        test_features.pt
        test_labels.pt

    Args:
        features_dir:
            Directory containing the serialized feature tensors.

        selection_mask:
            Optional binary instance-selection mask.

        batch_size:
            Number of feature vectors per batch.

        validation_fraction:
            Fraction of the original training features assigned to validation.

        split_seed:
            Seed controlling the stratified train/validation partition.

        loader_seed:
            Seed controlling training-data shuffling and worker initialization.

        num_workers:
            Number of DataLoader worker processes.

        pin_memory:
            Whether to use pinned host memory. If omitted, it is enabled only
            when CUDA is available.

    Returns:
        A tuple containing:

        - Dictionary with `train`, `valid`, and `test` DataLoaders.
        - Number of classes found in the training labels.

    Raises:
        ValueError:
            If feature and label tensors are inconsistent.
    """
    features_root = Path(
        features_dir
    )

    train_features = load_tensor_file(
        features_root
        / "train_features.pt"
    )

    train_labels = normalize_labels(
        load_tensor_file(
            features_root
            / "train_labels.pt"
        ),
        tensor_name="train_labels",
    )

    test_features = load_tensor_file(
        features_root
        / "test_features.pt"
    )

    test_labels = normalize_labels(
        load_tensor_file(
            features_root
            / "test_labels.pt"
        ),
        tensor_name="test_labels",
    )

    # --------------------------------------------------------------------------
    # Validate feature / label consistency
    # --------------------------------------------------------------------------

    if train_features.ndim < 2:
        raise ValueError(
            "'train_features' must contain at least two dimensions: "
            "[num_samples, ...]."
        )

    if test_features.ndim < 2:
        raise ValueError(
            "'test_features' must contain at least two dimensions: "
            "[num_samples, ...]."
        )

    if len(train_features) != len(train_labels):
        raise ValueError(
            "Training feature and label counts do not match. "
            f"Features: {len(train_features)}, "
            f"labels: {len(train_labels)}."
        )

    if len(test_features) != len(test_labels):
        raise ValueError(
            "Test feature and label counts do not match. "
            f"Features: {len(test_features)}, "
            f"labels: {len(test_labels)}."
        )

    if len(train_features) == 0:
        raise ValueError(
            "The training feature tensor is empty."
        )

    if len(test_features) == 0:
        raise ValueError(
            "The test feature tensor is empty."
        )

    full_train_dataset = TensorDataset(
        train_features,
        train_labels,
    )

    test_dataset = TensorDataset(
        test_features,
        test_labels,
    )

    # --------------------------------------------------------------------------
    # Create the same stratified split used by the raw-image pipeline
    # --------------------------------------------------------------------------

    (
        base_train_indices,
        validation_indices,
    ) = create_stratified_split_indices(
        targets=train_labels,
        validation_fraction=
            validation_fraction,
        split_seed=split_seed,
    )

    # --------------------------------------------------------------------------
    # Apply the instance-selection mask only to the base training partition
    # --------------------------------------------------------------------------

    if selection_mask is None:

        selected_original_indices = (
            base_train_indices
        )

    else:

        selected_positions = (
            get_selected_positions(
                selection_mask=
                    selection_mask,
                expected_size=
                    len(base_train_indices),
            )
        )

        selected_original_indices = [
            base_train_indices[position]
            for position in selected_positions
        ]

    train_dataset = Subset(
        full_train_dataset,
        selected_original_indices,
    )

    validation_dataset = Subset(
        full_train_dataset,
        validation_indices,
    )

    resolved_num_workers = resolve_num_workers(
        num_workers
    )

    resolved_pin_memory = (
        torch.cuda.is_available()
        if pin_memory is None
        else pin_memory
    )

    # --------------------------------------------------------------------------
    # Build reproducible DataLoaders
    # --------------------------------------------------------------------------

    loaders: DataLoaders = {
        DatasetSplit.TRAIN.value:
            create_data_loader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=
                    resolved_num_workers,
                pin_memory=
                    resolved_pin_memory,
                seed=loader_seed,
            ),

        DatasetSplit.VALIDATION.value:
            create_data_loader(
                dataset=validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=
                    resolved_num_workers,
                pin_memory=
                    resolved_pin_memory,
                seed=loader_seed + 1,
            ),

        DatasetSplit.TEST.value:
            create_data_loader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=
                    resolved_num_workers,
                pin_memory=
                    resolved_pin_memory,
                seed=loader_seed + 2,
            ),
    }

    unique_train_classes = torch.unique(
        train_labels,
        sorted=True,
    )

    num_classes = int(
        unique_train_classes.numel()
    )

    logger.info(
        "[Features] Tensors loaded | active_train=%d | base_train=%d | "
        "validation=%d | test=%d | classes=%d",
        len(train_dataset),
        len(base_train_indices),
        len(validation_dataset),
        len(test_dataset),
        num_classes,
    )

    return (
        loaders,
        num_classes,
    )