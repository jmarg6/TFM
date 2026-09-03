from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from algorithms.genetic import genetic_algorithm
from algorithms.local_search import local_search
from algorithms.memetic import memetic_algorithm
from algorithms.random_search import random_search
from algorithms.stratified_random import create_stratified_random_mask
from utils.algorithm_utils import HistoryEntry, Mask, Metrics
from utils.config import (
    Algorithm,
    DatasetName,
    DatasetSplit,
    Metric,
    MetricAveraging,
    ModelName,
)
from utils.data_utils import (
    create_data_loader,
    get_dataloaders,
    extract_dataset_targets,
    get_feature_dataloaders,
    get_selected_positions,
)
from utils.train_utils import (
    TrainingResult,
    clone_model_state_to_cpu,
    set_training_seed,
    train_and_evaluate,
)


# ==============================================================================
# LOGGER
# ==============================================================================

logger = logging.getLogger("run_experiments")


# ==============================================================================
# TYPE ALIASES
# ==============================================================================

FitnessFunction = Callable[[Mask], Metrics]
AlgorithmResult = tuple[
    Mask,
    float,
    list[HistoryEntry],
    list[float],
    int,
]


# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_SEEDS = (
    42,
    123,
    456,
    789,
    1024,
)

MODEL_DEFAULT_IMAGE_SIZE: dict[ModelName, int] = {
    ModelName.ALEXNET: 224,
    ModelName.RESNEXT: 224,
    ModelName.EFFICIENTNETV2: 384,
    ModelName.SWIN_TRANSFORMER: 224,
}

METHODOLOGY_ARGUMENT_FIELDS = (
    "algorithms",
    "datasets",
    "models",
    "metrics",
    "metric_averaging",
    "keep_percentages",
    "seeds",
    "split_seed",
    "training_seed_offset",
    "final_seed_offset",
    "final_repeats",
    "max_evaluations",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "validation_fraction",
    "population_size",
    "tournament_size",
    "mutation_probability",
    "mutation_fraction",
    "local_search_num_swaps",
    "memetic_local_search_probability",
    "memetic_local_search_steps",
    "memetic_local_search_num_swaps",
    "image_size",
    "use_features",
    "no_pretrained",
    "fine_tune_all",
    "allow_nondeterminism",
)

EXPECTED_FROZEN_HEAD_PARAMETERS: dict[ModelName, set[str]] = {
    ModelName.ALEXNET: {
        "model.classifier.6.weight",
        "model.classifier.6.bias",
    },
    ModelName.RESNEXT: {
        "model.fc.weight",
        "model.fc.bias",
    },
    ModelName.EFFICIENTNETV2: {
        "model.classifier.1.weight",
        "model.classifier.1.bias",
    },
    ModelName.SWIN_TRANSFORMER: {
        "model.head.weight",
        "model.head.bias",
    },
}

SEARCH_SUMMARY_FIELDS = (
    "experiment_name",
    "configuration_fingerprint",
    "run_id",
    "algorithm",
    "dataset",
    "model",
    "target_metric",
    "metric_averaging",
    "keep_percentage",
    "selected_instances",
    "selected_percentage",
    "total_instances",
    "search_seed",
    "training_seed",
    "split_seed",
    "max_evaluations",
    "evaluations_done",
    "generated_subsets",
    "uses_validation_for_selection",
    "selection_strategy",
    "best_fitness",
    "search_duration_seconds",
    "population_size",
    "tournament_size",
    "mutation_probability",
    "mutation_fraction",
    "local_search_num_swaps",
    "memetic_local_search_probability",
    "memetic_local_search_steps",
    "memetic_local_search_num_swaps",
    "batch_size",
    "num_epochs",
    "learning_rate",
    "weight_decay",
    "validation_fraction",
    "pretrained",
    "freeze_backbone",
    "deterministic_algorithms",
    "device",
    "status",
    "error",
)

TEST_SUMMARY_FIELDS = (
    "experiment_name",
    "configuration_fingerprint",
    "record_type",
    "run_id",
    "algorithm",
    "dataset",
    "model",
    "target_metric",
    "metric_averaging",
    "keep_percentage",
    "selected_instances",
    "selected_percentage",
    "total_instances",
    "search_seed",
    "final_repeat",
    "final_training_seed",
    "split_seed",
    "batch_size",
    "num_epochs",
    "learning_rate",
    "weight_decay",
    "pretrained",
    "freeze_backbone",
    "device",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "evaluation_loss",
    "best_validation_loss",
    "best_epoch",
    "epochs_trained",
    "evaluation_split",
    "test_duration_seconds",
    "status",
    "error",
)


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass(frozen=True, slots=True)
class DataBundle:
    """
    Hold one fixed dataset/model preprocessing context.

    The base training, validation, and test datasets are created only once and
    reused by every candidate evaluation. This avoids repeatedly scanning raw
    image directories, particularly for Tiny ImageNet.
    """

    base_train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset
    classes: tuple[str, ...]
    input_features: int | None = None
    representation: str = "images"
    base_train_targets: tuple[int, ...] = ()

    @property
    def total_instances(self) -> int:
        """Return the number of instances in the metaheuristic search space."""
        return len(self.base_train_dataset)

    @property
    def num_classes(self) -> int:
        """Return the number of classification classes."""
        return len(self.classes)

    @property
    def uses_preextracted_features(self) -> bool:
        """Return whether the bundle contains frozen-backbone feature vectors."""
        return self.input_features is not None


@dataclass(frozen=True, slots=True)
class SearchRunRecord:
    """Store one completed or resumed search result."""

    run_id: str
    run_dir: Path
    algorithm: Algorithm
    dataset: DatasetName
    model: ModelName
    target_metric: Metric
    metric_averaging: MetricAveraging
    keep_percentage: float
    search_seed: int
    training_seed: int
    best_mask: Mask
    best_fitness: float | None


class FrozenBackboneController(nn.Module):
    """
    Keep the complete frozen feature extractor in evaluation mode.

    A plain call to `model.train()` would reactivate BatchNorm updates and
    Dropout inside frozen layers. That behavior is undesirable for a strict
    head-only transfer-learning protocol because the supposedly fixed feature
    extractor would remain stochastic or stateful.

    This wrapper keeps every frozen module in evaluation mode and re-enables
    training mode only for modules that directly own trainable parameters. In
    the current architectures, those modules are the newly created final
    classification layers.
    """

    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward inputs through the wrapped model."""
        return self.model(inputs)

    def train(
        self,
        mode: bool = True,
    ) -> FrozenBackboneController:
        """Set head-only training mode while keeping the backbone fixed."""
        super().train(mode)

        if mode:
            # Disable BatchNorm updates and Dropout throughout the frozen
            # feature extractor.
            self.model.eval()

            # Re-enable training mode only on modules that directly own at
            # least one trainable parameter. For the supported architectures,
            # this is exactly the replacement classification layer.
            for module in self.model.modules():
                direct_parameters = tuple(
                    module.parameters(
                        recurse=False
                    )
                )

                if any(
                    parameter.requires_grad
                    for parameter in direct_parameters
                ):
                    module.train(True)

        return self


# ==============================================================================
# SERIALIZATION HELPERS
# ==============================================================================

def to_serializable(
    value: Any,
) -> Any:
    """Convert common project values into JSON-compatible representations."""
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.device):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Mapping):
        return {
            str(key): to_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_serializable(item)
            for item in value
        ]

    return value


def write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Write a mapping to a formatted JSON file."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            to_serializable(payload),
            output_file,
            indent=2,
            sort_keys=True,
        )


def read_json(
    path: Path,
) -> dict[str, Any]:
    """Read a JSON object from disk."""
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


def build_runtime_versions() -> dict[str, str]:
    """Return the runtime library versions that affect reproducibility."""
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def build_methodology_payload(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Build the canonical methodological configuration for one experiment."""
    return {
        "arguments": {
            field_name: to_serializable(
                getattr(args, field_name)
            )
            for field_name in METHODOLOGY_ARGUMENT_FIELDS
        },
        "resolved_device": str(device),
        "versions": build_runtime_versions(),
        "training_protocol": (
            (
                "pre-extracted activations from a pretrained frozen feature "
                "extractor with only a fresh linear classifier trainable"
            )
            if args.use_features
            else (
                "pretrained frozen feature extractor with only the "
                "replacement classification layer trainable"
            )
        ),
    }


def calculate_configuration_fingerprint(
    payload: Mapping[str, Any],
) -> str:
    """Calculate a stable SHA-256 fingerprint for a methodology payload."""
    canonical_payload = json.dumps(
        to_serializable(payload),
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical_payload.encode("utf-8")
    ).hexdigest()


def validate_existing_experiment(
    manifest_path: Path,
    expected_fingerprint: str,
) -> dict[str, Any] | None:
    """Validate that an existing experiment can be resumed safely."""
    if not manifest_path.is_file():
        return None

    manifest = read_json(
        manifest_path
    )

    existing_fingerprint = manifest.get(
        "configuration_fingerprint"
    )

    if existing_fingerprint != expected_fingerprint:
        raise ValueError(
            "The selected experiment directory already contains results from "
            "a different methodological configuration. Use a new "
            "'--experiment-name' instead of mixing incompatible results."
        )

    return manifest


def completed_result_exists(
    path: Path,
    expected_fingerprint: str,
) -> bool:
    """Return whether a result exists, is compatible, and completed."""
    if not path.is_file():
        return False

    try:
        payload = read_json(
            path
        )
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        logger.warning(
            "Ignoring unreadable result file and recomputing it: %s",
            path,
        )
        return False

    stored_fingerprint = payload.get(
        "configuration_fingerprint"
    )

    if stored_fingerprint != expected_fingerprint:
        raise ValueError(
            f"Result file '{path}' belongs to a different methodological "
            "configuration. Use a new experiment name."
        )

    return payload.get("status") == "completed"


def write_rows_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write a complete CSV file using the union of all row keys."""
    if not rows:
        raise ValueError(
            "'rows' must contain at least one record."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = sorted(
        {
            str(key)
            for row in rows
            for key in row.keys()
        }
    )

    with path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: to_serializable(
                        row.get(key)
                    )
                    for key in fieldnames
                }
            )


def append_row_csv(
    path: Path,
    row: Mapping[str, Any],
    fieldnames: Sequence[str],
) -> None:
    """Append one row to a CSV file with a stable predefined schema."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = path.is_file()

    with path.open(
        mode="a",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                key: to_serializable(
                    row.get(key)
                )
                for key in fieldnames
            }
        )


def save_mask(
    path: Path,
    mask: Mask,
) -> None:
    """Save a mask compactly as selected indices and total size."""
    selected_indices = [
        index
        for index, selected in mask.items()
        if selected == 1
    ]

    write_json(
        path,
        {
            "total_instances": len(mask),
            "selected_instances": len(selected_indices),
            "selected_indices": selected_indices,
        },
    )


def load_mask(
    path: Path,
    expected_size: int | None = None,
) -> Mask:
    """Load and validate a mask saved by `save_mask`."""
    payload = read_json(
        path
    )

    total_instances = int(
        payload["total_instances"]
    )

    if total_instances <= 0:
        raise ValueError(
            f"Saved mask '{path}' has an invalid total size."
        )

    if (
        expected_size is not None
        and total_instances != expected_size
    ):
        raise ValueError(
            f"Saved mask '{path}' contains {total_instances} positions, but "
            f"the current search space contains {expected_size}."
        )

    raw_selected_indices = [
        int(index)
        for index in payload[
            "selected_indices"
        ]
    ]

    selected_indices = set(
        raw_selected_indices
    )

    if len(selected_indices) != len(raw_selected_indices):
        raise ValueError(
            f"Saved mask '{path}' contains duplicate selected indices."
        )

    if any(
        index < 0
        or index >= total_instances
        for index in selected_indices
    ):
        raise ValueError(
            f"Saved mask '{path}' contains invalid indices."
        )

    stored_selected_instances = int(
        payload.get(
            "selected_instances",
            len(selected_indices),
        )
    )

    if stored_selected_instances != len(selected_indices):
        raise ValueError(
            f"Saved mask '{path}' contains an inconsistent selected count."
        )

    return {
        index: int(
            index in selected_indices
        )
        for index in range(
            total_instances
        )
    }


# ==============================================================================
# ENUM PARSING
# ==============================================================================

def parse_enum_value(
    enum_type: type[Enum],
    raw_value: str,
) -> Enum:
    """Parse either an enum member name or its stored value."""
    normalized = raw_value.strip().lower()

    for member in enum_type:
        if (
            member.name.lower() == normalized
            or str(member.value).lower() == normalized
        ):
            return member

    allowed = sorted(
        {
            member.name
            for member in enum_type
        }
        | {
            str(member.value)
            for member in enum_type
        }
    )

    raise argparse.ArgumentTypeError(
        f"Unsupported value '{raw_value}'. "
        f"Allowed values: {allowed}."
    )


def parse_algorithm(
    value: str,
) -> Algorithm:
    """Parse an algorithm command-line value."""
    return parse_enum_value(
        Algorithm,
        value,
    )  # type: ignore[return-value]


def parse_dataset(
    value: str,
) -> DatasetName:
    """Parse a dataset command-line value."""
    return parse_enum_value(
        DatasetName,
        value,
    )  # type: ignore[return-value]


def parse_model(
    value: str,
) -> ModelName:
    """Parse a model command-line value."""
    return parse_enum_value(
        ModelName,
        value,
    )  # type: ignore[return-value]


def parse_metric(
    value: str,
) -> Metric:
    """Parse a target-metric command-line value."""
    return parse_enum_value(
        Metric,
        value,
    )  # type: ignore[return-value]


def parse_metric_averaging(
    value: str,
) -> MetricAveraging:
    """Parse a metric-averaging command-line value."""
    return parse_enum_value(
        MetricAveraging,
        value,
    )  # type: ignore[return-value]


def parse_keep_percentage(
    value: str,
) -> float:
    """Parse a fixed retention percentage in the open interval (0, 1)."""
    try:
        percentage = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid percentage '{value}'."
        ) from exc

    if not 0.0 < percentage < 1.0:
        raise argparse.ArgumentTypeError(
            "Retention percentages must be in the interval (0, 1). "
            "The 100% search-space baseline is evaluated separately."
        )

    return percentage


# ==============================================================================
# MODEL FACTORY AND PREPROCESSING
# ==============================================================================

def get_default_weights(
    model_name: ModelName,
) -> Any:
    """Return torchvision's default weights enum for one architecture."""
    if model_name == ModelName.ALEXNET:
        return models.AlexNet_Weights.DEFAULT

    if model_name == ModelName.RESNEXT:
        return models.ResNeXt50_32X4D_Weights.DEFAULT

    if model_name == ModelName.EFFICIENTNETV2:
        return models.EfficientNet_V2_S_Weights.DEFAULT

    if model_name == ModelName.SWIN_TRANSFORMER:
        return models.Swin_T_Weights.DEFAULT

    raise ValueError(
        f"Unsupported model '{model_name}'."
    )


def build_preprocessing_transform(
    dataset_name: DatasetName,
    model_name: ModelName,
    pretrained: bool,
    explicit_image_size: int,
) -> tuple[Callable | None, int]:
    """
    Build deterministic preprocessing for one dataset/model pair.

    With pretrained weights and no explicit image size, the official
    torchvision weight transform is used. Supplying an explicit image size
    delegates preprocessing to `data_utils.build_default_image_transform`.
    """
    if explicit_image_size < 0:
        raise ValueError(
            "'image_size' must be greater than or equal to 0."
        )

    resolved_image_size = (
        explicit_image_size
        if explicit_image_size > 0
        else MODEL_DEFAULT_IMAGE_SIZE[
            model_name
        ]
    )

    if not pretrained or explicit_image_size > 0:
        return (
            None,
            resolved_image_size,
        )

    weights = get_default_weights(
        model_name
    )

    model_transform = weights.transforms()

    if dataset_name == DatasetName.MNIST:
        return (
            transforms.Compose(
                [
                    transforms.Grayscale(
                        num_output_channels=3
                    ),
                    model_transform,
                ]
            ),
            resolved_image_size,
        )

    return (
        model_transform,
        resolved_image_size,
    )


def build_model(
    model_name: ModelName,
    num_classes: int,
    load_pretrained_weights: bool,
    freeze_backbone: bool,
) -> nn.Module:
    """Build a supported architecture and replace its classification head."""
    if num_classes <= 1:
        raise ValueError(
            f"'num_classes' must be greater than 1, "
            f"but received {num_classes}."
        )

    if model_name == ModelName.ALEXNET:
        model = models.alexnet(
            weights=(
                models.AlexNet_Weights.DEFAULT
                if load_pretrained_weights
                else None
            )
        )

        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False

        input_features = (
            model.classifier[6].in_features
        )

        model.classifier[6] = nn.Linear(
            input_features,
            num_classes,
        )

    elif model_name == ModelName.RESNEXT:
        model = models.resnext50_32x4d(
            weights=(
                models.ResNeXt50_32X4D_Weights.DEFAULT
                if load_pretrained_weights
                else None
            )
        )

        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

    elif model_name == ModelName.EFFICIENTNETV2:
        model = models.efficientnet_v2_s(
            weights=(
                models.EfficientNet_V2_S_Weights.DEFAULT
                if load_pretrained_weights
                else None
            )
        )

        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes,
        )

    elif model_name == ModelName.SWIN_TRANSFORMER:
        model = models.swin_t(
            weights=(
                models.Swin_T_Weights.DEFAULT
                if load_pretrained_weights
                else None
            )
        )

        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False

        model.head = nn.Linear(
            model.head.in_features,
            num_classes,
        )

    else:
        raise ValueError(
            f"Unsupported model '{model_name}'."
        )

    if freeze_backbone:
        model = FrozenBackboneController(
            model
        )

        validate_frozen_backbone_model(
            model=model,
            model_name=model_name,
        )

    return model


def get_replacement_classifier(
    model: nn.Module,
    model_name: ModelName,
) -> nn.Linear:
    """
    Return the replacement classification layer from a supported model.

    The helper accepts both a raw torchvision model and the
    `FrozenBackboneController` wrapper used by the image pipeline.
    """
    base_model = (
        model.model
        if isinstance(
            model,
            FrozenBackboneController,
        )
        else model
    )

    if model_name == ModelName.ALEXNET:
        classifier = base_model.classifier[6]
    elif model_name == ModelName.RESNEXT:
        classifier = base_model.fc
    elif model_name == ModelName.EFFICIENTNETV2:
        classifier = base_model.classifier[1]
    elif model_name == ModelName.SWIN_TRANSFORMER:
        classifier = base_model.head
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'."
        )

    if not isinstance(
        classifier,
        nn.Linear,
    ):
        raise TypeError(
            "The replacement classifier must be an nn.Linear layer, but "
            f"received {type(classifier).__name__}."
        )

    return classifier


def create_initial_model_state(
    model_name: ModelName,
    num_classes: int,
    training_seed: int,
    pretrained: bool,
    freeze_backbone: bool,
    deterministic_algorithms: bool,
    input_features: int | None = None,
) -> dict[str, torch.Tensor]:
    """
    Create one reproducible CPU initial state shared within a run.

    In feature mode, the original complete architecture is instantiated once
    only to reproduce the exact seeded initialization of its replacement
    classifier. Only that classifier state is retained afterwards.
    """
    set_training_seed(
        seed=training_seed,
        deterministic_algorithms=
            deterministic_algorithms,
    )

    template_model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        load_pretrained_weights=pretrained,
        freeze_backbone=freeze_backbone,
    )

    if input_features is None:
        initial_state = clone_model_state_to_cpu(
            template_model
        )
    else:
        replacement_classifier = (
            get_replacement_classifier(
                model=template_model,
                model_name=model_name,
            )
        )

        if (
            replacement_classifier.in_features
            != input_features
        ):
            raise ValueError(
                "The extracted feature dimension does not match the "
                "replacement classifier input dimension. "
                f"Features: {input_features}, "
                f"classifier: {replacement_classifier.in_features}."
            )

        initial_state = clone_model_state_to_cpu(
            replacement_classifier
        )

    del template_model
    gc.collect()

    return initial_state


def instantiate_model_from_state(
    model_name: ModelName,
    num_classes: int,
    initial_state: Mapping[str, torch.Tensor],
    freeze_backbone: bool,
    device: torch.device,
    input_features: int | None = None,
) -> nn.Module:
    """Instantiate a fresh image model or feature-space linear classifier."""
    if input_features is None:
        model = build_model(
            model_name=model_name,
            num_classes=num_classes,
            load_pretrained_weights=False,
            freeze_backbone=freeze_backbone,
        )
    else:
        if input_features <= 0:
            raise ValueError(
                "'input_features' must be greater than 0 in feature mode."
            )

        model = nn.Linear(
            input_features,
            num_classes,
        )

    model.load_state_dict(
        initial_state,
        strict=True,
    )

    return model.to(
        device
    )

def get_trainable_parameters(
    model: nn.Module,
) -> list[nn.Parameter]:
    """Return the model's non-empty collection of trainable parameters."""
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not parameters:
        raise ValueError(
            "The model does not contain trainable parameters."
        )

    return parameters

def validate_frozen_backbone_model(
    model: nn.Module,
    model_name: ModelName,
) -> None:
    """Validate that only the intended replacement head is trainable."""
    actual_trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    expected_trainable_names = EXPECTED_FROZEN_HEAD_PARAMETERS[
        model_name
    ]

    if actual_trainable_names != expected_trainable_names:
        raise RuntimeError(
            "Frozen-backbone configuration is invalid for "
            f"'{model_name.value}'. Expected trainable parameters "
            f"{sorted(expected_trainable_names)}, but found "
            f"{sorted(actual_trainable_names)}."
        )


# ==============================================================================
# DATA AND DEVICE HELPERS
# ==============================================================================

def resolve_device(
    requested_device: str,
) -> torch.device:
    """Resolve `auto`, `cpu`, or `cuda` into a PyTorch device."""
    normalized = (
        requested_device
        .strip()
        .lower()
    )

    if normalized == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    if normalized == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device '{requested_device}'."
    )


def prepare_data_bundle(
    *,
    dataset: DatasetName,
    model_name: ModelName,
    data_dir: Path,
    features_dir: Path,
    use_features: bool,
    batch_size: int,
    validation_fraction: float,
    split_seed: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
) -> tuple[DataBundle, int]:
    """
    Load one fixed image or pre-extracted-feature preprocessing context.

    Feature tensors are expected under:

        FEATURES_DIR / DATASET / MODEL /

    and must contain the four files accepted by
    `utils.data_utils.get_feature_dataloaders`.
    """
    if use_features:
        feature_path = (
            features_dir
            / dataset.value
            / model_name.value
        )

        metadata_path = (
            feature_path
            / "metadata.json"
        )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                "Feature metadata was not found at "
                f"'{metadata_path}'. Run extract_features.py first."
            )

        feature_metadata = read_json(
            metadata_path
        )

        if feature_metadata.get("dataset") != dataset.value:
            raise ValueError(
                "Feature metadata dataset mismatch. Expected "
                f"'{dataset.value}', but found "
                f"'{feature_metadata.get('dataset')}'."
            )

        if feature_metadata.get("model") != model_name.value:
            raise ValueError(
                "Feature metadata model mismatch. Expected "
                f"'{model_name.value}', but found "
                f"'{feature_metadata.get('model')}'."
            )

        if feature_metadata.get("pretrained") is not True:
            raise ValueError(
                "Feature tensors must come from ImageNet-pretrained weights."
            )

        if feature_metadata.get("backbone_frozen") is not True:
            raise ValueError(
                "Feature tensors must come from a frozen backbone in "
                "evaluation mode."
            )

        loaders, num_classes = (
            get_feature_dataloaders(
                features_dir=feature_path,
                selection_mask=None,
                batch_size=batch_size,
                validation_fraction=
                    validation_fraction,
                split_seed=split_seed,
                loader_seed=split_seed,
                # The retained datasets are reused later. Worker processes are
                # therefore unnecessary while constructing this bundle.
                num_workers=0,
                pin_memory=False,
            )
        )

        sample_features, _ = loaders[
            DatasetSplit.TRAIN.value
        ].dataset[0]

        test_sample_features, _ = loaders[
            DatasetSplit.TEST.value
        ].dataset[0]

        if (
            not isinstance(
                sample_features,
                torch.Tensor,
            )
            or sample_features.ndim != 1
        ):
            raise ValueError(
                "Pre-extracted features must be one-dimensional vectors per "
                "sample. "
                f"Received {type(sample_features).__name__} with shape "
                f"{getattr(sample_features, 'shape', None)}."
            )

        input_features = int(
            sample_features.numel()
        )

        if input_features <= 0:
            raise ValueError(
                "The extracted feature dimension must be greater than 0."
            )

        if (
            not isinstance(
                test_sample_features,
                torch.Tensor,
            )
            or test_sample_features.ndim != 1
            or int(
                test_sample_features.numel()
            ) != input_features
        ):
            raise ValueError(
                "Training and test feature dimensions are inconsistent."
            )

        metadata_feature_dimension = int(
            feature_metadata.get(
                "feature_dimension",
                -1,
            )
        )

        if metadata_feature_dimension != input_features:
            raise ValueError(
                "Feature metadata dimension mismatch. Metadata: "
                f"{metadata_feature_dimension}, tensors: {input_features}."
            )

        metadata_num_classes = int(
            feature_metadata.get(
                "num_classes",
                -1,
            )
        )

        if metadata_num_classes != num_classes:
            raise ValueError(
                "Feature metadata class-count mismatch. Metadata: "
                f"{metadata_num_classes}, tensors: {num_classes}."
            )

        base_train_dataset = loaders[
            DatasetSplit.TRAIN.value
        ].dataset

        bundle = DataBundle(
            base_train_dataset=base_train_dataset,
            validation_dataset=loaders[
                DatasetSplit.VALIDATION.value
            ].dataset,
            test_dataset=loaders[
                DatasetSplit.TEST.value
            ].dataset,
            classes=tuple(
                str(class_index)
                for class_index in range(
                    num_classes
                )
            ),
            input_features=input_features,
            representation="preextracted_features",
            base_train_targets=tuple(
                extract_dataset_targets(
                    base_train_dataset
                )
            ),
        )

        del loaders
        gc.collect()

        # Image size is not used once frozen features have been extracted.
        return (
            bundle,
            0,
        )

    transform, resolved_image_size = (
        build_preprocessing_transform(
            dataset_name=dataset,
            model_name=model_name,
            pretrained=pretrained,
            explicit_image_size=image_size,
        )
    )

    loaders, classes = get_dataloaders(
        dataset_name=dataset,
        data_dir=data_dir,
        selection_mask=None,
        batch_size=batch_size,
        validation_fraction=
            validation_fraction,
        split_seed=split_seed,
        loader_seed=split_seed,
        image_size=resolved_image_size,
        # Only the dataset objects are retained here; candidate loaders are
        # built later with the requested worker count.
        num_workers=0,
        train_transform=transform,
        eval_transform=transform,
    )

    base_train_dataset = loaders[
        DatasetSplit.TRAIN.value
    ].dataset

    bundle = DataBundle(
        base_train_dataset=base_train_dataset,
        validation_dataset=loaders[
            DatasetSplit.VALIDATION.value
        ].dataset,
        test_dataset=loaders[
            DatasetSplit.TEST.value
        ].dataset,
        classes=tuple(classes),
        input_features=None,
        representation="images",
        base_train_targets=tuple(
            extract_dataset_targets(
                base_train_dataset
            )
        ),
    )

    del loaders
    gc.collect()

    return (
        bundle,
        resolved_image_size,
    )

def build_loaders_from_bundle(
    *,
    bundle: DataBundle,
    selection_mask: Mask | None,
    batch_size: int,
    loader_seed: int,
    num_workers: int,
    pin_memory: bool,
    include_test: bool,
) -> dict[str, DataLoader]:
    """Build reproducible loaders without reloading or rescanning datasets."""
    if selection_mask is None:
        train_dataset = (
            bundle.base_train_dataset
        )
    else:
        selected_positions = get_selected_positions(
            selection_mask=selection_mask,
            expected_size=
                bundle.total_instances,
        )

        train_dataset = Subset(
            bundle.base_train_dataset,
            selected_positions,
        )

    loaders = {
        DatasetSplit.TRAIN.value:
            create_data_loader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                seed=loader_seed,
            ),
        DatasetSplit.VALIDATION.value:
            create_data_loader(
                dataset=
                    bundle.validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                seed=loader_seed + 1,
            ),
    }

    if include_test:
        loaders[
            DatasetSplit.TEST.value
        ] = create_data_loader(
            dataset=bundle.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=loader_seed + 2,
        )

    return loaders


# ==============================================================================
# FITNESS FACTORY
# ==============================================================================

def create_fitness_function(
    *,
    bundle: DataBundle,
    model_name: ModelName,
    initial_model_state: Mapping[str, torch.Tensor],
    device: torch.device,
    training_seed: int,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    metric_averaging: MetricAveraging,
    freeze_backbone: bool,
    deterministic_algorithms: bool,
    empty_cuda_cache: bool,
) -> FitnessFunction:
    """Create a deterministic validation-fitness function."""
    pin_memory = (
        device.type == "cuda"
    )

    def fitness_function(
        selection_mask: Mask,
    ) -> dict[str, Any]:
        evaluation_start = (
            time.perf_counter()
        )

        set_training_seed(
            seed=training_seed,
            deterministic_algorithms=
                deterministic_algorithms,
        )

        loaders = build_loaders_from_bundle(
            bundle=bundle,
            selection_mask=selection_mask,
            batch_size=batch_size,
            loader_seed=training_seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            include_test=False,
        )

        model = instantiate_model_from_state(
            model_name=model_name,
            num_classes=bundle.num_classes,
            initial_state=initial_model_state,
            freeze_backbone=freeze_backbone,
            device=device,
            input_features=bundle.input_features,
        )

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            get_trainable_parameters(model),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        metrics = train_and_evaluate(
            model=model,
            loaders=loaders,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            eval_split=
                DatasetSplit.VALIDATION,
            metric_averaging=
                metric_averaging,
        )

        metrics["candidate_duration_seconds"] = (
            time.perf_counter()
            - evaluation_start
        )
        metrics["training_seed"] = training_seed

        del optimizer
        del criterion
        del model
        del loaders

        gc.collect()

        if (
            empty_cuda_cache
            and device.type == "cuda"
        ):
            torch.cuda.empty_cache()

        return metrics

    return fitness_function


# ==============================================================================
# ALGORITHM DISPATCH
# ==============================================================================

def execute_algorithm(
    *,
    algorithm: Algorithm,
    fitness_function: FitnessFunction,
    total_instances: int,
    keep_percentage: float,
    max_evaluations: int,
    target_metric: Metric,
    search_seed: int,
    population_size: int,
    tournament_size: int,
    mutation_probability: float,
    mutation_fraction: float,
    local_search_num_swaps: int,
    memetic_local_search_probability: float,
    memetic_local_search_steps: int,
    memetic_local_search_num_swaps: int,
) -> AlgorithmResult:
    """Execute one algorithm with an independent search RNG."""
    search_rng = random.Random(
        search_seed
    )

    common_arguments = {
        "fitness_func": fitness_function,
        "total_instances": total_instances,
        "keep_percentage": keep_percentage,
        "max_evaluations": max_evaluations,
        "target_metric": target_metric,
        "rng": search_rng,
    }

    if algorithm == Algorithm.RANDOM_SEARCH:
        return random_search(
            **common_arguments,
        )

    if algorithm == Algorithm.LOCAL_SEARCH:
        return local_search(
            **common_arguments,
            num_swaps=local_search_num_swaps,
        )

    if algorithm == Algorithm.GENETIC:
        return genetic_algorithm(
            **common_arguments,
            population_size=population_size,
            tournament_size=tournament_size,
            mutation_probability=
                mutation_probability,
            mutation_fraction=
                mutation_fraction,
        )

    if algorithm == Algorithm.MEMETIC:
        return memetic_algorithm(
            **common_arguments,
            population_size=population_size,
            tournament_size=tournament_size,
            mutation_probability=
                mutation_probability,
            mutation_fraction=
                mutation_fraction,
            local_search_probability=
                memetic_local_search_probability,
            local_search_steps=
                memetic_local_search_steps,
            local_search_num_swaps=
                memetic_local_search_num_swaps,
        )

    raise ValueError(
        f"Unsupported algorithm '{algorithm}'."
    )


# ==============================================================================
# FINAL TEST EVALUATION
# ==============================================================================

def evaluate_mask_on_test(
    *,
    selection_mask: Mask | None,
    bundle: DataBundle,
    model_name: ModelName,
    initial_model_state: Mapping[str, torch.Tensor],
    device: torch.device,
    final_training_seed: int,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    metric_averaging: MetricAveraging,
    freeze_backbone: bool,
    deterministic_algorithms: bool,
    empty_cuda_cache: bool,
) -> TrainingResult:
    """Retrain one mask and evaluate it once on the held-out test split."""
    set_training_seed(
        seed=final_training_seed,
        deterministic_algorithms=
            deterministic_algorithms,
    )

    loaders = build_loaders_from_bundle(
        bundle=bundle,
        selection_mask=selection_mask,
        batch_size=batch_size,
        loader_seed=final_training_seed,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        include_test=True,
    )

    model = instantiate_model_from_state(
        model_name=model_name,
        num_classes=bundle.num_classes,
        initial_state=initial_model_state,
        freeze_backbone=freeze_backbone,
        device=device,
        input_features=bundle.input_features,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        get_trainable_parameters(model),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    metrics = train_and_evaluate(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_split=DatasetSplit.TEST,
        metric_averaging=metric_averaging,
    )

    del optimizer
    del criterion
    del model
    del loaders

    gc.collect()

    if (
        empty_cuda_cache
        and device.type == "cuda"
    ):
        torch.cuda.empty_cache()

    return metrics


# ==============================================================================
# IDENTIFIERS AND LOGGING
# ==============================================================================

def build_run_id(
    *,
    algorithm: Algorithm,
    dataset: DatasetName,
    model: ModelName,
    metric: Metric,
    keep_percentage: float,
    search_seed: int,
) -> str:
    """Build a stable identifier for one search run."""
    percentage_tag = int(
        round(
            keep_percentage * 10_000
        )
    )

    return (
        f"{dataset.value}_"
        f"{model.value}_"
        f"{algorithm.value}_"
        f"{metric.value}_"
        f"kp{percentage_tag:04d}_"
        f"seed{search_seed}"
    )


def configure_logging(
    log_path: Path,
    verbose: bool,
) -> None:
    """Configure console and file logging."""
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.basicConfig(
        level=(
            logging.DEBUG
            if verbose
            else logging.INFO
        ),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        handlers=[
            logging.StreamHandler(
                sys.stdout
            ),
            logging.FileHandler(
                log_path,
                encoding="utf-8",
            ),
        ],
        force=True,
    )


# ==============================================================================
# ARGUMENT VALIDATION
# ==============================================================================

def validate_arguments(
    args: argparse.Namespace,
) -> None:
    """Validate cross-argument experiment constraints."""
    positive_integer_fields = (
        "epochs",
        "batch_size",
        "population_size",
        "tournament_size",
        "local_search_num_swaps",
        "memetic_local_search_steps",
        "memetic_local_search_num_swaps",
        "final_repeats",
    )

    for field_name in positive_integer_fields:
        value = int(
            getattr(
                args,
                field_name,
            )
        )

        if value <= 0:
            raise ValueError(
                f"'--{field_name.replace('_', '-')}' must be greater than 0."
            )

    if args.max_evaluations < 0:
        raise ValueError(
            "'--max-evaluations' must be greater than or equal to 0."
        )

    if (
        any(
            algorithm != Algorithm.STRATIFIED_RANDOM
            for algorithm in args.algorithms
        )
        and args.max_evaluations <= 0
    ):
        raise ValueError(
            "'--max-evaluations' must be greater than 0 whenever an "
            "optimization algorithm is requested. SRS alone may use 0 "
            "because it performs no fitness evaluations."
        )

    if args.num_workers < 0:
        raise ValueError(
            "'--num-workers' must be greater than or equal to 0."
        )

    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError(
            "'--validation-fraction' must be in the interval (0, 1)."
        )

    if args.learning_rate <= 0.0:
        raise ValueError(
            "'--learning-rate' must be greater than 0."
        )

    if args.weight_decay < 0.0:
        raise ValueError(
            "'--weight-decay' must be greater than or equal to 0."
        )

    if not 0.0 <= args.mutation_probability <= 1.0:
        raise ValueError(
            "'--mutation-probability' must be in [0, 1]."
        )

    if not 0.0 < args.mutation_fraction <= 1.0:
        raise ValueError(
            "'--mutation-fraction' must be in (0, 1]."
        )

    if not 0.0 <= args.memetic_local_search_probability <= 1.0:
        raise ValueError(
            "'--memetic-local-search-probability' must be in [0, 1]."
        )

    if args.tournament_size > args.population_size - 1:
        raise ValueError(
            "'--tournament-size' must be less than or equal to "
            "population_size - 1."
        )

    if any(
        algorithm in {
            Algorithm.GENETIC,
            Algorithm.MEMETIC,
        }
        for algorithm in args.algorithms
    ) and args.max_evaluations < args.population_size:
        raise ValueError(
            "For GEN or MEM, '--max-evaluations' must be greater than or "
            "equal to '--population-size'."
        )

    if any(
        seed < 0
        for seed in args.seeds
    ):
        raise ValueError(
            "All seeds must be non-negative."
        )

    if args.image_size < 0:
        raise ValueError(
            "'--image-size' must be greater than or equal to 0."
        )

    if args.use_features and args.no_pretrained:
        raise ValueError(
            "'--use-features' requires ImageNet-pretrained feature tensors. "
            "Do not combine it with '--no-pretrained'."
        )

    if args.use_features and args.fine_tune_all:
        raise ValueError(
            "'--use-features' is compatible only with a frozen backbone. "
            "Do not combine it with '--fine-tune-all'."
        )

    if args.use_features and args.image_size != 0:
        raise ValueError(
            "'--image-size' must remain 0 when '--use-features' is enabled. "
            "The preprocessing resolution is fixed during feature extraction."
        )


# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the experiment command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-cardinality instance-selection experiments with "
            "reproducible deep-learning fitness evaluations."
        )
    )

    parser.add_argument(
        "--algorithms",
        nargs="+",
        type=parse_algorithm,
        required=True,
        help=(
            "Algorithms: SRS RS LS GEN MEM, or their full enum names."
        ),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        type=parse_dataset,
        required=True,
        help=(
            "Datasets: MNIST CIFAR10 TINYIMAGENET."
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        type=parse_model,
        required=True,
        help=(
            "Models: alexnet resnext efficientnetv2 swintransformer."
        ),
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        type=parse_metric,
        default=[
            Metric.ACCURACY
        ],
        help=(
            "Validation metrics optimized by the search. Default: accuracy."
        ),
    )

    parser.add_argument(
        "--metric-averaging",
        type=parse_metric_averaging,
        default=MetricAveraging.MACRO,
        help=(
            "Averaging for precision, recall, and F1. Default: macro."
        ),
    )

    parser.add_argument(
        "--keep-percentages",
        nargs="+",
        type=parse_keep_percentage,
        required=True,
        help=(
            "Fixed retention fractions, e.g. 0.10 0.25 0.50 0.75."
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(
            DEFAULT_SEEDS
        ),
        help=(
            "Paired search seeds. Default: 42 123 456 789 1024."
        ),
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help=(
            "Fixed train/validation split seed. Default: 42."
        ),
    )

    parser.add_argument(
        "--training-seed-offset",
        type=int,
        default=100_000,
        help=(
            "Offset added to a search seed for candidate training."
        ),
    )

    parser.add_argument(
        "--final-seed-offset",
        type=int,
        default=200_000,
        help=(
            "Offset added to a search seed for final test retraining."
        ),
    )

    parser.add_argument(
        "--final-repeats",
        type=int,
        default=1,
        help=(
            "Independent final retrainings per selected mask. Default: 1."
        ),
    )

    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=100,
        help=(
            "Exact fitness budget for optimization algorithms. Use 0 for "
            "an SRS-only run because SRS performs no fitness evaluations. "
            "Default: 100."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help=(
            "Exact epochs per candidate evaluation. Default: 10."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=(
            "Batch size. Default: 32."
        ),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help=(
            "Adam learning rate. Default: 0.001."
        ),
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help=(
            "Adam weight decay. Default: 0."
        ),
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of original training data reserved for validation."
        ),
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=10,
        help=(
            "Population size for GEN and MEM. Default: 10."
        ),
    )

    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help=(
            "Tournament size for GEN and MEM. Default: 3."
        ),
    )

    parser.add_argument(
        "--mutation-probability",
        type=float,
        default=0.1,
        help=(
            "Probability of mutating an offspring. Default: 0.1."
        ),
    )

    parser.add_argument(
        "--mutation-fraction",
        type=float,
        default=0.05,
        help=(
            "Fraction of selected instances exchanged when mutation occurs. "
            "Default: 0.05."
        ),
    )

    parser.add_argument(
        "--local-search-num-swaps",
        type=int,
        default=10,
        help=(
            "Selected/unselected pairs exchanged by standalone LS. "
            "One swap changes two mask bits. Default: 10."
        ),
    )

    parser.add_argument(
        "--memetic-local-search-probability",
        type=float,
        default=0.2,
        help=(
            "Probability of refining a MEM offspring. Default: 0.2."
        ),
    )

    parser.add_argument(
        "--memetic-local-search-steps",
        type=int,
        default=10,
        help=(
            "Maximum additional local-neighbor evaluations per refined "
            "offspring. Default: 10."
        ),
    )

    parser.add_argument(
        "--memetic-local-search-num-swaps",
        type=int,
        default=5,
        help=(
            "Selected/unselected pairs exchanged per MEM local neighbor. "
            "One swap changes two mask bits. Default: 5."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help=(
            "Dataset root directory. Default: data."
        ),
    )

    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help=(
            "Root directory containing pre-extracted features. "
            "Default with '--use-features': DATA_DIR/features."
        ),
    )

    parser.add_argument(
        "--use-features",
        action="store_true",
        help=(
            "Train only a fresh linear classifier on cached frozen-backbone "
            "features instead of recomputing the backbone for every candidate."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help=(
            "Root directory for artifacts. Default: results."
        ),
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help=(
            "Experiment directory name. UTC timestamp by default."
        ),
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
        help=(
            "Execution device. Default: auto."
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help=(
            "Explicit square image size. Zero uses official pretrained "
            "transforms or model-specific defaults."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers. Default: 0."
        ),
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help=(
            "Disable ImageNet pretrained weights."
        ),
    )

    parser.add_argument(
        "--fine-tune-all",
        action="store_true",
        help=(
            "Train the complete network instead of only the new head."
        ),
    )

    parser.add_argument(
        "--allow-nondeterminism",
        action="store_true",
        help=(
            "Disable deterministic-algorithm enforcement."
        ),
    )

    parser.add_argument(
        "--skip-test",
        action="store_true",
        help=(
            "Run searches and save masks without evaluating the test split."
        ),
    )

    parser.add_argument(
        "--skip-full-baseline",
        action="store_true",
        help=(
            "Do not evaluate the separate 100%% base-training baseline."
        ),
    )

    parser.add_argument(
        "--empty-cuda-cache",
        action="store_true",
        help=(
            "Empty the CUDA cache after each candidate."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing artifacts with the same run ID."
        ),
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop immediately when one configuration fails."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate and display the experiment matrix without training."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Enable debug logging."
        ),
    )

    return parser


# ==============================================================================
# RUN EXECUTION
# ==============================================================================

def build_search_summary(
    *,
    args: argparse.Namespace,
    experiment_name: str,
    run_id: str,
    algorithm: Algorithm,
    dataset: DatasetName,
    model: ModelName,
    target_metric: Metric,
    keep_percentage: float,
    selected_instances: int,
    total_instances: int,
    search_seed: int,
    training_seed: int,
    evaluations_done: int,
    best_fitness: float | None,
    search_duration_seconds: float,
    device: torch.device,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    """Build one stable search-summary record."""
    return {
        "experiment_name": experiment_name,
        "configuration_fingerprint":
            args.configuration_fingerprint,
        "run_id": run_id,
        "algorithm": algorithm.value,
        "dataset": dataset.value,
        "model": model.value,
        "target_metric": target_metric.value,
        "metric_averaging": args.metric_averaging.value,
        "keep_percentage": keep_percentage,
        "selected_instances": selected_instances,
        "selected_percentage": (
            selected_instances / total_instances
            if total_instances > 0
            else None
        ),
        "total_instances": total_instances,
        "search_seed": search_seed,
        "training_seed": training_seed,
        "split_seed": args.split_seed,
        "max_evaluations": (
            0
            if algorithm == Algorithm.STRATIFIED_RANDOM
            else args.max_evaluations
        ),
        "evaluations_done": evaluations_done,
        "generated_subsets": (
            1
            if algorithm == Algorithm.STRATIFIED_RANDOM
            else evaluations_done
        ),
        "uses_validation_for_selection": (
            algorithm != Algorithm.STRATIFIED_RANDOM
        ),
        "selection_strategy": (
            "single_class_stratified_random_sample"
            if algorithm == Algorithm.STRATIFIED_RANDOM
            else "validation_guided_search"
        ),
        "best_fitness": best_fitness,
        "search_duration_seconds": search_duration_seconds,
        "population_size": args.population_size,
        "tournament_size": args.tournament_size,
        "mutation_probability": args.mutation_probability,
        "mutation_fraction": args.mutation_fraction,
        "local_search_num_swaps": args.local_search_num_swaps,
        "memetic_local_search_probability":
            args.memetic_local_search_probability,
        "memetic_local_search_steps":
            args.memetic_local_search_steps,
        "memetic_local_search_num_swaps":
            args.memetic_local_search_num_swaps,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "validation_fraction": args.validation_fraction,
        "pretrained": not args.no_pretrained,
        "freeze_backbone": not args.fine_tune_all,
        "deterministic_algorithms":
            not args.allow_nondeterminism,
        "device": str(device),
        "status": status,
        "error": error,
    }


def execute_stratified_random_run(
    *,
    args: argparse.Namespace,
    experiment_name: str,
    run_id: str,
    run_dir: Path,
    summary_path: Path,
    mask_path: Path,
    bundle: DataBundle,
    dataset: DatasetName,
    model_name: ModelName,
    target_metric: Metric,
    keep_percentage: float,
    search_seed: int,
    training_seed: int,
    device: torch.device,
) -> SearchRunRecord | None:
    """Generate and save one stratified random mask without fitness search."""
    start_time = time.perf_counter()

    try:
        if len(bundle.base_train_targets) != bundle.total_instances:
            raise ValueError(
                "Base-training targets are not aligned with the mask index "
                "space."
            )

        best_mask, selected_by_class, available_by_class = (
            create_stratified_random_mask(
                targets=bundle.base_train_targets,
                keep_percentage=keep_percentage,
                rng=random.Random(search_seed),
            )
        )

        selection_duration = (
            time.perf_counter()
            - start_time
        )

        selected_instances = sum(
            best_mask.values()
        )

        class_distribution = {
            str(class_label): {
                "available_instances":
                    available_by_class[class_label],
                "selected_instances":
                    selected_by_class[class_label],
                "selected_fraction_within_class": (
                    selected_by_class[class_label]
                    / available_by_class[class_label]
                ),
            }
            for class_label in sorted(available_by_class)
        }

        # The single row records mask generation only. No classifier is
        # trained and no validation metric is used to select this subset.
        write_rows_csv(
            run_dir / "search_history.csv",
            [
                {
                    "Experiment Name": experiment_name,
                    "Run ID": run_id,
                    "Algorithm":
                        Algorithm.STRATIFIED_RANDOM.value,
                    "Dataset": dataset.value,
                    "Model": model_name.value,
                    "Target Metric": target_metric.value,
                    "Search Seed": search_seed,
                    "Training Seed": training_seed,
                    "Split Seed": args.split_seed,
                    "Requested Keep Percentage":
                        keep_percentage,
                    "Selected Instances": selected_instances,
                    "Final Percentage": (
                        selected_instances
                        / bundle.total_instances
                    ),
                    "Candidate Type":
                        "single_stratified_random_sample",
                    "Evaluation": 0,
                    "Fitness": None,
                    "Uses Validation For Selection": False,
                    "Generated Subsets": 1,
                    "Selection Duration Seconds":
                        selection_duration,
                }
            ],
        )

        write_json(
            run_dir / "best_fitness_history.json",
            {
                "best_fitness_history": [],
                "note": (
                    "SRS performs no validation-fitness evaluations."
                ),
            },
        )

        save_mask(
            mask_path,
            best_mask,
        )

        write_json(
            run_dir / "class_distribution.json",
            {
                "total_instances": bundle.total_instances,
                "selected_instances": selected_instances,
                "keep_percentage": keep_percentage,
                "search_seed": search_seed,
                "allocation_method":
                    "proportional_largest_remainder",
                "sampling_method":
                    "uniform_without_replacement_within_class",
                "classes": class_distribution,
            },
        )

        write_json(
            run_dir / "configuration.json",
            {
                "experiment_name": experiment_name,
                "configuration_fingerprint":
                    args.configuration_fingerprint,
                "run_id": run_id,
                "algorithm":
                    Algorithm.STRATIFIED_RANDOM,
                "dataset": dataset,
                "model": model_name,
                "target_metric": target_metric,
                "metric_averaging":
                    args.metric_averaging,
                "keep_percentage": keep_percentage,
                "search_seed": search_seed,
                "training_seed": training_seed,
                "split_seed": args.split_seed,
                "total_instances":
                    bundle.total_instances,
                "num_classes": bundle.num_classes,
                "fitness_evaluations": 0,
                "generated_subsets": 1,
                "uses_validation_for_selection": False,
                "arguments": vars(args),
            },
        )

        summary = build_search_summary(
            args=args,
            experiment_name=experiment_name,
            run_id=run_id,
            algorithm=
                Algorithm.STRATIFIED_RANDOM,
            dataset=dataset,
            model=model_name,
            target_metric=target_metric,
            keep_percentage=keep_percentage,
            selected_instances=selected_instances,
            total_instances=bundle.total_instances,
            search_seed=search_seed,
            training_seed=training_seed,
            evaluations_done=0,
            best_fitness=None,
            search_duration_seconds=
                selection_duration,
            device=device,
            status="completed",
            error=None,
        )

        write_json(
            summary_path,
            summary,
        )

        logger.info(
            "Completed %s | generated_subsets=1 | "
            "fitness_evaluations=0 | selected_instances=%d",
            run_id,
            selected_instances,
        )

        return SearchRunRecord(
            run_id=run_id,
            run_dir=run_dir,
            algorithm=
                Algorithm.STRATIFIED_RANDOM,
            dataset=dataset,
            model=model_name,
            target_metric=target_metric,
            metric_averaging=
                args.metric_averaging,
            keep_percentage=keep_percentage,
            search_seed=search_seed,
            training_seed=training_seed,
            best_mask=best_mask,
            best_fitness=None,
        )

    except Exception as exc:
        selection_duration = (
            time.perf_counter()
            - start_time
        )

        summary = build_search_summary(
            args=args,
            experiment_name=experiment_name,
            run_id=run_id,
            algorithm=
                Algorithm.STRATIFIED_RANDOM,
            dataset=dataset,
            model=model_name,
            target_metric=target_metric,
            keep_percentage=keep_percentage,
            selected_instances=0,
            total_instances=bundle.total_instances,
            search_seed=search_seed,
            training_seed=training_seed,
            evaluations_done=0,
            best_fitness=None,
            search_duration_seconds=
                selection_duration,
            device=device,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )

        write_json(
            summary_path,
            summary,
        )

        logger.exception(
            "Run %s failed",
            run_id,
        )

        if args.fail_fast:
            raise

        return None


def execute_search_run(
    *,
    args: argparse.Namespace,
    experiment_name: str,
    experiment_root: Path,
    bundle: DataBundle,
    dataset: DatasetName,
    model_name: ModelName,
    algorithm: Algorithm,
    target_metric: Metric,
    keep_percentage: float,
    search_seed: int,
    training_seed: int,
    initial_model_state: Mapping[str, torch.Tensor] | None,
    device: torch.device,
) -> SearchRunRecord | None:
    """Execute, save, or resume one search configuration."""
    run_id = build_run_id(
        algorithm=algorithm,
        dataset=dataset,
        model=model_name,
        metric=target_metric,
        keep_percentage=keep_percentage,
        search_seed=search_seed,
    )

    run_dir = (
        experiment_root
        / "runs"
        / dataset.value
        / model_name.value
        / f"seed_{search_seed}"
        / target_metric.value
        / f"keep_{keep_percentage:.4f}"
        / algorithm.value
    )

    summary_path = (
        run_dir
        / "search_summary.json"
    )

    mask_path = (
        run_dir
        / "best_mask.json"
    )

    if (
        summary_path.is_file()
        and mask_path.is_file()
        and not args.overwrite
    ):
        if completed_result_exists(
            path=summary_path,
            expected_fingerprint=
                args.configuration_fingerprint,
        ):
            saved_summary = read_json(
                summary_path
            )

            best_mask = load_mask(
                mask_path,
                expected_size=
                    bundle.total_instances,
            )

            logger.info(
                "Skipping completed run %s",
                run_id,
            )

            return SearchRunRecord(
                run_id=run_id,
                run_dir=run_dir,
                algorithm=algorithm,
                dataset=dataset,
                model=model_name,
                target_metric=target_metric,
                metric_averaging=
                    args.metric_averaging,
                keep_percentage=
                    keep_percentage,
                search_seed=search_seed,
                training_seed=training_seed,
                best_mask=best_mask,
                best_fitness=(
                    None
                    if saved_summary.get(
                        "best_fitness"
                    ) is None
                    else float(
                        saved_summary[
                            "best_fitness"
                        ]
                    )
                ),
            )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Starting run %s",
        run_id,
    )

    if algorithm == Algorithm.STRATIFIED_RANDOM:
        return execute_stratified_random_run(
            args=args,
            experiment_name=experiment_name,
            run_id=run_id,
            run_dir=run_dir,
            summary_path=summary_path,
            mask_path=mask_path,
            bundle=bundle,
            dataset=dataset,
            model_name=model_name,
            target_metric=target_metric,
            keep_percentage=keep_percentage,
            search_seed=search_seed,
            training_seed=training_seed,
            device=device,
        )

    if initial_model_state is None:
        raise ValueError(
            "A candidate-training initial model state is required for "
            f"algorithm '{algorithm.value}'."
        )

    fitness_function = create_fitness_function(
        bundle=bundle,
        model_name=model_name,
        initial_model_state=
            initial_model_state,
        device=device,
        training_seed=training_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        metric_averaging=
            args.metric_averaging,
        freeze_backbone=
            not args.fine_tune_all,
        deterministic_algorithms=
            not args.allow_nondeterminism,
        empty_cuda_cache=
            args.empty_cuda_cache,
    )

    start_time = time.perf_counter()

    try:
        (
            best_mask,
            best_fitness,
            fitness_history,
            best_fitness_history,
            evaluations_done,
        ) = execute_algorithm(
            algorithm=algorithm,
            fitness_function=
                fitness_function,
            total_instances=
                bundle.total_instances,
            keep_percentage=
                keep_percentage,
            max_evaluations=
                args.max_evaluations,
            target_metric=target_metric,
            search_seed=search_seed,
            population_size=
                args.population_size,
            tournament_size=
                args.tournament_size,
            mutation_probability=
                args.mutation_probability,
            mutation_fraction=
                args.mutation_fraction,
            local_search_num_swaps=
                args.local_search_num_swaps,
            memetic_local_search_probability=
                args.memetic_local_search_probability,
            memetic_local_search_steps=
                args.memetic_local_search_steps,
            memetic_local_search_num_swaps=
                args.memetic_local_search_num_swaps,
        )

        search_duration = (
            time.perf_counter()
            - start_time
        )

        selected_instances = sum(
            best_mask.values()
        )

        enriched_history = []

        for entry in fitness_history:
            enriched_entry = dict(entry)
            enriched_entry.update(
                {
                    "Experiment Name":
                        experiment_name,
                    "Run ID": run_id,
                    "Dataset": dataset.value,
                    "Model": model_name.value,
                    "Search Seed": search_seed,
                    "Training Seed":
                        training_seed,
                    "Split Seed":
                        args.split_seed,
                    "Requested Keep Percentage":
                        keep_percentage,
                }
            )
            enriched_history.append(
                enriched_entry
            )

        write_rows_csv(
            run_dir
            / "search_history.csv",
            enriched_history,
        )

        write_json(
            run_dir
            / "best_fitness_history.json",
            {
                "best_fitness_history":
                    best_fitness_history,
            },
        )

        save_mask(
            mask_path,
            best_mask,
        )

        run_configuration = {
            "experiment_name": experiment_name,
            "configuration_fingerprint":
                args.configuration_fingerprint,
            "run_id": run_id,
            "algorithm": algorithm,
            "dataset": dataset,
            "model": model_name,
            "target_metric": target_metric,
            "metric_averaging":
                args.metric_averaging,
            "keep_percentage":
                keep_percentage,
            "search_seed": search_seed,
            "training_seed":
                training_seed,
            "split_seed": args.split_seed,
            "total_instances":
                bundle.total_instances,
            "num_classes":
                bundle.num_classes,
            "arguments": vars(args),
        }

        write_json(
            run_dir
            / "configuration.json",
            run_configuration,
        )

        summary = build_search_summary(
            args=args,
            experiment_name=
                experiment_name,
            run_id=run_id,
            algorithm=algorithm,
            dataset=dataset,
            model=model_name,
            target_metric=target_metric,
            keep_percentage=
                keep_percentage,
            selected_instances=
                selected_instances,
            total_instances=
                bundle.total_instances,
            search_seed=search_seed,
            training_seed=training_seed,
            evaluations_done=
                evaluations_done,
            best_fitness=best_fitness,
            search_duration_seconds=
                search_duration,
            device=device,
            status="completed",
            error=None,
        )

        write_json(
            summary_path,
            summary,
        )

        logger.info(
            "Completed %s | evaluations=%d | best_%s=%.6f",
            run_id,
            evaluations_done,
            target_metric.value,
            best_fitness,
        )

        return SearchRunRecord(
            run_id=run_id,
            run_dir=run_dir,
            algorithm=algorithm,
            dataset=dataset,
            model=model_name,
            target_metric=target_metric,
            metric_averaging=
                args.metric_averaging,
            keep_percentage=
                keep_percentage,
            search_seed=search_seed,
            training_seed=training_seed,
            best_mask=best_mask,
            best_fitness=best_fitness,
        )

    except Exception as exc:
        search_duration = (
            time.perf_counter()
            - start_time
        )

        summary = build_search_summary(
            args=args,
            experiment_name=
                experiment_name,
            run_id=run_id,
            algorithm=algorithm,
            dataset=dataset,
            model=model_name,
            target_metric=target_metric,
            keep_percentage=
                keep_percentage,
            selected_instances=0,
            total_instances=
                bundle.total_instances,
            search_seed=search_seed,
            training_seed=training_seed,
            evaluations_done=0,
            best_fitness=None,
            search_duration_seconds=
                search_duration,
            device=device,
            status="failed",
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )

        write_json(
            summary_path,
            summary,
        )

        logger.exception(
            "Run %s failed",
            run_id,
        )

        if args.fail_fast:
            raise

        return None


def build_test_summary(
    *,
    args: argparse.Namespace,
    experiment_name: str,
    record_type: str,
    run_id: str,
    algorithm: str,
    dataset: DatasetName,
    model_name: ModelName,
    target_metric: str,
    keep_percentage: float,
    selected_instances: int,
    total_instances: int,
    search_seed: int,
    final_repeat: int,
    final_training_seed: int,
    metrics: Mapping[str, Any] | None,
    duration_seconds: float,
    device: torch.device,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    """Build one stable final-test summary record."""
    metrics = (
        metrics
        if metrics is not None
        else {}
    )

    return {
        "experiment_name": experiment_name,
        "configuration_fingerprint":
            args.configuration_fingerprint,
        "record_type": record_type,
        "run_id": run_id,
        "algorithm": algorithm,
        "dataset": dataset.value,
        "model": model_name.value,
        "target_metric": target_metric,
        "metric_averaging":
            args.metric_averaging.value,
        "keep_percentage":
            keep_percentage,
        "selected_instances":
            selected_instances,
        "selected_percentage": (
            selected_instances / total_instances
            if total_instances > 0
            else None
        ),
        "total_instances":
            total_instances,
        "search_seed": search_seed,
        "final_repeat": final_repeat,
        "final_training_seed":
            final_training_seed,
        "split_seed": args.split_seed,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate":
            args.learning_rate,
        "weight_decay": args.weight_decay,
        "pretrained": not args.no_pretrained,
        "freeze_backbone":
            not args.fine_tune_all,
        "device": str(device),
        "accuracy": metrics.get(
            "accuracy"
        ),
        "precision": metrics.get(
            "precision"
        ),
        "recall": metrics.get(
            "recall"
        ),
        "f1": metrics.get(
            "f1"
        ),
        "evaluation_loss": metrics.get(
            "evaluation_loss"
        ),
        "best_validation_loss":
            metrics.get(
                "best_validation_loss"
            ),
        "best_epoch": metrics.get(
            "best_epoch"
        ),
        "epochs_trained": metrics.get(
            "epochs_trained"
        ),
        "evaluation_split": metrics.get(
            "evaluation_split"
        ),
        "test_duration_seconds":
            duration_seconds,
        "status": status,
        "error": error,
    }


def evaluate_record_and_save(
    *,
    args: argparse.Namespace,
    experiment_name: str,
    experiment_root: Path,
    bundle: DataBundle,
    model_name: ModelName,
    record: SearchRunRecord,
    initial_model_state: Mapping[str, torch.Tensor],
    final_repeat: int,
    final_training_seed: int,
    device: torch.device,
) -> None:
    """Evaluate one selected search mask on test and save its record."""
    output_path = (
        record.run_dir
        / f"test_repeat_{final_repeat:02d}.json"
    )

    if (
        not args.overwrite
        and completed_result_exists(
            path=output_path,
            expected_fingerprint=
                args.configuration_fingerprint,
        )
    ):
        logger.info(
            "Skipping completed final test for %s repeat %d",
            record.run_id,
            final_repeat,
        )
        return

    start_time = time.perf_counter()

    try:
        metrics = evaluate_mask_on_test(
            selection_mask=record.best_mask,
            bundle=bundle,
            model_name=model_name,
            initial_model_state=
                initial_model_state,
            device=device,
            final_training_seed=
                final_training_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_epochs=args.epochs,
            metric_averaging=
                args.metric_averaging,
            freeze_backbone=
                not args.fine_tune_all,
            deterministic_algorithms=
                not args.allow_nondeterminism,
            empty_cuda_cache=
                args.empty_cuda_cache,
        )

        duration = (
            time.perf_counter()
            - start_time
        )

        selected_instances = sum(
            record.best_mask.values()
        )

        summary = build_test_summary(
            args=args,
            experiment_name=
                experiment_name,
            record_type="selected_mask",
            run_id=record.run_id,
            algorithm=
                record.algorithm.value,
            dataset=record.dataset,
            model_name=model_name,
            target_metric=
                record.target_metric.value,
            keep_percentage=
                record.keep_percentage,
            selected_instances=
                selected_instances,
            total_instances=
                bundle.total_instances,
            search_seed=
                record.search_seed,
            final_repeat=final_repeat,
            final_training_seed=
                final_training_seed,
            metrics=metrics,
            duration_seconds=duration,
            device=device,
            status="completed",
            error=None,
        )

    except Exception as exc:
        duration = (
            time.perf_counter()
            - start_time
        )

        summary = build_test_summary(
            args=args,
            experiment_name=
                experiment_name,
            record_type="selected_mask",
            run_id=record.run_id,
            algorithm=
                record.algorithm.value,
            dataset=record.dataset,
            model_name=model_name,
            target_metric=
                record.target_metric.value,
            keep_percentage=
                record.keep_percentage,
            selected_instances=sum(
                record.best_mask.values()
            ),
            total_instances=
                bundle.total_instances,
            search_seed=
                record.search_seed,
            final_repeat=final_repeat,
            final_training_seed=
                final_training_seed,
            metrics=None,
            duration_seconds=duration,
            device=device,
            status="failed",
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )

        logger.exception(
            "Final test failed for %s",
            record.run_id,
        )

        if args.fail_fast:
            raise

    write_json(
        output_path,
        summary,
    )


def evaluate_baseline_and_save(
    *,
    args: argparse.Namespace,
    experiment_name: str,
    experiment_root: Path,
    bundle: DataBundle,
    dataset: DatasetName,
    model_name: ModelName,
    search_seed: int,
    initial_model_state: Mapping[str, torch.Tensor],
    final_repeat: int,
    final_training_seed: int,
    device: torch.device,
) -> None:
    """Evaluate the 100% base-training search-space baseline once."""
    baseline_id = (
        f"{dataset.value}_{model_name.value}_"
        f"BASELINE_100_seed{search_seed}_"
        f"repeat{final_repeat}"
    )

    output_path = (
        experiment_root
        / "baselines"
        / dataset.value
        / model_name.value
        / f"seed_{search_seed}"
        / f"repeat_{final_repeat:02d}.json"
    )

    if (
        not args.overwrite
        and completed_result_exists(
            path=output_path,
            expected_fingerprint=
                args.configuration_fingerprint,
        )
    ):
        logger.info(
            "Skipping completed baseline %s",
            baseline_id,
        )
        return

    start_time = time.perf_counter()

    try:
        metrics = evaluate_mask_on_test(
            selection_mask=None,
            bundle=bundle,
            model_name=model_name,
            initial_model_state=
                initial_model_state,
            device=device,
            final_training_seed=
                final_training_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_epochs=args.epochs,
            metric_averaging=
                args.metric_averaging,
            freeze_backbone=
                not args.fine_tune_all,
            deterministic_algorithms=
                not args.allow_nondeterminism,
            empty_cuda_cache=
                args.empty_cuda_cache,
        )

        duration = (
            time.perf_counter()
            - start_time
        )

        summary = build_test_summary(
            args=args,
            experiment_name=
                experiment_name,
            record_type="baseline_100pct",
            run_id=baseline_id,
            algorithm="100%",
            dataset=dataset,
            model_name=model_name,
            target_metric="not_applicable",
            keep_percentage=1.0,
            selected_instances=
                bundle.total_instances,
            total_instances=
                bundle.total_instances,
            search_seed=search_seed,
            final_repeat=final_repeat,
            final_training_seed=
                final_training_seed,
            metrics=metrics,
            duration_seconds=duration,
            device=device,
            status="completed",
            error=None,
        )

    except Exception as exc:
        duration = (
            time.perf_counter()
            - start_time
        )

        summary = build_test_summary(
            args=args,
            experiment_name=
                experiment_name,
            record_type="baseline_100pct",
            run_id=baseline_id,
            algorithm="100%",
            dataset=dataset,
            model_name=model_name,
            target_metric="not_applicable",
            keep_percentage=1.0,
            selected_instances=
                bundle.total_instances,
            total_instances=
                bundle.total_instances,
            search_seed=search_seed,
            final_repeat=final_repeat,
            final_training_seed=
                final_training_seed,
            metrics=None,
            duration_seconds=duration,
            device=device,
            status="failed",
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )

        logger.exception(
            "Baseline %s failed",
            baseline_id,
        )

        if args.fail_fast:
            raise

    write_json(
        output_path,
        summary,
    )


# ==============================================================================
# SUMMARY CONSOLIDATION
# ==============================================================================

def write_fixed_rows_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Write a complete CSV file with a stable predefined schema."""
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: to_serializable(
                        row.get(key)
                    )
                    for key in fieldnames
                }
            )


def consolidate_summaries(
    experiment_root: Path,
) -> None:
    """Rebuild global CSV summaries from per-run JSON artifacts."""
    search_rows = [
        read_json(path)
        for path in sorted(
            experiment_root.glob(
                "runs/**/search_summary.json"
            )
        )
    ]

    test_paths = list(
        experiment_root.glob(
            "runs/**/test_repeat_*.json"
        )
    )
    test_paths.extend(
        experiment_root.glob(
            "baselines/**/*.json"
        )
    )

    test_rows = [
        read_json(path)
        for path in sorted(test_paths)
    ]

    write_fixed_rows_csv(
        experiment_root
        / "search_summary.csv",
        search_rows,
        SEARCH_SUMMARY_FIELDS,
    )

    write_fixed_rows_csv(
        experiment_root
        / "test_summary.csv",
        test_rows,
        TEST_SUMMARY_FIELDS,
    )


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> int:
    """Run the complete configured experiment matrix."""
    parser = build_argument_parser()
    args = parser.parse_args()

    validate_arguments(
        args
    )

    experiment_name = (
        args.experiment_name
        or datetime.now(
            timezone.utc
        ).strftime(
            "experiment_%Y%m%dT%H%M%SZ"
        )
    )

    experiment_root = (
        args.output_dir
        / experiment_name
    )

    configure_logging(
        experiment_root
        / "logs"
        / "run.log",
        verbose=args.verbose,
    )

    device = resolve_device(
        args.device
    )

    methodology_payload = build_methodology_payload(
        args=args,
        device=device,
    )

    configuration_fingerprint = (
        calculate_configuration_fingerprint(
            methodology_payload
        )
    )

    args.configuration_fingerprint = (
        configuration_fingerprint
    )

    manifest_path = (
        experiment_root
        / "manifest.json"
    )

    existing_manifest = validate_existing_experiment(
        manifest_path=manifest_path,
        expected_fingerprint=
            configuration_fingerprint,
    )

    num_search_runs = (
        len(args.datasets)
        * len(args.models)
        * len(args.seeds)
        * len(args.metrics)
        * len(args.keep_percentages)
        * len(args.algorithms)
    )

    fitness_evaluations_per_context = sum(
        0
        if algorithm == Algorithm.STRATIFIED_RANDOM
        else args.max_evaluations
        for algorithm in args.algorithms
    )

    expected_fitness_evaluations = (
        len(args.datasets)
        * len(args.models)
        * len(args.seeds)
        * len(args.metrics)
        * len(args.keep_percentages)
        * fitness_evaluations_per_context
    )

    logger.info(
        "Experiment=%s | device=%s | search_runs=%d | "
        "planned_fitness_evaluations=%d",
        experiment_name,
        device,
        num_search_runs,
        expected_fitness_evaluations,
    )

    if existing_manifest is None:
        manifest = {
            "experiment_name": experiment_name,
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "configuration_fingerprint":
                configuration_fingerprint,
            "methodology_payload":
                methodology_payload,
            "arguments": vars(args),
            "device": device,
            "num_search_runs": num_search_runs,
            "planned_fitness_evaluations":
                expected_fitness_evaluations,
            "versions": build_runtime_versions(),
            "methodological_notes": {
                "fixed_cardinality": True,
                "full_baseline_is_separate": True,
                "early_stopping_in_search": False,
                "test_used_as_fitness": False,
                "candidate_training_seed_fixed_within_run": True,
                "search_rng_is_independent": True,
                "single_stratified_random_sample": {
                    "enabled": (
                        Algorithm.STRATIFIED_RANDOM
                        in args.algorithms
                    ),
                    "generated_subsets_per_seed": 1,
                    "fitness_evaluations": 0,
                    "uses_validation_for_selection": False,
                    "class_allocation":
                        "proportional_largest_remainder",
                    "sampling":
                        "uniform_without_replacement_within_class",
                },
                "data_representation": (
                    "preextracted_features"
                    if args.use_features
                    else "images"
                ),
                "pretrained_feature_extractor":
                    not args.no_pretrained,
                "backbone_frozen":
                    not args.fine_tune_all,
                "only_replacement_classifier_trainable":
                    not args.fine_tune_all,
                "frozen_modules_kept_in_eval_mode":
                    not args.fine_tune_all,
                "baseline_definition": (
                    "100% of the fixed base-training search space; the "
                    "validation partition remains held out for checkpoint "
                    "selection."
                ),
            },
        }
    else:
        manifest = dict(
            existing_manifest
        )
        manifest["last_resumed_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        manifest["arguments"] = vars(args)


    write_json(
        manifest_path,
        manifest,
    )

    if args.dry_run:
        logger.info(
            "Dry run completed. No datasets or models were loaded."
        )
        return 0

    pretrained = not args.no_pretrained
    freeze_backbone = not args.fine_tune_all
    deterministic_algorithms = (
        not args.allow_nondeterminism
    )

    logger.info(
        "Training protocol | pretrained=%s | freeze_backbone=%s | "
        "trainable_scope=%s | representation=%s",
        pretrained,
        freeze_backbone,
        (
            "replacement_classifier_only"
            if freeze_backbone
            else "complete_network"
        ),
        (
            "preextracted_features"
            if args.use_features
            else "images"
        ),
    )

    for dataset in args.datasets:
        for model_name in args.models:
            logger.info(
                "Preparing data | dataset=%s | model=%s",
                dataset.value,
                model_name.value,
            )

            bundle, resolved_image_size = (
                prepare_data_bundle(
                    dataset=dataset,
                    model_name=model_name,
                    data_dir=args.data_dir,
                    features_dir=(
                        args.features_dir
                        if args.features_dir is not None
                        else args.data_dir / "features"
                    ),
                    use_features=args.use_features,
                    batch_size=args.batch_size,
                    validation_fraction=
                        args.validation_fraction,
                    split_seed=args.split_seed,
                    image_size=args.image_size,
                    num_workers=
                        args.num_workers,
                    pretrained=pretrained,
                )
            )

            logger.info(
                "Data ready | representation=%s | base_train=%d | "
                "validation=%d | test=%d | classes=%d | image_size=%d | "
                "feature_dimension=%s",
                bundle.representation,
                bundle.total_instances,
                len(bundle.validation_dataset),
                len(bundle.test_dataset),
                bundle.num_classes,
                resolved_image_size,
                bundle.input_features,
            )

            for search_seed in args.seeds:
                training_seed = (
                    search_seed
                    + args.training_seed_offset
                )

                requires_candidate_training = any(
                    algorithm
                    != Algorithm.STRATIFIED_RANDOM
                    for algorithm in args.algorithms
                )

                initial_model_state = (
                    create_initial_model_state(
                        model_name=model_name,
                        num_classes=
                            bundle.num_classes,
                        training_seed=
                            training_seed,
                        pretrained=pretrained,
                        freeze_backbone=
                            freeze_backbone,
                        deterministic_algorithms=
                            deterministic_algorithms,
                        input_features=
                            bundle.input_features,
                    )
                    if requires_candidate_training
                    else None
                )

                completed_records: list[
                    SearchRunRecord
                ] = []

                for target_metric in args.metrics:
                    for keep_percentage in args.keep_percentages:
                        for algorithm in args.algorithms:
                            record = execute_search_run(
                                args=args,
                                experiment_name=
                                    experiment_name,
                                experiment_root=
                                    experiment_root,
                                bundle=bundle,
                                dataset=dataset,
                                model_name=
                                    model_name,
                                algorithm=algorithm,
                                target_metric=
                                    target_metric,
                                keep_percentage=
                                    keep_percentage,
                                search_seed=
                                    search_seed,
                                training_seed=
                                    training_seed,
                                initial_model_state=
                                    initial_model_state,
                                device=device,
                            )

                            if record is not None:
                                completed_records.append(
                                    record
                                )

                if initial_model_state is not None:
                    del initial_model_state

                gc.collect()

                if args.skip_test:
                    continue

                for final_repeat in range(
                    1,
                    args.final_repeats + 1,
                ):
                    final_training_seed = (
                        search_seed
                        + args.final_seed_offset
                        + final_repeat - 1
                    )

                    final_initial_state = (
                        create_initial_model_state(
                            model_name=
                                model_name,
                            num_classes=
                                bundle.num_classes,
                            training_seed=
                                final_training_seed,
                            pretrained=pretrained,
                            freeze_backbone=
                                freeze_backbone,
                            deterministic_algorithms=
                                deterministic_algorithms,
                            input_features=
                                bundle.input_features,
                        )
                    )

                    if not args.skip_full_baseline:
                        evaluate_baseline_and_save(
                            args=args,
                            experiment_name=
                                experiment_name,
                            experiment_root=
                                experiment_root,
                            bundle=bundle,
                            dataset=dataset,
                            model_name=
                                model_name,
                            search_seed=
                                search_seed,
                            initial_model_state=
                                final_initial_state,
                            final_repeat=
                                final_repeat,
                            final_training_seed=
                                final_training_seed,
                            device=device,
                        )

                    for record in completed_records:
                        evaluate_record_and_save(
                            args=args,
                            experiment_name=
                                experiment_name,
                            experiment_root=
                                experiment_root,
                            bundle=bundle,
                            model_name=
                                model_name,
                            record=record,
                            initial_model_state=
                                final_initial_state,
                            final_repeat=
                                final_repeat,
                            final_training_seed=
                                final_training_seed,
                            device=device,
                        )

                    del final_initial_state
                    gc.collect()

            del bundle
            gc.collect()

    consolidate_summaries(
        experiment_root
    )

    logger.info(
        "Experiment completed successfully: %s",
        experiment_name,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )