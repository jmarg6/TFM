from __future__ import annotations

import logging
import random
from math import isfinite
from typing import Mapping, TypeAlias

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils.config import (
    DatasetSplit,
    MetricAveraging,
)


# ==============================================================================
# TYPE ALIASES
# ==============================================================================

DataLoaders: TypeAlias = Mapping[
    str,
    DataLoader,
]

EvaluationMetrics: TypeAlias = dict[
    str,
    float,
]

TrainingResult: TypeAlias = dict[
    str,
    float | int | str,
]


# ==============================================================================
# LOGGER
# ==============================================================================

logger = logging.getLogger(__name__)


# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def set_training_seed(
    seed: int,
    deterministic_algorithms: bool = True,
) -> None:
    """
    Seed the main random number generators used during model training.

    This function should be called before creating the model so that model
    initialization is controlled by the requested training seed.

    The recommended order is:

        1. Set the training seed.
        2. Create the model.
        3. Move the model to the target device.
        4. Create the optimizer from the model parameters.
        5. Train and evaluate the model.

    Example:

        set_training_seed(training_seed)

        model = create_model(...).to(device)

        optimizer = create_optimizer(
            model.parameters()
        )

        metrics = train_and_evaluate(
            model=model,
            optimizer=optimizer,
            ...
        )

    Args:
        seed:
            Non-negative random seed.

        deterministic_algorithms:
            Whether PyTorch should request deterministic implementations when
            available.

            When enabled, unsupported non-deterministic operations raise an error
            instead of silently reducing reproducibility.

    Raises:
        ValueError:
            If the seed is negative.
    """
    if seed < 0:
        raise ValueError(
            f"'seed' must be greater than or equal to 0, "
            f"but received {seed}."
        )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = (
        deterministic_algorithms
    )

    torch.use_deterministic_algorithms(
        deterministic_algorithms,
        warn_only=False,
    )


# ==============================================================================
# CONFIGURATION NORMALIZATION
# ==============================================================================

def normalize_metric_averaging(
    metric_averaging: MetricAveraging | str,
) -> MetricAveraging:
    """
    Normalize a metric-averaging strategy.

    Args:
        metric_averaging:
            Metric-averaging enum member or string value.

    Returns:
        Validated `MetricAveraging` member.

    Raises:
        ValueError:
            If the averaging strategy is unsupported.
    """
    if isinstance(
        metric_averaging,
        MetricAveraging,
    ):
        return metric_averaging

    normalized_value = (
        str(metric_averaging)
        .strip()
        .lower()
    )

    try:
        return MetricAveraging(
            normalized_value
        )

    except ValueError as exc:
        allowed = [
            averaging.value
            for averaging in MetricAveraging
        ]

        raise ValueError(
            f"Unsupported metric averaging strategy "
            f"'{metric_averaging}'. "
            f"Allowed values: {allowed}."
        ) from exc


def normalize_evaluation_split(
    eval_split: DatasetSplit | str,
) -> DatasetSplit:
    """
    Normalize and validate the dataset split used for final evaluation.

    Only validation and test are accepted as target evaluation partitions.
    Training metrics are intentionally not exposed as final performance
    estimates.

    Args:
        eval_split:
            Validation or test split identifier.

    Returns:
        Validated `DatasetSplit`.

    Raises:
        ValueError:
            If the split is unsupported or refers to the training partition.
    """
    if isinstance(
        eval_split,
        DatasetSplit,
    ):
        normalized_split = (
            eval_split
        )

    else:
        normalized_value = (
            str(eval_split)
            .strip()
            .lower()
        )

        try:
            normalized_split = DatasetSplit(
                normalized_value
            )

        except ValueError as exc:
            allowed = [
                DatasetSplit.VALIDATION.value,
                DatasetSplit.TEST.value,
            ]

            raise ValueError(
                f"Unsupported evaluation split '{eval_split}'. "
                f"Allowed values: {allowed}."
            ) from exc

    if normalized_split == DatasetSplit.TRAIN:
        raise ValueError(
            "The training partition cannot be used as the final evaluation "
            "split. Use validation during instance-selection search and test "
            "only for the final held-out evaluation."
        )

    return normalized_split


# ==============================================================================
# VALIDATION HELPERS
# ==============================================================================

def validate_data_loader(
    loader: DataLoader,
    loader_name: str,
) -> None:
    """
    Validate that a DataLoader contains at least one sample and one batch.

    Args:
        loader:
            DataLoader to validate.

        loader_name:
            Human-readable loader identifier.

    Raises:
        ValueError:
            If the DataLoader is empty.
    """
    try:
        dataset_size = len(
            loader.dataset
        )

    except TypeError as exc:
        raise ValueError(
            f"Unable to determine the dataset size for loader "
            f"'{loader_name}'."
        ) from exc

    if dataset_size <= 0:
        raise ValueError(
            f"DataLoader '{loader_name}' contains no samples."
        )

    try:
        num_batches = len(
            loader
        )

    except TypeError as exc:
        raise ValueError(
            f"Unable to determine the number of batches for loader "
            f"'{loader_name}'."
        ) from exc

    if num_batches <= 0:
        raise ValueError(
            f"DataLoader '{loader_name}' contains no batches."
        )


def validate_optimizer_for_model(
    model: nn.Module,
    optimizer: Optimizer,
) -> None:
    """
    Validate that all optimizer parameters belong to the provided model.

    The optimizer may intentionally optimize only a subset of model
    parameters, for example when training a linear probe or partially frozen
    architecture.

    However, it must not contain parameters belonging to a different model
    instance.

    The recommended protocol is to move the model to the target device before
    creating the optimizer.

    Args:
        model:
            Model expected to be optimized.

        optimizer:
            Optimizer associated with the model.

    Raises:
        ValueError:
            If the optimizer has no parameters or contains parameters that do
            not belong to the model.
    """
    model_parameter_ids = {
        id(parameter)
        for parameter in model.parameters()
    }

    optimizer_parameters = [
        parameter
        for parameter_group
        in optimizer.param_groups
        for parameter
        in parameter_group["params"]
    ]

    if not optimizer_parameters:
        raise ValueError(
            "The optimizer does not contain any parameters."
        )

    invalid_parameters = [
        parameter
        for parameter in optimizer_parameters
        if id(parameter)
        not in model_parameter_ids
    ]

    if invalid_parameters:
        raise ValueError(
            "The optimizer contains parameters that do not belong to the "
            "provided model. A fresh optimizer must be created from the "
            "parameters of the fresh model instance after the model has been "
            "moved to the target device."
        )


def validate_loss_value(
    loss_value: float,
    loss_name: str,
) -> None:
    """
    Validate that a loss value is numerically finite.

    Args:
        loss_value:
            Loss value to validate.

        loss_name:
            Human-readable loss identifier.

    Raises:
        ValueError:
            If the loss is NaN or infinite.
    """
    if not isfinite(
        loss_value
    ):
        raise ValueError(
            f"'{loss_name}' is not finite: {loss_value}."
        )


# ==============================================================================
# MODEL STATE MANAGEMENT
# ==============================================================================

def clone_model_state_to_cpu(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """
    Create an independent CPU copy of a model state dictionary.

    Every tensor is explicitly detached, moved to CPU, and cloned.

    The explicit clone is important because calling `.cpu()` on a tensor that
    is already stored on CPU may return a tensor sharing the same underlying
    storage. Subsequent training could then modify the supposedly saved
    checkpoint.

    Args:
        model:
            Model whose state must be copied.

    Returns:
        Independent CPU copy of the complete model state.
    """
    return {
        name: (
            tensor
            .detach()
            .cpu()
            .clone()
        )
        for name, tensor
        in model.state_dict().items()
    }


# ==============================================================================
# LOSS ACCUMULATION
# ==============================================================================

def accumulate_batch_loss(
    current_total: float,
    loss: torch.Tensor,
    batch_size: int,
    criterion: nn.Module,
) -> float:
    """
    Add one batch loss to a sample-weighted running total.

    Standard PyTorch classification losses commonly use `reduction="mean"`.
    In that case, the batch loss is multiplied by the batch size so that the
    final epoch loss is averaged over samples rather than batches.

    `reduction="sum"` is also supported.

    Args:
        current_total:
            Current accumulated loss.

        loss:
            Scalar loss tensor returned by the criterion.

        batch_size:
            Number of samples in the current batch.

        criterion:
            Loss function used to compute the batch loss.

    Returns:
        Updated accumulated loss.

    Raises:
        ValueError:
            If the loss tensor is not scalar or the criterion uses
            `reduction="none"`.
    """
    if loss.ndim != 0:
        raise ValueError(
            "The training criterion must return a scalar loss. "
            f"Received a tensor with shape {tuple(loss.shape)}."
        )

    reduction = getattr(
        criterion,
        "reduction",
        "mean",
    )

    batch_loss = float(
        loss.detach().item()
    )

    validate_loss_value(
        loss_value=batch_loss,
        loss_name="batch_loss",
    )

    if reduction == "none":
        raise ValueError(
            "Loss functions using reduction='none' are not supported by this "
            "training utility."
        )

    if reduction == "sum":
        return (
            current_total
            + batch_loss
        )

    return (
        current_total
        + batch_loss * batch_size
    )


# ==============================================================================
# LOSS-ONLY EVALUATION
# ==============================================================================

def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate the sample-averaged loss of a model.

    This function is used during training to select the best validation
    checkpoint without calculating the complete set of classification metrics
    after every epoch.

    Args:
        model:
            Model to evaluate.

        loader:
            DataLoader containing evaluation samples.

        criterion:
            Loss function.

        device:
            Device on which the model is evaluated.

    Returns:
        Sample-averaged loss.

    Raises:
        ValueError:
            If the loader is empty or a non-finite loss is produced.
    """
    validate_data_loader(
        loader=loader,
        loader_name="evaluation",
    )

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():

        for inputs, labels in loader:

            inputs = inputs.to(
                device,
                non_blocking=True,
            )

            labels = (
                labels
                .to(
                    device,
                    non_blocking=True,
                )
                .long()
                .reshape(-1)
            )

            outputs = model(
                inputs
            )

            loss = criterion(
                outputs,
                labels,
            )

            batch_size = int(
                labels.shape[0]
            )

            total_loss = accumulate_batch_loss(
                current_total=total_loss,
                loss=loss,
                batch_size=batch_size,
                criterion=criterion,
            )

            total_samples += (
                batch_size
            )

    if total_samples <= 0:
        raise ValueError(
            "The evaluation DataLoader produced no samples."
        )

    average_loss = (
        total_loss
        / total_samples
    )

    validate_loss_value(
        loss_value=average_loss,
        loss_name="average_evaluation_loss",
    )

    return average_loss


# ==============================================================================
# COMPLETE MODEL EVALUATION
# ==============================================================================

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metric_averaging: MetricAveraging | str = MetricAveraging.MACRO,
) -> EvaluationMetrics:
    """
    Evaluate a trained classification model.

    The function calculates:

    - Sample-averaged loss.
    - Accuracy.
    - Precision.
    - Recall.
    - F1 score.

    Args:
        model:
            Trained classification model.

        loader:
            DataLoader containing the evaluation partition.

        criterion:
            Classification loss function.

        device:
            Device on which the model is evaluated.

        metric_averaging:
            Averaging strategy used for precision, recall, and F1.

    Returns:
        Dictionary containing the evaluation metrics.

    Raises:
        ValueError:
            If the loader is empty or the model produces invalid outputs.
    """
    validate_data_loader(
        loader=loader,
        loader_name="evaluation",
    )

    averaging = normalize_metric_averaging(
        metric_averaging
    )

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.inference_mode():

        for inputs, labels in loader:

            inputs = inputs.to(
                device,
                non_blocking=True,
            )

            labels = (
                labels
                .to(
                    device,
                    non_blocking=True,
                )
                .long()
                .reshape(-1)
            )

            outputs = model(
                inputs
            )

            if outputs.ndim != 2:
                raise ValueError(
                    "The model must return a two-dimensional classification "
                    "tensor with shape [batch_size, num_classes]. "
                    f"Received shape {tuple(outputs.shape)}."
                )

            if outputs.shape[0] != labels.shape[0]:
                raise ValueError(
                    "The model output batch size does not match the label "
                    f"batch size. Outputs: {outputs.shape[0]}, "
                    f"labels: {labels.shape[0]}."
                )

            loss = criterion(
                outputs,
                labels,
            )

            batch_size = int(
                labels.shape[0]
            )

            total_loss = accumulate_batch_loss(
                current_total=total_loss,
                loss=loss,
                batch_size=batch_size,
                criterion=criterion,
            )

            total_samples += (
                batch_size
            )

            predictions = (
                outputs
                .argmax(dim=1)
            )

            all_predictions.extend(
                predictions
                .detach()
                .cpu()
                .tolist()
            )

            all_labels.extend(
                labels
                .detach()
                .cpu()
                .tolist()
            )

    if total_samples <= 0:
        raise ValueError(
            "The evaluation DataLoader produced no samples."
        )

    average_loss = (
        total_loss
        / total_samples
    )

    validate_loss_value(
        loss_value=average_loss,
        loss_name="average_evaluation_loss",
    )

    return {
        "loss":
            float(
                average_loss
            ),

        "accuracy":
            float(
                accuracy_score(
                    all_labels,
                    all_predictions,
                )
            ),

        "precision":
            float(
                precision_score(
                    all_labels,
                    all_predictions,
                    average=averaging.value,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    all_labels,
                    all_predictions,
                    average=averaging.value,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    all_labels,
                    all_predictions,
                    average=averaging.value,
                    zero_division=0,
                )
            ),
    }


# ==============================================================================
# TRAINING AND EVALUATION
# ==============================================================================

def train_and_evaluate(
    model: nn.Module,
    loaders: DataLoaders,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int = 10,
    eval_split: DatasetSplit | str = DatasetSplit.VALIDATION,
    metric_averaging: MetricAveraging | str = MetricAveraging.MACRO,
) -> TrainingResult:
    """
    Train a classification model for a fixed number of epochs and evaluate it.

    The model is always trained for exactly `num_epochs`. No early stopping is
    used, ensuring that every fitness evaluation receives the same training
    epoch budget.

    After every epoch, validation loss is calculated. The model state with the
    lowest validation loss is stored in memory.

    Once all epochs have been completed:

        1. The best validation checkpoint is restored.
        2. The model is evaluated on the requested target split.
        3. Classification metrics and training metadata are returned.

    Methodological requirements:

        - During metaheuristic search, `eval_split` must be `valid`.
        - The test split must never be used by the fitness function.
        - A fresh model and a fresh optimizer must be created for every
          independent fitness evaluation.
        - The model should be moved to the target device before creating the
          optimizer.
        - For controlled candidate comparisons, the training seed should be
          fixed before creating each fresh model and optimizer.

    Example search protocol:

        set_training_seed(training_seed)

        model = create_model(
            ...
        ).to(
            device
        )

        optimizer = create_optimizer(
            model.parameters()
        )

        metrics = train_and_evaluate(
            model=model,
            loaders=loaders,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=10,
            eval_split=DatasetSplit.VALIDATION,
        )

    Args:
        model:
            Fresh model instance for the current training run.

        loaders:
            Mapping containing at least `train`, `valid`, and the requested
            evaluation split.

        criterion:
            Scalar classification loss function.

        optimizer:
            Fresh optimizer associated with the provided model.

        device:
            Device used for training and evaluation.

        num_epochs:
            Exact number of training epochs.

        eval_split:
            Partition used for the final metric calculation. Use validation
            during metaheuristic search and test only for final held-out
            evaluation.

        metric_averaging:
            Averaging strategy used for precision, recall, and F1.

    Returns:
        Dictionary containing:

        - accuracy
        - precision
        - recall
        - f1
        - evaluation_loss
        - best_validation_loss
        - best_epoch
        - epochs_trained
        - evaluation_split

    Raises:
        ValueError:
            If the training configuration, loaders, optimizer, or numerical
            losses are invalid.
    """
    if num_epochs <= 0:
        raise ValueError(
            f"'num_epochs' must be greater than 0, "
            f"but received {num_epochs}."
        )

    evaluation_split = normalize_evaluation_split(
        eval_split
    )

    averaging = normalize_metric_averaging(
        metric_averaging
    )

    train_key = (
        DatasetSplit.TRAIN.value
    )

    validation_key = (
        DatasetSplit.VALIDATION.value
    )

    evaluation_key = (
        evaluation_split.value
    )

    required_loader_keys = {
        train_key,
        validation_key,
        evaluation_key,
    }

    missing_loader_keys = (
        required_loader_keys
        - set(loaders.keys())
    )

    if missing_loader_keys:
        raise KeyError(
            "Missing required DataLoader entries: "
            f"{sorted(missing_loader_keys)}. "
            f"Available entries: {sorted(loaders.keys())}."
        )

    train_loader = loaders[
        train_key
    ]

    validation_loader = loaders[
        validation_key
    ]

    evaluation_loader = loaders[
        evaluation_key
    ]

    validate_data_loader(
        loader=train_loader,
        loader_name=train_key,
    )

    validate_data_loader(
        loader=validation_loader,
        loader_name=validation_key,
    )

    validate_data_loader(
        loader=evaluation_loader,
        loader_name=evaluation_key,
    )

    # --------------------------------------------------------------------------
    # Device and optimizer consistency
    # --------------------------------------------------------------------------

    # The model is moved to the target device before validating its relationship
    # with the optimizer. The recommended external protocol is still to move
    # the model to the device before creating the optimizer.
    model = model.to(
        device
    )

    validate_optimizer_for_model(
        model=model,
        optimizer=optimizer,
    )

    best_validation_loss = float(
        "inf"
    )

    best_epoch = 0

    best_model_state: dict[
        str,
        torch.Tensor,
    ] | None = None

    # --------------------------------------------------------------------------
    # Fixed-budget training loop
    # --------------------------------------------------------------------------

    for epoch in range(
        1,
        num_epochs + 1,
    ):

        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(
                device,
                non_blocking=True,
            )

            labels = (
                labels
                .to(
                    device,
                    non_blocking=True,
                )
                .long()
                .reshape(-1)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                inputs
            )

            if outputs.ndim != 2:
                raise ValueError(
                    "The model must return a two-dimensional classification "
                    "tensor with shape [batch_size, num_classes]. "
                    f"Received shape {tuple(outputs.shape)}."
                )

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            batch_size = int(
                labels.shape[0]
            )

            total_train_loss = accumulate_batch_loss(
                current_total=
                    total_train_loss,
                loss=loss,
                batch_size=batch_size,
                criterion=criterion,
            )

            total_train_samples += (
                batch_size
            )

        if total_train_samples <= 0:
            raise ValueError(
                "The training DataLoader produced no samples."
            )

        average_train_loss = (
            total_train_loss
            / total_train_samples
        )

        validate_loss_value(
            loss_value=average_train_loss,
            loss_name="average_train_loss",
        )

        # ----------------------------------------------------------------------
        # Validation checkpoint selection
        # ----------------------------------------------------------------------

        validation_loss = evaluate_loss(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        if (
            validation_loss
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_loss
            )

            best_epoch = (
                epoch
            )

            best_model_state = (
                clone_model_state_to_cpu(
                    model
                )
            )

        logger.debug(
            "Epoch %d/%d | train_loss=%.6f | validation_loss=%.6f | "
            "best_epoch=%d | best_validation_loss=%.6f",
            epoch,
            num_epochs,
            average_train_loss,
            validation_loss,
            best_epoch,
            best_validation_loss,
        )

    # --------------------------------------------------------------------------
    # Restore the best validation checkpoint
    # --------------------------------------------------------------------------

    if best_model_state is None:
        raise RuntimeError(
            "No valid model checkpoint was created during training."
        )

    model.load_state_dict(
        best_model_state
    )

    model.to(
        device
    )

    # --------------------------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------------------------

    evaluation_metrics = evaluate_model(
        model=model,
        loader=evaluation_loader,
        criterion=criterion,
        device=device,
        metric_averaging=averaging,
    )

    result: TrainingResult = {
        "accuracy":
            evaluation_metrics[
                "accuracy"
            ],

        "precision":
            evaluation_metrics[
                "precision"
            ],

        "recall":
            evaluation_metrics[
                "recall"
            ],

        "f1":
            evaluation_metrics[
                "f1"
            ],

        "evaluation_loss":
            evaluation_metrics[
                "loss"
            ],

        "best_validation_loss":
            float(
                best_validation_loss
            ),

        "best_epoch":
            best_epoch,

        "epochs_trained":
            num_epochs,

        "evaluation_split":
            evaluation_split.value,
    }

    logger.debug(
        "Training completed | evaluation_split=%s | best_epoch=%d | "
        "best_validation_loss=%.6f | accuracy=%.6f | f1=%.6f",
        evaluation_split.value,
        best_epoch,
        best_validation_loss,
        result["accuracy"],
        result["f1"],
    )

    return result