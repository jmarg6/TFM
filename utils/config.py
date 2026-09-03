from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any


# ==============================================================================
# BASE ENUM
# ==============================================================================

class StringEnum(str, Enum):
    """
    Base class for string-valued enumerations.

    Inheriting from `str` makes enum values easier to serialize to JSON, store
    in CSV files, use in command-line interfaces, and incorporate into paths.
    """

    def __str__(self) -> str:
        """Return the underlying string value."""
        return self.value


# ==============================================================================
# PROJECT ENUMS
# ==============================================================================

class Algorithm(StringEnum):
    """Supported instance-selection algorithms."""

    RANDOM_SEARCH = "RS"
    STRATIFIED_RANDOM = "SRS"
    K_CENTER_GREEDY = "KCG"
    LOCAL_SEARCH = "LS"
    GENETIC = "GEN"
    MEMETIC = "MEM"


class Metric(StringEnum):
    """Supported predictive performance metrics."""

    ACCURACY = "accuracy"
    F1_SCORE = "f1"
    PRECISION = "precision"
    RECALL = "recall"


class MetricAveraging(StringEnum):
    """
    Supported averaging strategies for multiclass precision, recall, and F1.

    The primary experimental protocol should use MACRO averaging so that every
    class contributes equally to the final metric.
    """

    MACRO = "macro"
    WEIGHTED = "weighted"
    MICRO = "micro"


class ModelName(StringEnum):
    """Supported deep-learning architectures."""

    ALEXNET = "alexnet"
    RESNEXT = "resnext"
    EFFICIENTNETV2 = "efficientnetv2"
    SWIN_TRANSFORMER = "swintransformer"


class DatasetName(StringEnum):
    """Supported image-classification datasets."""

    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    TINYIMAGENET = "TINYIMAGENET"


class DatasetSplit(StringEnum):
    """Standard dataset partitions used by the experimental protocol."""

    TRAIN = "train"
    VALIDATION = "valid"
    TEST = "test"


# ==============================================================================
# PROJECT CONFIGURATION
# ==============================================================================

@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """
    Immutable configuration describing one experimental context.

    The configuration contains stable information such as the dataset, model,
    target metric, metric-averaging strategy, task identifier, and project
    root directory.

    The object is immutable by design. A new configuration should be created
    for each experimental context instead of modifying shared global state.

    This prevents configuration values from accidentally leaking between
    different datasets, models, algorithms, or experimental runs.

    Attributes:
        dataset_name:
            Dataset used in the experiment.

        model_name:
            Deep-learning architecture used in the experiment.

        target_metric:
            Metric optimized by the instance-selection algorithm.

        metric_averaging:
            Averaging strategy used for multiclass precision, recall, and F1.

        task_id:
            Human-readable identifier for the experimental task.

        project_root:
            Root directory of the project.
    """

    dataset_name: DatasetName | str = DatasetName.MNIST
    model_name: ModelName | str = ModelName.ALEXNET
    target_metric: Metric | str = Metric.ACCURACY
    metric_averaging: MetricAveraging | str = MetricAveraging.MACRO
    task_id: str = "default_task"
    project_root: Path | str = Path(".")

    def __post_init__(self) -> None:
        """
        Normalize and validate all configuration values.

        String values are converted into their corresponding enum members so
        that the rest of the project always works with validated values.

        Raises:
            ValueError:
                If an unsupported dataset, model, metric, averaging strategy,
                or invalid task identifier is provided.
        """

        # ----------------------------------------------------------------------
        # Normalize dataset
        # ----------------------------------------------------------------------

        if isinstance(self.dataset_name, DatasetName):
            dataset_name = self.dataset_name
        else:
            normalized_dataset = str(
                self.dataset_name
            ).strip().upper()

            try:
                dataset_name = DatasetName(
                    normalized_dataset
                )
            except ValueError as exc:
                allowed = [
                    dataset.value
                    for dataset in DatasetName
                ]

                raise ValueError(
                    f"Unsupported dataset '{self.dataset_name}'. "
                    f"Allowed values: {allowed}."
                ) from exc

        # ----------------------------------------------------------------------
        # Normalize model
        # ----------------------------------------------------------------------

        if isinstance(self.model_name, ModelName):
            model_name = self.model_name
        else:
            normalized_model = str(
                self.model_name
            ).strip().lower()

            try:
                model_name = ModelName(
                    normalized_model
                )
            except ValueError as exc:
                allowed = [
                    model.value
                    for model in ModelName
                ]

                raise ValueError(
                    f"Unsupported model '{self.model_name}'. "
                    f"Allowed values: {allowed}."
                ) from exc

        # ----------------------------------------------------------------------
        # Normalize target metric
        # ----------------------------------------------------------------------

        if isinstance(self.target_metric, Metric):
            target_metric = self.target_metric
        else:
            normalized_metric = str(
                self.target_metric
            ).strip().lower()

            try:
                target_metric = Metric(
                    normalized_metric
                )
            except ValueError as exc:
                allowed = [
                    metric.value
                    for metric in Metric
                ]

                raise ValueError(
                    f"Unsupported target metric '{self.target_metric}'. "
                    f"Allowed values: {allowed}."
                ) from exc

        # ----------------------------------------------------------------------
        # Normalize metric averaging
        # ----------------------------------------------------------------------

        if isinstance(
            self.metric_averaging,
            MetricAveraging,
        ):
            metric_averaging = (
                self.metric_averaging
            )
        else:
            normalized_averaging = str(
                self.metric_averaging
            ).strip().lower()

            try:
                metric_averaging = (
                    MetricAveraging(
                        normalized_averaging
                    )
                )
            except ValueError as exc:
                allowed = [
                    averaging.value
                    for averaging in MetricAveraging
                ]

                raise ValueError(
                    f"Unsupported metric averaging strategy "
                    f"'{self.metric_averaging}'. "
                    f"Allowed values: {allowed}."
                ) from exc

        # ----------------------------------------------------------------------
        # Validate task identifier
        # ----------------------------------------------------------------------

        normalized_task_id = (
            self.task_id.strip()
        )

        if not normalized_task_id:
            raise ValueError(
                "'task_id' must be a non-empty string."
            )

        if (
            "/" in normalized_task_id
            or "\\" in normalized_task_id
        ):
            raise ValueError(
                "'task_id' must not contain path separators."
            )

        # ----------------------------------------------------------------------
        # Normalize project root
        # ----------------------------------------------------------------------

        project_root = Path(
            self.project_root
        ).expanduser()

        # Since the dataclass is frozen, normalized values must be assigned
        # explicitly through object.__setattr__.
        object.__setattr__(
            self,
            "dataset_name",
            dataset_name,
        )

        object.__setattr__(
            self,
            "model_name",
            model_name,
        )

        object.__setattr__(
            self,
            "target_metric",
            target_metric,
        )

        object.__setattr__(
            self,
            "metric_averaging",
            metric_averaging,
        )

        object.__setattr__(
            self,
            "task_id",
            normalized_task_id,
        )

        object.__setattr__(
            self,
            "project_root",
            project_root,
        )

    # ==========================================================================
    # PROJECT PATHS
    # ==========================================================================

    @property
    def data_root(self) -> Path:
        """
        Return the root directory containing all datasets.

        This path should be passed as the base data directory to dataset
        loaders.
        """
        return (
            self.project_root
            / "data"
        )

    @property
    def features_root(self) -> Path:
        """Return the root directory containing extracted features."""
        return (
            self.data_root
            / "features"
        )

    @property
    def features_path(self) -> Path:
        """
        Return the feature directory for the current dataset and model.

        Example:
            data/features/MNIST/alexnet/
        """
        return (
            self.features_root
            / self.dataset_name.value
            / self.model_name.value
        )

    @property
    def results_root(self) -> Path:
        """Return the root directory containing experiment results."""
        return (
            self.project_root
            / "results"
        )

    @property
    def results_path(self) -> Path:
        """
        Return the results directory for the current experimental context.

        Example:
            results/MNIST/alexnet/default_task/
        """
        return (
            self.results_root
            / self.dataset_name.value
            / self.model_name.value
            / self.task_id
        )

    @property
    def checkpoints_root(self) -> Path:
        """Return the root directory containing model checkpoints."""
        return (
            self.project_root
            / "checkpoints"
        )

    @property
    def checkpoints_path(self) -> Path:
        """
        Return the checkpoint directory for the current experimental context.
        """
        return (
            self.checkpoints_root
            / self.dataset_name.value
            / self.model_name.value
            / self.task_id
        )

    @property
    def logs_root(self) -> Path:
        """Return the root directory containing execution logs."""
        return (
            self.project_root
            / "logs"
        )

    @property
    def logs_path(self) -> Path:
        """
        Return the log directory for the current experimental context.
        """
        return (
            self.logs_root
            / self.dataset_name.value
            / self.model_name.value
            / self.task_id
        )

    # ==========================================================================
    # CONFIGURATION UTILITIES
    # ==========================================================================

    def with_updates(
        self,
        **changes: Any,
    ) -> ProjectConfig:
        """
        Create a new configuration with selected values replaced.

        The current configuration remains unchanged.

        Example:
            cifar_config = base_config.with_updates(
                dataset_name=DatasetName.CIFAR10,
            )

        Args:
            **changes:
                Configuration fields to replace.

        Returns:
            A new validated `ProjectConfig` instance.
        """
        return replace(
            self,
            **changes,
        )

    def create_output_directories(
        self,
    ) -> None:
        """
        Create directories used to store generated experiment artifacts.

        Dataset directories are intentionally not created here because raw
        dataset management belongs to the data-loading layer.
        """
        directories = (
            self.features_path,
            self.results_path,
            self.checkpoints_path,
            self.logs_path,
        )

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    def to_dict(
        self,
    ) -> dict[str, str]:
        """
        Convert the configuration into a serialization-friendly dictionary.

        This representation can be stored together with experiment results to
        preserve the exact experimental context.

        Returns:
            Dictionary containing normalized configuration values and paths.
        """
        return {
            "dataset_name":
                self.dataset_name.value,
            "model_name":
                self.model_name.value,
            "target_metric":
                self.target_metric.value,
            "metric_averaging":
                self.metric_averaging.value,
            "task_id":
                self.task_id,
            "project_root":
                str(self.project_root),
            "data_root":
                str(self.data_root),
            "features_path":
                str(self.features_path),
            "results_path":
                str(self.results_path),
            "checkpoints_path":
                str(self.checkpoints_path),
            "logs_path":
                str(self.logs_path),
        }