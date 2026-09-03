#!/usr/bin/env python3
"""
Extract deterministic frozen-backbone features for the TFM experiments.

The generated tensors preserve the original sample ordering of each dataset.
Consequently, `utils.data_utils.get_feature_dataloaders()` can recreate the
same stratified train/validation split and the same metaheuristic mask index
space used by the raw-image pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, MNIST, ImageFolder

from run_experiments import (
    build_preprocessing_transform,
    parse_dataset,
    parse_model,
    resolve_device,
)
from utils.config import DatasetName, ModelName
from utils.data_utils import (
    TinyImageNetValidationDataset,
    create_data_loader,
)


def build_feature_extractor(
    model_name: ModelName,
) -> nn.Module:
    """
    Build an ImageNet-pretrained model whose final classifier is an identity.

    The output is therefore exactly the vector received by the replacement
    linear classifier in the head-only transfer-learning pipeline.
    """
    if model_name == ModelName.ALEXNET:
        model = models.alexnet(
            weights=models.AlexNet_Weights.DEFAULT
        )
        model.classifier[6] = nn.Identity()

    elif model_name == ModelName.RESNEXT:
        model = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.DEFAULT
        )
        model.fc = nn.Identity()

    elif model_name == ModelName.EFFICIENTNETV2:
        model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT
        )
        model.classifier[1] = nn.Identity()

    elif model_name == ModelName.SWIN_TRANSFORMER:
        model = models.swin_t(
            weights=models.Swin_T_Weights.DEFAULT
        )
        model.head = nn.Identity()

    else:
        raise ValueError(
            f"Unsupported model '{model_name}'."
        )

    for parameter in model.parameters():
        parameter.requires_grad = False

    # Evaluation mode is essential because frozen Dropout layers must be
    # disabled and frozen BatchNorm statistics must remain unchanged.
    model.eval()

    return model


def load_full_datasets(
    *,
    dataset_name: DatasetName,
    data_dir: Path,
    transform: Callable,
) -> tuple[Dataset, Dataset, list[str]]:
    """
    Load the complete original training and held-out test datasets.

    No train/validation split is created here. The split is recreated later by
    `get_feature_dataloaders()` from the saved training labels, ensuring that
    image and feature modes use exactly the same positional mask space.
    """
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if dataset_name == DatasetName.MNIST:
        train_dataset = MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )
        classes = [
            str(class_index)
            for class_index in range(10)
        ]

    elif dataset_name == DatasetName.CIFAR10:
        train_dataset = CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )
        classes = list(
            train_dataset.classes
        )

    elif dataset_name == DatasetName.TINYIMAGENET:
        tiny_root = (
            data_dir
            / "tiny-imagenet-200"
        )
        train_path = (
            tiny_root
            / "train"
        )
        validation_path = (
            tiny_root
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

        train_dataset = ImageFolder(
            root=train_path,
            transform=transform,
        )

        test_dataset = TinyImageNetValidationDataset(
            root=validation_path,
            class_to_idx=train_dataset.class_to_idx,
            transform=transform,
        )

        classes = list(
            train_dataset.classes
        )

    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'."
        )

    return (
        train_dataset,
        test_dataset,
        classes,
    )


def extract_split(
    *,
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    split_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract one feature vector per sample while preserving dataset order.

    CPU tensors are preallocated after the first batch. This avoids holding a
    second full copy in memory during concatenation.
    """
    loader = create_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        seed=seed,
    )

    total_samples = len(dataset)

    if total_samples <= 0:
        raise ValueError(
            f"The '{split_name}' dataset is empty."
        )

    all_features: torch.Tensor | None = None
    all_labels = torch.empty(
        total_samples,
        dtype=torch.long,
    )

    offset = 0

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(
            loader,
            start=1,
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            outputs = model(
                images
            )

            if outputs.ndim < 2:
                raise ValueError(
                    "The feature extractor must return at least a "
                    "two-dimensional tensor."
                )

            features = (
                outputs
                .flatten(start_dim=1)
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.float32,
                )
            )

            labels = (
                labels
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.long,
                )
                .reshape(-1)
            )

            if features.shape[0] != labels.shape[0]:
                raise ValueError(
                    "Feature and label batch sizes do not match."
                )

            if all_features is None:
                all_features = torch.empty(
                    (
                        total_samples,
                        int(features.shape[1]),
                    ),
                    dtype=torch.float32,
                )

            if features.shape[1] != all_features.shape[1]:
                raise ValueError(
                    "The feature dimension changed between batches."
                )

            batch_size_actual = int(
                features.shape[0]
            )
            next_offset = (
                offset
                + batch_size_actual
            )

            if next_offset > total_samples:
                raise RuntimeError(
                    "The DataLoader produced more samples than expected."
                )

            all_features[
                offset:next_offset
            ].copy_(
                features
            )

            all_labels[
                offset:next_offset
            ].copy_(
                labels
            )

            offset = next_offset

            if (
                batch_index == 1
                or batch_index % 25 == 0
                or offset == total_samples
            ):
                percentage = (
                    100.0
                    * offset
                    / total_samples
                )
                print(
                    f"[{split_name}] "
                    f"{offset}/{total_samples} "
                    f"({percentage:.1f}%)",
                    flush=True,
                )

    if all_features is None:
        raise RuntimeError(
            f"No features were extracted for '{split_name}'."
        )

    if offset != total_samples:
        raise RuntimeError(
            f"Expected {total_samples} '{split_name}' samples, "
            f"but extracted {offset}."
        )

    return (
        all_features,
        all_labels,
    )


def save_tensor_atomically(
    tensor: torch.Tensor,
    output_path: Path,
) -> None:
    """Save a tensor through a temporary file before replacing the target."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix
        + ".tmp"
    )

    torch.save(
        tensor,
        temporary_path,
    )

    temporary_path.replace(
        output_path
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract deterministic pretrained frozen-backbone features."
        )
    )

    parser.add_argument(
        "--dataset",
        type=parse_dataset,
        required=True,
        help="MNIST, CIFAR10, or TINYIMAGENET.",
    )

    parser.add_argument(
        "--model",
        type=parse_model,
        required=True,
        help=(
            "alexnet, resnext, efficientnetv2, or swintransformer."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Raw dataset directory. Default: data.",
    )

    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("data/features"),
        help=(
            "Root output directory. Dataset and model subdirectories are "
            "created automatically."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Extraction batch size. Default: 32.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers. Default: 4.",
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
        help="Extraction device. Default: auto.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help=(
            "Explicit image size. Zero uses the same official pretrained "
            "transform as run_experiments.py."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="DataLoader seed. Default: 42.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing feature tensors.",
    )

    return parser


def main() -> int:
    """Extract and persist the requested dataset/model feature tensors."""
    args = build_argument_parser().parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "'--batch-size' must be greater than 0."
        )

    if args.num_workers < 0:
        raise ValueError(
            "'--num-workers' must be greater than or equal to 0."
        )

    if args.image_size < 0:
        raise ValueError(
            "'--image-size' must be greater than or equal to 0."
        )

    if args.seed < 0:
        raise ValueError(
            "'--seed' must be greater than or equal to 0."
        )

    dataset_name: DatasetName = args.dataset
    model_name: ModelName = args.model
    device = resolve_device(
        args.device
    )

    output_dir = (
        args.features_dir
        / dataset_name.value
        / model_name.value
    )

    expected_paths = {
        "train_features":
            output_dir / "train_features.pt",
        "train_labels":
            output_dir / "train_labels.pt",
        "test_features":
            output_dir / "test_features.pt",
        "test_labels":
            output_dir / "test_labels.pt",
        "metadata":
            output_dir / "metadata.json",
    }

    existing_paths = [
        path
        for path in expected_paths.values()
        if path.exists()
    ]

    if existing_paths and not args.overwrite:
        raise FileExistsError(
            "Feature artifacts already exist. Use '--overwrite' only when "
            "you intentionally want to replace them. Existing files: "
            f"{[str(path) for path in existing_paths]}"
        )

    transform, resolved_image_size = (
        build_preprocessing_transform(
            dataset_name=dataset_name,
            model_name=model_name,
            pretrained=True,
            explicit_image_size=
                args.image_size,
        )
    )

    if transform is None:
        raise RuntimeError(
            "Pretrained feature extraction requires a resolved transform."
        )

    train_dataset, test_dataset, classes = (
        load_full_datasets(
            dataset_name=dataset_name,
            data_dir=args.data_dir,
            transform=transform,
        )
    )

    model = build_feature_extractor(
        model_name=model_name,
    ).to(
        device
    )

    print(
        "============================================================",
        flush=True,
    )
    print(
        "TFM frozen-feature extraction",
        flush=True,
    )
    print(
        f"Dataset: {dataset_name.value}",
        flush=True,
    )
    print(
        f"Model: {model_name.value}",
        flush=True,
    )
    print(
        f"Device: {device}",
        flush=True,
    )
    print(
        f"Image size: {resolved_image_size}",
        flush=True,
    )
    print(
        f"Training samples: {len(train_dataset)}",
        flush=True,
    )
    print(
        f"Test samples: {len(test_dataset)}",
        flush=True,
    )
    print(
        f"Output directory: {output_dir}",
        flush=True,
    )
    print(
        "============================================================",
        flush=True,
    )

    train_features, train_labels = extract_split(
        model=model,
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
        split_name="train",
    )

    test_features, test_labels = extract_split(
        model=model,
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed + 1,
        split_name="test",
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_tensor_atomically(
        train_features,
        expected_paths[
            "train_features"
        ],
    )
    save_tensor_atomically(
        train_labels,
        expected_paths[
            "train_labels"
        ],
    )
    save_tensor_atomically(
        test_features,
        expected_paths[
            "test_features"
        ],
    )
    save_tensor_atomically(
        test_labels,
        expected_paths[
            "test_labels"
        ],
    )

    metadata = {
        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "dataset":
            dataset_name.value,
        "model":
            model_name.value,
        "pretrained":
            True,
        "backbone_frozen":
            True,
        "feature_definition":
            "input vector to the replacement final linear classifier",
        "train_samples":
            int(train_features.shape[0]),
        "test_samples":
            int(test_features.shape[0]),
        "feature_dimension":
            int(train_features.shape[1]),
        "num_classes":
            len(classes),
        "classes":
            classes,
        "image_size":
            resolved_image_size,
        "dtype":
            str(train_features.dtype),
        "torch_version":
            torch.__version__,
        "torchvision_version":
            torchvision.__version__,
        "python_version":
            sys.version,
        "command_arguments": {
            "data_dir":
                str(args.data_dir),
            "features_dir":
                str(args.features_dir),
            "batch_size":
                args.batch_size,
            "num_workers":
                args.num_workers,
            "device":
                str(device),
            "image_size":
                args.image_size,
            "seed":
                args.seed,
        },
    }

    with expected_paths[
        "metadata"
    ].open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            metadata,
            output_file,
            indent=2,
            sort_keys=True,
        )

    print(
        "Feature extraction completed successfully.",
        flush=True,
    )
    print(
        f"Feature dimension: {train_features.shape[1]}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
