import os
import torch
from typing import Dict, List, Tuple, Optional
from torch.utils.data import random_split, DataLoader, Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, ImageFolder

def get_dataloaders(
    dataset_name: str, 
    data_dir: str = "./data", 
    selection_mask: Optional[Dict[int, int]] = None, 
    batch_size: int = 32, 
    train_split: float = 0.8
) -> Tuple[Dict[str, DataLoader], List[str]]:
    """
    Loads MNIST, CIFAR10, or TinyImageNet datasets.
    Applies an evolutionary selection mask exclusively to the training set.
    
    Args:
        dataset_name: Name of the dataset (MNIST, CIFAR10, TINYIMAGENET).
        data_dir: Directory where the datasets are or will be downloaded.
        selection_mask: Dictionary mapping indices to 1 (keep) or 0 (discard).
        batch_size: Number of samples per batch.
        train_split: Proportion of the dataset to use for training (vs validation).
        
    Returns:
        Tuple containing:
        - A dictionary with the DataLoaders ('train', 'valid', 'test').
        - A list of class names.
    """
    dataset_name = dataset_name.upper()
    os.makedirs(data_dir, exist_ok=True)

    # Standard ImageNet normalization (Required for pre-trained models)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3), # Convert 1 channel to 3 channels for CNNs
            transforms.ToTensor(),
            normalize
        ])
        full_train_dataset = MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = MNIST(root=data_dir, train=False, download=True, transform=transform)
        classes = [str(i) for i in range(10)]

    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
        full_train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        classes = full_train_dataset.classes

    elif dataset_name == "TINYIMAGENET":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
        train_path = os.path.join(data_dir, "tiny-imagenet-200", "train")
        test_path = os.path.join(data_dir, "tiny-imagenet-200", "val")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"TinyImageNet data missing at {train_path}. Please download it manually.")
            
        full_train_dataset = ImageFolder(root=train_path, transform=transform)
        test_dataset = ImageFolder(root=test_path, transform=transform)
        classes = full_train_dataset.classes

    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    # 1. Split the full dataset into base Train and Validation sets
    total_full = len(full_train_dataset)
    train_size = int(train_split * total_full)
    valid_size = total_full - train_size
    
    # Fixed generator seed guarantees the validation split layout remains uncompromised
    generator = torch.Generator().manual_seed(42)
    base_train_dataset, valid_dataset = random_split(
        full_train_dataset, 
        [train_size, valid_size], 
        generator=generator
    )

    # 2. Apply the selection mask ONLY to the base training slice
    if selection_mask:
        # Sequential indices 0 to (train_size - 1) match your metaheuristics space exactly
        selected_indices = [i for i in range(len(base_train_dataset)) if selection_mask.get(i, 0) == 1]
        train_dataset = Subset(base_train_dataset, selected_indices)
    else:
        train_dataset = base_train_dataset

    # Windows OS Compatibility Patch for Jupyter Notebooks
    num_workers = 0 if os.name == 'nt' else 4

    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "valid": DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    }

    print(f"[{dataset_name}] Images loaded. Active Training Instances: {len(train_dataset)}/{len(base_train_dataset)}")
    return loaders, classes


def get_feature_dataloaders(
    features_dir: str,
    selection_mask: Optional[Dict[int, int]] = None,
    batch_size: int = 256,
    train_split: float = 0.8
) -> Tuple[Dict[str, DataLoader], int]:
    """
    Alternative fast loader that works directly with pre-extracted feature tensors (.pt).
    Bypasses all raw image transformations and forward-passes during the metaheuristic search.
    """
    train_features = torch.load(os.path.join(features_dir, "train_features.pt"), weights_only=True)
    train_labels = torch.load(os.path.join(features_dir, "train_labels.pt"), weights_only=True)
    
    test_features = torch.load(os.path.join(features_dir, "test_features.pt"), weights_only=True)
    test_labels = torch.load(os.path.join(features_dir, "test_labels.pt"), weights_only=True)
    
    full_train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    
    total_full = len(full_train_dataset)
    train_size = int(train_split * total_full)
    valid_size = total_full - train_size
    
    generator = torch.Generator().manual_seed(42)
    base_train_dataset, valid_dataset = random_split(
        full_train_dataset, [train_size, valid_size], generator=generator
    )
    
    if selection_mask:
        selected_indices = [i for i in range(len(base_train_dataset)) if selection_mask.get(i, 0) == 1]
        train_dataset = Subset(base_train_dataset, selected_indices)
    else:
        train_dataset = base_train_dataset
        
    num_workers = 0 if os.name == 'nt' else 4
    num_classes = len(torch.unique(train_labels))
    
    # Larger batch sizes are ideal when training purely on linear layers
    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "valid": DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    }
    
    print(f"[Features] Vectors loaded. Active instances: {len(train_dataset)}/{len(base_train_dataset)}")
    return loaders, num_classes