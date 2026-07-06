import os
import copy
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate(
    model: nn.Module, 
    loaders: Dict[str, DataLoader], 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    device: torch.device, 
    num_epochs: int = 10, 
    task_id: str = "default",
    eval_split: str = "valid"
) -> Dict[str, float]:
    """
    Trains the model, tracks the best validation weights in memory, 
    and evaluates performance on the target split (validation or test).

    Args:
        model: The PyTorch neural network model (or linear probe).
        loaders: Dictionary containing 'train', 'valid', and 'test' DataLoaders.
        criterion: The loss function (e.g., CrossEntropyLoss).
        optimizer: The optimization algorithm (e.g., Adam).
        device: The device to run the computations on (CPU or CUDA).
        num_epochs: Maximum number of training epochs.
        task_id: Unique identifier for logging (no longer hammers the disk).
        eval_split: The loader key to run final evaluation on ('valid' or 'test').

    Returns:
        A dictionary containing evaluation metrics: accuracy, precision, recall, and f1.
    """
    best_valid_loss = float('inf')
    # In-memory backup to avoid disk I/O bottlenecks during evolutionary loops
    best_model_state = None

    # --- TRAINING LOOP ---
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in loaders["train"]:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # --- VALIDATION LOOP ---
        model.eval()
        valid_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in loaders["valid"]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        avg_train_loss = train_loss / len(loaders["train"])
        avg_valid_loss = valid_loss / len(loaders["valid"])

        # Descomenta esta línea si necesitas depurar las épocas de entrenamiento:
        # print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")

        # Checkpoint en memoria RAM (veloz y seguro contra condiciones de carrera)
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            # Hacemos una copia profunda de los pesos mapeándolos a la CPU para evitar picos de VRAM
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # --- EVALUATION ON TARGET SPLIT ---
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loaders[eval_split]:
            inputs = inputs.to(device, non_blocking=True)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            # Nos aseguramos de pasar las etiquetas a la CPU de forma segura
            all_labels.extend(labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels)

    # Calculate final metrics with safe macro-averaging
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average='macro', zero_division=0),
        "recall": recall_score(all_labels, all_preds, average='macro', zero_division=0),
        "f1": f1_score(all_labels, all_preds, average='macro', zero_division=0)
    }