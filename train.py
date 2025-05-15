import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Tuple


def train_autoencoder(
    model: Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Module,
    device: torch.device
) -> float:
    """
    Trains an autoencoder for one epoch.

    Args:
        model (nn.Module): Autoencoder model (encoder + decoder).
        dataloader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (nn.Module): Loss function (e.g., L1 or MSE).
        device (torch.device): Computation device ("cpu" or "cuda").

    Returns:
        float: Average reconstruction loss over the epoch.
    """
    model.train()
    total_loss = 0
    total = 0

    for imgs, _ in dataloader:
        imgs = imgs.to(device)

        optimizer.zero_grad()
        recon = model(imgs)
        loss = criterion(recon, imgs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)

    return total_loss / total


def train_classifier(
    encoder: Module,
    mlp: Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Trains an encoder + MLP classifier for one epoch.

    Args:
        encoder (nn.Module): Encoder model.
        mlp (nn.Module): MLP classifier mapping latent vectors to logits.
        dataloader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        device (torch.device): Computation device ("cpu" or "cuda").

    Returns:
        Tuple[float, float]: (average loss, accuracy)
    """
    encoder.train()
    mlp.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        z = encoder(imgs)
        logits = mlp(z)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


