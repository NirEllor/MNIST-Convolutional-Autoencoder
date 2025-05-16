import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from encoder import Encoder
from train import train_classifier
from eval import evaluate_classifier
from plots import plot_metrics
from data import *
from classifier import ClassifierMLP
from torch.utils.data import Subset




def get_dataloaders(train_dataset, use_subset):
    """
    Returns a DataLoader (full or subset) along with training metadata.

    Args:
        train_dataset (Dataset): Full MNIST training dataset.
        use_subset (bool): Whether to use only 100 training samples.

    Returns:
        Tuple[DataLoader, str, int, bool, Optional[int]]:
        - training loader
        - label for title/printing
        - number of epochs
        - whether early stopping is enabled
        - patience for early stopping
    """
    if use_subset:
        indices = np.random.choice(len(train_dataset), 100, replace=False)
        train_loader = DataLoader(Subset(train_dataset, indices), batch_size=32, shuffle=True)
        label = "Subset of 100 samples"
        epochs = 30
        early_stopping = True
        patience = 5
    else:
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        label = "Full training set"
        epochs = 10
        early_stopping = False
        patience = None
    return train_loader, label, epochs, early_stopping, patience


def init_models(encoder_class, latent_dim, channels, device, use_subset, pre_trained=False, pre_trained_encoder_path=""):
    """
    Returns a DataLoader (full or subset) along with training metadata.

    Args:
        use_subset (bool): Whether to use only 100 training samples.

    Returns:
        Tuple[DataLoader, str, int, bool, Optional[int]]:
        - training loader
        - label for title/printing
        - number of epochs
        - whether early stopping is enabled
        - patience for early stopping
    """
    # Step 1: Create encoder and autoencoder
    _encoder = encoder_class(latent_dim=latent_dim, channels=channels).to(device)

    # Step 2: Load encoder weights from saved autoencoder if needed
    if pre_trained and pre_trained_encoder_path:
        _encoder = load_best_encoder(_encoder, latent_dim, channels, pre_trained, pre_trained_encoder_path, device)
        for param in _encoder.parameters():
            param.requires_grad = False
        _encoder.eval()

    # Step 3: Initialize classifier
    mlp = ClassifierMLP(latent_dim=latent_dim).to(device)

    # Step 4: Optimizer with only trainable params
    params = filter(lambda p: p.requires_grad, list(_encoder.parameters()) + list(mlp.parameters()))
    optimizer = optim.Adam(
        params,
        lr=1e-4,
        weight_decay=1e-4 if use_subset else 1e-5
    )

    criterion = nn.CrossEntropyLoss()
    return _encoder, mlp, optimizer, criterion


def train_with_logging(encoder, mlp, train_loader, test_loader, optimizer, criterion,
                       device, epochs, early_stopping, patience, label, latent_dim, channels):
    """
    Trains classifier with logging and optional early stopping.

    Args:
        encoder (nn.Module): The encoder.
        mlp (nn.Module): The MLP classifier.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Test data loader.
        optimizer (Optimizer): Optimizer for model parameters.
        criterion (nn.Module): Loss function.
        device (torch.device): Computation device.
        epochs (int): Max number of training epochs.
        early_stopping (bool): Whether to use early stopping.
        patience (int): Patience for early stopping.
        label (str): Experiment label for plots/titles.
        latent_dim (int): Latent space size.
        channels (tuple): Convolutional channel config.
    """
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_test_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    min_delta = 1e-4

    print(f"=====training {label}=====")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_classifier(encoder, mlp, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate_classifier(encoder, mlp, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Test Loss={test_loss:.4f}, Acc={test_acc:.4f}")

        if early_stopping:
            if test_loss < best_test_loss - min_delta:
                best_test_loss = test_loss
                best_model_state = (encoder.state_dict(), mlp.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    if early_stopping and best_model_state:
        encoder.load_state_dict(best_model_state[0])
        mlp.load_state_dict(best_model_state[1])

    plot_metrics(train_losses, test_losses, train_accs, test_accs,
                 title=f"Classification ({label}) – latent_dim={latent_dim}, channels={channels}")


def run_classifier_experiment(train_dataset, test_loader, encoder_class, latent_dim, channels, use_subset=False,
                              pre_trained=False, pre_trained_encoder_path=""):
    """
    Runs the training + evaluation for a given encoder/MLP configuration.

    Args:
        train_dataset (Dataset): Full MNIST training dataset.
        test_loader (DataLoader): DataLoader for test set.
        encoder_class (Type[Encoder]): The encoder class to instantiate.
        latent_dim (int): Latent vector dimension.
        channels (tuple): Conv channel configuration.
        use_subset (bool): Use only 100 training examples.
        pre_trained (bool): Use encoder pretrained from Q1.
        pre_trained_encoder_path (str): Path to the saved autoencoder weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, label, epochs, early_stopping, patience = get_dataloaders(train_dataset, use_subset)
    encoder, mlp, optimizer, criterion = init_models(encoder_class, latent_dim, channels, device, use_subset, pre_trained, pre_trained_encoder_path=pre_trained_encoder_path)

    train_with_logging(encoder, mlp, train_loader, test_loader, optimizer, criterion,
                       device, epochs, early_stopping, patience, label, latent_dim, channels)


def Q2():
    """
    Question 2:
    Trains encoder+MLP from scratch on:
    - Full dataset
    - Subset of 100 samples
    """
    run_classifier_experiment(
        train_dataset, test_loader,
        encoder_class=Encoder,
        latent_dim=16,
        channels=(16, 32, 64),
        use_subset=False,
        pre_trained=False
    )

    run_classifier_experiment(
        train_dataset, test_loader,
        encoder_class=Encoder,
        latent_dim=16,
        channels=(16, 32, 64),
        use_subset=True,
        pre_trained = False
    )


def Q3():
    """
    Question 3:
    Trains only MLP with frozen encoder from Q1:
    - Full dataset
    - Subset of 100 samples
    """
    encoder_path = 'best_large_latent16.pt'

    run_classifier_experiment(
        train_dataset, test_loader,
        encoder_class=Encoder,
        latent_dim=16,
        channels=(16, 32, 64),
        use_subset=False,
        pre_trained=True,
        pre_trained_encoder_path=encoder_path
    )

    run_classifier_experiment(
        train_dataset, test_loader,
        encoder_class=Encoder,
        latent_dim=16,
        channels=(16, 32, 64),
        use_subset=True,
        pre_trained = True,
        pre_trained_encoder_path=encoder_path
    )

