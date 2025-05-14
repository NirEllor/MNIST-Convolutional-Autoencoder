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


def init_models(encoder_class, latent_dim, channels, device, use_subset):
    encoder = encoder_class(latent_dim=latent_dim, channels=channels).to(device)
    mlp = ClassifierMLP(latent_dim=latent_dim).to(device)
    if use_subset:
        optimizer = optim.Adam(
            list(encoder.parameters()) + list(mlp.parameters()),
            lr=1e-4,  # reduced LR for stability on small data
            weight_decay=1e-4  # more regularization on smaller dataset
        )
    else:
        optimizer = optim.Adam(
            list(encoder.parameters()) + list(mlp.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )

    criterion = nn.CrossEntropyLoss()
    return encoder, mlp, optimizer, criterion


def train_with_logging(encoder, mlp, train_loader, test_loader, optimizer, criterion,
                       device, epochs, early_stopping, patience, label, latent_dim, channels):
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


def run_classifier_experiment(train_dataset, test_loader, encoder_class, latent_dim, channels, use_subset=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, label, epochs, early_stopping, patience = get_dataloaders(train_dataset, use_subset)
    encoder, mlp, optimizer, criterion = init_models(encoder_class, latent_dim, channels, device, use_subset)

    train_with_logging(encoder, mlp, train_loader, test_loader, optimizer, criterion,
                       device, epochs, early_stopping, patience, label, latent_dim, channels)


def Q2():
    run_classifier_experiment(
        train_dataset, test_loader,
        encoder_class=Encoder,
        latent_dim=16,
        channels=(4, 8, 16),
        use_subset=False
    )

    run_classifier_experiment(
        train_dataset, test_loader,
        encoder_class=Encoder,
        latent_dim=16,
        channels=(4, 8, 16),
        use_subset=True
    )

