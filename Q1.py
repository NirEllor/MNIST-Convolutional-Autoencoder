import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from autoencoder import ConvAutoencoder
from train import train_autoencoder
from eval import evaluate_autoencoder
from plots import show_reconstructions
from data import *

def Q1():
    latent_dims = [4, 16]
    model_configs = [("small", (4, 8, 16)), ("large", (16, 32, 64))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name, channels in model_configs:
        train_losses = []
        test_losses = []

        print(f"\n=== {model_name.upper()} model with channels {channels} ===")
        for latent_dim in latent_dims:
            print(f"\nTraining {model_name} model with latent_dim = {latent_dim}")
            model = ConvAutoencoder(latent_dim=latent_dim, channels=channels)
            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.L1Loss()
            best_test_loss = float('inf')
            best_model_state = None
            train_loss = None

            epochs_no_improve = 0
            patience = 5  #  גם 7 או 10
            min_delta = 1e-4

            for epoch in range(1, 11):
                train_loss = train_autoencoder(model, train_loader, optimizer, criterion, device)
                test_loss = evaluate_autoencoder(model, test_loader, criterion, device)
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

                if test_loss < best_test_loss - min_delta:
                    best_test_loss = test_loss
                    epochs_no_improve = 0
                    best_model_state = model.state_dict()

                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            model.load_state_dict(best_model_state)
            torch.save(best_model_state, f'best_{model_name}_latent{latent_dim}.pt')

            train_losses.append(train_loss)
            test_losses.append(best_test_loss)

            # Show reconstructions
            # print(f"\nShowing reconstructions for {model_name} model with latent_dim={latent_dim}")
            show_reconstructions(model, test_loader, device)

        # Plotting loss vs latent_dim for this model
        plt.figure(figsize=(8, 5))
        plt.plot(latent_dims, train_losses, marker='o', label='Train Loss')
        plt.plot(latent_dims, test_losses, marker='s', label='Test Loss')
        plt.title(f'{model_name.capitalize()} Model – Loss vs. Latent Dimension')
        plt.xlabel('Latent Dimension')
        plt.ylabel('Loss (L1)')
        plt.grid(True)
        plt.legend()
        plt.show()

