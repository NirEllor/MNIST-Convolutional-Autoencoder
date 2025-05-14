import torch.nn as nn
import torch.optim as optim
from encoder import Encoder
from train import train_autoencoder
from eval import evaluate_autoencoder
from plots import show_reconstructions
from data import *

latent_dims = [4, 16]
model_configs = [("small", (4, 8, 16)), ("large", (16, 32, 64))]


def Q1_4(pre_trained=False):
    """
    Runs Q1 and Q4:
    - If pre_trained=False: trains autoencoders from scratch (Q1)
    - If pre_trained=True: uses pretrained encoder (from Q2) and trains decoder only (Q4)

    Args:
        pre_trained (bool): Whether to load and freeze a pretrained encoder
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name, channels in model_configs:
        if pre_trained and (model_name == "small" or  channels ==(4, 8, 16)):
            break
        train_losses = []
        test_losses = []

        print(f"\n=== {model_name.upper()} model with channels {channels} ===")
        for latent_dim in latent_dims:
            pre_trained_decoder = " with pre_trained_encoder" if pre_trained else " from scratch"
            print(f"\nTraining {model_name} model with latent_dim = {latent_dim}" + pre_trained_decoder)
            encoder = Encoder(latent_dim=latent_dim, channels=channels).to(device)
            if pre_trained and model_name == "large" and channels == (16, 32, 64):
                encoder = load_best_encoder(
                    encoder,
                    latent_dim=latent_dim,
                    channels=channels,
                    pre_trained=True,
                    pre_trained_encoder_path=f'best_large_latent{latent_dim}.pt',  # adjust if needed
                    device=device
                )
                for p in encoder.parameters():
                    p.requires_grad = False

            model = ConvAutoencoder(latent_dim=latent_dim, channels=channels, encoder=encoder).to(device)
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

            if not pre_trained:
                model.load_state_dict(best_model_state)
                torch.save(best_model_state, f'best_{model_name}_latent{latent_dim}.pt')

            train_losses.append(train_loss)
            test_losses.append(best_test_loss)

            # Show reconstructions
            print(f"\nShowing reconstructions for {model_name} model with latent_dim={latent_dim}")
            show_reconstructions(model, test_loader, device)


