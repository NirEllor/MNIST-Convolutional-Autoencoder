import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



class Encoder(nn.Module):
    def __init__(self, latent_dim, channels=(4, 8, 16), input_shape=(1, 28, 28)):
        super().__init__()
        c1, c2, c3 = channels

        self.conv = nn.Sequential(
            nn.Conv2d(1, c1, 3, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c2, c3, 3, stride=1, padding=1),
            nn.BatchNorm2d(c3),
            nn.LeakyReLU(0.2)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.conv(dummy)
            self.flat_dim = out.view(1, -1).shape[1]
            self.unflatten_shape = out.shape[1:]

        self.fc = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, flat_dim, unflatten_shape, channels):
        super().__init__()
        c3, c2, c1 = channels

        self.fc = nn.Linear(latent_dim, flat_dim)
        self.unflatten_shape = unflatten_shape

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(c2, c1, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(c1, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), *self.unflatten_shape)
        return self.deconv(x)

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16, channels=(4, 8, 16)):
        super().__init__()
        self.encoder = Encoder(latent_dim, channels)
        self.decoder = Decoder(latent_dim, self.encoder.flat_dim, self.encoder.unflatten_shape, channels[::-1])

    def forward(self, x):
        z = self.encoder(x)
        # print("Latent std:", z.std().item())
        return self.decoder(z)

# Transform: convert to tensor and normalize to [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        imgs, _ = batch  # we don't need the labels
        imgs = imgs.to(device)

        optimizer.zero_grad()
        recon = model(imgs)
        loss = criterion(recon, imgs)  # compare reconstruction to original
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            imgs, _ = batch
            imgs = imgs.to(device)

            recon = model(imgs)
            loss = criterion(recon, imgs)
            total_loss += loss.item() * imgs.size(0)

    return total_loss / len(dataloader.dataset)

def show_reconstructions(model, dataloader, device, n=10):
    model.eval()
    imgs, _ = next(iter(dataloader))
    imgs = imgs[:n].to(device)

    with torch.no_grad():
        recons = model(imgs)

    # Undo normalization for visualization (if input was normalized to [-1, 1])
    imgs = denormalize(imgs).cpu().numpy()
    recons = denormalize(recons).cpu().numpy()
    print("Reconstruction range:", recons.min(), recons.max())

    plt.figure(figsize=(n, 2))
    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)
        plt.imshow(imgs[i][0], cmap='gray')
        plt.axis('off')

        # Reconstructed
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(recons[i][0], cmap='gray')
        plt.axis('off')

    plt.suptitle("Top: Original, Bottom: Reconstructed")
    plt.show()

def denormalize(tensor):
    return (tensor + 1) / 2  # for inputs normalized with mean=0.5, std=0.5

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
        train_loss = None
        test_loss = None

        best_loss = float('inf')
        epochs_no_improve = 0
        patience = 5  # אפשר גם 7 או 10

        for epoch in range(1, 100):  # שים מספר גדול – נעצור אוטומטית
            train_loss = train(model, train_loader, optimizer, criterion, device)
            test_loss = evaluate(model, test_loader, criterion, device)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

            if test_loss < best_loss - 1e-4:  # שיפור קטן נחשב
                best_loss = test_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Show reconstructions
        print(f"\nShowing reconstructions for {model_name} model with latent_dim={latent_dim}")
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




