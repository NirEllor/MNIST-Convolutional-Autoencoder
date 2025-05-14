import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from autoencoder import ConvAutoencoder

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


def load_best_encoder(_encoder, latent_dim, channels, pre_trained, pre_trained_encoder_path, device):
    _autoencoder = ConvAutoencoder(latent_dim=latent_dim, channels=channels).to(device) if pre_trained else None
    _autoencoder.load_state_dict(
        torch.load(pre_trained_encoder_path, map_location=device, weights_only=True)
    )
    encoder_state_dict = {
        k.replace("encoder.", ""): v
        for k, v in _autoencoder.state_dict().items()
        if k.startswith("encoder.")
    }
    _encoder.load_state_dict(encoder_state_dict)
    return _encoder