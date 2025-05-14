import torch.nn as nn


class ClassifierMLP(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 10)

    def forward(self, z):
        return self.fc(z)

