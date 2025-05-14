import torch.nn as nn


class ClassifierMLP(nn.Module):
    """
    Simple MLP classifier that maps a latent vector to digit class logits (0–9).

    Typically used on top of an encoder to perform digit classification.

    Args:
        latent_dim (int): Dimensionality of the input latent vector.

    Example:
        clf = ClassifierMLP(latent_dim=16)
        logits = clf(z)  # shape (B, 10)
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 10)

    def forward(self, z):
        """
        Forward pass through the classifier.

        Args:
            z (Tensor): Latent vector of shape (B, latent_dim)

        Returns:
            Tensor: Class logits of shape (B, 10)
        """
        return self.fc(z)  # Use nn.CrossEntropyLoss(), which applies softmax internally
