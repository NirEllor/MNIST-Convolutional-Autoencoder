import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16, channels=(16, 32, 64), encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else Encoder(latent_dim, channels)
        self.decoder = Decoder(latent_dim, self.encoder.flat_dim, self.encoder.unflatten_shape, channels[::-1])

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
