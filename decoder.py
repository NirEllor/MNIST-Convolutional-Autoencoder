import torch.nn as nn



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
