import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(chan, chan, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers=3):
        super().__init__()
        layers = [nn.Conv1d(input_dim, latent_dim, 3, padding=1)]
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv1d(latent_dim, latent_dim, 4, stride=2, padding=1),
                    ResBlock(latent_dim),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, num_layers=3):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.ConvTranspose1d(latent_dim, latent_dim, 4, stride=2, padding=1),
                    ResBlock(latent_dim),
                ]
            )
        layers.append(nn.Conv1d(latent_dim, output_dim, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
