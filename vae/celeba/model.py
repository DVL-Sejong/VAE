from vae import BaseVAE, Tensor
from typing import List
from torch import nn
from torch.nn import functional as F

import torch


class CelebAVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None) -> None:
        super(CelebAVAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims

        self._init_encoder()
        self._init_decoder()
        self._init_final_layer()

    def _init_encoder(self):
        modules = []
        in_channels = self.in_channels

        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1] * 4, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1] * 4, self.latent_dim)

    def _init_decoder(self):
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1] * 4)

        self.hidden_dims.reverse()

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

    def _init_final_layer(self):
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[-1],
                               self.hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, input_data: dict) -> List[Tensor]:
        input_tensor = input_data['data']
        mu, log_var = self.encode(input_tensor)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)
        return [output, input, mu, log_var]

    def loss_function(self, recons, input, mu, log_var, **kwargs) -> dict:
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kwargs['kld_weight'] * kld_loss

        return {'loss': loss, 'reconstruction_loss': recons_loss, 'kl-divergence': -kld_loss}

    def sample(self, num_samples:int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
