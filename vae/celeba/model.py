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
        # Build Encoder
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
        # Build Decoder
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
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        print('encode()')
        print(f'shape of input: {input.shape}')
        result = self.encoder(input)
        print(f'shape of result: {result.shape}')
        result = torch.flatten(result, start_dim=1)
        print(f'shape of result: {result.shape}')

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        print(f'shape of mu: {mu.shape}')
        log_var = self.fc_var(result)
        print(f'log_var: {log_var.shape}')
        print()

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        print(f'decode()')
        print(f'shape of z: {z.shape}')  # (144, 128)
        result = self.decoder_input(z)
        print(f'shape of decoder_input(z): {result.shape}')  # (144, 2048)
        result = result.view(-1, 512, 2, 2)
        print(f'shape of result: {result.shape}')  # (144, 512, 2, 2)
        result = self.decoder(result)
        print(f'shape of decoder result: {result.shape}')  # (144, 32, 32, 32)
        result = self.final_layer(result)
        print(f'shape of final_layer(result): {result.shape}')  # (144, 3, 64, 64)
        print()
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """

        print(f'reparameterize()')
        print(f'shape of mu: {mu.shape}')
        print(f'shape of logvar: {logvar.shape}')

        std = torch.exp(0.5 * logvar)
        print(f'shape of std: {std.shape}')
        eps = torch.randn_like(std)
        print(f'sahpe of eps: {eps.shape}')
        z = eps * std + mu
        print(f'shape of z: {z.shape}')
        print()
        return z

    def forward(self, input_data: dict) -> List[Tensor]:
        input_tensor = input_data['data']
        print(f'model forward()')
        print(f'shape of input: {input_tensor.shape}')
        mu, log_var = self.encode(input_tensor)
        print(f'shape of mu: {mu.shape}')
        print(f'shape of log_var: {log_var.shape}')
        z = self.reparameterize(mu, log_var)
        print(f'shape of z: {z.shape}')
        output = self.decode(z)
        print(f'shape of output: {output.shape}')
        print()
        return [output, input, mu, log_var]

    def loss_function(self, recons, input, mu, log_var, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kwargs['kld_weight'] * kld_loss

        return {'loss': loss, 'reconstruction_loss': recons_loss, 'kl-divergence': -kld_loss}

    def sample(self, num_samples:int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
