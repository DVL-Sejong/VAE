from vae import BaseVAE, Tensor
from torchdiffeq import odeint
from typing import List

import numpy as np

import torch.nn as nn
import torch


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, n_hidden=25, n_spiral=1, n_channel=2):
        super(RecognitionRNN, self).__init__()
        self.latent_dim = latent_dim
        self.n_hidden = n_hidden
        self.n_spiral = n_spiral
        self.n_channel = n_channel

        # print(f'n_spiral: {n_spiral}, n_channel: {n_channel}')

        self.i2h = nn.Linear(n_spiral * n_channel * 2, n_spiral * n_channel)
        self.h2o = nn.Linear(n_spiral * n_channel, n_spiral * latent_dim * 2)

    def forward(self, x, h):
        # print(f'RecognitionRNN')
        # print(f'shape of x: {x.shape}')  # (144, 28)
        # print(f'shape of h: {h.shape}')  # (144, 28)
        combined = torch.cat((x, h), dim=1)
        # print(f'shape of combined: {combined.shape}')  # (144, 56)
        h = torch.tanh(self.i2h(combined))
        # print(f'shape of h: {h.shape}')  # (144, hidden)
        out = self.h2o(h)
        # print(f'shape of out: {out.shape}')  # (144, 112)
        out = out.view(-1, self.n_spiral, self.latent_dim * 2)
        # print(f'shape of out: {out.shape}')  # (144, 14, 8)
        # print()
        return out, h

    def init_hidden(self, input_data: Tensor):
        return torch.zeros_like(input_data)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class LatentVAE(BaseVAE):
    def __init__(self, batch_size: int,
                 latent_dim: int, n_rnn_hidden: int, n_spiral: int, n_channel: int,
                 n_dec_hidden: int, n_hidden: int, noise_std: float,
                 device: torch.device):
        super(LatentVAE, self).__init__()

        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.n_rnn_hidden = n_rnn_hidden
        self.n_spiral = n_spiral
        self.n_channel = n_channel
        self.n_dec_hidden = n_dec_hidden
        self.n_hidden = n_hidden
        self.noise_std = noise_std
        self.device = device

        self.rec = RecognitionRNN(latent_dim, n_rnn_hidden, n_spiral, n_channel).to(device)
        self.decode_input = nn.Linear(1, 1)
        self.dec = Decoder(latent_dim, 2, n_dec_hidden).to(device)
        self.func = LatentODEfunc(latent_dim, n_hidden).to(device)

    def encode(self, input: Tensor) -> List[Tensor]:
        # print(f'encode()')
        # print(f'shape of input: {input.shape}')  # (144, 14, 10, 2)

        input = input.permute(0, 2, 1, 3)  # (144, 10, 14, 2)
        input = torch.flatten(input, start_dim=2)
        # print(f'shape of input: {input.shape}')   # (144, 10, 28)

        h = self.rec.init_hidden(input[:, 0, :]).to(self.device)
        # print(f'shape of h: {h.shape}')

        for t in reversed(range(input.size(1))):
            observation = input[:, t, :] # (144, 28)
            # print(f'shape of observation: {observation.shape}')
            out, h = self.rec(observation, h)

        # print(f'shape of out: {out.shape}')  # (144, 14, 8)
        # print(f'shape of h: {h.shape}')  # (144, 14, 8)
        mu, log_var = out[:, :, :self.latent_dim], out[:, :, self.latent_dim:]
        # print(f'shape of mu: {mu.shape}')  # (144, 14, 4)
        # print(f'shape of log_var: {log_var.shape}')  # (144, 14, 4)
        # print()
        return [mu, log_var]

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        # print(f'decode()')
        # aa = type(kwargs['timestamp'])
        # print(f'type of timestamp: {aa}')
        # bb = kwargs['timestamp'].shape
        # print(f'shape of timestamp: {bb}')

        # print(f'shape of z: {z.shape}')
        result = odeint(self.func, z, kwargs['timestamp'][0])
        # print(f'shape of odeint result: {result.shape}')  # (10, 144, 14, 4)
        result = result.permute(1, 2, 0, 3)
        # print(f'shape of permute result: {result.shape}')  # (144, 14, 10, 4)
        result = self.dec(result)
        # print(f'shape of decoded result: {result.shape}')  # TODO (144, 14, 10, 2)
        # print()

        # TODO it should be shaped as (144, 14, 10, 2)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        # print(f'reparameterize()')
        # print(f'shape of mu: {mu.shape}')
        # print(f'shape of logvar: {logvar.shape}')
        epsilon = torch.randn(mu.size()).to(self.device)
        # print(f'shape of epsilon: {epsilon.shape}')
        z = epsilon * torch.exp(.5 * logvar) + mu
        # print(f'shape of z: {z.shape}')
        # print()
        return z

    def forward(self, input_data: dict):
        # print('LatentVAE forward()')
        input_tensor = input_data['data']
        timestamp = input_data['timestamp']

        # print(f'shape of input_tensor: {input_tensor.shape}')
        # print(f'shape of timestamp: {timestamp.shape}')

        mu, log_var = self.encode(input_tensor)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z, timestamp=timestamp)

        # print()

        return [output, input_tensor, mu, log_var]

    def loss_function(self, recons, input, mu, log_var, **kwargs):
        # print(f'loss_function()')
        #
        # print(f'shape of recons: {recons.shape}')
        # print(f'shape of input: {input.shape}')
        # print(f'shape of mu: {mu.shape}')
        # print(f'shape of log_var: {log_var.shape}')
        noise_std_ = torch.zeros(recons.size()) + self.noise_std
        # print(f'shape of noise_std_: {noise_std_.shape}')
        noise_logvar = (2. * torch.log(noise_std_)).to(kwargs['device'])

        # print(f'shape of noise_nogvar: {noise_logvar.shape}')
        logpx = torch.mean(log_normal_pdf(input, recons, noise_logvar).sum(-1).sum(-1))
        # print(f'shape of logpx: {logpx.shape}')
        analytic_kl = torch.mean(normal_kl(mu,
                                           log_var,
                                           torch.zeros(mu.size()).to(kwargs['device']),
                                           torch.zeros(log_var.size()).to(kwargs['device']),
                                           device=kwargs['device']).sum(-1))

        # print(f'shape of analytic_kl: {analytic_kl.shape}')
        loss = torch.mean(-logpx + analytic_kl, dim=0)
        # print(f'shape of loss: {loss.shape}')
        # print()

        return {'loss': loss, 'reconstruction_loss': -logpx, 'kl-divergence': analytic_kl}


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    pdf = -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))
    # print(f'shape of log normal pdf: {pdf.shape}')
    return pdf


def normal_kl(mu1, lv1, mu2, lv2, **kwargs):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl
