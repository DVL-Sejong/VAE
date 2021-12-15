from vae import VAEDataLoader, VanillaVAE
from progress.bar import IncrementalBar
from os.path import abspath, join
from pathlib import Path

import pandas as pd
import numpy as np
import argparse
import yaml
import time

from torch.utils.data import DataLoader
import torch.optim as optim
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise FileNotFoundError(exc)


def manual_seed(**kwargs):
    seed = kwargs['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f'Manual seed to {seed}')
    print()


def init_loss_df():
    loss_df = pd.DataFrame(columns=['index', 'loss', 'reconstruction_loss', 'kl-divergence'])
    loss_df = loss_df.set_index('index')
    return loss_df


def update_loss_df(loss_df: pd.DataFrame, loss_dict: dict, epoch: int):
    loss_df.loc[epoch] = {'loss': loss_dict['loss'].item(),
                          'reconstruction_loss': loss_dict['reconstruction_loss'].item(),
                          'kl-divergence': loss_dict['kl-divergence'].item()}
    return loss_df


def get_mean_loss_dict(loss_df: pd.DataFrame):
    loss_dict = {'loss': np.mean(loss_df['loss'].to_list()),
                 'reconstruction_loss': np.mean(loss_df['reconstruction_loss'].to_list()),
                 'kl-divergence': np.mean(loss_df['kl-divergence'].to_list())}
    return loss_dict


def print_loss_dict(loss_dict: dict, break_line=True):
    loss = loss_dict['loss'].item()
    reconstruction_loss = loss_dict['reconstruction_loss'].item()
    kld = loss_dict['kl-divergence'].item()

    print(f'loss: {loss:4.8f}, '
          f'reconstruction loss: {reconstruction_loss:4.8f}, '
          f'kl-divergence: {kld:4.8f}')

    if break_line:
        print()


def experiment(model: VanillaVAE, data_loader: VAEDataLoader, device: torch.device,
               batch_size: int, learning_rate: float, epochs: int,
               model_path: str, exp_name: str):
    model.to(device)
    params = model.parameters()

    optimizer = optim.Adam(params, lr=learning_rate)
    loss_df = init_loss_df()

    train_loader = data_loader.get_data_loader()
    kld_weight = batch_size / data_loader.num_train_imgs

    for epoch in range(epochs):
        model, optimizer, loss_dict = train(epoch, model, train_loader, optimizer, kld_weight)
        loss_df = update_loss_df(loss_df, loss_dict, epoch)

    Path(abspath(model_path)).mkdir(exist_ok=True, parents=True)
    model_path = join(abspath(model_path), f'{exp_name}.pt')
    model.save(model_path)
    print(f'save trained model to {model_path}')


def train(epoch: int, model: VanillaVAE, train_loader: DataLoader, optimizer: optim.Adam, kld_weight: float):
    loss_df = init_loss_df()
    time_start = time.time()

    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    bar = IncrementalBar(f'Epoch: {epoch:>3}', max=len(train_loader))
    for i, batch in enumerate(train_loader):
        real_img, labels = batch
        real_img = real_img.to(device)
        labels = labels.to(device)

        recons, input, mu, log_var = model.forward(real_img, labels=labels)
        loss_dict = model.loss_function(recons, input, mu, log_var, kld_weight=kld_weight)
        loss_df = update_loss_df(loss_df, loss_dict, i)
        train_loss = loss_dict['loss']
        train_loss.backward()
        optimizer.step()

        bar.next()

    time_end = time.time()
    bar.finish()

    loss_dict = get_mean_loss_dict(loss_df)
    print_loss_dict(loss_dict, break_line=False)

    print(f'Took {(time_end - time_start):.4f} sec')
    return model, optimizer, loss_dict


if __name__ == '__main__':
    config = parse_args()
    manual_seed(**config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'set device to {device}')

    data_loader = VAEDataLoader(batch_size=config['exp_params']['batch_size'],
                                img_size=config['exp_params']['img_size'],
                                data_path=config['exp_params']['data_path'])
    model = VanillaVAE(in_channels=config['model_params']['in_channels'],
                       latent_dim=config['model_params']['latent_dim'],
                       hidden_dims=None)

    experiment(model=model,
               data_loader=data_loader,
               device=device,
               batch_size=config['exp_params']['batch_size'],
               learning_rate=config['exp_params']['LR'],
               epochs=config['exp_params']['epochs'],
               model_path=config['exp_params']['model_path'],
               exp_name=config['exp_params']['exp_name'])
