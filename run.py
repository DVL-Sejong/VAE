from vae import CelebALoader, CelebAVAE, LatentODELoader, LatentVAE
from progress.bar import IncrementalBar
import matplotlib.pyplot as plt
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
                        default='configs/celeba.yaml')

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


def experiment(model, data_loader, device: torch.device,
               batch_size: int, learning_rate: float, epochs: int,
               model_path: str, exp_name: str):
    model.to(device)
    params = model.parameters()

    loader_dict = data_loader.get_data_loader()

    optimizer = optim.Adam(params, lr=learning_rate)
    train_loss_df = init_loss_df()
    val_loss_df = init_loss_df()
    test_loss_df = init_loss_df()

    train_loader = loader_dict['train']
    val_loader = loader_dict['val']
    test_loader = loader_dict['test']

    train_path = join(abspath(model_path), exp_name, 'train')
    Path(train_path).mkdir(exist_ok=True, parents=True)
    val_path = join(abspath(model_path), exp_name, 'val')
    Path(val_path).mkdir(exist_ok=True, parents=True)
    test_path = join(abspath(model_path), exp_name, 'test')
    Path(test_path).mkdir(exist_ok=True, parents=True)

    if data_loader.num_train_imgs is not None:
        kld_weight = batch_size / data_loader.num_train_imgs
    else:
        kld_weight = None

    for epoch in range(epochs):
        model, optimizer, loss_dict = train(epoch, model, train_loader, optimizer, kld_weight, train_path, device)
        train_loss_df = update_loss_df(train_loss_df, loss_dict, epoch)
        val_loss_dict = validate(epoch, model, val_loader, kld_weight, val_path, device)
        val_loss_df = update_loss_df(val_loss_df, val_loss_dict, epoch)

    Path(join(abspath(model_path), exp_name)).mkdir(exist_ok=True, parents=True)
    model_path = join(abspath(model_path), exp_name,  f'{exp_name}.pt')
    torch.save(model, model_path)
    print(f'save trained model to {model_path}')

    test_loss_dict = test(model, test_loader, kld_weight, test_path, device)
    test_loss_df = update_loss_df(test_loss_df, test_loss_dict, -1)

    train_loss_df.to_csv(join(abspath(model_path), exp_name, 'train_loss.csv'))
    val_loss_df.to_csv(join(abspath(model_path), exp_name, 'val_loss.csv'))
    test_loss_df.to_csv(join(abspath(model_path), exp_name, 'test_loss.csv'))
    print(f'save train/val/test loss files under {join(abspath(model_path), exp_name)}')


def train(epoch: int, model, train_loader: DataLoader, optimizer: optim.Adam,
          kld_weight: float, train_path: str, device: torch.device):
    print(f'===========Epoch: {epoch:>3}')
    loss_df = init_loss_df()
    time_start = time.time()

    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    for i, (data1, data2) in enumerate(train_loader):
        data1 = data1.to(device)
        data2 = data2.to(device)

        data2_name = 'labels' if isinstance(model, CelebAVAE) else 'timestamp'
        recons, input, mu, log_var = model.forward({'data': data1, data2_name: data2})

        loss_dict = model.loss_function(recons, input, mu, log_var,
                                        kld_weight=kld_weight,
                                        device=device)
        loss_df = update_loss_df(loss_df, loss_dict, i)
        train_loss = loss_dict['loss']
        train_loss.backward()
        optimizer.step()

    with torch.no_grad():
        recons = recons.cpu().numpy()
        input = input.cpu().numpy()
        recons = recons[0, 0, :, :]
        input = input[0, 0, :, :]

        plt.figure()
        plt.plot(input[:, 0], input[:, 1],
                 'g', label='true trajectory')
        plt.plot(recons[:, 0], recons[:, 1], 'c',
                 label='learned trajectory')
        plt.legend()
        fig_path = join(train_path, f'{epoch}.png')
        plt.savefig(fig_path, dpi=500)
        print(f'Saved visualization figure at {fig_path}')

    time_end = time.time()

    loss_dict = get_mean_loss_dict(loss_df)
    print_loss_dict(loss_dict, break_line=False)

    print(f'Took {(time_end - time_start):.4f} sec')
    return model, optimizer, loss_dict


def validate(epoch: int, model, val_loader: DataLoader, kdl_weight: float, val_path: str, device: torch.device):
    loss_df = init_loss_df()

    model.eval()

    with torch.no_grad():
        for i, (data1, data2) in enumerate(val_loader):
            data1 = data1.to(device)
            data2 = data2.to(device)

            data2_name = 'labels' if isinstance(model, CelebAVAE) else 'timestamp'
            recons, input, mu, log_var = model.forward({'data': data1, data2_name: data2})
            loss_dict = model.loss_function(recons, input, mu, log_var,
                                            kld_weight=kdl_weight,
                                            device=device)
            loss_df = update_loss_df(loss_df, loss_dict, i)

        print(f'=========Validate() ', end='')
        loss_dict = get_mean_loss_dict(loss_df)
        print_loss_dict(loss_dict, break_line=False)

        recons = recons.cpu().numpy()
        input = input.cpu().numpy()
        recons = recons[0, 0, :, :]
        input = input[0, 0, :, :]

        plt.figure()
        plt.plot(input[:, 0], input[:, 1],
                 'g', label='true trajectory')
        plt.plot(recons[:, 0], recons[:, 1], 'c',
                 label='learned trajectory')
        plt.legend()
        fig_path = join(val_path, f'{epoch}.png')
        plt.savefig(fig_path, dpi=500)
        print(f'Saved visualization figure at {fig_path}')

    return loss_dict


def test(model, test_loader: DataLoader, kdl_weight: float,
         test_path: str, device: torch.device):
    loss_df = init_loss_df()

    model.eval()

    with torch.no_grad():
        for i, (data1, data2) in enumerate(test_loader):
            data1 = data1.to(device)
            data2 = data2.to(device)

            data2_name = 'labels' if isinstance(model, CelebAVAE) else 'timestamp'
            recons, input, mu, log_var = model.forward({'data': data1, data2_name: data2})
            loss_dict = model.loss_function(recons, input, mu, log_var,
                                            kld_weight=kdl_weight,
                                            device=device)
            loss_df = update_loss_df(loss_df, loss_dict, i)

        print(f'=========Test() ', end='')
        loss_dict = get_mean_loss_dict(loss_df)
        print_loss_dict(loss_dict, break_line=False)

        recons = recons.cpu().numpy()
        input = input.cpu().numpy()
        recons = recons[0, 0, :, :]
        input = input[0, 0, :, :]

        plt.figure()
        plt.plot(input[:, 0], input[:, 1],
                 'g', label='true trajectory')
        plt.plot(recons[:, 0], recons[:, 1], 'c',
                 label='learned trajectory')
        plt.legend()
        fig_path = join(test_path, f'test.png')
        plt.savefig(fig_path, dpi=500)
        print(f'Saved visualization figure at {fig_path}')

    return loss_dict


def celeba_main(device: torch.device, **kwargs):
    model = CelebAVAE(in_channels=kwargs['model_params']['in_channels'],
                      latent_dim=kwargs['model_params']['latent_dim'],
                      hidden_dims=None)

    data_loader = CelebALoader(batch_size=kwargs['exp_params']['batch_size'],
                               img_size=kwargs['exp_params']['img_size'],
                               data_path=kwargs['exp_params']['data_path'])

    experiment(model=model,
               data_loader=data_loader,
               device=device,
               batch_size=kwargs['exp_params']['batch_size'],
               learning_rate=kwargs['exp_params']['LR'],
               epochs=kwargs['exp_params']['epochs'],
               model_path=kwargs['exp_params']['model_path'],
               exp_name=kwargs['exp_params']['exp_name'])


def latent_ode_main(device: torch.device, **kwargs):
    model = LatentVAE(batch_size=kwargs['exp_params']['batch_size'],
                      latent_dim=kwargs['model_params']['latent_dim'],
                      n_rnn_hidden=kwargs['model_params']['rnn_hidden_dim'],
                      n_spiral=kwargs['data_params']['n_spiral'],
                      n_channel=kwargs['data_params']['n_channel'],
                      n_dec_hidden=kwargs['model_params']['dec_hidden_dim'],
                      n_hidden=kwargs['model_params']['hidden_dim'],
                      noise_std=kwargs['data_params']['noise_std'],
                      device=device)

    data_loader = LatentODELoader(batch_size=kwargs['exp_params']['batch_size'],
                                  n_frames=kwargs['data_params']['n_frames'],
                                  data_path=kwargs['exp_params']['data_path'],
                                  n_spiral=kwargs['data_params']['n_spiral'],
                                  n_total=kwargs['data_params']['n_total'],
                                  noise_std=kwargs['data_params']['noise_std'])

    experiment(model=model,
               data_loader=data_loader,
               device=device,
               batch_size=kwargs['exp_params']['batch_size'],
               learning_rate=kwargs['exp_params']['LR'],
               epochs=kwargs['exp_params']['epochs'],
               model_path=kwargs['exp_params']['model_path'],
               exp_name=kwargs['exp_params']['exp_name'])


if __name__ == '__main__':
    config = parse_args()
    manual_seed(**config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'set device on {device}')

    # celeba_main(device=device, **config)
    latent_ode_main(device=device, **config)
