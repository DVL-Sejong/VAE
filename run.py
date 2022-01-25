import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vae import CelebALoader, CelebAVAE, LatentODELoader, LatentVAE, LossLogger, Tensor
from os.path import abspath, join, isdir
from pathlib import Path

import numpy as np
import argparse
import yaml
import time
import re

from torch.utils.data import DataLoader
import torch.optim as optim
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/latent_ode_220112_1.yaml')

    args = parser.parse_args()
    filename = args.filename
    filename = re.sub('\s+', '', filename)

    with open(filename, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise FileNotFoundError(exc)


def manual_seed(**kwargs):
    seed = kwargs['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f'Manual seed to {seed}')


def set_device(**kwargs):
    device = torch.device(kwargs['model_params']['device'] if torch.cuda.is_available() else 'cpu')
    print(f'set device on {device}')

    return device


def set_model_paths(**kwargs):
    result_path = kwargs['exp_params']['result_path']
    exp_name = kwargs['exp_params']['exp_name']

    exp_path = join(abspath(result_path), exp_name)
    if isdir(exp_path):
        raise FileExistsError(f'{exp_path} already exists')

    path_dict = {'exp': exp_path,
                 'train': join(exp_path, 'train'),
                 'val': join(exp_path, 'val'),
                 'test': join(exp_path, 'test')}
    mkdir_paths(path_dict)

    return path_dict


def mkdir_paths(path_dict):
    for key, path in path_dict.items():
        Path(path).mkdir(exist_ok=True, parents=True)
        print(f'Make directory under {path}')


def plot_spirals(spiral_dict: dict, figpath: str = None):
    with torch.no_grad():
        plt.figure()
        for key, spiral in spiral_dict.items():
            linewidth = 1 if 'first' in key or 'second' in key else 3
            if isinstance(spiral, torch.Tensor):
                spiral = spiral.cpu().numpy()[0][0]
            plt.plot(spiral[:, 0], spiral[:, 1], label=f'spiral_{key}', linewidth=linewidth)
        plt.legend()

        if figpath is not None:
            plt.savefig(figpath)
            print(f'save spirals to {figpath}')

        plt.close()


def save_model(model, model_path: str):
    saving_path = join(model_path, f'model.pt')
    torch.save(model, saving_path)
    print(f'save trained model to {saving_path}')


def experiment(model, data_loader, device: torch.device, logger: LossLogger,
               batch_size: int, learning_rate: float, epochs: int, spiral_dict: dict = None):
    model.to(device)
    params = model.parameters()
    optimizer = optim.Adam(params, lr=learning_rate)

    loader_dict = data_loader.get_data_loader()

    if data_loader.num_train_imgs is not None:
        kld_weight = batch_size / data_loader.num_train_imgs
    else:
        kld_weight = None

    for epoch in range(epochs):
        print(f'===Epoch: {epoch:>4} ', end='')
        time_start = time.time()
        model, optimizer, logger = train(epoch, model, loader_dict['train'], optimizer, kld_weight, logger, device, spiral_dict)
        logger = validate(epoch, model, loader_dict['val'], kld_weight, logger, device, spiral_dict)
        logger.print_loss_by_epoch()

        time_end = time.time()
        print(f'Took {(time_end - time_start):.4f} sec')

    print()

    save_model(model, logger.path_dict['exp'])
    logger = test(model, loader_dict['test'], kld_weight, logger, device, spiral_dict)
    logger.print_test_loss()


def train(epoch: int, model, train_loader: DataLoader, optimizer: optim.Adam,
          kld_weight: float, logger: LossLogger, device: torch.device, spiral_dict: dict = None):
    plot_dict = dict(spiral_dict)

    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    logger.init_temp_loss()
    for i, (data1, data2) in enumerate(train_loader):
        data1 = data1.to(device)
        data2 = data2.to(device)

        data2_name = 'labels' if isinstance(model, CelebAVAE) else 'timestamp'
        recons, input, mu, log_var = model.forward({'data': data1, data2_name: data2})
        loss_dict = model.loss_function(recons, input, mu, log_var,
                                        kld_weight=kld_weight,
                                        device=device)
        logger.update_temp_loss(loss_dict, i)
        train_loss = loss_dict['loss']
        train_loss.backward()
        optimizer.step()

    if epoch % 30 == 0:
        fig_path = join(logger.path_dict['train'], f'{epoch}.png')
        plot_dict.update({'recons': recons, 'input': input})
        plot_spirals(plot_dict, fig_path)

    logger.mean_temp_loss()
    logger.update_loss(epoch, 'train')
    return model, optimizer, logger


def validate(epoch: int, model, val_loader: DataLoader, kdl_weight: float, logger: LossLogger, device: torch.device, spiral_dict: dict = None):
    plot_dict = dict(spiral_dict)

    model.eval()

    logger.init_temp_loss()
    with torch.no_grad():
        for i, (data1, data2) in enumerate(val_loader):
            data1 = data1.to(device)
            data2 = data2.to(device)

            data2_name = 'labels' if isinstance(model, CelebAVAE) else 'timestamp'
            recons, input, mu, log_var = model.forward({'data': data1, data2_name: data2})
            loss_dict = model.loss_function(recons, input, mu, log_var,
                                            kld_weight=kdl_weight,
                                            device=device)
            logger.update_temp_loss(loss_dict, i)

    if epoch % 30 == 0:
        fig_path = join(logger.path_dict['val'], f'{epoch}.png')
        plot_dict.update({'recons': recons, 'input': input})
        plot_spirals(plot_dict, fig_path)

    logger.mean_temp_loss()
    logger.update_loss(epoch, 'val')
    return logger


def test(model, test_loader: DataLoader, kdl_weight: float, logger: LossLogger, device: torch.device, spiral_dict: dict = None):
    plot_dict = dict(spiral_dict)

    model.eval()

    logger.init_temp_loss()
    with torch.no_grad():
        for i, (data1, data2) in enumerate(test_loader):
            data1 = data1.to(device)
            data2 = data2.to(device)

            data2_name = 'labels' if isinstance(model, CelebAVAE) else 'timestamp'
            recons, input, mu, log_var = model.forward({'data': data1, data2_name: data2})
            loss_dict = model.loss_function(recons, input, mu, log_var,
                                            kld_weight=kdl_weight,
                                            device=device)
            logger.update_temp_loss(loss_dict, i)

            fig_path = join(logger.path_dict['test'], f'test_{i:>03}.png')
            plot_dict.update({'recons': recons, 'input': input})
            plot_spirals(plot_dict, fig_path)

    logger.mean_temp_loss()
    logger.update_loss(-1, 'test')
    return logger


def celeba_main(device: torch.device, path_dict: dict, **kwargs):
    model = CelebAVAE(in_channels=kwargs['model_params']['in_channels'],
                      latent_dim=kwargs['model_params']['latent_dim'],
                      hidden_dims=None)

    data_loader = CelebALoader(batch_size=kwargs['exp_params']['batch_size'],
                               img_size=kwargs['exp_params']['img_size'],
                               data_path=kwargs['exp_params']['data_path'])

    logger = LossLogger(path_dict)

    experiment(model=model,
               data_loader=data_loader,
               device=device,
               logger=logger,
               batch_size=kwargs['exp_params']['batch_size'],
               learning_rate=kwargs['exp_params']['LR'],
               epochs=kwargs['exp_params']['epochs'])


def latent_ode_main(device: torch.device, path_dict: dict, **kwargs):
    model = LatentVAE(batch_size=kwargs['exp_params']['batch_size'],
                      latent_dim=kwargs['model_params']['latent_dim'],
                      n_rnn_hidden=kwargs['model_params']['rnn_hidden_dim'],
                      n_spiral=kwargs['data_params']['n_spiral'],
                      n_channel=kwargs['data_params']['n_channel'],
                      n_dec_hidden=kwargs['model_params']['dec_hidden_dim'],
                      n_hidden=kwargs['model_params']['hidden_dim'],
                      noise_std=kwargs['data_params']['noise_std'],
                      device=device).cuda()

    data_loader = LatentODELoader(batch_size=kwargs['exp_params']['batch_size'],
                                  n_frames=kwargs['data_params']['n_frames'],
                                  n_spiral=kwargs['data_params']['n_spiral'],
                                  n_total=kwargs['data_params']['n_total'],
                                  noise_std=kwargs['data_params']['noise_std'])

    first_spiral, second_spiral = data_loader.get_spirals()
    spiral_dict = {'first': first_spiral, 'second': second_spiral}

    logger = LossLogger(path_dict)

    experiment(model=model,
               data_loader=data_loader,
               device=device,
               logger=logger,
               batch_size=kwargs['exp_params']['batch_size'],
               learning_rate=kwargs['exp_params']['LR'],
               epochs=kwargs['exp_params']['epochs'],
               spiral_dict=spiral_dict)


if __name__ == '__main__':
    config = parse_args()
    manual_seed(**config)
    device = set_device(**config)
    path_dict = set_model_paths(**config)

    if config['model'] == 'celeba':
        celeba_main(device=device, path_dict=path_dict, **config)
    elif config['model'] == 'latent_ode':
        latent_ode_main(device=device, path_dict=path_dict, **config)
