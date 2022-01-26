from os.path import join

import pandas as pd
import numpy as np


class LossLogger:

    def __init__(self, path_dict: dict):
        self.path_dict = path_dict
        self.train_loss_df = init_loss_df()
        self.val_loss_df = init_loss_df()
        self.test_loss_df = init_loss_df()
        self.temp_loss_df = None
        self.temp_loss_dict = None

    def init_temp_loss(self):
        self.temp_loss_df = init_loss_df()
        self.temp_loss_dict = None

    def update_temp_loss(self, loss_dict:dict, epoch: int):
        self.temp_loss_df = update_loss_df(self.temp_loss_df, loss_dict, epoch)

    def mean_temp_loss(self):
        self.temp_loss_dict = get_mean_loss_dict(self.temp_loss_df)
        return self.temp_loss_dict

    def print_loss_by_epoch(self):
        train_loss = self.train_loss_df.iloc[-1]['loss']
        val_loss = self.val_loss_df.iloc[-1]['loss']
        print(f'Train loss: {train_loss}, val loss: {val_loss}, ', end ='')

    def print_test_loss(self):
        print_loss_dict(self.val_loss_df.iloc[-1])

    def update_loss(self, epoch: int, name: str):
        if name == 'train':
            self.train_loss_df = update_loss_df(self.train_loss_df, self.temp_loss_dict, epoch)
            self.save_loss('train')
        elif name == 'val':
            self.val_loss_df = update_loss_df(self.val_loss_df, self.temp_loss_dict, epoch)
            self.save_loss('val')
        elif name == 'test':
            self.test_loss_df = update_loss_df(self.test_loss_df, self.temp_loss_dict, epoch)
            self.save_loss('test')

    def save_loss(self, name: str):
        exp_path = self.path_dict['exp']

        if name == 'train':
            self.train_loss_df.to_csv(join(exp_path, 'train_loss.csv'), index=None)
        elif name == 'val':
            self.val_loss_df.to_csv(join(exp_path, 'val_loss.csv'), index=None)
        elif name == 'test':
            self.test_loss_df.to_csv(join(exp_path, 'test_loss.csv'), index=None)

def int_exp_loss():
    train_loss_df = init_loss_df()
    val_loss_df = init_loss_df()
    test_loss_df = init_loss_df()
    loss_dict = {'train': train_loss_df, 'val': val_loss_df, 'test': test_loss_df}
    return loss_dict


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


def print_loss_dict(loss_dict: dict):
    loss = loss_dict['loss'].item()
    reconstruction_loss = loss_dict['reconstruction_loss'].item()
    kld = loss_dict['kl-divergence'].item()

    print(f'loss: {loss:4.8f}, '
          f'reconstruction loss: {reconstruction_loss:4.8f}, '
          f'kl-divergence: {kld:4.8f}')
