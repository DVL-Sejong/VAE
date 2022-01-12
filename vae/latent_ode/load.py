from torch.utils.data import Dataset, DataLoader

import numpy.random as npr
import numpy as np
import torch


class ODEDataset(Dataset):

    def __init__(self, trajectories: np.array, timestamp: np.array, n_frames: int):
        self.trajectories = trajectories
        self.timestamp = timestamp
        self.n_frames = n_frames

    def __len__(self):
        return self.trajectories.shape[1] - self.n_frames

    def __getitem__(self, idx):
        data = self.trajectories[:, idx:idx+self.n_frames, :]
        timestamp = self.timestamp[idx:idx+self.n_frames]

        data = torch.from_numpy(data).float()
        timestamp = torch.from_numpy(timestamp).float()

        return data, timestamp


class LatentODELoader:

    def __init__(self, batch_size: int, n_frames: int,
                 n_spiral: int, n_total: int, noise_std: float):
        self.batch_size = batch_size
        self.n_frames = n_frames

        self.n_spiral = n_spiral
        self.n_total = n_total
        self.noise_std = noise_std

        self._setup()
        self._set_data_loader()

    def _generate_spiral(self, start: float, stop: float,
                         a: float, b: float, clockwise: bool = True):
        timestamp = np.linspace(start, stop, num=self.n_total)

        if clockwise:
            zs = stop + 1. - timestamp
            rs = a + b * 50. / zs
            xs, ys = rs * np.cos(zs) - 5, rs * np.sin(zs)
        else:
            zs = timestamp
            rw = a + b * zs
            xs, ys = rw * np.cos(zs) + 5, rw * np.sin(zs)

        spiral = np.stack((xs, ys), axis=1)
        return spiral

    def _setup(self):
        start = 0.
        stop = 6 * np.pi

        a = 0
        b = .3

        trajectory = []
        for _ in range(self.n_spiral):
            # a, b = npr.uniform(), npr.uniform()
            spiral = self._generate_spiral(start, stop, a, b, bool(npr.rand() > .5))
            trajectory.append(spiral)

        trajectory = np.stack(trajectory, axis=0)
        timestamp = np.linspace(start, stop, num=self.n_total)

        self.trajectories = trajectory
        self.timestamp = timestamp

        len_train = int(self.trajectories.shape[1] * 0.6)
        len_val = int(self.trajectories.shape[1] * 0.2)
        self.train_trajectories = self.trajectories[:, :len_train, :]
        self.train_timestamp = self.timestamp[:len_train]
        self.val_trajectories = self.trajectories[:, len_train:len_train+len_val, :]
        self.val_timestamp = self.timestamp[len_train:len_train+len_val]
        self.test_trajectories = self.trajectories[:, len_train+len_val:, :]
        self.test_timestamp = self.timestamp[len_train+len_val:]\

        self.num_train_imgs = None

    def _set_data_loader(self):
        self._set_train_loader()
        self._set_val_loader()
        self._set_test_loader()

    def _set_train_loader(self):
        train_dataset = ODEDataset(self.train_trajectories, self.train_timestamp, self.n_frames)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def _set_val_loader(self):
        val_dataset = ODEDataset(self.val_trajectories, self.val_timestamp, self.n_frames)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def _set_test_loader(self):
        test_dataset = ODEDataset(self.test_trajectories, self.test_timestamp, self.n_frames)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def get_data_loader(self):
        return {'train': self.train_loader, 'val': self.val_loader, 'test': self.test_loader}
