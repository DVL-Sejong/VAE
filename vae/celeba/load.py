from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from os.path import abspath


class CelebALoader:

    def __init__(self, batch_size: int, img_size: int, data_path: str):
        self.batch_size = batch_size
        self.img_size = img_size
        self.data_path = data_path

        self._setup()

    def _setup(self):
        Path(abspath(self.data_path)).mkdir(exist_ok=True, parents=True)
        transform = self.data_transforms()

        train_dataset = CelebA(root=self.data_path, split='train', transform=transform, download=False)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.num_train_imgs = len(train_dataset)

        val_dataset = CelebA(root=self.data_path, split='valid', transform=transform, download=False)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.num_val_imgs = len(val_dataset)

        test_dataset = CelebA(root=self.data_path, split='test', transform=transform, download=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.num_test_imgs = len(test_dataset)

    def get_data_loader(self):
        return {'train': self.train_loader, 'val': self.val_loader, 'test': self.test_loader}

    def data_transforms(self):
        SetRange = transforms.Lambda(lambda x: 2 * x - 1.)

        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(148),
                                        transforms.Resize(self.img_size),
                                        transforms.ToTensor(),
                                        SetRange])
        return transform
