"""Dataset loaders for ICVL and prepared Chikusei arrays."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms

from .. import utils
from . import transforms as hsi_transforms


def get_test_name(datadir, sort_files=True):
    files = list(Path(datadir).glob("*.npy"))
    if sort_files:
        files.sort()
    return [path.stem for path in files]


class ICVLDataset(Dataset):
    def __init__(self, datadir, crop_size=(512, 512), target_transform=None,
                 common_transforms=None, transform=None, val=False, band_norm=True):
        super().__init__()
        self.files = sorted(Path(datadir).glob("*.npy"))
        if not self.files:
            raise RuntimeError(f"No .npy cubes found in {datadir}")
        if dist.is_initialized():
            random.shuffle(self.files)
        self.loadfrom = tuple(torch.tensor(np.load(path), dtype=torch.float32) for path in self.files)
        self.band_norm = band_norm
        if val:
            self.base_transforms = tv_transforms.CenterCrop(crop_size)
            self.length = len(self.files)
        else:
            self.base_transforms = tv_transforms.Compose([
                tv_transforms.RandomCrop(crop_size),
                hsi_transforms.RandomReversal(p=0.5),
                hsi_transforms.RandChoice([
                    hsi_transforms.RandRot90Transform(),
                    tv_transforms.RandomVerticalFlip(p=0.9),
                    tv_transforms.RandomHorizontalFlip(p=0.9),
                ], combos=True),
            ])
            self.length = len(self.files) * 500
        self.target_transform = target_transform
        self.common_transforms = common_transforms
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = self.base_transforms(self.loadfrom[index % len(self.files)].unsqueeze(0))
        if self.common_transforms is not None:
            image = self.common_transforms(image)
        target = image.clone().detach().float()
        image = image.float()
        image, consts = utils.normalize(image, by_band=self.band_norm, band_dim=-3)
        target = utils.normalize_w_consts(target, consts, self.band_norm, -3)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class ICVLTestDataset(Dataset):
    def __init__(self, datadir, crop_size=(512, 512), target_transform=None,
                 common_transforms=None, transform=None, band_norm=True,
                 scene_id=-1, sort_files=True):
        super().__init__()
        files = list(Path(datadir).glob("*.npy"))
        if sort_files:
            files.sort()
        if dist.is_initialized():
            random.shuffle(files)
        selected = files if scene_id == -1 else files[scene_id:scene_id + 1]
        if not selected:
            raise RuntimeError(f"No matching .npy cubes found in {datadir}")
        self.loadfrom = tuple(torch.tensor(np.load(path), dtype=torch.float32) for path in selected)
        self.base_transforms = tv_transforms.CenterCrop(crop_size)
        self.band_norm = band_norm
        self.target_transform = target_transform
        self.common_transforms = common_transforms
        self.transform = transform

    def __len__(self):
        return len(self.loadfrom)

    def __getitem__(self, index):
        image = self.base_transforms(self.loadfrom[index].unsqueeze(0))
        if self.common_transforms is not None:
            image = self.common_transforms(image)
        target = image.clone().detach().float()
        image = image.float()
        image, consts = utils.normalize(image, by_band=self.band_norm, band_dim=-3)
        target = utils.normalize_w_consts(target, consts, self.band_norm, -3)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
