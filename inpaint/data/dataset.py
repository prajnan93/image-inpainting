import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from inpaint.utils import get_files


class BaseDataset(Dataset):
    """
    Base Dataset class that can be inherited to read from various datasets
    and perform data augmentations.

    """

    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def transform_initialize(
        self, crop_size, config=("random_crop", "to_tensor", "norm")
    ):
        """
        Initialize the transformation oprs and create transform function for img
        """
        self.transforms_oprs = {}
        self.transforms_oprs["hflip"] = transforms.RandomHorizontalFlip(0.5)
        self.transforms_oprs["vflip"] = transforms.RandomVerticalFlip(0.5)
        self.transforms_oprs["random_crop"] = transforms.RandomCrop(crop_size)
        self.transforms_oprs["to_tensor"] = transforms.ToTensor()
        self.transforms_oprs["norm"] = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        self.transforms_oprs["resize"] = transforms.Resize(crop_size)
        self.transforms_oprs["center_crop"] = transforms.CenterCrop(crop_size)
        self.transforms_oprs["rdresizecrop"] = transforms.RandomResizedCrop(
            crop_size, scale=(0.7, 1.0), ratio=(1, 1), interpolation=2
        )

        self.transforms_fun = transforms.Compose(
            [self.transforms_oprs[name] for name in config]
        )


class PlacesDataset(BaseDataset):
    """
    Class to read Places 365 Dataset (http://places2.csail.mit.edu/download.html).

    Params
    ------
    path_dir: str
        root director for the train, val or test split of Places 365 dataset.
    transform_config: tuple, default: ("to_tensor", "random_crop", "norm")
        data augmentation/transformation operations
    crop_size: tuple, default: (256, 256)
        crop size of the image
    """

    def __init__(
        self,
        path_dir,
        transform_config=("to_tensor", "random_crop", "norm"),
        crop_size=(256, 256),
    ):
        self.crop_size = crop_size
        self.imglist = get_files(path_dir)
        self.transform_initialize(crop_size=crop_size, config=transform_config)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        img = cv2.imread(self.imglist[index])

        if img.shape[0] < self.crop_size[0] or img.shape[1] < self.crop_size[1]:
            scale_percent = 150  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transforms_fun(img)
        return img
