import math
import os

# import imgcrop
import random
from time import sleep

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from IPython.display import display  # to display images
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

# import utils
from inpaint.utils import *
from inpaint.utils import utils

ALLMASKTYPES = ["single_bbox", "bbox", "free_form"]

SEED = 1


class PlacesDataset(Dataset):
    def __init__(self, cfg):
        #         assert cfg.mask_type in ALLMASKTYPES

        self.cfg = cfg
        self.imglist = get_files(cfg.path_dir)  # utils.

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        # image
        global SEED
        img = cv2.imread(self.imglist[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # set the different image size for each batch (data augmentation)
        if index % self.cfg.batch_size == 0:
            SEED += 2
        img, height, width = self.random_crop(img, SEED)

        img = (
            torch.from_numpy(img.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .contiguous()
        )
        #         mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
        #         mask = self.random_mask()[0]
        return img

    def random_crop(self, img, seed):
        width_list = [256, 320, 400, 480]
        height_list = [256, 320, 400, 480]
        random.seed(seed)
        width = random.choice(width_list)
        random.seed(seed + 1)
        height = random.choice(height_list)

        max_x = img.shape[1] - width
        max_y = img.shape[0] - height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop = img[y : y + height, x : x + width]

        return crop, height, width
