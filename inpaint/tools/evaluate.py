# Remove unused import

import os
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from inpaint.core.generator import GatedGenerator
from inpaint.data import PlacesDataset
from inpaint.utils import random_bbox_mask, random_ff_mask
from inpaint.utils.metrics import psnr, ssim


def flipCoin():
    f = random.random()
    return True if f < 0.5 else False


class Evaluate:
    # Evaluate args: generator and val_loader
    # Single instance of GatedGenerator should be be initialized in examples/evaluate.ipynb
    def __init__(self, cfg, val_loader):
        self.cfg = cfg
        self.val_loader = val_loader
        self.device = torch.device(cfg.device_id)

    def _create_mask(self, img):
        flip = flipCoin()
        B, C, H, W = img.shape
        mask = torch.empty(B, 1, H, W).cuda()
        # set the same masks for each batch
        for i in range(B):
            if flip == True:
                # Free form mask
                mask[i] = torch.from_numpy(
                    random_ff_mask(
                        shape=(H, W),
                        max_angle=4,
                        max_len=40,
                        max_width=10,
                        times=20,
                    ).astype(np.float32)
                )
            else:
                # Box mask
                mask[i] = torch.from_numpy(
                    random_bbox_mask(
                        shape=(H, W),
                        margin=(10, 10),
                        bbox_shape=(30, 30),
                        times=20,
                    ).astype(np.float32)
                )

        mask = mask.to(self.device)
        return mask

    def evaluate(self, checkpoints_dir):
        # Set models to eval state for validation
        checkpointslist = os.listdir(checkpoints_dir)

        for i in range(len(checkpointslist)):
            checkpointslist[i] = checkpoints_dir + "/" + checkpointslist[i]

        # print(checkpointslist)
        generator = GatedGenerator(self.cfg)

        val_iter = iter(self.val_loader)

        ssim_avgs = []
        psnr_avgs = []

        with torch.no_grad():
            for gen in checkpointslist:
                ssim_vals = []
                psnr_vals = []

                models = torch.load(gen)
                gen_model_state_dict = models["generator_state_dict"]
                generator.load_state_dict(gen_model_state_dict)

                generator.to(self.device)
                generator.eval()

                for step in range(self.cfg.val_steps):
                    try:
                        img = next(val_iter)
                    except:
                        val_iter = iter(self.val_loader)
                        img = next(val_iter)

                    img = img.to(self.device)
                    B, C, H, W = img.shape

                    mask = self._create_mask(img)

                    # Generate fake pixel values for the given mask and real images
                    coarse_out, refine_out = generator(img, mask)

                    coarse_out_wholeimg = img * (1 - mask) + coarse_out * mask
                    refine_out_wholeimg = img * (1 - mask) + refine_out * mask

                    psnr_val = psnr(refine_out_wholeimg, img)
                    ssim_val = ssim(refine_out_wholeimg, img)
                    psnr_vals.append(psnr_val)
                    ssim_vals.append(ssim_val)

                psnr_avgs.append(np.array(psnr_vals).mean())
                ssim_avgs.append(np.array(ssim_vals).mean())

        # Evaluate should simplt return the avg psnr and avg_ssim
        # Let's not plot any figures.
        fig = plt.figure()
        plt.plot(psnr_avgs)
        fig.suptitle("PSNR", fontsize=20)
        plt.xlabel("epochs", fontsize=18)
        plt.ylabel("PSNR", fontsize=16)
        # fig.savefig("Network psnr.jpg")

        fig2 = plt.figure()
        plt.plot(ssim_avgs)
        fig2.suptitle("SSIM", fontsize=20)
        plt.xlabel("epochs", fontsize=18)
        plt.ylabel("SSIM", fontsize=16)
        fig2.savefig("Network ssim.jpg")
