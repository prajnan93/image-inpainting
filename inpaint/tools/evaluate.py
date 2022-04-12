import argparse
import os
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
from tests import test_models

# from inpaint.core.modules import PerceptualNet


class Evaluate:
    def __init__(self):
        args = {}
        args["--in_channels"] = 4
        args["--out_channels"] = 3
        args["--latent_channels"] = 64
        args["--pad_type"] = "zero"
        args["--activation"] = "elu"
        args["--norm_d"] = "none"
        args["--norm_g"] = "batch"
        args["--init_type"] = "kaiming"
        args["--init_gain"] = 0.02
        args["--use_cuda"] = False

    def create_cfg(self):
        parser = argparse.ArgumentParser()
        for key, val in self.args.items():
            parser.add_argument(key, default=val)
            # print("--" + key, val)
        cfg = parser.parse_args()
        return cfg

    def _create_mask(self, img):
        B, C, H, W = img.shape
        mask = torch.empty(B, 1, H, W)  # .cuda()

        # set the same masks for each batch
        for i in range(B):
            if self.cfg.mask_type.lower() == "free_form":
                mask[i] = torch.from_numpy(
                    random_ff_mask(
                        shape=(H, W),
                        max_angle=self.cfg.max_angle,
                        max_len=self.cfg.max_len,
                        max_width=self.cfg.max_width,
                        times=self.cfg.mask_num,
                    ).astype(np.float32)
                )
            else:
                mask[i] = torch.from_numpy(
                    random_bbox_mask(
                        shape=(H, W),
                        margin=self.cfg.margin,
                        bbox_shape=self.cfg.bbox_shape,
                        times=self.cfg.mask_num,
                    ).astype(np.float32)
                )

        mask = mask.to(self.device)
        return mask

    def evaluate(self, checkpoints):
        # Set models to eval state for validation
        checkpointslist = os.listdir(checkpoints)
        print(checkpointslist)
        cfg = self.create_cfg()
        generator = GatedGenerator(cfg)
        generator.eval()
        val_ds = PlacesDataset(
            path_dir=cfg.val_ds_dir,
            transform_config=("to_tensor", "center_crop", "norm"),
            crop_size=cfg.crop_size,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=cfg.num_workers,
        )
        val_iter = iter(val_loader)
        ssim_vals = []
        psnr_vals = []

        ssim_avgs = []
        psnr_avgs = []
        with torch.no_grad():
            for gen in checkpointslist:
                gen_model = torch.load(gen)
                gen_model_state_dict = gen_model["generator_state_dict"]
                generator.load_state_dict(gen_model_state_dict)
                for step in range(cfg.val_steps):
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
                psnr_avgs.append(psnr_vals.mean())
                ssim_avgs.append(ssim_vals.mean())

        fig = plt.figure()
        plt.plot(psnr_avgs)
        fig.suptitle("PSNR", fontsize=20)
        plt.xlabel("epochs", fontsize=18)
        plt.ylabel("PSNR", fontsize=16)
        fig.savefig("Network psnr.jpg")

        fig2 = plt.figure()
        plt.plot(ssim_avgs)
        fig2.suptitle("SSIM", fontsize=20)
        plt.xlabel("epochs", fontsize=18)
        plt.ylabel("SSIM", fontsize=16)
        fig2.savefig("Network ssim.jpg")


eval = Evaluate()
eval.evaluate("/Users/aadit/D/cs6140/project/FinalProject/checkpoints/exp1/")
