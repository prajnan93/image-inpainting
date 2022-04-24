import os
import random

import numpy as np
import torch

from inpaint.utils import psnr, random_bbox_mask, random_ff_mask, ssim


def flipCoin():
    f = random.random()
    return True if f < 0.5 else False


class Evaluate:
    """
    Perform evaluation for a given generator.

    Params
    ------
    generator: torch.nn.Module
        an instance of a generator
    val_loader: torch.utils.data.DataLoader
        the validation dataloader
    """

    def __init__(self, generator, val_loader):

        self.generator = generator
        self.val_loader = val_loader
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _create_mask(self, img):
        flip = flipCoin()
        B, C, H, W = img.shape
        mask = torch.empty(B, 1, H, W)

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

    def evaluate(self):
        """
        Performs evaluation and returns the average psnr and ssim metric
        for the given dataset.

        Returns
        -------
        avg_psnr, avg_ssim: (float32, float32)

        """
        with torch.no_grad():
            # for gen in checkpointslist:
            ssim_vals = []
            psnr_vals = []

            self.generator = self.generator.to(self.device)

            # Set models to eval state for validation
            self.generator.eval()

            for _, img in enumerate(self.val_loader):
                img = img.to(self.device)
                B, C, H, W = img.shape

                mask = self._create_mask(img)

                # Generate fake pixel values for the given mask and real images
                _, refine_out = self.generator(img, mask)

                refine_out_wholeimg = img * (1 - mask) + refine_out * mask

                psnr_val = psnr(refine_out_wholeimg, img)
                ssim_val = ssim(refine_out_wholeimg, img)
                psnr_vals.append(psnr_val)
                ssim_vals.append(ssim_val)

            psnr_vals_mean = sum(psnr_vals) / len(psnr_vals)
            ssim_vals_mean = sum(ssim_vals) / len(ssim_vals)

        return psnr_vals_mean, ssim_vals_mean
