# Remove unused import

import os
import random

# import matplotlib.pyplot as plt
import numpy as np
import torch

from inpaint.utils import psnr, random_bbox_mask, random_ff_mask, ssim


def flipCoin():
    f = random.random()
    return True if f < 0.5 else False


class Evaluate:
    # Evaluate args: generator and val_loader
    # Single instance of GatedGenerator should be be initialized in examples/evaluate.ipynb
    def __init__(self, generator, val_loader, val_steps):
        # remove cfg
        self.generator = generator
        self.val_loader = val_loader
        #chage it cuda on windows
        self.device = torch.device('cpu')

        self.val_steps = val_steps

    def _create_mask(self, img):
        flip = flipCoin()
        B, C, H, W = img.shape
        mask = torch.empty(B, 1, H, W) #.cuda()
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
        # Set models to eval state for validation

        val_iter = iter(self.val_loader)

        ssim_avgs = []
        psnr_avgs = []

        with torch.no_grad():
            self.generator.eval()
            # for gen in checkpointslist:
            ssim_vals = []
            psnr_vals = []

            # models = torch.load(gen)
            # gen_model_state_dict = models["generator_state_dict"]
            # generator.load_state_dict(gen_model_state_dict)
            #
            # generator.to(self.device)
            # generator.eval()

            for step in range(self.val_steps):
                try:
                    img = next(val_iter)
                except:
                    val_iter = iter(self.val_loader)
                    img = next(val_iter)

                img = img.to(self.device)
                B, C, H, W = img.shape

                mask = self._create_mask(img)

                # Generate fake pixel values for the given mask and real images
                coarse_out, refine_out = self.generator(img, mask)

                coarse_out_wholeimg = img * (1 - mask) + coarse_out * mask
                refine_out_wholeimg = img * (1 - mask) + refine_out * mask

                psnr_val = psnr(refine_out_wholeimg, img)
                ssim_val = ssim(refine_out_wholeimg, img)
                psnr_vals.append(psnr_val)
                ssim_vals.append(ssim_val)
            psnr_vals_mean = sum(psnr_vals) / len(psnr_vals)
            ssim_vals_mean = sum(ssim_vals) / len(ssim_vals)
        return psnr_vals_mean, ssim_vals_mean

        # Evaluate should simplt return the avg psnr and avg_ssim
        # Let's not plot any figures.
        # fig = plt.figure()
        # plt.plot(psnr_avgs)
        # fig.suptitle("PSNR", fontsize=20)
        # plt.xlabel("epochs", fontsize=18)
        # plt.ylabel("PSNR", fontsize=16)
        # # fig.savefig("Network psnr.jpg")
        #
        # fig2 = plt.figure()
        # plt.plot(ssim_avgs)
        # fig2.suptitle("SSIM", fontsize=20)
        # plt.xlabel("epochs", fontsize=18)
        # plt.ylabel("SSIM", fontsize=16)
        # fig2.savefig("Network ssim.jpg")
