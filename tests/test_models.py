import sys

sys.argv = [""]
del sys

import argparse

import numpy as np
import torch

from inpaint.core.discriminator import PatchDiscriminator
from inpaint.core.generator import GatedGenerator
from inpaint.utils import random_ff_mask

# Model arguments
args = {}
args["--in_channels"] = 4
args["--out_channels"] = 3
args["--latent_channels"] = 64
args["--pad_type"] = "zero"
args["--activation"] = "elu"
args["--norm"] = "none"
args["--init_type"] = "kaiming"
args["--init_gain"] = 0.02
args["--use_cuda"] = False

img = torch.randn(1, 3, 256, 256)
mask = random_ff_mask(shape=(256, 256)).astype(np.float32)
mask = torch.from_numpy(mask)
mask = mask[None, :]


def create_cfg():
    parser = argparse.ArgumentParser()
    for key, val in args.items():
        parser.add_argument(key, default=val)
        # print("--" + key, val)
    cfg = parser.parse_args()
    return cfg


cfg = create_cfg()


def test_Generator():
    generator = GatedGenerator(cfg)
    coarse_output, refine_output = generator(img, mask)
    assert coarse_output.shape == img.shape
    assert refine_output.shape == img.shape
    del generator


def test_Discriminator():
    discriminator = PatchDiscriminator(cfg)
    output = discriminator(img, mask)
    assert output.shape == (1, 256, 8, 8)
    del discriminator
