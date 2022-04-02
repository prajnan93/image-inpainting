import torch

from inpaint.core.discriminator import PatchDiscriminator
from inpaint.core.generator import GatedGenerator
from inpaint.utils import random_ff_mask

img = torch.randn(1, 3, 256, 256)
mask = random_ff_mask(shape=(256, 256))

# create argparse cfg instance
# configure argparse cfg with network parameters


def test_Generator():
    pass


def test_Discriminator():
    pass
