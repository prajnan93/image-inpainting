import sys

sys.argv = [""]
del sys

# Training arguments
train_args = {}
train_args["--baseroot"] = "../../inpainting/dataset/Places/img_set"
train_args["--save_path"] = "./models"
train_args["--sample_path"] = ("./samples",)
train_args["--gpu_ids"] = "0,1"
train_args["--gan_type"] = "WGAN"
train_args["--cudnn_benchmark"] = True
train_args["--checkpoint_interval"] = 1
train_args["--multi_gpu"] = True
train_args["--load_name"] = ""
train_args["--epochs"] = 41
train_args["--batch_size"] = 2
train_args["--lr_g"] = 1e-4
train_args["--lr_d"] = 1e-4
train_args["--lambda_l1"] = 10
train_args["--lambda_perceptual"] = 10
train_args["--lambda_gan"] = 1
train_args["--lr_decrease_epoch"] = 10
train_args["--lr_decrease_factor"] = 0.5
train_args["--num_workers"] = 8
train_args["--in_channels"] = 4
train_args["--out_channels"] = 3
train_args["--latent_channels"] = 64
train_args["--pad_type"] = "zero"
train_args["--activation"] = "elu"
train_args["--norm"] = "none"
train_args["--init_type"] = "kaiming"
train_args["--init_gain"] = 0.02
train_args["--imgsize"] = 256
train_args["--mask_type"] = "free_form"
train_args["--margin"] = 10
train_args["--mask_num"] = 20
train_args["--bbox_shape"] = 30
train_args["--max_angle"] = 4
train_args["--max_len"] = 40
train_args["--max_width"] = 10

import argparse

import torch

from inpaint.core.discriminator import PatchDiscriminator
from inpaint.core.generator import GatedGenerator
from inpaint.utils import random_ff_mask

img = torch.randn(1, 3, 256, 256)
mask = torch.randn(1, 1, 256, 256)
# mask = random_ff_mask(shape=(256, 256))


def create_cfg():
    parser = argparse.ArgumentParser()
    for key, val in train_args.items():
        parser.add_argument(key, default=val)
        # print("--" + key, val)
    cfg = parser.parse_args()
    return cfg


cfg = create_cfg()

# def test_Generator():
#     generator = GatedGenerator(cfg)
#     coarse_output, refine_output = generator(img, mask)
#     assert coarse_output == img.shape
#     assert refine_output == img.shape
#     del generator


def test_Discriminator():
    discriminator = PatchDiscriminator(cfg)
    output = discriminator(img, mask)
    assert output.shape == (1, 256, 8, 8)
    del discriminator
