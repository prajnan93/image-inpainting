import torch
import torch.nn as nn
import torch.nn.init as init

from inpaint.core.modules import Conv2dLayer


# -----------------------------------------------
#                  Discriminator
# -----------------------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 30 * 30
class PatchDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(PatchDiscriminator, self).__init__()
        self.init_type = "kaiming"
        self.init_gain = 0.02

        # Down sampling
        self.block1 = Conv2dLayer(
            in_channels=cfg.in_channels,
            out_channels=cfg.latent_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            pad_type=cfg.pad_type,
            activation=cfg.activation,
            norm=cfg.norm_d,
            sn=True,
        )
        self.block2 = Conv2dLayer(
            in_channels=cfg.latent_channels,
            out_channels=cfg.latent_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            pad_type=cfg.pad_type,
            activation=cfg.activation,
            norm=cfg.norm_d,
            sn=True,
        )
        self.block3 = Conv2dLayer(
            in_channels=cfg.latent_channels * 2,
            out_channels=cfg.latent_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            pad_type=cfg.pad_type,
            activation=cfg.activation,
            norm=cfg.norm_d,
            sn=True,
        )
        self.block4 = Conv2dLayer(
            in_channels=cfg.latent_channels * 4,
            out_channels=cfg.latent_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            pad_type=cfg.pad_type,
            activation=cfg.activation,
            norm=cfg.norm_d,
            sn=True,
        )
        self.block5 = Conv2dLayer(
            in_channels=cfg.latent_channels * 4,
            out_channels=cfg.latent_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            pad_type=cfg.pad_type,
            activation=cfg.activation,
            norm=cfg.norm_d,
            sn=True,
        )
        self.block6 = Conv2dLayer(
            in_channels=cfg.latent_channels * 4,
            out_channels=cfg.latent_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            pad_type=cfg.pad_type,
            activation="none",
            norm="none",
            sn=True,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and classname.find("Conv") != -1:
            if self.init_type == "normal":
                init.normal_(m.weight.data, 0.0, self.init_gain)
            elif self.init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=self.init_gain)
            elif self.init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif self.init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=self.init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % self.init_type
                )
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find("Linear") != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    def forward(self, img, mask):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = torch.cat((img, mask), 1)
        x = self.block1(x)  # out: [B, 64, 256, 256]
        x = self.block2(x)  # out: [B, 128, 128, 128]
        x = self.block3(x)  # out: [B, 256, 64, 64]
        x = self.block4(x)  # out: [B, 256, 32, 32]
        x = self.block5(x)  # out: [B, 256, 16, 16]
        x = self.block6(x)  # out: [B, 256, 8, 8]
        return x
