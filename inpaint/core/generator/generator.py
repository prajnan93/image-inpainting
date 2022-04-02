import torch
import torch.nn as nn
import torch.nn.init as init

from inpaint.core.modules import GatedConv2d, TransposeGatedConv2d


# -----------------------------------------------
#                   Generator
# -----------------------------------------------
# Input: masked image + mask
# Output: filled image
class GatedGenerator(nn.Module):
    def __init__(self, cfg):
        super(GatedGenerator, self).__init__()
        self.coarse = nn.Sequential(
            # encoder
            GatedConv2d(
                in_channels=cfg.in_channels,
                out_channels=cfg.latent_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            # Bottleneck
            GatedConv2d(
                in_channles=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=8,
                dilation=8,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=16,
                dilation=16,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            # decoder
            TransposeGatedConv2d(
                in_channnels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channnels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            TransposeGatedConv2d(
                in_channnels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channnels=cfg.latent_channels,
                out_channels=cfg.latent_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels // 2,
                out_channels=cfg.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation="none",
                norm=cfg.norm,
            ),
            nn.Tanh(),
        )

        self.refine_conv = nn.Sequential(
            GatedConv2d(
                in_channels=cfg.in_channels,
                out_channels=cfg.latent_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels,
                out_channels=cfg.latent_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=8,
                dilation=8,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=16,
                dilation=16,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
        )
        self.refine_atten_1 = nn.Sequential(
            GatedConv2d(
                in_channels=cfg.in_channels,
                out_channels=cfg.latent_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels,
                out_channels=cfg.latent_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation="relu",
                norm=cfg.norm,
            ),
        )
        self.refine_atten_2 = nn.Sequential(
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
        )
        self.refine_combine = nn.Sequential(
            GatedConv2d(
                in_channels=cfg.latent_channels * 8,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            TransposeGatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            TransposeGatedConv2d(
                in_channels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels,
                out_channels=cfg.latent_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels // 2,
                out_channels=cfg.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation="none",
                norm=cfg.norm,
            ),
            nn.Tanh(),
        )
        self.context_attention = ContextualAttention(
            ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True
        )

    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # Coarse
        first_masked_img = img * (1 - mask) + mask
        first_in = torch.cat((first_masked_img, mask), dim=1)  # in: [B, 4, H, W]
        first_out = self.coarse(first_in)  # out: [B, 3, H, W]
        first_out = nn.functional.interpolate(first_out, (img.shape[2], img.shape[3]))
        # Refinement
        second_masked_img = img * (1 - mask) + first_out * mask
        second_in = torch.cat([second_masked_img, mask], dim=1)
        refine_conv = self.refine_conv(second_in)
        refine_atten = self.refine_atten_1(second_in)
        mask_s = nn.functional.interpolate(
            mask, (refine_atten.shape[2], refine_atten.shape[3])
        )
        refine_atten = self.context_attention(refine_atten, refine_atten, mask_s)
        refine_atten = self.refine_atten_2(refine_atten)
        second_out = torch.cat([refine_conv, refine_atten], dim=1)
        second_out = self.refine_combine(second_out)
        second_out = nn.functional.interpolate(second_out, (img.shape[2], img.shape[3]))
        return first_out, second_out
