import torch
import torch.nn as nn
import torch.nn.init as init

from inpaint.core.modules import ContextualAttention, GatedConv2d, TransposeGatedConv2d


class GatedGenerator(nn.Module):
    """
    Generator model from Free-Form Image Inpainting with Gated Convolution.
    https://arxiv.org/abs/1806.03589v2

    """

    def __init__(self, cfg):
        super(GatedGenerator, self).__init__()
        self.init_type = "kaiming"
        self.init_gain = 0.02

        # Setup Coarse Network
        self.coarse = CoarseNetwork(cfg)

        # Setup Two branch Refinement Network with
        # Configurable Contextual Attention
        self.refine_conv = ConvRefinementNetwork(cfg)

        combine_channels = cfg.latent_channels * 4
        self.use_context_attn_layer = cfg.add_context_attention

        if self.use_context_attn_layer:
            combine_channels = cfg.latent_channels * 8
            self.refine_attn = AttentionRefinementNetwork(cfg)

        # Combine Refinement outputs
        self.refine_combine = CombineRefinement(cfg)

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
        """
        Forward pass of the Generator.

        Params
        ------
        img: torch.tensor
            a tensor representing the image of shape (N, 3, H, W)
        mask: torch.tensor
            a tensor representing the mask of shape (N, 1, H, W)


        Returns
        -------
        (torch.tensor, torch.tensor):
            a tuple of two tensors of shape (N, 3, H, W) representing the reconstructed coarse and refine image.
        """

        # Coarse
        first_masked_img = img * (1 - mask) + mask
        first_in = torch.cat((first_masked_img, mask), dim=1)

        first_out = self.coarse(first_in)
        first_out = nn.functional.interpolate(first_out, (img.shape[2], img.shape[3]))

        # Refinement
        second_masked_img = img * (1 - mask) + first_out * mask
        second_in = torch.cat([second_masked_img, mask], dim=1)
        refine_conv = self.refine_conv(second_in)

        second_out = refine_conv

        if self.use_context_attn_layer:
            refine_atten = self.refine_attn(second_in, mask)
            second_out = torch.cat([refine_conv, refine_atten], dim=1)

        second_out = self.refine_combine(second_out)
        second_out = nn.functional.interpolate(second_out, (img.shape[2], img.shape[3]))

        return first_out, second_out


class CoarseNetwork(nn.Module):
    def __init__(self, cfg):
        super(CoarseNetwork, self).__init__()
        self.coarse_network = nn.Sequential(
            # encoder
            GatedConv2d(
                in_channels=cfg.in_channels,
                out_channels=cfg.latent_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            # Bottleneck
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
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
                norm=cfg.norm_g,
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
                norm=cfg.norm_g,
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
                norm=cfg.norm_g,
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
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            # decoder
            TransposeGatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            TransposeGatedConv2d(
                in_channels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels,
                out_channels=cfg.latent_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels // 2,
                out_channels=cfg.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation="none",
                norm=cfg.norm_g,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.coarse_network(x)


class ConvRefinementNetwork(nn.Module):
    def __init__(self, cfg):
        super(ConvRefinementNetwork, self).__init__()
        self.refine_network = nn.Sequential(
            GatedConv2d(
                in_channels=cfg.in_channels,
                out_channels=cfg.latent_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels,
                out_channels=cfg.latent_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
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
                norm=cfg.norm_g,
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
                norm=cfg.norm_g,
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
                norm=cfg.norm_g,
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
                norm=cfg.norm_g,
            ),
        )

    def forward(self, x):
        return self.refine_network(x)


class AttentionRefinementNetwork(nn.Module):
    def __init__(self, cfg):
        super(AttentionRefinementNetwork, self).__init__()
        combine_channels = cfg.latent_channels * 4

        if cfg.add_context_attention:
            combine_channels = cfg.latent_channels * 8

            self.refine_atten_1 = nn.Sequential(
                GatedConv2d(
                    in_channels=cfg.in_channels,
                    out_channels=cfg.latent_channels,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    pad_type=cfg.pad_type,
                    activation=cfg.activation,
                    norm=cfg.norm_g,
                ),
                GatedConv2d(
                    in_channels=cfg.latent_channels,
                    out_channels=cfg.latent_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    pad_type=cfg.pad_type,
                    activation=cfg.activation,
                    norm=cfg.norm_g,
                ),
                GatedConv2d(
                    in_channels=cfg.latent_channels,
                    out_channels=cfg.latent_channels * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pad_type=cfg.pad_type,
                    activation=cfg.activation,
                    norm=cfg.norm_g,
                ),
                GatedConv2d(
                    in_channels=cfg.latent_channels * 2,
                    out_channels=cfg.latent_channels * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    pad_type=cfg.pad_type,
                    activation=cfg.activation,
                    norm=cfg.norm_g,
                ),
                GatedConv2d(
                    in_channels=cfg.latent_channels * 4,
                    out_channels=cfg.latent_channels * 4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pad_type=cfg.pad_type,
                    activation=cfg.activation,
                    norm=cfg.norm_g,
                ),
                GatedConv2d(
                    in_channels=cfg.latent_channels * 4,
                    out_channels=cfg.latent_channels * 4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pad_type=cfg.pad_type,
                    activation="relu",
                    norm=cfg.norm_g,
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
                    norm=cfg.norm_g,
                ),
                GatedConv2d(
                    in_channels=cfg.latent_channels * 4,
                    out_channels=cfg.latent_channels * 4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pad_type=cfg.pad_type,
                    activation=cfg.activation,
                    norm=cfg.norm_g,
                ),
            )
            self.context_attention = ContextualAttention(
                ksize=3,
                stride=1,
                rate=2,
                fuse_k=3,
                softmax_scale=10,
                fuse=True,
                use_cuda=cfg.use_cuda,
            )

    def forward(self, x, mask):
        refine_atten = self.refine_atten_1(x)

        mask_s = nn.functional.interpolate(
            mask, (refine_atten.shape[2], refine_atten.shape[3])
        )
        refine_atten = self.context_attention(refine_atten, refine_atten, mask_s)

        refine_atten = self.refine_atten_2(refine_atten)

        return refine_atten


class CombineRefinement(nn.Module):
    def __init__(self, cfg):
        super(CombineRefinement, self).__init__()

        combine_channels = cfg.latent_channels * 4
        if cfg.add_context_attention:
            combine_channels = cfg.latent_channels * 8

        self.network = nn.Sequential(
            GatedConv2d(
                in_channels=combine_channels,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            TransposeGatedConv2d(
                in_channels=cfg.latent_channels * 4,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            TransposeGatedConv2d(
                in_channels=cfg.latent_channels * 2,
                out_channels=cfg.latent_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels,
                out_channels=cfg.latent_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation=cfg.activation,
                norm=cfg.norm_g,
            ),
            GatedConv2d(
                in_channels=cfg.latent_channels // 2,
                out_channels=cfg.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=cfg.pad_type,
                activation="none",
                norm=cfg.norm_g,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)
