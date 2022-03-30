import torch
from torch import nn
from torch.nn import functional as F

from inpaint.utils import (
    extract_image_patches,
    reduce_mean,
    reduce_std,
    reduce_sum,
    same_padding,
)


class ContextualAttention(nn.Module):
    """

    Params
    ------
    ksize: Kernel size for contextual attention.
    stride: Stride for extracting patches from b.
    rate: Dilation for matching.
    softmax_scale: Scaled softmax for attention.
    """

    def __init__(
        self,
        ksize=3,
        stride=1,
        rate=1,
        fuse_k=3,
        softmax_scale=10,
        fuse=True,
        use_cuda=True,
        device_ids=0,
    ):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = use_cuda
        self.device_ids = device_ids

    def forward(self, f, b, mask=None):
        """
        Contextual attention layer implementation.
        Contextual attention is first introduced in publication:

        Generative Image Inpainting with Contextual Attention, Yu et al.

        Params
        ------
        f: Input feature to match (foreground).
        b: Input feature for match (background).
        mask: Input mask for b, indicating patches not available.

        Returns
        -------
            torch.tensor: output
        """
        # get shapes
        raw_int_fs = list(f.size())  # b*c*h*w
        raw_int_bs = list(b.size())  # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(
            b,
            ksizes=[kernel, kernel],
            strides=[self.rate * self.stride, self.rate * self.stride],
            rates=[1, 1],
            padding="same",
        )  # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L] [4, 192, 4, 4, 1024]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1.0 / self.rate, mode="nearest")
        b = F.interpolate(b, scale_factor=1.0 / self.rate, mode="nearest")
        int_fs = list(f.size())  # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(
            b,
            ksizes=[self.ksize, self.ksize],
            strides=[self.stride, self.stride],
            rates=[1, 1],
            padding="same",
        )
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        mask = F.interpolate(mask, scale_factor=1.0 / self.rate, mode="nearest")
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(
            mask,
            ksizes=[self.ksize, self.ksize],
            strides=[self.stride, self.stride],
            rates=[1, 1],
            padding="same",
        )

        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)  # m shape: [N, L, C, k, k]
        m = m[0]  # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True) == 0.0).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3)  # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale  # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda()

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            """
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            """
            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda()
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(
                reduce_sum(torch.pow(wi, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True)
            )
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(
                xi, [self.ksize, self.ksize], [1, 1], [1, 1]
            )  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(
                    1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3]
                )  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(
                    1, int_bs[2], int_bs[3], int_fs[2], int_fs[3]
                )  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(
                    1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3]
                )
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(
                1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3]
            )  # (B=1, C=32*32, H=32, W=32)
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

            if int_bs != int_fs:
                # Normalize the offset value to match foreground dimension
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] * int_bs[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat(
                [offset // int_fs[3], offset % int_fs[3]], dim=1
            )  # 1*2*H*W

            # deconv for patch pasting
            wi_center = raw_wi[0]
            # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            yi = (
                F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.0
            )  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs)

        return
