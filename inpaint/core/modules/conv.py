import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F


class Conv2dLayer(nn.Module):
    """
    Basic 2D Convolution Block.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        pad_type="zero",
        activation="elu",
        norm="none",
        sn=False,
    ):
        super(Conv2dLayer, self).__init__()

        # Initialize the padding scheme
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == "layer":
            self.norm = nn.LayerNorm(out_channels)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = nn.utils.spectral_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=0,
                    dilation=dilation,
                )
            )
        else:
            self.conv2d = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=0,
                dilation=dilation,
            )

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class TransposeConv2dLayer(nn.Module):
    """
    Transpose 2D Convolution.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        pad_type="zero",
        activation="lrelu",
        norm="none",
        sn=False,
        scale_factor=2,
    ):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            pad_type,
            activation,
            norm,
            sn,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv2d(x)
        return x


class GatedConv2d(nn.Module):
    """
    Gated 2D Convolution.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        pad_type="reflect",
        activation="elu",
        norm="none",
        sn=False,
    ):

        super(GatedConv2d, self).__init__()

        # Initialize the padding scheme
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == "layer":
            self.norm = nn.LayerNorm(out_channels)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = nn.utils.spectral_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=0,
                    dilation=dilation,
                )
            )
            self.mask_conv2d = nn.utils.spectral_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=0,
                    dilation=dilation,
                )
            )
        else:
            self.conv2d = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=0,
                dilation=dilation,
            )
            self.mask_conv2d = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=0,
                dilation=dilation,
            )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        if self.activation:
            conv = self.activation(conv)
        x = conv * gated_mask
        return x


class TransposeGatedConv2d(nn.Module):
    """
    Transpose Gated 2D Convolution.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        pad_type="zero",
        activation="lrelu",
        norm="none",
        sn=True,
        scale_factor=2,
    ):

        super(TransposeGatedConv2d, self).__init__()

        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            pad_type,
            activation,
            norm,
            sn,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.gated_conv2d(x)
        return x
