import torch
import torch.nn as nn
from .. import FCNN_3D_XY_new as fn

__all__ = ['BasicConv3d', 'BasicDeConv3d', 'BasicUpsampleConv3d', 'UpsampleConv3d', 'HeadConv3d', 'TailConv3d']

_smooth = True
_inscale = 0.1


class UpsampleConv3d(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(
        self, tranNum, in_channels, out_channels, kernel_size, stride, padding, bias=True, upsample=None, smooth=_smooth, inscale=_inscale
    ):
        super(UpsampleConv3d, self).__init__()
        self.upsample = upsample
        
        if upsample:
            self.upsample_layer = torch.nn.Upsample(
                scale_factor=upsample, mode="trilinear", align_corners=True
            )


        self.conv3d = fn.Fconv_3D_PCA(
                sizeP = kernel_size,
                inNum = in_channels,
                outNum = out_channels,
                tranNum = tranNum,
                inP = None,
                padding = padding,
                stride = stride,
                ifIni = 0,
                bias = bias,
                Smooth=smooth,
                iniScale=inscale,
            )
    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.conv3d(x_in)
        return out


class BasicConv3d(nn.Sequential):
    def __init__(self, tranNum, in_channels, channels, k=3, s=1, p=1, ifIni=0, bias=False, bn=True, is_2d=False, cat=False, smooth=_smooth, inscale=_inscale):
        super(BasicConv3d, self).__init__()
        if bn:
            self.add_module("bn", fn.F_BN_3D(in_channels,tranNum=int((tranNum-1)*(1-ifIni) + 1)))
        self.add_module("conv", fn.Fconv_3D_PCA(
                sizeP = k,
                inNum = in_channels,
                outNum = channels,
                tranNum = tranNum,
                inP = None,
                padding = p,
                stride = s,
                ifIni = ifIni,
                bias = bias,
                Smooth=smooth,
                iniScale=inscale,
            )
        )

class HeadConv3d(nn.Sequential):
    def __init__(self, tranNum, in_channels, channels, k=3, s=1, p=1, bias=False, bn=True, ifIni=1,is_2d=False, cat=False, smooth=_smooth):
        super(HeadConv3d, self).__init__()
        if bn:
            self.add_module("bn", fn.F_BN_3D(in_channels,tranNum=int((tranNum-1)*(1-ifIni) + 1)))
        self.add_module("conv", fn.Fconv_3D_PCA(
                sizeP = k,
                inNum = in_channels,
                outNum = channels,
                tranNum = tranNum,
                inP = None,
                padding = p,
                stride = s,
                ifIni=1,
                bias = bias,
                Smooth=smooth,
            )
        )

class TailConv3d(nn.Sequential):
    def __init__(self, tranNum, in_channels, channels, k=3, s=1, p=1, bias=False, bn=True, is_2d=False, cat=False, smooth=_smooth):
        super(TailConv3d, self).__init__()
        if bn:
            self.add_module("bn", fn.F_BN_3D(in_channels,tranNum=tranNum))
        self.add_module("conv", fn.Fconv_3D_PCA_out(
                sizeP = k,
                inNum = in_channels,
                outNum = channels,
                tranNum = tranNum,
                inP = None,
                padding = p,
                bias = bias,
                Smooth=smooth,
            )
        )


class BasicDeConv3d(nn.Sequential):
    def __init__(self, tranNum, in_channels, channels, k=3, s=1, p=1, ifIni=0, bias=False, bn=True, smooth=_smooth, out=False, inscale=_inscale):
        super(BasicDeConv3d, self).__init__()
        if bn:
            self.add_module("bn", fn.F_BN_3D(in_channels,tranNum=int((tranNum-1)*(1-ifIni) + 1)))
        if not out:
            self.add_module("deconv", fn.Fconv_3D_PCA(
                    sizeP = k,
                    inNum = in_channels,
                    outNum = channels,
                    tranNum = tranNum,
                    inP = None,
                    padding = p,
                    stride = s,
                    ifIni = ifIni,
                    bias = bias,
                    Smooth=smooth,
                    iniScale=inscale,
                )
            )
        else:
            self.add_module("deconv", fn.Fconv_3D_PCA_out(
                    sizeP = k,
                    inNum = in_channels,
                    outNum = channels,
                    tranNum = tranNum,
                    inP = None,
                    padding = p,
                    bias = bias,
                    Smooth=smooth,
                    iniScale=inscale,
                )
            )


class BasicUpsampleConv3d(nn.Sequential):
    def __init__(self, tranNum, in_channels, channels, k=3, s=1, p=1, ifIni=0, upsample=(1, 2, 2), bn=True, smooth=_smooth):
        super(BasicUpsampleConv3d, self).__init__()
        if bn:
            self.add_module("bn", fn.F_BN_3D(in_channels,tranNum=int((tranNum-1)*(1-ifIni) + 1)))
        self.add_module(
            "upsample_conv",
            UpsampleConv3d(tranNum, in_channels, channels, k, s, p, bias=False, upsample=upsample,smooth=smooth),
        )
