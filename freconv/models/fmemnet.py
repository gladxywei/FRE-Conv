"""3D FRE-Conv MemNet variants retained for checkpoint compatibility."""

import torch
import torch.nn as nn

from . import FCNN_3D_XY as fn

_RS = 1


class FMemNet2(nn.Module):
    def __init__(self, tranNum, in_channels, channels, num_memblock, num_resblock,
                 conv3d=False, hyres=False, smooth=False, kernel_size=3):
        super().__init__()
        if not conv3d:
            raise ValueError("The clean release only supports the paper's 3D FMemNet configuration.")
        if hyres:
            raise ValueError("HyRes preprocessing is outside the clean release scope.")
        in_channels = 1
        self.tranNum = tranNum
        channels = int(channels / self.tranNum)
        padding = (kernel_size - 1) // 2
        self.feature_extractor = BNReLUConv(
            tranNum, in_channels, channels, ifIni=1, k=kernel_size, p=padding, smooth=smooth
        )
        self.reconstructor = BNReLUConv(
            tranNum, channels, in_channels, out=True, k=kernel_size, p=padding, smooth=smooth
        )
        self.dense_memory = nn.ModuleList([
            MemoryBlock(tranNum, channels, num_resblock, i + 1, k=kernel_size, p=padding, smooth=smooth)
            for i in range(num_memblock)
        ])
        self.freeze_bn = True
        self.freeze_bn_affine = True
        self.conv3d = True
        self.hyres = None

    def forward(self, x):
        out = self.feature_extractor(x)
        out_temp = out
        ys = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out = out + out_temp
        return self.reconstructor(out)


class MemoryBlock(nn.Module):
    def __init__(self, tranNum, channels, num_resblock, num_memblock, k=3, p=1, smooth=True):
        super().__init__()
        self.recursive_unit = nn.ModuleList([
            ResidualBlock(tranNum, channels, k=k, p=p, smooth=smooth) for _ in range(num_resblock)
        ])
        self.gate_unit = BNReLUConv(
            tranNum, (num_resblock + num_memblock) * channels, channels, k=1, p=0, smooth=smooth
        )

    def forward(self, x, ys):
        xs = []
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)
        gate_out = self.gate_unit(torch.cat(xs + ys, 1))
        ys.append(gate_out)
        return gate_out


class ResidualBlock(nn.Module):
    def __init__(self, tranNum, channels, k=3, s=1, p=1, smooth=True):
        super().__init__()
        self.relu_conv1 = BNReLUConv(tranNum, channels, channels, k=k, s=s, p=p, smooth=smooth)
        self.relu_conv2 = BNReLUConv(tranNum, channels, channels, k=k, s=s, p=p, smooth=smooth)

    def forward(self, x):
        return _RS * self.relu_conv2(self.relu_conv1(x)) + x


class BNReLUConv(nn.Sequential):
    def __init__(self, tranNum, in_channels, out_channels, ifIni=0, out=False,
                 k=3, s=1, p=1, inplace=True, smooth=True):
        super().__init__()
        self.add_module("bn", fn.F_BN_3D(in_channels, tranNum=1 if ifIni == 1 else tranNum))
        self.add_module("relu", nn.ReLU(inplace=inplace))
        if not out and k != 1:
            conv = fn.Fconv_3D_PCA(
                sizeP=k, inNum=in_channels, outNum=out_channels, tranNum=tranNum,
                inP=None, padding=p, stride=s, ifIni=ifIni, bias=False, Smooth=smooth,
            )
        elif out and k != 1:
            conv = fn.Fconv_3D_PCA_out(
                sizeP=k, inNum=in_channels, outNum=out_channels, tranNum=tranNum,
                inP=None, padding=p, bias=False, Smooth=smooth,
            )
        else:
            conv = fn.Fconv_1X1X1(
                inNum=in_channels, outNum=out_channels, tranNum=tranNum,
                ifIni=ifIni, bias=False, Smooth=smooth, iniScale=1.0,
            )
        self.add_module("conv", conv)
