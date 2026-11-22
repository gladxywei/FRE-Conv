"""3D MemNet baseline retained for the paper comparison."""

import torch
import torch.nn as nn


class MemNet2(nn.Module):
    def __init__(self, in_channels, channels, num_memblock, num_resblock, conv3d=False, hyres=False):
        super().__init__()
        if not conv3d:
            raise ValueError("The clean release only supports the paper's 3D MemNet configuration.")
        if hyres:
            raise ValueError("HyRes preprocessing is outside the clean release scope.")
        in_channels = 1
        self.feature_extractor = BNReLUConv(in_channels, channels)
        self.reconstructor = BNReLUConv(channels, in_channels)
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels, num_resblock, i + 1) for i in range(num_memblock)]
        )
        self.freeze_bn = True
        self.freeze_bn_affine = True
        self.conv3d = True
        self.hyres = None

    def forward(self, x):
        residual = x
        out = self.feature_extractor(x)
        out_temp = out
        ys = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out = out + out_temp
        out = self.reconstructor(out)
        _ = residual
        return out


class MemoryBlock(nn.Module):
    def __init__(self, channels, num_resblock, num_memblock):
        super().__init__()
        self.recursive_unit = nn.ModuleList([ResidualBlock(channels) for _ in range(num_resblock)])
        self.gate_unit = BNReLUConv((num_resblock + num_memblock) * channels, channels, 1, 1, 0)

    def forward(self, x, ys):
        xs = []
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)
        gate_out = self.gate_unit(torch.cat(xs + ys, 1))
        ys.append(gate_out)
        return gate_out


class ResidualBlock(nn.Module):
    def __init__(self, channels, k=3, s=1, p=1):
        super().__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, k, s, p)
        self.relu_conv2 = BNReLUConv(channels, channels, k, s, p)

    def forward(self, x):
        return self.relu_conv2(self.relu_conv1(x)) + x


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super().__init__()
        self.add_module("bn", nn.BatchNorm3d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=inplace))
        self.add_module("conv", nn.Conv3d(in_channels, channels, k, s, p, bias=False))
