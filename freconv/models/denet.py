"""DeNet3D baseline and FRE-Conv variants used in the paper."""

import torch.nn as nn
import torch.nn.init as init

from . import FCNN_3D_XY as fn

class DeNetLite3D2(nn.Module):
    def __init__(self, in_channels=1, kernel_size=3, init_weights=True):
        super(DeNetLite3D2, self).__init__()
        m_head = []
        m_body = []
        m_tail = []
        out_ch = in_channels
        out_channels = 32  # 64
        # add CR
        m_head.append(
            nn.Conv3d(  # nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=True,
            )
        )
        m_head.append(nn.ReLU(inplace=True))
        # add CBR1 to CBR16
        in_channels = out_channels
        for out_channels, dilation in cfg3d:
            if dilation == 1:
                padding = 1
            elif dilation == 2:
                padding = 2
            m_body.append(
                nn.Conv3d(  # nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                )
            )
            m_body.append(
                nn.BatchNorm3d(num_features=out_channels)
            )  # BatchNorm2d(num_features=out_channels))
            m_body.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # add C
        m_tail.append(
            nn.Conv3d(  # nn.Conv2d(
                in_channels=32,  # 64
                out_channels=out_ch,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res = res + x

        out = self.tail(res)

        return out  # y - out

    def _initialize_weights(self):
        print("===> Start initializing weights")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

"""
each tuple unit: [out_channels, dilation]
"""
cfg3d = (
    [2 ** 5, 1],  # [64, 1],
    [2 ** 5, 1],  # [64, 1],
    [2 ** 5, 1],  # [64, 1],
    [2 ** 6, 1],  # [128, 1],
    [2 ** 6, 1],  # [128, 1],
    [2 ** 6, 1],  # [128, 1],
    [2 ** 7, 1],  # [256, 2],
    [2 ** 6, 1],  # [128, 1],
    [2 ** 5, 1],  # [64, 1],
    [2 ** 5, 1],  # [64, 1],
    [2 ** 5, 1],  # [64, 1],
)

class FDeNetLite3D2(nn.Module):
    def __init__(self, tranNum=8, in_channels=1, kernel_size=5, smooth = False,init_weights=True):
        super(FDeNetLite3D2, self).__init__()
        head = []
        body = []
        tail = []
        out_ch = in_channels
        out_channels = int(32/tranNum)  # 64/tranNum
        smooth = smooth
        # add CR
        head.append(
            fn.Fconv_3D_PCA(
                sizeP = kernel_size,
                inNum = in_channels,
                outNum = out_channels,
                tranNum = tranNum,
                inP = None,
                padding = (kernel_size-1)//2,
                ifIni = 1,
                bias = True,
                Smooth = smooth,
            )
        )
        head.append(nn.ReLU(inplace=True))
        # add CBR1 to CBR16
        in_channels = out_channels
        for out_channels, dilation in cfg3d:
            out_channels = int(out_channels/tranNum)
            if dilation == 1:
                padding = 2
            elif dilation == 2:
                padding = 3
            body.append(
                fn.Fconv_3D_PCA(
                sizeP = kernel_size,
                inNum = in_channels,
                outNum = out_channels,
                tranNum = tranNum,
                inP = None,
                padding = (kernel_size-1)//2,
                ifIni = 0,
                bias = False,
                Smooth = smooth,
            )
                
            )
            body.append(
                fn.F_BN_3D(out_channels,tranNum=tranNum)
            )  # BatchNorm2d(num_features=out_channels))
            body.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # add C
        tail.append(
            fn.Fconv_3D_PCA_out(
                sizeP = kernel_size,
                inNum = int(32/tranNum),
                outNum = out_ch,
                tranNum = tranNum,
                inP = None,
                padding = (kernel_size-1)//2,
                bias = False,
                Smooth = smooth,
            )
        )
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res = res + x

        out = self.tail(res)

        return out  # y - out
