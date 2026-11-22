# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:05:23 2023

@author: admin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import FCNN_3D_XY_new as fn


""" Utility block """

_smooth = True
_inscale = 0.1
_rs = 0.5
_k =3
_p = 1


def _kernel_padding(kernel_size):
    return (kernel_size - 1) // 2


class UpsampleConv3d(torch.nn.Module):
    """ UpsampleConvLayer
    """

    def __init__(self,tranNum, in_channels, out_channels, kernel_size, stride, padding, bias=True, upsample=None, group=1):
        super(UpsampleConv3d, self).__init__()
        self.upsample_layer = nn.Upsample(scale_factor=upsample, mode='trilinear', align_corners=True)
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
                Smooth=_smooth,
                iniScale=_inscale,
            )

    def forward(self, x):
        x = self.upsample_layer(x)
        x = self.conv3d(x)
        return x


class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP)
    """

    def __init__(self,tranNum, channel, bias=True):
        super().__init__()
        self.w_1 = fn.Fconv_1X1X1(channel, channel, tranNum=tranNum, ifIni=0, bias=bias, Smooth=_smooth, iniScale=_inscale)
        self.w_2 = fn.Fconv_1X1X1(channel, channel, tranNum=tranNum, ifIni=0, bias=bias, Smooth=_smooth, iniScale=_inscale)

    def forward(self, x):
        return self.w_2(torch.tanh(self.w_1(x)))


""" The proposed blocks
"""


class PSCA(nn.Module):
    """ Progressive Spectral Channel Attention (PSCA) 
    """

    def __init__(self,tranNum, d_model, d_ff):
        super().__init__()


        self.w_1 = fn.Fconv_1X1X1(d_model, d_ff, tranNum=tranNum, ifIni=0, bias=False, Smooth=_smooth, iniScale=_inscale)
        self.w_2 = fn.Fconv_1X1X1(d_ff, d_model, tranNum=tranNum, ifIni=0, bias=False, Smooth=_smooth, iniScale=_inscale)
        self.w_3 = fn.Fconv_1X1X1(d_model, d_model, tranNum=tranNum, ifIni=0, bias=False, Smooth=_smooth, iniScale=_inscale, zeros=True)
        
    def forward(self, x):
        x = self.w_3(x) * x + x
        x = self.w_1(x)
        x = F.gelu(x)
        x = self.w_2(x)
        return x


class ASC(nn.Module):
    """ Attentive Skip Connection
    """

    def __init__(self,tranNum, channel, kernel_size=_k):
        super().__init__()
        padding = _kernel_padding(kernel_size)
        
        self.weight = nn.Sequential(
            fn.Fconv_1X1X1(channel * 2, channel, tranNum=tranNum, ifIni=0, Smooth=_smooth, iniScale=_inscale),
            nn.LeakyReLU(),
            fn.Fconv_3D_PCA(
                sizeP = kernel_size,
                inNum = channel,
                outNum = channel,
                tranNum = tranNum,
                inP = None,
                padding = padding,
                stride = 1,
                ifIni = 0,
                Smooth=_smooth,
                iniScale=_inscale,
            ),
            nn.Sigmoid()
        )

    def forward(self, x, y):
 
        w = self.weight(torch.cat([x, y], dim=1))
        out = (1 - w) * x + w * y
        return out


class MHRSA(nn.Module):
    """ Multi-Head Recurrent Spectral Attention
    """

    def __init__(self,tranNum, channels, multi_head=True, ffn=True):
        super().__init__()
        self.tranNum = tranNum
        self.channels = channels
        self.multi_head = multi_head
        self.ffn = ffn

        if ffn:
            self.ffn1 = MLP(tranNum,channels)
            self.ffn2 = MLP(tranNum,channels)

    def _conv_step(self, inputs):
        if self.ffn:
            Z = self.ffn1(inputs).tanh()
            F = self.ffn2(inputs).sigmoid()
        else:
            Z, F = inputs.split(split_size=self.channels*self.tranNum, dim=1)
            Z, F = Z.tanh(), F.sigmoid()
        return Z, F

    def _rnn_step(self, z, f, h):
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, reverse=False):
        Z, F = self._conv_step(inputs)

        if self.multi_head:
            Z1, Z2 = Z.split(self.channels*self.tranNum // 2, 1)
            Z2 = torch.flip(Z2, [2])
            Z = torch.cat([Z1, Z2], dim=1)

            F1, F2 = F.split(self.channels*self.tranNum // 2, 1)
            F2 = torch.flip(F2, [2])
            F = torch.cat([F1, F2], dim=1)

        h = None
        h_time = []

        if not reverse:
            for _, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for _, (z, f) in enumerate((zip(
                reversed(Z.split(1, 2)), reversed(F.split(1, 2))
            ))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)

        y = torch.cat(h_time, dim=2)

        if self.multi_head:
            y1, y2 = y.split(self.channels*self.tranNum // 2, 1)
            y2 = torch.flip(y2, [2])
            y = torch.cat([y1, y2], dim=1)

        return y


class MAB(nn.Module):
    def __init__(self,tranNum, conv_layer, channels, multi_head=True, ffn=True):
        super().__init__()
        self.conv = conv_layer
        self.inter_sa = MHRSA(tranNum,channels, multi_head=multi_head, ffn=ffn)
        self.intra_sa = PSCA(tranNum,channels, channels * 2)

    def forward(self, x, reverse=False):
        x = self.conv(x)
        x = self.inter_sa(x, reverse=reverse)
        x = self.intra_sa(x)
        return x


""" Encoder-Decoder
"""


def PlainMAB(tranNum,in_ch, out_ch, bias=False,ifIni=0, kernel_size=_k):
    padding = _kernel_padding(kernel_size)
    return MAB(
        tranNum,
        fn.Fconv_3D_PCA(
            sizeP = kernel_size,
            inNum = in_ch,
            outNum = out_ch,
            tranNum = tranNum,
            inP = None,
            padding = padding,
            stride = 1,
            ifIni = ifIni,
            bias=bias,
            Smooth=_smooth,
            iniScale=_inscale,
        ),
        out_ch
        )


def DownMAB(tranNum, in_ch, out_ch, bias=False, kernel_size=_k):
    padding = _kernel_padding(kernel_size)
    return MAB(
        tranNum,
        fn.Fconv_3D_PCA(
            sizeP = kernel_size,
            inNum = in_ch,
            outNum = out_ch,
            tranNum = tranNum,
            inP = None,
            padding = padding,
            stride = (1,2,2),
            ifIni = 0,
            bias=bias,
            Smooth=_smooth,
            iniScale=_inscale,
        ),
        out_ch
        )


def UpMAB(tranNum, in_ch, out_ch, bias=False, kernel_size=_k):
    padding = _kernel_padding(kernel_size)
    return MAB(
        tranNum,
        UpsampleConv3d(tranNum, in_ch, out_ch, kernel_size, 1, padding, bias=bias, upsample=(1, 2, 2)),
        out_ch
        )


class Encoder(nn.Module):
    def __init__(self, tranNum, channels, num_half_layer, sample_idx, kernel_size=_k):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_half_layer):
            if i not in sample_idx:
                encoder_layer = PlainMAB(tranNum, channels, channels, kernel_size=kernel_size)
            else:
                encoder_layer = DownMAB(tranNum, channels, 2 * channels, kernel_size=kernel_size)
                channels *= 2
            self.layers.append(encoder_layer)

    def forward(self, x, xs, reverse=False):
        num_half_layer = len(self.layers)
        for i in range(num_half_layer - 1):
            x = self.layers[i](x, reverse)
            reverse = not reverse
            xs.append(x)
        x = self.layers[-1](x, reverse)
        reverse = not reverse
        return x


class Decoder(nn.Module):
    def __init__(self, tranNum, channels, num_half_layer, sample_idx, Fusion=None, kernel_size=_k):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.enable_fusion = Fusion is not None

        if self.enable_fusion:
            self.fusions = nn.ModuleList()
            ch = channels
            for i in reversed(range(1,num_half_layer)):
                fusion_layer = Fusion(tranNum, ch, kernel_size=kernel_size)
                if i-1 in sample_idx:
                    ch //= 2
                self.fusions.append(fusion_layer)

        for i in reversed(range(num_half_layer)):
            if i not in sample_idx:
                decoder_layer = PlainMAB(tranNum, channels, channels, kernel_size=kernel_size)
            else:
                decoder_layer = UpMAB(tranNum, channels, channels // 2, kernel_size=kernel_size)
                channels //= 2
            self.layers.append(decoder_layer)

    def forward(self, x, xs, reverse=False):
        num_half_layer = len(self.layers)
        x = self.layers[0](x, reverse)
        reverse = not reverse
        for i in range(1, num_half_layer):
            if self.enable_fusion:
                x = self.fusions[i-1](_rs * x, xs.pop())
            else:
                x = _rs * x + xs.pop()
            x = self.layers[i](x, reverse)
            reverse = not reverse
        return x


class FMAN(nn.Module):
    def __init__(self, tranNum, in_channels, channels, num_half_layer, sample_idx, Fusion=ASC, kernel_size=_k):
        super().__init__()
        padding = _kernel_padding(kernel_size)
        channels = channels//tranNum
        self.head = PlainMAB(tranNum, in_channels, channels, ifIni=1, kernel_size=kernel_size)
        self.encoder = Encoder(tranNum, channels, num_half_layer, sample_idx, kernel_size=kernel_size)
        self.decoder = Decoder(
            tranNum,
            channels * (2**len(sample_idx)),
            num_half_layer,
            sample_idx,
            Fusion=Fusion,
            kernel_size=kernel_size,
        )
        self.tail = fn.Fconv_3D_PCA_out(
                sizeP = kernel_size,
                inNum = channels,
                outNum = in_channels,
                tranNum = tranNum,
                inP = None,
                padding = padding,
                bias = True,
                Smooth=_smooth,
                iniScale=_inscale
            )

    def forward(self, x):
        xs = [x]
        out = self.head(xs[0])
        xs.append(out)
        reverse = True
        out = self.encoder(out, xs, reverse)
        out = self.decoder(out, xs, reverse)
        out = _rs * out + xs.pop()
        out = self.tail(out)
        out = out + xs.pop()
        return out


""" Models
"""
