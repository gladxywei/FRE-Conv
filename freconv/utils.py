"""Small numerical utility subset required by the public training/evaluation paths."""

from typing import Union

import numpy as np
import torch


def normalize(image: Union[torch.Tensor, np.ndarray], by_band=False, band_dim: int = -1):
    out = torch.zeros_like(image)
    if by_band:
        mins, maxs = [], []
        for band in range(image.shape[band_dim]):
            sl = [slice(None)] * image.ndim
            sl[band_dim] = band
            sl = tuple(sl)
            max_y = image[sl].max()
            min_y = image[sl].min()
            maxs.append(max_y)
            mins.append(min_y)
            out[sl] = (image[sl] - min_y) / (max_y - min_y)
        min_y, max_y = tuple(mins), tuple(maxs)
    else:
        max_y, min_y = image.max(), image.min()
        out = (image - min_y) / (max_y - min_y)
    return out, {"mins": min_y, "maxs": max_y}


def normalize_w_consts(image, consts, by_band=True, band_dim=-1):
    out = torch.zeros_like(image)
    mins, maxs = consts["mins"], consts["maxs"]
    if by_band:
        for band in range(image.shape[band_dim]):
            sl = [slice(None)] * image.ndim
            sl[band_dim] = band
            sl = tuple(sl)
            out[sl] = (image[sl] - mins[band]) / (maxs[band] - mins[band])
    else:
        out = (image - mins) / (maxs - mins)
    return out


def peak_snr(img1, img2, bandwise=False, band_dim=-1):
    if not bandwise:
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(img2.max() / torch.sqrt(mse))
    out = torch.zeros(img1.shape[band_dim], device=img1.device)
    for band in range(img1.shape[band_dim]):
        sl = [slice(None)] * img1.ndim
        sl[band_dim] = band
        sl = tuple(sl)
        mse = torch.mean((img1[sl] - img2[sl]) ** 2)
        out[band] = 20 * torch.log10(img2[sl].max() / torch.sqrt(mse))
    return out


def sam(noise, reference):
    if len(noise.shape) == 1:
        numerator = torch.dot(noise, reference)
        denominator = torch.linalg.norm(noise) * torch.linalg.norm(reference)
    else:
        numerator = torch.sum(noise * reference, dim=-1)
        denominator = torch.linalg.norm(noise, dim=-1) * torch.linalg.norm(reference, dim=-1)
    eps = torch.finfo(denominator.dtype).eps
    return torch.arccos(numerator / (denominator + eps))
