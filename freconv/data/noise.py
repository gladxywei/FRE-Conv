"""Noise primitives used by the paper training and evaluation protocols."""

from typing import Callable, Dict, Iterable, Optional, Union

import numpy as np
import torch


def add_noise_std(
    image: torch.Tensor,
    sigma: Union[float, list[float]],
    scale_factor: float = 1,
    noise_type: str = "additive",
    iid: bool = True,
    band_dim: int = -1,
) -> torch.Tensor:
    """Add IID Gaussian, non-IID Gaussian, or Poisson noise."""
    if noise_type == "additive" and iid:
        return image + sigma / scale_factor * torch.randn_like(image)
    if noise_type == "additive" and not iid:
        if sigma is None:
            sigmas = torch.rand(image.shape[band_dim], dtype=image.dtype, device=image.device) * 60 + 10
        elif isinstance(sigma, list):
            sigmas = sigma
        else:
            sigmas = [10, 30, 50, 70]
        generated = torch.randn_like(image)
        for band in range(image.shape[band_dim]):
            index = np.random.randint(len(sigmas)) if sigma is not None else band
            if band_dim == -1:
                generated[:, :, :, band] *= sigmas[index] / scale_factor
            elif band_dim == -3:
                generated[:, band, :, :] *= sigmas[index] / scale_factor
            else:
                raise ValueError("band_dim must be -1 or -3")
        return image + generated
    if noise_type == "poisson":
        snr_set = torch.exp(
            15 * torch.log(torch.tensor(10, device=image.device, dtype=image.dtype)) / 10
        )
        if band_dim == -1:
            pixels = image.shape[-2] * image.shape[-3]
        elif band_dim == -3:
            pixels = image.shape[-1] * image.shape[-2]
        else:
            raise ValueError("band_dim must be -1 or -3")
        generated = torch.zeros((image.shape[band_dim], pixels), dtype=image.dtype, device=image.device)
        for band in range(image.shape[band_dim]):
            current = image[:, :, :, band].unsqueeze(0) if band_dim == -1 else image[:, band, :, :].unsqueeze(0)
            current[current <= 0] = 0
            factor = snr_set / ((current ** 2).sum() / current.sum())
            generated[band] = torch.poisson(factor * current).reshape(pixels)
        return image + generated.reshape(image.shape) / scale_factor
    raise ValueError(f"Unsupported noise type: {noise_type}")


def add_noise_on_bands(
    signal: torch.Tensor,
    bands: float,
    noise_fn: Callable,
    noise_fn_args: Optional[Dict] = None,
    band_dim: int = -1,
    inplace: bool = True,
) -> torch.Tensor:
    """Apply one corruption primitive to a random fraction of spectral bands."""
    noise_fn_args = noise_fn_args or {}
    band_count = signal.shape[band_dim]
    selected = torch.randperm(band_count, device=signal.device)
    if 0 < bands <= 1:
        bands = int(bands * band_count)
    selected = selected[:bands]
    output = signal if inplace else signal.clone()
    return noise_fn(output, selected, band_dim=band_dim, **noise_fn_args)


def add_noise_impulse(
    signal: torch.Tensor,
    bands: float,
    amounts: Iterable = (0.1, 0.3, 0.5, 0.7),
    salt_vs_pepper: float = 0.5,
    band_dim: int = -1,
    inplace: bool = True,
) -> torch.Tensor:
    return add_noise_on_bands(
        signal,
        bands,
        _add_noise_impulse,
        {"amounts": amounts, "salt_vs_pepper": salt_vs_pepper},
        band_dim,
        inplace,
    )


def _add_noise_impulse(signal, bands, amounts, salt_vs_pepper, band_dim):
    slices = [slice(None)] * signal.ndim
    band_amounts = [amounts[index.item()] for index in torch.randint(len(amounts), size=(len(bands),))]
    for band, amount in zip(bands, band_amounts):
        slices[band_dim] = band
        current = signal[slices]
        flipped = torch.from_numpy(
            np.random.choice([True, False], size=tuple(current.shape), p=[amount, 1 - amount])
        ).to(signal.device)
        salted = torch.from_numpy(
            np.random.choice([True, False], size=tuple(current.shape), p=[salt_vs_pepper, 1 - salt_vs_pepper])
        ).to(signal.device)
        current[flipped & salted] = 1
        current[flipped & ~salted] = 0
        signal[slices] = current
    return signal


def add_noise_stripe(
    signal: torch.Tensor,
    bands: float = 1 / 3,
    min_amount: float = 0.05,
    max_amount: float = 0.15,
    band_dim: int = -1,
    inplace: bool = True,
) -> torch.Tensor:
    return add_noise_on_bands(
        signal,
        bands,
        _add_noise_stripe,
        {"min_amount": min_amount, "max_amount": max_amount},
        band_dim,
        inplace,
    )


def _add_noise_stripe(signal, bands, min_amount, max_amount, band_dim):
    width_dim = -2 if band_dim == -1 else -1
    width = signal.shape[width_dim]
    counts = np.random.randint(int(min_amount * width), int(max_amount * width), len(bands))
    slices = [slice(None)] * signal.ndim
    for band, count in zip(bands, counts):
        locations = torch.randperm(width)[:count]
        slices[band_dim] = band
        slices[width_dim] = locations
        stripe = torch.rand_like(locations, dtype=signal.dtype, device=signal.device)
        signal[slices] -= stripe * 0.5 - 0.25
    return signal


def add_noise_deadline(
    signal: torch.Tensor,
    bands: float,
    min_amount: float = 0.05,
    max_amount: float = 0.15,
    band_dim: int = -1,
    inplace: bool = True,
) -> torch.Tensor:
    return add_noise_on_bands(
        signal,
        bands,
        _add_noise_deadline,
        {"min_amount": min_amount, "max_amount": max_amount},
        band_dim,
        inplace,
    )


def _add_noise_deadline(signal, bands, min_amount, max_amount, band_dim):
    width_dim = -2 if band_dim == -1 else -1
    width = signal.shape[width_dim]
    counts = np.random.randint(int(min_amount * width), int(max_amount * width), len(bands))
    slices = [slice(None)] * signal.ndim
    for band, count in zip(bands, counts):
        locations = torch.randperm(width, device=signal.device)[:count]
        slices[band_dim] = band
        slices[width_dim] = locations
        signal[slices] *= 0
    return signal


def add_noise_mixed(
    signal: torch.Tensor,
    num_bands: Iterable,
    noise_deadline_args,
    noise_stripe_args,
    noise_impulse_args,
    band_dim: int = -1,
    inplace: bool = True,
):
    """Apply impulse, stripe, and deadline noise to disjoint random bands."""
    bank = (
        (_add_noise_impulse, noise_impulse_args),
        (_add_noise_stripe, noise_stripe_args),
        (_add_noise_deadline, noise_deadline_args),
    )
    band_count = signal.shape[band_dim]
    all_bands = np.random.permutation(range(band_count))
    position = 0
    output = signal if inplace else signal.clone()
    for (noise_maker, arguments), count in zip(bank, num_bands):
        if 0 < count <= 1:
            count = int(np.floor(count * band_count))
        selected = all_bands[position:position + count]
        position += count
        output = noise_maker(output, selected, band_dim=band_dim, **arguments)
    return output
