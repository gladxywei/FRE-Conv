"""Noise and augmentation transforms used by the paper protocols."""

import random
from itertools import combinations

import numpy as np
import torch
import torchvision.transforms.functional as TF

from . import noise


class GaussianNoiseLevel:
    def __init__(self, sigma, scale_factor=255.0, noise_type="additive", iid=True, band_dim=-3):
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.noise_type = noise_type
        self.iid = iid
        self.band_dim = band_dim

    def __call__(self, image):
        return noise.add_noise_std(image, self.sigma, self.scale_factor, self.noise_type, self.iid, self.band_dim)


class GaussianBlindNoiseLevel:
    def __init__(self, sigmas=(10, 30, 50, 70), min_sigma=30, max_sigma=70,
                 scale_factor=255.0, noise_type="additive", iid=True, band_dim=-3):
        self.sigmas = sigmas
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.scale_factor = scale_factor
        self.noise_type = noise_type
        self.iid = iid
        self.band_dim = band_dim

    def __call__(self, image):
        sigma = ((self.max_sigma - self.min_sigma) * np.random.rand() + self.min_sigma
                 if self.sigmas is None else self.sigmas[np.random.randint(len(self.sigmas))])
        return noise.add_noise_std(image, sigma, self.scale_factor, self.noise_type, self.iid, self.band_dim)


class AddNoiseNonIID:
    def __init__(self, sigma=(10, 30, 50, 70), scale_factor=255.0,
                 noise_type="additive", iid=False, band_dim=-3):
        self.sigma = list(sigma)
        self.scale_factor = scale_factor
        self.noise_type = noise_type
        self.iid = iid
        self.band_dim = band_dim

    def __call__(self, image):
        return noise.add_noise_std(image, self.sigma, self.scale_factor, self.noise_type, self.iid, self.band_dim)


class AddNoiseMixed:
    def __init__(self, num_bands=(1 / 3, 1 / 3, 1 / 3), min_amount_deadline=0.05,
                 max_amount_deadline=0.15, min_amount_stripe=0.05, max_amount_stripe=0.15,
                 amounts=(0.1, 0.3, 0.5, 0.7), salt_vs_pepper=0.5, band_dim=-3, inplace=True):
        self.num_bands = list(num_bands)
        self.deadline = {"min_amount": min_amount_deadline, "max_amount": max_amount_deadline}
        self.stripe = {"min_amount": min_amount_stripe, "max_amount": max_amount_stripe}
        self.impulse = {"amounts": amounts, "salt_vs_pepper": salt_vs_pepper}
        self.band_dim = band_dim
        self.inplace = inplace

    def __call__(self, image):
        return noise.add_noise_mixed(
            image, self.num_bands, self.deadline, self.stripe, self.impulse,
            self.band_dim, self.inplace,
        )


class AddNoiseImpulse:
    def __init__(self, amounts=(0.1, 0.3, 0.5, 0.7), s_vs_p=0.5, bands=1 / 3, band_dim=-3):
        self.amounts = np.array(amounts)
        self.s_vs_p = s_vs_p
        self.bands = bands
        self.band_dim = band_dim

    def __call__(self, image):
        return noise.add_noise_impulse(image, self.bands, amounts=self.amounts,
                                       salt_vs_pepper=self.s_vs_p, band_dim=self.band_dim)


class AddNoiseStripe:
    def __init__(self, min_amount=0.05, max_amount=0.15, bands=1 / 3, band_dim=-3):
        self.min_amount, self.max_amount = min_amount, max_amount
        self.bands, self.band_dim = bands, band_dim

    def __call__(self, image):
        return noise.add_noise_stripe(image, self.bands, min_amount=self.min_amount,
                                      max_amount=self.max_amount, band_dim=self.band_dim)


class AddNoiseDeadline:
    def __init__(self, min_amount=0.05, max_amount=0.15, bands=1 / 3, band_dim=-3):
        self.min_amount, self.max_amount = min_amount, max_amount
        self.bands, self.band_dim = bands, band_dim

    def __call__(self, image):
        return noise.add_noise_deadline(image, self.bands, min_amount=self.min_amount,
                                        max_amount=self.max_amount, band_dim=self.band_dim)


class RandChoice:
    def __init__(self, transforms, p=None, combos=False):
        self.transforms = list(transforms)
        self.p = [p] * len(self.transforms) if isinstance(p, float) else p
        self.combos = []
        for count in range(1, len(self.transforms) + 1):
            self.combos.extend(combinations(self.transforms, count))
        self.use_combos = combos

    def __call__(self, value, *args):
        choices = self.combos if self.use_combos else self.transforms
        if self.p is None:
            index = random.randrange(len(choices) + 1)
            if index == len(choices):
                return value
            selected = random.choice(choices)
            if not self.use_combos:
                return selected(value, *args)
            for transform in selected:
                value = transform(value, *args)
            return value
        return random.choices(self.transforms, weights=self.p)[0](value, *args)


class RandRot90Transform:
    def __init__(self, angles=(-90, 0, 90, 180)):
        self.angles = tuple(angles)

    def __call__(self, value):
        return TF.rotate(value, random.choice(self.angles))


class RandomReversal:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, value):
        return value.flip(-3) if torch.rand(1) < self.p else value
