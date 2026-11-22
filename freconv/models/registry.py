"""Checkpoint-compatible registry for the twelve paper models."""

from dataclasses import dataclass
from typing import Callable

from .denet import DeNetLite3D2, FDeNetLite3D2
from .fman import FMAN
from .fmemnet import FMemNet2
from .man import MAN
from .memnet import MemNet2
from .qrnn import FQRNNREDC3D, QRNNREDC3D

PAPER_MODELS = (
    "denet3d2", "fdenet3d2", "fdenet3d_t4",
    "memnet3d2_plus", "fmemnet3d2_plus", "fmemnet3d2_plus_t4",
    "qrnn3d_wf", "fqrnn3d", "fqrnn3d_t4",
    "man", "fman", "fman_t4",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    variant: str
    factory: Callable[[], object]


def _mark_3d(net):
    net.use_2dconv = False
    return net


MODEL_REGISTRY = {
    "denet3d2": ModelSpec("denet3d2", "denet", "baseline", lambda: _mark_3d(DeNetLite3D2())),
    "fdenet3d2": ModelSpec("fdenet3d2", "denet", "t8", lambda: _mark_3d(FDeNetLite3D2(smooth=True, kernel_size=5))),
    "fdenet3d_t4": ModelSpec("fdenet3d_t4", "denet", "t4", lambda: _mark_3d(FDeNetLite3D2(tranNum=4, smooth=True, kernel_size=5))),
    "memnet3d2_plus": ModelSpec("memnet3d2_plus", "memnet", "baseline", lambda: _mark_3d(MemNet2(1, 64, 2, 2, conv3d=True))),
    "fmemnet3d2_plus": ModelSpec("fmemnet3d2_plus", "memnet", "t8", lambda: _mark_3d(FMemNet2(8, 1, 64, 2, 2, conv3d=True, smooth=True, kernel_size=5))),
    "fmemnet3d2_plus_t4": ModelSpec("fmemnet3d2_plus_t4", "memnet", "t4", lambda: _mark_3d(FMemNet2(4, 1, 64, 2, 2, conv3d=True, smooth=True, kernel_size=5))),
    "qrnn3d_wf": ModelSpec("qrnn3d_wf", "qrnn", "baseline", lambda: _mark_3d(QRNNREDC3D(in_channels=1, channels=16, num_half_layer=5, sample_idx=[1, 3], has_ad=True, clamp=False))),
    "fqrnn3d": ModelSpec("fqrnn3d", "qrnn", "t8", lambda: _mark_3d(FQRNNREDC3D(tranNum=8, in_channels=1, channels=16, num_half_layer=5, sample_idx=[1, 3], has_ad=True, clamp=False, kernel_size=5))),
    "fqrnn3d_t4": ModelSpec("fqrnn3d_t4", "qrnn", "t4", lambda: _mark_3d(FQRNNREDC3D(tranNum=4, in_channels=1, channels=16, num_half_layer=5, sample_idx=[1, 3], has_ad=True, clamp=False))),
    "man": ModelSpec("man", "man", "baseline", lambda: _mark_3d(MAN(in_channels=1, channels=16, num_half_layer=5, sample_idx=[1, 3]))),
    "fman": ModelSpec("fman", "man", "t8", lambda: _mark_3d(FMAN(tranNum=8, in_channels=1, channels=16, num_half_layer=5, sample_idx=[1, 3], kernel_size=5))),
    "fman_t4": ModelSpec("fman_t4", "man", "t4", lambda: _mark_3d(FMAN(tranNum=4, in_channels=1, channels=16, num_half_layer=5, sample_idx=[1, 3]))),
}


def build_model(name: str):
    try:
        return MODEL_REGISTRY[name].factory()
    except KeyError as error:
        raise KeyError(f"Unknown paper model: {name}") from error
