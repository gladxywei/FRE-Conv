"""Spatially rotation-equivariant convolution for spectral image volumes."""

__version__ = "1.0.0rc1"
__all__ = [
    "FREConvLift3d", "FREConv3d", "FREConvReadout3d", "FREBatchNorm3d",
    "MODEL_REGISTRY", "PAPER_MODELS", "build_model",
]


def __getattr__(name):
    if name in __all__[:4]:
        from . import ops
        return getattr(ops, name)
    if name in __all__[4:]:
        from .models import registry
        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
