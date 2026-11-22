"""Paper backbones, loaded only when requested."""

__all__ = ["MODEL_REGISTRY", "PAPER_MODELS", "build_model"]


def __getattr__(name):
    if name in __all__:
        from . import registry
        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
