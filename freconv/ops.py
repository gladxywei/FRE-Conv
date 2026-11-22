"""Public FRE-Conv layers for tensors shaped [N, channels, bands, H, W].

Channels in group features are flattened in [feature, rotation] order.
The numerical implementation is shared with the DeNet/MemNet paper models.
See docs/operators.md for shapes, initialization, and checkpoint conventions.
"""

from .models.FCNN_3D_XY import F_BN_3D, Fconv_3D_PCA, Fconv_3D_PCA_out

__all__ = ["FREConvLift3d", "FREConv3d", "FREConvReadout3d", "FREBatchNorm3d"]


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate(in_channels, out_channels, kernel_size, rotations, padding, stride=1):
    for name, value in (("in_channels", in_channels), ("out_channels", out_channels)):
        _positive_int(name, value)
    if isinstance(rotations, bool) or not isinstance(rotations, int) or rotations not in (4, 8):
        raise ValueError("rotations must be 4 or 8")
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int) or kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3")
    values = (padding,) * 3 if isinstance(padding, int) else padding
    if not isinstance(values, tuple) or len(values) != 3 or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values
    ):
        raise ValueError("padding must be a nonnegative integer or a tuple of three integers")
    values = (stride,) * 3 if isinstance(stride, int) else stride
    if not isinstance(values, tuple) or len(values) != 3:
        raise ValueError("stride must be a positive integer or a tuple of three integers")
    for value in values:
        _positive_int("stride", value)


class _PublicLayer:
    """Keep expanded evaluation filters out of portable layer checkpoints."""

    def train(self, mode=True):
        result = super().train(mode)
        # These tensors are derived from Basis/weights/c, not learned state.
        for name in ("filter", "bias"):
            if name in self._buffers:
                self._non_persistent_buffers_set.add(name)
        return result

    def _load_from_state_dict(self, *args, **kwargs):
        was_training = self.training
        if not was_training:
            self.train(True)
        super()._load_from_state_dict(*args, **kwargs)
        if not was_training:
            self.train(False)

    def forward(self, x):
        expected = self.in_channels
        if not isinstance(self, FREConvLift3d):
            expected *= self.rotations
        if x.ndim != 5 or x.shape[1] != expected:
            raise ValueError(f"Expected [N, {expected}, bands, H, W], got {tuple(x.shape)}")
        return super().forward(x)

    def extra_repr(self):
        return (f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.sizeP}, rotations={self.rotations}, padding={self.padding}")


class FREConvLift3d(_PublicLayer, Fconv_3D_PCA):
    """Lift C ordinary channels to out_channels * rotations group channels.

    kernel_size is a cubic odd kernel; padding/stride follow (bands, H, W).
    in_channels and out_channels count features, excluding rotation copies.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, *, rotations=8,
                 padding=1, stride=1, bias=True, smooth=True, init_scale=1.0):
        _validate(in_channels, out_channels, kernel_size, rotations, padding, stride)
        super().__init__(kernel_size, in_channels, out_channels, tranNum=rotations,
                         padding=padding, stride=stride, ifIni=1, bias=bias,
                         Smooth=smooth, iniScale=init_scale)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.rotations = rotations


class FREConv3d(_PublicLayer, Fconv_3D_PCA):
    """Map in_channels * rotations to out_channels * rotations channels.

    Input must use the same rotation count and feature-major layout as the
    preceding FRE-Conv layer. Use FREConvLift3d for ordinary input features.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, *, rotations=8,
                 padding=1, stride=1, bias=True, smooth=True, init_scale=1.0):
        _validate(in_channels, out_channels, kernel_size, rotations, padding, stride)
        super().__init__(kernel_size, in_channels, out_channels, tranNum=rotations,
                         padding=padding, stride=stride, ifIni=0, bias=bias,
                         Smooth=smooth, iniScale=init_scale)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.rotations = rotations


class FREConvReadout3d(_PublicLayer, Fconv_3D_PCA_out):
    """Map in_channels * rotations group channels to ordinary output channels.

    This is a learned spatial-spectral readout, not global rotation pooling.
    Readout uses unit stride; kernel_size and padding control its output size.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, *, rotations=8,
                 padding=1, bias=True, smooth=True, init_scale=1.0):
        _validate(in_channels, out_channels, kernel_size, rotations, padding)
        super().__init__(kernel_size, in_channels, out_channels, tranNum=rotations,
                         padding=padding, bias=bias, Smooth=smooth, iniScale=init_scale)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.rotations = rotations


class FREBatchNorm3d(F_BN_3D):
    """Shared BatchNorm for num_features * rotations group channels."""

    def __init__(self, num_features, *, rotations=8):
        _positive_int("num_features", num_features)
        if isinstance(rotations, bool) or not isinstance(rotations, int) or rotations not in (4, 8):
            raise ValueError("rotations must be 4 or 8")
        super().__init__(num_features, tranNum=rotations)
        self.num_features, self.rotations = num_features, rotations

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != self.num_features * self.rotations:
            raise ValueError("Expected [N, num_features * rotations, bands, H, W]")
        return super().forward(x)

    def extra_repr(self):
        return f"num_features={self.num_features}, rotations={self.rotations}"
