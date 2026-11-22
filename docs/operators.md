# Using FRE-Conv in your network

English | [简体中文](operators.zh-CN.md)

The public layers live in [`freconv.ops`](../freconv/ops.py) and can also be imported directly from `freconv`. They are PyTorch modules: compose them with `nn.Sequential`, optimize their parameters, and move them with `.to(device)` as usual. No dataset or pretrained checkpoint is needed to use the operator.

## Three stages

Input tensors have shape `[N, C, B, H, W]`: batch, feature channels, spectral bands, height, width. A raw hyperspectral cube usually has `C = 1`; its bands belong on the third axis, not the feature-channel axis.

For `T` rotations, intermediate tensors flatten feature and rotation axes in **feature-major order**: `[N, C, T, B, H, W]` becomes `[N, C*T, B, H, W]`. Keep the same `T` throughout a connected group-feature path.

| Layer | Input channels | Output channels | Role |
| --- | --- | --- | --- |
| `FREConvLift3d(Cin, Cout, rotations=T)` | `Cin` | `Cout*T` | Introduce the rotation axis. |
| `FREConv3d(Cin, Cout, rotations=T)` | `Cin*T` | `Cout*T` | Process group-indexed features. |
| `FREConvReadout3d(Cin, Cout, rotations=T)` | `Cin*T` | `Cout` | Return to ordinary spatial-spectral features. |
| `FREBatchNorm3d(C, rotations=T)` | `C*T` | `C*T` | Share normalization across rotations. |

`in_channels` and `out_channels` count feature types **before** multiplication by `T`. Readout is a learned convolution, not global rotation-invariant pooling: the output remains a spatial image.

```python
import torch
from torch import nn
from freconv import FREConvLift3d, FREConv3d, FREConvReadout3d, FREBatchNorm3d

net = nn.Sequential(
    FREConvLift3d(1, 4, rotations=4),  # 1 -> 16 channels
    FREBatchNorm3d(4, rotations=4),
    nn.ReLU(),
    FREConv3d(4, 4, rotations=4),      # 16 -> 16 channels
    nn.ReLU(),
    FREConvReadout3d(4, 1, rotations=4),  # 16 -> 1 channel
)
x = torch.randn(1, 1, 31, 32, 32)
y = net(x)
assert y.shape == x.shape
y.square().mean().backward()
```

This small network illustrates composition; it is not a pretrained denoiser. A complete runnable example, including inference and state-dict loading, is in [`examples/operator_quickstart.py`](../examples/operator_quickstart.py):

```bash
python examples/operator_quickstart.py
python -m unittest discover -s tests -v
```

## Constructor options

All three convolution layers accept `in_channels`, `out_channels`, and `kernel_size=3`. Other arguments are keyword-only:

| Argument | Default | Meaning |
| --- | --- | --- |
| `rotations` | `8` | Number of sampled in-plane rotations: `4` or `8`. |
| `padding` | `1` | Integer or `(spectral, height, width)` tuple. For unchanged sizes with unit stride, use `kernel_size // 2`. |
| `bias` | `True` | Learn an additive bias. With `False`, the underlying implementation retains a fixed zero `c` parameter. |
| `smooth` | `True` | Apply the basis reweighting used by the implementation. |
| `init_scale` | `1.0` | Scale initial Fourier-basis coefficients. |
| `stride` | `1` | Integer or `(spectral, height, width)` tuple; lifting and group layers only. Readout has unit stride. |

The public API supports odd cubic kernels of size at least 3. It does not expose arbitrary dilation, grouped convolution, or a `1x1x1` specialization. `padding=1` is a fixed default, not automatic same padding for larger kernels. With kernel size `K`, padding `P`, and stride `S`, each output dimension is `floor((input + 2*P - K)/S) + 1`.

`FREBatchNorm3d(num_features, rotations=8)` uses the underlying shared BatchNorm defaults. Pointwise activations such as ReLU can follow group layers directly. Skip additions must match both feature count and rotation layout. Do not treat the flattened rotation channels as unrelated ordinary channels when adding convolutions or normalization.

The rotation action is restricted to `(H, W)`, leaving the ordered spectral axis unchanged. Spatial downsampling, interpolation, boundaries, and other backbone layers affect the behavior of the full network; using these layers alone does not guarantee exact equivariance of a sampled network. In particular, T8 includes rotations not aligned with the square pixel grid. See the theoretical scope in the main README.

## Training, inference, and checkpoints

Initialization uses NumPy as well as PyTorch. Seed both when repeatable initialization is needed:

```python
import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)
```

Construct the network after setting the seeds. Use `net.train()` for optimization and `net.eval()` with `torch.no_grad()` for inference. Evaluation expands and caches filters; returning to training discards these caches.

The public layers save `Basis`, `weights`, and `c`, but omit derived evaluation caches. Consequently, their state dictionaries load in either training or evaluation mode; loading while in evaluation mode refreshes the filters. Shared BatchNorm also saves its usual parameters and running statistics. Save and load custom networks with ordinary PyTorch state dictionaries:

```python
torch.save(net.state_dict(), "my_network.pt")
# Recreate the same network architecture as `restored` before loading.
restored.load_state_dict(torch.load("my_network.pt", map_location="cpu"))
restored.eval()
```

Only load checkpoint files from trusted sources. A custom network must be trained for its task; paper checkpoints belong to their matching complete backbone architectures, not this illustrative network.

## Relationship to the paper models

The public layer API wraps the original DeNet/MemNet FRE-Conv implementation without adding a submodule prefix to its learned parameter names:

| Public layer | Underlying implementation |
| --- | --- |
| `FREConvLift3d` | `FCNN_3D_XY.Fconv_3D_PCA`, `ifIni=1` |
| `FREConv3d` | `FCNN_3D_XY.Fconv_3D_PCA`, `ifIni=0` |
| `FREConvReadout3d` | `FCNN_3D_XY.Fconv_3D_PCA_out` |
| `FREBatchNorm3d` | `FCNN_3D_XY.F_BN_3D` |

The low-level names `sizeP`, `inNum`, `outNum`, and `tranNum` correspond to `kernel_size`, `in_channels`, `out_channels`, and `rotations`. The public API uses the full basis (`inP=kernel_size`).

QRNN and MAN use the separate `FCNN_3D_XY_new.py` implementation, with different initialization, masking, and bias handling. The twelve paper models retain their respective implementations and checkpoint conventions. To use published weights, select the matching model through `build_model` and use the documented evaluation commands; do not replace its layers with the public wrappers before loading weights.

```python
from freconv import build_model

paper_model = build_model("fman_t4")
```

## Installing and distributing the library

For operator use, install from a local checkout into a Python 3.9 environment with the desired PyTorch build:

```bash
python -m pip install .
```

This installs the package's core dependencies, NumPy and PyTorch. It does not import the paper backbones or require image/data tooling when you import the operators. For editable paper workflows, use the reference environment in the main README; alternatively, install the experiment dependencies with `python -m pip install -e ".[experiments]"` after selecting the appropriate PyTorch/torchvision builds.

Build a wheel and source distribution with the standard Python build frontend:

```bash
python -m pip install build
python -m build
```

The wheel in `dist/` provides the importable library and can be installed with `python -m pip install dist/freconv-1.0.0rc1-py3-none-any.whl`. Paper experiment commands use the repository's `data/`, `checkpoints/`, and `results/` resources, so run those from an editable checkout or unpacked source distribution. Obtain datasets and pretrained weights separately using the project guide.
