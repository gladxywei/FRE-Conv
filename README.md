# FRE-Conv

English | [简体中文](README.zh-CN.md)

FRE-Conv is a Fourier-basis parameterized, rotation-equivariant 3D convolution operator for hyperspectral image denoising. It introduces spatial rotation structure into convolution filters while retaining the ordered spectral dimension. Use it as a PyTorch layer in your own network, or explore its integration into four denoising backbones with pretrained weights and training/evaluation tools.

The accompanying manuscript is titled *Equivariant 3D Convolution Operator for Hyperspectral Image Denoising*.

[Use the operator](#use-fre-conv-in-your-own-network) · [Pretrained weights](#pretrained-weights) · [Paper results](#results) · [API guide](docs/operators.md)

## How it works

A hyperspectral image has two spatial dimensions and one spectral dimension. FRE-Conv uses steerable Fourier bases to parameterize group-indexed 3D filters, sharing filter structure across spatial rotations while modeling local spectral neighborhoods. The rotation action applies only to the spatial plane: it does not rotate or permute the spectral axis.

Each backbone has three variants: standard 3D convolution (`Conv`), FRE-Conv with eight sampled rotations (`T8`), and FRE-Conv with four sampled rotations (`T4`). Parameter sharing reduces the number of learnable parameters; its denoising benefit depends on the backbone and noise setting.

The theoretical guarantees distinguish continuous and discrete operators. The continuous input, intermediate, and output operators are strictly spatially rotation-equivariant under the paper's assumptions. The sampled implementation has bounded discretization error under regularity, finite-support, and fixed-filter-size assumptions.

## Installation

Clone the repository and enter its root directory, or download and extract its source ZIP:

```bash
git clone https://github.com/gladxywei/FRE-Conv.git
cd FRE-Conv
```

### Operator library

In a Python 3.9 environment with your chosen PyTorch build, install from this repository's root:

```bash
python -m pip install .
```

The operator library depends only on NumPy and PyTorch. It supports CPU use and does not require the paper's CUDA environment, datasets, or pretrained weights. After installation, import `freconv` from your own project. See the [operator guide](docs/operators.md#installing-and-distributing-the-library) for building and installing a wheel.

### Paper training and evaluation

For the paper's training and evaluation workflows, the reference environment is Python 3.9.5, PyTorch 1.8.1 with CUDA 10.2, torchvision 0.9.1 with CUDA 10.2, and scikit-image 0.19.2. Install from the repository root:

```bash
conda env create -f environment.yml
conda activate freconv
python -m pip install -e . --no-deps
```

[environment.yml](environment.yml) specifies the reference environment; [pyproject.toml](pyproject.toml) lists the Python package dependencies. Use a Linux/CUDA environment compatible with these versions.

The `experiments` extra in `pyproject.toml` groups the additional workflow dependencies; the Conda environment above already supplies them. Keep this editable checkout for training, evaluation, data preparation, and weight downloads: these tools use its `data/`, `checkpoints/`, and `results/` resources. Run their commands from the repository root. Training and evaluation default to CUDA device 0; `--gpu` selects another device.

## Use FRE-Conv in your own network

Compose the input lifting, group-feature convolution, and output readout layers just like other PyTorch modules:

```python
import torch
from torch import nn
from freconv import FREConvLift3d, FREConv3d, FREConvReadout3d

net = nn.Sequential(
    FREConvLift3d(1, 4, rotations=4),     # ordinary input -> group features
    nn.ReLU(),
    FREConv3d(4, 4, rotations=4),         # process group features
    nn.ReLU(),
    FREConvReadout3d(4, 1, rotations=4),  # group features -> ordinary output
)
x = torch.randn(1, 1, 31, 32, 32)  # [batch, channels, bands, height, width]
y = net(x)                         # same shape as x
assert y.shape == x.shape
```

Intermediate feature channels include a rotation axis: `4` features with `4` rotations occupy `16` tensor channels. Constructor channel counts exclude this rotation multiplier, and connected layers must use the same rotation count (`4` or `8`). Rotations act on the spatial plane, not the spectral bands. For normalization of group features, use `FREBatchNorm3d`, also available from `freconv`.

For a single float32 hyperspectral cube stored as `[bands, height, width]`, add batch and feature-channel axes with `x = torch.from_numpy(cube)[None, None]`. Thus a 31-band image has one input feature channel, not 31. With the default `kernel_size=3`, `padding=1`, and unit stride, the example preserves all three volume dimensions.

See the [operator guide](docs/operators.md) for all parameters, shared normalization, checkpoints, and wheel packaging. Run [the complete example](examples/operator_quickstart.py) with `python examples/operator_quickstart.py`. The example is a small trainable network, not a pretrained denoiser; the paper models and published weights are described below.

The public layers use the DeNet/MemNet operator implementation. To use a complete paper model, select its ID through `build_model` or the evaluation commands below; each backbone retains its own operator configuration and matching checkpoint format. See [the implementation mapping](docs/operators.md#relationship-to-the-paper-models) when adapting an existing model.

## Repository guide

| Path | Purpose |
| --- | --- |
| [freconv/ops.py](freconv/ops.py) | Public, reusable FRE-Conv layers with PyTorch-style constructors. |
| [docs/operators.md](docs/operators.md) | Operator API, tensor layout, integration, and packaging guide. |
| [examples/](examples/) | Standalone operator example; no dataset or pretrained weights needed. |
| [tests/](tests/) | Operator shape, gradient, numerical-parity, and checkpoint tests. |
| [freconv/models/](freconv/models/) | Low-level operator implementations, paper backbones, and the model registry. |
| [freconv/data/](freconv/data/) | Dataset preparation, array loaders, augmentation, and noise models. |
| [freconv/train.py](freconv/train.py) | ICVL training for Gaussian and complex noise. |
| [freconv/evaluate.py](freconv/evaluate.py) | Synthetic-noise evaluation on ICVL and Chikusei. |
| [freconv/evaluate_real.py](freconv/evaluate_real.py) | Qualitative evaluation on Washington DC Mall. |
| [freconv/](freconv/) | Shared inference, metrics, losses, logging, and checkpoint utilities. |
| [data/](data/README.md) | Dataset instructions and the path/preprocessing registry. |
| [checkpoints/](checkpoints/README.md) | Weight installation instructions, model groups, and file checksums. |
| [results/](results/README.md) | Paper result tables, metric definitions, and evidence provenance. |
| [LICENSES/](LICENSES/) | Third-party license texts and copyright notices. |

Training and evaluation write new runs under `experiments/` by default. Data and pretrained weights are obtained separately, as described below.

## Models

Use these IDs with `--model` for training or `--models` for evaluation:

| Backbone | Conv | T8 | T4 |
| --- | --- | --- | --- |
| DeNet3D | `denet3d2` | `fdenet3d2` | `fdenet3d_t4` |
| MemNet3D | `memnet3d2_plus` | `fmemnet3d2_plus` | `fmemnet3d2_plus_t4` |
| QRNN3D | `qrnn3d_wf` | `fqrnn3d` | `fqrnn3d_t4` |
| MAN | `man` | `fman` | `fman_t4` |

For evaluation, `paper12` selects all twelve models; `baseline_t4` selects the four Conv/T4 pairs. Model definitions are in [registry.py](freconv/models/registry.py), and evaluation groups are in [checkpoints/registry.yaml](checkpoints/registry.yaml).

To construct a paper backbone in Python:

```python
from freconv import build_model

model = build_model("fman_t4")
```

`build_model` returns the architecture with freshly initialized parameters. The evaluation commands below load the matching pretrained checkpoint after you install the [published weights](#pretrained-weights).

## Datasets

ICVL is used for supervised training and synthetic-noise testing. Chikusei tests zero-shot transfer of ICVL-trained models under synthetic noise. Washington DC Mall provides real-sensor inputs for qualitative zero-shot evaluation, without clean ground truth.

For ICVL, download the **31-band MAT data** (400–700 nm) from the [official dataset page](https://icvl.cs.bgu.ac.il/pages/researches/hyperspectral-imaging.html) or its [Hugging Face mirror](https://huggingface.co/datasets/ICVL-BGU/ICVL_HS_2016), using the mirror's `mat/` subset. Convert each `rad` array to bands-first float32 `.npy`; see the [ICVL preparation notes](data/README.md#icvl).

Prepared cubes use float32 NumPy arrays with shape `[bands, height, width]`:

```text
data/
  ICVL/
    train/*.npy
    vval/*.npy
    test/*.npy
  Chikusei/
    test/*.npy
  WDC_Mall/
    real/wdc_mall.npy
```

See the [dataset guide](data/README.md) for source links, conversion instructions, disjoint ICVL split selection, and preprocessing. The ICVL helper checks the prepared array layout. Dataset access and use are subject to the providers' terms.

## Pretrained weights

The pretrained weights are available in [GitHub Release `checkpoints-v1`](https://github.com/gladxywei/FRE-Conv/releases/tag/checkpoints-v1). The archive contains the `paper-v1` checkpoint set: 24 ICVL-trained weights, with Gaussian and complex-noise checkpoints for every model above.

Download and install the published archive from the repository root:

```bash
python -m freconv.download_checkpoints
```

Alternatively, download [freconv-paper-v1-checkpoints.tar.gz](https://github.com/gladxywei/FRE-Conv/releases/download/checkpoints-v1/freconv-paper-v1-checkpoints.tar.gz) (386.4 MB) and install the local file:

```bash
python -m freconv.download_checkpoints --archive freconv-paper-v1-checkpoints.tar.gz
```

The tool verifies the archive and all 24 weights against the sizes and SHA-256 digests in the [checkpoint manifest](checkpoints/manifest.json), then installs files under `checkpoints/paper-v1/<model>/`. The GitHub release tag is `checkpoints-v1`; the checkpoint-set ID and local directory remain `paper-v1`. Check an existing installation with:

```bash
python -m freconv.download_checkpoints --verify-only
```

See the [checkpoint guide](checkpoints/README.md) for the archive layout and verification behavior. Evaluation requires these weights; training from scratch does not.

## Training

Prepare the ICVL training and validation splits, then choose a model and noise task:

```bash
python -m freconv.train --model fman_t4 --task gaussian
python -m freconv.train --model fman_t4 --task complex
```

To run both tasks as separate training runs:

```bash
python -m freconv.train --model fman_t4 --task all
```

The default schedule uses Adam, learning rate 0.001, and 100 epochs, with a tenfold learning-rate reduction from epoch 50. Model initialization uses seed 42, the loader setup resets the seed to 0, and each epoch uses `epoch + 2022`.

Gaussian training uses blind Gaussian noise and saves validation PSNR-best and loss-best weights. Complex training combines non-IID Gaussian noise with random impulse, stripe, and deadline corruption, saving PSNR-best and loss-best weights for both non-IID and mixture validation. Both tasks also save periodic checkpoints every ten epochs.

Use `--dry-run` to write the resolved protocol without loading data or starting optimization, or `--help` for all options. The training entrypoint supports ICVL only.

## Evaluation

With prepared data and installed checkpoints, evaluate all nine synthetic-noise modes:

```bash
python -m freconv.evaluate --dataset icvl --models paper12 --modes all
python -m freconv.evaluate --dataset chikusei --models baseline_t4 --modes all
```

The modes are Gaussian `30`, `50`, `70`, and `blind`, plus complex `case1` through `case5`. Select individual models and modes for a smaller run:

```bash
python -m freconv.evaluate --dataset chikusei --models man fman_t4 --modes 50
```

Evaluation writes per-scene metrics and aggregate CSV summaries to the run directory. `--save-images` additionally saves visualizations; `--output-root` changes the output location.

For Washington DC Mall:

```bash
python -m freconv.evaluate_real --dataset wdc_mall --models man fman_t4 --task complex
```

This entrypoint saves input, denoised output, and residual visualizations, together with no-reference diagnostics. Without clean ground truth, these diagnostics do not establish PSNR, SSIM, SAM, or a quantitative denoising ranking.

## Results

The following tables summarize the ICVL results reported in the manuscript and the recorded Chikusei evaluation results. Full PSNR, SSIM, and SAM tables, together with metric definitions and data sources, are available in [results/](results/README.md).

Learnable parameters, in millions:

| Backbone | Conv | T8 | T4 |
| --- | ---: | ---: | ---: |
| DeNet3D | 0.92 | 0.53 | 0.24 |
| MemNet3D | 0.92 | 0.52 | 0.23 |
| QRNN3D | 0.86 | 0.52 | 0.23 |
| MAN | 0.89 | 0.37 | 0.19 |

ICVL mean PSNR (dB) across the nine reported noise modes:

| Backbone | Conv | T8 | T4 |
| --- | ---: | ---: | ---: |
| DeNet3D | 37.63 | 38.88 | 38.34 |
| MemNet3D | 38.43 | 38.96 | 38.96 |
| QRNN3D | 39.49 | 39.29 | 39.36 |
| MAN | 40.09 | 39.89 | 40.51 |

These are arithmetic means of the rounded paper-table values. They summarize the nine settings, rather than implying an improvement for every backbone or noise mode.

Chikusei zero-shot mean PSNR (dB), across nine modes and eight fixed crops:

| Backbone | Conv | T8 | T4 |
| --- | ---: | ---: | ---: |
| DeNet3D | 36.48 | 37.46 | 37.29 |
| MemNet3D | 37.20 | 37.92 | 38.03 |
| QRNN3D | 34.04 | 36.53 | 35.92 |
| MAN | 34.69 | 34.15 | 36.54 |

All four T4 variants improve aggregate PSNR and SSIM and reduce SAM relative to their Conv baselines on Chikusei. These results use synthetic noise on pseudo-clean crops. Washington DC Mall supplies qualitative, no-ground-truth evidence only and has no quantitative result table.

## Citation

The associated manuscript is *Equivariant 3D Convolution Operator for Hyperspectral Image Denoising*. For the software citation, see [CITATION.cff](CITATION.cff).

## Acknowledgments

FRE-Conv builds on the Fourier filter-parameterization ideas of [F-Conv](https://github.com/XieQi2015/F-Conv), extending the construction to local spatial-spectral 3D filtering with spatial-plane rotations. We thank its authors for the foundational method and implementation.

The backbone comparisons follow [HSI-DeNet](https://doi.org/10.1109/TGRS.2018.2859203), [MemNet](https://github.com/tyshiwo/MemNet), [QRNN3D](https://github.com/Vandermode/QRNN3D), and [MAN](https://github.com/zeqiang-lai/MAN). DeNet3D and MemNet3D here are adapted 3D configurations, not the original papers' unmodified implementations.

We acknowledge [HyDe](https://github.com/Helmholtz-AI-Energy/HyDe) for backbone and training/data utilities, [HSIR](https://github.com/bit-isp/HSIR) for its MIT-licensed MAN implementation, [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) for progress-display code, and [e2cnn](https://github.com/QUVA-Lab/e2cnn) for related equivariant-layer utility lineage. Component-specific credits are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Original FRE-Conv contributions are licensed under the [MIT License](LICENSE). Third-party portions retain their applicable terms and copyright notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES/](LICENSES/).

## Source and file integrity

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records the source revision and upstream components. [RELEASE_MANIFEST.json](RELEASE_MANIFEST.json) lists file sizes and SHA-256 checksums.
