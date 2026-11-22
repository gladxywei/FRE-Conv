# FRE-Conv

[English](README.md) | 简体中文

FRE-Conv 是一种采用傅里叶基参数化的旋转等变三维卷积算子，用于高光谱图像去噪。它将空间旋转结构引入卷积滤波器，同时保留光谱维度的有序性。你可以将它作为 PyTorch 层接入自己的网络，也可以使用其在四种去噪骨干中的实现、预训练权重及训练评测工具。

对应论文题为 *Equivariant 3D Convolution Operator for Hyperspectral Image Denoising*（用于高光谱图像去噪的等变三维卷积算子）。

[使用算子](#在自己的网络中使用-fre-conv) · [预训练权重](#预训练权重) · [论文结果](#结果) · [API 指南](docs/operators.zh-CN.md)

## 核心方法

高光谱图像具有两个空间维度和一个光谱维度。FRE-Conv 使用可转向傅里叶基，对按群元素索引的三维滤波器进行参数化，在不同空间旋转之间共享滤波器结构，同时建模局部光谱邻域。旋转作用仅发生在空间平面内，不会旋转或置换光谱轴。

每种骨干网络均提供三个版本：标准三维卷积（`Conv`）、采用八个采样旋转的 FRE-Conv（`T8`），以及采用四个采样旋转的 FRE-Conv（`T4`）。参数共享减少了可学习参数量；去噪效果的提升则取决于骨干网络和噪声设置。

理论保证区分连续算子与离散算子。在论文规定的假设下，连续域的输入、中间和输出算子满足严格的空间旋转等变性。采样实现则在正则性、有限支撑和固定滤波器尺寸等假设下具有有界离散化误差。

## 安装

克隆仓库并进入根目录，或下载并解压仓库的源码 ZIP：

```bash
git clone https://github.com/gladxywei/FRE-Conv.git
cd FRE-Conv
```

### 算子库

在已选好 PyTorch 构建的 Python 3.9 环境中，从本仓库根目录安装：

```bash
python -m pip install .
```

算子库只依赖 NumPy 和 PyTorch，支持 CPU 使用，不需要论文的 CUDA 环境、数据集或预训练权重。安装后即可在自己的项目中导入 `freconv`。构建和安装 wheel 的方式见[算子指南](docs/operators.zh-CN.md#安装和分发库)。

### 论文训练与评测

论文训练与评测工作流的参考环境为 Python 3.9.5、PyTorch 1.8.1（CUDA 10.2）、torchvision 0.9.1（CUDA 10.2）和 scikit-image 0.19.2。在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate freconv
python -m pip install -e . --no-deps
```

[environment.yml](environment.yml) 指定了参考环境，[pyproject.toml](pyproject.toml) 列出了 Python 包依赖。请使用与这些版本兼容的 Linux/CUDA 环境。

`pyproject.toml` 中的 `experiments` 可选依赖组列出了工作流所需的额外依赖，上述 Conda 环境已经包含这些依赖。训练、评测、数据准备和权重下载需要使用仓库中的 `data/`、`checkpoints/` 和 `results/` 资源，请保留这一可编辑安装的仓库，并从仓库根目录运行相关命令。训练和评测默认使用 CUDA 设备 0，可通过 `--gpu` 选择其他设备。

## 在自己的网络中使用 FRE-Conv

像组合其他 PyTorch 模块一样，连接输入提升层、群特征卷积层和输出映射层：

```python
import torch
from torch import nn
from freconv import FREConvLift3d, FREConv3d, FREConvReadout3d

net = nn.Sequential(
    FREConvLift3d(1, 4, rotations=4),     # 普通输入 -> 群特征
    nn.ReLU(),
    FREConv3d(4, 4, rotations=4),         # 处理群特征
    nn.ReLU(),
    FREConvReadout3d(4, 1, rotations=4),  # 群特征 -> 普通输出
)
x = torch.randn(1, 1, 31, 32, 32)  # [批量, 通道, 波段, 高, 宽]
y = net(x)                         # 与 x 形状相同
assert y.shape == x.shape
```

中间特征通道包含旋转轴：`4` 种特征、`4` 个旋转对应 `16` 个张量通道。构造参数中的通道数不包含这个旋转倍数，相连的层应使用相同的旋转数（`4` 或 `8`）。旋转作用于空间平面，不作用于光谱波段。需要对群特征归一化时，使用同样可从 `freconv` 导入的 `FREBatchNorm3d`。

对于以 `[波段, 高, 宽]` 保存的单个 float32 高光谱立方体，可用 `x = torch.from_numpy(cube)[None, None]` 添加批量轴和特征通道轴。因此，31 波段图像对应一个输入特征通道，而非 31 个。示例采用默认的 `kernel_size=3`、`padding=1` 和单位步长，保持三个体数据维度的尺寸不变。

完整参数、共享归一化、权重保存加载和 wheel 打包说明见[算子指南](docs/operators.zh-CN.md)。可通过 `python examples/operator_quickstart.py` 运行[完整示例](examples/operator_quickstart.py)。示例是一个可训练的小网络，并非预训练去噪器；论文模型和已发布权重见下文。

公开层使用 DeNet/MemNet 的算子实现。使用完整论文模型时，通过 `build_model` 或下方评测命令选择模型 ID；各骨干网络保留自己的算子配置和对应权重格式。改造现有模型时，可查阅[实现对应关系](docs/operators.zh-CN.md#与论文模型的关系)。

## 仓库导航

| 路径 | 作用 |
| --- | --- |
| [freconv/ops.py](freconv/ops.py) | 可复用的公开 FRE-Conv 层，使用 PyTorch 风格构造参数。 |
| [docs/operators.zh-CN.md](docs/operators.zh-CN.md) | 算子 API、张量布局、集成和打包说明。 |
| [examples/](examples/) | 独立算子示例，不需要数据集或预训练权重。 |
| [tests/](tests/) | 算子形状、梯度、数值一致性和权重保存加载测试。 |
| [freconv/models/](freconv/models/) | 底层算子实现、论文骨干网络和模型注册表。 |
| [freconv/data/](freconv/data/) | 数据准备、数组加载、数据增强和噪声模型。 |
| [freconv/train.py](freconv/train.py) | ICVL 上的高斯噪声与复杂噪声训练。 |
| [freconv/evaluate.py](freconv/evaluate.py) | ICVL 和 Chikusei 上的合成噪声评测。 |
| [freconv/evaluate_real.py](freconv/evaluate_real.py) | Washington DC Mall 上的定性评测。 |
| [freconv/](freconv/) | 公共推理、指标、损失、日志和权重工具。 |
| [data/](data/README.md) | 数据集说明，以及路径与预处理配置表。 |
| [checkpoints/](checkpoints/README.md) | 权重安装说明、模型分组和文件校验值。 |
| [results/](results/README.md) | 论文结果表、指标定义和证据来源。 |
| [LICENSES/](LICENSES/) | 第三方许可证原文与版权声明。 |

训练和评测默认将新运行的输出写入 `experiments/`。数据和预训练权重需要单独获取，具体见下文。

## 模型

训练时通过 `--model`、评测时通过 `--models` 使用以下 ID：

| 骨干网络 | Conv | T8 | T4 |
| --- | --- | --- | --- |
| DeNet3D | `denet3d2` | `fdenet3d2` | `fdenet3d_t4` |
| MemNet3D | `memnet3d2_plus` | `fmemnet3d2_plus` | `fmemnet3d2_plus_t4` |
| QRNN3D | `qrnn3d_wf` | `fqrnn3d` | `fqrnn3d_t4` |
| MAN | `man` | `fman` | `fman_t4` |

评测时，`paper12` 选择全部十二个模型；`baseline_t4` 选择四组 Conv/T4 模型对。模型定义位于 [registry.py](freconv/models/registry.py)，评测分组位于 [checkpoints/registry.yaml](checkpoints/registry.yaml)。

在 Python 中构建论文骨干网络：

```python
from freconv import build_model

model = build_model("fman_t4")
```

`build_model` 返回网络结构及新初始化的参数。安装[已发布权重](#预训练权重)后，下方的评测命令会加载对应的预训练 checkpoint。

## 数据集

ICVL 用于监督训练和合成噪声测试。Chikusei 用于检验 ICVL 训练模型在合成噪声条件下的零样本迁移能力。Washington DC Mall 提供真实传感器输入，用于无干净真值的定性零样本评测。

ICVL 请下载 **31 波段 MAT 数据**（400–700 nm），来源为[官方数据集页面](https://icvl.cs.bgu.ac.il/pages/researches/hyperspectral-imaging.html)或其 [Hugging Face 镜像](https://huggingface.co/datasets/ICVL-BGU/ICVL_HS_2016)，选择镜像中的 `mat/` 子目录。将每个 `rad` 数组转换为波段轴在前的 float32 `.npy`；详见 [ICVL 准备说明](data/README.md#icvl)。

准备好的数据立方体应为 float32 NumPy 数组，形状为 `[bands, height, width]`：

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

数据来源链接、格式转换、互不重叠的 ICVL 子集划分及预处理方式见[数据集指南](data/README.md)。ICVL 辅助工具用于检查准备好的数组布局。数据的获取与使用应遵守提供方的条款。

## 预训练权重

预训练权重已发布于 [GitHub Release `checkpoints-v1`](https://github.com/gladxywei/FRE-Conv/releases/tag/checkpoints-v1)。压缩包包含 `paper-v1` 权重集：24 个在 ICVL 上训练的权重，上述每个模型各有一份高斯噪声权重和一份复杂噪声权重。

在仓库根目录下载并安装已发布的压缩包：

```bash
python -m freconv.download_checkpoints
```

也可以手动下载 [freconv-paper-v1-checkpoints.tar.gz](https://github.com/gladxywei/FRE-Conv/releases/download/checkpoints-v1/freconv-paper-v1-checkpoints.tar.gz)（386.4 MB），然后安装本地文件：

```bash
python -m freconv.download_checkpoints --archive freconv-paper-v1-checkpoints.tar.gz
```

工具会按照[权重清单](checkpoints/manifest.json)核对压缩包及全部 24 个权重的大小与 SHA-256 校验值，然后将文件安装到 `checkpoints/paper-v1/<model>/`。GitHub 发布标签为 `checkpoints-v1`，权重集 ID 和本地目录仍为 `paper-v1`。检查已安装的文件：

```bash
python -m freconv.download_checkpoints --verify-only
```

归档目录结构和校验行为见[权重指南](checkpoints/README.md)。评测需要这些权重；从头训练则不需要。

## 训练

准备好 ICVL 训练集与验证集后，选择模型和噪声任务：

```bash
python -m freconv.train --model fman_t4 --task gaussian
python -m freconv.train --model fman_t4 --task complex
```

将两种任务作为独立训练分别运行：

```bash
python -m freconv.train --model fman_t4 --task all
```

默认训练设置为 Adam、学习率 0.001、100 个 epoch，并从 epoch 50 起将学习率降为原来的十分之一。模型初始化使用种子 42，数据加载器设置阶段将种子重置为 0，每个 epoch 使用 `epoch + 2022`。

高斯噪声训练采用盲高斯噪声，并保存验证集 PSNR 最优与损失最优的权重。复杂噪声训练将非独立同分布高斯噪声与随机脉冲、条纹和失效线噪声组合，分别保存非独立同分布噪声验证与混合噪声验证中 PSNR 最优、损失最优的权重。两种任务还会每十个 epoch 定期保存 checkpoint。

使用 `--dry-run` 可写出解析后的训练协议，而不加载数据或开始优化；使用 `--help` 查看全部选项。训练入口仅支持 ICVL。

## 评测

准备好数据并安装权重后，评测全部九种合成噪声模式：

```bash
python -m freconv.evaluate --dataset icvl --models paper12 --modes all
python -m freconv.evaluate --dataset chikusei --models baseline_t4 --modes all
```

噪声模式包括高斯噪声 `30`、`50`、`70` 和 `blind`，以及复杂噪声 `case1` 至 `case5`。也可选择单独的模型和模式，进行较小规模的运行：

```bash
python -m freconv.evaluate --dataset chikusei --models man fman_t4 --modes 50
```

评测会将逐场景指标和聚合 CSV 汇总写入运行目录。`--save-images` 额外保存可视化，`--output-root` 可更改输出位置。

Washington DC Mall 评测：

```bash
python -m freconv.evaluate_real --dataset wdc_mall --models man fman_t4 --task complex
```

该入口保存输入、去噪输出、残差的可视化，以及无参考诊断指标。由于没有干净真值，这些诊断不能用于得出 PSNR、SSIM、SAM 或定量去噪排名。

## 结果

下列表格汇总了论文报告的 ICVL 结果和已记录的 Chikusei 评测结果。完整 PSNR、SSIM、SAM 表格、指标定义和数据来源见 [results/](results/README.md)。

可学习参数量，单位为百万：

| 骨干网络 | Conv | T8 | T4 |
| --- | ---: | ---: | ---: |
| DeNet3D | 0.92 | 0.53 | 0.24 |
| MemNet3D | 0.92 | 0.52 | 0.23 |
| QRNN3D | 0.86 | 0.52 | 0.23 |
| MAN | 0.89 | 0.37 | 0.19 |

ICVL 九种已报告噪声模式的平均 PSNR（dB）：

| 骨干网络 | Conv | T8 | T4 |
| --- | ---: | ---: | ---: |
| DeNet3D | 37.63 | 38.88 | 38.34 |
| MemNet3D | 38.43 | 38.96 | 38.96 |
| QRNN3D | 39.49 | 39.29 | 39.36 |
| MAN | 40.09 | 39.89 | 40.51 |

这些数值是对论文表格中已舍入数值计算的算术平均，用于概括九种设置，并不表示每种骨干网络或每种噪声模式都获得提升。

Chikusei 零样本平均 PSNR（dB），覆盖九种模式和八个固定裁剪区域：

| 骨干网络 | Conv | T8 | T4 |
| --- | ---: | ---: | ---: |
| DeNet3D | 36.48 | 37.46 | 37.29 |
| MemNet3D | 37.20 | 37.92 | 38.03 |
| QRNN3D | 34.04 | 36.53 | 35.92 |
| MAN | 34.69 | 34.15 | 36.54 |

在 Chikusei 上，四种 T4 变体相对于各自的 Conv 基线，聚合 PSNR 和 SSIM 均提高，SAM 均降低。这些结果使用伪干净裁剪区域上的合成噪声。Washington DC Mall 仅提供无真值的定性证据，不提供定量结果表。

## 引用

对应论文题为 *Equivariant 3D Convolution Operator for Hyperspectral Image Denoising*。软件引用信息见 [CITATION.cff](CITATION.cff)。

## 致谢

FRE-Conv 基于 [F-Conv](https://github.com/XieQi2015/F-Conv) 的傅里叶滤波器参数化思想，将其扩展到仅在空间平面旋转的局部空间—光谱三维滤波。感谢其作者提供的基础方法与实现。

骨干网络比较基于 [HSI-DeNet](https://doi.org/10.1109/TGRS.2018.2859203)、[MemNet](https://github.com/tyshiwo/MemNet)、[QRNN3D](https://github.com/Vandermode/QRNN3D) 和 [MAN](https://github.com/zeqiang-lai/MAN)。本仓库的 DeNet3D 与 MemNet3D 是经过调整的三维配置，而非原论文未经修改的实现。

感谢 [HyDe](https://github.com/Helmholtz-AI-Energy/HyDe) 提供的骨干网络及训练、数据工具，[HSIR](https://github.com/bit-isp/HSIR) 提供的 MIT 许可 MAN 实现，[pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) 提供的进度显示代码，以及 [e2cnn](https://github.com/QUVA-Lab/e2cnn) 的相关等变层辅助实现。各组件的具体归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 许可证

FRE-Conv 自有代码与文档采用 [MIT 许可证](LICENSE)。第三方部分保留其适用的条款与版权声明，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [LICENSES/](LICENSES/)。

## 代码来源与文件完整性

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) 记录来源版本和上游组件。[RELEASE_MANIFEST.json](RELEASE_MANIFEST.json) 列出文件大小与 SHA-256 校验值。
