# 在自己的网络中使用 FRE-Conv

[English](operators.md) | 简体中文

公开层位于 [`freconv.ops`](../freconv/ops.py)，也可以直接从 `freconv` 导入。它们是 PyTorch 模块，可以用 `nn.Sequential` 组合、优化其参数，并用 `.to(device)` 移动设备。使用算子不需要数据集或预训练权重。

## 三个阶段

输入张量形状为 `[N, C, B, H, W]`，依次表示批量、特征通道、光谱波段、高和宽。原始高光谱立方体通常对应 `C = 1`；波段位于第三个轴，不是特征通道轴。

采用 `T` 个旋转时，中间张量按**特征优先**顺序展平特征轴和旋转轴：`[N, C, T, B, H, W]` 变为 `[N, C*T, B, H, W]`。相连的群特征路径必须保持相同的 `T`。

| 层 | 输入通道数 | 输出通道数 | 作用 |
| --- | --- | --- | --- |
| `FREConvLift3d(Cin, Cout, rotations=T)` | `Cin` | `Cout*T` | 引入旋转轴。 |
| `FREConv3d(Cin, Cout, rotations=T)` | `Cin*T` | `Cout*T` | 处理按群元素索引的特征。 |
| `FREConvReadout3d(Cin, Cout, rotations=T)` | `Cin*T` | `Cout` | 返回普通空间—光谱特征。 |
| `FREBatchNorm3d(C, rotations=T)` | `C*T` | `C*T` | 在旋转维度之间共享归一化。 |

`in_channels` 和 `out_channels` 表示乘以 `T` **之前**的特征种类数。Readout 是可学习的卷积，而不是全局旋转不变池化：输出仍然是空间图像。

```python
import torch
from torch import nn
from freconv import FREConvLift3d, FREConv3d, FREConvReadout3d, FREBatchNorm3d

net = nn.Sequential(
    FREConvLift3d(1, 4, rotations=4),  # 1 -> 16 个通道
    FREBatchNorm3d(4, rotations=4),
    nn.ReLU(),
    FREConv3d(4, 4, rotations=4),      # 16 -> 16 个通道
    nn.ReLU(),
    FREConvReadout3d(4, 1, rotations=4),  # 16 -> 1 个通道
)
x = torch.randn(1, 1, 31, 32, 32)
y = net(x)
assert y.shape == x.shape
y.square().mean().backward()
```

这个小网络用于演示如何组合算子，并不是预训练去噪器。包含推理及 state-dict 加载的完整示例位于 [`examples/operator_quickstart.py`](../examples/operator_quickstart.py)：

```bash
python examples/operator_quickstart.py
python -m unittest discover -s tests -v
```

## 构造参数

三个卷积层均接受 `in_channels`、`out_channels` 和 `kernel_size=3`。其他参数必须按名称传入：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `rotations` | `8` | 空间平面内的采样旋转数，可取 `4` 或 `8`。 |
| `padding` | `1` | 整数或 `(光谱, 高, 宽)` 三元组。单位步长时，使用 `kernel_size // 2` 保持尺寸不变。 |
| `bias` | `True` | 是否学习加性偏置。为 `False` 时，底层实现仍保留固定为零的 `c` 参数。 |
| `smooth` | `True` | 是否启用实现中的基函数重加权。 |
| `init_scale` | `1.0` | 傅里叶基系数的初始化缩放。 |
| `stride` | `1` | 整数或 `(光谱, 高, 宽)` 三元组，仅输入提升层和群卷积层支持；输出层步长固定为 1。 |

公开 API 支持边长至少为 3 的奇数立方卷积核，不提供任意 dilation、分组卷积或 `1x1x1` 专用层。`padding=1` 是固定默认值，不会随更大的卷积核自动调整。核大小为 `K`、填充为 `P`、步长为 `S` 时，每个输出维度为 `floor((input + 2*P - K)/S) + 1`。

`FREBatchNorm3d(num_features, rotations=8)` 使用底层共享 BatchNorm 的默认设置。ReLU 等逐点激活可直接接在群特征层之后。跳跃连接相加时，特征数和旋转布局都必须一致。添加卷积或归一化时，不要将展平的旋转通道视为互不相关的普通通道。

旋转仅作用于 `(H, W)`，有序的光谱轴保持不变。空间下采样、插值、边界和其他骨干层都会影响整个网络的行为；使用这些层并不能单独保证采样网络精确等变。尤其是 T8 包含不与方形像素网格对齐的旋转。理论适用范围见主 README。

## 训练、推理和权重保存

初始化同时使用 NumPy 和 PyTorch。需要可重复的初始化时，请在构建网络前为两者设置种子：

```python
import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)
```

优化时使用 `net.train()`，推理时结合 `net.eval()` 与 `torch.no_grad()`。进入评测模式会展开并缓存滤波器；回到训练模式时会清除缓存。

公开卷积层保存 `Basis`、`weights` 和 `c`，不保存派生的评测缓存。因此，其 state dictionary 可在训练或评测模式下加载；在评测模式下加载会刷新滤波器。共享 BatchNorm 另外保存其常规参数和运行统计量。自定义网络使用普通 PyTorch state dictionary 保存和加载：

```python
torch.save(net.state_dict(), "my_network.pt")
# 加载前，先以相同结构创建 restored。
restored.load_state_dict(torch.load("my_network.pt", map_location="cpu"))
restored.eval()
```

只加载可信来源的权重。自定义网络需要针对相应任务训练；论文权重属于各自对应的完整骨干网络，而不是上面的演示网络。

## 与论文模型的关系

公开层 API 封装了 DeNet/MemNet 使用的原始 FRE-Conv 实现，不为可学习参数增加子模块名称前缀：

| 公开层 | 底层实现 |
| --- | --- |
| `FREConvLift3d` | `FCNN_3D_XY.Fconv_3D_PCA`，`ifIni=1` |
| `FREConv3d` | `FCNN_3D_XY.Fconv_3D_PCA`，`ifIni=0` |
| `FREConvReadout3d` | `FCNN_3D_XY.Fconv_3D_PCA_out` |
| `FREBatchNorm3d` | `FCNN_3D_XY.F_BN_3D` |

底层名称 `sizeP`、`inNum`、`outNum` 和 `tranNum` 分别对应 `kernel_size`、`in_channels`、`out_channels` 和 `rotations`。公开 API 使用完整基函数配置，即 `inP=kernel_size`。

QRNN 和 MAN 使用独立的 `FCNN_3D_XY_new.py` 实现，其初始化、掩模和偏置处理有所不同。十二个论文模型保留各自的实现与权重约定。使用已发布权重时，通过 `build_model` 选择对应模型，并使用文档中的评测命令；不要在加载权重前将其中的层替换为公开封装层。

```python
from freconv import build_model

paper_model = build_model("fman_t4")
```

## 安装和分发库

单独使用算子时，在已选好 PyTorch 构建的 Python 3.9 环境中，从本地仓库安装：

```bash
python -m pip install .
```

这会安装核心依赖 NumPy 和 PyTorch。导入算子不会加载论文骨干网络，也不需要图像或数据工具。论文工作流使用主 README 中的参考环境和可编辑安装；也可以先选择合适的 PyTorch/torchvision 构建，再用 `python -m pip install -e ".[experiments]"` 安装实验依赖。

使用标准 Python 构建工具生成 wheel 和源码发行包：

```bash
python -m pip install build
python -m build
```

`dist/` 中的 wheel 提供可导入的库，可通过 `python -m pip install dist/freconv-1.0.0rc1-py3-none-any.whl` 安装。论文实验命令需要仓库中的 `data/`、`checkpoints/` 和 `results/` 资源，因此应在可编辑安装的仓库或解压的源码发行包中运行。数据集和预训练权重按照项目指南单独获取。
