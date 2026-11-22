# Third-party notices

FRE-Conv builds on the research implementations and utilities listed below. We thank their authors for making this work available. Original FRE-Conv contributions are released under the [MIT License](LICENSE); third-party components retain their respective licenses and copyright notices.

## Components and licenses

| Source | Relevant portions of this package | Terms and copyright notice |
| --- | --- | --- |
| [HyDe](https://github.com/Helmholtz-AI-Energy/HyDe/tree/251a4f461cd8678462cfda09009c183da1b5a092) | DeNet/MemNet/QRNN backbones and data, noise, normalization, loss, training, logging, and initialization utilities. | [BSD-3-Clause](LICENSES/HyDe-BSD-3-Clause.txt), copyright 2021 Helmholtz AI Energy. |
| [HSIR / MAN](https://github.com/bit-isp/HSIR/blob/cf6fd69e9f839ef4262d629ce1b11884c9b39fc0/hsir/model/man/v2.py) | MAN backbone and the corresponding portions of the FRE-equipped variants. | [MIT](LICENSES/HSIR-MIT.txt), copyright 2022 Zeqiang Lai. |
| [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar/blob/49b7aa97b0c12fe0d4054e670403a16b6b834ddd/utils.py) | Progress-display rendering in `freconv/progress.py`. | [MIT](LICENSES/pytorch-cifar-MIT.txt), copyright 2017 liukuang. |
| [e2cnn](https://github.com/QUVA-Lab/e2cnn/tree/022d6ca4a78ab666f3c149d1849e324ed1d40238) | Filter-cache utilities derived through F-Conv. The e2cnn package is not a runtime dependency. | [BSD-3-Clause-Clear](LICENSES/e2cnn-BSD-3-Clause-Clear.txt), copyright 2019 Qualcomm Innovation Center, Inc. |
| [F-Conv](https://github.com/XieQi2015/F-Conv/tree/0b71ad8e0505fc825c0272509db2319ef4e17b25) | Fourier-basis parameterization, initialization, and filter-construction routines in the `FCNN_3D_XY` modules. | Used and distributed with permission. |

Full third-party license texts and copyright notices are provided in [LICENSES/](LICENSES/).

## Research credit

The backbone designs originate from HSI-DeNet (Yi Chang and coauthors), MemNet (Ying Tai and coauthors), QRNN3D (Kaixuan Wei, Ying Fu, and Hua Huang), and MAN (Zeqiang Lai and Ying Fu). FRE-Conv's Fourier-parameterization foundation is F-Conv by Qi Xie, Qian Zhao, Zongben Xu, and Deyu Meng. The backbone implementations draw on HyDe and HSIR, as listed above.

Pinned upstream revisions and the corresponding local files are recorded in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json).

## Dependencies and datasets

PyTorch, torchvision, NumPy, SciPy, h5py, PyYAML, scikit-image, OpenCV, and transitive dependencies are installed separately and retain their own licenses. Dependency declarations are in [environment.yml](environment.yml) and [pyproject.toml](pyproject.toml).

Dataset access and use follow the respective providers' terms; see the [dataset guide](data/README.md).
