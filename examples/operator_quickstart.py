"""Run with: python examples/operator_quickstart.py (after installation)."""

import numpy as np
import torch
from torch import nn

from freconv import FREBatchNorm3d, FREConv3d, FREConvLift3d, FREConvReadout3d


def make_network(rotations=4):
    return nn.Sequential(
        FREConvLift3d(1, 4, rotations=rotations),
        FREBatchNorm3d(4, rotations=rotations),
        nn.ReLU(),
        FREConv3d(4, 4, rotations=rotations),
        nn.ReLU(),
        FREConvReadout3d(4, 1, rotations=rotations),
    )


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    network = make_network()
    cube = torch.randn(1, 1, 7, 16, 16)
    prediction = network(cube)
    prediction.square().mean().backward()
    print(f"Training: {tuple(cube.shape)} -> {tuple(prediction.shape)}; backward complete")

    network.eval()
    with torch.no_grad():
        prediction = network(cube)
    print(f"Inference: {tuple(prediction.shape)}")

    # The same state dictionary works whether saved in training or eval mode.
    restored = make_network()
    restored.load_state_dict(network.state_dict())
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(prediction, restored(cube), atol=1e-6)
    print("State-dict round trip complete")


if __name__ == "__main__":
    main()
