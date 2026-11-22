"""CPU tests: python -m unittest discover -s tests -v."""

import subprocess
import sys
import unittest

import numpy as np
import torch
from torch import nn

from freconv import FREBatchNorm3d, FREConv3d, FREConvLift3d, FREConvReadout3d
from freconv.models.FCNN_3D_XY import Fconv_3D_PCA, Fconv_3D_PCA_out


class OperatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        np.random.seed(0)
        torch.manual_seed(0)

    def test_shapes_gradients_and_eval(self):
        for rotations in (4, 8):
            with self.subTest(rotations=rotations):
                lift = FREConvLift3d(1, 2, rotations=rotations)
                group = FREConv3d(2, 3, rotations=rotations)
                readout = FREConvReadout3d(3, 1, rotations=rotations, bias=False)
                net = nn.Sequential(lift, FREBatchNorm3d(2, rotations=rotations), nn.ReLU(), group, readout)
                x = torch.randn(1, 1, 5, 7, 7, requires_grad=True)
                self.assertEqual(lift(x).shape, (1, 2 * rotations, 5, 7, 7))
                y = net(x)
                self.assertEqual(y.shape, x.shape)
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                for parameter in net.parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                net.eval()
                with torch.no_grad():
                    self.assertTrue(torch.isfinite(net(x)).all())
                net.train()
                self.assertFalse(hasattr(lift, "filter"))

    def test_numerical_parity_with_paper_operator(self):
        for rotations in (4, 8):
            for public, original, incoming in (
                (FREConvLift3d, lambda: Fconv_3D_PCA(3, 2, 3, tranNum=rotations, padding=1, ifIni=1), 2),
                (FREConv3d, lambda: Fconv_3D_PCA(3, 2, 3, tranNum=rotations, padding=1), 2 * rotations),
                (FREConvReadout3d, lambda: Fconv_3D_PCA_out(3, 2, 3, tranNum=rotations, padding=1), 2 * rotations),
            ):
                with self.subTest(rotations=rotations, layer=public.__name__):
                    layer = public(2, 3, rotations=rotations)
                    reference = original()
                    reference.load_state_dict(layer.state_dict(), strict=True)
                    x = torch.randn(1, incoming, 5, 7, 7)
                    self.assertTrue(torch.equal(layer(x), reference(x)))
                    layer.eval()
                    reference.eval()
                    self.assertTrue(torch.equal(layer(x), reference(x)))

    def test_checkpoint_modes_and_cache_refresh(self):
        for cls in (FREConvLift3d, FREConv3d, FREConvReadout3d):
            for source_eval in (False, True):
                for target_eval in (False, True):
                    with self.subTest(layer=cls.__name__, source_eval=source_eval, target_eval=target_eval):
                        source, target = cls(1, 2, rotations=4), cls(1, 2, rotations=4)
                        if source_eval:
                            source.eval()
                        if target_eval:
                            target.eval()
                        state = source.state_dict()
                        self.assertEqual(set(state), {"weights", "c", "Basis"})
                        target.load_state_dict(state, strict=True)
                        source.eval()
                        target.eval()
                        x = torch.randn(1, 1 if cls is FREConvLift3d else 4, 5, 7, 7)
                        self.assertTrue(torch.equal(source(x), target(x)))

    def test_dtype_and_stride(self):
        layer = FREConvLift3d(1, 2, rotations=4, stride=(1, 2, 2)).eval().double()
        y = layer(torch.randn(1, 1, 5, 7, 7, dtype=torch.float64))
        self.assertEqual(y.shape, (1, 8, 5, 4, 4))
        self.assertEqual(y.dtype, torch.float64)

    def test_invalid_arguments_and_layout(self):
        for kwargs in ({"rotations": 3}, {"rotations": 8.0}, {"kernel_size": 1},
                       {"kernel_size": 4}, {"padding": -1}, {"stride": 0}):
            with self.assertRaises(ValueError):
                FREConv3d(1, 2, **kwargs)
        with self.assertRaises(ValueError):
            FREConv3d(2, 3, rotations=4)(torch.zeros(1, 2, 5, 7, 7))
        with self.assertRaises(ValueError):
            FREBatchNorm3d(2, rotations=4)(torch.zeros(1, 2, 5, 7, 7))

    def test_import_does_not_load_backbones(self):
        code = ("from freconv import FREConv3d; import sys; "
                "assert 'freconv.models.registry' not in sys.modules; "
                "assert 'freconv.models.man' not in sys.modules; "
                "assert 'torchvision' not in sys.modules")
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_registry_compatibility(self):
        from freconv import MODEL_REGISTRY, PAPER_MODELS, build_model
        from freconv.models import build_model as original_build
        self.assertIs(build_model, original_build)
        self.assertEqual(len(MODEL_REGISTRY), 12)
        self.assertEqual(len(PAPER_MODELS), 12)


if __name__ == "__main__":
    unittest.main()
