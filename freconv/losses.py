import torch.nn as nn

from . import utils


__all__ = ["SAMLoss"]


class SAMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return utils.sam(input, target).mean()
