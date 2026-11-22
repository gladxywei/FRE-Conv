import torch.nn as nn
import torch.nn.init as init

from . import logging

logger = logging.get_logger()


def adjust_learning_rate(optimizer, lr):
    logger.info("Adjust Learning Rate => %.4e" % lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def init_network(net, init_type="kn"):
    logger.info("Use init Scheme: %s" % init_type)
    if init_type != "edsr":
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                if init_type == "kn":
                    init.kaiming_normal_(m.weight, mode="fan_out")
                if init_type == "ku":
                    init.kaiming_uniform_(m.weight, mode="fan_out")
                if init_type == "xn":
                    init.xavier_normal_(m.weight)
                if init_type == "xu":
                    init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(
                m,
                (nn.BatchNorm2d, nn.BatchNorm3d),
            ):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
