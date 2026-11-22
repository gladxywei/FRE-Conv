"""Unified Gaussian and complex-noise training entrypoint.

Gaussian and complex-noise tasks share setup, seeding, optimization,
validation, and checkpoint handling.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tv_transforms
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader

from . import helper, training_utils
from .cli import parse_train_args
from .data.datasets import ICVLDataset
from .data.transforms import (
    AddNoiseDeadline,
    AddNoiseImpulse,
    AddNoiseNonIID,
    AddNoiseStripe,
    GaussianBlindNoiseLevel,
    RandChoice,
)
from .experiment import create_experiment_dir, display_path, write_json
from .losses import SAMLoss
from .models.registry import MODEL_REGISTRY, build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = {
    "gaussian": {
        "seed_base": 2022,
        "loader_seed": 0,
        "epochs": 100,
        "batch_size": 16,
        "lr_drop_epoch": 50,
        "checkpoint_rule": "validation PSNR best and validation loss best",
    },
    "complex": {
        "seed_base": 2022,
        "loader_seed": 0,
        "epochs": 100,
        "batch_size": 16,
        "lr_drop_epoch": 50,
        "checkpoint_rule": "non-IID and mixture validation PSNR/loss best",
    },
}


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def criterion_for(name: str):
    return {
        "l1": nn.L1Loss,
        "l2": nn.MSELoss,
        "smooth_l1": nn.SmoothL1Loss,
        "sam": SAMLoss,
    }[name]()


def choose_device(args):
    if args.device == "cpu":
        return torch.device("cpu"), True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch.cuda.set_device(args.gpu)
    return torch.device("cuda", args.gpu), False


def task_transforms(task: str):
    if task == "gaussian":
        blind = GaussianBlindNoiseLevel()
        return blind, {"blind": blind}
    mixture = tv_transforms.Compose(
        [
            AddNoiseNonIID(),
            RandChoice([AddNoiseImpulse(), AddNoiseStripe(), AddNoiseDeadline()], p=None, combos=True),
        ]
    )
    return mixture, {"noniid": AddNoiseNonIID(), "mixture": mixture}


def make_loaders(args, task: str, dataset_entry: dict, device):
    train_transform, validation_transforms = task_transforms(task)
    train_dir = project_path(dataset_entry["splits"]["train"]["path"])
    val_dir = project_path(dataset_entry["splits"]["vval"]["path"])
    crop_size = tuple(dataset_entry.get("train_crop_size", [64, 64]))
    val_crop_size = tuple(dataset_entry.get("validation_crop_size", [64, 64]))
    band_norm = dataset_entry.get("band_norm", True)
    batch_size = args.batch_size or PROTOCOLS[task]["batch_size"]
    persistent = args.num_workers > 0

    train_dataset = ICVLDataset(
        train_dir,
        transform=train_transform,
        crop_size=crop_size,
        band_norm=band_norm,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent,
    )
    val_loaders = {}
    for name, transform in validation_transforms.items():
        dataset = ICVLDataset(
            val_dir,
            transform=transform,
            val=True,
            crop_size=val_crop_size,
            band_norm=band_norm,
        )
        val_loaders[name] = DataLoader(
            dataset,
            batch_size=args.val_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    return train_loader, val_loaders


def checkpoint_path(run_dir: Path, model: str, name: str, loss: str) -> Path:
    return run_dir / model / f"{model}-{name}-{loss}.pth"


def run_task(args, task: str, run_dir: Path, dataset_entry: dict) -> None:
    protocol = PROTOCOLS[task]
    device, no_cuda = choose_device(args)
    seed_everything(args.seed)
    network = build_model(args.model)
    if MODEL_REGISTRY[args.model].variant == "baseline":
        helper.init_network(network, "xn")
    network = network.to(device)
    criterion = criterion_for(args.loss).to(device)
    optimizer = Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=False)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(str(project_path(args.resume)), map_location=device)
        network.load_state_dict(checkpoint["net"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1

    seed_everything(protocol["loader_seed"])
    torch.backends.cudnn.benchmark = device.type == "cuda"
    train_loader, val_loaders = make_loaders(args, task, dataset_entry, device)
    cla = SimpleNamespace(
        arch=args.model,
        prefix=args.model,
        device=device,
        gpu=args.gpu,
        rank=0,
        no_cuda=no_cuda,
        save_dir=str(run_dir),
        loss=args.loss,
    )
    best = {name: {"psnr": float("-inf"), "loss": float("inf")} for name in val_loaders}
    max_epochs = args.max_epochs or protocol["epochs"]
    for epoch in range(start_epoch, max_epochs):
        seed_everything(epoch + protocol["seed_base"])
        lr = args.lr * (0.1 if epoch >= protocol["lr_drop_epoch"] else 1.0)
        helper.adjust_learning_rate(optimizer, lr)
        training_utils.train(
            train_loader,
            network,
            cla,
            epoch,
            optimizer,
            criterion,
            iterations=args.iterations_per_epoch,
        )
        for name, loader in val_loaders.items():
            psnr, loss = training_utils.validate(loader, f"validate_{name}", network, cla, epoch, criterion)
            prefix = "gaussian-noise" if task == "gaussian" else f"complex-noise-{name}"
            if psnr > best[name]["psnr"]:
                best[name]["psnr"] = psnr
                training_utils.save_checkpoint(
                    cla,
                    epoch,
                    network,
                    optimizer,
                    model_out_path=str(checkpoint_path(run_dir, args.model, f"{prefix}-psnr-best", args.loss)),
                )
            if loss < best[name]["loss"]:
                best[name]["loss"] = loss
                training_utils.save_checkpoint(
                    cla,
                    epoch,
                    network,
                    optimizer,
                    model_out_path=str(checkpoint_path(run_dir, args.model, f"{prefix}-loss-best", args.loss)),
                )
        if (epoch + 1) % 10 == 0:
            training_utils.save_checkpoint(
                cla,
                epoch,
                network,
                optimizer,
                model_out_path=str(checkpoint_path(run_dir, args.model, f"hsidenoising-epoch-{epoch + 1}", args.loss)),
            )


def main(argv=None):
    args = parse_train_args(argv)
    data_registry = read_yaml(PROJECT_ROOT / "data" / "registry.yaml")
    dataset_entry = data_registry["datasets"][args.dataset]
    run_dir = create_experiment_dir(args.output_root, note=args.note, run_id=args.run_id)
    tasks = ["gaussian", "complex"] if args.task == "all" else [args.task]
    write_json(
        run_dir / "run_manifest.json",
        {
            "entrypoint": "python -m freconv.train",
            "output_dir": display_path(run_dir),
            "args": vars(args),
            "tasks": tasks,
            "protocols": {task: PROTOCOLS[task] for task in tasks},
            "dataset": dataset_entry,
            "validation_scope": "runtime validation is performed only when this command is executed",
        },
    )
    if args.dry_run:
        print(json.dumps({"run_dir": display_path(run_dir), "tasks": tasks}, indent=2))
        return 0
    for task in tasks:
        task_dir = run_dir / task
        task_dir.mkdir(parents=True, exist_ok=False)
        run_task(args, task, task_dir, dataset_entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
