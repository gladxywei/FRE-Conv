"""Public command-line parsers for training and evaluation."""

from __future__ import annotations

import argparse

from .models.registry import PAPER_MODELS


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--note", default=None)


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the twelve paper FRE-Conv models.")
    parser.add_argument("--model", required=True, choices=PAPER_MODELS)
    parser.add_argument("--task", choices=("gaussian", "complex", "all"), default="gaussian")
    parser.add_argument("--dataset", choices=("icvl",), default="icvl")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", choices=("l1", "l2", "smooth_l1", "sam"), default="l2")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--iterations-per-epoch", type=int, default=250)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--dry-run", action="store_true")
    add_runtime_arguments(parser)
    add_output_arguments(parser)
    return parser


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FRE-Conv on synthetic noise.")
    parser.add_argument("--dataset", choices=("icvl", "chikusei"), default="icvl")
    parser.add_argument("--checkpoint-set", default="paper-v1")
    parser.add_argument("--models", nargs="+", default=["paper12"])
    parser.add_argument("--modes", nargs="+", default=["all"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--scene-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--tile-halo", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--reference-summary", nargs="*", default=[])
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    add_runtime_arguments(parser)
    add_output_arguments(parser)
    return parser


def parse_train_args(argv=None):
    return build_train_parser().parse_args(argv)


def parse_eval_args(argv=None):
    return build_eval_parser().parse_args(argv)
