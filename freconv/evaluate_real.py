#!/usr/bin/env python3
"""Qualitative real-noise HSI evaluation for FRE-Conv checkpoints."""

from __future__ import annotations

import argparse
import csv
import random
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from .experiment import append_log, create_experiment_dir, display_path, write_json
from . import utils
from .models.registry import MODEL_REGISTRY, build_model


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def available_real_datasets():
    registry = read_yaml(PROJECT_ROOT / "data" / "registry.yaml")
    names = [name for name, entry in registry["datasets"].items() if entry.get("kind") == "real_noisy"]
    return sorted(names) or ["wdc_mall"]


def resolve_models(requested, checkpoint_registry, checkpoint_set):
    if len(requested) == 1 and requested[0] in checkpoint_registry.get("model_groups", {}):
        return list(checkpoint_registry["model_groups"][requested[0]])
    if requested == ["all"]:
        return list(checkpoint_registry["checkpoint_sets"][checkpoint_set]["models"].keys())
    invalid = [name for name in requested if name not in MODEL_REGISTRY]
    if invalid:
        raise SystemExit(f"未知模型: {', '.join(invalid)}")
    return list(dict.fromkeys(requested))


def choose_device(args):
    if args.device == "cpu":
        return torch.device("cpu"), True
    if not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但 torch.cuda.is_available() 为 False。")
    torch.cuda.set_device(args.gpu)
    return torch.device("cuda", args.gpu), False


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_model(model_name, weight_path, device):
    net = build_model(model_name)
    net.to(device)
    net.eval()
    checkpoint = torch.load(str(weight_path), map_location=device)
    net.load_state_dict(checkpoint["net"], strict=True)
    return net


def make_cla(args, model_name, device, no_cuda):
    return SimpleNamespace(
        arch=model_name,
        prefix=model_name,
        device=device,
        gpu=args.gpu,
        rank=0,
        no_cuda=no_cuda,
        seed=args.seed,
    )


def center_crop_cube(cube: np.ndarray, crop_size: tuple[int, int] | None):
    if crop_size is None:
        return cube
    crop_h, crop_w = crop_size
    _, height, width = cube.shape
    if crop_h > height or crop_w > width:
        raise ValueError(f"Crop {crop_size} exceeds cube spatial shape {(height, width)}")
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return cube[:, top : top + crop_h, left : left + crop_w]


def normalize_input(cube: np.ndarray, band_norm: bool):
    tensor = torch.tensor(cube, dtype=torch.float32).unsqueeze(0)
    tensor, _ = utils.normalize(tensor, by_band=band_norm, band_dim=-3)
    return tensor.unsqueeze(0)


def denoise_cube(cube: np.ndarray, network, device, no_cuda: bool, band_norm: bool):
    inputs = normalize_input(cube, band_norm=band_norm)
    if not no_cuda and torch.cuda.is_available():
        inputs = inputs.to(device)
    with torch.no_grad():
        outputs = network(inputs).squeeze(0).squeeze(0)
    before = inputs.squeeze(0).squeeze(0).detach().cpu().numpy()
    after = outputs.detach().cpu().numpy()
    residual = before - after
    return before, after, residual


def spectral_roughness(cube: np.ndarray) -> float:
    if cube.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(cube, axis=0))))


def robust_image(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    low, high = np.percentile(array[np.isfinite(array)], [1, 99])
    if high <= low:
        high = low + 1.0
    image = np.clip((array - low) / (high - low), 0, 1)
    return (image * 255).astype(np.uint8)


def valid_bands(requested, bands: int):
    return [band for band in requested if 0 <= band < bands]


def save_visualizations(output_dir: Path, model_name: str, scene_name: str, before, after, residual, gray_bands, rgb_bands):
    vis_dir = output_dir / "visualizations" / model_name / scene_name
    vis_dir.mkdir(parents=True, exist_ok=True)
    bands = before.shape[0]
    gray = valid_bands(gray_bands, bands) or [bands // 2]
    rgb = valid_bands(rgb_bands, bands)

    for band in gray:
        cv2.imwrite(str(vis_dir / f"before_b{band:03d}.png"), robust_image(before[band]))
        cv2.imwrite(str(vis_dir / f"after_b{band:03d}.png"), robust_image(after[band]))
        cv2.imwrite(str(vis_dir / f"residual_b{band:03d}.png"), robust_image(np.abs(residual[band])))

    if len(rgb) >= 3:
        rgb = rgb[:3]
        before_rgb = np.stack([robust_image(before[band]) for band in rgb], axis=-1)
        after_rgb = np.stack([robust_image(after[band]) for band in rgb], axis=-1)
        residual_rgb = np.stack([robust_image(np.abs(residual[band])) for band in rgb], axis=-1)
        cv2.imwrite(str(vis_dir / "before_false_color.png"), before_rgb[:, :, ::-1])
        cv2.imwrite(str(vis_dir / "after_false_color.png"), after_rgb[:, :, ::-1])
        cv2.imwrite(str(vis_dir / "residual_false_color.png"), residual_rgb[:, :, ::-1])


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def failure_row(model_name, scene_name, error):
    return {
        "model": model_name,
        "scene": scene_name,
        "status": "FAILED",
        "task": "",
        "weight_path": "",
        "bands": "",
        "height": "",
        "width": "",
        "seconds": 0,
        "residual_abs_mean": "",
        "residual_std": "",
        "spectral_roughness_before": "",
        "spectral_roughness_after": "",
        "error": str(error),
    }


def run_one(args, output_dir, model_name, scene_path, dataset_entry, checkpoint_entry, device, no_cuda):
    scene_name = scene_path.stem
    log_path = output_dir / "logs" / model_name / f"{scene_name}.log"
    weight_info = checkpoint_entry["models"][model_name][args.task]
    weight_path = resolve_project_path(weight_info["path"])
    crop_size = tuple(args.crop_size) if args.crop_size else tuple(dataset_entry.get("default_eval_crop_size", []))
    if args.full_scene:
        crop_size = None

    seed_everything(args.seed)
    cla = make_cla(args, model_name, device, no_cuda)
    append_log(log_path, f"Args: {vars(args)}")
    append_log(log_path, f"Init weight from {display_path(weight_path)}")
    append_log(log_path, f"Device: {device}")
    append_log(log_path, f"Scene: {display_path(scene_path)}")
    append_log(log_path, f"Crop size: {crop_size or 'full_scene'}")

    network = load_model(model_name, weight_path, device)
    _ = cla
    cube = np.load(scene_path).astype(np.float32, copy=False)
    cube = center_crop_cube(cube, crop_size)

    start = time.time()
    before, after, residual = denoise_cube(cube, network, device, no_cuda, dataset_entry.get("band_norm", True))
    seconds = round(time.time() - start, 3)

    save_visualizations(
        output_dir,
        model_name,
        scene_name,
        before,
        after,
        residual,
        gray_bands=args.gray_bands,
        rgb_bands=args.rgb_bands,
    )
    if args.save_arrays:
        arrays_dir = output_dir / "arrays" / model_name
        arrays_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arrays_dir / f"{scene_name}.npz", before=before, after=after, residual=residual)

    row = {
        "model": model_name,
        "scene": scene_name,
        "status": "OK",
        "task": args.task,
        "weight_path": display_path(weight_path),
        "bands": before.shape[0],
        "height": before.shape[1],
        "width": before.shape[2],
        "seconds": seconds,
        "residual_abs_mean": float(np.mean(np.abs(residual))),
        "residual_std": float(np.std(residual)),
        "spectral_roughness_before": spectral_roughness(before),
        "spectral_roughness_after": spectral_roughness(after),
        "error": "",
    }
    append_log(log_path, f"Summary: {row}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def build_parser():
    parser = argparse.ArgumentParser(description="FRE-Conv real-noise HSI qualitative evaluation")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--note", default=None)
    parser.add_argument("--dataset", default="wdc_mall", choices=available_real_datasets())
    parser.add_argument("--checkpoint-set", default="paper-v1")
    parser.add_argument("--models", nargs="+", default=["baseline_t4"], help="Model names, registry group, or all.")
    parser.add_argument("--task", choices=["gaussian", "complex"], default="complex")
    parser.add_argument("--scene-limit", type=int, default=None)
    parser.add_argument("--crop-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--full-scene", action="store_true")
    parser.add_argument("--gray-bands", type=int, nargs="+", default=[12, 60, 120])
    parser.add_argument("--rgb-bands", type=int, nargs=3, default=[60, 30, 10])
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    data_registry = read_yaml(PROJECT_ROOT / "data" / "registry.yaml")
    checkpoint_registry = read_yaml(PROJECT_ROOT / "checkpoints" / "registry.yaml")
    dataset_entry = data_registry["datasets"][args.dataset]
    checkpoint_entry = checkpoint_registry["checkpoint_sets"][args.checkpoint_set]
    selected_models = resolve_models(args.models, checkpoint_registry, args.checkpoint_set)
    real_dir = resolve_project_path(dataset_entry["splits"]["real"]["path"])
    scene_paths = sorted(real_dir.glob("*.npy"))
    if args.scene_limit is not None:
        scene_paths = scene_paths[: args.scene_limit]
    if not scene_paths:
        raise RuntimeError(f"No .npy real scenes found in {real_dir}")

    output_dir = create_experiment_dir(args.output_root, note=args.note, run_id=args.run_id)
    device, no_cuda = choose_device(args)
    torch.backends.cudnn.benchmark = True
    write_json(
        output_dir / "run_manifest.json",
        {
            "entrypoint": "python -m freconv.evaluate_real",
            "output_dir": display_path(output_dir),
            "args": vars(args),
            "models": selected_models,
            "dataset": dataset_entry,
            "checkpoint_set": args.checkpoint_set,
            "device": str(device),
            "scenes": [display_path(path) for path in scene_paths],
            "metric_note": "real-noise run has no clean GT; summary contains no-reference diagnostics only",
        },
    )

    rows = []
    fieldnames = [
        "model",
        "scene",
        "status",
        "task",
        "weight_path",
        "bands",
        "height",
        "width",
        "seconds",
        "residual_abs_mean",
        "residual_std",
        "spectral_roughness_before",
        "spectral_roughness_after",
        "error",
    ]
    for model_name in selected_models:
        for scene_path in scene_paths:
            print(f"[RUN] {model_name} {scene_path.name}", flush=True)
            try:
                row = run_one(args, output_dir, model_name, scene_path, dataset_entry, checkpoint_entry, device, no_cuda)
            except Exception as error:
                row = failure_row(model_name, scene_path.stem, error)
                append_log(output_dir / "logs" / model_name / f"{scene_path.stem}.log", traceback.format_exc())
                print(f"[FAILED] {model_name} {scene_path.name}: {error}", flush=True)
                if args.fail_fast:
                    raise
            rows.append(row)
            write_csv(output_dir / "summary.csv", rows, fieldnames)

    failed = sum(1 for row in rows if row["status"] != "OK")
    print(f"Done. output={output_dir} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
