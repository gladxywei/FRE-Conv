#!/usr/bin/env python3
"""Official FRE-Conv evaluation entrypoint."""

from __future__ import annotations

import csv
import random
import time
import traceback
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torchvision.transforms as tv_transforms
import yaml
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from .data import datasets as ds_utils
from .inference import spatial_tiled_inference
from .data.transforms import (
    AddNoiseDeadline,
    AddNoiseImpulse,
    AddNoiseMixed,
    AddNoiseNonIID,
    AddNoiseStripe,
    GaussianBlindNoiseLevel,
    GaussianNoiseLevel,
)
from .experiment import append_log, collect_git_metadata, create_experiment_dir, display_path, write_json
from . import utils
from .models.registry import PAPER_MODELS, MODEL_REGISTRY, build_model
from .cli import parse_eval_args


GAUSSIAN_MODES = ("30", "50", "70", "blind")
COMPLEX_MODES = ("case1", "case2", "case3", "case4", "case5")
ALL_MODES = GAUSSIAN_MODES + COMPLEX_MODES


class Bandwise:
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, x, y):
        values = []
        for channel in range(x.shape[-3]):
            x_channel = torch.squeeze(x[..., channel, :, :].data).cpu().numpy()
            y_channel = torch.squeeze(y[..., channel, :, :].data).cpu().numpy()
            values.append(self.index_fn(x_channel, y_channel))
        return values


CAL_BWSSIM = Bandwise(compare_ssim)
CAL_BWPSNR = Bandwise(partial(compare_psnr, data_range=1))


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_models(requested, checkpoint_registry, checkpoint_set):
    if requested == ["paper12"]:
        return list(PAPER_MODELS)
    if len(requested) == 1 and requested[0] in checkpoint_registry.get("model_groups", {}):
        return list(checkpoint_registry["model_groups"][requested[0]])
    if requested == ["all"]:
        models = checkpoint_registry["checkpoint_sets"][checkpoint_set]["models"].keys()
        return list(models)
    invalid = [name for name in requested if name not in MODEL_REGISTRY]
    if invalid:
        raise SystemExit(f"未知模型: {', '.join(invalid)}")
    return list(dict.fromkeys(requested))


def resolve_modes(requested):
    if requested == ["all"]:
        return list(ALL_MODES)
    invalid = [mode for mode in requested if mode not in ALL_MODES]
    if invalid:
        raise SystemExit(f"未知噪声模式: {', '.join(invalid)}")
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


def build_transform(mode):
    if mode in ("30", "50", "70"):
        return GaussianNoiseLevel(int(mode))
    if mode == "blind":
        return GaussianBlindNoiseLevel(sigmas=[30, 50, 70])
    if mode == "case1":
        return AddNoiseNonIID()
    if mode == "case2":
        return tv_transforms.Compose([AddNoiseNonIID(), AddNoiseStripe()])
    if mode == "case3":
        return tv_transforms.Compose([AddNoiseNonIID(), AddNoiseDeadline()])
    if mode == "case4":
        return tv_transforms.Compose([AddNoiseNonIID(), AddNoiseImpulse()])
    if mode == "case5":
        return tv_transforms.Compose([AddNoiseNonIID(), AddNoiseMixed()])
    raise ValueError(f"Unsupported mode: {mode}")


def cal_sam(x, y, eps=1e-8):
    x = torch.squeeze(x.data).cpu().numpy()
    y = torch.squeeze(y.data).cpu().numpy()
    tmp = (np.sum(x * y, axis=0) + eps) / (np.sqrt(np.sum(x**2, axis=0)) + eps)
    tmp = tmp / (np.sqrt(np.sum(y**2, axis=0)) + eps)
    return np.mean(np.real(np.arccos(tmp)))


def msiqa(x, y):
    psnr = np.mean(CAL_BWPSNR(x, y))
    ssim = np.mean(CAL_BWSSIM(x, y))
    sam = cal_sam(x, y)
    return psnr, ssim, sam


def to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def make_cla(args, model_name, device, no_cuda):
    return SimpleNamespace(
        arch=model_name,
        prefix=model_name,
        device=device,
        gpu=args.gpu,
        rank=0,
        no_cuda=no_cuda,
        seed=args.seed,
        tile_size=args.tile_size,
        tile_halo=args.tile_halo,
    )


def resolve_crop_size(args, dataset_entry):
    if args.crop_size is not None:
        return tuple(args.crop_size)
    crop_size = dataset_entry.get("default_eval_crop_size", [512, 512])
    return tuple(crop_size)


def load_model(model_name, weight_path, device):
    net = build_model(model_name)
    net.to(device)
    net.eval()
    checkpoint = torch.load(str(weight_path), map_location=device)
    net.load_state_dict(checkpoint["net"], strict=True)
    return net


def evaluate_loader(loader, network, cla, save=False):
    psnrs = []
    my_psnrs = []
    my_ssims = []
    my_sams = []
    noise_imgs = []
    predict_imgs = []
    groundtruth_imgs = []

    with torch.no_grad():
        for inputs, targets in loader:
            if not cla.no_cuda and torch.cuda.is_available():
                inputs, targets = inputs.to(cla.device), targets.to(cla.device)

            if cla.tile_size is None:
                outputs = network(inputs)
            else:
                outputs = spatial_tiled_inference(
                    network,
                    inputs,
                    tile_size=cla.tile_size,
                    halo=cla.tile_halo,
                )
            outputs = outputs.squeeze(1)
            targets = targets.squeeze(1)

            if save:
                noise_imgs.append(np.squeeze(inputs.detach().cpu().numpy()))
                groundtruth_imgs.append(np.squeeze(targets.detach().cpu().numpy()))
                predict_imgs.append(np.squeeze(outputs.detach().cpu().numpy()))

            batch_psnrs = []
            for index in range(outputs.shape[0]):
                batch_psnrs.append(
                    to_float(utils.peak_snr(outputs[index], targets[index], bandwise=False, band_dim=-3))
                )
            psnrs.extend(batch_psnrs)

            my_psnr, my_ssim, my_sam = msiqa(outputs, targets)
            my_psnrs.append(my_psnr)
            my_ssims.append(my_ssim)
            my_sams.append(my_sam)

    result = (
        sum(psnrs) / float(len(psnrs)),
        sum(my_psnrs) / float(len(my_psnrs)),
        sum(my_ssims) / float(len(my_ssims)),
        sum(my_sams) / float(len(my_sams)),
    )
    if not save:
        return result

    noise_imgs = np.array(noise_imgs, dtype=np.float32).squeeze()
    noise_imgs = np.transpose(noise_imgs, (1, 2, 0))
    predict_imgs = np.array(predict_imgs, dtype=np.float32).squeeze()
    predict_imgs = np.transpose(predict_imgs, (1, 2, 0))
    groundtruth_imgs = np.array(groundtruth_imgs, dtype=np.float32).squeeze()
    groundtruth_imgs = np.transpose(groundtruth_imgs, (1, 2, 0))
    return noise_imgs, groundtruth_imgs, predict_imgs, *result


def image_dirs(output_dir, model_name, mode):
    return {
        "noise": output_dir / "noise" / model_name / mode,
        "predict": output_dir / "predict" / model_name / mode,
        "groundth": output_dir / "groundth" / model_name / mode,
    }


def ensure_image_dirs(dirs):
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=False)
    (dirs["groundth"] / "gray").mkdir(parents=True, exist_ok=False)


def save_scene_images(dirs, file_name, noise_img, groundth_img, predict_img):
    cv2.imwrite(str(dirs["noise"] / f"{file_name}.png"), (noise_img[:, :, 12] * 255).astype(np.uint8))
    cv2.imwrite(str(dirs["predict"] / f"{file_name}.png"), (predict_img[:, :, 12] * 255).astype(np.uint8))
    cv2.imwrite(
        str(dirs["groundth"] / "gray" / f"{file_name}.png"),
        (groundth_img[:, :, 12] * 255).astype(np.uint8),
    )
    cv2.imwrite(str(dirs["noise"] / f"{file_name}_c.png"), (noise_img[:, :, [30, 10, 5]] * 255).astype(np.uint8))
    cv2.imwrite(
        str(dirs["predict"] / f"{file_name}_c.png"),
        (predict_img[:, :, [30, 10, 5]] * 255).astype(np.uint8),
    )
    cv2.imwrite(
        str(dirs["groundth"] / f"{file_name}_c.png"),
        (groundth_img[:, :, [30, 10, 5]] * 255).astype(np.uint8),
    )


def read_reference_summaries(paths):
    references = {}
    for path in paths:
        with Path(path).open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") == "OK":
                    references[(row["model"], row["mode"])] = {
                        "my_psnr": float(row["my_psnr"]),
                        "ssim": float(row["ssim"]),
                        "sam": float(row["sam"]),
                        "psnr": float(row["psnr"]),
                    }
    return references


def add_reference(row, references, tolerance):
    comparison = dict(row)
    reference = references.get((row["model"], row["mode"]))
    if row["status"] != "OK":
        comparison["comparison_status"] = "FAILED"
        return comparison
    if not reference:
        comparison["comparison_status"] = "NO_REFERENCE"
        return comparison
    deltas = {
        "delta_my_psnr": row["my_psnr"] - reference["my_psnr"],
        "delta_ssim": row["ssim"] - reference["ssim"],
        "delta_sam": row["sam"] - reference["sam"],
        "delta_psnr": row["psnr"] - reference["psnr"],
    }
    comparison.update({f"reference_{key}": value for key, value in reference.items()})
    comparison.update(deltas)
    comparison["comparison_status"] = "OK" if max(abs(value) for value in deltas.values()) <= tolerance else "DIFF"
    return comparison


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def failure_row(model_name, mode, error):
    return {
        "model": model_name,
        "mode": mode,
        "status": "FAILED",
        "weight_path": "",
        "scenes": 0,
        "seconds": 0,
        "my_psnr": "",
        "ssim": "",
        "sam": "",
        "psnr": "",
        "error": str(error),
    }


def run_one(args, output_dir, model_name, mode, dataset_entry, checkpoint_entry, device, no_cuda):
    log_path = output_dir / "logs" / model_name / mode / "log.txt"
    kind = "gaussian" if mode in GAUSSIAN_MODES else "complex"
    weight_info = checkpoint_entry["models"][model_name][kind]
    weight_path = resolve_project_path(weight_info["path"])
    split_entry = dataset_entry["splits"].get(args.split)
    if split_entry is None:
        raise ValueError(f"Dataset split not found: {args.split}")
    test_datadir = resolve_project_path(split_entry["path"])
    crop_size = resolve_crop_size(args, dataset_entry)

    seed_everything(args.seed)
    cla = make_cla(args, model_name, device, no_cuda)
    append_log(log_path, f"Args: {vars(args)}")
    append_log(log_path, f"Init weight from {display_path(weight_path)}")
    append_log(log_path, f"Mode: {mode}")
    append_log(log_path, f"Split: {args.split}")
    append_log(log_path, f"Device: {device}")
    append_log(log_path, f"Crop size: {crop_size}")
    append_log(log_path, f"Tile size: {args.tile_size} | Tile halo: {args.tile_halo}")

    net = load_model(model_name, weight_path, device)
    num_params = sum(param.numel() for param in net.parameters() if param.requires_grad)
    append_log(log_path, f"Network: {model_name} | Number of parameters: {num_params}")

    sort_files = dataset_entry.get("deterministic_file_order", False)
    file_names = ds_utils.get_test_name(str(test_datadir), sort_files=sort_files)
    if args.scene_limit is not None:
        file_names = file_names[: args.scene_limit]
    if not file_names:
        raise RuntimeError(f"No test scenes found in {test_datadir}")

    dirs = image_dirs(output_dir, model_name, mode)
    if args.save_images:
        ensure_image_dirs(dirs)

    transform = build_transform(mode)
    totals = {"psnr": 0.0, "my_psnr": 0.0, "ssim": 0.0, "sam": 0.0}
    scene_rows = []
    sam_count = 0
    start = time.time()

    for scene_index, name in enumerate(file_names):
        # Keep stochastic noise deterministic and model-independent.  The
        # network constructor consumes RNG before weights are loaded, and
        # multi-worker DataLoaders seed NumPy from the current torch RNG state.
        # Resetting per scene prevents blind noise from collapsing to a single
        # sigma and lets every model see the same noisy scene for a mode.
        seed_everything(args.seed + scene_index)
        dataset = ds_utils.ICVLTestDataset(
            str(test_datadir),
            transform=transform,
            crop_size=crop_size,
            band_norm=dataset_entry.get("band_norm", True),
            scene_id=scene_index,
            sort_files=sort_files,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        if args.save_images:
            noise_img, groundth_img, predict_img, psnr, my_psnr, ssim, sam = evaluate_loader(loader, net, cla, save=True)
            save_scene_images(dirs, name, noise_img, groundth_img, predict_img)
        else:
            psnr, my_psnr, ssim, sam = evaluate_loader(loader, net, cla, save=False)

        if np.isnan(sam):
            sam = 0
        else:
            sam_count += 1
        append_log(log_path, f"{name} | PSNR: {psnr} | MY_PSNR: {my_psnr} | SSIM: {ssim} | SAM: {sam}")
        scene_rows.append(
            {
                "model": model_name,
                "mode": mode,
                "scene_index": scene_index,
                "scene": name,
                "my_psnr": my_psnr,
                "ssim": ssim,
                "sam": sam,
                "psnr": psnr,
            }
        )
        totals["psnr"] += psnr
        totals["my_psnr"] += my_psnr
        totals["ssim"] += ssim
        totals["sam"] += sam

        if device.type == "cuda":
            torch.cuda.empty_cache()

    count = len(file_names)
    sam_divisor = sam_count if sam_count else count
    row = {
        "model": model_name,
        "mode": mode,
        "status": "OK",
        "weight_path": display_path(weight_path),
        "scenes": count,
        "seconds": round(time.time() - start, 3),
        "my_psnr": totals["my_psnr"] / count,
        "ssim": totals["ssim"] / count,
        "sam": totals["sam"] / sam_divisor,
        "psnr": totals["psnr"] / count,
        "error": "",
    }
    append_log(
        log_path,
        "Average ({mode}) MY_PSNR: {my_psnr} | SSIM: {ssim} | SAM: {sam} | PSNR : {psnr}".format(
            mode=mode,
            my_psnr=row["my_psnr"],
            ssim=row["ssim"],
            sam=row["sam"],
            psnr=row["psnr"],
        ),
    )
    return row, scene_rows


def main(argv=None):
    args = parse_eval_args(argv)
    data_registry = read_yaml(PROJECT_ROOT / "data" / "registry.yaml")
    checkpoint_registry = read_yaml(PROJECT_ROOT / "checkpoints" / "registry.yaml")
    dataset_entry = data_registry["datasets"][args.dataset]
    checkpoint_entry = checkpoint_registry["checkpoint_sets"][args.checkpoint_set]
    selected_models = resolve_models(args.models, checkpoint_registry, args.checkpoint_set)
    selected_modes = resolve_modes(args.modes)
    references = read_reference_summaries(args.reference_summary)

    output_dir = create_experiment_dir(args.output_root, note=args.note, run_id=args.run_id)
    device, no_cuda = choose_device(args)
    torch.backends.cudnn.benchmark = True

    write_json(
        output_dir / "run_manifest.json",
        {
            "entrypoint": "python -m freconv.evaluate",
            "output_dir": display_path(output_dir),
            "args": vars(args),
            "models": selected_models,
            "modes": selected_modes,
            "split": args.split,
            "dataset": dataset_entry,
            "checkpoint_set": args.checkpoint_set,
            "device": str(device),
            "effective_crop_size": list(resolve_crop_size(args, dataset_entry)),
            **collect_git_metadata(),
        },
    )

    rows = []
    comparisons = []
    scene_rows = []
    for model_name in selected_models:
        for mode in selected_modes:
            print(f"[RUN] {model_name} {mode}", flush=True)
            try:
                row, current_scene_rows = run_one(
                    args, output_dir, model_name, mode, dataset_entry, checkpoint_entry, device, no_cuda
                )
                scene_rows.extend(current_scene_rows)
            except Exception as error:
                row = failure_row(model_name, mode, error)
                err_log = output_dir / "logs" / model_name / mode / "log.txt"
                append_log(err_log, "FAILED")
                append_log(err_log, traceback.format_exc())
                print(f"[FAILED] {model_name} {mode}: {error}", flush=True)
                if args.fail_fast:
                    raise

            rows.append(row)
            comparisons.append(add_reference(row, references, args.tolerance))
            write_csv(
                output_dir / "summary.csv",
                rows,
                ["model", "mode", "status", "weight_path", "scenes", "seconds", "my_psnr", "ssim", "sam", "psnr", "error"],
            )
            write_csv(
                output_dir / "comparison.csv",
                comparisons,
                [
                    "model",
                    "mode",
                    "status",
                    "comparison_status",
                    "weight_path",
                    "scenes",
                    "seconds",
                    "my_psnr",
                    "reference_my_psnr",
                    "delta_my_psnr",
                    "ssim",
                    "reference_ssim",
                    "delta_ssim",
                    "sam",
                    "reference_sam",
                    "delta_sam",
                    "psnr",
                    "reference_psnr",
                    "delta_psnr",
                    "error",
                ],
            )
            write_csv(
                output_dir / "scene_metrics.csv",
                scene_rows,
                ["model", "mode", "scene_index", "scene", "my_psnr", "ssim", "sam", "psnr"],
            )

    failed = sum(1 for row in rows if row["status"] != "OK")
    diff = sum(1 for row in comparisons if row.get("comparison_status") == "DIFF")
    print(f"Done. output={output_dir} failed={failed} diff={diff}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
