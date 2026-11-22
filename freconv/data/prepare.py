"""Prepare public ICVL, Chikusei, and Washington DC Mall data layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENVI_DTYPES = {
    1: np.dtype("u1"), 2: np.dtype("i2"), 3: np.dtype("i4"),
    4: np.dtype("f4"), 5: np.dtype("f8"), 12: np.dtype("u2"),
    13: np.dtype("u4"), 14: np.dtype("i8"), 15: np.dtype("u8"),
}
CHIKUSEI_SHA256 = {
    "cube": "89715ed9a6ab725644707b2c1732642dd1634755f3914d0ecec9bc77f62a741b",
    "header": "8e68b2d13005a96d8d5e057e626c47dd8eb1993c78d4492038a49544360d5d14",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} exists; use --force to replace generated data")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def project_output(value: str) -> Path:
    path = (PROJECT_ROOT / value).resolve()
    if PROJECT_ROOT.resolve() not in path.parents:
        raise ValueError("Output must be a project-relative child path")
    return path


def numeric_arrays(mapping):
    return {
        key: np.asarray(value)
        for key, value in mapping.items()
        if not key.startswith("__") and np.asarray(value).ndim == 3
    }


def load_cube(path: Path, key: str | None):
    if path.suffix.lower() == ".npy":
        return np.load(path), path.stem
    if path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            selected = key or max(archive.files, key=lambda name: archive[name].size)
            return archive[selected], selected
    if path.suffix.lower() != ".mat":
        raise ValueError("Input must be .mat, .npy, or .npz")
    try:
        arrays = numeric_arrays(sio.loadmat(path))
    except NotImplementedError:
        with h5py.File(path, "r") as handle:
            arrays = numeric_arrays(handle)
    selected = key or max(arrays, key=lambda name: arrays[name].size)
    return arrays[selected], selected


def bands_first(cube: np.ndarray, axis: str):
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, found {cube.shape}")
    if axis == "auto":
        candidates = [index for index, size in enumerate(cube.shape) if size <= 256]
        if not candidates:
            raise ValueError("Cannot infer the spectral axis; pass --bands-axis")
        band_axis = min(candidates, key=lambda index: cube.shape[index])
    else:
        band_axis = int(axis)
    cube = np.moveaxis(cube, band_axis, 0) if band_axis else cube
    return np.ascontiguousarray(cube, dtype=np.float32), band_axis


def read_envi_header(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    values = {}
    for key in ("samples", "lines", "bands", "header offset", "data type", "byte order", "interleave"):
        match = re.search(rf"(?im)^\s*{re.escape(key)}\s*=\s*([^\r\n]+)", text)
        if not match:
            raise ValueError(f"Missing ENVI header field: {key}")
        value = match.group(1).strip().strip("{} ")
        values[key] = value.lower() if key == "interleave" else int(value)
    for key in ("reflectance scale factor", "data ignore value"):
        match = re.search(rf"(?im)^\s*{re.escape(key)}\s*=\s*([^\r\n]+)", text)
        if match:
            values[key] = float(match.group(1).strip().strip("{} "))
    return values


def open_envi_cube(data_path: Path, header_path: Path):
    header = read_envi_header(header_path)
    dtype = ENVI_DTYPES[header["data type"]].newbyteorder("<" if header["byte order"] == 0 else ">")
    bands, lines, samples = header["bands"], header["lines"], header["samples"]
    interleave = header["interleave"]
    raw_shape = {"bil": (lines, bands, samples), "bip": (lines, samples, bands), "bsq": (bands, lines, samples)}[interleave]
    expected = header["header offset"] + int(np.prod(raw_shape)) * dtype.itemsize
    if data_path.stat().st_size != expected:
        raise ValueError(f"ENVI size mismatch: expected {expected} bytes")
    raw = np.memmap(data_path, dtype=dtype, mode="r", offset=header["header offset"], shape=raw_shape)
    cube = np.moveaxis(raw, 1, 0) if interleave == "bil" else np.moveaxis(raw, 2, 0) if interleave == "bip" else raw
    return cube, header


def write_manifest(root: Path, payload: dict) -> None:
    (root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare_wdc(args) -> None:
    source, key = load_cube(args.input, args.key)
    cube, axis = bands_first(source, args.bands_axis)
    root = project_output(args.output_root)
    clean_output(root, args.force)
    real = root / "real"
    real.mkdir()
    np.save(real / "wdc_mall.npy", cube)
    write_manifest(root, {"source": {"file": args.input.name, "key": key, "bands_axis": axis, "sha256": sha256_file(args.input)}, "shape": list(cube.shape)})


def coverage8(height: int, width: int, crop_size):
    crop_h, crop_w = crop_size
    if crop_h > height or crop_w > width:
        raise ValueError("Crop is larger than the Chikusei cube")
    row_margin = min(128, max(0, (height - 2 * crop_h) // 2))
    col_margin = min(128, max(0, (width - 4 * crop_w) // 2))
    rows = [row_margin, height - crop_h - row_margin]
    cols = np.linspace(col_margin, width - crop_w - col_margin, num=4, dtype=int).tolist()
    return [(top, left) for top in rows for left in cols]


def prepare_chikusei(args) -> None:
    cube_hash = sha256_file(args.input)
    header_hash = sha256_file(args.header)
    if not args.skip_source_hash_check:
        if cube_hash != CHIKUSEI_SHA256["cube"]:
            raise ValueError(f"Chikusei cube SHA-256 mismatch: {cube_hash}")
        if header_hash != CHIKUSEI_SHA256["header"]:
            raise ValueError(f"Chikusei header SHA-256 mismatch: {header_hash}")
    cube, header = open_envi_cube(args.input, args.header)
    root = project_output(args.output_root)
    clean_output(root, args.force)
    test = root / "test"
    test.mkdir()
    crop_size = tuple(args.crop_size)
    scale = header.get("reflectance scale factor", 1.0)
    rows = []
    for index, (top, left) in enumerate(coverage8(cube.shape[1], cube.shape[2], crop_size)):
        crop = np.asarray(cube[:, top:top + crop_size[0], left:left + crop_size[1]], dtype=np.float32) / scale
        name = f"chikusei_r{top:04d}_c{left:04d}_{index:02d}.npy"
        np.save(test / name, crop)
        rows.append({"file": name, "top": top, "left": left, "shape": list(crop.shape)})
    write_manifest(
        root,
        {
            "source": {"cube": args.input.name, "header": args.header.name, "cube_sha256": cube_hash, "header_sha256": header_hash},
            "protocol": {"name": "coverage8", "crop_size": list(crop_size), "band_norm": True},
            "test": rows,
        },
    )


def inspect_icvl(args) -> None:
    root = Path(args.root)
    report = {}
    for split in ("train", "vval", "test"):
        files = sorted((root / split).glob("*.npy"))
        if not files:
            raise RuntimeError(f"No .npy files in {root / split}")
        shapes = sorted({tuple(np.load(path, mmap_mode="r").shape) for path in files})
        if any(len(shape) != 3 for shape in shapes):
            raise ValueError(f"ICVL {split} contains a non-3D cube")
        report[split] = {"files": len(files), "shapes": [list(shape) for shape in shapes]}
    print(json.dumps(report, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare FRE-Conv public datasets.")
    commands = parser.add_subparsers(dest="dataset", required=True)
    icvl = commands.add_parser("icvl", help="Validate an existing ICVL .npy layout.")
    icvl.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "ICVL")
    icvl.set_defaults(func=inspect_icvl)
    wdc = commands.add_parser("wdc_mall", help="Convert Washington DC Mall to bands-first .npy.")
    wdc.add_argument("--input", type=Path, required=True)
    wdc.add_argument("--key", default=None)
    wdc.add_argument("--bands-axis", choices=("auto", "0", "1", "2"), default="auto")
    wdc.add_argument("--output-root", default="data/WDC_Mall")
    wdc.add_argument("--force", action="store_true")
    wdc.set_defaults(func=prepare_wdc)
    chikusei = commands.add_parser("chikusei", help="Prepare the fixed coverage8 evaluation crops.")
    chikusei.add_argument("--input", type=Path, required=True)
    chikusei.add_argument("--header", type=Path, required=True)
    chikusei.add_argument("--crop-size", type=int, nargs=2, default=(512, 512))
    chikusei.add_argument("--skip-source-hash-check", action="store_true", help="Use alternative source files without enforcing the benchmark source checksums.")
    chikusei.add_argument("--output-root", default="data/Chikusei")
    chikusei.add_argument("--force", action="store_true")
    chikusei.set_defaults(func=prepare_chikusei)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
