"""Download, safely extract, and verify the paper checkpoint archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "checkpoints" / "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights(root: Path, manifest: dict) -> None:
    errors = []
    for item in manifest["weights"]:
        path = root / item["target_path"]
        if not path.is_file():
            errors.append(f"missing: {item['target_path']}")
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != item["bytes"]:
            errors.append(f"size: {item['target_path']} expected {item['bytes']} found {size}")
        if digest != item["sha256"]:
            errors.append(f"sha256: {item['target_path']} expected {item['sha256']} found {digest}")
    if errors:
        raise RuntimeError("Checkpoint verification failed:\n" + "\n".join(errors))


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Links are not allowed in the archive: {member.name}")
        handle.extractall(destination)


def build_parser():
    parser = argparse.ArgumentParser(description="Download the FRE-Conv paper checkpoints.")
    parser.add_argument("--release-tag", help="Override the release tag in checkpoints/manifest.json (the archive must still match its checksums).")
    parser.add_argument("--archive", type=Path, help="Verify and extract a local archive instead of downloading.")
    parser.add_argument("--verify-only", action="store_true", help="Verify files already present under checkpoints/paper-v1.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if args.verify_only:
        verify_weights(PROJECT_ROOT, manifest)
        print(f"Verified {manifest['weight_count']} checkpoint files.")
        return 0
    if manifest["archive"]["availability"] != "available" and args.archive is None:
        raise SystemExit(
            "Hosted download is unavailable according to checkpoints/manifest.json; "
            "use --archive to install a matching local archive."
        )
    release_tag = args.release_tag or manifest["archive"].get("release_tag")
    if args.archive is None and not release_tag:
        raise SystemExit("--release-tag is required for a hosted archive")

    target = PROJECT_ROOT / "checkpoints" / "paper-v1"
    if target.exists():
        raise FileExistsError(f"{target} already exists; use --verify-only to check it. It was not overwritten.")

    with tempfile.TemporaryDirectory(prefix="freconv-checkpoints-") as temporary:
        temporary_root = Path(temporary)
        archive = args.archive
        if archive is None:
            archive = temporary_root / manifest["archive"]["filename"]
            url = manifest["archive"]["url_template"].replace("<release-tag>", release_tag)
            print(f"Downloading {manifest['archive']['filename']} from release {release_tag}...", flush=True)
            urllib.request.urlretrieve(url, archive)
        archive = archive.resolve()
        expected_archive_bytes = manifest["archive"].get("archive_bytes")
        if expected_archive_bytes is not None and archive.stat().st_size != expected_archive_bytes:
            raise RuntimeError("Archive byte-size verification failed")
        expected_archive_hash = manifest["archive"].get("archive_sha256")
        if expected_archive_hash and sha256_file(archive) != expected_archive_hash:
            raise RuntimeError("Archive SHA-256 verification failed")
        extracted = temporary_root / "extracted"
        extracted.mkdir()
        safe_extract(archive, extracted)
        verify_weights(extracted, manifest)
        if target.exists():
            raise FileExistsError(f"{target} already exists; it was not overwritten")
        source = extracted / "checkpoints" / "paper-v1"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    verify_weights(PROJECT_ROOT, manifest)
    print(f"Installed and verified {manifest['weight_count']} checkpoint files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
