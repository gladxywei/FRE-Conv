"""Experiment directory, manifest, and timestamped log helpers."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def log_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def slugify_note(note: str | None) -> str:
    if not note:
        return ""
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", note.strip())
    return slug.strip("_")


def make_run_id(note: str | None = None, run_id: str | None = None) -> str:
    if run_id:
        return run_id
    suffix = slugify_note(note)
    return f"{timestamp()}_{suffix}" if suffix else timestamp()


def create_experiment_dir(
    output_root: str | Path | None = None,
    note: str | None = None,
    run_id: str | None = None,
) -> Path:
    root = Path(output_root) if output_root else EXPERIMENTS_ROOT
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    output_dir = root / make_run_id(note=note, run_id=run_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    note_text = note or "No run note was provided."
    (output_dir / "note.md").write_text(f"# {output_dir.name}\n\n{note_text}\n", encoding="utf-8")
    return output_dir


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def collect_git_metadata(root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(root)

    def git_output(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

    try:
        head = git_output("rev-parse", "HEAD")
        short_head = git_output("rev-parse", "--short", "HEAD")
        status_short = git_output("status", "--short")
        return {
            "git_head": head,
            "git_short_head": short_head,
            "git_dirty": bool(status_short),
            "git_status_short": status_short,
        }
    except Exception as error:
        return {
            "git_head": None,
            "git_short_head": None,
            "git_dirty": None,
            "git_status_short": "",
            "git_error": str(error),
        }


def append_log(path: str | Path, message: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{log_timestamp()} - {message.rstrip()}\n")


def display_path(path: str | Path, root: str | Path = PROJECT_ROOT) -> str:
    path = Path(path)
    root = Path(root)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
