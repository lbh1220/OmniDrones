#!/usr/bin/env python3

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


# =========================
# Configuration (edit here)
# =========================

# Root directory to clean
ROOT_DIR = Path("/home/liang/Projects/OmniDrones/runs")

# Dry run mode: when True, only print the files that would be deleted
DRY_RUN = False

# Directories to skip (relative to ROOT_DIR or absolute paths)
# Example: "traffic/ttc_sweep/u10e1" or "/abs/path/to/keep"
SKIP_DIRS: List[str] = [
    "traffic/ttc_sweep/u10e1",
]

# Directory base names to skip anywhere in the tree (e.g., skip all dirs named 'wandb')
SKIP_NAMES: List[str] = [
    # "wandb",
]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def normalize_skip_dirs(root_dir: Path, skip_dirs: Iterable[str]) -> List[Path]:
    normalized: List[Path] = []
    for p in skip_dirs:
        pp = Path(p)
        if not pp.is_absolute():
            pp = (root_dir / pp).resolve()
        else:
            pp = pp.resolve()
        normalized.append(pp)
    return normalized


def is_within(path: Path, ancestor: Path) -> bool:
    try:
        path.resolve().relative_to(ancestor.resolve())
        return True
    except Exception:
        return False


def main() -> int:
    root = ROOT_DIR.resolve()
    if not root.exists() or not root.is_dir():
        log(f"Error: ROOT_DIR does not exist: {root}")
        return 1

    abs_skip_dirs = normalize_skip_dirs(root, SKIP_DIRS)

    log(f"Root directory: {root}")
    if abs_skip_dirs:
        log("Skipping directories (by path):")
        for d in abs_skip_dirs:
            log(f"  - {d}")
    if SKIP_NAMES:
        log("Skipping by directory names:")
        for n in SKIP_NAMES:
            log(f"  - {n}")
    if DRY_RUN:
        log("DRY RUN enabled (no deletions will occur)")

    deleted_count = 0

    # Walk the tree top-down so we can prune directories before descending
    for current_root, dirnames, filenames in os.walk(root, topdown=True):
        current_path = Path(current_root)

        # If current path is inside any skip path, prune entirely
        if any(is_within(current_path, skip) for skip in abs_skip_dirs):
            dirnames[:] = []
            continue

        # Prune by directory base names globally
        dirnames[:] = [d for d in dirnames if d not in SKIP_NAMES]

        # Also prune by absolute skip dirs (children)
        pruned: List[str] = []
        for d in dirnames:
            candidate = current_path / d
            if any(is_within(candidate, skip) for skip in abs_skip_dirs):
                pruned.append(d)
        if pruned:
            dirnames[:] = [d for d in dirnames if d not in pruned]

        # Determine whether we are in a subtree that includes a 'checkpoints' directory
        try:
            rel_parts = current_path.relative_to(root).parts
        except Exception:
            rel_parts = current_path.parts

        in_checkpoints_subtree = "checkpoints" in rel_parts

        # Evaluate files
        for fname in filenames:
            fpath = current_path / fname
            lower = fname.lower()

            # Delete videos: *.mp4 anywhere
            if lower.endswith(".mp4"):
                log(str(fpath))
                if not DRY_RUN:
                    try:
                        fpath.unlink(missing_ok=True)
                    except Exception as e:
                        log(f"Warning: failed to delete {fpath}: {e}")
                    else:
                        deleted_count += 1
                continue

            # Delete model artifacts inside any 'checkpoints' directory: *.zip, *.pkl
            if in_checkpoints_subtree and (lower.endswith(".zip") or lower.endswith(".pkl")):
                log(str(fpath))
                if not DRY_RUN:
                    try:
                        fpath.unlink(missing_ok=True)
                    except Exception as e:
                        log(f"Warning: failed to delete {fpath}: {e}")
                    else:
                        deleted_count += 1

    log("Cleanup completed.")
    if not DRY_RUN:
        log(f"Deleted files: {deleted_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


