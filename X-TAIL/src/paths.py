"""X-TAIL root and default paths (overridable via env or CLI)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import torch

# .../X-TAIL/src/paths.py -> .../X-TAIL
XTAIL_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_DATA = XTAIL_ROOT / "data"
_DEFAULT_CHECKPOINTS = XTAIL_ROOT / "checkpoints"

LATEST_MODEL_NAME = "latest.pth"
LATEST_GAU_NAME = "latest_gau.pth"

GaussianStats = Tuple[torch.Tensor, torch.Tensor]
TaskGaussians = List[GaussianStats]


def get_data_root() -> Path:
    return Path(os.environ.get("XTAIL_DATA_ROOT", _DEFAULT_DATA))


def get_checkpoint_root() -> Path:
    env = os.environ.get("XTAIL_CHECKPOINT_ROOT")
    return Path(env) if env else _DEFAULT_CHECKPOINTS


def get_latest_model_path(checkpoint_dir: Path | None = None) -> Path:
    root = checkpoint_dir if checkpoint_dir is not None else get_checkpoint_root()
    return root / LATEST_MODEL_NAME


def get_latest_gau_path(checkpoint_dir: Path | None = None) -> Path:
    root = checkpoint_dir if checkpoint_dir is not None else get_checkpoint_root()
    return root / LATEST_GAU_NAME


def load_task_gaussians(checkpoint_dir: Path) -> TaskGaussians:
    path = get_latest_gau_path(checkpoint_dir)
    if not path.is_file():
        return []
    return torch.load(path, map_location="cpu")


def save_task_gaussians(checkpoint_dir: Path, task_gaussians: TaskGaussians) -> Path:
    path = get_latest_gau_path(checkpoint_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        [(mean.cpu(), cov.cpu()) for mean, cov in task_gaussians],
        path,
    )
    return path


def get_all_classes_path(data_root: Path | None = None) -> Path:
    env = os.environ.get("XTAIL_ALL_CLASSES")
    if env:
        return Path(env)
    root = data_root if data_root is not None else get_data_root()
    return root / "all_classes.pt"
