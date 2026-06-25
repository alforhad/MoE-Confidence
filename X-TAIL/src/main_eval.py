from __future__ import annotations

from pathlib import Path

import clip.clip as clip

from . import utils
from .args import parse_arguments
from .models.evaluation import evaluate
from .paths import get_latest_model_path, load_task_gaussians


def main(args):
    utils.seed_all(args.seed)
    if not args.eval_only:
        raise SystemExit("Only supports --eval-only.")

    artifact_root = Path(args.checkpoint_dir)
    latest_path = get_latest_model_path(artifact_root)
    model, _, val_preprocess = clip.load_model_from_path(str(latest_path))

    task_gaussians = load_task_gaussians(artifact_root)
    expected = args.tid + 1
    if len(task_gaussians) < expected:
        raise ValueError(
            f"latest_gau has {len(task_gaussians)} task(s), need at least {expected} "
            f"for --tid {args.tid}"
        )
    gau = task_gaussians[:expected]

    if args.tid < args.experts_num:
        _, _, val_preprocess = clip.load("ViT-B/16", jit=False)

    evaluate(model, gau, args, val_preprocess)


if __name__ == "__main__":
    main(parse_arguments())
