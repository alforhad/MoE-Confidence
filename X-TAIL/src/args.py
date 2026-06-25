import argparse

import torch

from .paths import get_checkpoint_root, get_data_root


def parse_arguments():
    parser = argparse.ArgumentParser(description="X-TAIL adapter train / eval")

    # training
    parser.add_argument("--train-dataset", default=None, help="Dataset class name (train).")
    parser.add_argument("--tid", type=int, default=0, help="Task id; trains adaptmlp_list.{tid}.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--ls", type=float, default=0.2, help="Label smoothing.")
    parser.add_argument("--warmup_length", type=int, default=100)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--loss-interval", type=int, default=1000)
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument(
        "--use-all-classes-file",
        action="store_true",
        help="Eval with all_classes.pt instead of train-aligned prompts (default: build 1201 from datasets).",
    )

    parser.add_argument("--tau_UB", type=float, default=1.0, help="Logit scale for correct samples.")
    parser.add_argument("--tau_LB", type=float, default=0.9, help="Logit scale for filtered samples.")
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=99.0,
        help="Percentile on past-expert confidence to mark filtered samples.",
    )
    parser.add_argument(
        "--filtered-lr",
        type=float,
        default=None,
        help="Fixed LR for filtered phase (task >= 1). Default: same as --lr (see cachemodels_ddp).",
    )

    # evaluation
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-datasets", default=None, type=lambda x: x.split(","))
    parser.add_argument(
        "--experts_num",
        type=int,
        default=0,
        help="Eval dataset index (0–10) for label offset and preprocessing.",
    )

    # data & checkpoints
    parser.add_argument(
        "--data-location",
        type=str,
        default=str(get_data_root()),
        help="Dataset root (default: X-TAIL/data). Env: XTAIL_DATA_ROOT.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(get_checkpoint_root()),
        help="latest.pth and latest_gau.pth directory. Env: XTAIL_CHECKPOINT_ROOT.",
    )

    # loader
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-size-eval", type=int, default=128)

    # misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args
