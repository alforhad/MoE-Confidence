from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np

from .xtail_constants import NUM_XTAIL_TASKS

_LOG_LINE = re.compile(
    r"Learned:\s*(\d+)\s+Dataset:\s*(\d+)\s+Top-1\s+accuracy:\s*([\d.]+)",
    re.IGNORECASE,
)


def parse_log_line(line: str) -> tuple[int, int, float] | None:
    match = _LOG_LINE.search(line)
    if not match:
        return None
    learned, dataset, accuracy = match.groups()
    return int(learned), int(dataset), float(accuracy)


def load_fusion_acc_table(log_path: Path, num_tasks: int) -> np.ndarray:
    fusion_acc_table = np.zeros((num_tasks, num_tasks))
    if not log_path.is_file():
        raise FileNotFoundError(f"Eval log not found: {log_path}")

    with log_path.open(encoding="utf-8") as file:
        for line in file:
            parsed = parse_log_line(line)
            if parsed is None:
                continue
            learned, dataset, accuracy = parsed
            if learned >= num_tasks or dataset >= num_tasks:
                raise ValueError(
                    f"Index out of range (num_tasks={num_tasks}): "
                    f"learned={learned}, dataset={dataset}"
                )
            fusion_acc_table[learned, dataset] = accuracy

    return fusion_acc_table


def print_metrics(fusion_acc_table: np.ndarray) -> None:
    print(fusion_acc_table)

    upper_triangle_no_diag = np.triu(fusion_acc_table, k=1)
    masked_matrix = np.ma.masked_equal(upper_triangle_no_diag, 0)
    transfer_acc = np.mean(masked_matrix, axis=0)
    transfer_avg_acc = float(np.mean(transfer_acc))
    avg_acc = np.mean(fusion_acc_table, axis=0)
    avg_avg_acc = float(np.mean(avg_acc))
    last_avg_acc = float(np.mean(fusion_acc_table[-1, :]))

    print("average transfer acc:", transfer_avg_acc)
    print("average average acc:", avg_avg_acc)
    print("average last acc:", last_avg_acc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize X-TAIL eval log metrics.")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path(os.environ.get("XTAIL_EVAL_LOG", "output_eval_clean.txt")),
        help="Eval log from evaluation (env: XTAIL_EVAL_LOG).",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=NUM_XTAIL_TASKS,
        help="Number of tasks / matrix size (default: 11).",
    )
    args = parser.parse_args()

    fusion_acc_table = load_fusion_acc_table(args.log_path, args.num_tasks)
    print_metrics(fusion_acc_table)


if __name__ == "__main__":
    main()
