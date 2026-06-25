"""X-TAIL 11-task layout (datasets, class counts, default LRs)."""

from __future__ import annotations

XTAIL_DATASETS = [
    "Aircraft",
    "Caltech101",
    "CIFAR100",
    "DTD",
    "EuroSAT",
    "Flowers",
    "Food",
    "MNIST",
    "OxfordPet",
    "StanfordCars",
    "SUN397",
]

# Classes per task (same order as XTAIL_DATASETS).
XTAIL_LENS = [100, 101, 100, 47, 10, 102, 101, 10, 37, 196, 397]

# Cumulative offsets; head_sizes[0] is unused padding in eval slicing.
XTAIL_HEAD_SIZES = [0, *XTAIL_LENS]

# Learning rate same order as XTAIL_DATASETS.
XTAIL_LR = [5e-3, 1e-3, 5e-3, 1e-3, 1e-4, 1e-3, 1e-3, 1e-4, 1e-3, 1e-3, 1e-3]

NUM_XTAIL_TASKS = len(XTAIL_DATASETS)
NUM_XTAIL_CLASSES = sum(XTAIL_LENS)


def task_class_start(task_id: int) -> int:
    return sum(XTAIL_LENS[:task_id])


def task_class_end(task_id: int) -> int:
    return task_class_start(task_id) + XTAIL_LENS[task_id]


def slice_all_class_prompts(all_prompts: list, task_id: int) -> list:
    start, end = task_class_start(task_id), task_class_end(task_id)
    return all_prompts[start:end]
