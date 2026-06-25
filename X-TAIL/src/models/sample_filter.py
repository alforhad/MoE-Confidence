from __future__ import annotations

from pathlib import Path
from typing import List, Set, Tuple

import clip.clip as clip
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .. import datasets
from ..xtail_constants import XTAIL_DATASETS


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        if isinstance(sample, dict):
            return sample["images"], sample["labels"], index
        images, labels = sample
        return images, labels, index


def _tokenize_past_task_texts(
    learned: int,
    train_preprocess,
    data_location: str | Path,
    batch_size: int,
    batch_size_eval: int,
    device: torch.device,
) -> torch.Tensor:
    dataset_name = XTAIL_DATASETS[learned]
    dataset_class = getattr(datasets, dataset_name)
    past_dataset = dataset_class(
        train_preprocess,
        location=str(data_location),
        batch_size=batch_size,
        batch_size_eval=batch_size_eval,
    )
    template = past_dataset.template
    text_strings = [template(c) for c in past_dataset.classnames]
    return clip.tokenize(text_strings).to(device)


@torch.no_grad()
def ada_evaluate(
    model: torch.nn.Module,
    train_dataset: Dataset,
    task_id: int,
    train_preprocess,
    data_location: str | Path,
    *,
    batch_size: int,
    batch_size_eval: int,
    conf_threshold: float,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    if task_id <= 0:
        raise ValueError("Only defined for task_id >= 1")
    if task_id > len(XTAIL_DATASETS):
        raise ValueError(f"task_id {task_id} exceeds X-TAIL task count {len(XTAIL_DATASETS)}")

    model.eval()

    previous_texts = [
        _tokenize_past_task_texts(
            learned,
            train_preprocess,
            data_location,
            batch_size,
            batch_size_eval,
            device,
        )
        for learned in range(task_id)
    ]

    indexed = IndexedDataset(train_dataset)
    loader = DataLoader(indexed, batch_size=batch_size, shuffle=False, num_workers=0)

    confidence_scores_per_task = {learned: [] for learned in range(task_id)}
    confidence_records_per_task = {learned: [] for learned in range(task_id)}

    for images, _labels, sample_indices in tqdm(loader, desc="ada_evaluate"):
        images = images.to(device, non_blocking=True)
        sample_indices = sample_indices.cpu().tolist()

        for learned in range(task_id):
            texts = previous_texts[learned]
            logits, _ = model(images, texts, learned, is_train=False)
            probs = logits.softmax(dim=-1)
            max_confidences, _ = torch.max(probs, dim=-1)
            scores = max_confidences.cpu().tolist()
            confidence_scores_per_task[learned].extend(scores)
            confidence_records_per_task[learned].extend(
                zip(sample_indices, scores)
            )

    adaptive_thresholds = {}
    for learned in range(task_id):
        adaptive_thresholds[learned] = float(
            np.percentile(confidence_scores_per_task[learned], conf_threshold)
        )

    incorrect_indices: Set[int] = set()
    for learned in range(task_id):
        threshold = adaptive_thresholds[learned]
        for sample_index, confidence_score in confidence_records_per_task[learned]:
            if confidence_score >= threshold:
                incorrect_indices.add(int(sample_index))

    all_indices = set(range(len(train_dataset)))
    incorrect_list = sorted(incorrect_indices)
    correct_list = sorted(all_indices - incorrect_indices)
    return incorrect_list, correct_list
