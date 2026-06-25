from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from . import datasets, templates
from .xtail_constants import XTAIL_DATASETS, XTAIL_LENS


def _expected_num_classes() -> int:
    return sum(XTAIL_LENS)


def build_xtail_train_aligned_prompts(
    data_location: str | Path,
    preprocess,
    *,
    batch_size: int = 64,
    batch_size_eval: int = 128,
    template_name: str | None = None,
) -> List[str]:
    data_location = Path(data_location)
    prompts: List[str] = []

    for task_id, dataset_name in enumerate(XTAIL_DATASETS):
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            preprocess,
            location=str(data_location),
            batch_size=batch_size,
            batch_size_eval=batch_size_eval,
        )
        expected = XTAIL_LENS[task_id]
        if len(dataset.classnames) != expected:
            raise ValueError(
                f"{dataset_name}: {len(dataset.classnames)} class names, "
                f"expected {expected} (XTAIL_LENS[{task_id}])"
            )

        if template_name is not None:
            template = getattr(templates, template_name)[0]
        else:
            template = dataset.template

        prompts.extend(template(c) for c in dataset.classnames)

    expected_total = _expected_num_classes()
    if len(prompts) != expected_total:
        raise ValueError(f"Built {len(prompts)} prompts, expected {expected_total}")
    return prompts


def load_eval_text_strings(
    data_location: str | Path,
    preprocess,
    *,
    use_all_classes_file: bool,
    batch_size: int,
    batch_size_eval: int,
    template_name: str | None,
    all_classes_path: Path,
) -> Sequence[str]:
    if use_all_classes_file:
        import torch

        return torch.load(all_classes_path, map_location="cpu")
    return build_xtail_train_aligned_prompts(
        data_location,
        preprocess,
        batch_size=batch_size,
        batch_size_eval=batch_size_eval,
        template_name=template_name,
    )
