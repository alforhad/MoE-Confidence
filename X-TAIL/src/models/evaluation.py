from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import clip.clip as clip
import torch
from tqdm import tqdm

from .. import datasets
from ..paths import get_all_classes_path
from ..xtail_constants import XTAIL_HEAD_SIZES, XTAIL_LENS
from ..xtail_prompts import load_eval_text_strings

ACCURACY_LOG_PATH = os.environ.get("XTAIL_EVAL_LOG", "output_eval_clean.txt")


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def extend_logits(logits: torch.Tensor, zero_logits: torch.Tensor) -> torch.Tensor:
    _n, c = logits.shape
    extended_logits = zero_logits.clone()
    extended_logits[:, :c] = logits
    return extended_logits


def mahalanobis_distance(
    batch_x: torch.Tensor,
    mean: torch.Tensor,
    covariance_matrix: torch.Tensor,
) -> torch.Tensor:
    device = batch_x.device
    mean = mean.to(device)
    covariance_matrix = covariance_matrix.to(device)
    centered_x = batch_x - mean
    cov_inv = torch.inverse(covariance_matrix)
    left_term = torch.mm(centered_x, cov_inv)
    return torch.sqrt(torch.sum(left_term * centered_x, dim=1))


@torch.no_grad()
def _eval_loop(
    model: torch.nn.Module,
    gau: List,
    loader,
    texts: torch.Tensor,
    args,
) -> float:
    top1 = n = 0.0
    head_sizes = XTAIL_HEAD_SIZES
    lens = XTAIL_LENS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_taskid = args.tid

    for _, (images, target) in enumerate(tqdm(loader)):
        target = target.long()
        target = target + sum(lens[: args.experts_num])

        images, target, texts = (
            images.to(device),
            target.to(device),
            texts.to(device),
        )

        logits_per_image_list: List[torch.Tensor] = []
        mahalanobis_list: List[torch.Tensor] = []

        for i in range(max_taskid + 1):
            logits_per_image_i, encode_image = model(images, texts, i, is_train=False)
            logits_per_image_list.append(logits_per_image_i.softmax(dim=-1))
            mahalanobis_list.append(
                mahalanobis_distance(encode_image, gau[i][0], gau[i][1])
            )

        mahalanobiss = torch.stack(mahalanobis_list, dim=0)

        epsilon = 1e-6
        inverse_distances = 1.0 / (mahalanobiss + epsilon)
        temperature = 0.01
        weights = (inverse_distances / temperature).softmax(dim=0)

        selected_logits = [
            logits_per_image_list[i][
                :,
                sum(head_sizes[: i + 1]) : sum(head_sizes[: i + 1]) + head_sizes[i + 1],
            ]
            * weights[i, :].view(weights.size(1), 1)
            for i in range(max_taskid + 1)
        ]

        logits = torch.cat(selected_logits, dim=1)

        zero_logits, _ = model(images, texts, None, is_train=False)
        zero_logits = zero_logits.softmax(dim=-1)

        extended_logits = extend_logits(logits, zero_logits)

        probs = extended_logits

        acc1, _acc5 = accuracy(probs, target, topk=(1, 5))
        top1 += acc1
        n += images.size(0)

    return (top1 / n) * 100.0


def eval_single_dataset(
    image_classifier, gau, dataset, texts: torch.Tensor, args
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_classifier = image_classifier.to(device)
    image_classifier.eval()

    top1 = _eval_loop(image_classifier, gau, dataset.test_loader, texts, args)

    with open(ACCURACY_LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(
            f"Learned: {args.tid} Dataset: {args.experts_num} Top-1 accuracy: {top1:.2f}\n"
        )


def evaluate(image_classifier, gau, args, val_preprocess) -> None:
    if args.eval_datasets is None:
        return

    all_texts = load_eval_text_strings(
        args.data_location,
        val_preprocess,
        use_all_classes_file=args.use_all_classes_file,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        template_name=args.template,
        all_classes_path=get_all_classes_path(Path(args.data_location)),
    )
    texts = clip.tokenize(all_texts)

    for dataset_name in args.eval_datasets:
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
        eval_single_dataset(image_classifier, gau, dataset, texts, args)
