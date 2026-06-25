from __future__ import annotations

from pathlib import Path
from typing import Set, Tuple

import clip.clip as clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import datasets, templates, utils
from ..paths import get_latest_model_path, load_task_gaussians, save_task_gaussians
from .sample_filter import IndexedDataset, ada_evaluate


def _scale_logits_per_sample(
    logits: torch.Tensor,
    sample_indices,
    filtered_indices: Set[int],
    tau_ub: float,
    tau_lb: float,
    device: torch.device,
) -> torch.Tensor:
    taus = torch.tensor(
        [
            tau_lb if int(idx) in filtered_indices else tau_ub
            for idx in sample_indices
        ],
        device=device,
        dtype=logits.dtype,
    ).view(-1, 1)
    return logits / taus


@torch.no_grad()
def fit_gaussian(
    model: torch.nn.Module,
    dataset_loader,
    texts: torch.Tensor,
    task_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    embed_dim = model.text_projection.shape[1]
    mean = torch.zeros(embed_dim, device=device)
    m2 = torch.zeros(embed_dim, embed_dim, device=device)
    n_samples = 0

    model.eval()
    for batch in dataset_loader:
        images = batch[0].to(device)
        _, image_features = model(images, texts, task_id, is_train=False)

        batch_size = image_features.size(0)
        n_samples += batch_size
        delta = image_features.mean(dim=0) - mean
        mean += delta * batch_size / n_samples
        centered_features = image_features - mean
        m2 += torch.mm(centered_features.t(), centered_features)

    covariance_matrix = m2 / (n_samples - 1)
    return mean, covariance_matrix


def finetune(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.tid > 0:
        init_path = get_latest_model_path(ckpt_dir)
        if init_path.is_file():
            model, train_preprocess, _val_preprocess = clip.load_model_from_path(
                str(init_path), device=device, jit=False
            )
        else:
            raise FileNotFoundError(
                f"Task {args.tid} requires previous checkpoint {init_path}"
            )
    else:
        model, train_preprocess, _val_preprocess = clip.load(
            "ViT-B/16", device=device, jit=False
        )

    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
    )

    if args.template is not None:
        template = getattr(templates, args.template)[0]
    else:
        template = dataset.template

    for _name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if utils.adaptmlp_param_matches(name, args.tid):
            param.requires_grad = True

    params = [
        v
        for k, v in model.named_parameters()
        if utils.adaptmlp_param_matches(k, args.tid)
    ]
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2)
    )

    text_strings = [template(x) for x in dataset.classnames]
    texts = clip.tokenize(text_strings).to(device)
    model = model.to(device)

    train_source = dataset.train_dataset
    phase_steps = args.iterations + 1
    filtered_indices: Set[int] = set()

    if args.tid > 0:
        incorrect_list, _correct_list = ada_evaluate(
            model,
            train_source,
            args.tid,
            train_preprocess,
            args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
            conf_threshold=args.conf_threshold,
            device=device,
        )
        filtered_indices = set(incorrect_list)
        print(
            f"Task {args.tid} mixed train: {len(filtered_indices)} filtered / "
            f"{len(train_source)} total (tau_LB={args.tau_LB}, tau_UB={args.tau_UB})"
        )

    scheduler = utils.cosine_lr(
        optimizer, args.lr, args.warmup_length, args.iterations
    )

    indexed_train = IndexedDataset(train_source)
    train_loader = DataLoader(
        indexed_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    data_iter = iter(train_loader)

    desc = (
        f"train task {args.tid} (mixed tau)"
        if args.tid > 0
        else f"train task {args.tid}"
    )
    for iteration in tqdm(range(phase_steps), desc=desc):
        scheduler(iteration)
        try:
            images, labels, sample_indices = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels, sample_indices = next(data_iter)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        logits, _ = model(images, texts, args.tid, is_train=True)
        if args.tid > 0 and filtered_indices:
            logits = _scale_logits_per_sample(
                logits,
                sample_indices,
                filtered_indices,
                args.tau_UB,
                args.tau_LB,
                device,
            )
        elif args.tau_UB != 1.0:
            logits = logits / torch.tensor(
                args.tau_UB, device=device, dtype=logits.dtype
            )

        loss = F.cross_entropy(logits, labels, label_smoothing=args.ls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % args.loss_interval == 0:
            print("Loss:", loss.item())

    model.eval()
    model_save_path = get_latest_model_path(ckpt_dir)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    mean, cov = fit_gaussian(model, dataset.train_loader, texts, args.tid)

    task_gaussians = load_task_gaussians(ckpt_dir)
    if len(task_gaussians) != args.tid:
        raise ValueError(
            f"Expected {args.tid} Gaussian(s) in latest_gau before task {args.tid}, "
            f"found {len(task_gaussians)}"
        )
    task_gaussians.append((mean, cov))
    gau_save_path = save_task_gaussians(ckpt_dir, task_gaussians)
    print(f"Gaussian stats saved to {gau_save_path} (tasks 0–{args.tid})")
