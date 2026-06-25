import os
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import clip.clip as clip
from . import utils
from .utils import cfg_truthy, get_class_ids_per_task, get_class_names


def _strip_module_prefix_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(str(k).startswith("module.") for k in state_dict):
        return dict(state_dict)
    return {k[7:] if str(k).startswith("module.") else k: v for k, v in state_dict.items()}


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return (*sample, index)


class ClassIncrementalDDP(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        self._model_weights_already_from_checkpoint = False
        if cfg_truthy(cfg, "eval_only"):
            path = self._latest_checkpoint_path(cfg)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"eval_only=True but no file at {path}. "
                    "Place latest.pth there or set checkpoint_dir."
                )
            ckpt = torch.load(path, map_location="cpu")
            init_path = cfg.get("eval_only_clip_checkpoint", None)
            if init_path and os.path.isfile(str(init_path)):
                self.model, self.transforms, self.ttransforms = clip.load_model_from_path(
                    str(init_path), device=device, jit=jit
                )
            else:
                self.model, self.transforms, self.ttransforms = clip.load(
                    cfg.model_name, device=device, jit=jit
                )
            incomp = self.model.load_state_dict(
                _strip_module_prefix_state_dict(ckpt["model_state_dict"]),
                strict=False,
            )
            if incomp.missing_keys or incomp.unexpected_keys:
                logging.warning(
                    "eval_only init load_state_dict strict=False: missing_keys=%d unexpected_keys=%d",
                    len(incomp.missing_keys),
                    len(incomp.unexpected_keys),
                )
            self._model_weights_already_from_checkpoint = True
            self.task_gaussians_input = []
        else:
            self.model, self.transforms, self.ttransforms = clip.load(
                cfg.model_name, device=device, jit=jit
            )
            self.task_gaussians_input = []
        self.epsilon = float(cfg.get("epsilon", 1e-6))
        self.zeta = float(cfg.get("zeta", 0.01))
        self.conf_threshold = float(cfg.get("conf_threshold", 95))
        self.tau_UB = float(cfg.get("tau_UB", 1.0))
        self.tau_LB = float(cfg.get("tau_LB", 0.8))
        self.eval_classnames = []
        self.task_reference_sets = []
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.datasetn = cfg.dataset
        self.initial_increment = int(cfg.initial_increment)
        self.increment = int(cfg.increment)
        self._eval_latest_loaded = False

    @property
    def distributed(self):
        return dist.is_available() and dist.is_initialized()

    @property
    def rank(self):
        return dist.get_rank() if self.distributed else 0

    @property
    def world_size(self):
        return dist.get_world_size() if self.distributed else 1

    @property
    def is_rank0(self):
        return self.rank == 0

    def wrap_ddp(self, local_rank):
        if not self.distributed:
            self.model = self.model.to(self.device)
            return

        self.model = self.model.to(self.device)
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    def _clip_model(self):
        return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

    def _loader_kwargs(self):
        num_workers = int(getattr(self, "num_workers", 8))
        return {
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": num_workers > 0,
        }

    @staticmethod
    def _set_optimizer_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _train_loader(self, dataset, batch_size, shuffle=True):
        if self.distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=False,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                **self._loader_kwargs(),
            )
            return loader, sampler

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **self._loader_kwargs(),
        )
        return loader, None

    def _rank_indices(self, dataset):
        if not self.distributed:
            return list(range(len(dataset)))
        return list(range(self.rank, len(dataset), self.world_size))

    def _sharded_loader(self, dataset, batch_size, with_indices=False):
        source = IndexedDataset(dataset) if with_indices else dataset
        shard = Subset(source, self._rank_indices(source))
        return DataLoader(
            shard,
            batch_size=batch_size,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def _all_gather_object(self, value):
        if not self.distributed:
            return [value]

        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, value)
        return gathered

    def _task_class_start(self, task_id):
        if task_id == 0:
            return 0
        return self.initial_increment + (task_id - 1) * self.increment

    def _task_class_end(self, task_id):
        task_size = self.initial_increment if task_id == 0 else self.increment
        return self._task_class_start(task_id) + task_size

    def _target_shift(self, task_id, _cfg=None):
        return self._task_class_start(task_id)

    def _latest_checkpoint_path(self, cfg):
        checkpoint_dir = str(cfg.get("checkpoint_dir", "checkpoints"))
        return os.path.join(checkpoint_dir, "latest.pth")

    def save_task_checkpoint(self, cfg):
        if not self.is_rank0:
            return

        checkpoint_path = self._latest_checkpoint_path(cfg)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._clip_model().state_dict(),
                "task_gaussians_input": [
                    (mean.cpu(), covariance.cpu())
                    for mean, covariance in self.task_gaussians_input
                ],
            },
            checkpoint_path,
        )

    def load_task_checkpoint(self, cfg):
        if self._eval_latest_loaded:
            return
        checkpoint_path = self._latest_checkpoint_path(cfg)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not getattr(self, "_model_weights_already_from_checkpoint", False):
            incomp = self._clip_model().load_state_dict(
                _strip_module_prefix_state_dict(checkpoint["model_state_dict"]),
                strict=False,
            )
            if incomp.missing_keys or incomp.unexpected_keys:
                logging.warning(
                    "Checkpoint weight load used strict=False: missing_keys=%d unexpected_keys=%d",
                    len(incomp.missing_keys),
                    len(incomp.unexpected_keys),
                )
        self.task_gaussians_input = [
            (mean.to(self.device), covariance.to(self.device))
            for mean, covariance in checkpoint["task_gaussians_input"]
        ]
        n_tasks = len(self.task_gaussians_input)
        n_cfg = len(self.class_ids_per_task)
        if n_tasks != n_cfg:
            raise ValueError(
                f"Checkpoint has {n_tasks} Gaussian(s) but config has {n_cfg} task(s). "
                "Use the same class_order / initial_increment / increment as training."
            )
        self.current_class_names = []
        for learned in range(n_tasks):
            self.current_class_names += get_class_names(
                self.classes_names,
                self.class_ids_per_task[learned],
            )
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._eval_latest_loaded = True

    def mahalanobis_distance(self, batch_x, mean, covariance_matrix, epsilon=1e-6):
        if (
            not torch.all(torch.isfinite(batch_x))
            or not torch.all(torch.isfinite(mean))
            or not torch.all(torch.isfinite(covariance_matrix))
        ):
            raise ValueError("Inputs contain NaN or Inf values")

        _, dim = batch_x.shape
        if mean.shape != (dim,) or covariance_matrix.shape != (dim, dim):
            raise ValueError("Incompatible shapes for batch_x, mean, or covariance_matrix")

        centered_x = batch_x - mean
        covariance_matrix = covariance_matrix + epsilon * torch.eye(
            dim,
            device=covariance_matrix.device,
        )
        try:
            cov_inv = torch.inverse(covariance_matrix)
        except RuntimeError as exc:
            raise ValueError("Covariance matrix is not invertible even with regularization") from exc

        left_term = torch.mm(centered_x, cov_inv)
        mahalanobis_dists = torch.sum(left_term * centered_x, dim=1)
        mahalanobis_dists = torch.clamp(mahalanobis_dists, min=0.0)
        mahalanobis_dists = torch.sqrt(mahalanobis_dists)
        return torch.clamp(mahalanobis_dists, max=1e9)

    def _forward_base_and_cache_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        model = self._clip_model()
        visual = model.visual
        cached_attention_outputs = []
        cached_mlp_outputs = []

        x = visual.conv1(image.type(visual.conv1.weight.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x,
        ], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        cached_preprocessing = x

        for block in visual.transformer.resblocks:
            x = x + block.attention(block.ln_1(x))
            cached_attention_outputs.append(x)
            mlp_out = block.mlp(block.ln_2(x))
            cached_mlp_outputs.append(mlp_out)
            x = x + mlp_out

        x = x.permute(1, 0, 2)
        x = visual.ln_post(x[:, 0, :])
        base_output = x @ visual.proj if visual.proj is not None else x
        return base_output, cached_preprocessing, cached_attention_outputs, cached_mlp_outputs

    def _forward_base_and_cache_text(self, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        model = self._clip_model()
        transformer = model.transformer
        cached_attention_outputs = []
        cached_mlp_outputs = []

        x = model.token_embedding(text).type(model.dtype)
        x = x + model.positional_embedding.type(model.dtype)
        x = x.permute(1, 0, 2)
        cached_preprocessing = x

        for block in transformer.resblocks:
            x = x + block.attention(block.ln_1(x))
            cached_attention_outputs.append(x)
            mlp_out = block.mlp(block.ln_2(x))
            cached_mlp_outputs.append(mlp_out)
            x = x + mlp_out

        x = x.permute(1, 0, 2)
        x = model.ln_final(x).type(model.dtype)
        base_output = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ model.text_projection
        return base_output, cached_preprocessing, cached_attention_outputs, cached_mlp_outputs

    def _forward_with_adapters_image(
        self,
        cached_preprocessing: torch.Tensor,
        cached_attention_outputs: List[torch.Tensor],
        cached_mlp_outputs: List[torch.Tensor],
        task_id: int,
    ) -> torch.Tensor:
        visual = self._clip_model().visual
        x = cached_preprocessing

        adapter_outputs = [
            block.adaptmlp_list[task_id](cached_attn_out, add_residual=False)
            for block, cached_attn_out in zip(visual.transformer.resblocks, cached_attention_outputs)
        ]

        for i, (block, cached_attn_out, cached_mlp_out, adapter_out) in enumerate(
            zip(visual.transformer.resblocks, cached_attention_outputs, cached_mlp_outputs, adapter_outputs)
        ):
            if i == 0:
                x = cached_attn_out + cached_mlp_out + adapter_out
            else:
                x_after_attn = x + block.attention(block.ln_1(x))
                mlp_out = block.mlp(block.ln_2(x_after_attn))
                adapter_out = block.adaptmlp_list[task_id](x_after_attn, add_residual=False)
                x = x_after_attn + mlp_out + adapter_out

        x = x.permute(1, 0, 2)
        x = visual.ln_post(x[:, 0, :])
        return x @ visual.proj if visual.proj is not None else x

    def _forward_with_adapters_text(
        self,
        cached_preprocessing: torch.Tensor,
        cached_attention_outputs: List[torch.Tensor],
        cached_mlp_outputs: List[torch.Tensor],
        task_id: int,
        text: torch.Tensor,
    ) -> torch.Tensor:
        model = self._clip_model()
        transformer = model.transformer
        x = cached_preprocessing

        adapter_outputs = [
            block.adaptmlp_list[task_id](cached_attn_out, add_residual=False)
            for block, cached_attn_out in zip(transformer.resblocks, cached_attention_outputs)
        ]

        for i, (block, cached_attn_out, cached_mlp_out, adapter_out) in enumerate(
            zip(transformer.resblocks, cached_attention_outputs, cached_mlp_outputs, adapter_outputs)
        ):
            if i == 0:
                x = cached_attn_out + cached_mlp_out + adapter_out
            else:
                x_after_attn = x + block.attention(block.ln_1(x))
                mlp_out = block.mlp(block.ln_2(x_after_attn))
                adapter_out = block.adaptmlp_list[task_id](x_after_attn, add_residual=False)
                x = x_after_attn + mlp_out + adapter_out

        x = x.permute(1, 0, 2)
        x = model.ln_final(x).type(model.dtype)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ model.text_projection

    def forward(self, image, taskid, loop=None):
        with torch.no_grad():
            logits_per_image_list = []
            mahalanobis_list = []

            _, image_cached_preprocessing, image_cached_attention, image_cached_mlp = self._forward_base_and_cache_image(image)
            _, text_cached_preprocessing, text_cached_attention, text_cached_mlp = self._forward_base_and_cache_text(self.text_tokens)
            logit_scale = self._clip_model().logit_scale.exp()

            for i in range(taskid + 1):
                image_features = self._forward_with_adapters_image(
                    image_cached_preprocessing, image_cached_attention, image_cached_mlp, i
                )
                text_features = self._forward_with_adapters_text(
                    text_cached_preprocessing, text_cached_attention, text_cached_mlp, i, self.text_tokens
                )

                encode_image = image_features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits_per_image_i = logit_scale * image_features @ text_features.t()
                mahalanobis = self.mahalanobis_distance(
                    encode_image,
                    self.task_gaussians_input[i][0],
                    self.task_gaussians_input[i][1],
                )
                logits_per_image_list.append(logits_per_image_i.softmax(dim=-1))
                mahalanobis_list.append(mahalanobis)

            mahalanobiss = torch.stack(mahalanobis_list, dim=0)
            inverse_distances = 1.0 / (mahalanobiss + self.epsilon)
            weights = (inverse_distances / self.zeta).softmax(dim=0)

            selected_logits = [
                logits_per_image_list[i][
                    :, self._task_class_start(i):self._task_class_end(i)
                ] * weights[i, :].view(weights.size(1), 1)
                for i in range(taskid + 1)
            ]

            return torch.cat(selected_logits, dim=1)

    def adaptation(self, task_id, cfg, train_dataset, train_eval_dataset, train_classes_names):
        if self.is_rank0:
            print("USING DDP CACHED ADAPTER PYTHON FILE")

        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)

        if cfg.method != "zeroshot":
            self.train(task_id, cfg, train_dataset, train_eval_dataset, train_classes_names)

    def train(self, task_id, cfg, train_dataset1, train_eval_dataset1, train_classes_names):

        train_dataset = train_dataset1[task_id:task_id + 1]
        self.model = self.model.to(self.device)
        self.num_workers = int(cfg.get("num_workers", 8))

        if task_id > 0:
            incorrect_indices, correct_indices = self.ada_evaluate(train_dataset, task_id, cfg)
            incorrect_dataset = Subset(train_dataset, incorrect_indices)
            correct_dataset = Subset(train_dataset, correct_indices)
        else:
            incorrect_dataset = None
            correct_dataset = train_dataset

        train_loader, train_sampler = self._train_loader(correct_dataset, cfg.batch_size, shuffle=True)
        if train_sampler is not None:
            train_sampler.set_epoch(task_id)

        incorrect_loader = None
        if task_id > 0 and incorrect_dataset is not None and len(incorrect_dataset) > 0:
            incorrect_loader, incorrect_sampler = self._train_loader(incorrect_dataset, cfg.batch_size, shuffle=True)
            if incorrect_sampler is not None:
                incorrect_sampler.set_epoch(task_id)

        correct_steps = max(len(train_loader), 1)
        filtered_steps = max(len(incorrect_loader), 1) if incorrect_loader is not None else 0

        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        texts = clip.tokenize([self.prompt_template.format(c) for c in classnames]).to(self.device)

        for _, param in self.model.named_parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if utils.adaptmlp_param_matches(name, task_id):
                param.requires_grad = True

        params = [
            v for k, v in self.model.named_parameters()
            if utils.adaptmlp_param_matches(k, task_id)
        ]

        warmup_length = max(1, min(30, correct_steps - 1))
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        correct_scheduler = utils.cosine_lr(optimizer, cfg.lr, warmup_length, correct_steps)

        self.model.train()
        train_iter = iter(train_loader)
        for step in tqdm(range(correct_steps), disable=not self.is_rank0):
            correct_scheduler(step)
            try:
                inputs, targets, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets, _ = next(train_iter)

            targets = (targets - self._target_shift(task_id, cfg)).long()
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits_per_image, _ = self.model(inputs, texts, task_id, is_train=True)
            logits_per_image = logits_per_image / torch.tensor(
                self.tau_UB,
                device=self.device,
                dtype=logits_per_image.dtype,
            )
            loss = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if incorrect_loader is not None:
            filtered_lr = float(cfg.get("filtered_lr", cfg.lr))
            self._set_optimizer_lr(optimizer, filtered_lr)
            if self.is_rank0:
                logging.info(
                    "Task %d filtered phase: n=%d steps=%d lr=%.6g tau_LB=%.4f",
                    task_id,
                    len(incorrect_dataset),
                    filtered_steps,
                    filtered_lr,
                    self.tau_LB,
                )
            filtered_iter = iter(incorrect_loader)
            for _step in tqdm(range(filtered_steps), disable=not self.is_rank0):
                try:
                    filtered_inputs, filtered_targets, _ = next(filtered_iter)
                except StopIteration:
                    filtered_iter = iter(incorrect_loader)
                    filtered_inputs, filtered_targets, _ = next(filtered_iter)

                filtered_targets = (filtered_targets - self._target_shift(task_id, cfg)).long()
                filtered_inputs = filtered_inputs.to(self.device, non_blocking=True)
                filtered_targets = filtered_targets.to(self.device, non_blocking=True)
                filtered_logits, _ = self.model(filtered_inputs, texts, task_id, is_train=True)
                filtered_logits = filtered_logits / torch.tensor(
                    self.tau_LB,
                    device=self.device,
                    dtype=filtered_logits.dtype,
                )
                filtered_loss = F.cross_entropy(filtered_logits, filtered_targets, label_smoothing=cfg.ls)

                optimizer.zero_grad()
                filtered_loss.backward()
                optimizer.step()

        self.model.eval()
        train_loader_gau = self._sharded_loader(train_dataset, cfg.batch_size)
        mean, cov = self.fit_gaussian(train_loader_gau, texts, task_id)
        self.task_gaussians_input.append((mean, cov))
        self.save_task_checkpoint(cfg)

    def fit_gaussian(self, dataset_loader, texts, task_id):
        model = self._clip_model()
        embed_dim = model.text_projection.shape[1]
        sum_x = torch.zeros(embed_dim, device=self.device, dtype=torch.float64)
        sum_xx = torch.zeros(embed_dim, embed_dim, device=self.device, dtype=torch.float64)
        n_samples = torch.zeros((), device=self.device, dtype=torch.float64)

        with torch.no_grad():
            for images, _, _ in dataset_loader:
                images = images.to(self.device, non_blocking=True)
                _, image_features = model(images, texts, task_id, is_train=False)
                image_features = image_features.to(torch.float64)
                sum_x += image_features.sum(dim=0)
                sum_xx += image_features.t().mm(image_features)
                n_samples += image_features.size(0)

        if self.distributed:
            dist.all_reduce(sum_x, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_xx, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)

        mean = sum_x / n_samples.clamp_min(1.0)
        centered_sum_xx = sum_xx - n_samples * torch.outer(mean, mean)
        covariance_matrix = centered_sum_xx / (n_samples - 1).clamp_min(1.0)
        return mean.float(), covariance_matrix.float()

    def ada_evaluate(self, train_task_set, task_id, cfg):
        train_loader = self._sharded_loader(train_task_set, cfg.batch_size, with_indices=True)
        self.model.eval()
        confidence_scores_per_task = {learned: [] for learned in range(task_id)}
        confidence_records_per_task = {learned: [] for learned in range(task_id)}
        logit_scale = self._clip_model().logit_scale.exp()

        text_cached_data = {}
        with torch.no_grad():
            for learned in range(task_id):
                self.eval_classnames = get_class_names(self.classes_names, self.class_ids_per_task[learned])
                eval_texts = [self.prompt_template.format(c) for c in self.eval_classnames]
                texts = clip.tokenize(eval_texts).to(self.device)
                _, text_cached_preprocessing, text_cached_attention, text_cached_mlp = self._forward_base_and_cache_text(texts)
                text_cached_data[learned] = {
                    "texts": texts,
                    "text_cached_preprocessing": text_cached_preprocessing,
                    "text_cached_attention": text_cached_attention,
                    "text_cached_mlp": text_cached_mlp,
                }

        with torch.no_grad():
            for inputs, _, _, sample_indices in train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                _, image_cached_preprocessing, image_cached_attention, image_cached_mlp = self._forward_base_and_cache_image(inputs)
                sample_indices = sample_indices.cpu().tolist()

                for learned in range(task_id):
                    text_data = text_cached_data[learned]
                    image_features = self._forward_with_adapters_image(
                        image_cached_preprocessing, image_cached_attention, image_cached_mlp, learned
                    )
                    text_features = self._forward_with_adapters_text(
                        text_data["text_cached_preprocessing"],
                        text_data["text_cached_attention"],
                        text_data["text_cached_mlp"],
                        learned,
                        text_data["texts"],
                    )
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    probs = logits_per_image.softmax(dim=-1)
                    max_confidences, _ = torch.max(probs, 1)
                    confidence_scores = max_confidences.cpu().tolist()
                    confidence_scores_per_task[learned].extend(confidence_scores)
                    confidence_records_per_task[learned].extend(
                        zip(sample_indices, confidence_scores)
                    )

        adaptive_thresholds = {}
        for learned in range(task_id):
            gathered_scores = self._all_gather_object(confidence_scores_per_task[learned])
            all_scores = [score for rank_scores in gathered_scores for score in rank_scores]
            adaptive_thresholds[learned] = np.percentile(all_scores, self.conf_threshold)

        local_incorrect_indices = set()
        for learned in range(task_id):
            confidence_threshold = adaptive_thresholds[learned]
            for sample_index, confidence_score in confidence_records_per_task[learned]:
                if confidence_score >= confidence_threshold:
                    local_incorrect_indices.add(int(sample_index))

        gathered_indices = self._all_gather_object(sorted(local_incorrect_indices))
        incorrect_indices_set = {
            index for rank_indices in gathered_indices for index in rank_indices
        }
        all_indices = set(range(len(train_task_set)))
        correct_indices = sorted(all_indices - incorrect_indices_set)
        incorrect_indices = sorted(incorrect_indices_set)

        return incorrect_indices, correct_indices


class DomainIncremental(nn.Module):
    pass


class TaskAgnostic(nn.Module):
    pass


def load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    if cfg.scenario == "class":
        return ClassIncrementalDDP(cfg, device)
    if cfg.scenario == "domain":
        return DomainIncremental()
    if cfg.scenario == "task-aganostic":
        return TaskAgnostic()
    raise ValueError(f"""
        `{cfg.scenarios}` is not a valid scenario,
        Please choose from ['class', "domain', 'task-agnostic']
    """)
