from asyncio.proactor_events import _ProactorBasePipeTransport
from omegaconf import DictConfig
from tqdm import tqdm
import torch.nn.functional as F

import clip.clip as clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from .utils import get_class_ids_per_task, get_class_names
from . import utils
import os
import random
import numpy as np
from typing import List, Tuple

class ClassIncremental(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None

        self.model, self.transforms, self.ttransforms = clip.load(cfg.model_name, device=device, jit=jit)
        self.epsilon = float(cfg.get("epsilon", 1e-6))
        self.zeta = float(cfg.get("zeta", 0.01))
        self.conf_threshold = float(cfg.get("conf_threshold", 95))
        self.tau_UB = float(cfg.get("tau_UB", 1.0))
        self.tau_LB = float(cfg.get("tau_LB", 0.8))
        self.eval_classnames = []
        self.task_gaussians_input = [] 
        self.task_reference_sets = []

        self.class_ids_per_task = list(get_class_ids_per_task(cfg)) 
        self.current_class_names = []
        self.text_tokens = None
        
        self.incrmnt = cfg.increment
        self.datasetn = cfg.dataset

    def _forward_base_and_cache_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        visual = self.model.visual
        cached_attention_outputs = []
        cached_mlp_outputs = []
        
        x = visual.conv1(image.type(visual.conv1.weight.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), 
            x
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
        if visual.proj is not None:
            base_output = x @ visual.proj
        else:
            base_output = x
            
        return base_output, cached_preprocessing, cached_attention_outputs, cached_mlp_outputs
    
    def _forward_base_and_cache_text(self, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        transformer = self.model.transformer
        cached_attention_outputs = []
        cached_mlp_outputs = []
        
        x = self.model.token_embedding(text).type(self.model.dtype)
        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  
        cached_preprocessing = x  
        
        for block in transformer.resblocks:
            x = x + block.attention(block.ln_1(x))
            
            cached_attention_outputs.append(x)
            
            mlp_out = block.mlp(block.ln_2(x))
            cached_mlp_outputs.append(mlp_out)
            x = x + mlp_out
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)

        base_output = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection
        
        return base_output, cached_preprocessing, cached_attention_outputs, cached_mlp_outputs
    
    def _forward_with_adapters_image(self, cached_preprocessing: torch.Tensor, 
                                     cached_attention_outputs: List[torch.Tensor],
                                     cached_mlp_outputs: List[torch.Tensor],
                                     task_id: int) -> torch.Tensor:

        visual = self.model.visual
        
        x = cached_preprocessing

        adapter_outputs = []
        for block, cached_attn_out in zip(visual.transformer.resblocks, cached_attention_outputs):
            adapter_output = block.adaptmlp_list[task_id](cached_attn_out, add_residual=False)
            adapter_outputs.append(adapter_output)

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
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = visual.ln_post(x[:, 0, :])
        if visual.proj is not None:
            output = x @ visual.proj
        else:
            output = x
            
        return output
    
    def _forward_with_adapters_text(self, cached_preprocessing: torch.Tensor,
                                    cached_attention_outputs: List[torch.Tensor],
                                    cached_mlp_outputs: List[torch.Tensor],
                                    task_id: int, text: torch.Tensor) -> torch.Tensor:
        
        transformer = self.model.transformer
        
        x = cached_preprocessing
        
        adapter_outputs = []
        for block, cached_attn_out in zip(transformer.resblocks, cached_attention_outputs):
            adapter_output = block.adaptmlp_list[task_id](cached_attn_out, add_residual=False)
            adapter_outputs.append(adapter_output)
        
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
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)
        output = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection
        
        return output

    def forward(self, image, taskid, loop=None):
        with torch.no_grad():   
            max_taskid = taskid
            logits_per_image_list = []
            mahalanobis_list = []

            _, image_cached_preprocessing, image_cached_attention, image_cached_mlp = self._forward_base_and_cache_image(image)
            _, text_cached_preprocessing, text_cached_attention, text_cached_mlp = self._forward_base_and_cache_text(self.text_tokens)
            
            logit_scale = self.model.logit_scale.exp()
            
            for i in range(max_taskid + 1):
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
                

                mahalanobis = self.mahalanobis_distance(encode_image, self.task_gaussians_input[i][0], self.task_gaussians_input[i][1])
                logits_per_image_list.append(logits_per_image_i.softmax(dim=-1))
                mahalanobis_list.append(mahalanobis)

            mahalanobiss = torch.stack(mahalanobis_list, dim=0)
            
            inverse_distances = 1.0 / (mahalanobiss + self.epsilon)
            weights = (inverse_distances / self.zeta).softmax(dim=0)

            if self.datasetn == "tinyimagenet":
                selected_logits = [
                    logits_per_image_list[i][:, :100] * weights[i, :].view(weights.size(1), 1) if i == 0
                    else logits_per_image_list[i][:, 100 + (i - 1) * self.incrmnt : 100 + i * self.incrmnt] * weights[i, :].view(weights.size(1), 1)
                    for i in range(max_taskid + 1)
                ]

            else: 
                selected_logits = [logits_per_image_list[i][:, i*self.incrmnt:(i+1)*self.incrmnt] * weights[i, :].view(weights.size(1), 1) for i in range(max_taskid + 1)]

            logits_per_image = torch.cat(selected_logits, dim=1)
            
            probs = logits_per_image
                
        return probs
    
    def adaptation(self, task_id, cfg, train_dataset, train_classes_names):

        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id]) #+

        self.text_tokens = clip.tokenize(
                [self.prompt_template.format(c) for c in self.current_class_names]
            ).to(self.device)
        
        if cfg.method != "zeroshot":
            self.train(task_id, cfg, train_dataset, train_classes_names)

    def train(self, task_id, cfg, train_dataset1, train_classes_names):

        train_dataset = train_dataset1[task_id:task_id + 1]

        self.model = self.model.cuda()

        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])

        total_loader1 = 0
        total_loader2 = 0 
        
        if task_id > 0:
            incorrect_dataset, correct_dataset = self.ada_evaluate(train_dataset, task_id, cfg)

            if len(incorrect_dataset) > 0:
                incorrect_loader = DataLoader(incorrect_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8)
                total_loader1 += len(incorrect_loader)

            train_loader = DataLoader(correct_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8)
            total_loader2 += len(train_loader)

        else:
            train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8)
            total_loader2 += len(train_loader)

        train_iter = iter(train_loader)

        EPOCH = 1 #2
        num_batches = total_loader1 + total_loader2
        total_iterations = EPOCH * num_batches

        texts = [self.prompt_template.format(c) for c in classnames]
        texts = clip.tokenize(texts).to(self.device)

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if utils.adaptmlp_param_matches(name, task_id):
                param.requires_grad = True
        
        params = [
            v for k, v in self.model.named_parameters()
            if utils.adaptmlp_param_matches(k, task_id)
        ]

        warmup_length = min(30, total_iterations - 1)
        warmup_length = max(1, warmup_length) 

        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = utils.cosine_lr(
            optimizer, cfg.lr, warmup_length, total_iterations
        )

        self.model.train()

        for iteration in tqdm(range(0, total_loader2 + 1)):
            scheduler(iteration)
            try:
                inputs, targets, task_ids = next(train_iter)
            except:
                train_iter = iter(train_loader)
                inputs, targets, task_ids = next(train_iter)

            if cfg.dataset == "tinyimagenet" and task_id != 0:
                shift = 100 + (task_id - 1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "imagenet100" and task_id != 0:
                shift = cfg.initial_increment + (task_id - 1) * cfg.increment
                targets -= shift
            else:
                shift = task_id * cfg.increment
                targets -= shift

            targets = targets.long()
            inputs, targets = inputs.cuda(), targets.cuda()

            logits_per_image, _ = self.model(inputs, texts, task_id, is_train=True)

            temperature = torch.tensor(self.tau_UB).expand(logits_per_image.size(0), logits_per_image.size(1)).cuda()
            logits_per_image = logits_per_image / temperature

            loss = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if task_id > 0 and len(incorrect_dataset) > 0:
            for iteration in tqdm(range(total_loader2 + 1, total_iterations + 2)):
                
                scheduler(iteration)
                try:
                    filtered_inputs, filtered_targets, task_ids = next(filtered_iter)
                except:
                    filtered_iter = iter(incorrect_loader)
                    filtered_inputs, filtered_targets, task_ids = next(filtered_iter)
                    
                if cfg.dataset == "tinyimagenet" and task_id != 0:
                    shift = 100 + (task_id - 1) * cfg.increment
                    filtered_targets -= shift
                elif cfg.dataset == "imagenet100" and task_id != 0:
                    shift = cfg.initial_increment + (task_id - 1) * cfg.increment
                    filtered_targets -= shift
                else: 
                    shift = task_id * cfg.increment
                    filtered_targets = (filtered_targets - shift).long()
                
                filtered_inputs, filtered_targets = filtered_inputs.cuda(), filtered_targets.cuda()
                filtered_logits, _ = self.model(filtered_inputs, texts, task_id, is_train=True)

                temperature = torch.tensor(self.tau_LB).expand(filtered_logits.size(0), filtered_logits.size(1)).cuda()
                filtered_logits = filtered_logits / temperature

                filtered_loss = F.cross_entropy(filtered_logits, filtered_targets, label_smoothing=cfg.ls)
                loss = filtered_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model.eval()

        train_loader_gau = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8)
        mean, cov = self.fit_gaussian(train_loader_gau, texts, task_id)
        self.task_gaussians_input.append((mean, cov))

    def fit_gaussian(self, dataset_loader, texts, task_id):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mean = torch.zeros(512, device=device) 
        m2 = torch.zeros(512, 512, device=device)  
        n_samples = 0

        for images, _, _ in dataset_loader:
            with torch.no_grad():
                images = images.to(device)  
                _, image_features = self.model(images, texts, task_id, is_train=False)

            batch_size = image_features.size(0)

            n_samples_old = n_samples
            n_samples += batch_size

            delta = image_features.mean(dim=0) - mean
            mean += delta * batch_size / n_samples

            centered_features = image_features - mean
            m2 += torch.mm(centered_features.t(), centered_features)

        covariance_matrix = m2 / (n_samples - 1)

        return mean, covariance_matrix


    def mahalanobis_distance(self, batch_x, mean, covariance_matrix, epsilon=1e-6):

        if not torch.all(torch.isfinite(batch_x)) or not torch.all(torch.isfinite(mean)) or not torch.all(torch.isfinite(covariance_matrix)):
            raise ValueError("Inputs contain NaN or Inf values")
        
        N, D = batch_x.shape
        if mean.shape != (D,) or covariance_matrix.shape != (D, D):
            raise ValueError("Incompatible shapes for batch_x, mean, or covariance_matrix")
        
        centered_x = batch_x - mean
        
        covariance_matrix = covariance_matrix + epsilon * torch.eye(D, device=covariance_matrix.device)
        
        try:
            cov_inv = torch.inverse(covariance_matrix)
        except RuntimeError:
            raise ValueError("Covariance matrix is not invertible even with regularization")
        
        left_term = torch.mm(centered_x, cov_inv)
        mahalanobis_dists = torch.sum(left_term * centered_x, dim=1)
        
        mahalanobis_dists = torch.clamp(mahalanobis_dists, min=0.0)
        mahalanobis_dists = torch.sqrt(mahalanobis_dists)
        
        mahalanobis_dists = torch.clamp(mahalanobis_dists, max=1e9)
        
        return mahalanobis_dists


    def ada_evaluate(self, train_task_set, task_id, cfg):

        train_loader = DataLoader(train_task_set, batch_size=cfg.batch_size, shuffle=False, num_workers=8)
        self.model.eval()
        incorrect_indices_set = set()
        adaptive_thresholds = {}
        confidence_scores_per_task = {learned: [] for learned in range(task_id)}

        logit_scale = self.model.logit_scale.exp()
        
        if cfg.dataset == "tinyimagenet" and task_id != 0:
            shift = 100 + (task_id - 1) * cfg.increment
        elif cfg.dataset == "imagenet100" and task_id != 0:
            shift = cfg.initial_increment + (task_id - 1) * cfg.increment
        else:
            shift = task_id * cfg.increment
        
        text_cached_data = {}
        with torch.no_grad():
            for learned in range(task_id):
                self.eval_classnames = get_class_names(self.classes_names, self.class_ids_per_task[learned]) 
                eval_texts = [self.prompt_template.format(c) for c in self.eval_classnames]
                texts = clip.tokenize(eval_texts).to(self.device)
                
                _, text_cached_preprocessing, text_cached_attention, text_cached_mlp = self._forward_base_and_cache_text(texts)
                text_cached_data[learned] = {
                    'texts': texts,
                    'text_cached_preprocessing': text_cached_preprocessing,
                    'text_cached_attention': text_cached_attention,
                    'text_cached_mlp': text_cached_mlp
                }

        with torch.no_grad():
            for batch_idx, (inputs, targets, task_ids) in enumerate(train_loader):

                targets = (targets - shift).long()
                inputs, targets = inputs.cuda(), targets.cuda()
                
                _, image_cached_preprocessing, image_cached_attention, image_cached_mlp = self._forward_base_and_cache_image(inputs)
                
                for learned in range(task_id):
                    text_data = text_cached_data[learned]
                    
                    image_features = self._forward_with_adapters_image(
                        image_cached_preprocessing, image_cached_attention, image_cached_mlp, learned
                    )
                    text_features = self._forward_with_adapters_text(
                        text_data['text_cached_preprocessing'], 
                        text_data['text_cached_attention'], 
                        text_data['text_cached_mlp'], 
                        learned, 
                        text_data['texts']
                    )
                    
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    
                    probs = logits_per_image.softmax(dim=-1)
                    max_confidences, _ = torch.max(probs, 1)

                    confidence_scores_per_task[learned].extend(max_confidences.cpu().numpy())

                    del logits_per_image, probs, image_features, text_features
                
                del inputs, targets, image_cached_preprocessing, image_cached_attention, image_cached_mlp
                if batch_idx % 10 == 0:  
                    torch.cuda.empty_cache()

        for learned in range(task_id):
            adaptive_thresholds[learned] = np.percentile(confidence_scores_per_task[learned], self.conf_threshold)

        with torch.no_grad():
            for batch_idx, (inputs, targets, task_ids) in enumerate(train_loader):

                targets = (targets - shift).long()
                inputs, targets = inputs.cuda(), targets.cuda()

                _, image_cached_preprocessing, image_cached_attention, image_cached_mlp = self._forward_base_and_cache_image(inputs)
                
                for learned in range(task_id):
                    text_data = text_cached_data[learned]
                    
                    image_features = self._forward_with_adapters_image(
                        image_cached_preprocessing, image_cached_attention, image_cached_mlp, learned
                    )
                    text_features = self._forward_with_adapters_text(
                        text_data['text_cached_preprocessing'], 
                        text_data['text_cached_attention'], 
                        text_data['text_cached_mlp'], 
                        learned, 
                        text_data['texts']
                    )
                    
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    
                    probs = logits_per_image.softmax(dim=-1)
                    max_confidences, _ = torch.max(probs, 1)

                    confidence_threshold = adaptive_thresholds[learned]

                    high_confidence_mask = max_confidences >= confidence_threshold

                    high_confidence_indices = [
                        batch_idx * cfg.batch_size + idx for idx, high_conf in enumerate(high_confidence_mask) if high_conf
                    ]

                    incorrect_indices_set.update(high_confidence_indices) 

                    del logits_per_image, probs, image_features, text_features
                
                del inputs, targets, image_cached_preprocessing, image_cached_attention, image_cached_mlp
                if batch_idx % 10 == 0: 
                    torch.cuda.empty_cache()

        all_indices = set(range(len(train_task_set)))
        correct_indices = list(all_indices - incorrect_indices_set)

        incorrect_indices = list(incorrect_indices_set)
        incorrect_dataset = Subset(train_task_set, incorrect_indices)
        correct_dataset = Subset(train_task_set, correct_indices)

        return incorrect_dataset, correct_dataset
    

class DomainIncremental(nn.Module):
    pass


class TaskAgnostic(nn.Module):
    pass


def load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.

    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.

    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        return ClassIncremental(cfg, device)
    elif cfg.scenario == "domain":
        return DomainIncremental(cfg, device)
    elif cfg.scenario == "task-aganostic":
        return TaskAgnostic(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)

