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

class ClassIncremental(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None

        self.model, self.transforms, self.ttransforms = clip.load(cfg.model_name, device=device, jit=jit)

        self.temp_list = [1.0,0.8]
        self.eval_classnames = []
        self.task_gaussians_input = [] 
        self.task_reference_sets = []

        self.class_ids_per_task = list(get_class_ids_per_task(cfg)) 
        self.current_class_names = []
        self.text_tokens = None
        
        self.incrmnt = cfg.increment
        self.datasetn = cfg.dataset

    def forward(self, image, taskid, loop=None):
        with torch.no_grad():   
            max_taskid = taskid
            logits_per_image_list = []
            mahalanobis_list = []

            for i in range(max_taskid + 1):
                logits_per_image_i, encode_image = self.model(image, self.text_tokens, i, is_train=False)
                mahalanobis = self.mahalanobis_distance(encode_image, self.task_gaussians_input[i][0], self.task_gaussians_input[i][1])
    
                logits_per_image_list.append(logits_per_image_i.softmax(dim=-1))
                mahalanobis_list.append(mahalanobis)

            mahalanobiss = torch.stack(mahalanobis_list, dim=0)
            epsilon = 1e-6
            inverse_distances = 1.0 / (mahalanobiss + epsilon)
            temperature = 0.01
            weights = (inverse_distances / temperature).softmax(dim=0)

            if self.datasetn == "tinyimagenet":
                #initial increment 100
                selected_logits = [
                    logits_per_image_list[i][:, :100] * weights[i, :].view(weights.size(1), 1) if i == 0
                    else logits_per_image_list[i][:, 100 + (i - 1) * self.incrmnt : 100 + i * self.incrmnt] * weights[i, :].view(weights.size(1), 1)
                    for i in range(max_taskid + 1)
                ]

            else:
                selected_logits = [logits_per_image_list[i][:, i*self.incrmnt:(i+1)*self.incrmnt] * weights[i, :].view(weights.size(1), 1) for i in range(max_taskid + 1)]
                
            logits_per_image = torch.cat(selected_logits, dim=1)
            
        return logits_per_image
    
    def adaptation(self, task_id, cfg, train_dataset, train_classes_names):

        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.text_tokens = clip.tokenize(
                [self.prompt_template.format(c) for c in self.current_class_names]
            ).to(self.device)
        
        if cfg.method != "zeroshot":
            self.train(task_id, cfg, train_dataset, train_classes_names)

    def train(self, task_id, cfg, train_dataset1, train_classes_names):

        train_dataset = train_dataset1[task_id:task_id + 1]
        self.model = self.model.cuda()

        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        print("Class names for task {}: {}".format(task_id, classnames))

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

        EPOCH = 1
        num_batches = total_loader1 + total_loader2
        print("Total batches:", num_batches)
        total_iterations = EPOCH * num_batches
        print("Total iterations:", total_iterations)

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
        warmup_length = max(1, warmup_length) ##30 chilo

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


            temperature = torch.tensor(self.temp_list[0]).expand(logits_per_image.size(0), logits_per_image.size(1)).cuda()
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

                temperature = torch.tensor(self.temp_list[1]).expand(filtered_logits.size(0), filtered_logits.size(1)).cuda()
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

        for learned in range(task_id):

            confidence_scores = []

            self.eval_classnames = get_class_names(self.classes_names, self.class_ids_per_task[learned]) 
            eval_texts = [self.prompt_template.format(c) for c in self.eval_classnames]
            texts = clip.tokenize(eval_texts).to(self.device)

            with torch.no_grad():
                for batch_idx, (inputs, targets, task_ids) in enumerate(train_loader):

                    if cfg.dataset == "tinyimagenet" and task_id != 0:
                        shift = 100 + (task_id - 1) * cfg.increment
                        targets -= shift
                    elif cfg.dataset == "imagenet100" and task_id != 0:
                        shift = cfg.initial_increment + (task_id - 1) * cfg.increment
                        targets -= shift
                    else:
                        shift = task_id * cfg.increment
                        targets = (targets - shift).long()

                    inputs, targets = inputs.cuda(), targets.cuda()
                    logits_per_image, _ = self.model(inputs, texts, learned, is_train=False)
                    
                    probs = logits_per_image.softmax(dim=-1)
                    max_confidences, _ = torch.max(probs, 1)

                    confidence_scores.extend(max_confidences.cpu().numpy())

                    del inputs, targets, logits_per_image, probs
                    torch.cuda.empty_cache()

            adaptive_thresholds[learned] = np.percentile(confidence_scores, 95)

        for learned in range(task_id):

            self.eval_classnames = get_class_names(self.classes_names, self.class_ids_per_task[learned])
            eval_texts = [self.prompt_template.format(c) for c in self.eval_classnames]
            texts = clip.tokenize(eval_texts).to(self.device)

            with torch.no_grad():
                for batch_idx, (inputs, targets, task_ids) in enumerate(train_loader):

                    if cfg.dataset == "tinyimagenet" and task_id != 0:
                        shift = 100 + (task_id - 1) * cfg.increment
                        targets -= shift
                    elif cfg.dataset == "imagenet100" and task_id != 0:
                        shift = cfg.initial_increment + (task_id - 1) * cfg.increment
                        targets -= shift
                    else:
                        shift = task_id * cfg.increment
                        targets = (targets - shift).long()

                    inputs, targets = inputs.cuda(), targets.cuda()

                    logits_per_image, _ = self.model(inputs, texts, learned, is_train=False)
                    probs = logits_per_image.softmax(dim=-1)
                    max_confidences, _ = torch.max(probs, 1)

                    confidence_threshold = adaptive_thresholds[learned]

                    high_confidence_mask = max_confidences >= confidence_threshold

                    high_confidence_indices = [
                        batch_idx * cfg.batch_size + idx for idx, high_conf in enumerate(high_confidence_mask) if high_conf
                    ]

                    incorrect_indices_set.update(high_confidence_indices) 

                    del inputs, targets, logits_per_image, probs
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

