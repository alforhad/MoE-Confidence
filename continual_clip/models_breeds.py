from omegaconf import DictConfig
from tqdm import tqdm
import torch.nn.functional as F

import clip.clip as clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from . import utils
import os
import numpy as np

class ClassIncremental(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()

        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = []
        self.model, self.transforms, self.ttransforms = clip.load(cfg.model_name, device=device, jit=jit)
        self.temp_list = cfg.tempture
        self.task_gaussians_input = []
        self.text_tokens = None

    def forward(self, image, taskid):
        with torch.no_grad():

            max_taskid = taskid
            logits_per_image_list = []
            mahalanobis_list = []

            for i in range(max_taskid + 1):
                logits_per_image_i, encode_image = self.model(image, self.text_tokens, i, is_train=False)
                logits_per_image_list.append(logits_per_image_i.softmax(dim=-1))
                mahalanobis = self.mahalanobis_distance(encode_image, self.task_gaussians_input[i][0], self.task_gaussians_input[i][1])
                mahalanobis_list.append(mahalanobis)

            mahalanobiss = torch.stack(mahalanobis_list, dim=0)
            
            epsilon = 1e-6
            inverse_distances = 1.0 / (mahalanobiss + epsilon)
            temperature = 0.01
            weights = (inverse_distances / temperature).softmax(dim=0)

            selected_logits = [logits_per_image_list[i] * weights[i, :].view(weights.size(1), 1) for i in range(max_taskid + 1)]
            ensemble_outputs = torch.stack(selected_logits)
            row_max_values, _ = ensemble_outputs.max(dim=2)
            _, max_indices = torch.max(row_max_values, dim=0)
            max_confidence_output = torch.stack([ensemble_outputs[max_indices[i], i, :] for i in range(ensemble_outputs.shape[1])])
            probs = max_confidence_output

        return probs

    def adaptation(self, task_id, cfg, train_dataset, train_classes_names):
        self.classes_names = train_classes_names
        self.text_tokens = clip.tokenize(
                [self.prompt_template.format(c) for c in self.classes_names]
            ).to(self.device)
        if cfg.method != "zeroshot": 
            self.train(task_id, cfg, train_dataset)

    def train(self, task_id, cfg, train_dataset1):

        train_dataset = train_dataset1
        self.model = self.model.cuda()
        classnames = self.classes_names
        total_loader1 = 0 
        total_loader2 = 0 
        
        if task_id > 0:
            incorrect_dataset, correct_dataset = self.ada_evaluate(train_dataset, task_id, cfg)
            if len(incorrect_dataset) > 0:
                incorrect_loader = DataLoader(incorrect_dataset, batch_size=128, shuffle=True, num_workers=8)
                total_loader1 += len(incorrect_loader)
            train_loader = DataLoader(correct_dataset, batch_size=128, shuffle=True, num_workers=8)
            total_loader2 += len(train_loader)
        else:
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
            total_loader2 += len(train_loader)

        train_iter = iter(train_loader)
        EPOCH = 1
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
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = utils.cosine_lr(
            optimizer, cfg.lr, 30, total_iterations
        )
        self.model.train()

        for iteration in tqdm(range(0, total_loader2 + 1)):
            scheduler(iteration)
            try:
                inputs, targets = next(train_iter)
            except:
                train_iter = iter(train_loader)
                inputs, targets = next(train_iter)

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
                    filtered_inputs, filtered_targets = next(filtered_iter)
                except:
                    filtered_iter = iter(incorrect_loader)
                    filtered_inputs, filtered_targets  = next(filtered_iter)
                    
                filtered_targets = filtered_targets.long()
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
        train_loader_gau = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
        mean, cov = self.fit_gaussian(train_loader_gau,texts, task_id)
        self.task_gaussians_input.append((mean, cov))

    def fit_gaussian(self, dataset_loader, texts, task_id):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mean = torch.zeros(512, device=device) 
        m2 = torch.zeros(512, 512, device=device)  
        n_samples = 0

        for images, labels in dataset_loader:
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
    

    def mahalanobis_distance(self, batch_x, mean, covariance_matrix):
        centered_x = batch_x - mean  
        cov_inv = torch.inverse(covariance_matrix)
        left_term = torch.mm(centered_x, cov_inv)
        mahalanobis_dists = torch.sqrt(torch.sum(left_term * centered_x, dim=1))
        return mahalanobis_dists

    def ada_evaluate(self, train_task_set, task_id, cfg):
        train_loader = DataLoader(train_task_set, batch_size=128, shuffle=False, num_workers=8)
        self.model.eval()
        incorrect_indices_set = set()
        adaptive_thresholds = {}
        for learned in range(task_id):
            confidence_scores = []
            eval_texts = [self.prompt_template.format(c) for c in self.classes_names]
            texts = clip.tokenize(eval_texts).to(self.device)
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    targets = targets.long()
                    inputs, targets = inputs.cuda(), targets.cuda()
                    logits_per_image, _ = self.model(inputs, texts, learned, is_train=False)
                    probs = logits_per_image.softmax(dim=-1)
                    max_confidences, _ = torch.max(probs, 1)
                    confidence_scores.extend(max_confidences.cpu().numpy())

                    del inputs, targets, logits_per_image, probs
                    torch.cuda.empty_cache()

            adaptive_thresholds[learned] = np.percentile(confidence_scores, 95)  

        for learned in range(task_id):
            eval_texts = [self.prompt_template.format(c) for c in self.classes_names]
            texts = clip.tokenize(eval_texts).to(self.device)
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    targets = targets.long()
                    inputs, targets = inputs.cuda(), targets.cuda()
                    logits_per_image, _ = self.model(inputs, texts, learned, is_train=False)
                    probs = logits_per_image.softmax(dim=-1)
                    max_confidences, predicted = torch.max(probs, 1)
                    confidence_threshold = adaptive_thresholds[learned]
                    incorrect_mask = (predicted != targets) & (max_confidences >= confidence_threshold)
                    incorrect_batch_indices = [batch_idx * 128 + idx for idx, incorrect in enumerate(incorrect_mask) if incorrect]
                    incorrect_indices_set.update(incorrect_batch_indices)
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

