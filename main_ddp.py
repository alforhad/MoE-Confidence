import json
import logging
import os
import statistics

import hydra
import torch
import torch.distributed as dist
from continuum.metrics import Logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from continual_clip import utils
from continual_clip.cachemodels_ddp import load_model
from continual_clip.datasets import build_cl_scenarios


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if is_distributed() else 0


def world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_rank0():
    return rank() == 0


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    requested_world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        n_cuda = torch.cuda.device_count()
        if local_rank >= n_cuda:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {n_cuda} CUDA device(s) are visible. "
                "Use --nproc_per_node equal to the number of visible GPUs "
                "(e.g. with CUDA_VISIBLE_DEVICES=0,1 use --nproc_per_node=2)."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if requested_world_size > 1:
        dist.init_process_group(backend=backend)

    return device, local_rank


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def barrier():
    if is_distributed():
        dist.barrier()


def rank_indices(dataset):
    if not is_distributed():
        return list(range(len(dataset)))
    return list(range(rank(), len(dataset), world_size()))


def all_gather_object(value):
    if not is_distributed():
        return [value]

    gathered = [None for _ in range(world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def flatten_gathered(gathered):
    return [item for rank_items in gathered for item in rank_items]


def build_scenarios(cfg, model):
    if is_distributed() and not is_rank0():
        barrier()

    eval_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=model.ttransforms
    )
    train_dataset, train_classes_names = build_cl_scenarios(
        cfg, is_train=True, transforms=model.transforms
    )
    train_eval_dataset, _ = build_cl_scenarios(
        cfg, is_train=True, transforms=model.ttransforms
    )

    if is_distributed() and is_rank0():
        barrier()

    barrier()
    return eval_dataset, classes_names, train_dataset, train_eval_dataset, train_classes_names


@hydra.main(config_path=None, config_name=None, version_base="1.1")
def continual_clip(cfg: DictConfig) -> None:
    device, local_rank = setup_distributed()

    try:
        cfg.workdir = utils.get_workdir(path=os.getcwd())
        cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)
        cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))

        if is_rank0():
            utils.save_config(cfg)
            logging.info(
                f"DDP setup: rank={rank()}, world_size={world_size()}, device={device}"
            )
            if torch.cuda.is_available() and not is_distributed():
                n = torch.cuda.device_count()
                if n > 1:
                    logging.warning(
                        "Multiple CUDA devices visible but WORLD_SIZE=1 (plain python). "
                        "Only one GPU will be used. For DDP use torchrun, for example:\n"
                        "  torchrun --standalone --nproc_per_node=%d main_ddp.py ...",
                        n,
                    )

        model = load_model(cfg, device)
        model.wrap_ddp(local_rank)

        (
            eval_dataset,
            classes_names,
            train_dataset,
            train_eval_dataset,
            train_classes_names,
        ) = build_scenarios(cfg, model)
        model.classes_names = classes_names
        train_only = utils.cfg_truthy(cfg, "train_only")
        eval_only = utils.cfg_truthy(cfg, "eval_only")
        if train_only and eval_only:
            raise ValueError("Only one of train_only or eval_only can be true.")

        if is_rank0():
            with open(cfg.log_path, "w+") as f:
                pass
            metric_logger = Logger(list_subsets=["test"])
            acc_list = []
        else:
            metric_logger = None
            acc_list = None

        for task_id, _ in enumerate(eval_dataset):
            if is_rank0():
                if train_only:
                    logging.info(f"Training for task {task_id} has started.")
                elif eval_only:
                    logging.info(f"Checkpoint evaluation for task {task_id} has started.")
                else:
                    logging.info(f"Training and evaluation for task {task_id} has started.")

            if eval_only:
                model.load_task_checkpoint(cfg)
            else:
                model.adaptation(
                    task_id,
                    cfg,
                    train_dataset,
                    train_eval_dataset,
                    train_classes_names,
                )
            model.model.eval()
            barrier()

            if train_only:
                continue

            eval_seen_dataset = eval_dataset[:task_id + 1]
            if is_distributed():
                eval_seen_dataset = Subset(
                    eval_seen_dataset,
                    rank_indices(eval_seen_dataset),
                )
            eval_loader = DataLoader(
                eval_seen_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=int(cfg.get("num_workers", 8)),
                pin_memory=torch.cuda.is_available(),
            )

            local_preds = []
            local_targets = []
            local_task_ids = []

            iterator = tqdm(eval_loader, disable=not is_rank0())
            for inputs, targets, task_ids in iterator:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs, task_id)

                local_preds.extend(outputs.cpu().argmax(dim=1).tolist())
                local_targets.extend(targets.cpu().tolist())
                local_task_ids.extend(task_ids.cpu().tolist())

            gathered_preds = all_gather_object(local_preds)
            gathered_targets = all_gather_object(local_targets)
            gathered_task_ids = all_gather_object(local_task_ids)

            if is_rank0():
                metric_logger.add([
                    torch.tensor(flatten_gathered(gathered_preds)),
                    torch.tensor(flatten_gathered(gathered_targets)),
                    torch.tensor(flatten_gathered(gathered_task_ids)),
                ], subset="test")

            if is_rank0():
                acc_list.append(100 * metric_logger.accuracy)
                with open(cfg.log_path, "a+") as f:
                    f.write(json.dumps({
                        "task": task_id,
                        "acc": round(100 * metric_logger.accuracy, 2),
                        "avg_acc": round(100 * metric_logger.average_incremental_accuracy, 2),
                        "forgetting": round(100 * metric_logger.forgetting, 6),
                        "acc_per_task": [
                            round(100 * acc_t, 2)
                            for acc_t in metric_logger.accuracy_per_task
                        ],
                        "bwt": round(100 * metric_logger.backward_transfer, 2),
                        "fwt": round(100 * metric_logger.forward_transfer, 2),
                    }) + "\n")
                    metric_logger.end_task()

            barrier()

        if is_rank0() and not train_only:
            with open(cfg.log_path, "a+") as f:
                f.write(json.dumps({
                    "last": round(acc_list[-1], 2),
                    "avg": round(statistics.mean(acc_list), 2),
                }) + "\n")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    continual_clip()
