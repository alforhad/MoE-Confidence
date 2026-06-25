# Adopted from https://github.com/wuyujack/ISL

import os
import json
import hydra
import logging
from omegaconf import DictConfig

from tqdm import tqdm

import torch
import statistics
from continuum.metrics import Logger

from continual_clip import utils
from continual_clip.models_breeds import load_model
from continual_clip.breeds_inc import BREEDSFactory

from torchvision import transforms
import pickle

# lighting transform
# https://git.io/fhBOc
IMAGENET_PCA = {
    'eigval':torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec':torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}
class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


def br_dt_get():
    
    ds_name = 'entity30'
    info_dir="/ILSVRC/imagenet_class_hierarchy/modified/"
    data_dir="/ILSVRC/Data/CLS-LOC/"
    task_stat_path = "/breeds_protocol/entity30_4_tasks.pkl"
    batch_size = 128
    classes = ['serpentes', 'passerine', 'saurian', 'arachnid', 'aquatic bird', 'crustacean', 'carnivore', 'insect', 'ungulate', 'primate', 'bony fish', 'barrier', 'building', 'electronic equipment', 'footwear', 'garment', 'headdress', 'home appliance', 'kitchen utensil', 'measuring instrument', 'motor vehicle', 'musical instrument', 'neckwear', 'sports equipment', 'tableware', 'tool', 'vessel', 'dish', 'vegetable', 'fruit']
    breeds_factory = BREEDSFactory(info_dir=info_dir,
                                   data_dir=data_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    source_train_dataset = breeds_factory.get_breeds(
            ds_name = ds_name,
            partition = 'train',
            source = True,
            mode = "coarse",
            transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1),    
            transforms.ToTensor(),
            Lighting(0.05, IMAGENET_PCA['eigval'], 
                      IMAGENET_PCA['eigvec']),
            normalize,
        ]),
            split = 'rand'
        )
    
    source_val_dataset = breeds_factory.get_breeds(
            ds_name = ds_name,
            partition = 'val',
            source = True,
            mode = "coarse",
            transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
            split = 'rand'
        )
    
    source_val_loader = torch.utils.data.DataLoader(
        source_val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True)

    ''' create target_train dataset '''
    target_train_dataset = breeds_factory.get_breeds(
        ds_name=ds_name,
        partition='train',
        source=False,
        mode='coarse',
        transforms=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1),
            transforms.ToTensor(),
            Lighting(0.05, IMAGENET_PCA['eigval'],
                     IMAGENET_PCA['eigvec']),
            normalize,
        ]),
        split='rand'
    )

    ''' create target_train_val dataset '''
    target_train_val_augment_dataset = breeds_factory.get_breeds(
        ds_name=ds_name,
        partition='train',
        source=False,
        mode='coarse',
        transforms=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        split='rand'
    )

    ''' create target_val dataset (i.e., test set) '''
    val_val_dataset = breeds_factory.get_breeds(
        ds_name=ds_name,
        partition='val',
        source=False,
        mode='coarse',
        transforms=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        split='rand'
    )

    if ds_name == 'entity30':
        class_number = 30
    elif ds_name == 'entity13':
        class_number = 13

    coarse_to_fine_map = dict()
    for i in range(0, class_number):
        coarse_to_fine_map[i] = list()

    for key in target_train_dataset.class_to_idx.keys():
        coarse_to_fine_map[target_train_dataset.class_to_idx[key]].append(key)

    inc_step_num = 4

    task_coarse_class_dict = dict()
    task_size = inc_step_num + 1  
    for i in range(1, task_size):
        task_coarse_class_dict[i] = list()

    """ 
    create the subclasses list for each step.
    this can be different for each protocols
    """
    with open(task_stat_path, 'rb') as f:
        task_coarse_class_dict = pickle.load(f)

    assert set.union(set(task_coarse_class_dict[1]), set(task_coarse_class_dict[2]), set(task_coarse_class_dict[3]),
                     set(task_coarse_class_dict[4])) == set(target_train_dataset.class_to_idx.keys())

    '''
    create the train, train_val, test set and corresponding loaders
    '''

    ''' create each step's training images index dict '''
    task_training_idx_list_dict = dict()
    for i in range(1, task_size):
        task_training_idx_list_dict[i] = list()
        for subclass in task_coarse_class_dict[i]:
            temp_list = [j for j in range(0, len(target_train_dataset.samples)) if
                         target_train_dataset.samples[j][2] == subclass]
            task_training_idx_list_dict[i].extend(temp_list)

    ''' create each step's training Subset dict '''
    dset_train_train_task_dict = dict()
    for i in range(1, task_size):
        dset_train_train_task_dict[i] = torch.utils.data.dataset.Subset(target_train_dataset,
                                                                        task_training_idx_list_dict[i])

    ''' create each step's training loader dict '''
    target_train_train_task_loader_dict = dict()
    for i in range(1, task_size):
        train_sampler = None
        target_train_train_task_loader_dict[i] = torch.utils.data.DataLoader(dset_train_train_task_dict[i],
                                                                             batch_size=128,
                                                                             shuffle=True,
                                                                             num_workers=16,
                                                                             pin_memory=True,
                                                                             sampler=train_sampler)

    ''' create each step's testing images index dict '''
    task_val_idx_list_dict = dict()
    for i in range(1, task_size):
        task_val_idx_list_dict[i] = list()
        for subclass in task_coarse_class_dict[i]:
            temp_list = [j for j in range(0, len(val_val_dataset.samples)) if val_val_dataset.samples[j][2] == subclass]
            task_val_idx_list_dict[i].extend(temp_list)


    ''' create each step's testing Subset dict '''
    dset_val_val_task_dict = dict()
    for i in range(1, task_size):
        dset_val_val_task_dict[i] = torch.utils.data.dataset.Subset(val_val_dataset, task_val_idx_list_dict[i])

    ''' create each step's testing loader dict'''
    target_val_val_task_loader_dict = dict()
    for i in range(1, task_size):
        target_val_val_task_loader_dict[i] = torch.utils.data.DataLoader(dset_val_val_task_dict[i],
                                                                         batch_size=128,
                                                                         shuffle=False,
                                                                         num_workers=16,
                                                                         pin_memory=True)

    ''' Create the target_train dataset using the val augmentation. 
    This is used for calculate the mean feature in each previous step '''

    task_target_train_val_augment_idx_list_dict = dict()
    for i in range(1, task_size):
        task_target_train_val_augment_idx_list_dict[i] = list()
        for subclass in task_coarse_class_dict[i]:
            temp_list = [j for j in range(0, len(target_train_val_augment_dataset.samples)) if
                         target_train_val_augment_dataset.samples[j][2] == subclass]
            task_target_train_val_augment_idx_list_dict[i].extend(temp_list)


    dset_target_train_val_augment_task_dict = dict()
    for i in range(1, task_size):
        dset_target_train_val_augment_task_dict[i] = torch.utils.data.dataset.Subset(target_train_val_augment_dataset,
                                                                                     task_target_train_val_augment_idx_list_dict[i])

    target_train_val_augment_task_loader_dict = dict()
    for i in range(1, task_size):
        train_sampler = None
        target_train_val_augment_task_loader_dict[i] = torch.utils.data.DataLoader(
            dset_target_train_val_augment_task_dict[i], batch_size=128, shuffle=False,
            num_workers=16, pin_memory=True)


    ''' create each step's train_val images index dict '''
    train_val_class_size = 50
    task_training_val_idx_list_dict = dict()
    for i in range(1, task_size):
        task_training_val_idx_list_dict[i] = list()
        for subclass in task_coarse_class_dict[i]:
            temp_list = [j for j in range(0, len(target_train_val_augment_dataset.samples)) if
                         target_train_val_augment_dataset.samples[j][2] == subclass]
            task_training_val_idx_list_dict[i].extend(temp_list[0:train_val_class_size])

    ''' create each step's train_val Subset dict '''
    dset_train_train_val_task_dict = dict()
    for i in range(1, task_size):
        dset_train_train_val_task_dict[i] = torch.utils.data.dataset.Subset(target_train_val_augment_dataset,
                                                                            task_training_val_idx_list_dict[i])

    target_train_val_task_loader_dict = dict()
    for i in range(1, task_size):
        target_train_val_task_loader_dict[i] = torch.utils.data.DataLoader(dset_train_train_val_task_dict[i],
                                                                           batch_size=128,
                                                                           shuffle=False,
                                                                           num_workers=16,
                                                                           pin_memory=True)
        
    return classes, task_size, source_train_dataset, dset_train_train_task_dict, source_val_loader, target_val_val_task_loader_dict


@hydra.main(config_path=None, config_name=None, version_base="1.1") 
def continual_clip(cfg: DictConfig) -> None:

    cfg.workdir = utils.get_workdir(path=os.getcwd())

    utils.save_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(cfg, device)

    classes, task_size, source_train_dataset, dset_train_train_task_dict, source_val_loader, target_val_val_task_loader_dict = br_dt_get()
    model.classes_names = classes
    
    with open(cfg.log_path, 'w+') as f: 
        pass

    acc_list = []
    metric_logger = Logger(list_subsets=["test"])
    train_classes_names = classes
    
    for task_id in range(task_size):

        logging.info(f"Evaluation for task {task_id} has started.")

        if task_id == 0:
            model.adaptation(task_id, cfg, source_train_dataset, train_classes_names)
        else:    
            model.adaptation(task_id, cfg, dset_train_train_task_dict[task_id], train_classes_names)
        
        model.model.eval()
        
        for loop in range(task_size):

            if loop == 0:
                eval_loader = source_val_loader
            else:
                eval_loader = target_val_val_task_loader_dict[loop] 
            
            for _, (inputs, targets) in enumerate(tqdm(eval_loader)):

                assert torch.max(targets) <= 29, "Error: Found a target value greater than 29."

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, task_id)
                metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), torch.full_like(targets, loop)], subset="test")
            
        acc_list.append(100 * metric_logger.accuracy)
        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'acc': round(100 * metric_logger.accuracy, 2),
                'avg_acc': round(100 * metric_logger.average_incremental_accuracy, 2),
                'forgetting': round(100 * metric_logger.forgetting, 6),
                'acc_per_task': [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
                'bwt': round(100 * metric_logger.backward_transfer, 2),
                'fwt': round(100 * metric_logger.forward_transfer, 2),
            }) + '\n')
            metric_logger.end_task()
        
    with open(cfg.log_path, 'a+') as f:
        f.write(json.dumps({
            'last': round(acc_list[-1], 2), 
            'avg': round(statistics.mean(acc_list), 2)
        }) + '\n')


if __name__ == "__main__":
    continual_clip()