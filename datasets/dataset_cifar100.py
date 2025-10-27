
import os
import numpy as np


import torch
from torch.utils.data import Subset
from torchvision import transforms, datasets


def make_cifar100_dataset(cfg, replay_indices, augmentation, train=True):


    target_task = cfg.continual.target_task
    cls_per_task = cfg.continual.cls_per_task


    # 現在タスクのクラスリスト
    if train:
        target_classes = list(range(target_task*cls_per_task, (target_task+1)*cls_per_task))
    else:
        target_classes = list(range(0, (target_task+1)*cls_per_task))
    print(target_classes)


    # 一旦，データセット全体を用意
    subset_indices = []
    _train_dataset = datasets.CIFAR100(root=cfg.dataset.data_folder,
                                       transform=augmentation,
                                       train=train,
                                       download=True)
    
    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()
    

    # リプレイバッファのインデックスを加える
    subset_indices += replay_indices

    dataset =  Subset(_train_dataset, subset_indices)
    print('Dataset size: {}'.format(len(subset_indices)))
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
    print(uc[np.argsort(uk)])



    return dataset





