
import random
import math
import numpy as np

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, Subset



# ring buffer
def replay_ring(cfg, model, prev_indices=None):

    is_training = model.training
    model.eval()

    class IdxDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            return self.indices[idx], self.dataset[idx]

    # データローダの仮作成（ラベルがほしいだけ）
    val_transform = transforms.Compose([
        transforms.Resize(cfg.dataset.size),
        transforms.ToTensor(),
    ])

    if cfg.dataset.type == 'cifar100':
        subset_indices = []
        val_dataset = datasets.CIFAR100(root=cfg.dataset.data_folder,
                                         transform=val_transform,
                                         download=True)
        val_targets = np.array(val_dataset.targets)


    else:
        raise ValueError('dataset not supported: {}'.format(cfg.dataset.type))
    

    if prev_indices is None:
        prev_indices = []
        observed_classes = list(range(0, cfg.continual.target_task*cfg.continual.cls_per_task))
    else:

        # 過去タスクのデータに割り当てるバッファのサイズ
        shrink_size = ((cfg.continual.target_task - 1) * cfg.continual.buffer_size / cfg.continual.target_task)

        if len(prev_indices) > 0:
            unique_cls = np.unique(val_targets[prev_indices])
            _prev_indices = prev_indices
            prev_indices = []

            for c in unique_cls:
                mask = val_targets[_prev_indices] == c
                size_for_c = shrink_size / len(unique_cls)
                p = size_for_c - (shrink_size // len(unique_cls))
                if random.random() < p:
                    size_for_c = math.ceil(size_for_c)
                else:
                    size_for_c = math.floor(size_for_c)

                # 各クラス均等になるようにバッファ内のデータを削除
                prev_indices += torch.tensor(_prev_indices)[mask][torch.randperm(mask.sum())[:size_for_c]].tolist()

            print(np.unique(val_targets[prev_indices], return_counts=True))

        # 前回タスクのクラス範囲
        observed_classes = list(range(max(cfg.continual.target_task-1, 0)*cfg.continual.cls_per_task, (cfg.continual.target_task)*cfg.continual.cls_per_task))
    
    print("buffer_er.py observed_classes: ", observed_classes)

    # 確認済みのクラス（前回タスク）がない場合終了
    if len(observed_classes) == 0:
        return prev_indices
    

    # 確認済みクラスのインデックスを獲得
    observed_indices = []
    for tc in observed_classes:
        observed_indices += np.where(val_targets == tc)[0].tolist()


    val_observed_targets = val_targets[observed_indices]
    val_unique_cls = np.unique(val_observed_targets)
    print("val_unique_cls: ", val_unique_cls)


    print("cfg.continual.buffer_size: ", cfg.continual.buffer_size)
    selected_observed_indices = []
    for c_idx, c in enumerate(val_unique_cls):
        size_for_c_float = ((cfg.continual.buffer_size - len(prev_indices) - len(selected_observed_indices)) / (len(val_unique_cls) - c_idx))
        print("size_for_c_flaot: ", size_for_c_float)
        p = size_for_c_float -  ((cfg.continual.buffer_size - len(prev_indices) - len(selected_observed_indices)) // (len(val_unique_cls) - c_idx))
        if random.random() < p:
            size_for_c = math.ceil(size_for_c_float)
        else:
            size_for_c = math.floor(size_for_c_float)
        mask = val_targets[observed_indices] == c
        selected_observed_indices += torch.tensor(observed_indices)[mask][torch.randperm(mask.sum())[:size_for_c]].tolist()
    print(np.unique(val_targets[selected_observed_indices], return_counts=True))


    model.is_training = is_training

    return prev_indices + selected_observed_indices




