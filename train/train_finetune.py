

import os
import numpy as np

import torch

from utils import AverageMeter, WindowAverageMeter, ProgressMeter



def train_finetune(model, criterion, optimizer, scheduler, train_loader, epoch, cfg, writer=None):

    model.train()

    # 学習記録
    losses = AverageMeter('Loss', ':.4e')
    accuracies = AverageMeter('Acc', ':.4e')
    batch_time = WindowAverageMeter('Time', fmt=':6.3f')
    data_time = WindowAverageMeter('Data', fmt=':6.3f')
    lr_meter = AverageMeter('LR', ':.4e')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, lr_meter] + [losses],
                             prefix="Epoch: [{}]".format(epoch), tbwriter=writer)
    

    # corr = [0.] * sum(cfg.continual.cls_per_task[:cfg.continual.target_task])
    # cnt  = [0.] * sum(cfg.continual.cls_per_task[:cfg.continual.target_task])
    # correct_task = 0.0


    for idx, (images, labels) in enumerate(train_loader):

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # モデルにデータを入力して出力を取得
        y_pred = model(images)
        print("len(y_pred): ", len(y_pred))
        print("y_pred[0].shape: ", y_pred[0].shape)

        assert False







    return



