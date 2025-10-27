

import os
import time
import numpy as np

import torch

from utils import AverageMeter, WindowAverageMeter, ProgressMeter



def train_finetune(model, criterion, optimizer, scheduler, train_loader, epoch, cfg, writer=None):

    model.train()

    # 学習記録
    losses = AverageMeter('Loss', ':.4e')
    accuracies = AverageMeter('Acc', ':.4f')
    batch_time = WindowAverageMeter('Time', fmt=':6.3f')
    data_time = WindowAverageMeter('Data', fmt=':6.3f')
    lr_meter = AverageMeter('LR', ':.4e')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, lr_meter] + [losses] + [accuracies],
                             prefix="Epoch: [{}]".format(epoch), tbwriter=writer)
    

    # corr = [0.] * sum(cfg.continual.cls_per_task[:cfg.continual.target_task])
    # cnt  = [0.] * sum(cfg.continual.cls_per_task[:cfg.continual.target_task])
    # correct_task = 0.0

    taskid = cfg.continual.target_task

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        data_time.update(time.time() - end)

        # モデルにデータを入力して出力を取得
        logits = model(images)
        # print("len(y_pred): ", len(y_pred))
        # print("y_pred[0].shape: ", y_pred[0].shape)

        # 損失計算
        loss = criterion(logits, labels, taskid)
        losses.update(loss.item(), images[0].size(0))

        # 訓練精度の計算 
        y_pred = torch.cat(list(logits[:taskid+1]), dim=1)
        preds = y_pred.argmax(dim=1)               # 予測ラベル
        # print("y_pred.shape: ", y_pred.shape)
        # print("preds.shape: ", preds.shape)
        # print('labels.shape: ', labels.shape)

        correct = preds.eq(labels).sum().item()    # 正解数

        acc = correct / bsz
        accuracies.update(acc, bsz)

        # 最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        # 現在の学習率
        current_lr = optimizer.param_groups[0]['lr']
        lr_meter.update(current_lr)


        # 学習状況の表示
        # print("len(trainloader.batch_sampler.buffer): ", len(trainloader.batch_sampler.buffer))
        if idx % cfg.log.print_freq == 0:
            tb_step = (epoch * len(train_loader.dataset) // cfg.optimizer.train.batch_size + idx)
            progress.display(idx)
            progress.tbwrite(tb_step)




    return



