

import os
import sys
import csv
import time
import numpy as np
from itertools import zip_longest

import torch

from utils import AverageMeter, WindowAverageMeter, ProgressMeter



def write_csv(value, path, file_name, task, epoch):

    # ファイルパスを生成
    file_path = f"{path}/{file_name}.csv"

    # ファイルが存在しなければ新規作成、かつヘッダー行を記入する
    # value がリストの場合は、ヘッダーの値部分は要素数に合わせて "value_1", "value_2", ... とする例
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # ヘッダー行を定義（必要に応じて適宜変更）
            if isinstance(value, list):
                header = ["task"] + ["epoch"] + [f"task_{i+1}" for i in range(len(value))]
            else:
                header = ["task", "epoch", "value"]
            writer.writerow(header)

    # CSV に実際のデータを追加記録する
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if isinstance(value, list):
            row = [task] + [epoch] + value
        else:
            row = [task, epoch, value]
        writer.writerow(row)





def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def eval_finetune(model, criterion, optimizer, scheduler, test_loader, epoch, cfg, writer=None):

    model.eval()

    all_stats = {}
    targets = []
    preds = []

    # 学習記録
    batch_time = AverageMeter('Time', ':6.3f', tbname='val/time')
    losses = AverageMeter('Loss', ':.4e', tbname='val/loss')
    top1 = AverageMeter('Acc@1', ':6.2f', tbname='val/top1')
    top5 = AverageMeter('Acc@5', ':6.2f', tbname='val/top5')
    progress = ProgressMeter(len(test_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ',
                             tbwriter=writer)
    

    # amp の使用状況
    use_amp = bool(getattr(cfg, "amp", None) and cfg.amp.use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and str(cfg.amp.dtype).lower() == "bf16") else torch.float16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    taskid = cfg.continual.target_task

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(test_loader):

            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # モデルにデータを入力して出力を取得
            logits = model(images)

            # 損失計算
            loss = criterion(logits, labels, taskid)

            # 訓練精度の計算 
            y_pred = torch.cat(list(logits[:taskid+1]), dim=1)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(y_pred, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if idx % cfg.log.print_freq == 0:
                progress.display(idx)
                sys.stdout.flush()


        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))

        progress.tbwrite(0)
        all_stats['acc1'] = top1.avg
        all_stats['acc5'] = top5.avg

        print("all_stats['acc1']: ", all_stats['acc1'])
        print("all_stats['acc5']: ", all_stats['acc5'])


        
    write_csv(top1.avg.item(), cfg.log.result_path, file_name="top1_acc", task=taskid, epoch=epoch)
    write_csv(top5.avg.item(), cfg.log.result_path, file_name="top5_acc", task=taskid, epoch=epoch)








