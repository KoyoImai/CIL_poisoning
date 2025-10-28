

import os
import hydra
import random


import torch


from utils import seed_everything
from models import make_model
from losses import make_criterion
from optimizers import make_optimizer, make_scheduler
from augmentations import make_augmentation
from datasets import set_buffer, make_dataset
from dataloaders import make_dataloader
from train import train




def preparation(cfg):

    # データセット毎にタスク数・タスク毎のクラス数を決定
    # （現状不要）

    # 総タスク数
    # （現状不要）

    # モデルの保存，実験記録などの保存先パス
    if cfg.dataset.data_folder is None:
        cfg.dataset.data_folder = '~/data/'
    cfg.log.model_path = f'./logs/{cfg.method.name}/{cfg.log.name}/model/'      # modelの保存先
    cfg.log.explog_path = f'./logs/{cfg.method.name}/{cfg.log.name}/exp_log/'   # 実験記録の保存先
    cfg.log.mem_path = f'./logs/{cfg.method.name}/{cfg.log.name}/mem_log/'      # リプレイバッファ内の保存先
    cfg.log.result_path = f'./logs/{cfg.method.name}/{cfg.log.name}/result/'    # 結果の保存先

    # ディレクトリ作成
    if not os.path.isdir(cfg.log.model_path):
        os.makedirs(cfg.log.model_path)
    if not os.path.isdir(cfg.log.explog_path):
        os.makedirs(cfg.log.explog_path)
    if not os.path.isdir(cfg.log.mem_path):
        os.makedirs(cfg.log.mem_path)
    if not os.path.isdir(cfg.log.result_path):
        os.makedirs(cfg.log.result_path)



def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...'+save_file)
    if opt.method in ["cclis-pcgrad"]:
        state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer._optim.state_dict(),
        'epoch': epoch,
    }
    else:
        state = {
            'opt': opt,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
    torch.save(state, save_file)
    del state


@hydra.main(config_path='configs/default/', config_name='default', version_base=None)
def main(cfg):

    # ===========================================
    # シード値固定
    # ===========================================
    seed_everything(cfg.seed)

    # logの名前を決定
    cfg.log.name = f"{cfg.log.base}_{cfg.method.name}_{cfg.continual.buffer_type}{cfg.continual.buffer_size}_{cfg.dataset.type}_seed{cfg.seed}_date{cfg.date}"



    # ===========================================
    # データローダ作成やディレクトリ作成などの前処理
    # ===========================================
    preparation(cfg)


    # ===========================================
    # modelの作成
    # ===========================================
    model = make_model(cfg)


    # ===========================================
    # 損失関数の作成
    # ===========================================
    criterion = make_criterion(cfg)


    # バッファ内データのインデックス
    replay_indices = None


    # ===========================================
    # タスクを順番に処理
    # ===========================================
    n_task = cfg.continual.n_task
    for taskid in range(n_task):

        print(f"=== Training task {taskid}/{n_task-1} ===")
        cfg.continual.target_task = taskid

        # ===========================================
        # Optimizer の作成
        # ===========================================
        optimizer = make_optimizer(cfg, model)


        # ===========================================
        # データローダーの作成
        # ===========================================
        replay_indices = set_buffer(cfg=cfg, model=model, prev_indices=replay_indices)
        train_augmentation, test_augmentation = make_augmentation(cfg)
        train_dataset, test_dataset = make_dataset(cfg=cfg, replay_indices=replay_indices,
                                                   train_augmentation=train_augmentation,
                                                   test_augmentation=test_augmentation)
        
        train_loader, test_loader = make_dataloader(cfg, train_dataset, test_dataset)


        # ===========================================
        # scheduler の作成
        # ===========================================
        scheduler = make_scheduler(cfg, optimizer)
        

        # ===========================================
        # 学習を実行
        # ===========================================
        epochs = cfg.optimizer.train.epochs
        for epoch in range(epochs):
            
            # model, criterion, optimizer, scheduler, train_loader, epoch, cfg
            train(model=model, criterion=criterion, optimizer=optimizer,
                  scheduler=scheduler, train_loader=train_loader, epoch=epoch, cfg=cfg)

            # 学習率の調整
            scheduler.step()

            # 学習途中のパラメータを保存
            dir_path = f"{cfg.log.model_path}/task{cfg.continual.target_task:02d}"
            file_path = f"{dir_path}/model_epoch{epoch:03d}.pth"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            save_model(model, optimizer, cfg, cfg.optimizer.train.epochs, file_path)



        # 保存（opt.model_path）
        file_path = f"{cfg.log.model_path}/model_{cfg.continual.target_task:02d}.pth"
        # save_model(model, method_tools["optimizer"], opt, opt.epochs, file_path)
        save_model(model, optimizer, cfg, cfg.optimizer.train.epochs, file_path)










if __name__ == "__main__":
    main()


