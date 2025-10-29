
import os
import hydra
import numpy as np


import torch
import torch.nn as nn



from utils import seed_everything
from models import make_model
from optimizers.proj_adam import ProjAdam





class BatchNormHook:
    def __init__(self, model):
        self.data = {}  # Dictionary to store information
        self.handles = []  # List to store hook handles

        def hook_fn(module, input, output):
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                self.data[module] = {
                    'input': input[0].clone(),
                    'running_mean': module.running_mean.clone().detach(),
                    'running_var': module.running_var.clone().detach()
                }

        # Register hooks for each batch norm layer in the model
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                handle = layer.register_forward_hook(hook_fn)
                self.handles.append(handle)
                


    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()




def cal_tv_loss(x):
    if x.shape[-1] == 784:
        x = x.reshape(x.shape[0], 1, 28, 28)

    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
                torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    
    return tv_loss




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

    # model inversion用のディレクトリ
    cfg.inversion.save_dir = f"./logs/{cfg.method.name}/{cfg.log.name}/inversion_resilt/"

    # ディレクトリ作成
    if not os.path.isdir(cfg.log.model_path):
        os.makedirs(cfg.log.model_path)
    if not os.path.isdir(cfg.log.explog_path):
        os.makedirs(cfg.log.explog_path)
    if not os.path.isdir(cfg.log.mem_path):
        os.makedirs(cfg.log.mem_path)
    if not os.path.isdir(cfg.log.result_path):
        os.makedirs(cfg.log.result_path)
    if not os.path.isdir(cfg.inversion.save_dir):
        os.makedirs(cfg.inversion.save_dir)




@hydra.main(config_path='configs/default/', config_name='default', version_base=None)
def main(cfg):

    # ===========================================
    # シード値固定
    # ===========================================
    seed_everything(cfg.seed)

    # logの名前を決定
    cfg.log.name = f"{cfg.log.base}_{cfg.method.name}_{cfg.continual.buffer_type}{cfg.continual.buffer_size}_{cfg.dataset.type}_seed{cfg.seed}_date{cfg.date}"

    task_list = cfg.inversion.task_list
    cls_list = list(range(0, cfg.continual.cls_per_task * (cfg.inversion.target_task)))

    n_iters = cfg.inversion.n_iters
    

    # ===========================================
    # データローダ作成やディレクトリ作成などの前処理
    # ===========================================
    preparation(cfg)


    # ===========================================
    # modelの作成 & 学習済みパラメータの読み込み
    # ===========================================
    model = make_model(cfg)

    # # ---------- 観察対象（最初の1パラメータ）のスナップショット ----------
    # name, before = next(iter(model.state_dict().items()))
    # before = before.detach().cpu().clone()
    # print(f"[TARGET] {name}")
    # print("before[:5]:", before.flatten()[:5])

    target_task = int(cfg.inversion.target_task)
    pretrained_path = os.path.join(cfg.log.model_path, f"model_{target_task-1:02d}.pth")
    state_dict = torch.load(pretrained_path, map_location="cpu")["model"]
    model.load_state_dict(state_dict, strict=False)
    

    # # ---------- 読み込み後の同パラメータを確認 ----------
    # after = model.state_dict()[name].detach().cpu()
    # print("after[:5]: ", after.flatten()[:5])
    # print("changed?: ", not torch.equal(before, after))
    # print("max|diff|: ", (before - after).abs().max().item())



    # ===========================================
    # BN のフック登録
    # ===========================================
    bn_hook = BatchNormHook(model)


    # ===========================================
    # 損失関数の作成
    # ===========================================
    loss_fn = torch.nn.CrossEntropyLoss()


    # ===========================================
    # BrainWashとは異なり，single headなので一括で実行
    # ===========================================
    x_dst = torch.rand(cfg.inversion.num_samples, 3, cfg.dataset.size, cfg.dataset.size).cuda()
    x_dst.requires_grad = True

    if cfg.inversion.num_samples > len(cls_list):
        y_dst = torch.randint(0, len(cls_list), (cfg.inversion.num_samples,)).cuda()
    else:
        assert False
    
    print("y_dst.shape: ", y_dst.shape)    # y_dst.shape:  torch.Size([128])
    print("y_dst: ", y_dst)

    

    optim = ProjAdam([x_dst], lr=1e-2, nrm=1, norm_type='inf')

    sch = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[5000, 10000], gamma=0.1)

    loss_ = []
    save_every = 100

    for i in range(n_iters):
        model.zero_grad()
        optim.zero_grad()

        pred = model(x_dst)
        pred = torch.cat(list(pred[:cfg.inversion.target_task]), dim=1)
        # print("pred.shape: ", pred.shape)    # pred.shape:  torch.Size([128, 80])

        # assert False

        tv_loss = cal_tv_loss(x_dst)    
        norm_loss = torch.norm(x_dst, p=2, dim=1).mean()
        task_loss = loss_fn(pred, y_dst)

        loss_bn = 0

        for bn_k in bn_hook.data.keys():
            bn_mean = bn_hook.data[bn_k]['running_mean']  
            bn_var = bn_hook.data[bn_k]['running_var']    

            bn_in = bn_hook.data[bn_k]['input'] 
            bn_in = bn_in.transpose(0, 1).reshape(bn_in.shape[1], -1)
            
            bn_in_mean = bn_in.mean(dim=-1)   
            bn_in_var = bn_in.var(dim=-1)     

            loss_bn += (bn_in_mean - bn_mean).norm() + (bn_in_var - bn_var).norm()


        if cfg.inversion.reg == True:
            loss = 1e5 * task_loss + tv_loss * 1e2 + norm_loss * 1e4 + loss_bn * 1e4        

        else:
            loss = task_loss
        

        loss.backward()
        optim.step()
        sch.step()
        loss_.append(loss.item())

        if (i+1) % save_every == 0:
            print(f'iter {i+1} loss: {loss.item()} tv_loss: {tv_loss.item()} norm_loss: {norm_loss.item()} task_loss: {task_loss.item()} loss_bn: {loss_bn.item()}')


    np.savez(f'{cfg.inversion.save_dir}/target_task_{target_task}.npz', 
                x_dst=x_dst.detach().cpu().numpy(), y_dst=y_dst.detach().cpu().numpy(), target_task=target_task)

    torch.save(x_dst, f'{cfg.inversion.save_dir}/xdst_target_task{target_task}.pth')
    torch.save(y_dst, f'{cfg.inversion.save_dir}/ydst_target_task{target_task}.pth')



if __name__ == "__main__":

    main()


# /home/kouyou/ContinualLearning/repexp/Brainwash/afec_ewc_lamb_500000.0_lambemp_100.0__model_type_resnet_dataset_split_cifar100_class_num_10_bs_16_lr_0.01_n_epochs_20__model_name_ResNet_task_num_9__seed_0_emb_fact_1_im_sz_32__last_task_9___.pkl

# CUDA_VISIBLE_DEVICES=3 python main_inv.py --pretrained_model_add=./afec_ewc_lamb_500000.0_lambemp_100.0__model_type_resnet_dataset_split_cifar100_class_num_10_bs_16_lr_0.01_n_epochs_20__model_name_ResNet_task_num_9__seed_0_emb_fact_1_im_sz_32__last_task_9___.pkl --num_samples=128 --save_dir=./save_dir --task_lst=0,1,2,3,4,5,6,7,8 --save_every=1000 --batch_reg --init_acc --n_iters=10000


# CUDA_VISIBLE_DEVICES=0 python main_brainwash.py --extra_desc=reckless_test --pretrained_model_add=./afec_ewc_lamb_500000.0_lambemp_100.0__model_type_resnet_dataset_split_cifar100_class_num_10_bs_16_lr_0.01_n_epochs_20__model_name_ResNet_task_num_9__seed_0_emb_fact_1_im_sz_32__last_task_9___.pkl --mode='reckless' --target_task_for_eval=0 --delta=0.3 --seed=0 --eval_every=10 --distill_folder=./save_dir --init_acc --noise_norm=inf --cont_learner_lr=0.001 --n_epochs=5000 --save_every=100 

# CUDA_VISIBLE_DEVICES=0 python main_baselines.py --experiment split_mini_imagenet --approach afec_ewc --lasttask 9 --tasknum 10 --nepochs 20 --batch-size 16 --lr 0.01 --clip 100. --lamb 500000 --lamb_emp 100  --checkpoint <noise_pkl_file>  --init_acc --addnoise
