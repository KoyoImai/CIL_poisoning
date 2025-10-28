
import os
import hydra


import torch



from utils import seed_everything
from models import make_model




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
    pretrained_path = os.path.join(cfg.log.model_path, f"model_{target_task:02d}.pth")
    state_dict = torch.load(pretrained_path, map_location="cpu")["model"]
    model.load_state_dict(state_dict, strict=False)
    

    # # ---------- 読み込み後の同パラメータを確認 ----------
    # after = model.state_dict()[name].detach().cpu()
    # print("after[:5]: ", after.flatten()[:5])
    # print("changed?: ", not torch.equal(before, after))
    # print("max|diff|: ", (before - after).abs().max().item())





if __name__ == "__main__":

    main()


# /home/kouyou/ContinualLearning/repexp/Brainwash/afec_ewc_lamb_500000.0_lambemp_100.0__model_type_resnet_dataset_split_cifar100_class_num_10_bs_16_lr_0.01_n_epochs_20__model_name_ResNet_task_num_9__seed_0_emb_fact_1_im_sz_32__last_task_9___.pkl

# python main_inv.py --pretrained_model_add=./afec_ewc_lamb_500000.0_lambemp_100.0__model_type_resnet_dataset_split_cifar100_class_num_10_bs_16_lr_0.01_n_epochs_20__model_name_ResNet_task_num_9__seed_0_emb_fact_1_im_sz_32__last_task_9___.pkl --num_samples=128 --save_dir=./save_dir --task_lst=0,1,2,3,4,5,6,7,8 --save_every=1000 --batch_reg --init_acc --n_iters=10000


CUDA_VISIBLE_DEVICES=0 python main_brainwash.py --extra_desc=reckless_test --pretrained_model_add=./afec_ewc_lamb_500000.0_lambemp_100.0__model_type_resnet_dataset_split_cifar100_class_num_10_bs_16_lr_0.01_n_epochs_20__model_name_ResNet_task_num_9__seed_0_emb_fact_1_im_sz_32__last_task_9___.pkl --mode='reckless' --target_task_for_eval=0 --delta=0.3 --seed=0 --eval_every=10 --distill_folder=./save_dir --init_acc --noise_norm=inf --cont_learner_lr=0.001 --n_epochs=5000 --save_every=100 

