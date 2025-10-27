

import torch



def make_model(cfg):

    if cfg.method.name in ["er", "finetune"]:

        from models.resnet_cifar_er_before import BackboneResNet
        # model  = BackboneResNet(name=cfg.model.type, seed=cfg.seed, cfg=cfg)

        from models.resnet_cifar_er import ResNet18
        # model  = ResNet18(task_num=cfg.continual.n_task, nclasses=cfg.dataset.num_classes)
        model  = ResNet18(task_num=cfg.continual.n_task, nclasses=cfg.dataset.num_classes, include_head=False)

        print("model: ", model)


    else:
        assert False


    if torch.cuda.is_available():
        model = model.cuda()
    

    return model







