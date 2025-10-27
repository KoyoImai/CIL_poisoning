





def make_model(cfg):

    if cfg.method.name in ["er"]:

        from models.resnet_cifar_er_before import BackboneResNet
        # model  = BackboneResNet(name=cfg.model.type, seed=cfg.seed, cfg=cfg)

        from models.resnet_cifar_er import ResNet18
        model  = ResNet18(task_num=cfg.continual.n_task, nclasses=cfg.dataset.num_classes)
        print("model: ", model)


    else:
        assert False


    return model







