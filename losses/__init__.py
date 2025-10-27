


from losses.cross_entropy import SeenCrossEntropyLoss


def make_criterion(cfg):

    if cfg.method.name in ["er", "finetune"]:
        criterion = SeenCrossEntropyLoss()
    
    else:
        assert False

    return criterion





