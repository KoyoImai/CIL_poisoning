


import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler



def make_optimizer(cfg, model):

    
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.optimizer.train.learning_rate,
                          momentum=cfg.optimizer.train.momentum,
                          weight_decay=cfg.optimizer.train.weight_decay)


    return optimizer




def make_scheduler(cfg, optimizer):

    milestones = cfg.scheduler.train.milestones
    gamma = cfg.scheduler.train.gamma

    if cfg.method.name in ["er", "finetune"]:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    else:
        assert False


    return scheduler






