


from train.train_finetune import train_finetune



def train(model, criterion, optimizer, scheduler, train_loader, epoch, cfg):

    if cfg.method.name in ["er", "finetune"]:

        train_finetune(model=model, criterion=criterion, optimizer=optimizer,
                       scheduler=scheduler, train_loader=train_loader, epoch=epoch, cfg=cfg)
    
    else:
        assert False













