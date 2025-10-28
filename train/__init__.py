


from train.train_finetune import train_finetune
from train.eval_finetune import eval_finetune



def train(model, criterion, optimizer, scheduler, train_loader, epoch, cfg):

    if cfg.method.name in ["er", "finetune"]:

        train_finetune(model=model, criterion=criterion, optimizer=optimizer,
                       scheduler=scheduler, train_loader=train_loader, epoch=epoch, cfg=cfg)
    
    else:
        assert False





def evaluation(model, criterion, optimizer, scheduler, test_loader, epoch, cfg):

    if cfg.method.name in ["er", "finetune"]:

        return eval_finetune(model=model, criterion=criterion, optimizer=optimizer,
                             scheduler=scheduler, test_loader=test_loader, epoch=epoch, cfg=cfg)


    return 







