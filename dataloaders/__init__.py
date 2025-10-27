



import torch




def make_dataloader(cfg, train_dataset, test_dataset):

    batch_size = cfg.optimizer.train.batch_size
    num_workers = cfg.workers

    if cfg.method.name in ["er", "finetune"]:
        
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=num_workers,
                                                   pin_memory=True,
                                                   sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers,
                                                  pin_memory=True)

    else:
        assert False


    return train_loader, test_loader

