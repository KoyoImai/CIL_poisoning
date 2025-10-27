


from datasets.dataset_cifar100 import make_cifar100_dataset


def set_buffer(cfg, prev_indices):

    replay_indices = []

    return replay_indices





def make_dataset(cfg, replay_indices, train_augmentation, test_augmentation):

    if cfg.dataset.type == "cifar100":

        # cfg, replay_indices, train_augmentation, test_augmentation, train=True
        train_dataset = make_cifar100_dataset(cfg=cfg, replay_indices=replay_indices,
                                              augmentation=train_augmentation, train=True)
        
        test_dataset = make_cifar100_dataset(cfg=cfg, replay_indices=replay_indices,
                                             augmentation=test_augmentation, train=False)
    

    else:
        assert False

    return train_dataset, test_dataset



