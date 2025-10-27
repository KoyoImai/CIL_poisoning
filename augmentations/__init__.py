




import torchvision.transforms as transforms

def make_augmentation(cfg):

    mean=(0.5071, 0.4867, 0.4408)
    std=(0.2675, 0.2565, 0.2761)

    if cfg.method.name in ["er", "finetune"]:
        augmentation = transforms.Compose([
            transforms.Resize(size=(cfg.dataset.size, cfg.dataset.size)),
            transforms.RandomCrop(cfg.dataset.size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_augmentation = transforms.Compose([
            transforms.Resize(size=(cfg.dataset.size, cfg.dataset.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:

        assert False




    return augmentation, test_augmentation










