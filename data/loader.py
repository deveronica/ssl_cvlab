from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from data.dataset import DATASET_GETTERS

def get_dataset(cfg):
    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[cfg.dataset](
        root=cfg.root, cfg=cfg)
    return labeled_dataset, unlabeled_dataset, test_dataset


def get_dataloader(cfg):
    labeled_dataset, unlabeled_dataset, test_dataset = get_dataset(cfg)

    train_sampler = RandomSampler
    test_sampler = SequentialSampler
    
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=cfg.batch_size*cfg.mu,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True)

    testloader = DataLoader(
        test_dataset,
        sampler=test_sampler(test_dataset),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True)
    
    return labeled_trainloader, unlabeled_trainloader, testloader
