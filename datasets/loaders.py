import numpy as np
import torch
from torch.utils.data import DataLoader

from .image import get_image_datasets
from .weighted_uniform_sampler import WeightedUniformWithReplacementSampler


def get_loaders_from_config(cfg, device, **kwargs):
    if cfg["net"] == "cnn":
        flatten = False
    elif cfg["net"] == "mlp":
        flatten = True
    elif cfg["net"] == "logistic":
        flatten = True
    else:
        raise ValueError(f"Unknown net type {cfg['net']} for flattening")

    train_loader, valid_loader, test_loader = get_loaders(
        dataset=cfg["dataset"],
        device=device,
        data_root=cfg.get("data_root", "/root/autodl-tmp/data/"),
        train_batch_size=cfg["train_batch_size"],
        valid_batch_size=cfg["valid_batch_size"],
        test_batch_size=cfg["test_batch_size"],
        group_ratios=cfg["group_ratios"],
        seed=cfg["seed"],
        protected_group=cfg["protected_group"],
        make_valid_loader=cfg["make_valid_loader"],
        flatten=flatten,
        weighted_sampling=cfg["weighted_sampling"],
        sample_rate=cfg["sample_rate"]
    )

    if cfg["dataset"] in ["celeba"]:
        train_dataset_shape = train_loader.dataset.shape
    else:
        train_dataset_shape = train_loader.dataset.x.shape
    cfg["train_dataset_size"] = train_dataset_shape[0]
    cfg["data_shape"] = tuple(train_dataset_shape[1:])
    cfg["data_dim"] = int(np.prod(cfg["data_shape"]))

    if not cfg["make_valid_loader"]:
        valid_loader = test_loader
        print("WARNING: Using test loader for validation")

    return train_loader, valid_loader, test_loader


def get_loaders(
        dataset,
        device,
        data_root,
        train_batch_size,
        valid_batch_size,
        test_batch_size,
        group_ratios,
        seed,
        protected_group,
        make_valid_loader,
        flatten,
        weighted_sampling,
        sample_rate,
):
    # NOTE: only training and validation sets sampled according to group_ratios
    if dataset in ["mnist", "fashion-mnist", "cifar10", "svhn", "celeba"]:
        train_dset, valid_dset, test_dset = get_image_datasets(dataset, data_root, seed, group_ratios,
                                                               make_valid_loader, flatten)

    # NOTE: entire dataset sampled according to group_ratios
    # elif dataset in ["adult", "dutch"]:
    #     train_dset, valid_dset, test_dset = get_tabular_datasets(dataset, data_root, seed, protected_group,
    #                                                              group_ratios, make_valid_loader)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    train_loader = get_loader(train_dset, device, train_batch_size, drop_last=False,
                              weighted_sampling=weighted_sampling, sample_rate=sample_rate)

    if make_valid_loader:
        valid_loader = get_loader(valid_dset, device, valid_batch_size, drop_last=False)
    else:
        valid_loader = None

    test_loader = get_loader(test_dset, device, test_batch_size, drop_last=False)

    return train_loader, valid_loader, test_loader


def get_loader(dset, device, batch_size, drop_last, weighted_sampling=None, sample_rate=None):
    if weighted_sampling is not None:
        num_groups, group_counts = torch.unique(dset.z, return_counts=True)
        groups = dset.z
        if weighted_sampling == 1:
            # for auto-s
            group_weights = 1 / group_counts
        elif weighted_sampling == 2:
            # for fdpog
            group_weights = torch.ones(len(num_groups))
        weights = group_weights[groups]
        weights = weights / weights.sum() * len(dset)
        sampler = WeightedUniformWithReplacementSampler(weights,
                                                        num_samples=len(dset), sample_rate=sample_rate)

        return DataLoader(
            dset.to(device),
            shuffle=False,
            batch_sampler=sampler,
            collate_fn=None,
            batch_size=1,
            num_workers=0,
            pin_memory=False)
    else:
        return DataLoader(
            dset.to(device),
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=0,
            pin_memory=False
        )
