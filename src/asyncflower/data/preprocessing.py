from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner, DirichletPartitioner
from asyncflower.partitioner.group_labels_inner_dirichlet import GroupLabelsInnerDirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Lambda
import datasets
import numpy as np
import os
import shutil

transforms = {
    "FashionMNIST": lambda batch: {
        "label": batch["label"], "image": [
            Compose([ToTensor(), Normalize(0.5, 0.5)])(img) for img in batch["image"]
        ]
    }, 
    
    "CIFAR-10": lambda batch: {
        "label": batch["label"], "image": [
            Compose([ToTensor(), Normalize(0.5, 0.5)])(img) for img in batch["img" if "img" in batch else "image"]
        ]
    }, 

    "CIFAR-100": lambda batch: {
        "label": batch["fine_label"], "image": [
            Compose([ToTensor(), Normalize(0.5, 0.5)])(img) for img in batch["img" if "img" in batch else "image"]
        ]
    }, 

    "TinyImageNet": lambda batch: {
        "label": batch["label"], "image": [
            Compose([Lambda(lambda img: img.convert("RGB")), ToTensor(), Normalize(0.5, 0.5)])(img) for img in batch["img" if "img" in batch else "image"]
        ]
    }
}

dataset_names = {
    "FashionMNIST": "zalando-datasets/fashion_mnist",
    "CIFAR-10": "uoft-cs/cifar10",
    "CIFAR-100": "uoft-cs/cifar100",
    "TinyImageNet": "zh-plus/tiny-imagenet"
}

def get_dirichlet_partitioner(
    num_partitions: int, 
    partition_by: str, 
    alpha: float = 0.1, 
    seed: int = 10, 
    shuffle: bool = True
):
    partitioner = DirichletPartitioner(
        num_partitions = num_partitions,
        alpha = alpha,
        seed = seed,
        partition_by = partition_by,
        shuffle = shuffle
    )
    return partitioner

def get_labels_dirichlet_partitioner(
    labels_per_partition: dict, 
    partition_by: str = "label", 
    alpha: float = 0.1, 
    max_attempts: int = 5, 
    min_num_samples: int = None,
    seed: int = 0
):
    partitioner = GroupLabelsInnerDirichletPartitioner(
        partition_by = partition_by,
        labels_per_partition = labels_per_partition,
        alpha = alpha,
        max_attempts = max_attempts,
        min_num_samples = min_num_samples,
        seed = seed
    )
    return partitioner

def generate_datasets(dataset: str, num_partitions: int, partitioner: Partitioner, save_dir: str) -> None:
    filepath = save_dir
    
    if os.path.exists(filepath):
        shutil.rmtree(filepath)

    fed_dataset = FederatedDataset(dataset = dataset_names[dataset], partitioners = {"train": partitioner})
    
    for i in range(num_partitions):
        partition_filepath = f"{filepath}/partition-{i}"
        fed_dataset.load_partition(i).save_to_disk(partition_filepath)

    test_filepath = f"{filepath}/test"
    fed_dataset.load_split("test").save_to_disk(test_filepath)

    train_filepath = f"{filepath}/train"
    fed_dataset.load_split("train").save_to_disk(train_filepath)

def load_test_data(data_dir: str, dataset: str, batch_size: int) -> DataLoader:
    filename = f"{data_dir}/test"
    test_dataset = datasets.load_from_disk(filename).with_transform(transforms[dataset])
    test_loader = DataLoader(test_dataset, batch_size = batch_size)
    return test_loader

def load_train_data(data_dir: str, dataset: str, batch_size: int) -> DataLoader:
    filename = f"{data_dir}/train"
    train_dataset = datasets.load_from_disk(filename).with_transform(transforms[dataset])
    train_loader = DataLoader(train_dataset, batch_size = batch_size)
    return train_loader

def load_client_data(
    cid: int, data_dir: str, dataset: str, batch_size: int, seed: int, return_train_val_splits: bool = True
) -> tuple[DataLoader, DataLoader] | DataLoader:
    partition_filename = f"{data_dir}/partition-{cid}"
    partition = datasets.load_from_disk(partition_filename).with_transform(transforms[dataset])
    
    if return_train_val_splits:
        np.random.seed(seed)
        train_val_splits = partition.train_test_split(test_size = 0.2)
        train_loader = DataLoader(train_val_splits["train"], batch_size = batch_size, shuffle = True)
        val_loader = DataLoader(train_val_splits["test"], batch_size = batch_size)
        return train_loader, val_loader
    
    return DataLoader(partition, batch_size = batch_size, shuffle = True)