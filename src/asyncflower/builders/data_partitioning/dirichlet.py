import os
import argparse
from flwr_datasets.partitioner import DirichletPartitioner
from asyncflower.data.preprocessing import generate_datasets
from typing import Callable

def build_dirichlet_partitions(
    dataset: str,
    num_partitions: int, 
    partition_by: str,
    alpha: float,
    min_num_samples: int,
    shuffle: bool,
    seed: int,
    save_dir: str
) -> Callable:
    partitions_dir = f"{save_dir}/data/{dataset}/n{num_partitions}-dirichlet-alpha{str(alpha).replace('.', 'd')}/partitions-seed{seed}"

    if not os.path.exists(partitions_dir):
        partitioner = DirichletPartitioner(
            num_partitions = num_partitions,
            partition_by = partition_by,
            alpha = alpha, 
            min_partition_size = min_num_samples,
            shuffle = shuffle,
            seed = seed
        )

        generate_datasets(dataset, num_partitions, partitioner, partitions_dir)

    return partitions_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_partitions", required = True, dest = "num_partitions", type = int)
    parser.add_argument("--save_dir", required = True, dest = "save_dir", type = str)    
    parser.add_argument("--dataset", required = True, dest = "dataset", type = str)
    parser.add_argument("--partition_by", required = True, dest = "partition_by", type = str)
    parser.add_argument("--alpha", required = True, dest = "alpha", type = float)
    parser.add_argument("--min_num_samples", required = True, dest = "min_num_samples", type = int)
    parser.add_argument("--shuffle", dest = "shuffle", action = "store_true")
    parser.add_argument("--seed", required = True, dest = "seed", type = int)
    
    args = parser.parse_args()

    print("Preparing Data...")
    partitions_dir = build_dirichlet_partitions(
        dataset = args.dataset,
        num_partitions = args.num_partitions,
        partition_by = args.partition_by,
        alpha = args.alpha, 
        min_num_samples = args.min_num_samples, 
        shuffle = args.shuffle, 
        seed = args.seed,
        save_dir = args.save_dir
    )
    print(f"Partitions were stored in {partitions_dir}...")