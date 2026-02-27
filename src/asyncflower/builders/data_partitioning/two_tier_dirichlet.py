import os
import json
import argparse
import numpy as np
from asyncflower.partitioner.two_tier_dirichlet import TwoTierDirichletPartitioner
from asyncflower.data.preprocessing import generate_datasets
from typing import Callable

def build_two_tier_dirichlet_partitions(
    dataset: str,
    num_partitions: int,
    partition_by: str, 
    group_names: list[str],
    group_fractions: list[float],
    alpha_first_level: float, 
    alpha_second_level: float, 
    use_group_size_aware_partitioning: bool,
    max_attempts: int, 
    min_num_samples: int,
    seed: int,
    save_dir: str,
    force_sort_groups: bool = True
) -> Callable:
    if force_sort_groups:
        group_idx_order = np.argsort(group_names)
        group_names = np.array(group_names)[group_idx_order].tolist()
        group_fractions = np.array(group_fractions)[group_idx_order]

    alpha1_str = str(alpha_first_level).replace('.', 'd')
    alpha2_str = str(alpha_second_level).replace('.', 'd')
    groups_str = "_".join([f"{name}{str(fraction).replace('.', 'd')}" for name, fraction in zip(group_names, group_fractions)])
    group_aware_str = "-group_size_aware" if use_group_size_aware_partitioning else ""
    setting_str_id = f"{dataset}/n{num_partitions}-2tier_dirichlet{group_aware_str}-alpha1_{alpha1_str}-alpha2_{alpha2_str}-{groups_str}/partitions-seed{seed}"
    partitions_dir = f"{save_dir}/data/{setting_str_id}"
    
    group_total_partitions = np.floor(np.array(group_fractions) * num_partitions).astype(int)
    remaining = num_partitions - group_total_partitions.sum().item()
    group_total_partitions[:remaining] += 1

    group_to_partition_ids = {}
    offset = 0
    for group_name, group_total in zip(group_names, group_total_partitions):
        group_to_partition_ids[group_name] = list(range(offset, offset + group_total))
        offset += group_total

    if not os.path.exists(partitions_dir):
        partitioner = TwoTierDirichletPartitioner(
            partition_by = partition_by,
            group_to_partition_ids = group_to_partition_ids,
            alpha_first_level = alpha_first_level,
            alpha_second_level = alpha_second_level,
            max_attempts = max_attempts,
            min_num_samples = min_num_samples,
            use_group_size_aware_partitioning = use_group_size_aware_partitioning,
            seed = seed
        )

        generate_datasets(dataset, num_partitions, partitioner, partitions_dir)

        with open(f"{partitions_dir}/groups_specification.json", "w") as f:
            json.dump(group_to_partition_ids, f)

    return partitions_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_dir", required = True, dest = "save_dir", type = str)    
    parser.add_argument("--num_partitions", required = True, dest = "num_partitions", type = int)
    parser.add_argument("--dataset", required = True, dest = "dataset", type = str)
    parser.add_argument("--partition_by", required = True, dest = "partition_by", type = str)
    parser.add_argument("--group_names", required = True, dest = "group_names", nargs = "+", type = str)
    parser.add_argument("--group_fractions", required = True, dest = "group_fractions", nargs = "+", type = float)
    parser.add_argument("--alpha1", required = True, dest = "alpha1", type = float)
    parser.add_argument("--alpha2", required = True, dest = "alpha2", type = float)
    parser.add_argument("--min_num_samples", required = True, dest = "min_num_samples", type = int)
    parser.add_argument("--max_attempts", required = True, dest = "max_attempts", type = int)
    parser.add_argument("--use_group_size_aware_partitioning", dest = "use_group_size_aware_partitioning", action = "store_true")
    parser.add_argument("--seed", required = True, dest = "seed", type = int)
    
    args = parser.parse_args()

    print("Preparing Data...")
    partitions_dir = build_two_tier_dirichlet_partitions(
        dataset = args.dataset,
        num_partitions = args.num_partitions,
        partition_by = args.partition_by, 
        group_names = args.group_names, 
        group_fractions = args.group_fractions, 
        alpha_first_level = args.alpha1, 
        alpha_second_level = args.alpha2, 
        max_attempts = args.max_attempts, 
        min_num_samples = args.min_num_samples,
        use_group_size_aware_partitioning = args.use_group_size_aware_partitioning,
        seed = args.seed,
        save_dir = args.save_dir
    )
    print(f"Partitions were stored in {partitions_dir}...")