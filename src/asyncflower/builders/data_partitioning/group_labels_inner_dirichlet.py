import os
import json
import argparse
import numpy as np
from asyncflower.partitioner.group_labels_inner_dirichlet import GroupLabelsInnerDirichletPartitioner
from asyncflower.data.preprocessing import generate_datasets
from typing import Callable

def build_group_labels_inner_dirichlet_partitions(
    dataset: str,
    num_partitions: int,
    partition_by: str, 
    group_names: list[str],
    group_fractions: list[float],
    group_label_lists: list[list[float]],
    alpha: float, 
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
        group_label_lists = [group_label_lists[idx.item()] for idx in group_idx_order]

    alpha_str = str(alpha).replace('.', 'd')
    group_label_lists_str = ['_'.join([str(label) for label in label_list]) for label_list in group_label_lists]
    groups_str = "_".join([
        f"{name}{str(fraction).replace('.', 'd')}_{label_list_str}" for name, fraction, label_list_str in zip(group_names, group_fractions, group_label_lists_str)
    ])
    setting_str_id = f"{dataset}/n{num_partitions}-group_labels_inner_dirichlet-alpha_{alpha_str}-{groups_str}/partitions-seed{seed}"
    partitions_dir = f"{save_dir}/data/{setting_str_id}"
    
    group_total_partitions = np.floor(np.array(group_fractions) * num_partitions).astype(int)
    remaining = num_partitions - group_total_partitions.sum().item()
    group_total_partitions[:remaining] += 1

    group_to_partition_ids = {}
    labels_per_partition = {}

    offset = 0
    for group_name, group_total, group_labels in zip(group_names, group_total_partitions, group_label_lists):
        group_cids = list(range(offset, offset + group_total))
        group_to_partition_ids[group_name] = group_cids
        labels_per_partition.update({cid: group_labels for cid in group_cids})
        offset += group_total

    if not os.path.exists(partitions_dir):
        partitioner = GroupLabelsInnerDirichletPartitioner(
            partition_by = partition_by, 
            labels_per_partition = labels_per_partition,
            alpha = alpha,
            max_attempts = max_attempts, 
            min_num_samples = min_num_samples,
            seed = 0
        )

        generate_datasets(dataset, num_partitions, partitioner, partitions_dir)

        with open(f"{partitions_dir}/groups_specification.json", "w") as f:
            json.dump(group_to_partition_ids, f)

        with open(f"{partitions_dir}/cid_labels.json", "w") as f:
            json.dump(labels_per_partition, f)

    return partitions_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_dir", required = True, dest = "save_dir", type = str)    
    parser.add_argument("--num_partitions", required = True, dest = "num_partitions", type = int)
    parser.add_argument("--dataset", required = True, dest = "dataset", type = str)
    parser.add_argument("--partition_by", required = True, dest = "partition_by", type = str)
    parser.add_argument("--group_names", required = True, dest = "group_names", nargs = "+", type = str)
    parser.add_argument("--group_fractions", required = True, dest = "group_fractions", nargs = "+", type = float)
    parser.add_argument("--group_label_lists", required = True, dest = "group_label_lists", type = json.loads)
    parser.add_argument("--alpha", required = True, dest = "alpha", type = float)
    parser.add_argument("--min_num_samples", required = True, dest = "min_num_samples", type = int)
    parser.add_argument("--max_attempts", required = True, dest = "max_attempts", type = int)
    parser.add_argument("--seed", required = True, dest = "seed", type = int)

    args = parser.parse_args()

    print("Preparing Data...")
    partitions_dir = build_group_labels_inner_dirichlet_partitions(
        dataset = args.dataset,
        num_partitions = args.num_partitions,
        partition_by = args.partition_by, 
        group_names = args.group_names, 
        group_fractions = args.group_fractions, 
        group_label_lists = args.group_label_lists, 
        alpha = args.alpha,
        max_attempts = args.max_attempts, 
        min_num_samples = args.min_num_samples,
        seed = args.seed,
        save_dir = args.save_dir
    )
    print(f"Partitions were stored in {partitions_dir}...")