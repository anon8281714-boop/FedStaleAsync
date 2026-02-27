import numpy as np
from datasets import Dataset
from flwr_datasets.partitioner import Partitioner

class TwoTierDirichletPartitioner(Partitioner):
    def __init__(
        self, 
        partition_by: str, 
        group_to_partition_ids: dict, 
        alpha_first_level: float = 1, 
        alpha_second_level: float = 0.1, 
        max_attempts: int = 5, 
        min_num_samples: int = None,
        use_group_size_aware_partitioning = False,
        seed: int = 0
    ):
        self._num_partitions_per_group = {
            group_name: len(partition_ids) for group_name, partition_ids in group_to_partition_ids.items()
        }
        self._num_partitions = sum(self._num_partitions_per_group.values())
        self._num_groups = len(group_to_partition_ids)
        self._group_to_partition_ids = group_to_partition_ids

        self.partition_by = partition_by
        self.alpha_first_level = alpha_first_level
        self.alpha_second_level = alpha_second_level
        self.max_attempts = max_attempts
        self.min_num_samples = min_num_samples
        self.seed = seed
        self.use_group_size_aware_partitioning = use_group_size_aware_partitioning
        self._random_generator = np.random.default_rng(seed)
        self._idx_list_by_partition = None
        super().__init__()

    @property
    def dataset(self) -> Dataset:
        return self._dataset
    
    @dataset.setter
    def dataset(self, value: Dataset) -> None:
        if self._dataset is not None:
            raise ValueError(
                "The dataset should be assigned only once to the partitioner."
                "This operation might also wipe out the saved references to the "
                "created partitions (in case the partitioning scheme needs to create "
                "the full partitioning also in order to return a single partition)."
            )
        if not isinstance(value, Dataset):
            raise TypeError(
                f"The dataset object you want to assign to the partitioner should be "
                f"of type `datasets.Dataset` but given {type(value)}."
            )
        self._dataset = value
        self._idx_list_by_partition = None

    def _determine_partition_id_to_indices_if_needed(self) -> None:
        if self._idx_list_by_partition is not None:
            return

        rng = self._random_generator

        label_to_indices = {}
        for label in set(self._dataset[self.partition_by]):
            idxs = np.where(np.array(self._dataset[self.partition_by]) == label)[0]
            rng.shuffle(idxs)
            label_to_indices[label] = idxs.tolist()

        for _ in range(self.max_attempts):
            idx_list_by_partition = {partition_id: [] for partition_id in range(self._num_partitions)}

            for label, indices in label_to_indices.items():
                n_samples = len(indices)

                num_partitions_per_group = np.array(list(self._num_partitions_per_group.values()))

                # ---------- First level: between groups ----------
                alphas_first_level = np.array([self.alpha_first_level] * self._num_groups)
                if self.use_group_size_aware_partitioning:
                    alphas_first_level = (num_partitions_per_group / self._num_partitions) * alphas_first_level

                group_props = rng.dirichlet(alphas_first_level)
                group_sizes = np.floor(group_props * n_samples).astype(int)

                # Fix rounding
                while group_sizes.sum() < n_samples:
                    group_sizes[np.argmax(group_props)] += 1

                offset = 0
                for group_size, (group, partition_ids) in zip(
                    group_sizes, self._group_to_partition_ids.items()
                ):
                    if group_size == 0:
                        continue

                    group_indices = indices[offset: offset + group_size]
                    offset += group_size

                    # ---------- Second level: inside group ----------
                    n_partitions = len(partition_ids)
                    partition_props = rng.dirichlet(
                        [self.alpha_second_level] * n_partitions
                    )
                    partition_sizes = np.floor(partition_props * group_size).astype(int)

                    while partition_sizes.sum() < group_size:
                        partition_sizes[np.argmax(partition_props)] += 1

                    partition_offset = 0
                    for pid, psize in zip(partition_ids, partition_sizes):
                        if psize > 0:
                            idx_list_by_partition[pid].extend(
                                group_indices[partition_offset: partition_offset + psize]
                            )
                            partition_offset += psize

            # ---------- Check minimum samples ----------
            if self.min_num_samples is not None:
                min_samples = min(len(v) for v in idx_list_by_partition.values())
                if min_samples >= self.min_num_samples:
                    self._idx_list_by_partition = idx_list_by_partition
                    return

        raise RuntimeError(
            f"Minimum number of samples {self.min_num_samples} "
            f"could not be satisfied in {self.max_attempts} attempts."
        )


    def load_partition(self, partition_id: int) -> Dataset:
        assert self.is_dataset_assigned()
        self._determine_partition_id_to_indices_if_needed()
        idx_list = self._idx_list_by_partition[partition_id]
        return self._dataset.select(idx_list)
    
    @property
    def num_partitions(self):
        return self._num_partitions