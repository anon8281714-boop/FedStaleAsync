import numpy as np
import pandas as pd
from datasets import Dataset
from flwr_datasets.partitioner import Partitioner

class GroupLabelsInnerDirichletPartitioner(Partitioner):
    def __init__(
        self, 
        partition_by: str, 
        labels_per_partition: dict, 
        alpha: float = 0.1, 
        max_attempts: int = 5, 
        min_num_samples: int = None,
        seed: int = 0
    ):
        self._num_partitions = len(labels_per_partition)
        self.partition_by = partition_by
        self.labels_per_partition = labels_per_partition
        self.alpha = alpha
        self.max_attempts = max_attempts
        self.min_num_samples = min_num_samples
        self.seed = seed
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
            return None 
        
        idx_to_label = {idx: label for idx, label in enumerate(self._dataset[self.partition_by])}
        label_to_idx_list = pd.DataFrame(
            data = idx_to_label.items(), columns = ["idx", "label"]
        ).groupby("label").agg(lambda group: self._random_generator.choice(list(group), size = group.shape[0], replace = False).tolist())["idx"].to_dict()

        partitions_per_label = pd.DataFrame(
            self.labels_per_partition.items(), columns = ["partition_id_list", "label_set"]
        ).explode("label_set")
        partitions_per_label = partitions_per_label.groupby("label_set").agg(list).to_dict(orient = "index")

        for i in range(self.max_attempts):
            partitioning_df = []

            for label in partitions_per_label:
                total_label_samples = len(label_to_idx_list[label])
                num_clients = len(partitions_per_label[label]["partition_id_list"])
                label_sample_idx_list = label_to_idx_list[label]
                
                proportions = self._random_generator.dirichlet(alpha = [self.alpha] * num_clients, size = 1)[0]
                num_samples = (proportions * total_label_samples).round().astype(int)
                partition_id_list = partitions_per_label[label]["partition_id_list"]

                df = pd.DataFrame(data = {"partition_id": partition_id_list, "proportion": proportions, "num_samples": num_samples})
                df["label"] = label

                df["end_sample_idx"] = df["num_samples"].cumsum().astype(int)
                df["start_sample_idx"] = df["end_sample_idx"].shift(1).fillna(0).astype(int)
                df["assigned_sample_idx_list"] = df.apply(lambda row: label_sample_idx_list[int(row.start_sample_idx) : int(row.end_sample_idx)], axis = 1)

                partitioning_df += [df]

            partitioning_df = pd.concat(partitioning_df)
            
            idx_list_by_partition_df = partitioning_df[
                ["partition_id", "assigned_sample_idx_list"]
            ].groupby("partition_id").agg(lambda group: np.concat(list(group)))
            idx_list_by_partition_df["num_samples"] = idx_list_by_partition_df["assigned_sample_idx_list"].apply(len)
            min_num_samples = idx_list_by_partition_df.num_samples.min()
            
            if min_num_samples >= self.min_num_samples:
                self._idx_list_by_partition = idx_list_by_partition_df["assigned_sample_idx_list"].to_dict()
                break
        
        if not self._idx_list_by_partition:
            raise Exception(
                f"Mininum number of samples {self.min_num_samples} could not be satisfied in all of the {self.max_attempts} attempts."
            )

    def load_partition(self, partition_id: int) -> Dataset:
        assert self.is_dataset_assigned()
        self._determine_partition_id_to_indices_if_needed()
        idx_list = self._idx_list_by_partition[partition_id]
        return self._dataset.select(idx_list)
    
    @property
    def num_partitions(self):
        return self._num_partitions