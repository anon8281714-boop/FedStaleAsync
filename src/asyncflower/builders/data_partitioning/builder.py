from typing import Literal
from asyncflower.builders.data_partitioning.dirichlet import build_dirichlet_partitions
from asyncflower.builders.data_partitioning.two_tier_dirichlet import build_two_tier_dirichlet_partitions
from asyncflower.builders.data_partitioning.group_labels_inner_dirichlet import build_group_labels_inner_dirichlet_partitions

PartitioningScheme = Literal["dirichlet", "two_tier_dirichlet", "group_labels_inner_dirichlet"]

def build_data_partitioning(
    scheme: PartitioningScheme = "dirichlet",
    **kwargs
) -> str:
    if scheme == "dirichlet":
        return build_dirichlet_partitions(**kwargs)

    elif scheme == "two_tier_dirichlet":
        return build_two_tier_dirichlet_partitions(**kwargs)
    
    elif scheme == "group_labels_inner_dirichlet":
        return build_group_labels_inner_dirichlet_partitions(**kwargs)
    
    raise ValueError(f"{scheme} is not a valid scheme.")