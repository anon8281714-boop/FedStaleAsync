import numpy as np
from asyncflower.utils.prob_estimator import ClientGroupBasedParticipationProbabilityEstimator

def build_client_group_based_estimator(
    num_clients: int, 
    group_names: list[str],
    group_fractions: list[float],
    min_allowed_prob: float | None = None,
    force_sort_groups: bool = True
):
    if force_sort_groups:
        group_idx_order = np.argsort(group_names)
        group_names = np.array(group_names)[group_idx_order].tolist()
        group_fractions = np.array(group_fractions)[group_idx_order]

    group_total_clients = np.floor(np.array(group_fractions) * num_clients).astype(int)
    remaining = num_clients - group_total_clients.sum().item()
    group_total_clients[:remaining] += 1

    cid_to_group = {}
    offset = 0
    for group_name, group_total in zip(group_names, group_total_clients):
        cid_to_group.update({cid: group_name for cid in list(range(offset, offset + group_total))})
        offset += group_total

    return ClientGroupBasedParticipationProbabilityEstimator(
        cid_to_group = cid_to_group, 
        min_allowed_prob = min_allowed_prob
    )