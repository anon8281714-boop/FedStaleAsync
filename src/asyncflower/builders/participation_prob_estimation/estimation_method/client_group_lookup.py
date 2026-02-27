import numpy as np
from asyncflower.utils.prob_estimator import LookupParticipationProbabilityEstimator

def build_client_group_lookup_estimator(
    num_clients: int, 
    group_names: list[str],
    group_fractions: list[float],
    group_probs: list[float],
    min_allowed_prob: float | None = None,
    force_sort_groups: bool = True
):
    if force_sort_groups:
        group_idx_order = np.argsort(group_names)
        group_names = np.array(group_names)[group_idx_order].tolist()
        group_fractions = np.array(group_fractions)[group_idx_order]
        group_probs = np.array(group_probs)[group_idx_order].tolist()

    group_total_clients = np.floor(np.array(group_fractions) * num_clients).astype(int)
    remaining = num_clients - group_total_clients.sum().item()
    group_total_clients[:remaining] += 1

    cid_probabilities = {}
    offset = 0
    for group_total, group_prob in zip(group_total_clients, group_probs):
        cid_probabilities.update({cid: group_prob for cid in list(range(offset, offset + group_total))})
        offset += group_total

    return LookupParticipationProbabilityEstimator(
        cid_probabilities = cid_probabilities,
        min_allowed_prob = min_allowed_prob
    )