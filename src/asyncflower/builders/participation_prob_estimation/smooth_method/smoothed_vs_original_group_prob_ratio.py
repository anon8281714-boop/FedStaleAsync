import numpy as np

def build_smoothed_vs_original_group_prob_ratio_ratio(
    num_clients: int, 
    group_names: list[str],
    group_fractions: list[float],
    group_prob_ratio: list[float],
    force_sort_groups: bool = True
):
    if force_sort_groups:
        group_idx_order = np.argsort(group_names)
        group_names = np.array(group_names)[group_idx_order].tolist()
        group_fractions = np.array(group_fractions)[group_idx_order]
        group_prob_ratio = np.array(group_prob_ratio)[group_idx_order].tolist()

    group_total_clients = np.floor(np.array(group_fractions) * num_clients).astype(int)
    remaining = num_clients - group_total_clients.sum().item()
    group_total_clients[:remaining] += 1

    cid_prob_ratio = {}
    offset = 0
    for group_total, group_prob_ratio in zip(group_total_clients, group_prob_ratio):
        assert group_prob_ratio >= 1, "Ratio should be greater than 1."
        cid_prob_ratio.update({cid: group_prob_ratio for cid in list(range(offset, offset + group_total))})
        offset += group_total

    smooth_term_fn = lambda cid, prob: min((cid_prob_ratio[int(cid)] - 1) * prob, 1 - prob)

    return smooth_term_fn