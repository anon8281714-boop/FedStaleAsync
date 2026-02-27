import numpy as np
from asyncflower.utils import experiments as utils
from flwr.common import Parameters
from typing import Literal, Callable
from asyncflower.strategy import AsyncStrategy, FedStaleAsync
from asyncflower.utils.prob_estimator import BaseParticipationProbabilityEstimator

def build_fedstaleasync_hinge_scaling(
    concurrency: int,
    aggregation_scheme: Literal["buffer_size", "round_duration"],
    aggregation_value: int|float, 
    participating_hinge_a: float, 
    participating_hinge_b: float, 
    stored_hinge_a: float, 
    stored_hinge_b: float, 
    use_relative_data_size_weights: bool,
    use_personalized_hinge_b_participation_prob: bool,
    participation_prob_estimator: BaseParticipationProbabilityEstimator,
    full_participation_memory_initialization: bool,
    server_learning_rate: float,
    min_available_clients: int,
    initial_parameters: Parameters,
    on_fit_config_fn: Callable,
    fit_metrics_aggregation_fn: Callable, 
    evaluate_metrics_aggregation_fn: Callable,
    evaluate_fn: Callable, 
    **kwargs
) -> AsyncStrategy:
    def alpha_staleness_weighting_fn(staleness, cid = None): 
        a = participating_hinge_a
        b = participating_hinge_b
        if use_personalized_hinge_b_participation_prob:
            # Hinge b is computed relative to the expected number of rounds without the participation of client i (1 / p_i)
            cid_participation_prob = participation_prob_estimator.get_participation_probability(cid)
            b = b / cid_participation_prob
        
        scaling = utils.hinge_fn(value = staleness, a = a, b = b)

        return scaling
    
    def beta_staleness_weighting_fn(staleness, cid = None): 
        a = stored_hinge_a
        b = stored_hinge_b
        if use_personalized_hinge_b_participation_prob:
            # Hinge b is computed relative to the expected number of rounds without the participation of client i (1 / p_i)
            cid_participation_prob = participation_prob_estimator.get_participation_probability(cid)
            b = b / cid_participation_prob
        
        scaling = utils.hinge_fn(value = staleness, a = a, b = b)

        return scaling

    strategy = FedStaleAsync(
        concurrency = concurrency,
        aggregation_scheme = aggregation_scheme,
        aggregation_value = aggregation_value,
        alpha_staleness_weighting_fn = alpha_staleness_weighting_fn,
        beta_staleness_weighting_fn = beta_staleness_weighting_fn,
        use_relative_data_size_weights = use_relative_data_size_weights,
        participation_prob_estimator = participation_prob_estimator,
        full_participation_memory_initialization = full_participation_memory_initialization,
        learning_rate = server_learning_rate,
        min_available_clients = min_available_clients,
        initial_parameters = initial_parameters,
        evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
        evaluate_fn = evaluate_fn,
        on_fit_config_fn = on_fit_config_fn
    )

    return strategy

def build_fedstaleasync_constant_scaling(
    concurrency: int,
    aggregation_scheme: Literal["buffer_size", "round_duration"],
    aggregation_value: int|float, 
    alpha: float, 
    beta: float, 
    use_relative_data_size_weights: bool,
    participation_prob_estimator: BaseParticipationProbabilityEstimator,
    full_participation_memory_initialization: bool,
    server_learning_rate: float,
    min_available_clients: int,
    initial_parameters: Parameters,
    on_fit_config_fn: Callable,
    fit_metrics_aggregation_fn: Callable, 
    evaluate_metrics_aggregation_fn: Callable,
    evaluate_fn: Callable, 
    **kwargs
) -> AsyncStrategy: 
    def get_staleness_weighting_fn(scaling_value: float):
        def staleness_weighting_fn(staleness, cid = None): 
            return scaling_value
        return staleness_weighting_fn

    strategy = FedStaleAsync(
        concurrency = concurrency,
        aggregation_scheme = aggregation_scheme,
        aggregation_value = aggregation_value,
        alpha_staleness_weighting_fn = get_staleness_weighting_fn(alpha),
        beta_staleness_weighting_fn = get_staleness_weighting_fn(beta),
        use_relative_data_size_weights = use_relative_data_size_weights,
        participation_prob_estimator = participation_prob_estimator,
        full_participation_memory_initialization = full_participation_memory_initialization,
        learning_rate = server_learning_rate,
        min_available_clients = min_available_clients,
        initial_parameters = initial_parameters,
        evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
        evaluate_fn = evaluate_fn,
        on_fit_config_fn = on_fit_config_fn
    )

    return strategy

def build_fedstaleasync_group_constant_scaling(
    num_clients: int, 
    concurrency: int,
    aggregation_scheme: Literal["buffer_size", "round_duration"],
    aggregation_value: int|float, 
    group_names: list[str],
    group_fractions: list[float],
    group_alpha_values: list[float],
    group_beta_values: list[float],
    use_relative_data_size_weights: bool,
    participation_prob_estimator: BaseParticipationProbabilityEstimator,
    full_participation_memory_initialization: bool,
    server_learning_rate: float,
    min_available_clients: int,
    initial_parameters: Parameters,
    on_fit_config_fn: Callable,
    fit_metrics_aggregation_fn: Callable, 
    evaluate_metrics_aggregation_fn: Callable,
    evaluate_fn: Callable, 
    force_sort_groups: bool = True,
    **kwargs
) -> AsyncStrategy: 
    if force_sort_groups:
        group_idx_order = np.argsort(group_names)
        group_names = np.array(group_names)[group_idx_order].tolist()
        group_alpha_values = np.array(group_alpha_values)[group_idx_order]
        group_beta_values = np.array(group_beta_values)[group_idx_order]

    group_total_clients = np.floor(np.array(group_fractions) * num_clients).astype(int)
    remaining = num_clients - group_total_clients.sum().item()
    group_total_clients[:remaining] += 1

    cid_to_alpha = {}
    cid_to_beta = {}

    offset = 0
    for group_total, group_alpha, group_beta in zip(
        group_total_clients, group_alpha_values, group_beta_values
    ):
        cid_list = list(range(offset, offset + group_total))
        cid_to_alpha.update({cid: group_alpha for cid in cid_list})
        cid_to_beta.update({cid: group_beta for cid in cid_list})
        offset += group_total

    def alpha_staleness_weighting_fn(staleness, cid = None): 
        cid = int(cid)
        scaling = cid_to_alpha[cid]
        return scaling
    
    def beta_staleness_weighting_fn(staleness, cid = None): 
        cid = int(cid)
        scaling = cid_to_beta[cid]
        return scaling

    strategy = FedStaleAsync(
        concurrency = concurrency,
        aggregation_scheme = aggregation_scheme,
        aggregation_value = aggregation_value,
        alpha_staleness_weighting_fn = alpha_staleness_weighting_fn,
        beta_staleness_weighting_fn = beta_staleness_weighting_fn,
        use_relative_data_size_weights = use_relative_data_size_weights,
        participation_prob_estimator = participation_prob_estimator,
        full_participation_memory_initialization = full_participation_memory_initialization,
        learning_rate = server_learning_rate,
        min_available_clients = min_available_clients,
        initial_parameters = initial_parameters,
        evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
        evaluate_fn = evaluate_fn,
        on_fit_config_fn = on_fit_config_fn
    )

    return strategy

# def build_fedstaleasync_hinge_scaling(
#     concurrency: int,
#     aggregation_scheme: Literal["buffer_size", "round_duration"],
#     aggregation_value: int|float, 
#     participating_hinge_a: float, 
#     participating_hinge_b: float, 
#     stored_hinge_a: float, 
#     stored_hinge_b: float, 
#     use_relative_data_size_weights: bool,
#     participation_prob_estimator: BaseParticipationProbabilityEstimator,
#     full_participation_memory_initialization: bool,
#     server_learning_rate: float,
#     min_available_clients: int,
#     initial_parameters: Parameters,
#     on_fit_config_fn: Callable,
#     fit_metrics_aggregation_fn: Callable, 
#     evaluate_metrics_aggregation_fn: Callable,
#     evaluate_fn: Callable, 
#     **kwargs
# ) -> AsyncStrategy:
#     alpha_staleness_weighting_fn = lambda staleness, cid: utils.hinge_fn(
#         value = staleness, 
#         a = participating_hinge_a, 
#         b = participating_hinge_b
#     )

#     beta_staleness_weighting_fn = lambda staleness, cid: utils.hinge_fn(
#         value = staleness, 
#         a = stored_hinge_a, 
#         b = stored_hinge_b
#     )

#     strategy = FedStaleAsync(
#         concurrency = concurrency,
#         aggregation_scheme = aggregation_scheme,
#         aggregation_value = aggregation_value,
#         alpha_staleness_weighting_fn = alpha_staleness_weighting_fn,
#         beta_staleness_weighting_fn = beta_staleness_weighting_fn,
#         use_relative_data_size_weights = use_relative_data_size_weights,
#         participation_prob_estimator = participation_prob_estimator,
#         full_participation_memory_initialization = full_participation_memory_initialization,
#         learning_rate = server_learning_rate,
#         min_available_clients = min_available_clients,
#         initial_parameters = initial_parameters,
#         evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn,
#         fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
#         evaluate_fn = evaluate_fn,
#         on_fit_config_fn = on_fit_config_fn
#     )

#     return strategy