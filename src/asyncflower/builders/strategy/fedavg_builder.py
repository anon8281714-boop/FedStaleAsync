from asyncflower.utils.prob_estimator import BaseParticipationProbabilityEstimator
from asyncflower.strategy import AsyncStrategy, FedAvg, UnbiasedFedAvg
from flwr.common import Parameters
from typing import Callable

def build_fedavg(
    fraction_fit: int,
    min_available_clients: int,
    initial_parameters: Parameters,
    on_fit_config_fn: Callable,
    fit_metrics_aggregation_fn: Callable, 
    evaluate_metrics_aggregation_fn: Callable,
    evaluate_fn: Callable,
    **kwargs
) -> AsyncStrategy:
    strategy = FedAvg(
        fraction_fit = fraction_fit,
        min_available_clients = min_available_clients,
        initial_parameters = initial_parameters,
        evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
        evaluate_fn = evaluate_fn,
        on_fit_config_fn = on_fit_config_fn
    )

    return strategy

def build_unbiased_fedavg(
    num_fit_clients: int,
    use_relative_data_size_weights: bool,
    participation_prob_estimator: BaseParticipationProbabilityEstimator, 
    min_available_clients: int,
    initial_parameters: Parameters,
    on_fit_config_fn: Callable,
    fit_metrics_aggregation_fn: Callable, 
    evaluate_metrics_aggregation_fn: Callable,
    evaluate_fn: Callable,
    **kwargs
) -> AsyncStrategy:
    strategy = BaseParticipationProbabilityEstimator(
        num_fit_clients = num_fit_clients,
        use_relative_data_size_weights = use_relative_data_size_weights,
        participation_prob_estimator = participation_prob_estimator,
        min_available_clients = min_available_clients,
        initial_parameters = initial_parameters,
        evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
        evaluate_fn = evaluate_fn,
        on_fit_config_fn = on_fit_config_fn
    )

    return strategy