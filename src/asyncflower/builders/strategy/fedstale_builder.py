from flwr.common import Parameters
from typing import Callable
from asyncflower.strategy import AsyncStrategy, FedStale
from asyncflower.utils.prob_estimator import BaseParticipationProbabilityEstimator

def build_fedstale(
    num_clients_fit: int,
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
    strategy = FedStale(
        num_clients_fit = num_clients_fit,
        learning_rate = server_learning_rate,
        beta = beta,
        use_relative_data_size_weights = use_relative_data_size_weights,
        participation_prob_estimator = participation_prob_estimator,
        full_participation_memory_initialization = full_participation_memory_initialization,
        min_available_clients = min_available_clients,
        initial_parameters = initial_parameters,
        evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
        evaluate_fn = evaluate_fn,
        on_fit_config_fn = on_fit_config_fn
    )

    return strategy