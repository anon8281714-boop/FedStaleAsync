from flwr.common import Parameters
from typing import Literal, Callable
from asyncflower.strategy import AsyncStrategy, CA2FL, UnbiasedCA2FL
from asyncflower.utils.prob_estimator import BaseParticipationProbabilityEstimator

def build_ca2fl(
    concurrency: int,
    buffer_size: int,
    server_learning_rate: float,
    min_available_clients: int,
    full_participation_memory_initialization: bool,
    initial_parameters: Parameters,
    on_fit_config_fn: Callable,
    fit_metrics_aggregation_fn: Callable, 
    evaluate_metrics_aggregation_fn: Callable,
    evaluate_fn: Callable, 
    **kwargs
) -> AsyncStrategy:
    
    strategy = CA2FL(
        concurrency = concurrency,
        buffer_size = buffer_size,
        learning_rate = server_learning_rate,
        full_participation_memory_initialization = full_participation_memory_initialization,
        min_available_clients = min_available_clients,
        initial_parameters = initial_parameters,
        evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
        evaluate_fn = evaluate_fn,
        on_fit_config_fn = on_fit_config_fn
    )

    return strategy

def build_unbiased_ca2fl(
    concurrency: int,
    aggregation_value: int|float,
    aggregation_scheme: Literal["buffer_size", "round_duration"],
    server_learning_rate: float,
    min_available_clients: int,
    use_relative_data_size_weights: bool,
    participation_prob_estimator: BaseParticipationProbabilityEstimator,
    full_participation_memory_initialization: bool,
    initial_parameters: Parameters,
    on_fit_config_fn: Callable,
    fit_metrics_aggregation_fn: Callable, 
    evaluate_metrics_aggregation_fn: Callable,
    evaluate_fn: Callable, 
    **kwargs
) -> AsyncStrategy:
    
    strategy = UnbiasedCA2FL(
        concurrency = concurrency, 
        aggregation_value = aggregation_value, 
        aggregation_scheme = aggregation_scheme, 
        learning_rate = server_learning_rate, 
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