from flwr.common import Parameters
from typing import Callable, Literal
from asyncflower.strategy import AsyncStrategy, ACE, BufferedACE

def build_ace(
    concurrency: int,
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
    
    strategy = ACE(
        concurrency = concurrency,
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

def build_buffered_ace(
    concurrency: int,
    server_learning_rate: float,
    min_available_clients: int,
    aggregation_scheme: Literal["buffer_size", "round_duration"], 
    aggregation_value: int|float, 
    use_relative_data_size_weights: bool,
    full_participation_memory_initialization: bool,
    initial_parameters: Parameters,
    on_fit_config_fn: Callable,
    fit_metrics_aggregation_fn: Callable, 
    evaluate_metrics_aggregation_fn: Callable,
    evaluate_fn: Callable, 
    **kwargs
) -> AsyncStrategy:
    
    strategy = BufferedACE(
        concurrency = concurrency,
        learning_rate = server_learning_rate,
        aggregation_scheme = aggregation_scheme,
        aggregation_value = aggregation_value,
        use_relative_data_size_weights = use_relative_data_size_weights,
        full_participation_memory_initialization = full_participation_memory_initialization,
        min_available_clients = min_available_clients,
        initial_parameters = initial_parameters,
        evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
        evaluate_fn = evaluate_fn,
        on_fit_config_fn = on_fit_config_fn
    )

    return strategy