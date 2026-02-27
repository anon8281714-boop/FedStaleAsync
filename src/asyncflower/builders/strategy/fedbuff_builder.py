from asyncflower.utils.prob_estimator import BaseParticipationProbabilityEstimator
from asyncflower.strategy import AsyncStrategy, FedBuff, UnbiasedFedBuff
from asyncflower.utils import experiments as utils
from flwr.common import Parameters
from typing import Callable, Literal

def build_fedbuff_hinge_scaling(
    concurrency: int,
    buffer_size: int, 
    hinge_a: float, 
    hinge_b: float, 
    server_learning_rate: float,
    min_available_clients: int,
    initial_parameters: Parameters,
    on_fit_config_fn: Callable,
    fit_metrics_aggregation_fn: Callable, 
    evaluate_metrics_aggregation_fn: Callable,
    evaluate_fn: Callable,
    **kwargs
) -> AsyncStrategy:
    staleness_weighting_fn = lambda staleness: utils.hinge_fn(
        value = staleness, 
        a = hinge_a, 
        b = hinge_b
    )

    strategy = FedBuff(
        concurrency = concurrency,
        buffer_size = buffer_size,
        learning_rate = server_learning_rate,
        staleness_scaling_fn = staleness_weighting_fn,
        min_available_clients = min_available_clients,
        initial_parameters = initial_parameters,
        evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
        evaluate_fn = evaluate_fn,
        on_fit_config_fn = on_fit_config_fn
    )

    return strategy

def build_unbiased_fedbuff_hinge_scaling(
    concurrency: int,
    aggregation_scheme: Literal["buffer_size", "round_duration"],
    aggregation_value: int|float, 
    use_relative_data_size_weights: bool,
    participation_prob_estimator: BaseParticipationProbabilityEstimator,
    hinge_a: float, 
    hinge_b: float, 
    server_learning_rate: float,
    min_available_clients: int,
    initial_parameters: Parameters,
    on_fit_config_fn: Callable,
    fit_metrics_aggregation_fn: Callable, 
    evaluate_metrics_aggregation_fn: Callable,
    evaluate_fn: Callable,
    **kwargs
) -> AsyncStrategy:
    def staleness_weighting_fn(staleness, cid = None):
        scaling = utils.hinge_fn(
            value = staleness, 
            a = hinge_a, 
            b = hinge_b
        )
        return scaling

    strategy = UnbiasedFedBuff(
        concurrency = concurrency,
        aggregation_scheme = aggregation_scheme,
        aggregation_value = aggregation_value,
        learning_rate = server_learning_rate,
        use_relative_data_size_weights = use_relative_data_size_weights,
        participation_prob_estimator = participation_prob_estimator,
        staleness_scaling_fn = staleness_weighting_fn,
        min_available_clients = min_available_clients,
        initial_parameters = initial_parameters,
        evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
        evaluate_fn = evaluate_fn,
        on_fit_config_fn = on_fit_config_fn
    )

    return strategy