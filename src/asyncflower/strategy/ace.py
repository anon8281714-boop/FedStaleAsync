from asyncflower.strategy.fedstaleasync import FedStaleAsync
from asyncflower.utils.prob_estimator import MaximumParticipationProbabilityEstimator
from flwr.common.typing import Scalar, Parameters, NDArrays
from typing import Optional, Dict, Callable, List, Literal

class ACE(FedStaleAsync):
    # ACE's paper: https://arxiv.org/pdf/2511.19066#page=63.15
    # The method is characterized by:
    #   - Buffer size of 1
    #   - No scaling weights
    #   - No adjustments due to participation heterogeneity
    #   - No use of relative dataset sizes
    #   - One single local epoch (this helps to eliminate the bias error when all historical updates are used in the aggregation)

    def __init__(
        self, 
        concurrency: int,
        learning_rate: float,
        full_participation_memory_initialization: bool = False, 
        fraction_eval: float = 1.0,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        evaluate_fn: Optional[Callable[[int, NDArrays], Optional[tuple[float, Dict[str, Scalar]]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable[[List[tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]] = None,
        fit_metrics_aggregation_fn: Optional[Callable[[List[tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]] = None,
    ):
        super().__init__(
            concurrency = concurrency, 
            learning_rate = learning_rate, 
            alpha_staleness_weighting_fn = lambda staleness: 1, 
            beta_staleness_weighting_fn = lambda staleness: 1, 
            use_relative_data_size_weights = False,
            aggregation_value = 1, 
            aggregation_scheme = "buffer_size", 
            participation_prob_estimator = MaximumParticipationProbabilityEstimator(), 
            full_participation_memory_initialization = full_participation_memory_initialization, 
            fraction_eval = fraction_eval, 
            min_evaluate_clients = min_evaluate_clients, 
            min_available_clients = min_available_clients, 
            initial_parameters = initial_parameters, 
            evaluate_fn = evaluate_fn, 
            on_fit_config_fn = on_fit_config_fn, 
            accept_failures = accept_failures, 
            on_evaluate_config_fn = on_evaluate_config_fn, 
            evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn, 
            fit_metrics_aggregation_fn = fit_metrics_aggregation_fn, 
        )

        if self.on_fit_config_fn:
            self._last_on_fit_config_fn = self.on_fit_config_fn
            self.on_fit_config_fn = lambda server_round: {
                **self._last_on_fit_config_fn(server_round), 
                "iteration_mode": "minibatch", 
                "num_iterations": 1,
            }

class BufferedACE(FedStaleAsync):
    # Adapted from the ACE's paper: https://arxiv.org/pdf/2511.19066#page=63.15
    # The original method is characterized by:
    #   - Buffer size of 1
    #   - No scaling weights
    #   - No adjustments due to participation heterogeneity
    #   - No use of relative dataset sizes
    #   - One single local epoch (this helps to eliminate the bias error when all historical updates are used in the aggregation)
    # This version implements the following changes:
    #   - Arbitrary buffer size
    #   - Allow usage of relative dataset sizes
    #   - Arbitrary number of local epochs or minibatches

    def __init__(
        self, 
        concurrency: int,
        learning_rate: float,
        aggregation_scheme: Literal["buffer_size", "round_duration"] = "buffer_size", 
        aggregation_value: int|float = 1, 
        use_relative_data_size_weights: bool = False,
        full_participation_memory_initialization: bool = False, 
        fraction_eval: float = 1.0,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        evaluate_fn: Optional[Callable[[int, NDArrays], Optional[tuple[float, Dict[str, Scalar]]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable[[List[tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]] = None,
        fit_metrics_aggregation_fn: Optional[Callable[[List[tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]] = None,
    ):
        super().__init__(
            concurrency = concurrency, 
            learning_rate = learning_rate, 
            alpha_staleness_weighting_fn = lambda staleness, cid: 1, 
            beta_staleness_weighting_fn = lambda staleness, cid: 1, 
            use_relative_data_size_weights = use_relative_data_size_weights,
            aggregation_value = aggregation_value, 
            aggregation_scheme = aggregation_scheme, 
            participation_prob_estimator = MaximumParticipationProbabilityEstimator(), 
            full_participation_memory_initialization = full_participation_memory_initialization, 
            fraction_eval = fraction_eval, 
            min_evaluate_clients = min_evaluate_clients, 
            min_available_clients = min_available_clients, 
            initial_parameters = initial_parameters, 
            evaluate_fn = evaluate_fn, 
            on_fit_config_fn = on_fit_config_fn, 
            accept_failures = accept_failures, 
            on_evaluate_config_fn = on_evaluate_config_fn, 
            evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn, 
            fit_metrics_aggregation_fn = fit_metrics_aggregation_fn, 
        )