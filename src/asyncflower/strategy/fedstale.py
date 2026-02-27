from asyncflower.strategy.fedstaleasync import FedStaleAsync
from asyncflower.utils.prob_estimator import BaseParticipationProbabilityEstimator
from flwr.common.typing import Scalar, Parameters, NDArrays
from typing import Optional, Dict, Callable, List

class FedStale(FedStaleAsync):
    def __init__(
        self, 
        num_clients_fit: int,
        learning_rate: float,
        beta: float,
        use_relative_data_size_weights: bool = True,
        participation_prob_estimator: BaseParticipationProbabilityEstimator = None, 
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
            concurrency = num_clients_fit, 
            learning_rate = learning_rate, 
            alpha_staleness_weighting_fn = lambda staleness, cid: 1, 
            beta_staleness_weighting_fn = lambda staleness, cid: beta, 
            use_relative_data_size_weights = use_relative_data_size_weights,
            aggregation_scheme = "buffer_size", 
            aggregation_value = num_clients_fit,
            participation_prob_estimator = participation_prob_estimator,
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
            fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        )

        self.num_clients_fit = num_clients_fit