from asyncflower.strategy.async_strategy import AsyncStrategy
from asyncflower.strategy.fedstaleasync import FedStaleAsync
from asyncflower.utils.prob_estimator import BaseParticipationProbabilityEstimator
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import Scalar, Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes, NDArrays
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from typing import Optional, Dict, Callable, Union, List, Tuple, Literal
import numpy as np

class FedAvg(AsyncStrategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_eval: float = 1.0,
        min_fit_clients: int = 2,
        min_evalulate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        evaluate_fn: Optional[Callable[[int, NDArrays], Optional[tuple[float, Dict[str, Scalar]]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        fit_metrics_aggregation_fn: Optional[Callable[[List[tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable[[List[tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]] = None,
        accept_failures: bool = True
    ):
        self.initial_parameters = initial_parameters
        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evalulate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.accept_failures = accept_failures

        self._round_num_active_clients = 0

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[tuple[float, Dict[str, Scalar]]]:
        if self.evaluate_fn is None:
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays)
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        initial_parameters = self.initial_parameters
        self.initial_parameters = None
        return initial_parameters

    def round_delimiter(self, server_round: int) -> Tuple[Literal["buffer_size", "round_duration"], int]:
        return ("buffer_size", self._round_num_active_clients)

    def num_fit_clients(self, num_available_clients: int) -> int:
        num_clients = max(int(self.fraction_fit * num_available_clients), self.min_fit_clients)
        return num_clients

    def num_evaluation_clients(self, num_available_clients: int) -> int:
        num_clients = max(int(self.fraction_eval * num_available_clients), self.min_evaluate_clients)
        return num_clients

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[tuple[ClientProxy, FitIns]]:
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        client_manager.wait_for(self.min_available_clients)
        sample_size = self.num_fit_clients(client_manager.num_available())
        sampled_clients = client_manager.sample(num_clients = sample_size, min_num_clients = sample_size)
        clients_fitting_configuration = []
        for client_proxy in sampled_clients:
            instruction = FitIns(parameters = parameters, config = config)
            clients_fitting_configuration.append((client_proxy, instruction))

        self._round_num_active_clients = sample_size

        return clients_fitting_configuration

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[tuple[ClientProxy, EvaluateIns]]:
        if self.fraction_eval == 0.0:
            return []
        
        config = {}
        if self.on_evaluate_config_fn is not None: 
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters = parameters, config = config)

        client_manager.wait_for(self.min_available_clients)
        sample_size = self.num_evaluation_clients(client_manager.num_available())
        sampled_clients = client_manager.sample(num_clients = sample_size, min_num_clients = sample_size)
        clients_evaluation_configuration = [(client, evaluate_ins) for client in sampled_clients]

        return clients_evaluation_configuration

    def aggregate_fit(
        self,
        server_round: int, 
        results: List[tuple[ClientProxy, FitRes]], 
        failures: List[Union[tuple[ClientProxy, FitRes], BaseException]],
        global_params: Optional[Parameters] = None
    ) -> tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        if not self.accept_failures and failures:
            return None, {}
        
        num_examples = [res.num_examples for _, res in results]
        clients_parameters_by_layer = zip(*[parameters_to_ndarrays(res.parameters) for _, res in results])
        aggregated_ndarrays = [np.average(layer_params, weights = num_examples, axis = 0) for layer_params in clients_parameters_by_layer]
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        self._round_num_active_clients = 0

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int, 
        results: List[tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        if not self.accept_failures and failures:
            return None, {}

        num_examples = [res.num_examples for _, res in results]
        losses = [res.loss for _, res in results]

        loss_aggregated = np.average(losses, weights = num_examples)

        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)

        return loss_aggregated, metrics_aggregated
    
class UnbiasedFedAvg(FedStaleAsync):
    def __init__(
        self, 
        num_fit_clients: int,
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
            concurrency = num_fit_clients,
            learning_rate = 1,
            alpha_staleness_weighting_fn = lambda staleness, cid: 1,
            beta_staleness_weighting_fn = lambda staleness, cid: 0,
            aggregation_value = num_fit_clients,
            aggregation_scheme = "buffer_size",
            use_relative_data_size_weights = use_relative_data_size_weights,
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
            fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
        )

    def aggregate_fit(
        self,
        server_round: int, 
        results: List[tuple[ClientProxy, FitRes]], 
        failures: List[Union[tuple[ClientProxy, FitRes], BaseException]],
        global_params: Parameters
    ) -> tuple[Optional[Parameters], Dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round = server_round, 
            results = results, 
            failures = failures, 
            global_params = global_params
        )
        self._last_updates = {}
        return parameters_aggregated, metrics_aggregated