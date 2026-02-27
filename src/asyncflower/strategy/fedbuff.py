from asyncflower.strategy.async_strategy import AsyncStrategy
from asyncflower.strategy.fedstaleasync import FedStaleAsync
from asyncflower.utils.criterions import AvailableClientsCriterion
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import Scalar, Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes, NDArrays
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from asyncflower.utils.prob_estimator import BaseParticipationProbabilityEstimator
from typing import Optional, Dict, Callable, Union, List, Tuple, Literal
import numpy as np

class FedBuff(AsyncStrategy):
    def __init__(
        self, 
        concurrency: int,
        buffer_size: int,
        learning_rate: float,
        staleness_scaling_fn: Optional[Callable[[int, int], float]] = None,
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
        self.concurrency = concurrency
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.staleness_scaling_fn = staleness_scaling_fn 
        if self.staleness_scaling_fn is None:
            self.staleness_scaling_fn = lambda staleness, cid: 1
        self.fraction_eval = fraction_eval
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.on_fit_config_fn = on_fit_config_fn
        self.accept_failures = accept_failures
        self.evaluate_fn = evaluate_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.initial_parameters = initial_parameters

        self._busy_clients = set()

    def round_delimiter(self, server_round: int) -> Tuple[Literal["buffer_size", "round_duration"], int]:
        return ("buffer_size", self.buffer_size)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[tuple[ClientProxy, FitIns]]:
        sample_size = self.concurrency if server_round == 0 else self.buffer_size

        config = {"delta": True}
        if self.on_fit_config_fn is not None:
            config.update(self.on_fit_config_fn(server_round))
        
        client_manager.wait_for(self.min_available_clients)
        sampled_clients = client_manager.sample(
            num_clients = sample_size, 
            min_num_clients = sample_size,
            criterion = AvailableClientsCriterion(
                all_client_cids = [cid for cid, client in client_manager.all().items()],
                busy_client_cids = self._busy_clients
            )
        )
        clients_fitting_configuration = []
        for client_proxy in sampled_clients:
            instruction = FitIns(parameters = parameters, config = config)
            clients_fitting_configuration.append((client_proxy, instruction))
            self._busy_clients.add(client_proxy.cid)

        return clients_fitting_configuration
        
    def aggregate_fit(
        self,
        server_round: int, 
        results: List[tuple[ClientProxy, FitRes]], 
        failures: List[Union[tuple[ClientProxy, FitRes], BaseException]],
        global_params: Parameters
    ) -> tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}
        
        global_params = parameters_to_ndarrays(global_params)

        for client_proxy, update_res in [*results, *failures]:
            self._busy_clients.remove(client_proxy.cid)

        clients_scaled_updates = []
        for client_proxy, update_res in results:
            update_params = parameters_to_ndarrays(update_res.parameters)
            update_staleness = update_res.metrics["staleness"]
            staleness_scaling = self.staleness_scaling_fn(update_staleness, update_res.metrics["cid"])
            update_scaled_params = [
                layer_params * staleness_scaling for layer_params in update_params
            ]
            clients_scaled_updates.append(update_scaled_params)

        clients_update_params_by_layer = zip(*clients_scaled_updates)
        aggregated_ndarrays = [
            global_layer_params - self.learning_rate * np.mean(update_layer_params, axis = 0) 
            for global_layer_params, update_layer_params in zip(global_params, clients_update_params_by_layer)
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
 
        return parameters_aggregated, metrics_aggregated

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

    def num_evaluation_clients(self, num_available_clients: int) -> int:
        num_clients = max(int(self.fraction_eval * num_available_clients), self.min_evaluate_clients)
        return num_clients
    
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
    
class UnbiasedFedBuff(FedStaleAsync):
    def __init__(
        self, 
        concurrency: int,
        aggregation_scheme: Literal["buffer_size", "round_duration"],
        aggregation_value: int | float,
        learning_rate: float,
        participation_prob_estimator: BaseParticipationProbabilityEstimator, 
        full_participation_memory_initialization: bool = False,
        use_relative_data_size_weights: bool = True, 
        staleness_scaling_fn: Optional[Callable[[float], float]] = None,
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
            alpha_staleness_weighting_fn = staleness_scaling_fn,
            beta_staleness_weighting_fn = lambda staleness: 0,
            aggregation_value = aggregation_value,
            aggregation_scheme = aggregation_scheme,
            participation_prob_estimator = participation_prob_estimator,
            use_relative_data_size_weights = use_relative_data_size_weights,
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