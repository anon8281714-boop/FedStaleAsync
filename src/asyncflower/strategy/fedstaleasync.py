from asyncflower.strategy.async_strategy import AsyncStrategy
from asyncflower.utils.criterions import AvailableClientsCriterion
from asyncflower.utils.prob_estimator import BaseParticipationProbabilityEstimator
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import Scalar, Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes, NDArrays
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from typing import Optional, Dict, Callable, Union, List, Tuple, Literal
import numpy as np
import copy

class FedStaleAsync(AsyncStrategy):
    def __init__(
        self, 
        concurrency: int,
        learning_rate: float,
        alpha_staleness_weighting_fn: Optional[Callable[[int, int], float]],
        beta_staleness_weighting_fn: Optional[Callable[[int, int], float]],
        aggregation_value: float|int,
        aggregation_scheme: Literal["buffer_size", "round_duration"] = "round_duration",
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
        self.concurrency = concurrency
        self.aggregation_scheme = aggregation_scheme
        self.aggregation_value = aggregation_value
        self.learning_rate = learning_rate
        self.alpha_staleness_weighting_fn = alpha_staleness_weighting_fn or (lambda staleness, cid: 1)
        self.beta_staleness_weighting_fn = beta_staleness_weighting_fn or (lambda staleness, cid: 1)
        self.use_relative_data_size_weights = use_relative_data_size_weights
        self.participation_prob_estimator = participation_prob_estimator
        self.full_participation_memory_initialization = full_participation_memory_initialization
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

        self._last_updates = {}
        self._beta_scaling_history = {}
        self._last_total_consumed_updates = None
        self._num_sampled_clients_memory_initialization = None

        self._busy_clients = set()

    def round_delimiter(self, server_round: int) -> Tuple[Literal["buffer_size", "round_duration"], int]:
        if self.full_participation_memory_initialization and server_round == 0:
            return ("buffer_size", self._num_sampled_clients_memory_initialization)
        return (self.aggregation_scheme, self.aggregation_value)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[tuple[ClientProxy, FitIns]]:
        
        client_manager.wait_for(self.min_available_clients)
        
        if server_round == 0:
            if self.full_participation_memory_initialization:
                sample_size = client_manager.num_available()  
                self._num_sampled_clients_memory_initialization = sample_size
            else:
                sample_size = self.concurrency

        elif server_round == 1:
            if self.full_participation_memory_initialization:
                sample_size = self.concurrency 
            else:
                sample_size = self._last_total_consumed_updates

        else:
            sample_size = self._last_total_consumed_updates

        config = {"delta": True}
        if self.on_fit_config_fn is not None:
            config.update(self.on_fit_config_fn(server_round))
        
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

        stored_fit_metrics = []
        participating_fit_metrics = []

        self._last_total_consumed_updates = 0

        for client_proxy, _ in failures:
            self._last_total_consumed_updates += 1
            self._busy_clients.remove(client_proxy.cid)

        participating_cids = []
        round_received_updates = {}

        for client_proxy, new_update in results:
            cid = new_update.metrics["cid"]
            self._last_total_consumed_updates += 1
            participating_cids.append(cid)
            self._busy_clients.remove(client_proxy.cid)
            round_received_updates[cid] = new_update

        stored_cids = set(self._last_updates.keys())
        participating_cids = set(participating_cids)
        all_cids = stored_cids.union(participating_cids)

        total_clients = len(all_cids)

        self.participation_prob_estimator.update_estimations(server_round, participating_cids)
        
        total_examples_per_cid = {}
        processed_updates = {}

        for cid in all_cids:
            processed_updates[cid] = [
                np.zeros_like(global_layer_params, dtype = "float64") for global_layer_params in global_params
            ]

            stored_update = self._last_updates.get(cid, None)
            if stored_update:
                cid_is_participating = cid in participating_cids
                total_examples_per_cid[cid] = stored_update.num_examples

                stored_update.metrics["source"] = "server_memory"
                stored_update.metrics["num_examples"] = stored_update.num_examples
                stored_update.metrics["staleness"] = server_round - stored_update.metrics["server_round_start"]
                stored_update.metrics["scaling"] = self.beta_staleness_weighting_fn(stored_update.metrics["staleness"], cid)
                stored_update.metrics["stored_for_n_rounds"] = server_round - stored_update.metrics["server_round_end"]
            
                self._beta_scaling_history[cid][0] += stored_update.metrics["scaling"] # Update Beta Sum
                self._beta_scaling_history[cid][1] += 1                                # Increment Total Rounds
                cumulative_beta_non_participating, total_non_participating_rounds = self._beta_scaling_history[cid] 

                stored_update.metrics["scaling_avg_non_participating_rounds"] = cumulative_beta_non_participating / total_non_participating_rounds
            
                stored_update.metrics["participation_discount_applied"] = cid_is_participating
                stored_update.metrics["participation_prob"] = self.participation_prob_estimator.get_participation_probability(cid)
                
                stored_update.metrics["update_weight"] = stored_update.metrics["scaling"] - (int(cid_is_participating) * stored_update.metrics["scaling_avg_non_participating_rounds"] / stored_update.metrics["participation_prob"])

                stored_fit_metrics += [(stored_update.num_examples, stored_update.metrics)]

                for layer, layer_params in enumerate(parameters_to_ndarrays(stored_update.parameters)):
                    processed_updates[cid][layer] += stored_update.metrics["update_weight"] * layer_params
        
            new_update = round_received_updates.get(cid, None)
            if new_update:
                new_update.metrics["source"] = "participating_client"
                new_update.metrics["num_examples"] = new_update.num_examples
                new_update.metrics["staleness"] = server_round - new_update.metrics["server_round_start"]
                new_update.metrics["scaling"] = self.alpha_staleness_weighting_fn(new_update.metrics["staleness"], cid)
                new_update.metrics["stored_for_n_rounds"] = server_round - new_update.metrics["server_round_end"]

                total_examples_per_cid[cid] = new_update.num_examples

                self._beta_scaling_history[cid] = [0,0]  # restart (cumulative_beta, total_rounds), which accounts only for non participating rounds

                new_update.metrics["participation_prob"] = self.participation_prob_estimator.get_participation_probability(cid)
                new_update.metrics["update_weight"] = new_update.metrics["scaling"] / new_update.metrics["participation_prob"]

                participating_fit_metrics += [(new_update.num_examples, new_update.metrics)]

                for layer, layer_params in enumerate(parameters_to_ndarrays(new_update.parameters)):
                    processed_updates[cid][layer] += new_update.metrics["update_weight"] * layer_params

                self._last_updates[cid] = new_update

        total_examples = sum(total_examples_per_cid.values())

        aggregation_weights = {
            "cid": [], "rel_data_size": [], "uniform_aggregation_weight": [], "aggregation_applied_weight": []
        }

        updated_global_params = [
            layer.astype(dtype = "float64")
            for layer in copy.deepcopy(global_params)
        ]

        for cid, processed_update in processed_updates.items():
            rel_data_size = total_examples_per_cid[cid] / total_examples
            uniform_aggregation_weight = 1 / total_clients
            aggregation_applied_weight = rel_data_size if self.use_relative_data_size_weights else uniform_aggregation_weight

            for layer, processed_layer_params in enumerate(processed_update):
                updated_global_params[layer] -= self.learning_rate * aggregation_applied_weight * processed_layer_params

            aggregation_weights["cid"] += [cid]
            aggregation_weights["rel_data_size"] += [rel_data_size]
            aggregation_weights["uniform_aggregation_weight"] += [uniform_aggregation_weight]
            aggregation_weights["aggregation_applied_weight"] += [aggregation_applied_weight]

        parameters_aggregated = ndarrays_to_parameters(updated_global_params)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            metrics_aggregated = {
                "server_memory": self.fit_metrics_aggregation_fn(stored_fit_metrics),
                "participating_client": self.fit_metrics_aggregation_fn(participating_fit_metrics),
                "aggregation": aggregation_weights,
                "total_clients": total_clients,
                "total_examples": total_examples
            }

        print(metrics_aggregated)

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