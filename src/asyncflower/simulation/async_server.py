from flwr.server import Server, ClientManager, History
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns, Code, Parameters, Scalar, GetPropertiesIns, Metrics, FitRes
from flwr.common.logger import log
from logging import INFO, ERROR, DEBUG
from typing import Optional, Callable
import timeit
import concurrent.futures
from asyncflower.strategy.async_strategy import AsyncStrategy
from asyncflower.simulation.device import DevicesSimulation
from queue import PriorityQueue
from threading import Condition

class AsyncServerSimulation(Server):
    def __init__(
        self,
        *,
        devices: DevicesSimulation,
        client_manager: ClientManager,
        strategy: AsyncStrategy, 
        checkpoint_rounds_interval = None,
        checkpoint_handler_fn = None,
        metrics_enrichment_fn: Callable[[FitRes], Metrics] | None = None, 
        verbose = False,
    ) -> None:
        self._client_manager = client_manager
        self.strategy = strategy
        super().__init__(client_manager = client_manager, strategy = strategy)

        self._devices = devices

        self._global_round = 0
        self._global_simulation_time = 0
        self._round_start_time = 0

        self._received_updates = {}
        self._updates_scheduling_info = PriorityQueue()
        
        self._verbose = verbose
        self._executor = None
        self._condition = Condition()

        self._checkpoint_rounds_interval = checkpoint_rounds_interval
        self._checkpoint_handler_fn = checkpoint_handler_fn
        self._metrics_enrichment_fn = metrics_enrichment_fn

    def fit_client(
        self, client: ClientProxy, ins: FitIns, timeout: Optional[float], group_id: int, server_round: Optional[int]
    ) -> None:
        """Refine parameters on a single client."""
        cid = client.get_properties(GetPropertiesIns({}), None, None).properties["cid"]
        try:
            # TODO: Change to DEBUG
            if self._verbose:
                log(
                    INFO, 
                    f"Client {cid} is starting its local training."
                )

            fit_res = client.fit(ins, timeout=timeout, group_id=group_id)
            
            result = (client, fit_res)

            status = "Ok" if fit_res.status.code == Code.OK else "Failure"
            with self._condition:
                self._received_updates[cid] = (status, result)
                self._condition.notify_all()

        except Exception as failure:
            if failure is not None:
                with self._condition:
                    self._received_updates[cid] = ("Failure", failure)
                    self._condition.notify_all()
        
        finally:
            # TODO: Change to DEBUG
            if self._verbose:
                log(
                    INFO, 
                    f"Client {cid} has finished its local training and the result was stored by the server."
                )

    def fit_clients(
        self,
        client_instructions: list[tuple[ClientProxy, FitIns]],
        timeout: Optional[float],
        group_id: int,
        server_round: Optional[int]
    ) -> None:
        for client_proxy, ins in client_instructions:
            self._executor.submit(self.fit_client, client_proxy, ins, timeout, group_id, server_round)

    def sample_and_fit_clients(self, server_round: int, timeout: Optional[float]) -> None:
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        for client, _ in client_instructions:
            cid = client.get_properties(GetPropertiesIns({}), None, None).properties["cid"]
            sampled_service_time = self._devices.sample_service_time(cid)
            completion_time = self._global_simulation_time + sampled_service_time
            self._updates_scheduling_info.put((
                completion_time, 
                {"cid": cid, "server_round": self._global_round, "service_time": sampled_service_time}
            )) 

        if not client_instructions:
            log(INFO, "configure_fit: no clients selected, cancel")
            return None
        log(
            INFO,
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        self.fit_clients(
            client_instructions,
            timeout=timeout,
            group_id=server_round,
            server_round = server_round
        )

    def fit_round(self, server_round: int, timeout: Optional[float]) -> Optional[
        tuple[Optional[Parameters], dict[str, Scalar]]
    ]:
        results = []
        failures = []
        
        round_delimiter_type, round_delimiter_param = self.strategy.round_delimiter(server_round = server_round)

        round_timeout_at_time = None
        
        if round_delimiter_type == "buffer_size":
            num_updates_to_consume = round_delimiter_param

        elif round_delimiter_type == "round_duration":
            round_duration = round_delimiter_param

            # TODO: Change to DEBUG
            if self._verbose:
                log(
                    INFO, 
                    f"Round duration is of {round_duration} seconds."
                )

            with self._updates_scheduling_info.mutex:
                num_updates_to_consume = 0
                for completion_time, _ in self._updates_scheduling_info.queue:
                    if completion_time <= self._round_start_time + round_duration:
                        num_updates_to_consume += 1
                    else:
                        break

            # Simulating round timeout.
            if num_updates_to_consume < self._updates_scheduling_info.qsize():
                round_timeout_at_time = self._global_simulation_time + round_duration

        # TODO: Change to DEBUG
        if self._verbose:
            log(
                INFO, 
                f"Strategy required {num_updates_to_consume} updates."
            )

        for i in range(num_updates_to_consume):
            completion_time, update_info = self._updates_scheduling_info.get()
            self._global_simulation_time = completion_time
            
            cid = update_info["cid"]

            # TODO: Change to DEBUG
            if self._verbose:
                log(
                    INFO, 
                    f"Server is waiting for the update from client {cid}, scheduled to be completed at time {self._global_simulation_time:.4}s."
                )
            
            with self._condition:
                while self._received_updates.get(cid, None) is None:
                    self._condition.wait()

                status, update_res = self._received_updates.get(cid)
                self._received_updates[cid] = None
            
            client, fit_res = update_res

            fit_res.metrics["cid"] = cid
            fit_res.metrics["completion_time"] = completion_time
            fit_res.metrics["service_time"] =  update_info["service_time"]
            fit_res.metrics["server_round_start"] = update_info["server_round"]
            fit_res.metrics["server_round_end"] = self._global_round
            fit_res.metrics["staleness"] = fit_res.metrics["server_round_end"] - fit_res.metrics["server_round_start"]

            if self._metrics_enrichment_fn:
                fit_res.metrics.update(self._metrics_enrichment_fn(fit_res))

            if status == "Ok":
                # TODO: Change to DEBUG
                if self._verbose:
                    log(
                        INFO, 
                        f"Server consumed the update from client {cid} with success."
                    )
                results.append(update_res)

            else:
                # TODO: Change to DEBUG
                if self._verbose:
                    log(
                        INFO, 
                        f"Server failed to consumed the update from client {cid}: {fit_res.metrics}."
                    ) 
                failures.append(update_res)

        if round_timeout_at_time:
            self._global_simulation_time = round_timeout_at_time

        log(
            INFO,
            "aggregate_fit: received %s results and %s failures",
            len(results),
            len(failures),
        )

        parameters_aggregated, metrics_aggregated = self.strategy.aggregate_fit(
            server_round = server_round,
            results = results, 
            failures = failures,
            global_params = self.parameters
        )

        metrics_aggregated["round_stats"] = {
            "server_round": server_round,
            "num_received_updates": num_updates_to_consume
        }

        return parameters_aggregated, metrics_aggregated

    def fit(self, num_rounds: int, timeout: Optional[float]) -> tuple[History, float]:
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers = self.max_workers)
        
        history = History()
    
        log(INFO, "[INIT]")
        self.parameters = self._get_initial_parameters(server_round = self._global_round, timeout = None)
        
        start_time = timeit.default_timer()
        
        try:
            while self._global_round < num_rounds and self._global_simulation_time <= timeout:
                round_metrics = {}

                self._round_start_time = self._global_simulation_time

                log(INFO, "")
                log(INFO, "[ROUND %s]", self._global_round)
                self.sample_and_fit_clients(server_round = self._global_round, timeout = None)

                res_cen = self.strategy.evaluate(self._global_round, parameters=self.parameters)
                if res_cen is not None:
                    loss_cen, metrics_cen = res_cen
                    log(
                        INFO,
                        "fit progress: (%s, %s, %s, %s)",
                        self._global_round,
                        loss_cen,
                        metrics_cen,
                        timeit.default_timer() - start_time,
                    )
                    
                    round_metrics["loss_centralized"] = {"server_round": self._global_round, "loss": loss_cen}
                    round_metrics["metrics_centralized"] = {"server_round": self._global_round, "metrics": metrics_cen}

                res_fit = self.fit_round(
                    server_round=self._global_round,
                    timeout=timeout,
                )
                if res_fit is not None:
                    parameters_prime, fit_metrics = res_fit
                    if parameters_prime:
                        self.parameters = parameters_prime
                    fit_metrics["round_stats"] = {
                        **fit_metrics.get("round_stats", {}), 
                        "round_start_time": self._round_start_time,
                        "round_end_time": self._global_simulation_time
                    }
                    round_metrics["metrics_distributed_fit"] = {"server_round": self._global_round, "metrics": fit_metrics}

                if self._checkpoint_handler_fn:
                    if (self._global_round > 0) and (self._global_round % self._checkpoint_rounds_interval == 0):
                        self._checkpoint_handler_fn(history, self._global_round)

                if "loss_centralized" in round_metrics:
                    history.add_loss_centralized(**round_metrics["loss_centralized"])
                
                if "metrics_centralized" in round_metrics:
                    history.add_metrics_centralized(**round_metrics["metrics_centralized"])

                if "metrics_distributed_fit" in round_metrics:
                    history.add_metrics_distributed_fit(**round_metrics["metrics_distributed_fit"])

                self._global_round += 1
                
            log(INFO, "Training has been executed successfully.")
            self._executor.shutdown()
            self._executor = None
            end_time = timeit.default_timer()
            elapsed = end_time - start_time
        
        except Exception as e:
            print(f"Error: {e}")
            raise e
        
        finally:
            if self._checkpoint_handler_fn:
                self._checkpoint_handler_fn(history, self._global_round)

        return history, elapsed