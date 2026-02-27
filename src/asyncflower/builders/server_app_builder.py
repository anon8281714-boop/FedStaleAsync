from typing import Dict
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common.context import Context
from flwr.server import SimpleClientManager
from asyncflower.simulation.device import DevicesSimulation
from asyncflower.strategy.async_strategy import AsyncStrategy
from asyncflower.simulation.async_server import AsyncServerSimulation
from torch.distributions import Distribution

def build_server_app(
    num_clients: int,
    strategy: AsyncStrategy, 
    max_num_rounds: int, 
    max_training_time: float, 
    service_time_distributions: Dict[str, Distribution], 
    verbose = False, 
    checkpoint_rounds_interval = None,
    checkpoint_handler_fn = None,
    metrics_enrichment_fn = None
) -> ServerApp:
    def server_fn(context: Context):
        server = AsyncServerSimulation(
            devices = DevicesSimulation(service_time_distributions = service_time_distributions),
            client_manager = SimpleClientManager(), 
            strategy = strategy,
            verbose = verbose,
            checkpoint_rounds_interval = checkpoint_rounds_interval,
            checkpoint_handler_fn = checkpoint_handler_fn,
            metrics_enrichment_fn = metrics_enrichment_fn
        )

        server.set_max_workers(num_clients)

        server_config = ServerConfig(num_rounds = max_num_rounds, round_timeout = max_training_time)

        app = ServerAppComponents(
            server = server,
            config = server_config,
        )

        return app
    
    server_app = ServerApp(server_fn = server_fn)

    return server_app