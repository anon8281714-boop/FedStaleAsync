from typing import Dict
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.client import ClientApp
from flwr.common.context import Context
from flwr.simulation import run_simulation
from flwr.server import SimpleClientManager
from asyncflower.simulation.device import DevicesSimulation
from asyncflower.strategy.async_strategy import AsyncStrategy
from asyncflower.model.model import CustomModel
from asyncflower.simulation.async_server import AsyncServerSimulation
from asyncflower.client import CustomClient
from asyncflower.data.preprocessing import load_client_data
from torch.distributions import Distribution
from asyncflower.utils import experiments as utils
import os

def generate_server_fn(
    num_clients: int,
    strategy: AsyncStrategy, 
    max_num_rounds: int, 
    max_training_time: float, 
    service_time_distributions: Dict[str, Distribution], 
    verbose = False, 
    history_handler_fn = None
):
    def server_fn(context: Context):
        server = AsyncServerSimulation(
            devices = DevicesSimulation(service_time_distributions = service_time_distributions),
            client_manager = SimpleClientManager(), 
            strategy = strategy,
            verbose = verbose,
            history_handler_fn = history_handler_fn
        )

        server.set_max_workers(num_clients)

        server_config = ServerConfig(num_rounds = max_num_rounds, round_timeout = max_training_time)

        app = ServerAppComponents(
            server = server,
            config = server_config,
        )

        return app
    
    return server_fn

def generate_client_fn(
    dataset: str,
    dataset_dir: str,
    batch_size: int, 
    seed: int, 
    model_architecture: str, 
    device: str = "cpu"
):
    def client_fn(context: Context):
        cid = str(context.node_config["partition-id"])

        train_loader, val_loader = load_client_data(
            cid = cid, 
            data_dir = dataset_dir,
            dataset = dataset,
            batch_size = batch_size,
            seed = seed
        )

        model = CustomModel(model = model_architecture)

        client = CustomClient(
            cid = cid, 
            train_loader = train_loader, 
            val_loader = val_loader, 
            model = model, 
            device = device,
        )

        return client
    
    return client_fn

def run_experiment(
    num_clients: int, 
    max_num_rounds: int,
    max_training_time: float, 
    dataset: str, 
    dataset_dir: str, 
    model_architecture: str,
    batch_size: int, 
    strategy: AsyncStrategy,
    service_time_distributions, 
    results_dir: str, 
    seed: int = 0,
    verbose: bool = False,
    device: str = "cpu", 
    num_cpus_per_client: int = 2, 
    num_gpus_per_client: float = 0.25
):
    utils.set_seed(seed)
    os.makedirs(results_dir, exist_ok = True)

    history_handler_fn = utils.generate_history_handler_fn(results_path = f"{results_dir}/history.json")

    server_fn = generate_server_fn(
        num_clients = num_clients,
        strategy = strategy, 
        max_num_rounds = max_num_rounds,
        max_training_time = max_training_time, 
        service_time_distributions = service_time_distributions,
        verbose = verbose,
        history_handler_fn = history_handler_fn
    )

    client_fn = generate_client_fn(
        dataset = dataset, 
        dataset_dir = dataset_dir, 
        batch_size = batch_size,
        device = device,
        model_architecture = model_architecture,
        seed = seed
    )

    run_simulation(
        server_app = ServerApp(server_fn = server_fn),
        client_app = ClientApp(client_fn = client_fn),
        num_supernodes = num_clients, 
        backend_name = "ray",
        backend_config = {"client_resources": {
            "num_cpus": num_cpus_per_client, "num_gpus": num_gpus_per_client
        }} if num_cpus_per_client and num_gpus_per_client else None
    )
