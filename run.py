import os
import json
import hydra
import joblib
import torch
import socket
import platform
import numpy as np
from flwr.common import log
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from flwr.simulation import run_simulation
from asyncflower.utils import experiments as utils
from asyncflower.builders.client_app_builder import build_client_app
from asyncflower.builders.server_app_builder import build_server_app
from asyncflower.builders.metrics_enrichment_builder import build_update_metrics_enrichment_fn
from asyncflower.builders.strategy.builder import build_strategy
from asyncflower.builders.data_partitioning.builder import build_data_partitioning
from asyncflower.builders.device_specifications.builder import build_devices
from asyncflower.builders.participation_prob_estimation.estimation_method.builder import build_participation_prob_estimator
from asyncflower.builders.participation_prob_estimation.smooth_method.builder import build_smoothed_participation_prob_estimator
from asyncflower.data.preprocessing import load_test_data, load_train_data
from asyncflower.model.model import CustomModel
from logging import DEBUG, INFO
from time import sleep

def dictconfig_to_cli(cfg: DictConfig, prefix="") -> list:
    cli_args = []
    for k, v in cfg.items():
        if isinstance(v, DictConfig):
            cli_args += dictconfig_to_cli(v, prefix=f"{prefix}{k}.")
        else:
            cli_args.append(f"{prefix}{k}={v}")
    return cli_args

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    runtime_cfg = HydraConfig.get()
    job_num = int(runtime_cfg.job.num)
    sleep((job_num + 1) * 15)

    log(DEBUG, f"Running on {socket.gethostname()}/{platform.node()}")

    log(DEBUG, f"Cuda is available: {torch.cuda.is_available()}")

    log(DEBUG, f"Storing config file...")
    job_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    resolved_yaml = OmegaConf.to_yaml(resolved_cfg)

    with open(f"{job_output_dir}/config.yaml", "w") as f:
        f.write(resolved_yaml)

    with open(f"{job_output_dir}/cli_args.json", "w") as f:
        json.dump({
            "config_args": " ".join(dictconfig_to_cli(DictConfig(resolved_cfg)))
        }, f)
        
    best_save = {
        "best_params": None, 
        "best_loss": np.inf,
        "best_round": None,
        "best_model_class": None
    }

    def update_best_model(loss, accuracy, model: CustomModel, server_round) -> None:
        if loss <= best_save["best_loss"]:
            log(
                INFO, 
                f"BEST_MODEL: Found a best model at round {server_round}: last_loss = {best_save['best_loss']}, new_loss = {loss}"
            )
            best_save["best_loss"] = loss
            best_save["best_params"] = model.get_model_parameters()
            best_save["best_round"] = server_round
            best_save["best_model_class"] = type(model.model)

    utils.set_seed(cfg.trainer.seed)
    os.makedirs(cfg.trainer.results_dir, exist_ok = True)

    log(DEBUG, f"Creating/loading device specifications...")

    device_specifications_dir = build_devices(
        scheme = cfg.device_specifications.scheme,
        **cfg.device_specifications.params
    )

    service_time_distributions = joblib.load(f"{device_specifications_dir}/specifications.joblib")

    log(DEBUG, f"Device specifications path: {device_specifications_dir}")

    log(DEBUG, f"Creating/loading data partitions...")

    partitions_dir = build_data_partitioning(
        scheme = cfg.data_partitioning.scheme,
        **cfg.data_partitioning.params
    )

    log(INFO, f"Data partitions path: {partitions_dir}")

    log(DEBUG, f"Creating/loading data partitions...")

    test_loader = load_test_data(
        data_dir = partitions_dir,
        dataset = cfg.task.dataset,
        batch_size = cfg.trainer.batch_size
    )

    calibration_loader = load_train_data(
        data_dir = partitions_dir,
        dataset = cfg.task.dataset,
        batch_size = cfg.trainer.batch_size
    )

    checkpoint_handler_fn = utils.generate_checkpoint_handler_fn(
        results_dir = cfg.trainer.results_dir, 
        best_save = best_save, 
        num_clients = cfg.trainer.num_clients, 
        partitions_dir = partitions_dir, 
        dataset = cfg.task.dataset, 
        batch_size = cfg.trainer.batch_size, 
        seed = cfg.trainer.seed, 
        device = cfg.trainer.device.device,
    )

    log(DEBUG, f"Instantiating strategy...")

    participation_prob_estimator = None
    if cfg.participation_prob_estimation:
        participation_prob_estimator = build_participation_prob_estimator(
            scheme = cfg.participation_prob_estimation.estimation_method.scheme,
            **cfg.participation_prob_estimation.estimation_method.params,
        )
        participation_prob_estimator = build_smoothed_participation_prob_estimator(
            participation_prob_estimator = participation_prob_estimator,
            scheme = cfg.participation_prob_estimation.smooth_method.scheme,
            **cfg.participation_prob_estimation.smooth_method.params,
        )

    strategy = build_strategy(
        scheme = cfg.strategy.scheme,
        model_architecture = cfg.task.model_architecture,
        test_loader = test_loader,
        calibration_loader = calibration_loader,
        on_evaluate_callback = update_best_model,
        participation_prob_estimator = participation_prob_estimator,
        device = cfg.trainer.device.device,
        optimizer = cfg.trainer.local_optimizer.name,
        local_learning_rate = cfg.trainer.local_optimizer.learning_rate,
        iteration_mode = cfg.trainer.local_optimizer.iteration_mode,
        num_iterations = cfg.trainer.local_optimizer.num_iterations,
        weight_decay = cfg.trainer.local_optimizer.weight_decay,
        sgd_momentum = OmegaConf.select(cfg, "trainer.local_optimizer.momentum"),
        adam_betas = (
            OmegaConf.select(cfg, "trainer.local_optimizer.beta0"), 
            OmegaConf.select(cfg, "trainer.local_optimizer.beta1")
        ),
        seed = cfg.trainer.seed,
        **cfg.strategy.params
    )

    metrics_enrichment_fn = None
    if cfg.update_metrics_enrichment.enable_metrics_enrichment:
        metrics_enrichment_fn = build_update_metrics_enrichment_fn(**cfg.update_metrics_enrichment.params)

    log(DEBUG, f"Instantiating server app...")

    server_app = build_server_app(
        num_clients = cfg.trainer.num_clients,
        strategy = strategy,
        max_num_rounds = cfg.trainer.max_num_rounds,
        max_training_time = cfg.trainer.max_training_time,
        service_time_distributions = service_time_distributions,
        verbose = cfg.trainer.verbose, 
        checkpoint_rounds_interval = cfg.trainer.checkpoint_rounds_interval,
        checkpoint_handler_fn = checkpoint_handler_fn,
        metrics_enrichment_fn = metrics_enrichment_fn
    )
    
    log(DEBUG, f"Instantiating client app...")

    client_app = build_client_app(
        dataset = cfg.task.dataset,
        dataset_dir = partitions_dir,
        batch_size = cfg.trainer.batch_size,
        seed = cfg.trainer.seed,
        model_architecture = cfg.task.model_architecture,
        device = cfg.trainer.device.device
    )

    log(DEBUG, f"Starting simulation...")

    backend_config = {"init_args": {"address": "local", "log_to_driver": True, "ignore_reinit_error": True}}

    if cfg.trainer.device.num_cpus_per_client and cfg.trainer.device.num_gpus_per_client:
        backend_config.update({
            "client_resources": {
                "num_cpus": cfg.trainer.device.num_cpus_per_client, 
                "num_gpus": cfg.trainer.device.num_gpus_per_client
            }
        })

    run_simulation(
        server_app = server_app,
        client_app = client_app,
        num_supernodes = cfg.trainer.num_clients, 
        backend_name = "ray",
        backend_config = backend_config
    )

if __name__ == "__main__":
    main()