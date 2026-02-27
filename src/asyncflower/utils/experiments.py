import random
import numpy as np
import torch
import json
from typing import List, Dict, Literal
from flwr.common import Scalar, NDArrays
from flwr.server import History
from torch.utils.data import DataLoader
from torch import nn
from asyncflower.data.preprocessing import load_client_data
from asyncflower.model.model import CustomModel

def hinge_fn(value, a, b): 
    if value <= b:
        return 1
    return 1 / (a * (value - b) + 1)

def clamped_linear_interpolation_fn(x, start_x, end_x, start_y, end_y, return_int = False) -> float | int:
    if x < start_x:
        y = start_y
    
    elif x > end_x:
        y = end_y

    else:
        y = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)

    y = int(y) if return_int else float(y)
    return y

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.mps.manual_seed(seed)

def evaluate_metrics_aggregation_fn(metrics: List[tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    num_examples, metrics = list(zip(*metrics))
    losses = [metrics_dict["loss"] for metrics_dict in metrics]
    accuracies = [metrics_dict["acc"] for metrics_dict in metrics]
    avg_loss = np.average(losses, weights = num_examples)
    avg_acc = np.average(accuracies, weights = num_examples)
    return {"acc": avg_acc, "loss": avg_loss}

def fit_metrics_aggregation_fn(metrics: List[tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    results = {}
    
    for num_samples, update_metrics in metrics:
        for metric_name, metric_value in update_metrics.items():
            results[metric_name] = results.get(metric_name, [])
            results[metric_name].append(metric_value)
    
    return {"individual_metrics": results}

def get_evaluate_fn(test_loader: DataLoader, model: nn.Module, device: str, callback_fn = None):
    def evaluate_fn(server_round: int, parameters: NDArrays) -> tuple[float, Dict[str, Scalar]]:
        model.set_model_parameters(parameters)

        criterion = torch.nn.CrossEntropyLoss()
        loss, acc = model.evaluate(test_loader = test_loader, criterion = criterion, device = device)

        if callback_fn:
            callback_fn(loss = loss, accuracy = acc, model = model, server_round = server_round)

        return loss, {"acc": acc}
    
    return evaluate_fn

def get_on_fit_config_fn(
    optimizer: Literal["SGD", "Adam"] = "SGD",
    learning_rate: float = 0.01, 
    momentum: float = 0, 
    betas: tuple = (0.9, 0.999), 
    weight_decay: float = 0, 
    iteration_mode: Literal["epoch", "minibatch"] = "epoch",
    num_iterations: int = 1
):
    def on_fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        if optimizer.lower() == "sgd":
            config_dict = {
                "optimizer": "sgd",
                "learning_rate": learning_rate,
                "sgd_momentum": momentum, 
                "weight_decay": weight_decay,
                "iteration_mode": iteration_mode,
                "num_iterations": num_iterations
            }

        elif optimizer.lower() == "adam":
            config_dict = {
                "optimizer": "adam",
                "learning_rate": learning_rate,
                "adam_beta0": betas[0], 
                "adam_beta1": betas[1], 
                "weight_decay": weight_decay,
                "iteration_mode": iteration_mode,
                "num_iterations": num_iterations
            }
        
        else:
            raise ValueError(f"{optimizer} is not a valid optimizer.")
        return config_dict
    
    return on_fit_config_fn

def generate_checkpoint_handler_fn(
    results_dir: str, 
    best_save: dict, 
    num_clients: int, 
    partitions_dir: str,
    dataset: str,
    batch_size: int,
    seed: int,
    device: str
):
    assert "best_params" in best_save
    assert "best_loss" in best_save
    assert "best_round" in best_save
    assert "best_model_class" in best_save

    def checkpoint_handler_fn(
        history: History, 
        server_round: int
    ):
        history.__dict__["server_round_checkpoint"] = server_round

        # Evaluating best model on local test datasets
        local_performance_best_model = []
        for cid in range(num_clients):
            cid = str(cid)
            local_train_loader, local_test_loader = load_client_data(
                cid = cid, 
                data_dir = partitions_dir, 
                dataset = dataset, 
                batch_size = batch_size,
                seed = seed
            )
            
            model_class = best_save["best_model_class"]
            best_model = CustomModel(model = model_class(), calibration_loader = local_train_loader)
            best_model.set_model_parameters(best_save["best_params"])

            loss, acc = best_model.evaluate(
                test_loader = local_test_loader, 
                criterion = nn.CrossEntropyLoss(),
                device = device
            )

            local_performance_best_model.append({"cid": cid, "loss": loss, "accuracy": acc})

        history.__dict__["best_model"] = {
            "server_round": best_save["best_round"], 
            "loss": best_save["best_loss"], 
            "model_class_str": str(best_save["best_model_class"]),
            "local_evaluation": local_performance_best_model
        }

        with open(f"{results_dir}/history.json", "w") as f:
            json.dump(history.__dict__, f)

        best_model.save(f"{results_dir}/model.joblib")
    
    return checkpoint_handler_fn
    