from typing import Literal, Callable
from asyncflower.builders.strategy.fedstaleasync_builder import build_fedstaleasync_hinge_scaling, build_fedstaleasync_constant_scaling, build_fedstaleasync_group_constant_scaling
from asyncflower.builders.strategy.fedbuff_builder import build_fedbuff_hinge_scaling, build_unbiased_fedbuff_hinge_scaling
from asyncflower.builders.strategy.fedavg_builder import build_fedavg, build_unbiased_fedavg
from asyncflower.builders.strategy.ca2fl_builder import build_ca2fl, build_unbiased_ca2fl
from asyncflower.builders.strategy.ace_builder import build_ace, build_buffered_ace
from asyncflower.builders.strategy.fedstale_builder import build_fedstale
from asyncflower.data.preprocessing import load_train_data
from asyncflower.utils import experiments as utils
from asyncflower.model.model import CustomModel
from asyncflower.strategy import AsyncStrategy
from asyncflower.utils.prob_estimator import BaseParticipationProbabilityEstimator
from flwr.common import ndarrays_to_parameters
import torch

registered_strategies = {
    "fedavg": build_fedavg,  
    "unbiased_fedavg": build_unbiased_fedavg,  
    "ace": build_ace, 
    "buffered_ace": build_buffered_ace, 
    "ca2fl": build_ca2fl, 
    "unbiased_ca2fl": build_unbiased_ca2fl, 
    "fedbuff_hinge_scaling": build_fedbuff_hinge_scaling,  
    "unbiased_fedbuff_hinge_scaling": build_unbiased_fedbuff_hinge_scaling,  
    "fedstaleasync_hinge_scaling": build_fedstaleasync_hinge_scaling,  
    "fedstaleasync_constant_scaling": build_fedstaleasync_constant_scaling,  
    "fedstaleasync_group_constant_scaling": build_fedstaleasync_group_constant_scaling, 
    "fedstale": build_fedstale
}

def build_strategy(
    scheme: str,
    model_architecture: str,
    test_loader: torch.utils.data.DataLoader,
    on_evaluate_callback: Callable,
    calibration_loader: torch.utils.data.DataLoader | None = None,
    participation_prob_estimator: BaseParticipationProbabilityEstimator | None = None, 
    device: torch.device = "cpu",
    optimizer: Literal["SGD", "Adam"] = "SGD",
    local_learning_rate: float = 0.01,
    iteration_mode: Literal["epoch", "minibatch"] = "epoch",
    num_iterations: int = 2,
    weight_decay: float = 0.0005,
    sgd_momentum: float = 0,
    adam_betas: tuple = (0.9, 0.999),
    seed: int = 0,
    **kwargs
) -> AsyncStrategy:
    utils.set_seed(seed)

    initial_parameters = ndarrays_to_parameters(
        CustomModel(model = model_architecture).get_model_parameters()
    )

    on_fit_config = utils.get_on_fit_config_fn(
        optimizer = optimizer,
        learning_rate = local_learning_rate, 
        momentum = sgd_momentum,
        betas = adam_betas,
        weight_decay = weight_decay,
        iteration_mode = iteration_mode,
        num_iterations = num_iterations
    )

    evaluate_fn = utils.get_evaluate_fn(
        test_loader = test_loader, 
        model = CustomModel(
            model = model_architecture, 
            calibration_loader = calibration_loader
        ),
        device = device, 
        callback_fn = on_evaluate_callback
    )

    parameters = {
        "initial_parameters": initial_parameters,
        "on_fit_config_fn": on_fit_config,
        "fit_metrics_aggregation_fn": utils.fit_metrics_aggregation_fn,
        "evaluate_metrics_aggregation_fn": utils.evaluate_metrics_aggregation_fn,
        "evaluate_fn": evaluate_fn,
        "participation_prob_estimator": participation_prob_estimator
    }
    parameters.update(kwargs)

    build_strategy_fn = registered_strategies.get(scheme, None)
    if not build_strategy_fn:
        raise ValueError(f"{scheme} is not a valid strategy.")

    return build_strategy_fn(**parameters)
    