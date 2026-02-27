from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes,
    GetPropertiesIns, GetPropertiesRes, GetParametersIns, GetParametersRes,
    ndarrays_to_parameters, Status, Code, parameters_to_ndarrays
)

from asyncflower.model.model import CustomModel
from torch.utils.data import DataLoader
from flwr.client import Client
import torch
from flwr.common import log
from logging import INFO

class CustomClient(Client):
    def __init__(
        self, 
        cid: str, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: CustomModel,
        device: torch.device,
    ) -> None:
        self.cid = cid
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.device = device

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        properties = {"cid": self.cid}
        response = GetPropertiesRes(
            Status(code = Code.OK, message = ""),
            properties = properties
        )
        return response

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        parameters = ndarrays_to_parameters(self.model.get_model_parameters())
        response = GetParametersRes(
            Status(code = Code.OK, message = ""),
            parameters = parameters
        )
        return response

    def fit(self, ins: FitIns) -> FitRes:
        parameters, config = ins.parameters, ins.config
        parameters = parameters_to_ndarrays(parameters)
        self.model.set_model_parameters(parameters)

        optimizer_str = config.get("optimizer", "SGD")
        iteration_mode = config.get("iteration_mode", "epoch")
        num_iterations = config.get("num_iterations", 1)
        learning_rate = config.get("learning_rate", 0.01)
        weight_decay = config.get("weight_decay", 0)
        momentum = config.get("sgd_momentum", 0)
        adam_beta0 = config.get("adam_beta0",0.9)
        adam_beta1 = config.get("adam_beta1",0.999)
        delta = config.get("delta", False)
        verbose = config.get("verbose", False)

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = None
        if optimizer_str.lower() == "adam":
            optimizer = torch.optim.Adam(
                params = self.model.parameters(),
                lr = learning_rate, 
                weight_decay = weight_decay, 
                betas = (adam_beta0, adam_beta1) 
            )
        elif optimizer_str.lower() == "sgd":
            optimizer = torch.optim.SGD(
                params = self.model.parameters(), 
                lr = learning_rate, 
                weight_decay = weight_decay,
                momentum = momentum
            )

        losses, accuracies = self.model.fit(
            train_loader = self.train_loader,
            iteration_mode = iteration_mode,
            num_iterations = num_iterations,
            criterion = criterion, 
            optimizer = optimizer,
            device = self.device,
            verbose = verbose
        )   
        num_examples = len(self.train_loader.dataset)

        if delta:
            updated_parameters = self.model.get_model_parameters()
            parameters = [
                initial_layer_params - updated_layer_params 
                for initial_layer_params, updated_layer_params in zip(parameters, updated_parameters)
            ]

        else:
            parameters = self.model.get_model_parameters()
        
        parameters = ndarrays_to_parameters(parameters)

        response = FitRes(
            Status(code = Code.OK, message = ""),
            parameters = parameters, 
            num_examples = num_examples, 
            metrics = {"loss": losses[-1], "acc": accuracies[-1]}
        )

        return response

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        parameters, config = ins.parameters, ins.config
        parameters = parameters_to_ndarrays(parameters)
        self.model.set_model_parameters(parameters)

        criterion = torch.nn.CrossEntropyLoss()
        loss, acc = self.model.evaluate(test_loader = self.val_loader, criterion = criterion, device = self.device)
        num_examples = len(self.val_loader.dataset)

        response = EvaluateRes(
            Status(code = Code.OK, message = ""),
            loss = loss, 
            num_examples = num_examples, 
            metrics = {"loss": loss, "acc": acc}
        )
        return response

    def to_client(self) -> Client:
        return self