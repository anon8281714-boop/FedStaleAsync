from torch import nn
from typing import List, Type, Literal
from torch.utils.data import DataLoader
from collections import OrderedDict
from asyncflower.model.fmnist import FashionMNIST_CNN
from asyncflower.model.cifar10 import CIFAR10_CNN, CIFAR10_ResNet18, CIFAR10_ResNet8
from asyncflower.model.cifar100 import CIFAR100_ResNet18
from asyncflower.model.tinyimagenet import TinyImageNet_ResNet18
import torch
import numpy as np

REGISTERED_MODELS = {
    "FashionMNIST_CNN": FashionMNIST_CNN,
    "CIFAR10_CNN": CIFAR10_CNN,
    "CIFAR10_ResNet18": CIFAR10_ResNet18,
    "CIFAR10_ResNet8": CIFAR10_ResNet8,
    "CIFAR100_ResNet18": CIFAR100_ResNet18,
    "TinyImageNet_ResNet18": TinyImageNet_ResNet18
}

class CustomModel():
    def __init__(self, model: nn.Module | str, calibration_loader: DataLoader = None):
        if type(model) is str:
            assert model in REGISTERED_MODELS, f"Unknowm model name: `{model}` is not registered."
            self.model = REGISTERED_MODELS[model]()
        else:
            self.model = model

        self._calibration_loader = calibration_loader
    
    def _fit_epochs(
        self, 
        train_loader: DataLoader,
        epochs: int, 
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        verbose: bool = False,
    ) -> tuple[float, float]:
        self.model = self.model.to(device)
        self.model.train()

        losses = []
        accuracies = []

        for epoch in range(1, epochs + 1):
            running_loss = 0
            correct = 0
            total = 0

            for batch in train_loader:
                images, labels = batch["image"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(dim = 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            losses.append(running_loss / total)
            accuracies.append(correct / total)

            if verbose:
                print(f"Epoch {epoch}/{epochs}: loss = {losses[-1]}, acc = {accuracies[-1]}")

        return losses, accuracies

    def _fit_minibatch_iterations(
        self, 
        train_loader: DataLoader,
        num_iterations: int, 
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        verbose: bool = False,
    ) -> tuple[float, float]:
        self.model = self.model.to(device)
        self.model.train()
        
        running_loss = 0
        correct = 0
        total = 0

        train_loader_iter = iter(train_loader)
        minibatch_iterations_count = 0
        while minibatch_iterations_count < num_iterations:
            try:
                batch = next(train_loader_iter)
                images, labels = batch["image"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(dim = 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                minibatch_iterations_count += 1

                if verbose:
                    print(f"Iteration {minibatch_iterations_count}/{num_iterations}: loss = {running_loss / total}, acc = {correct / total}")

            except StopIteration:
                train_loader_iter = iter(train_loader)

        loss = [running_loss / total]
        acc = [correct / total]

        return loss, acc

    def fit(
        self, 
        train_loader: DataLoader,
        iteration_mode: Literal["epoch", "minibatch"],
        num_iterations: int, 
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        verbose: bool = False,
    ) -> tuple[float, float]:
        if iteration_mode == "epoch":
            loss, acc = self._fit_epochs(
                train_loader = train_loader,
                epochs = num_iterations,
                criterion = criterion,
                optimizer = optimizer,
                device = device,
                verbose = verbose,
            )
        
        elif iteration_mode == "minibatch":
            loss, acc = self._fit_minibatch_iterations(
                train_loader = train_loader,
                num_iterations = num_iterations,
                criterion = criterion,
                optimizer = optimizer,
                device = device,
                verbose = verbose,
            )

        else:
            raise ValueError(f"{iteration_mode} is not valid.")
        
        return loss, acc

    def evaluate(
        self, 
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[float, float]:
        self.model = self.model.to(device)
        
        if callable(getattr(self.model, "recalibrate_buffers", None)):
            if not self._calibration_loader:
                raise ValueError("Calibration data loader function was not provided.")
            
            self.model.recalibrate_buffers(self._calibration_loader)

        self.model.eval()

        running_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch["image"].to(device), batch["label"].to(device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(dim = 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        loss = running_loss / total
        accuracy = correct / total

        return loss, accuracy
    
    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse = recurse)
    
    def get_model_parameters(self) -> List[np.ndarray]:
        ignored_parameters = set()
        if callable(getattr(self.model, "ignored_parameters", None)):
            ignored_parameters = self.model.ignored_parameters()

        model_parameters = [
            param_value.cpu().detach().numpy() 
            for param_name, param_value in self.model.state_dict().items() if param_name not in ignored_parameters
        ]

        return model_parameters
    
    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        ignored_parameters = set()
        if callable(getattr(self.model, "ignored_parameters", None)):
            ignored_parameters = self.model.ignored_parameters()

        param_names = [param_name for param_name in self.model.state_dict().keys() if param_name not in ignored_parameters]

        state_dict = OrderedDict()
        for key, param in zip(param_names, parameters):
            tensor = torch.tensor(param, dtype=torch.float32) 
            tensor = tensor.to(next(self.model.parameters()).device)
            state_dict[key] = tensor
        self.model.load_state_dict(state_dict, strict=True)

    def save(self, path: str):
        parameters = self.model.to("cpu").state_dict()
        torch.save(parameters, path)

    @classmethod
    def load(cls, path: str, model: nn.Module | str, device = "cpu") -> Type["CustomModel"]:
        parameters = torch.load(path, map_location = device)
        custom_model = cls(model = model)
        custom_model.model.load_state_dict(state_dict = parameters, strict = True)
        return custom_model