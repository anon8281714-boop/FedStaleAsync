import torch
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader
from asyncflower.model.resnet import ResNet8

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, img_batch):
        x = self.relu(self.conv1(img_batch))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CIFAR10_ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights = None, num_classes = 10)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = nn.Identity()
        self._ignored_parameters = set([name for name, value in self.resnet.named_buffers()])

    def forward(self, img_batch):
        return self.resnet(img_batch)
    
    def ignored_parameters(self) -> set[str]:
        return self._ignored_parameters
    
    def recalibrate_buffers(self, calibration_loader: DataLoader) -> None:
        """Recalibrates the buffers (running_mean and running_var) of all BatchNorm layers of the model using the provided calibration/train DataLoader. Does not compute gradients.""" 
        print("Recalibrating buffers for the global model in the central server...")
        self.train()
        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in calibration_loader:
                images = batch["image"].to(device)
                _ = self(images)

class CIFAR10_ResNet8(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet8()
        self._ignored_parameters = set([name for name, value in self.resnet.named_buffers()])

    def forward(self, img_batch):
        return self.resnet(img_batch)
    
    def ignored_parameters(self) -> set[str]:
        return self._ignored_parameters
    
    def recalibrate_buffers(self, calibration_loader: DataLoader) -> None:
        """Recalibrates the buffers (running_mean and running_var) of all BatchNorm layers of the model using the provided calibration/train DataLoader. Does not compute gradients.""" 
        print("Recalibrating buffers for the global model in the central server...")
        self.train()
        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in calibration_loader:
                images = batch["image"].to(device)
                _ = self(images)