import torch
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader

class CIFAR100_ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights = None, num_classes = 100)
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
        self.train()
        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in calibration_loader:
                images = batch["image"].to(device)
                _ = self(images)