from torch import nn

class FashionMNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
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
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128) 
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