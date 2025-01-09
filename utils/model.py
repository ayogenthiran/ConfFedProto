import torch
import torch.nn as nn
import torch.nn.functional as F

class Cifar10Net(nn.Module):
    def __init__(self) -> None:
        super(Cifar10Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        proto = self.fc1(x)
        proto = F.normalize(proto, p=2, dim=1)
        x = F.relu(self.fc2(self.dropout(proto)))
        x = self.fc3(x)
        return x, proto

class MnistNet(nn.Module):
    def __init__(self) -> None:
        super(MnistNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(-1, 16 * 4 * 4)
        proto = F.relu(self.fc1(x))
        x = F.relu(self.fc2(self.dropout(proto)))
        x = self.fc3(x)
        return x, proto