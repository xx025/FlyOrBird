import torch
import torchvision
from torch import nn
from torchvision.models import ResNet18_Weights

from device import device


class FlyOrBird(nn.Module):
    def __init__(self):
        super(FlyOrBird, self).__init__()
        self.model = nn.Sequential()

        self.model.append(  # resnet18
            torchvision.models.resnet18(
                # weights=ResNet18_Weights.IMAGENET1K_V1 # 使用预训练权重
            )
        )
        self.model.append(nn.Linear(1000, 2))
        # softmax
        # self.model.append(nn.Softmax(dim=1)) # CrossEntropyLoss()包含了softmax

    def forward(self, x):
        return self.model(x)


model = FlyOrBird().to(device=device)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 损失函数
criterion = nn.CrossEntropyLoss().to(device=device)
