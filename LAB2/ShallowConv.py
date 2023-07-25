import torch
import torch.nn as nn
import numpy as np


class ShallowConvNet(nn.Module):
    def __init__(self, activation_func, device) -> None:
        super(ShallowConvNet, self).__init__()
        self.device = device
        self.conv0 = nn.Conv2d(1, 40, kernel_size=(1, 13))
        self.conv1 = nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=(2, 1), bias=False),
            nn.BatchNorm2d(
                40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            activation_func,
        )
        self.pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7)), nn.Dropout(p=0.5)
        )
        self.classify = nn.Linear(4040, 2)

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.classify(x)
        return x
