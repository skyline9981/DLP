import torch
import torch.nn as nn
import numpy as np


class DeepConvNet(nn.Module):
    def __init__(self, activation_func, device) -> None:
        super(DeepConvNet, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1, 1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1)),
            nn.BatchNorm2d(
                25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            activation_func,
            nn.MaxPool2d(
                kernel_size=(1, 2),
                stride=(1, 2),
                padding=0,
                dilation=1,
                ceil_mode=False,
            ),
            nn.Dropout(p=0.5),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1)),
            nn.BatchNorm2d(
                50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            activation_func,
            nn.MaxPool2d(
                kernel_size=(1, 2),
                stride=(1, 2),
                padding=0,
                dilation=1,
                ceil_mode=False,
            ),
            nn.Dropout(p=0.5),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1)),
            nn.BatchNorm2d(
                100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            activation_func,
            nn.MaxPool2d(
                kernel_size=(1, 2),
                stride=(1, 2),
                padding=0,
                dilation=1,
                ceil_mode=False,
            ),
            nn.Dropout(p=0.5),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1)),
            nn.BatchNorm2d(
                200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            activation_func,
            nn.MaxPool2d(
                kernel_size=(1, 2),
                stride=(1, 2),
                padding=0,
                dilation=1,
                ceil_mode=False,
            ),
            nn.Dropout(p=0.5),
        )
        self.classifier = nn.Linear(8600, 2)

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
