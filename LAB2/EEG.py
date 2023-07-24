import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, activation_func, device):
        super(EEGNet, self).__init__()
        self.device = device
        self.firstconv = nn.Sequential(
            nn.Conv2d(
                1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False
            ),
            nn.BatchNorm2d(
                16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(
                32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            activation_func,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25),
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(
                32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False
            ),
            nn.BatchNorm2d(
                32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            activation_func,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25),
        )
        self.classify = nn.Linear(736, 2)

    def forward(self, x):
        x = x.to(self.device)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.shape[0], -1)
        x = self.classify(x)
        return x
