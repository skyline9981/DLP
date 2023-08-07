import torch
import torch.nn as nn

"""
    ResNet
    this is a simple ResNet model
    include resnet18, resnet34, resnet50, resnet101, resnet152
"""


class Block(nn.Module):
    """
    ## Block
    this is a simple block of ResNet
    args:
        - in_channel: 3
        - out_channel: 64
        - i_downsample: None
        - stride: 1
    """

    expansion = 1

    def __init__(self, in_channel, out_channel, i_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 1
        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.i_downsample = i_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    """
    ## Bottleneck
    This is a simple Bottleneck of ResNet
    args:
        - in_channel: 3
        - out_channel: 64
        - i_downsample: None
        - stride: 1
    """

    expansion = 4

    def __init__(self, in_channel, out_channel, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(
            out_channel,
            out_channel * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU()
        self.i_downsample = i_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    """
    ## ResNet
    This is a simple ResNet
    When you use this model, you can use ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    args:
        - block: Block or Bottleneck
        - layers: [2, 2, 2, 2] or [3, 4, 6, 3] or [3, 8, 36, 3]
        - image_channel: 3
        - num_classes: 2
    """

    def __init__(self, block, layers, image_channel, num_classes):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(image_channel, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], out_channel=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channel=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channel=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channel=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_residual_blocks, out_channel, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channel != out_channel * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channel,
                    out_channel * block.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channel * block.expansion),
            )

        layers.append(block(self.in_channel, out_channel, identity_downsample, stride))
        self.in_channel = out_channel * block.expansion

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channel, out_channel))

        return nn.Sequential(*layers)

    def ResNet18(img_channel=3, num_classes=2):
        return ResNet(Block, [2, 2, 2, 2], img_channel, num_classes)

    def ResNet50(img_channel=3, num_classes=2):
        return ResNet(Bottleneck, [3, 4, 6, 3], img_channel, num_classes)

    def ResNet152(img_channel=3, num_classes=2):
        return ResNet(Bottleneck, [3, 8, 36, 3], img_channel, num_classes)
