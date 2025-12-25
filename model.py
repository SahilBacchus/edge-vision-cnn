import torch
import torch.nn as nn

# TODO: 
# Implement depthwise separable convolution --> seems like a recurring theme when searching for edge optimized vision cnn
# use Depthwise sparable conv blocks to build our cnn 


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        # Depthwise convolution --> applies a single convolution per input channel (groups=in_channel)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False
        )

        # Pointwise convolution --> applies 1x1 convolution to combine outputs of depthwise conv
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): 
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class EdgeCNN(nn.Module): 
    def __init__(self, num_classes=10): 
        super().__init__()

        # Intial standard Convolution --> "stem" are usually the intial simple set of layers
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True)
        )

        # Depthwwsie seperable blocks
        self.features = nn.Sequential(
            # 32x32
            DepthwiseSeparableConv(32, 64, stride=1),

            # 16x16
            DepthwiseSeparableConv(64, 128, stride=2), 
            DepthwiseSeparableConv(128, 128, stride=1),

            # 8x8 
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1)
        )

        # Standard feed forward network to do classification based on features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), 
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

