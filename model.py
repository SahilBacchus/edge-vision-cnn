import torch
import torch.nn as nn

# TODO: 
# Implement depthwise separable convolution --> seems like a recurring theme when searching for edge optimized vision cnn
# use Depthwise sparable conv blocks to build our cnn 


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        # Depthwise convolution --> applies a single convolution per input channel (groups=in_channel)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, stride=stride, padding=1, groups=in_channels, bias=False
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