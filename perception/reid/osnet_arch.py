"""
Standalone OSNet-x0.25 implementation — no torchreid dependency.
Architecture matches torchreid exactly so pretrained .pth weights load cleanly.
Reference: Zhou et al., "Omni-Scale Feature Learning for Person Re-Identification" (ICCV 2019).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------

class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0, groups=1, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                              padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.IN = nn.InstanceNorm2d(out_ch, affine=True) if IN else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.IN is not None:
            x = self.IN(x)
        return self.relu(x)


class Conv1x1(ConvLayer):
    def __init__(self, in_ch, out_ch, groups=1):
        super().__init__(in_ch, out_ch, 1, groups=groups)


class Conv1x1Linear(nn.Module):
    """1x1 conv + BN, no ReLU — used in residual paths."""
    def __init__(self, in_ch, out_ch, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """Lightweight 3x3: pointwise + depthwise."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    """Channel-wise attention gate."""
    def __init__(self, in_ch):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, in_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.gap(x).view(x.size(0), -1)
        w = self.sigmoid(self.fc(w))
        return x * w.unsqueeze(2).unsqueeze(3)


class OSBlock(nn.Module):
    """Omni-scale block with 4 receptive field scales and channel gating."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 4
        self.conv1 = Conv1x1(in_ch, mid)

        # 4 branches with increasing receptive fields
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid),
                                    LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid),
                                    LightConv3x3(mid, mid), LightConv3x3(mid, mid))

        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_ch)
        self.downsample = Conv1x1Linear(in_ch, out_ch) if in_ch != out_ch else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = (self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) +
               self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1)))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity, inplace=True)


# ------------------------------------------------------------------
# OSNet
# ------------------------------------------------------------------

class OSNet(nn.Module):
    """
    OSNet with configurable width multiplier.
    For x0.25: channels=[16, 64, 96, 128].
    feature_dim=512 always (final embedding size).
    """

    def __init__(self, num_classes, channels, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = self._make_layer(channels[0], channels[1], num_blocks=2)
        self.pool2 = nn.Sequential(Conv1x1(channels[1], channels[1]),
                                   nn.AvgPool2d(2, stride=2))

        self.conv3 = self._make_layer(channels[1], channels[2], num_blocks=2)
        self.pool3 = nn.Sequential(Conv1x1(channels[2], channels[2]),
                                   nn.AvgPool2d(2, stride=2))

        self.conv4 = self._make_layer(channels[2], channels[3], num_blocks=2)
        self.conv5 = Conv1x1(channels[3], feature_dim)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

        self._init_weights()

    def _make_layer(self, in_ch, out_ch, num_blocks):
        layers = [OSBlock(in_ch, out_ch)]
        for _ in range(num_blocks - 1):
            layers.append(OSBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feats=False):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        v = self.gap(x).view(x.size(0), -1)
        v = self.fc(v)
        return v  # 512-d embedding (classifier not used for ReID inference)


def osnet_x0_25(num_classes=1041):
    """OSNet with 0.25x channel width. num_classes=1041 matches MSMT17 pretrained weights."""
    return OSNet(num_classes=num_classes, channels=[16, 64, 96, 128], feature_dim=512)
