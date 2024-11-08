import torch
import torch.nn as nn

# Squeeze-and-Excitation Block
class SE_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, int(in_channels/16))
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(int(in_channels/16), in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.pool(x)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.shape[0], out.shape[1], 1).expand_as(x)
        return out * x


# Residual Block
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, padding=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        dorate = 0.4
        self.downsample = downsample

        self.dropout1 = nn.Dropout1d(dorate)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout1d(dorate)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SE_Block(in_channels=planes)


    def forward(self, x):
        identity = x

        out = self.dropout1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.dropout2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        
        return out


# HAR用のsmall_resnetを定義
class ResNet1D_small(nn.Module):
    def __init__(self, block, layers, num_classes, num_channel):
        super(ResNet1D_small, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv1d(num_channel, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1a = self._make_layer(block, 64, layers[0], stride=1, padding=1)
        self.layer1b = self._make_layer(block, 64, layers[1], stride=1, padding=1)
        self.layer2a = self._make_layer(block, 128, layers[2], stride=2, padding=1)
        self.layer2b = self._make_layer(block, 128, layers[3], stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, padding=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                # nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, padding, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1a(x)
        x = self.layer1b(x)
        x = self.layer2a(x)
        x = self.layer2b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Human Activity Recognition Based on Residual Network
def SE_HAR(num_classes=4, num_channel=1):
    return ResNet1D_small(BasicBlock1D, [1, 1, 1, 1], num_classes, num_channel=num_channel)