import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class AlexNetCIFAR(nn.Module):
    """
    适配 32x32 CIFAR 的 AlexNet 变体（更稳）：
    - 特征：Conv + BN + ReLU，32→16→8→4 的下采样
    - 分类头：4096→1024→1024→num_classes（较原版4096两层更稳/更省）
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),            # 32 -> 16

            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),            # 16 -> 8

            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),            # 8 -> 4
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # 输出 [B, 256, 4, 4]，展平后 4096
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)           # [B, 256, 4, 4]
        x = torch.flatten(x, 1)       # [B, 4096]
        x = self.classifier(x)        # [B, C]
        return x


# class AlexNetCIFAR(nn.Module):
#     """
#     适配 32x32 CIFAR 输入的 AlexNet 变体：
#     特征：Conv(64)-Pool-Conv(192)-Pool-Conv(384)-Conv(256)-Conv(256)-Pool
#     32→16→8→4 的下采样；最后展平到 256*4*4=4096
#     分类头：4096→4096→num_classes（保留经典的两层大 FC）
#     """
#     def __init__(self, num_classes: int = 10):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),            # 32 -> 16

#             nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),            # 16 -> 8

#             nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),            # 8 -> 4
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # 保底，兼容性更好
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 4 * 4, 4096, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes, bias=True),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)           # [B, 256, 4, 4]
#         x = torch.flatten(x, 1)       # [B, 4096]
#         x = self.classifier(x)        # [B, C]
#         return x


# ===== VGG16 for CIFAR (BN + GAP head) =====
class VGG16CIFAR(nn.Module):
    """
    适配 CIFAR 的 VGG16-BN：
      - 输入 32x32
      - 特征层：经典 VGG16 的 13 个 conv（全部 3x3, stride=1, padding=1）+ BN + ReLU
      - 下采样：MaxPool 2x2, stride=2, 共 5 次（32→16→8→4→2→1）
      - 头：GAP (AdaptiveAvgPool2d(1)) → Linear(512, num_classes)
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        cfg = [64, 64, 'M',
               128, 128, 'M',
               256, 256, 256, 'M',
               512, 512, 512, 'M',
               512, 512, 512, 'M']
        layers = []
        in_ch = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_ch, v, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        head = [nn.Flatten(),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(512, num_classes, bias=True)]
        self.classifier = nn.Sequential(*head)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)     # [B, 512, 1, 1]
        x = self.classifier(x)  # [B, C]
        return x


# ===== LeNet for CIFAR (32x32) =====
class LeNetCIFAR(nn.Module):
    """
    经典 LeNet 结构的 CIFAR 版：
      Conv(6, 5x5) → ReLU → AvgPool(2) →
      Conv(16, 5x5) → ReLU → AvgPool(2) →
      Conv(120, 5x5 valid) → ReLU →
      FC(84) → ReLU → FC(num_classes)
    32x32 输入到最后有效卷积会把 5x5 的有效大小对齐（中间尺寸：32→28→14→10→5→1）。
    """
    def __init__(self, num_classes: int = 10, use_bn: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(6) if use_bn else nn.Identity()
        self.pool = nn.AvgPool2d(2, 2)   # LeNet 原版是平均池化
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(16) if use_bn else nn.Identity()
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)  # valid conv -> 1x1
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))   # 32->28->14
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))   # 14->10->5
        x = torch.relu(self.conv3(x))                        # 5->1
        x = x.view(x.size(0), -1)                            # [B,120]
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
