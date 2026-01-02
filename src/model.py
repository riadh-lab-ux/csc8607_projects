"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""

import torch
import torch.nn as nn


class InceptionLikeModule(nn.Module):
    """
    Module multibranches :
      - branche 1 : Conv 1x1 -> ReLU
      - branche 2 : Conv 3x3 (padding=1) -> ReLU
      - branche 3 : MaxPool 3x3 (stride=1, padding=1) -> Conv 1x1 -> ReLU
    Concat (channels) puis BatchNorm.
    """
    def __init__(self, in_channels: int, b1: int, b2: int, b3: int):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, b1, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, b2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, b3, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        out_channels = b1 + b2 + b3
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.bn(out)
        return out


class InceptionTinyNet(nn.Module):
    """
    - Stem : Conv 3x3 -> 64 canaux, padding=1
    - M modules Inception-like (sortie = 192 canaux)
    - MaxPool 2x2 après chaque 2 modules
      * M=4 : après module 2
      * M=6 : après modules 2 et 4
    - GAP -> Linear(192 -> num_classes)
    """
    def __init__(self, num_classes: int, num_modules: int, branch_channels: tuple[int, int, int]):
        super().__init__()

        b1, b2, b3 = branch_channels
        assert num_modules in (4, 6), "num_modules doit être 4 ou 6"
        assert (b1 + b2 + b3) == 192, "branch_channels doit sommer à 192"
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        layers = []
        in_ch = 64
        for i in range(num_modules):
            layers.append(InceptionLikeModule(in_ch, b1, b2, b3))
            in_ch = 192

            # Pool after module 2, and after module 4 if M=6
            if i == 1 or (num_modules == 6 and i == 3):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(192, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.gap(x)           # (B, 192, 1, 1)
        x = torch.flatten(x, 1)   # (B, 192)
        x = self.classifier(x)    # (B, num_classes)
        return x

def build_model(config: dict) -> nn.Module:

    mcfg = config["model"]
    num_classes = int(mcfg.get("num_classes", 200))
    num_modules = int(mcfg["num_modules"])
    bc = mcfg["branch_channels"]
    branch_channels = (int(bc[0]), int(bc[1]), int(bc[2]))
    return InceptionTinyNet(
        num_classes=num_classes,
        num_modules=num_modules,
        branch_channels=branch_channels,
    )
