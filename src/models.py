"""モデル定義と、転移学習の freeze / unfreeze ヘルパー。

  build_baseline        : 自作 CNN（BatchNorm 付き, 1ch 入力）
  build_resnet18        : ImageNet 事前学習 ResNet18（3ch 入力）
  build_efficientnet_b0 : ImageNet 事前学習 EfficientNet-B0（3ch 入力）

ResNet の2段階学習用:
  freeze_backbone_resnet : 全層を凍結し、最終層(fc)だけ学習対象にする（特徴抽出）
  unfreeze_resnet_later  : 後半の層(layer4 など)を解凍して fine-tune できるようにする
"""

import torch.nn as nn
from torchvision import models


# Baseline CNN: 1ch 入力, 224x224
def build_baseline(num_classes=2):
    def block(c_in, c_out, pool):
        return [nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out), nn.ReLU(), pool]
    layers = (block(1, 32, nn.MaxPool2d(2))
            + block(32, 64, nn.MaxPool2d(2))
            + block(64, 128, nn.MaxPool2d(2))
            + block(128, 256, nn.AvgPool2d(2))
            + block(256, 512, nn.AvgPool2d(14)))
    layers += [nn.Flatten(), nn.Linear(512, 256), nn.ReLU(),
               nn.Dropout(0.5), nn.Linear(256, num_classes)]
    return nn.Sequential(*layers)


# 転移学習: 事前学習済み (3ch 入力), 最終層を2クラスに差し替え
def build_resnet18(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes=2):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# ---- ResNet 2段階学習用ヘルパー ----
def freeze_backbone_resnet(model):
    """全層を凍結し、最終層(fc)だけ学習対象にする（特徴抽出フェーズ）。"""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    return model


def unfreeze_resnet_later(model, layers=("layer4",)):
    """指定した後半層 + fc を解凍する（fine-tune フェーズ）。

    layers 例: ("layer4",) や ("layer3", "layer4")
    """
    for name in layers:
        for p in getattr(model, name).parameters():
            p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
    return model
