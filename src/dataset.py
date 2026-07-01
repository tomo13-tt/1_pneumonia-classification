# 前処理（transform）と DataLoader 生成。

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# 3ch（転移学習）用の正規化に使う ImageNet 統計量
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

## Augmentationパイプライン
def make_transforms(channels):
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=channels),
        transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ] + ([transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)] if channels == 3 else []))

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=channels),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ] + ([transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)] if channels == 3 else []))
    return train_tf, eval_tf

## 訓練/検証データの再分割 + DataLoader生成
## data_dir の下に train/ test/ がある前提
def build_loaders(data_dir, channels, batch_size=16, train_limit=2500, seed=42):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_tf, eval_tf = make_transforms(channels)

    full_data = datasets.ImageFolder(train_dir, transform=train_tf)
    full_data_for_val = datasets.ImageFolder(train_dir, transform=eval_tf)
    test_data = datasets.ImageFolder(test_dir, transform=eval_tf)

    targets = [label for _, label in full_data.samples]
    indices = list(range(len(full_data)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=targets, random_state=seed)
    train_idx = train_idx[:train_limit]   # 学習時間短縮のため間引き (None で全件)

    train_data = Subset(full_data, train_idx)
    val_data = Subset(full_data_for_val, val_idx)

    train_dl = DataLoader(train_data, batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size, shuffle=False)
    test_dl = DataLoader(test_data, batch_size, shuffle=False)
    return train_dl, val_dl, test_dl, test_data
