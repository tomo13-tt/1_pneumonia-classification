# Grad-CAM：モデルが画像のどこを見て判断したかを可視化。

## 外部ライブラリは使わず、forward/backward フックでスクラッチ実装している。
## 3モデル(Baseline / ResNet / EfficientNet)に対応:
## - ResNet         -> model.layer4[-1]
## - EfficientNet   -> model.features[-1]
## - Baseline(Seq)  -> 最後の Conv2d 層


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _to_display(img):
    arr = img.cpu().numpy()
    if arr.shape[0] == 3:
        arr = arr * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
        arr = np.clip(np.transpose(arr, (1, 2, 0)), 0, 1)
    else:
        arr = arr.squeeze()
    return arr


def get_target_layer(model):
    """モデル種別に応じて Grad-CAM 対象の最終畳み込み層を返す。"""
    if hasattr(model, "layer4"):        # ResNet
        return model.layer4[-1]
    if hasattr(model, "features"):      # EfficientNet など
        return model.features[-1]
    last = None                         # nn.Sequential (Baseline)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def compute_gradcam(model, img, device, target_layer=None, target_class=None):
    """1枚分の Grad-CAM ヒートマップ(0-1)と予測クラスを返す。"""
    model.eval()
    if target_layer is None:
        target_layer = get_target_layer(model)

    store = {}
    h1 = target_layer.register_forward_hook(
        lambda m, i, o: store.__setitem__("act", o.detach()))
    h2 = target_layer.register_full_backward_hook(
        lambda m, gi, go: store.__setitem__("grad", go[0].detach()))

    x = img.unsqueeze(0).to(device)
    out = model(x)
    if target_class is None:
        target_class = int(out.argmax(dim=1).item())
    model.zero_grad()
    out[0, target_class].backward()

    act = store["act"][0]               # (C,H,W)
    grad = store["grad"][0]             # (C,H,W)
    weights = grad.mean(dim=(1, 2))     # (C,) チャンネルごとの重要度
    cam = torch.relu((weights[:, None, None] * act).sum(dim=0))
    cam = cam / (cam.max() + 1e-8)

    h1.remove(); h2.remove()
    return cam.cpu().numpy(), target_class


def show_gradcam(model, test_data, indices, device, class_names, target_layer=None):
    """指定した複数画像について、元画像と Grad-CAM 重ね合わせを並べて表示。"""
    n = len(indices)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4))
    axes = np.array(axes).reshape(2, n)
    for col, idx in enumerate(indices):
        img, label = test_data[idx]
        cam, pred = compute_gradcam(model, img, device, target_layer)
        disp = _to_display(img)

        cam_t = torch.tensor(cam)[None, None]
        cam_up = F.interpolate(cam_t, size=disp.shape[:2],
                               mode="bilinear", align_corners=False)[0, 0].numpy()

        axes[0, col].imshow(disp, cmap="gray")
        axes[0, col].set_title(f"True:{class_names[label]} / Pred:{class_names[pred]}", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(disp, cmap="gray")
        axes[1, col].imshow(cam_up, cmap="jet", alpha=0.5)
        axes[1, col].set_title("Grad-CAM", fontsize=9)
        axes[1, col].axis("off")
    plt.tight_layout(); plt.show()
