# 評価・誤分類分析・推論速度/モデルサイズ
## evaluate() は指標に加えて、各サンプルの予測確率も返す。
## それを使って次の2種類の誤分類分析ができる:
## - show_misclassified_by_type : 誤りの方向別(見逃しFN / 過検出FP)に表示
## - show_confidence_cases      : 確信度別(高確率なのに誤り / 確率が低く迷っている)に表示


import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

# 表示時の逆正規化に使う (3ch 転移学習の入力を見やすく戻すため)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _to_display(img):
    """画像テンソル (C,H,W) を matplotlib 表示用の配列に変換。

    3ch は ImageNet 正規化を逆算して 0-1 に戻す。1ch は squeeze。
    """
    arr = img.cpu().numpy()
    if arr.shape[0] == 3:
        arr = arr * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
        arr = np.clip(np.transpose(arr, (1, 2, 0)), 0, 1)
    else:
        arr = arr.squeeze()
    return arr

## テストデータの評価
def evaluate(model, test_dl, device, class_names, title=""):
    """test_dl で評価。指標を表示・混同行列を描画し、結果を dict で返す。

    返り値の dict には y_true / y_pred / conf(予測クラスの確率) /
    p_pos(PNEUMONIA=陽性の確率) が入り、誤分類分析に使える。
    """
    model.eval()
    y_true, y_pred, conf, p_pos = [], [], [], []
    with torch.no_grad():
        for x_batch, y_batch in test_dl:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            prob = torch.softmax(logits, dim=1).cpu()
            pred = prob.argmax(dim=1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(pred.numpy())
            conf.extend(prob.max(dim=1).values.numpy())
            p_pos.extend(prob[:, 1].numpy())

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    conf = np.array(conf); p_pos = np.array(p_pos)

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f'[{title}] Accuracy : {acc:.4f}')
    print(f'[{title}] Recall   : {rec:.4f}')
    print(f'[{title}] Precision: {prec:.4f}')
    print(f'[{title}] F1-score : {f1:.4f}')
    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=class_names, yticklabels=class_names,
           xlabel='Predicted', ylabel='True', title=f'{title} Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout(); plt.show()

    return {'accuracy': acc, 'recall': rec, 'precision': prec, 'f1': f1,
            'y_true': y_true, 'y_pred': y_pred, 'conf': conf, 'p_pos': p_pos}


def _grid(test_data, indices, captions, suptitle, cols=3):
    """指定インデックスの画像を格子状に表示する補助関数。"""
    if len(indices) == 0:
        print(f'[{suptitle}] 該当画像なし'); return
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, idx, cap in zip(axes, indices, captions):
        img, _ = test_data[idx]
        ax.imshow(_to_display(img), cmap='gray')
        ax.set_title(cap, color='red', fontsize=10)
        ax.axis('off')
    for ax in axes[len(indices):]:
        ax.axis('off')
    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout(); plt.show()


def show_misclassified_by_type(test_data, result, class_names, n=6):
    """誤分類を「方向別」に表示する。

    見逃し (False Negative): True=PNEUMONIA を NORMAL と誤判定 → 医療的に最重要
    過検出 (False Positive): True=NORMAL を PNEUMONIA と誤判定
    """
    y_true, y_pred, p_pos = result['y_true'], result['y_pred'], result['p_pos']

    fn = np.where((y_true == 1) & (y_pred == 0))[0]   # 見逃し
    fp = np.where((y_true == 0) & (y_pred == 1))[0]   # 過検出

    caps_fn = [f'True:PNEUMONIA / Pred:NORMAL\np(PNEU)={p_pos[i]:.2f}' for i in fn[:n]]
    _grid(test_data, fn[:n], caps_fn, '見逃し False Negative (肺炎を見落とし)')

    caps_fp = [f'True:NORMAL / Pred:PNEUMONIA\np(PNEU)={p_pos[i]:.2f}' for i in fp[:n]]
    _grid(test_data, fp[:n], caps_fp, '過検出 False Positive (正常を肺炎と誤判定)')


def show_confidence_cases(test_data, result, class_names, n=6):
    """誤分類を「確信度別」に表示する。

    確信誤り: 予測確率が高い(自信満々)のに間違えた例 → モデルが誤って確信
    迷い    : 予測確率が 0.5 付近で迷っている例 (正解/不正解どちらも含む)
    """
    y_true, y_pred, conf = result['y_true'], result['y_pred'], result['conf']
    wrong = np.where(y_true != y_pred)[0]

    # 高確率なのに誤り: wrong を conf 降順
    confident_wrong = wrong[np.argsort(-conf[wrong])][:n]
    caps_cw = [f'T:{class_names[y_true[i]]} P:{class_names[y_pred[i]]}\nconf={conf[i]:.2f}'
               for i in confident_wrong]
    _grid(test_data, confident_wrong, caps_cw, '確信していたのに誤り (high-conf wrong)')

    # 迷っている: conf が小さい(0.5 に近い)順。正誤両方含む
    uncertain = np.argsort(conf)[:n]
    caps_un = [f'T:{class_names[y_true[i]]} P:{class_names[y_pred[i]]}\nconf={conf[i]:.2f}'
               + ('  ✗' if y_true[i] != y_pred[i] else '  ✓') for i in uncertain]
    _grid(test_data, uncertain, caps_un, '確率が低く迷っている例 (uncertain)')


# 推論速度
def measure_speed(model, test_dl, device, n_batches=20):
    model.eval()
    with torch.no_grad():
        for x, _ in test_dl:
            model(x.to(device)); break  # ウォームアップ
    total_t = total_n = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(test_dl):
            x = x.to(device)
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time(); model(x)
            if device.type == 'cuda': torch.cuda.synchronize()
            total_t += time.time() - t0; total_n += x.size(0)
            if i + 1 >= n_batches: break
    return (total_t / total_n) * 1000  # ms/枚

## モデルサイズ
def model_stats(model):
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    return n_params, size_mb
