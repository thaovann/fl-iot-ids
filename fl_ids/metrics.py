"""Độ đo nhị phân, chọn ngưỡng trên val liên kết (methodology 6.1, 7, 10.11)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

TAU_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def confusion(y, pred) -> np.ndarray:
    """[[TN, FP], [FN, TP]]"""
    cm = np.zeros((2, 2), np.int64)
    np.add.at(cm, (np.asarray(y, int), np.asarray(pred, int)), 1)
    return cm


def from_confusion(cm: np.ndarray) -> dict:
    tn, fp, fn, tp = cm.ravel().astype(float)
    def div(a, b):
        return a / b if b else 0.0
    p1, r1 = div(tp, tp + fp), div(tp, tp + fn)
    p0, r0 = div(tn, tn + fn), div(tn, tn + fp)
    f1_1 = div(2 * p1 * r1, p1 + r1)
    f1_0 = div(2 * p0 * r0, p0 + r0)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "acc": div(tp + tn, cm.sum()), "macro_f1": (f1_0 + f1_1) / 2,
        "macro_precision": (p0 + p1) / 2, "macro_recall": (r0 + r1) / 2,
        "attack_recall": r1, "fpr": div(fp, fp + tn), "mcc": div(tp * tn - fp * fn, mcc_den),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def fed_val_threshold(prob_by_client: list[np.ndarray], y_by_client: list[np.ndarray]) -> tuple[float, float]:
    """Mỗi client gửi ma trận nhầm lẫn cho từng τ; server cộng và chọn τ làm macro-F1 lớn nhất."""
    tot = np.zeros((len(TAU_GRID), 2, 2), np.int64)
    for p, y in zip(prob_by_client, y_by_client):
        if len(y) == 0:
            continue
        for i, t in enumerate(TAU_GRID):
            tot[i] += confusion(y, p >= t)
    f1 = np.array([from_confusion(cm)["macro_f1"] for cm in tot])
    if tot.sum() == 0:
        return 0.5, float("nan")
    i = int(np.argmax(f1 - 1e-9 * np.abs(TAU_GRID - 0.5)))   # hoà thì gần 0,5 nhất
    return float(TAU_GRID[i]), float(f1[i])


def fed_val_f1(prob_by_client, y_by_client, tau=0.5) -> float:
    cm = sum((confusion(y, p >= tau) for p, y in zip(prob_by_client, y_by_client) if len(y)),
             np.zeros((2, 2), np.int64))
    return from_confusion(cm)["macro_f1"] if cm.sum() else float("nan")


def evaluate(y, prob, tau: float, groups: np.ndarray | None = None, min_group: int = 50) -> dict:
    y = np.asarray(y, int)
    res = from_confusion(confusion(y, prob >= tau))
    res["tau"] = tau
    if len(np.unique(y)) == 2:
        res["auc_pr"] = float(average_precision_score(y, prob))
        res["roc_auc"] = float(roc_auc_score(y, prob))
    if groups is not None:
        for g in np.unique(groups[y == 1]):
            m = (groups == g) & (y == 1)
            if m.sum() >= min_group:
                res[f"recall_{g}"] = float((prob[m] >= tau).mean())
    return res
