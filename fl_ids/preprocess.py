"""Tiền xử lý liên kết (methodology mục 3, định hướng 05-06/11 mục 3.2).

Mọi thống kê đều tính ở client rồi gộp ở server; không có bước nào nhìn train gộp.
- Bỏ cột hằng: client gửi (min, max); server bỏ cột có min = max trên toàn cục.
- Chuẩn hoá: client gửi (n, mean, M2); server gộp bằng công thức Chan -> FedScaler.
- Trọng số lớp: client gửi số cửa sổ mỗi lớp; w_c = N / (C · n_c).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .partition import Client


def chan_merge(stats: list[tuple[int, np.ndarray, np.ndarray]]):
    n, mean, M2 = 0, None, None
    for nk, mk, M2k in stats:
        if nk == 0:
            continue
        if mean is None:
            n, mean, M2 = nk, mk.copy(), M2k.copy()
            continue
        delta = mk - mean
        tot = n + nk
        mean = mean + delta * nk / tot
        M2 = M2 + M2k + delta ** 2 * n * nk / tot
        n = tot
    return n, mean, M2


@dataclass
class FedScaler:
    keep: np.ndarray          # mặt nạ cột giữ lại (trên D gốc)
    mean: np.ndarray
    std: np.ndarray
    gmin: np.ndarray          # min/max toàn cục sau chuẩn hoá (cho χ²)
    gmax: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X[:, self.keep] - self.mean) / self.std).astype(np.float32)


def fit_fed_scaler(clients: list[Client]) -> FedScaler:
    mins = [c.Xtr.min(0) for c in clients if c.n]
    maxs = [c.Xtr.max(0) for c in clients if c.n]
    gmin, gmax = np.min(mins, 0), np.max(maxs, 0)
    stats = [(c.n, c.Xtr.mean(0).astype(np.float64), ((c.Xtr - c.Xtr.mean(0)) ** 2).sum(0).astype(np.float64))
             for c in clients if c.n]
    n, mean, M2 = chan_merge(stats)
    std = np.sqrt(M2 / max(n, 1))
    keep = (gmax > gmin) & (std > 1e-8)
    m, s = mean[keep], std[keep]
    return FedScaler(keep, m.astype(np.float32), s.astype(np.float32),
                     ((gmin[keep] - m) / s).astype(np.float32), ((gmax[keep] - m) / s).astype(np.float32))


def apply_scaler(clients: list[Client], sc: FedScaler) -> list[Client]:
    for c in clients:
        c.Xtr = sc.transform(c.Xtr) if c.n else np.empty((0, int(sc.keep.sum())), np.float32)
        c.Xva = sc.transform(c.Xva) if len(c.yva) else np.empty((0, int(sc.keep.sum())), np.float32)
    return clients


def fed_class_weights(clients: list[Client], n_classes: int = 2) -> np.ndarray:
    counts = np.zeros(n_classes)
    for c in clients:
        counts += np.bincount(c.ytr, minlength=n_classes)
    N = counts.sum()
    w = np.where(counts > 0, N / (n_classes * np.maximum(counts, 1)), 0.0)
    return w.astype(np.float32)
