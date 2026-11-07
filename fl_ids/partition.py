"""Chia client theo khối (methodology 1.5). Test không chia (đánh giá toàn cục).

Tên cách chia: "iid", "dirichlet:<alpha>", "filepart".
Khối val đi theo cùng quy tắc với khối train để mỗi client có val của mình (val liên kết, mục 6.1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .windowing import WindowSet


@dataclass
class Client:
    id: int
    Xtr: np.ndarray
    ytr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    gva: np.ndarray            # group của cửa sổ val (để báo recall theo nhóm)
    tr_idx: np.ndarray         # chỉ số cửa sổ gốc trong WindowSet
    va_idx: np.ndarray

    @property
    def n(self) -> int:
        return len(self.ytr)

    @property
    def classes(self) -> np.ndarray:
        return np.unique(self.ytr)


def _split_even(items: list, K: int) -> list[list]:
    return [list(a) for a in np.array_split(np.array(items, dtype=object), K)] if items else [[] for _ in range(K)]


def assign_blocks(ws: WindowSet, scheme: str, K: int, rng: np.random.Generator,
                  pcfg: dict) -> dict[int, int]:
    """Trả về {block_id: client} cho mọi khối train và val."""
    sel = (ws.split == "train") | (ws.split == "val")
    blocks, first = np.unique(ws.block[sel], return_index=True)
    idx = np.flatnonzero(sel)[first]
    b_split = ws.split[idx]
    b_file = ws.file_idx[idx]
    files = ws.files
    assign: dict[int, int] = {}

    if scheme == "iid":
        for sp in ("train", "val"):
            bs = blocks[b_split == sp]
            bs = bs[rng.permutation(len(bs))]
            for k, part in enumerate(np.array_split(bs, K)):
                for b in part:
                    assign[int(b)] = k
        return assign

    if scheme.startswith("dirichlet"):
        alpha = float(scheme.split(":")[1]) if ":" in scheme else 0.5
        key = pcfg.get("dirichlet_by", "attack_type")
        cat = np.array([str(files[f][key]) if key != "label" else str(files[f]["label"]) for f in b_file])
        for c in np.unique(cat):
            p = rng.dirichlet(alpha * np.ones(K))
            for sp in ("train", "val"):
                bs = blocks[(cat == c) & (b_split == sp)]
                bs = bs[rng.permutation(len(bs))]
                cuts = (np.cumsum(p) * len(bs)).astype(int)[:-1]
                for k, part in enumerate(np.split(bs, cuts)):
                    for b in part:
                        assign[int(b)] = k
        return assign

    if scheme == "filepart":
        smap = pcfg.get("single_file_map", {})
        n_parts: dict[str, int] = {}
        for f in files:
            if f["split_src"] == "train":
                n_parts[f["attack_type"]] = n_parts.get(f["attack_type"], 0) + 1
        for sp in ("train", "val"):
            leftovers: dict[str, list[int]] = {}
            for b, fi in zip(blocks[b_split == sp], b_file[b_split == sp]):
                f = files[fi]
                at = f["attack_type"]
                if n_parts.get(at, 1) > 1:
                    assign[int(b)] = (f["part"] - 1) % K
                elif at in smap:
                    assign[int(b)] = int(smap[at]) % K
                else:                                   # Benign và kiểu một file khác: khối liên tiếp
                    leftovers.setdefault(at, []).append(int(b))
            for at, bs in leftovers.items():
                for k, part in enumerate(_split_even(sorted(bs), K)):
                    for b in part:
                        assign[int(b)] = k
        return assign

    raise ValueError(f"Cách chia không hỗ trợ: {scheme}")


def make_clients(ws: WindowSet, scheme: str, seed: int, pcfg: dict) -> list[Client]:
    K = pcfg["n_clients_filepart"] if scheme == "filepart" else pcfg["n_clients"]
    rng = np.random.default_rng(seed)
    assign = assign_blocks(ws, scheme, K, rng, pcfg)
    cl = np.full(len(ws.y), -1, np.int32)
    sel = ws.split != "test"
    cl[sel] = [assign.get(int(b), -1) for b in ws.block[sel]]
    groups = ws.group
    clients = []
    for k in range(K):
        tr = np.flatnonzero((cl == k) & (ws.split == "train"))
        va = np.flatnonzero((cl == k) & (ws.split == "val"))
        clients.append(Client(k, ws.X[tr], ws.y[tr].astype(int), ws.X[va], ws.y[va].astype(int),
                              groups[va], tr, va))
    return clients


def client_table(clients: list[Client]):
    import pandas as pd
    rows = []
    for c in clients:
        rows.append({"client": c.id, "train_benign": int((c.ytr == 0).sum()),
                     "train_attack": int((c.ytr == 1).sum()), "val_benign": int((c.yva == 0).sum()),
                     "val_attack": int((c.yva == 1).sum())})
    return pd.DataFrame(rows)
