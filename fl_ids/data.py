"""Đọc CSV CICFlowMeter, gán nhãn theo tên file, chia train/val/test theo file, cắt khối.

Quy ước (methodology mục 1 và 10.5):
- Nhãn nhị phân theo tên file: Benign_* = 0, còn lại = 1. `group`, `attack_type` chỉ là metadata.
- Sắp xếp theo Timestamp trong file rồi mới bỏ cột định danh.
- Test = file *_test. Val không bao giờ lấy từ file test.
- Kiểu tấn công nhiều phần (ICMP1..8): phần số lớn nhất làm val. Kiểu một file: 20% cuối
  theo thời gian làm val, có khoảng đệm FW dòng.
- Cửa sổ chỉ cắt bên trong một khối, nên train/val/test không chung dòng nào.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_RULES = [
    ("Benign", "Benign"),
    ("ARP_Spoofing", "Spoofing"),
    ("MQTT-", "MQTT"),
    ("Recon-", "Recon"),
    ("TCP_IP-DDoS-", "DDoS"),
    ("TCP_IP-DoS-", "DoS"),
]

# Tên (đã chuẩn hoá) không bao giờ được là đặc trưng. Lỗi Window_Index (báo cáo 05-06/11) là lý do có kiểm tra này.
META_NAMES = {
    "flow_id", "src_ip", "src_port", "dst_ip", "dst_port", "timestamp", "label",
    "window_index", "window_id", "block_id", "file_id", "file", "split", "client",
    "attack_type", "group", "part", "row", "index", "unnamed_0",
}
META_PATTERN = re.compile(r"(^|_)(index|idx|id|timestamp|window|offset|pos|row)(_|$)")


def norm_name(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", s.strip().lower()).strip("_")


def assert_no_metadata(feature_names: list[str]) -> None:
    """Dừng ngay nếu một cột metadata/định danh lọt vào đặc trưng."""
    bad = []
    for f in feature_names:
        base = norm_name(f.split("__")[0])
        if base in META_NAMES or META_PATTERN.search(base):
            bad.append(f)
    if bad:
        raise ValueError(f"Cột metadata lọt vào đặc trưng: {bad[:10]} (tổng {len(bad)})")


@dataclass
class FileInfo:
    path: str
    stem: str
    attack_type: str
    group: str
    label: int
    part: int
    split_src: str            # "train" | "test" theo tên file / thư mục


def parse_name(path: str | Path) -> FileInfo:
    p = Path(path)
    stem = p.name
    for suf in (".csv", "_Flow", ".pcap"):
        if stem.lower().endswith(suf.lower()):
            stem = stem[: -len(suf)]
    split_src = ""
    m = re.search(r"[_-](train|test)$", stem, flags=re.I)
    if m:
        split_src = m.group(1).lower()
        base = stem[: m.start()]
    else:
        base = stem
        for parent in p.parents:
            if parent.name.lower() in ("train", "test"):
                split_src = parent.name.lower()
                break
    if not split_src:
        raise ValueError(f"Không xác định được train/test cho {p}")
    mp = re.search(r"(\d+)$", base)
    part = int(mp.group(1)) if mp else 1
    attack_type = base[: mp.start()] if mp else base
    attack_type = attack_type.rstrip("_-")
    group = next((g for pre, g in GROUP_RULES if attack_type.startswith(pre)), "Other")
    label = 0 if group == "Benign" else 1
    return FileInfo(str(p), stem, attack_type, group, label, part, split_src)


def discover_files(root: str | Path, pattern: str = "**/*.csv") -> list[FileInfo]:
    files = sorted(Path(root).glob(pattern))
    if not files:
        raise FileNotFoundError(f"Không thấy CSV nào trong {root} (pattern {pattern})")
    return [parse_name(f) for f in files]


def load_file(path: str, id_cols: list[str], label_col: str = "Label",
              drop_features: list[str] = (), dayfirst: bool = True) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    if "Timestamp" in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=dayfirst)
        bad = ts.isna().mean()
        if bad > 0.01:
            warnings.warn(f"{Path(path).name}: {bad:.1%} Timestamp không parse được, "
                          f"ví dụ {df['Timestamp'].head(3).tolist()}")
        df = df.assign(_ts=ts, _row=np.arange(len(df)))
        df = df.sort_values(["_ts", "_row"], kind="mergesort")
    drop = list(id_cols) + [label_col, "_ts", "_row"] + list(drop_features)
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.to_numpy(np.float32), list(df.columns)


def take_segments(X: np.ndarray, cap: int, n_seg: int) -> list[np.ndarray]:
    """Lấy n_seg đoạn liên tiếp rải đều trong file (không lấy ngẫu nhiên từng dòng)."""
    n = len(X)
    if cap <= 0 or n <= cap:
        return [X]
    L = cap // n_seg
    starts = np.linspace(0, n - L, n_seg).astype(int)
    return [X[s:s + L] for s in starts]


def to_blocks(segments: list[np.ndarray], block_size: int, min_len: int) -> list[np.ndarray]:
    out = []
    for seg in segments:
        for s in range(0, len(seg), block_size):
            b = seg[s:s + block_size]
            if len(b) >= min_len:
                out.append(b)
    return out


@dataclass
class Block:
    rows: np.ndarray
    file_idx: int
    split: str               # train | val | test
    block_idx: int


@dataclass
class SplitPlan:
    """Quyết định split cho từng file train: 'train', 'val' hoặc 'tail' (chia theo thời gian)."""
    decisions: dict[str, str] = field(default_factory=dict)


def plan_splits(files: list[FileInfo], split_cfg: dict) -> SplitPlan:
    plan = SplitPlan()
    holdout = split_cfg.get("val_holdout") or {}
    by_type: dict[str, list[FileInfo]] = {}
    for f in files:
        if f.split_src == "train":
            by_type.setdefault(f.attack_type, []).append(f)
    for at, fs in by_type.items():
        if holdout.get(at) == "val":
            for f in fs:
                plan.decisions[f.path] = "val"
        elif len(fs) > 1 and split_cfg.get("multipart_val", "last_part") == "last_part":
            last = max(f.part for f in fs)
            for f in fs:
                plan.decisions[f.path] = "val" if f.part == last else "train"
        else:
            for f in fs:
                plan.decisions[f.path] = "tail"
    for f in files:
        if f.split_src == "test":
            plan.decisions[f.path] = "test"
    return plan


def file_blocks(X: np.ndarray, info: FileInfo, decision: str, file_idx: int, dcfg: dict,
                fw: int, val_frac: float) -> list[Block]:
    is_benign = info.label == 0 and dcfg.get("no_cap_benign", True)
    cap_tr = 0 if is_benign else dcfg["max_rows_train"]
    cap_te = 0 if is_benign else dcfg["max_rows_test"]
    bs, nseg = dcfg["block_size"], dcfg["n_segments"]

    def mk(rows_list, split, start_idx=0):
        blocks = to_blocks(rows_list, bs, fw)
        return [Block(b, file_idx, split, start_idx + i) for i, b in enumerate(blocks)]

    if decision == "test":
        return mk(take_segments(X, cap_te, nseg), "test")
    if decision in ("train", "val"):
        return mk(take_segments(X, cap_tr, nseg), decision)
    # "tail": 80% đầu train, đệm FW dòng, 20% cuối val
    n = len(X)
    cut = int(round(n * (1 - val_frac)))
    Xtr, Xva = X[:max(0, cut - fw // 2)], X[min(n, cut + fw - fw // 2):]
    cap_va = int(cap_tr * val_frac) if cap_tr else 0
    cap_tr2 = cap_tr - cap_va if cap_tr else 0
    tr = mk(take_segments(Xtr, cap_tr2, nseg), "train")
    va = mk(take_segments(Xva, cap_va, max(1, nseg // 2)), "val", start_idx=len(tr))
    return tr + va
