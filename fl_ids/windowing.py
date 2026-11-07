"""Cửa sổ trượt + 5 thống kê (methodology mục 2), và dựng/cache toàn bộ tập cửa sổ."""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import kurtosis, skew

from .config import Config, data_fingerprint
from .data import (FileInfo, assert_no_metadata, discover_files, file_blocks, load_file,
                   plan_splits)

STAT_FUNCS = {
    "mean": lambda W: W.mean(-1),
    "std": lambda W: W.std(-1),
    "median": lambda W: np.median(W, -1),
    "skew": lambda W: skew(W, axis=-1, bias=True),
    "kurtosis": lambda W: kurtosis(W, axis=-1, fisher=True, bias=True),
}


def windows_of_block(Xb: np.ndarray, fw: int, fs: int, stats=("mean", "std", "median", "skew", "kurtosis"),
                     batch: int = 256) -> np.ndarray:
    d = Xb.shape[1]
    if len(Xb) < fw:
        return np.empty((0, len(stats) * d), np.float32)
    W_all = sliding_window_view(Xb.astype(np.float64), fw, axis=0)[::fs]  # (n_win, d, fw)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s in range(0, len(W_all), batch):
            W = W_all[s:s + batch]
            feats = [STAT_FUNCS[k](W) for k in stats]
            out.append(np.concatenate(feats, axis=1))
    F = np.concatenate(out, axis=0)
    # Cột hằng trong cửa sổ -> skew/kurtosis NaN -> 0 (giá trị trung tính)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@dataclass
class WindowSet:
    X: np.ndarray            # (N, D) float32
    y: np.ndarray            # (N,) int8, nhãn nhị phân
    split: np.ndarray        # (N,) "train"/"val"/"test"
    block: np.ndarray        # (N,) id khối toàn cục
    file_idx: np.ndarray     # (N,) chỉ số file trong `files`
    feature_names: list[str]
    files: list[dict]        # FileInfo dạng dict

    @property
    def attack_type(self) -> np.ndarray:
        at = np.array([f["attack_type"] for f in self.files])
        return at[self.file_idx]

    @property
    def group(self) -> np.ndarray:
        g = np.array([f["group"] for f in self.files])
        return g[self.file_idx]

    def mask(self, split: str) -> np.ndarray:
        return self.split == split

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, X=self.X, y=self.y, split=self.split, block=self.block,
                            file_idx=self.file_idx,
                            meta=np.array(json.dumps({"feature_names": self.feature_names,
                                                      "files": self.files})))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "WindowSet":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(z["X"], z["y"], z["split"], z["block"], z["file_idx"],
                   meta["feature_names"], meta["files"])


def build_windows(cfg: Config, verbose: bool = True) -> WindowSet:
    dcfg, wcfg = cfg["data"], cfg["window"]
    fw, fs, stats = wcfg["fw"], wcfg["fs"], wcfg["stats"]
    files = discover_files(dcfg["root"], dcfg.get("glob", "**/*.csv"))
    plan = plan_splits(files, cfg["split"])
    Xs, ys, sps, blks, fids = [], [], [], [], []
    cols_ref = None
    short_files = []
    gblock = 0
    t0 = time.time()
    for fi, info in enumerate(files):
        X, cols = load_file(info.path, dcfg["id_cols"], dcfg.get("label_col", "Label"),
                            dcfg.get("drop_features", []), dcfg.get("timestamp_dayfirst", True))
        if cols_ref is None:
            cols_ref = cols
        elif cols != cols_ref:
            missing = set(cols_ref) ^ set(cols)
            raise ValueError(f"{info.stem}: cột khác file đầu tiên: {sorted(missing)[:10]}")
        if len(X) < fw:
            short_files.append((info.stem, len(X)))
        blocks = file_blocks(X, info, plan.decisions[info.path], fi, dcfg, fw, cfg["split"]["val_frac"])
        for b in blocks:
            W = windows_of_block(b.rows, fw, fs, stats, wcfg.get("batch", 256))
            if len(W) == 0:
                continue
            Xs.append(W)
            n = len(W)
            ys.append(np.full(n, info.label, np.int8))
            sps.append(np.full(n, b.split, dtype="<U5"))
            blks.append(np.full(n, gblock, np.int32))
            fids.append(np.full(n, fi, np.int32))
            gblock += 1
        if verbose:
            print(f"[{fi + 1}/{len(files)}] {info.stem}: {len(X)} dòng, "
                  f"{plan.decisions[info.path]}, {len(blocks)} khối ({time.time() - t0:.0f}s)")
    if short_files and verbose:
        print("File có ít hơn FW dòng (không tạo được cửa sổ):", short_files)
    names = [f"{c}__{s}" for s in stats for c in cols_ref]
    assert_no_metadata(names)
    ws = WindowSet(np.concatenate(Xs), np.concatenate(ys), np.concatenate(sps),
                   np.concatenate(blks), np.concatenate(fids), names,
                   [f.__dict__ for f in files])
    return ws


def get_windows(cfg: Config, force: bool = False, verbose: bool = True) -> WindowSet:
    """Dựng tập cửa sổ hoặc nạp từ cache trên Drive (khoá = băm tham số data + window)."""
    cache = Path(cfg["paths"]["cache_dir"]) / f"windows_{data_fingerprint(cfg)}.npz"
    if cache.exists() and not force:
        if verbose:
            print(f"Nạp cửa sổ từ cache {cache}")
        return WindowSet.load(cache)
    ws = build_windows(cfg, verbose)
    ws.save(cache)
    with open(cache.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"data": cfg["data"], "window": cfg["window"]}, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"Đã lưu cache {cache} ({ws.X.shape[0]} cửa sổ × {ws.X.shape[1]} đặc trưng)")
    return ws


def window_counts(ws: WindowSet):
    import pandas as pd
    df = pd.DataFrame({"split": ws.split, "group": ws.group, "attack_type": ws.attack_type, "y": ws.y})
    return df.groupby(["split", "group", "attack_type"]).size().unstack("split", fill_value=0)
