"""Cấu hình thí nghiệm.

Mọi tham số nằm trong một dict lồng nhau. Giá trị mặc định ở DEFAULTS, ghi đè bằng
file YAML (configs/*.yaml) rồi bằng danh sách "khoá.con=giá_trị" (từ CLI hoặc notebook).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULTS: dict[str, Any] = {
    "name": "default",
    "data": {
        # Thư mục gốc chứa CSV CICFlowMeter; tìm đệ quy mọi *.csv.
        # Train/test xác định theo hậu tố _train/_test trong tên file hoặc tên thư mục cha.
        "root": "/content/drive/MyDrive/CICIoMT2024/WiFI_and_MQTT/attacks/CSV",
        "glob": "**/*.csv",
        "id_cols": ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"],
        "label_col": "Label",
        "drop_features": [],            # ví dụ ["Protocol"] cho ablation
        "timestamp_dayfirst": True,
        "max_rows_train": 50000,        # 0 = không giới hạn
        "max_rows_test": 100000,
        "n_segments": 5,                # số đoạn liên tiếp khi file dài hơn cap
        "no_cap_benign": True,
        "block_size": 5000,             # đơn vị chia val và chia client
    },
    "window": {"fw": 512, "fs": 8, "stats": ["mean", "std", "median", "skew", "kurtosis"],
               "batch": 256},  # số cửa sổ tính mỗi lần; giảm nếu thiếu RAM
    "split": {
        # Val: file nhiều phần -> phần số lớn nhất làm val; file đơn -> khối cuối theo thời gian.
        "val_frac": 0.2,
        "val_holdout": {},              # {attack_type: "val"} để giữ trọn kiểu con làm val (mục 10.5)
        "multipart_val": "last_part",   # "last_part" | "tail_blocks"
    },
    "partition": {
        "schemes": ["iid", "dirichlet:0.5", "filepart"],
        "n_clients": 5,
        "n_clients_filepart": 4,
        "dirichlet_by": "attack_type",  # "attack_type" | "group" | "label"
        "single_file_map": {
            "ARP_Spoofing": 0, "MQTT-DDoS-Connect_Flood": 0, "Recon-OS_Scan": 0,
            "MQTT-DDoS-Publish_Flood": 1, "MQTT-DoS-Connect_Flood": 1, "Recon-Ping_Sweep": 1,
            "MQTT-DoS-Publish_Flood": 2, "Recon-Port_Scan": 2,
            "MQTT-Malformed_Data": 3, "Recon-VulScan": 3,
        },
    },
    "fs": {
        # Phương pháp chọn đặc trưng chạy cho mỗi (partition, seed). Tên hợp lệ:
        #   full, random, central_mi_rfe, fed_mi, fed_chi2, fed_mi_rfe,
        #   rank_<scorer>_<agg>  với scorer ∈ {mi, chi2, rf, mi_rfe}, agg ∈ {mean, median, trimmed, borda}
        #   scoresum_<scorer>
        "methods": ["full", "random", "central_mi_rfe", "fed_mi", "fed_chi2", "fed_mi_rfe",
                    "rank_chi2_median", "rank_mi_rfe_median"],
        "k_list": [10, 20],
        "mi_bins": 32,
        "top_k_mi": 100,
        "rfe_step_frac": 0.05,
        "rfe_max_samples": 20000,
        "rf_trees": 100,
        "rf_max_depth": 15,
        "trim_frac": 0.2,
    },
    "attack": {
        # Đầu độc bước chọn đặc trưng (Phần III). m = số client độc hại.
        "enabled": False,
        "m_list": [0],
        "kind": "promote_worst",        # "promote_worst" | "score_dominate" | "inflate_n"
    },
    "models": {
        # central_xgb, central_mlp, fedxgb_bag, fedxgb_cyclic, fedavg_mlp, fedprox_mlp:<mu>
        "methods": ["central_xgb", "fedxgb_bag", "fedavg_mlp"],
        "xgb": {"max_depth": 6, "eta": 0.1, "lambda": 1.0, "tree_method": "hist",
                "n_estimators_central": 100},
        "fedxgb": {"rounds": 20, "trees_per_client_round": 5},
        "mlp": {"hidden": [128, 64], "dropout": 0.2, "lr": 1e-3, "batch": 256,
                "local_epochs": 2, "rounds": 60, "fedprox_mu": 0.0, "device": "auto"},
        "threshold": "val_macro_f1",    # "val_macro_f1" | "0.5"
        "local_xgb": False,             # True: thêm cận dưới Local-only XGB (FULL đặc trưng)
    },
    "seeds": [42, 43, 44, 45, 46],
    "paths": {
        "cache_dir": "/content/drive/MyDrive/fl-iot-ids/cache",
        "out_dir": "/content/drive/MyDrive/fl-iot-ids/results",
    },
}


class Config(dict):
    """dict có truy cập kiểu thuộc tính: cfg.window.fw."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) and not isinstance(v, Config) else v

    def __setattr__(self, k, v):
        self[k] = v

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self))


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _parse_override(s: str) -> tuple[list[str], Any]:
    key, _, raw = s.partition("=")
    return key.strip().split("."), yaml.safe_load(raw)


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> Config:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    for ov in overrides:
        keys, val = _parse_override(ov)
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val
    return Config(cfg)


def data_fingerprint(cfg: Config) -> str:
    """Băm các tham số ảnh hưởng tới cache cửa sổ (dữ liệu + cửa sổ)."""
    key = {"data": cfg["data"], "window": cfg["window"]}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
