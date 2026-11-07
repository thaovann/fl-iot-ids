"""Vòng lặp thí nghiệm: partition × seed × số client độc hại × chọn đặc trưng × k × mô hình.

Mỗi dòng kết quả được ghi ngay vào `<out_dir>/<name>/final_summary.csv`, nên khi Colab ngắt
kết nối chỉ cần chạy lại cùng lệnh: các dòng đã có sẽ được bỏ qua (resume).
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import feature_selection as fsel
from .config import Config
from .metrics import evaluate, fed_val_threshold
from .models import train, train_local_xgb
from .partition import Client, client_table, make_clients
from .preprocess import apply_scaler, fed_class_weights, fit_fed_scaler
from .windowing import WindowSet, get_windows, window_counts

KEY_COLS = ["partition", "seed", "m_malicious", "fs_method", "k", "model", "client"]


class ResultWriter:
    def __init__(self, path: Path):
        self.path = path
        self.done: set[tuple] = set()
        self.fields: list[str] | None = None
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                rd = csv.DictReader(f)
                self.fields = rd.fieldnames
                for r in rd:
                    self.done.add(tuple(str(r[k]) for k in KEY_COLS))

    def key(self, row: dict) -> tuple:
        return tuple(str(row.get(k, "")) for k in KEY_COLS)

    def has(self, **kw) -> bool:
        return self.key(kw) in self.done

    def write(self, row: dict) -> None:
        if self.fields is None:
            self.fields = KEY_COLS + [k for k in row if k not in KEY_COLS]
        extra = [k for k in row if k not in self.fields]
        if extra:                                      # cột mới (ví dụ recall_<nhóm>): viết lại header
            self.fields += extra
            rows = []
            if self.path.exists():
                with open(self.path, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, self.fields)
                wr.writeheader()
                wr.writerows(rows)
        new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, self.fields)
            if new:
                wr.writeheader()
            wr.writerow({k: row.get(k, "") for k in self.fields})
        self.done.add(self.key(row))


def _slice(clients: list[Client], S: np.ndarray) -> list[Client]:
    return [replace(c, Xtr=c.Xtr[:, S], Xva=c.Xva[:, S]) for c in clients]


def _pick_malicious(clients: list[Client], m: int, seed: int) -> set[int]:
    elig = [c.id for c in clients if c.n and len(c.classes) >= 2]
    rng = np.random.default_rng(seed + 7919)
    return set(map(int, rng.choice(elig, min(m, len(elig)), replace=False))) if m else set()


def run(cfg: Config, ws: WindowSet | None = None, verbose: bool = True) -> Path:
    out = Path(cfg["paths"]["out_dir"]) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
    ws = ws or get_windows(cfg, verbose=verbose)
    window_counts(ws).to_csv(out / "window_counts.csv")

    te = ws.split == "test"
    Xte_raw, yte, gte = ws.X[te], ws.y[te].astype(int), ws.group[te]
    writer = ResultWriter(out / "final_summary.csv")
    rank_path = out / "rankings.jsonl"
    rank_cache: dict[str, list[int]] = {}
    if rank_path.exists():
        for line in rank_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rank_cache[r["key"]] = r["ranking"]

    fcfg, mcfg = cfg["fs"], cfg["models"]
    m_list = cfg["attack"]["m_list"] if cfg["attack"]["enabled"] else [0]
    for scheme in cfg["partition"]["schemes"]:
        for seed in cfg["seeds"]:
            t_ps = time.time()
            clients = make_clients(ws, scheme, seed, cfg["partition"])
            client_table(clients).to_csv(out / f"clients_{scheme.replace(':', '')}_s{seed}.csv", index=False)
            sc = fit_fed_scaler(clients)
            clients = apply_scaler(clients, sc)
            w = fed_class_weights(clients)
            Xte = sc.transform(Xte_raw)
            names = [n for n, k in zip(ws.feature_names, sc.keep) if k]
            D = len(names)
            if verbose:
                print(f"\n=== {scheme} | seed {seed} | {len(clients)} client | D = {D} ===")
                print(client_table(clients).to_string(index=False))

            for m in m_list:
                mal = _pick_malicious(clients, m, seed)
                central_rank = None
                for fs_method in fcfg["methods"]:
                    ks = [D] if fs_method == "full" else [k for k in fcfg["k_list"] if k <= D]
                    todo = [(k, mod) for k in ks for mod in mcfg["methods"]
                            if not writer.has(partition=scheme, seed=seed, m_malicious=m,
                                              fs_method=fs_method, k=k, model=mod, client="")]
                    rkey = f"{scheme}|{seed}|{m}|{fs_method}"
                    if rkey in rank_cache:
                        ranking = np.array(rank_cache[rkey])
                        fs_time = float("nan")
                    elif not todo and not fs_method.startswith("central_"):
                        continue
                    else:
                        ctx = fsel.FSContext(clients, sc.gmin, sc.gmax, fcfg, seed,
                                             malicious=mal if not fs_method.startswith("central_") else set(),
                                             attack_kind=cfg["attack"]["kind"])
                        t0 = time.time()
                        ranking = fsel.select(fs_method, ctx)
                        fs_time = time.time() - t0
                        rank_cache[rkey] = ranking.tolist()
                        with open(rank_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"key": rkey, "ranking": ranking.tolist(),
                                                "top_names": [names[i] for i in ranking[:100]],
                                                "fs_time_s": fs_time, "comm_bytes": ctx.comm_bytes}) + "\n")
                        if verbose:
                            print(f"  [{fs_method}] chọn xong {fs_time:.1f}s; top-5: "
                                  f"{[names[i] for i in ranking[:5]]}")
                    if fs_method == "central_mi_rfe":
                        central_rank = ranking
                    for k, mod in todo:
                        S = np.sort(ranking[:k])
                        cs = _slice(clients, S)
                        predict, info = train(mod, cs, w, mcfg, seed)
                        vc = [c for c in cs if len(c.yva)]
                        if mcfg.get("threshold", "val_macro_f1") == "val_macro_f1":
                            tau, val_f1 = fed_val_threshold([predict(c.Xva) for c in vc], [c.yva for c in vc])
                        else:
                            tau, val_f1 = 0.5, float("nan")
                        t1 = time.time()
                        prob = predict(Xte[:, S])
                        infer_ms = (time.time() - t1) * 1000 / max(1, len(prob))
                        res = evaluate(yte, prob, tau, gte)
                        res05 = evaluate(yte, prob, 0.5)
                        row = {"partition": scheme, "seed": seed, "m_malicious": m, "fs_method": fs_method,
                               "k": k, "model": mod, "client": "", "D": D, "val_macro_f1": val_f1,
                               **res, "macro_f1_tau05": res05["macro_f1"], "fpr_tau05": res05["fpr"],
                               "fs_time_s": fs_time, "infer_ms_per_window": infer_ms,
                               "jaccard_vs_central": fsel.jaccard(ranking[:k], central_rank[:k])
                               if central_rank is not None and fs_method != "full" else "",
                               "malicious": ";".join(map(str, sorted(mal))), **info}
                        writer.write(row)
                        if verbose:
                            print(f"    {fs_method:>20} k={k:<4} {mod:<14} macro-F1 {res['macro_f1']:.4f} "
                                  f"MCC {res['mcc']:.4f} FPR {res['fpr']:.4f} τ={tau:.2f}")

            if mcfg.get("local_xgb", False):          # cận dưới Local-only, mỗi client một dòng
                for c in clients:
                    if not c.n or writer.has(partition=scheme, seed=seed, m_malicious=0, fs_method="full",
                                             k=D, model="local_xgb", client=c.id):
                        continue
                    pred = train_local_xgb(c, w, mcfg, seed)
                    tau = fed_val_threshold([pred(c.Xva)], [c.yva])[0] if len(c.yva) else 0.5
                    res = evaluate(yte, pred(Xte), tau, gte)
                    writer.write({"partition": scheme, "seed": seed, "m_malicious": 0, "fs_method": "full",
                                  "k": D, "model": "local_xgb", "client": c.id, "D": D, **res})
            if verbose:
                print(f"--- xong {scheme} seed {seed} trong {time.time() - t_ps:.0f}s")
    return out / "final_summary.csv"


def summarize(csv_path: str | Path, by=("partition", "fs_method", "k", "model", "m_malicious"),
              metrics=("macro_f1", "mcc", "attack_recall", "fpr", "auc_pr")):
    """Bảng mean ± std theo seed."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df["client"].isna()] if "client" in df else df
    metrics = [m for m in metrics if m in df]
    g = df.groupby(list(by))[metrics].agg(["mean", "std"])
    g["n_seed"] = df.groupby(list(by)).size()
    return g
