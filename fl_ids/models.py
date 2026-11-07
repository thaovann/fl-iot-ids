"""Mô hình tập trung và liên kết (methodology mục 5, 10.8).

Mỗi hàm train_* nhận danh sách client (đã chuẩn hoá, đã cắt cột theo bộ đặc trưng) và trả về
(predict_proba, info). Vòng tốt nhất chọn theo macro-F1 trên val liên kết (cộng ma trận nhầm lẫn).
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import xgboost as xgb

from .metrics import fed_val_f1
from .partition import Client

Predict = Callable[[np.ndarray], np.ndarray]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _xgb_params(mcfg: dict, seed: int) -> dict:
    x = mcfg["xgb"]
    return {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": x["max_depth"],
            "eta": x["eta"], "lambda": x["lambda"], "tree_method": x["tree_method"],
            "base_score": 0.5, "seed": seed, "verbosity": 0}


def _concat(clients):
    X = np.concatenate([c.Xtr for c in clients])
    y = np.concatenate([c.ytr for c in clients])
    return X, y


# ---------------------------------------------------------------- tập trung
def train_central_xgb(clients: list[Client], w: np.ndarray, mcfg: dict, seed: int):
    t0 = time.time()
    X, y = _concat(clients)
    d = xgb.DMatrix(X, label=y, weight=w[y])
    b = xgb.train(_xgb_params(mcfg, seed), d, num_boost_round=mcfg["xgb"]["n_estimators_central"])
    return (lambda Z: b.predict(xgb.DMatrix(Z))), {"train_s": time.time() - t0, "comm_mb": 0.0}


def train_local_xgb(c: Client, w: np.ndarray, mcfg: dict, seed: int):
    d = xgb.DMatrix(c.Xtr, label=c.ytr, weight=w[c.ytr])
    b = xgb.train(_xgb_params(mcfg, seed), d, num_boost_round=mcfg["xgb"]["n_estimators_central"])
    return lambda Z: b.predict(xgb.DMatrix(Z))


# ---------------------------------------------------------------- FedXGB bagging (10.8)
def train_fedxgb_bag(clients: list[Client], w: np.ndarray, mcfg: dict, seed: int):
    t0 = time.time()
    P = _xgb_params(mcfg, seed)
    eta = mcfg["xgb"]["eta"]
    R, T = mcfg["fedxgb"]["rounds"], mcfg["fedxgb"]["trees_per_client_round"]
    act = [c for c in clients if c.n]
    N = sum(c.n for c in act)
    dtr = {c.id: xgb.DMatrix(c.Xtr) for c in act}
    dva = {c.id: xgb.DMatrix(c.Xva) for c in clients if len(c.yva)}
    Ftr = {c.id: np.zeros(c.n) for c in act}
    Fva = {k: np.zeros(d.num_row()) for k, d in dva.items()}
    ens: list[xgb.Booster] = []
    best = (-1.0, 0)
    comm = 0
    for r in range(R):
        new = []
        for c in act:
            d = xgb.DMatrix(c.Xtr, label=c.ytr, weight=w[c.ytr], base_margin=Ftr[c.id])
            p = dict(P, eta=eta * c.n / N, seed=seed + r * 100 + c.id)
            new.append(xgb.train(p, d, num_boost_round=T))
        for b in new:
            sz = len(b.save_raw())
            comm += sz + len(act) * sz
            for k in Ftr:
                Ftr[k] += b.predict(dtr[k], output_margin=True)
            for k in Fva:
                Fva[k] += b.predict(dva[k], output_margin=True)
        ens += new
        f1 = fed_val_f1([_sigmoid(Fva[c.id]) for c in clients if c.id in Fva],
                        [c.yva for c in clients if c.id in Fva])
        if f1 > best[0]:
            best = (f1, len(ens))
    keep = ens[: best[1]] if best[1] else ens

    def predict(Z):
        d = xgb.DMatrix(Z)
        return _sigmoid(sum(b.predict(d, output_margin=True) for b in keep))
    return predict, {"train_s": time.time() - t0, "comm_mb": comm / 2**20,
                     "best_round": best[1] // max(1, len(act)), "val_f1_05": best[0]}


# ---------------------------------------------------------------- FedXGB cyclic (5.1)
def train_fedxgb_cyclic(clients: list[Client], w: np.ndarray, mcfg: dict, seed: int):
    t0 = time.time()
    P = _xgb_params(mcfg, seed)
    R, T = mcfg["fedxgb"]["rounds"], mcfg["fedxgb"]["trees_per_client_round"]
    rng = np.random.default_rng(seed)
    act = [c for c in clients if c.n]
    booster, best, comm = None, (-1.0, None), 0
    for r in range(R):
        for i in rng.permutation(len(act)):
            c = act[i]
            d = xgb.DMatrix(c.Xtr, label=c.ytr, weight=w[c.ytr])
            booster = xgb.train(P, d, num_boost_round=T, xgb_model=booster)
            comm += 2 * len(booster.save_raw())
        f1 = fed_val_f1([booster.predict(xgb.DMatrix(c.Xva)) for c in clients if len(c.yva)],
                        [c.yva for c in clients if len(c.yva)])
        if f1 > best[0]:
            best = (f1, booster.copy())
    b = best[1] or booster
    return (lambda Z: b.predict(xgb.DMatrix(Z))), {"train_s": time.time() - t0, "comm_mb": comm / 2**20,
                                                    "val_f1_05": best[0]}


# ---------------------------------------------------------------- FedAvg / FedProx MLP
def train_fed_mlp(clients: list[Client], w: np.ndarray, mcfg: dict, seed: int, mu: float | None = None,
                  central: bool = False):
    import torch
    from torch import nn

    m = mcfg["mlp"]
    mu = m.get("fedprox_mu", 0.0) if mu is None else mu
    dev = m.get("device", "auto")
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev
    torch.manual_seed(seed)
    t0 = time.time()
    if central:
        X, y = _concat(clients)
        clients = [Client(0, X, y, np.concatenate([c.Xva for c in clients]),
                          np.concatenate([c.yva for c in clients]), np.empty(0), np.empty(0), np.empty(0))]
    act = [c for c in clients if c.n]
    D = act[0].Xtr.shape[1]
    layers, prev = [], D
    for h in m["hidden"]:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(m["dropout"])]
        prev = h
    layers.append(nn.Linear(prev, 2))
    glob = nn.Sequential(*layers).to(dev)
    n_params = sum(p.numel() for p in glob.parameters())
    wt = torch.tensor(w, dtype=torch.float32, device=dev)
    data = {c.id: (torch.tensor(c.Xtr, device=dev), torch.tensor(c.ytr, device=dev)) for c in act}
    N = sum(c.n for c in act)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def proba(model, X):
        model.eval()
        with torch.no_grad():
            out = []
            for s in range(0, len(X), 8192):
                xb = torch.tensor(X[s:s + 8192], device=dev)
                out.append(torch.softmax(model(xb), 1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)

    best = (-1.0, None)
    R = m["rounds"]
    for r in range(R):
        states, weights = [], []
        gstate = {k: v.detach().clone() for k, v in glob.state_dict().items()}
        for c in act:
            glob.load_state_dict(gstate)
            model = glob
            opt = torch.optim.Adam(model.parameters(), lr=m["lr"])
            Xc, yc = data[c.id]
            g_params = [p.detach().clone() for p in model.parameters()]
            model.train()
            for _ in range(m["local_epochs"]):
                perm = torch.randperm(len(yc), generator=g).to(dev)
                for s in range(0, len(yc), m["batch"]):
                    i = perm[s:s + m["batch"]]
                    loss = nn.functional.cross_entropy(model(Xc[i]), yc[i], weight=wt)
                    if mu > 0:
                        loss = loss + mu / 2 * sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), g_params))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            weights.append(c.n / N)
        avg = {k: sum(wk * s[k] for wk, s in zip(weights, states)) for k in states[0]}
        glob.load_state_dict(avg)
        vc = [c for c in clients if len(c.yva)]
        f1 = fed_val_f1([proba(glob, c.Xva) for c in vc], [c.yva for c in vc])
        if f1 > best[0]:
            best = (f1, {k: v.clone() for k, v in glob.state_dict().items()})
    if best[1] is not None:
        glob.load_state_dict(best[1])
    comm = 0 if central else n_params * 4 * 2 * len(act) * R
    return (lambda Z: proba(glob, Z)), {"train_s": time.time() - t0, "comm_mb": comm / 2**20,
                                        "val_f1_05": best[0]}


def train(method: str, clients: list[Client], w: np.ndarray, mcfg: dict, seed: int):
    if method == "central_xgb":
        return train_central_xgb(clients, w, mcfg, seed)
    if method == "fedxgb_bag":
        return train_fedxgb_bag(clients, w, mcfg, seed)
    if method == "fedxgb_cyclic":
        return train_fedxgb_cyclic(clients, w, mcfg, seed)
    if method == "fedavg_mlp":
        return train_fed_mlp(clients, w, mcfg, seed, mu=0.0)
    if method.startswith("fedprox_mlp"):       # fedprox_mlp hoặc fedprox_mlp:0.1
        mu = float(method.split(":")[1]) if ":" in method else mcfg["mlp"].get("fedprox_mu", 0.01)
        return train_fed_mlp(clients, w, mcfg, seed, mu=mu)
    if method == "central_mlp":
        return train_fed_mlp(clients, w, mcfg, seed, mu=0.0, central=True)
    raise ValueError(f"Mô hình không hỗ trợ: {method}")
