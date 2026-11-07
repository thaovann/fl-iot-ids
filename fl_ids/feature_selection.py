"""Chọn đặc trưng liên kết (methodology mục 4, định hướng 05-06/11 mục 2.1).

Ba dạng thông điệp client gửi lên server:
- Thống kê đủ cộng được: Fed-MI (bảng đếm theo bin × lớp), Fed-χ² (tổng theo lớp). Bằng đúng tập trung.
- Điểm cục bộ: scoresum_<scorer> (server cộng điểm).
- Hạng cục bộ: rank_<scorer>_<agg> (server gộp hạng bằng mean/median/trimmed/borda).

Mọi hàm trả về `ranking`: mảng chỉ số đặc trưng, tốt nhất trước, dài bằng số đặc trưng.
Mô hình đe dọa (Phần III) cài ở `poison_*`: client độc hại sửa thông điệp của mình trước khi gửi.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import trim_mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from .partition import Client

N_CLASSES = 2


@dataclass
class FSContext:
    clients: list[Client]
    gmin: np.ndarray
    gmax: np.ndarray
    cfg: dict
    seed: int
    malicious: set[int] = field(default_factory=set)
    attack_kind: str = "promote_worst"
    comm_bytes: dict[str, int] = field(default_factory=dict)

    def eligible(self) -> list[Client]:
        """Client có đủ 2 lớp (mới chấm điểm có giám sát được)."""
        return [c for c in self.clients if c.n and len(c.classes) >= 2]


def _subsample(X, y, n, seed):
    if len(y) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ic = np.flatnonzero(y == c)
        take = max(1, int(round(n * len(ic) / len(y))))
        idx.append(rng.choice(ic, min(take, len(ic)), replace=False))
    idx = np.concatenate(idx)
    return X[idx], y[idx]


def rank_from_scores(scores: np.ndarray, tie: np.ndarray | None = None) -> np.ndarray:
    s = np.nan_to_num(np.asarray(scores, float), nan=-np.inf)
    if tie is None:
        return np.argsort(-s, kind="stable")
    return np.lexsort((-tie, -s))


def positions(ranking: np.ndarray) -> np.ndarray:
    pos = np.empty(len(ranking), int)
    pos[ranking] = np.arange(len(ranking))
    return pos


# ---------------------------------------------------------------- thống kê đủ
def fed_mi_scores(ctx: FSContext, feat: np.ndarray | None = None, bins: int | None = None) -> np.ndarray:
    bins = bins or ctx.cfg["mi_bins"]
    cl = [c for c in ctx.clients if c.n]
    feat = np.arange(cl[0].Xtr.shape[1]) if feat is None else feat
    q = np.linspace(0, 1, bins + 1)
    # Vòng 1: biên bin = trung bình có trọng số các phân vị cục bộ
    Q = np.stack([np.quantile(c.Xtr[:, feat], q, axis=0) for c in cl])
    w = np.array([c.n for c in cl], float)
    edges = np.tensordot(w / w.sum(), Q, axes=1)            # (B+1, F)
    # Vòng 2: bảng đếm counts[f, b, y]
    counts = np.zeros((len(feat), bins, N_CLASSES))
    for c in cl:
        ck = np.zeros_like(counts)
        for j in range(len(feat)):
            e = np.unique(edges[:, j])
            b = np.clip(np.searchsorted(e[1:-1], c.Xtr[:, feat[j]], side="right"), 0, bins - 1)
            np.add.at(ck[j], (b, c.ytr), 1)
        if c.id in ctx.malicious and ctx.attack_kind == "inflate_n":
            ck *= 100
        counts += ck
        ctx.comm_bytes["fed_mi"] = ctx.comm_bytes.get("fed_mi", 0) + ck.size * 4 + Q[0].size * 4
    p = counts / counts.sum(axis=(1, 2), keepdims=True)
    pb = p.sum(2, keepdims=True)
    pc = p.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = p * np.log(p / (pb * pc))
    return np.nansum(t, axis=(1, 2))


def _chi2_stats(X, y, gmin, gmax):
    Xn = np.clip((X - gmin) / np.where(gmax > gmin, gmax - gmin, 1), 0, None)
    S = np.stack([Xn[y == c].sum(0) for c in range(N_CLASSES)])   # (C, D)
    n = np.bincount(y, minlength=N_CLASSES).astype(float)
    return S, n


def chi2_from_stats(S, n):
    tot = S.sum(0)
    exp = np.outer(n / n.sum(), tot)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = ((S - exp) ** 2 / exp).sum(0)
    return np.nan_to_num(chi)


def fed_chi2_scores(ctx: FSContext) -> np.ndarray:
    S_tot, n_tot = 0, 0
    for c in ctx.clients:
        if not c.n:
            continue
        S, n = _chi2_stats(c.Xtr, c.ytr, ctx.gmin, ctx.gmax)
        if c.id in ctx.malicious and ctx.attack_kind == "inflate_n":
            S, n = S * 100, n * 100
        S_tot, n_tot = S_tot + S, n_tot + n
        ctx.comm_bytes["fed_chi2"] = ctx.comm_bytes.get("fed_chi2", 0) + (S.size + n.size) * 4
    return chi2_from_stats(S_tot, n_tot)


# ---------------------------------------------------------------- điểm cục bộ
def local_scores(c: Client, scorer: str, ctx: FSContext) -> np.ndarray:
    cfg, seed = ctx.cfg, ctx.seed
    X, y = _subsample(c.Xtr, c.ytr, cfg["rfe_max_samples"], seed + c.id)
    if scorer == "chi2":
        return chi2_from_stats(*_chi2_stats(X, y, ctx.gmin, ctx.gmax))
    if scorer == "mi":
        return mutual_info_classif(X, y, random_state=seed)
    if scorer == "rf":
        rf = RandomForestClassifier(cfg["rf_trees"], max_depth=cfg["rf_max_depth"], class_weight="balanced",
                                    n_jobs=-1, random_state=seed).fit(X, y)
        return rf.feature_importances_
    if scorer == "mi_rfe":                     # MI top-k rồi RFE cục bộ; điểm = −hạng
        sub = FSContext([c], ctx.gmin, ctx.gmax, cfg, seed)
        ranking = mi_rfe_ranking(sub, mi_fn=lambda: mutual_info_classif(X, y, random_state=seed))
        return -positions(ranking).astype(float)
    raise ValueError(scorer)


def _poison_scores(s: np.ndarray, kind: str, top: int = 20) -> np.ndarray:
    worst = np.argsort(s)[:top]
    if kind == "promote_worst":               # đảo hạng: đặc trưng tệ nhất lên đầu
        return -s
    if kind == "score_dominate":              # dồn điểm rất lớn vào 20 đặc trưng tệ nhất
        out = np.zeros_like(s)
        out[worst] = (np.abs(s).max() + 1) * 1e3
        return out
    return s


def aggregate_ranks(pos_list: list[np.ndarray], agg: str, trim: float = 0.2) -> np.ndarray:
    P = np.stack(pos_list).astype(float)       # (K, D), 0 = tốt nhất
    if agg == "mean":
        s = P.mean(0)
    elif agg == "median":
        s = np.median(P, 0)
    elif agg == "trimmed":
        s = trim_mean(P, trim, axis=0)
    elif agg == "borda":
        s = -(P.shape[1] - P).sum(0)
    else:
        raise ValueError(agg)
    return np.lexsort((P.mean(0), s))          # nhỏ hơn = tốt hơn; hoà thì theo hạng trung bình


def rank_method(ctx: FSContext, scorer: str, agg: str) -> np.ndarray:
    pos_list = []
    for c in ctx.eligible():
        s = local_scores(c, scorer, ctx)
        if c.id in ctx.malicious:
            s = _poison_scores(s, ctx.attack_kind)
        pos_list.append(positions(rank_from_scores(s)))
        ctx.comm_bytes[f"rank_{scorer}"] = ctx.comm_bytes.get(f"rank_{scorer}", 0) + len(s) * 2
    return aggregate_ranks(pos_list, agg, ctx.cfg.get("trim_frac", 0.2))


def scoresum_method(ctx: FSContext, scorer: str) -> np.ndarray:
    tot = 0
    for c in ctx.eligible():
        s = local_scores(c, scorer, ctx)
        if c.id in ctx.malicious:
            s = _poison_scores(s, ctx.attack_kind if ctx.attack_kind != "promote_worst" else "score_dominate")
        tot = tot + s
    return rank_from_scores(tot)


# ---------------------------------------------------------------- MI -> RFE
def fed_rfe(ctx: FSContext, S: list[int], tie: np.ndarray) -> np.ndarray:
    cfg = ctx.cfg
    S = list(S)
    elim: list[int] = []
    while len(S) > 1:
        imps, ws = [], []
        for c in ctx.eligible():
            X, y = _subsample(c.Xtr[:, S], c.ytr, cfg["rfe_max_samples"], ctx.seed + c.id)
            rf = RandomForestClassifier(cfg["rf_trees"], max_depth=cfg["rf_max_depth"],
                                        class_weight="balanced", n_jobs=-1, random_state=ctx.seed).fit(X, y)
            imp = rf.feature_importances_
            if c.id in ctx.malicious:
                imp = _poison_scores(imp, ctx.attack_kind)
            imps.append(imp)
            ws.append(len(y))
        imp = np.average(np.stack(imps), axis=0, weights=ws)
        r = max(1, int(cfg["rfe_step_frac"] * len(S)))
        worst = sorted(range(len(S)), key=lambda j: (imp[j], tie[S[j]]))[:r]
        elim += [S[j] for j in worst]
        ws_set = set(worst)
        S = [S[j] for j in range(len(S)) if j not in ws_set]
    elim += S
    return np.array(elim[::-1])


def mi_rfe_ranking(ctx: FSContext, mi_fn=None) -> np.ndarray:
    mi = mi_fn() if mi_fn else fed_mi_scores(ctx)
    order = rank_from_scores(mi)
    top = order[: ctx.cfg["top_k_mi"]]
    rest = order[ctx.cfg["top_k_mi"]:]
    return np.concatenate([fed_rfe(ctx, list(top), mi), rest])


# ---------------------------------------------------------------- điểm vào
def select(method: str, ctx: FSContext) -> np.ndarray:
    D = ctx.clients[0].Xtr.shape[1]
    if method == "full":
        return np.arange(D)
    if method == "random":
        return np.random.default_rng(ctx.seed).permutation(D)
    if method == "fed_mi":
        return rank_from_scores(fed_mi_scores(ctx))
    if method == "fed_chi2":
        return rank_from_scores(fed_chi2_scores(ctx))
    if method == "fed_mi_rfe":
        return mi_rfe_ranking(ctx)
    if method.startswith("central_"):         # tham chiếu: một client chứa toàn bộ train
        allc = Client(0, np.concatenate([c.Xtr for c in ctx.clients]),
                      np.concatenate([c.ytr for c in ctx.clients]), np.empty((0, D)), np.empty(0, int),
                      np.empty(0), np.empty(0, int), np.empty(0, int))
        sub = FSContext([allc], ctx.gmin, ctx.gmax, ctx.cfg, ctx.seed)
        return select(method.replace("central_", "fed_"), sub)
    if method.startswith("rank_"):
        _, rest = method.split("_", 1)
        scorer, agg = rest.rsplit("_", 1)
        return rank_method(ctx, scorer, agg)
    if method.startswith("scoresum_"):
        return scoresum_method(ctx, method.split("_", 1)[1])
    raise ValueError(f"Phương pháp chọn đặc trưng không hỗ trợ: {method}")


def jaccard(a, b) -> float:
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / max(1, len(a | b))
