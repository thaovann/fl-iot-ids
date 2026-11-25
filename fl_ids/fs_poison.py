"""B2: tấn công chiếm chỗ vào chọn đặc trưng χ² (tập trung và liên kết) và tác hại lên detector.

Dùng cho `scripts/run_b2.py`. Tách khỏi pipeline Colab: chỉ cần một file .npz cửa sổ
(X_train, y_train, X_val, y_val, [X_test, y_test], feature_names; y là tên họ dạng chuỗi),
cùng định dạng với `chung_minh_A4.py` (B1).

Hai phần:
  * Oracle B2 (giao thức chốt trước 19/11, `giao-thuc-v6-B2-B5-nhap.md`): khi j trong k chỗ của top-k bị
    đặc trưng rác chiếm, recall từng họ giảm bao nhiêu. Ba nhánh (a) chỉ top-r, (b) top-r + j rác,
    (c) top-r ngoài cl(F*) + j rác. Đây là TRẦN của mọi tấn công chiếm chỗ.
  * CSP-chiếm-chỗ liên kết: client độc gửi thống kê đủ Fed-χ² (n_k[c], S_k[c, f]) tự nhất quán
    (giữ n_k và tổng cột, 0 ≤ S ≤ n) để đẩy F* ra và đẩy J (rác) vào top-k gộp. Đo tập chọn dưới các
    bộ gộp fed_chi2 / median_chi2 / robust_chi2 / rootmed rồi huấn luyện detector trên tập đó.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Ngưỡng đã chốt trước (B1/B2)
DROP_TARGET_RECALL = 0.10
FIXED_FPR = 0.01


# ------------------------------------------------------------------ dữ liệu
@dataclass
class Data:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xev: np.ndarray
    yev: np.ndarray
    names: list
    classes: list          # benign đứng đầu
    benign: str


def load_npz(path, benign="Benign", split="val", max_train_rows=0, seed=0):
    d = np.load(path, allow_pickle=True)
    Xtr, ytr = d["X_train"].astype(np.float32), d["y_train"].astype(str)
    if split == "test":
        Xev, yev = d["X_test"].astype(np.float32), d["y_test"].astype(str)
    else:
        Xev, yev = d["X_val"].astype(np.float32), d["y_val"].astype(str)
    names = [str(s) for s in d["feature_names"]]
    if max_train_rows and len(Xtr) > max_train_rows:          # rút gọn có phân tầng theo họ, cố định
        rng = np.random.default_rng(seed)
        keep = []
        for c in np.unique(ytr):
            idx = np.where(ytr == c)[0]
            keep.append(rng.choice(idx, max(1, int(round(len(idx) * max_train_rows / len(ytr)))), replace=False)
                        if len(idx) > 1 else idx)
        keep = np.sort(np.concatenate(keep))
        Xtr, ytr = Xtr[keep], ytr[keep]
    nz = Xtr.std(0) > 0                                         # bỏ cột hằng (như B1)
    classes = [benign] + sorted(set(ytr) - {benign})
    return Data(Xtr[:, nz], ytr, Xev[:, nz], yev, [n for n, k in zip(names, nz) if k], classes, benign)


def synthetic(seed=1, n_per=1500, nb=60):
    """Dữ liệu giả có dư thừa: 60 flow × 5 thống kê, benign + 6 họ; mỗi họ lệch ở một khối flow chồng lấn.
    Chỉ để kiểm mã, không mang ý nghĩa về tác hại thật."""
    rng = np.random.default_rng(seed)
    stats = {"mean": 0.0, "median": 0.15, "std": 0.8, "skew": 1.5, "kurtosis": 2.0}
    fams = {"Benign": (3, []), "DDoS": (3, list(range(0, 8))), "DoS": (2, list(range(4, 12))),
            "Recon": (1, list(range(14, 20))), "Mirai": (2, list(range(22, 30))),
            "Spoof": (0.5, list(range(30, 36))), "Web": (0.3, list(range(38, 44)))}
    shift = {f: rng.uniform(0.6, 1.6, size=len(b)) for f, (_, b) in fams.items()}

    def gen(scale):
        X, y = [], []
        for f, (w, blk) in fams.items():
            n = int(n_per * w * scale)
            B = rng.normal(size=(n, nb))
            B[:, blk] += shift[f]
            if f != "Benign":
                B[:, 50:54] += 0.5
            X.append(np.hstack([B + s * rng.normal(size=B.shape) for s in stats.values()]) + 10)
            y += [f] * n
        return np.vstack(X).astype(np.float32), np.array(y)
    Xtr, ytr = gen(1.0)
    Xva, yva = gen(0.5)
    Xte, yte = gen(0.5)
    names = np.array([f"f{c}__{s}" for s in stats for c in range(nb)])
    return dict(X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva, X_test=Xte, y_test=yte, feature_names=names)


# ------------------------------------------------------------------ χ², F*, bao đóng, rác
def minmax(X, gmin, gmax):
    return np.clip((X - gmin) / np.where(gmax > gmin, gmax - gmin, 1), 0, 1)


def chi2_from(S, n):
    """χ² từ tổng theo lớp. S: (C, d), n: (C,). Giống chi2_scores của chung_minh_A4.py."""
    T = S.sum(0)
    E = np.outer(n / n.sum(), T)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(((S - E) ** 2 / E).sum(0))


def class_sums(Xn, y, classes):
    return (np.stack([Xn[y == c].sum(0) for c in classes]),
            np.array([(y == c).sum() for c in classes], float))


def clean_ranking(Xn, y, classes):
    """Xếp hạng χ² đa lớp theo họ (rank-mode multiclass của B1)."""
    return np.argsort(-chi2_from(*class_sums(Xn, y, classes)), kind="stable")


def target_features(Xn, y, target, benign, n):
    """F* = top-n χ² họ đích so với benign (định nghĩa A4)."""
    S, N = class_sums(Xn, y, [benign, target])
    return list(np.argsort(-chi2_from(S, N), kind="stable")[:n])


def closure(F, names, X, rho=0.9, sep="__", max_rows=20000, seed=0):
    """cl(F): F + mọi thống kê của cùng đặc trưng flow + đặc trưng có |corr| ≥ rho (như B1)."""
    if not any(sep in nm for nm in names):
        sep = "_"
    base = [nm.rsplit(sep, 1)[0] for nm in names]
    fb = {base[i] for i in F}
    out = set(F) | {i for i, b in enumerate(base) if b in fb}
    if rho < 1:
        rng = np.random.default_rng(seed)
        R = X[rng.choice(len(X), min(len(X), max_rows), replace=False)].astype(float)
        sd = R.std(0)
        ok = sd > 0
        Z = np.zeros_like(R)
        Z[:, ok] = (R[:, ok] - R[:, ok].mean(0)) / sd[ok]
        C = np.abs(Z.T @ Z[:, sorted(out)] / len(Z))
        out |= set(np.where(C.max(1) >= rho)[0].tolist())
    return sorted(out)


def junk_features(ranking, j, exclude=()):
    """j đặc trưng χ² thấp nhất (cột hằng đã bỏ khi nạp) = định nghĩa A1 promote_worst."""
    ex = set(exclude)
    return [f for f in ranking[::-1] if f not in ex][:j]


# ------------------------------------------------------------------ detector và độ đo
def fit_predict_proba(Xtr, ytr_int, Xev, n_class, seed=42, n_jobs=-1):
    """XGB đa lớp, cùng siêu tham số với B1 (chung_minh_A4.fit_predict)."""
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist",
                        n_jobs=n_jobs, random_state=seed, eval_metric="mlogloss")
    clf.fit(Xtr, ytr_int)
    P = np.zeros((len(Xev), n_class))
    P[:, clf.classes_] = clf.predict_proba(Xev)
    return P


def detector_metrics(P, yev_int, classes, benign_idx=0, fpr=FIXED_FPR, min_eval=100):
    """Recall đa lớp theo họ, macro-F1, recall nhị phân, TPR tại FPR cố định (tổng và theo họ).

    Điểm tấn công = 1 − P(benign). Ngưỡng = phân vị (1 − fpr) của điểm trên cửa sổ benign của chính tập
    đánh giá (ở val đây là ngưỡng oracle; ở test nên lấy ngưỡng từ val, xem docs/B2.md)."""
    from sklearn.metrics import f1_score
    pred = P.argmax(1)
    score = 1 - P[:, benign_idx]
    ben = yev_int == benign_idx
    thr = np.quantile(score[ben], 1 - fpr) if ben.any() else 0.5
    out = {"macro_f1": float(f1_score(yev_int, pred, average="macro")),
           "recall_binary": float(np.mean(pred[~ben] != benign_idx)) if (~ben).any() else np.nan,
           "fpr_argmax": float(np.mean(pred[ben] != benign_idx)) if ben.any() else np.nan,
           "tpr_at_fpr": float(np.mean(score[~ben] > thr)) if (~ben).any() else np.nan,
           "recall": {}, "tpr_at_fpr_family": {}}
    for ci, c in enumerate(classes):
        m = yev_int == ci
        if ci == benign_idx or m.sum() < min_eval:
            continue
        out["recall"][c] = float(np.mean(pred[m] == ci))
        out["tpr_at_fpr_family"][c] = float(np.mean(score[m] > thr))
    return out


def eligible_families(D, min_train=300, min_eval=100):
    """Quy tắc G1: họ có ≥ 300 cửa sổ train và ≥ 100 cửa sổ đánh giá."""
    return [c for c in D.classes[1:] if (D.ytr == c).sum() >= min_train and (D.yev == c).sum() >= min_eval]


# ------------------------------------------------------------------ liên kết: chia client và thông điệp
def partition(y, classes, benign, K, mode, rng):
    idx = np.arange(len(y))
    if mode == "iid":
        rng.shuffle(idx)
        return np.array_split(idx, K)
    if mode != "by_source":
        raise ValueError(mode)
    # by_source: benign chia đều; họ tấn công thứ j giao vòng tròn cho client j, j+F, ...
    fams = [c for c in classes if c != benign]
    parts = [[] for _ in range(K)]
    b = idx[y == benign]
    rng.shuffle(b)
    for k, ch in enumerate(np.array_split(b, K)):
        parts[k] += ch.tolist()
    for j, f in enumerate(fams):
        ow = [k for k in range(K) if k % len(fams) == j] or [j % K]
        fi = idx[y == f]
        rng.shuffle(fi)
        for k, ch in zip(ow, np.array_split(fi, len(ow))):
            parts[k] += ch.tolist()
    return [np.array(p, int) for p in parts]


def client_msgs(Xn, y, parts, classes):
    """Thông điệp Fed-χ² của mỗi client: S_k[c, f] (tổng giá trị min-max theo họ) và n_k[c]."""
    S = np.zeros((len(parts), len(classes), Xn.shape[1]))
    N = np.zeros((len(parts), len(classes)))
    for k, p in enumerate(parts):
        yk = y[p]
        for ci, c in enumerate(classes):
            m = yk == c
            N[k, ci] = m.sum()
            if m.any():
                S[k, ci] = Xn[p[m]].sum(0)
    return S, N


# ------------------------------------------------------------------ tấn công CSP (nhất quán)
def _project_box_sum(target, w, lo, hi, total, iters=80):
    """argmin Σ (x − target)² / w  với lo ≤ x ≤ hi, Σ x = total (chia đôi nhân tử Lagrange)."""
    total = float(np.clip(total, lo.sum(), hi.sum()))
    a, b = -1e9, 1e9
    for _ in range(iters):
        lam = (a + b) / 2
        if np.clip(target + lam * w, lo, hi).sum() > total:
            b = lam
        else:
            a = lam
    return np.clip(target + (a + b) / 2 * w, lo, hi)


def csp_suppress(S, N, mal, fstar):
    """CSP-loại (A7c): trên cột F*, kéo tổng gộp theo họ về giá trị kỳ vọng (χ² gộp → 0),
    giữ n_k và tổng cột T_k của từng client độc. Kẻ tấn công biết tổng của client lành (xấu nhất)."""
    S = S.copy()
    hon = [k for k in range(len(S)) if k not in set(mal)]
    H = S[hon].sum(0)
    Ntot = N.sum(0)
    for f in fstar:
        Tm = np.array([S[j, :, f].sum() for j in mal])
        E = Ntot / Ntot.sum() * (H[:, f].sum() + Tm.sum())
        need = E - H[:, f]
        for jj, j in enumerate(mal):
            share = Tm[jj] / max(Tm.sum(), 1e-12)
            S[j, :, f] = _project_box_sum(need * share, np.maximum(E, 1e-9), np.zeros(len(N[j])), N[j].copy(), Tm[jj])
    return S


def csp_promote(S, N, mal, junk):
    """CSP-chèn: trên cột J, dồn toàn bộ tổng cột T_k[f] của client độc vào các họ hiếm nhất (theo số cửa sổ
    gộp) mà nó có, mỗi họ tối đa n_k[c] (giá trị min-max ≤ 1). Đây là đỉnh của hộp ràng buộc làm χ² lớn
    nhất theo hướng tham lam; vẫn giữ n_k, tổng cột và 0 ≤ S ≤ n nên qua lớp kiểm tra cấu trúc."""
    S = S.copy()
    order = np.argsort(N.sum(0))                   # họ hiếm nhất trước
    for j in mal:
        for f in junk:
            T = S[j, :, f].sum()
            new = np.zeros(N.shape[1])
            for c in order:
                if T <= 0:
                    break
                put = min(T, N[j, c])
                new[c] = put
                T -= put
            S[j, :, f] = new
    return S


def csp_hijack(S, N, mal, fstar, junk):
    """CSP-chiếm-chỗ = loại F* + chèn J, cùng một thông điệp nhất quán."""
    if not mal:
        return S
    S = csp_suppress(S, N, mal, fstar) if len(fstar) else S
    return csp_promote(S, N, mal, junk) if len(junk) else S


# ------------------------------------------------------------------ bộ gộp phía server
def _means(S, N):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(N[..., None] > 0, S / np.maximum(N[..., None], 1), np.nan)


def agg_fed_chi2(S, N, **_):
    return chi2_from(S.sum(0), N.sum(0))


def agg_median_chi2(S, N, **_):
    loc = [chi2_from(S[k], N[k]) for k in range(len(S)) if (N[k] > 0).sum() >= 2]
    return np.median(np.stack(loc), 0)


def agg_robust_chi2(S, N, min_n=20, **_):
    """BẢN DỰNG LẠI (cần Van đối chiếu với robust_chi2 thật): trung vị theo họ của trung bình họ μ_k[c, f],
    chỉ trên client có ≥ min_n cửa sổ họ c, rồi nhân lại số cửa sổ gộp."""
    mu = _means(S, N)
    mu[N < min_n] = np.nan
    with np.errstate(all="ignore"):
        med = np.nanmedian(mu, 0)
    Nt = N.sum(0)
    return chi2_from(np.nan_to_num(med) * Nt[:, None], Nt)


def agg_rootmed(S, N, root=None, z=3.0, min_n=20, **_):
    """rootmed: kẹp μ_k[c, f] vào μ_root ± z·sqrt(σ²/n_k + σ²/r), rồi trung vị theo họ."""
    mu_r, sd_r, n_r = root
    mu = _means(S, N)
    half = z * np.sqrt(sd_r[None] ** 2 / np.maximum(N[..., None], 1) + sd_r[None] ** 2 / np.maximum(n_r, 1)[None, :, None])
    mu = np.clip(mu, mu_r[None] - half, mu_r[None] + half)
    mu[N < min_n] = np.nan
    with np.errstate(all="ignore"):
        med = np.nanmedian(mu, 0)
    Nt = N.sum(0)
    return chi2_from(np.nan_to_num(med) * Nt[:, None], Nt)


AGGREGATORS = {"fed_chi2": agg_fed_chi2, "median_chi2": agg_median_chi2,
               "robust_chi2": agg_robust_chi2, "rootmed": agg_rootmed}


def make_root(Xn, y, classes, per_class, rng):
    """Tập gốc của server: per_class cửa sổ mỗi họ (rút khỏi dữ liệu client)."""
    idx = []
    for c in classes:
        ci = np.where(y == c)[0]
        idx.append(rng.choice(ci, min(per_class, max(1, len(ci) // 5)), replace=False))
    idx = np.concatenate(idx)
    Xr, yr = Xn[idx], y[idx]
    root = (np.stack([Xr[yr == c].mean(0) for c in classes]),
            np.stack([Xr[yr == c].std(0) + 1e-6 for c in classes]),
            np.array([(yr == c).sum() for c in classes], float))
    return root, idx
