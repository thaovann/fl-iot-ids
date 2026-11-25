"""Kiểm tra B2: python -m pytest -q tests/test_fs_poison.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fl_ids import fs_poison as fp  # noqa: E402


def _setup(mode="iid", seed=0):
    d = fp.synthetic(seed=1, n_per=300)
    X, y = d["X_train"], d["y_train"]
    classes = ["Benign"] + sorted(set(y) - {"Benign"})
    Xn = fp.minmax(X, X.min(0), X.max(0))
    rng = np.random.default_rng(seed)
    parts = fp.partition(y, classes, "Benign", 10, mode, rng)
    S, N = fp.client_msgs(Xn, y, parts, classes)
    return Xn, y, classes, S, N


def test_partition_covers_all():
    for mode in ("iid", "by_source"):
        Xn, y, classes, S, N = _setup(mode)
        assert N.sum() == len(y)
        assert np.allclose(S.sum((0, 1)), Xn.sum(0))


def test_csp_hijack_is_consistent():
    """Thông điệp độc phải qua mọi kiểm tra cấu trúc: n_k giữ nguyên, tổng cột giữ nguyên, 0 ≤ S ≤ n."""
    Xn, y, classes, S, N = _setup("iid")
    rank = list(fp.clean_ranking(Xn, y, classes))
    F = fp.target_features(Xn, y, "DDoS", "Benign", 20)
    J = fp.junk_features(rank, 10, F)
    mal = [0, 1]
    S2 = fp.csp_hijack(S, N, mal, F, J)
    for j in mal:
        assert np.allclose(S2[j].sum(0), S[j].sum(0), atol=1e-6)
        assert (S2[j] >= -1e-9).all() and (S2[j] <= N[j][:, None] + 1e-6).all()
    assert np.array_equal(S2[2:], S[2:])


def test_csp_hijack_moves_fed_chi2_selection():
    Xn, y, classes, S, N = _setup("iid")
    rank = list(fp.clean_ranking(Xn, y, classes))
    k = 10
    F = fp.target_features(Xn, y, "DDoS", "Benign", 20)
    J = fp.junk_features(rank, k, F)
    clean = set(np.argsort(-fp.agg_fed_chi2(S, N))[:k])
    S2 = fp.csp_hijack(S, N, [0, 1, 2], F, J)
    att = set(np.argsort(-fp.agg_fed_chi2(S2, N))[:k])
    assert len(att & set(J)) > len(clean & set(J))
    assert len(att & set(F)) < len(clean & set(F))


def test_detector_metrics():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 200)
    P = np.eye(3)[y] * 0.8 + 0.2 / 3
    m = fp.detector_metrics(P + rng.uniform(0, 0.01, P.shape), y, ["Benign", "A", "B"])
    assert m["recall"] == {"A": 1.0, "B": 1.0}
    assert 0.99 <= m["tpr_at_fpr"] <= 1.0
