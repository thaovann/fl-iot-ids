"""Kiểm tra nhanh: python -m pytest -q"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fl_ids import feature_selection as fsel  # noqa: E402
from fl_ids.config import load_config  # noqa: E402
from fl_ids.data import assert_no_metadata, parse_name  # noqa: E402
from fl_ids.partition import Client  # noqa: E402
from fl_ids.preprocess import chan_merge  # noqa: E402
from fl_ids.windowing import get_windows, windows_of_block  # noqa: E402


def test_parse_name():
    f = parse_name("x/train/TCP_IP-DDoS-ICMP3_train.pcap.csv")
    assert (f.attack_type, f.part, f.group, f.label, f.split_src) == ("TCP_IP-DDoS-ICMP", 3, "DDoS", 1, "train")
    f = parse_name("x/Benign_test.pcap_Flow.csv")
    assert (f.attack_type, f.label, f.split_src) == ("Benign", 0, "test")
    assert parse_name("MQTT-DDoS-Connect_Flood_train.pcap.csv").group == "MQTT"


def test_metadata_guard():
    assert_no_metadata(["Flow Duration__mean", "Idle Mean__std", "Protocol__mean", "FWD Init Win Bytes__skew"])
    with pytest.raises(ValueError):
        assert_no_metadata(["Flow Duration__mean", "Window_Index__mean"])


def test_windows_shape():
    X = np.random.default_rng(0).normal(size=(5000, 67)).astype(np.float32)
    W = windows_of_block(X, 512, 8)
    assert W.shape == (562, 335) and not np.isnan(W).any()


def test_chan_merge_exact():
    rng = np.random.default_rng(0)
    parts = [rng.normal(size=(n, 3)) for n in (10, 50, 7)]
    n, mean, M2 = chan_merge([(len(p), p.mean(0), ((p - p.mean(0)) ** 2).sum(0)) for p in parts])
    allx = np.concatenate(parts)
    assert np.allclose(mean, allx.mean(0)) and np.allclose(M2 / n, allx.var(0))


def _clients(rng, K=3, n=400, D=8):
    out = []
    for k in range(K):
        y = rng.integers(0, 2, n)
        X = rng.normal(size=(n, D)).astype(np.float32)
        X[:, 0] += 2 * y
        out.append(Client(k, X, y, X[:50], y[:50], np.array(["g"] * 50), np.arange(n), np.arange(50)))
    return out


def test_fed_chi2_equals_central():
    rng = np.random.default_rng(1)
    cl = _clients(rng)
    allX = np.concatenate([c.Xtr for c in cl])
    gmin, gmax = allX.min(0), allX.max(0)
    ctx = fsel.FSContext(cl, gmin, gmax, load_config()["fs"], 0)
    from sklearn.feature_selection import chi2
    Xn = (allX - gmin) / (gmax - gmin)
    ref, _ = chi2(Xn, np.concatenate([c.ytr for c in cl]))
    assert np.allclose(fsel.fed_chi2_scores(ctx), ref, rtol=1e-4)
    assert fsel.select("fed_mi", ctx)[0] == 0


def test_bagging_margin_additive():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] > 0).astype(int)
    F = rng.normal(size=300)
    p = {"objective": "binary:logistic", "base_score": 0.5, "max_depth": 3, "verbosity": 0}
    b = xgb.train(p, xgb.DMatrix(X, label=y, base_margin=F), 3)
    full = xgb.DMatrix(X, base_margin=F)
    assert np.allclose(b.predict(full, output_margin=True), F + b.predict(xgb.DMatrix(X), output_margin=True),
                       atol=1e-5)


def test_smoke_end_to_end(tmp_path):
    data = tmp_path / "data"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic.py"), "--out", str(data), "--scale", "0.5"],
                   check=True)
    cfg = load_config(ROOT / "configs/smoke.yaml", [
        f"data.root={data}", f"paths.cache_dir={tmp_path/'cache'}", f"paths.out_dir={tmp_path/'out'}",
        "partition.schemes=[iid, filepart]", "models.methods=[central_xgb, fedxgb_bag]",
        "fs.methods=[full, fed_chi2, rank_chi2_median]", "models.local_xgb=false"])
    ws = get_windows(cfg, verbose=False)
    tr, va, te = (set(ws.block[ws.split == s]) for s in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    from fl_ids.experiment import run
    import pandas as pd
    df = pd.read_csv(run(cfg, ws, verbose=False))
    assert len(df) == 2 * 2 * (1 + 2 * 2) * 2 and df["macro_f1"].notna().all()
