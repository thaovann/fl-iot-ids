"""B2: tấn công chiếm chỗ vào chọn đặc trưng χ² và tác hại lên detector. Xem docs/B2.md.

  python scripts/run_b2.py synth  --out data/b2_synth.npz
  python scripts/run_b2.py oracle --data ciciot_512_8.npz --out results/b2/oracle.jsonl
  python scripts/run_b2.py fed    --data ciciot_512_8.npz --target DoS --out results/b2/fed_DoS.jsonl
  python scripts/run_b2.py report --oracle results/b2/oracle.jsonl --fed results/b2/fed_*.jsonl --out results/b2/BAO_CAO.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fl_ids import fs_poison as fp  # noqa: E402

K_GRID = [5, 10, 20]                 # chốt 19/11: 5 = điểm vận hành bài v1, 10 và 20 = cấu hình phụ/chính
R_GRID = [1, 2, 3, 5, 10]            # số chỗ tốt còn lại, giao với [1, k]
SEEDS = list(range(42, 52))


# ------------------------------------------------------------------ tiện ích
class Detector:
    """Huấn luyện XGB trên một tập đặc trưng; nhớ kết quả theo (tập, seed) để không huấn luyện lại."""

    def __init__(self, D, seed_frac, n_jobs):
        self.D, self.seed_frac, self.n_jobs = D, seed_frac, n_jobs
        self.enc = {c: i for i, c in enumerate(D.classes)}
        self.ytr = np.array([self.enc[c] for c in D.ytr])
        self.yev = np.array([self.enc.get(c, -1) for c in D.yev])
        self.cache = {}
        self._idx = {}

    def _train_idx(self, seed):
        if self.seed_frac >= 1:
            return slice(None)
        if seed not in self._idx:                       # rút phân tầng theo họ, cố định theo seed
            rng = np.random.default_rng(seed)
            keep = [rng.choice(np.where(self.ytr == c)[0], max(1, int(round((self.ytr == c).sum() * self.seed_frac))),
                               replace=False) for c in np.unique(self.ytr)]
            self._idx[seed] = np.sort(np.concatenate(keep))
        return self._idx[seed]

    def __call__(self, feats, seed):
        key = (tuple(sorted(int(f) for f in feats)), seed)
        if key not in self.cache:
            f = list(key[0])
            ix = self._train_idx(seed)
            ok = self.yev >= 0
            P = fp.fit_predict_proba(self.D.Xtr[ix][:, f], self.ytr[ix], self.D.Xev[ok][:, f],
                                     len(self.D.classes), seed, self.n_jobs)
            self.cache[key] = fp.detector_metrics(P, self.yev[ok], self.D.classes)
        return self.cache[key]


def prep(a):
    D = fp.load_npz(a.data, a.benign, a.split, a.max_train_rows)
    gmin, gmax = D.Xtr.min(0), D.Xtr.max(0)
    Xn = fp.minmax(D.Xtr, gmin, gmax)
    ranking = fp.clean_ranking(Xn, D.ytr, D.classes)
    fams = fp.eligible_families(D)
    print(f"[b2] {len(D.Xtr)} cửa sổ train, {len(D.Xev)} cửa sổ {a.split}, {Xn.shape[1]} đặc trưng; "
          f"họ: {D.classes}; họ được báo (G1): {fams}", file=sys.stderr)
    return D, Xn, ranking, fams


def write(fh, row):
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()


def done_keys(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(line)["key"] for line in open(path) if line.strip()}


# ------------------------------------------------------------------ oracle B2
def cmd_oracle(a):
    D, Xn, ranking, fams = prep(a)
    targets = a.targets or fams
    det = Detector(D, a.seed_frac, a.n_jobs)
    clF = {}
    for t in targets:
        F = fp.target_features(Xn, D.ytr, t, D.benign, a.nF)
        clF[t] = fp.closure(F, D.names, D.Xtr, a.rho)
        print(f"[b2] {t}: |cl(F*)| = {len(clF[t])}", file=sys.stderr)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    seen = done_keys(a.out)
    rank = [int(f) for f in ranking]
    with open(a.out, "a") as fh:
        for seed in a.seeds:
            for k in a.k:
                jobs = [("clean", None, k, rank[:k])]
                for r in [r for r in R_GRID if r <= k]:
                    j = k - r
                    top_r = rank[:r]
                    jobs.append(("a_top_r", None, r, top_r))
                    jobs.append(("b_top_r_junk", None, r, top_r + fp.junk_features(rank, j, top_r)))
                    for t in targets:
                        good = [f for f in rank if f not in set(clF[t])][:r]
                        jobs.append(("c_no_clF_junk", t, r, good + fp.junk_features(rank, j, set(good) | set(clF[t]))))
                for branch, t, r, feats in jobs:
                    key = f"oracle|{seed}|{k}|{r}|{branch}|{t}"
                    if key in seen:
                        continue
                    t0 = time.time()
                    m = det(feats, seed)
                    write(fh, dict(key=key, kind="oracle", seed=seed, k=k, r=r, j=k - r, branch=branch, target=t,
                                   n_feats=len(set(feats)), sec=round(time.time() - t0, 1), **m))
            print(f"[b2] oracle seed {seed} xong", file=sys.stderr)


# ------------------------------------------------------------------ CSP-chiếm-chỗ liên kết
def cmd_fed(a):
    D, Xn, ranking, fams = prep(a)
    t = a.target
    F = fp.target_features(Xn, D.ytr, t, D.benign, a.nF)
    rank = [int(f) for f in ranking]
    det = Detector(D, a.seed_frac, a.n_jobs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    seen = done_keys(a.out)
    ti = D.classes.index(t)
    with open(a.out, "a") as fh:
        for part in a.parts:
            for seed in a.seeds:
                rng = np.random.default_rng(seed)
                root, root_idx = fp.make_root(Xn, D.ytr, D.classes, a.root_per_class, rng)
                rest = np.setdiff1d(np.arange(len(D.ytr)), root_idx)
                parts = [rest[p] for p in fp.partition(D.ytr[rest], D.classes, D.benign, a.K, part, rng)]
                S0, N0 = fp.client_msgs(Xn, D.ytr, parts, D.classes)
                holders = [k for k in range(a.K) if N0[k, ti] > 0]
                order = holders + [k for k in rng.permutation(a.K) if k not in holders]
                for k in a.k:
                    J = fp.junk_features(rank, k, F)            # kẻ tấn công cố lấp cả k chỗ bằng rác
                    clean_central = rank[:k]
                    for m in a.m:
                        mal = [int(c) for c in order[:int(round(m * a.K))]]
                        S = fp.csp_hijack(S0, N0, mal, F, J) if mal else S0
                        for agg in a.aggs:
                            key = f"fed|{t}|{part}|{seed}|{k}|{m}|{agg}"
                            if key in seen:
                                continue
                            sc = fp.AGGREGATORS[agg](S, N0, root=root)
                            sel = [int(f) for f in np.argsort(-sc, kind="stable")[:k]]
                            row = dict(key=key, kind="fed", target=t, part=part, seed=seed, k=k, m=m, agg=agg,
                                       n_mal=len(mal), sel_F=len(set(sel) & set(F)), sel_J=len(set(sel) & set(J)),
                                       jaccard_central=len(set(sel) & set(clean_central)) / len(set(sel) | set(clean_central)))
                            if seed in a.det_seeds:
                                row.update(det(sel, seed))
                            write(fh, row)
                print(f"[b2] fed {t} {part} seed {seed} xong", file=sys.stderr)


# ------------------------------------------------------------------ báo cáo
def _mean_sd(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return (np.mean(v), np.std(v), len(v)) if v else (np.nan, np.nan, 0)


def report_oracle(rows, thr):
    lines = ["## Oracle B2: trần của tấn công chiếm chỗ (tập trung, XGB đa lớp)", "",
             "Δ = giá trị top-k sạch − giá trị của tập bị chiếm, cùng k và cùng seed (dương = bị hại). "
             f"Ô có **Δ ≥ {thr:.2f}** là đạt ngưỡng tác hại đã chốt.", ""]
    base = {(r["seed"], r["k"]): r for r in rows if r["branch"] == "clean"}
    fams = sorted({f for r in rows for f in r["recall"]})
    hits = []
    for metric, label in [("recall", "recall đa lớp theo họ"), ("tpr_at_fpr_family", f"TPR tại FPR {fp.FIXED_FPR:.0%} theo họ")]:
        lines += [f"### Δ {label}", "", "| k | r (chỗ tốt) | j (rác) | nhánh | đích (c) | " + " | ".join(fams) + " |",
                  "|" + "---|" * (5 + len(fams))]
        g = defaultdict(list)
        for r in rows:
            if r["branch"] != "clean" and (r["seed"], r["k"]) in base:
                g[(r["k"], r["r"], r["branch"], r["target"] or "")].append(r)
        for key in sorted(g):
            cells = []
            for f in fams:
                d = [base[(x["seed"], x["k"])][metric].get(f, np.nan) - x[metric].get(f, np.nan) for x in g[key]]
                mu, sd, n = _mean_sd(d)
                if np.isnan(mu):
                    cells.append("–")
                    continue
                s = f"{mu:+.3f}±{sd:.3f}"
                if mu >= thr:
                    s = f"**{s}**"
                    hits.append((metric, key, f, mu))
                cells.append(s)
            k, r, br, tg = key
            lines.append(f"| {k} | {r} | {k - r} | {br} | {tg} | " + " | ".join(cells) + " |")
        lines.append("")
    small = [h for h in hits if h[1][0] <= 10 and h[1][2] != "a_top_r"]
    lines += ["### Kết luận tự động (oracle)", ""]
    if small:
        lines.append(f"- **CÓ** ô đạt ngưỡng ở k ≤ 10 trong nhánh có rác (b/c): {len(small)} ô. Ví dụ: " + "; ".join(
            f"k={h[1][0]}, r={h[1][1]}, {h[1][2]}, {h[2]}: Δ{'recall' if h[0] == 'recall' else 'TPR@FPR'} = {h[3]:.3f}"
            for h in sorted(small, key=lambda h: -h[3])[:5]))
        lines.append("- Trần cho phép threat thật ở k nhỏ. Bước tiếp: xem phần liên kết, tấn công thật có chạm trần không.")
    else:
        lines.append("- **KHÔNG** ô nào ở k ≤ 10 (nhánh b/c) đạt ngưỡng. Theo quy tắc đã chốt, không tấn công chiếm chỗ "
                     "nào gây tác hại ≥ 0,10 ở cấu hình này: định vị bài thành phân tích độ bền (docs/B2.md, nhánh 2).")
    only_a = [h for h in hits if h[1][2] == "a_top_r"]
    if only_a:
        lines.append(f"- Nhánh (a) chỉ top-r có {len(only_a)} ô đạt ngưỡng: so (a) với (b) để biết tác hại do mất chỗ hay do rác.")
    return lines, bool(small)


def report_fed(rows, thr):
    lines = ["## CSP-chiếm-chỗ liên kết", "",
             "sel_F = |top-k ∩ F*|, sel_J = |top-k ∩ J| (rác do kẻ tấn công đẩy vào). Δ so với chính bộ gộp đó ở m = 0, cùng seed.", ""]
    g = defaultdict(list)
    for r in rows:
        g[(r["target"], r["part"], r["k"], r["agg"], r["m"])].append(r)
    clean = {(r["target"], r["part"], r["k"], r["agg"], r["seed"]): r for r in rows if r["m"] == 0}
    hits = []
    lines += ["| đích | phân vùng | k | bộ gộp | m | sel_F | sel_J | Δrecall đích | ΔTPR@FPR đích | Δrecall lớn nhất (họ) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(g):
        t, part, k, agg, m = key
        rs = g[key]
        sF, sJ = np.mean([r["sel_F"] for r in rs]), np.mean([r["sel_J"] for r in rs])
        dr, dt, worst = [], [], defaultdict(list)
        for r in rs:
            c = clean.get((t, part, k, agg, r["seed"]))
            if c is None or "recall" not in r or "recall" not in c:
                continue
            dr.append(c["recall"].get(t, np.nan) - r["recall"].get(t, np.nan))
            dt.append(c["tpr_at_fpr_family"].get(t, np.nan) - r["tpr_at_fpr_family"].get(t, np.nan))
            for f in r["recall"]:
                worst[f].append(c["recall"].get(f, np.nan) - r["recall"][f])
        mr, _, n = _mean_sd(dr)
        mt, _, _ = _mean_sd(dt)
        wf = max(worst, key=lambda f: np.nanmean(worst[f])) if worst else None
        wv = np.nanmean(worst[wf]) if wf else np.nan

        def fmt(x):
            return "–" if np.isnan(x) else (f"**{x:+.3f}**" if x >= thr else f"{x:+.3f}")
        if m > 0 and max(np.nan_to_num([mr, mt, wv], nan=-1)) >= thr:
            hits.append(key)
        lines.append(f"| {t} | {part} | {k} | {agg} | {m:.0%} | {sF:.1f} | {sJ:.1f} | {fmt(mr)} | {fmt(mt)} | "
                     f"{fmt(wv)}{f' ({wf})' if wf else ''} |")
    lines += ["", "### Kết luận tự động (liên kết)", ""]
    no_def = [h for h in hits if h[3] == "fed_chi2" and h[2] <= 10]
    if no_def:
        lines.append(f"- **CÓ threat**: {len(no_def)} cấu hình fed_chi2 ở k ≤ 10 đạt Δ ≥ {thr:.2f}. "
                     "Đi tiếp phòng thủ (rootmed) theo docs/B2.md, nhánh 1.")
        for agg in sorted({h[3] for h in hits if h[3] != "fed_chi2"}):
            lines.append(f"- Bộ gộp {agg} vẫn bị hại ở {sum(1 for h in hits if h[3] == agg)} cấu hình.")
    else:
        lines.append(f"- **KHÔNG** cấu hình fed_chi2 nào ở k ≤ 10 đạt Δ ≥ {thr:.2f}: tấn công thật chưa gây hại đáng kể.")
    return lines, bool(no_def)


def cmd_report(a):
    out = ["# Báo cáo B2 (tự sinh)", "", f"Ngưỡng tác hại: Δ ≥ {a.thr:.2f} (DROP_TARGET_RECALL, chốt trước B1). "
           "Số trên val là để quyết định hướng, chưa phải số cho bài (G3).", ""]
    verdict = {}
    if a.oracle:
        rows = [json.loads(line) for p in a.oracle for f in glob.glob(p) for line in open(f) if line.strip()]
        ls, verdict["oracle"] = report_oracle(rows, a.thr)
        out += ls + [""]
    if a.fed:
        rows = [json.loads(line) for p in a.fed for f in glob.glob(p) for line in open(f) if line.strip()]
        ls, verdict["fed"] = report_fed(rows, a.thr)
        out += ls + [""]
    out += ["## Hướng đi", ""]
    if verdict.get("oracle") and verdict.get("fed", True):
        out.append("**Nhánh 1 – threat thật.** Giữ khung tấn công + phòng thủ; làm tiếp rootmed và CSP-thích nghi (docs/B2.md).")
    elif verdict.get("oracle"):
        out.append("**Trần cho phép tác hại nhưng tấn công liên kết chưa chạm tới.** Xem lại ngân sách/m của CSP trước khi kết luận.")
    else:
        out.append("**Nhánh 2 – phân tích độ bền.** Bề mặt tấn công có thật, tác hại bị chặn bởi dư thừa đặc trưng; nhắm Q2 (docs/B2.md).")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


def cmd_synth(a):
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fp.synthetic(a.seed))
    print(f"đã ghi {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data", required=True, help=".npz: X_train, y_train, X_val, y_val, [X_test, y_test], feature_names")
        p.add_argument("--benign", default="Benign")
        p.add_argument("--split", choices=["val", "test"], default="val", help="test chỉ sau khi đóng băng (G3)")
        p.add_argument("--k", type=int, nargs="+", default=K_GRID)
        p.add_argument("--nF", type=int, default=20)
        p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
        p.add_argument("--seed-frac", type=float, default=0.8,
                       help="tỉ lệ train rút phân tầng theo seed cho XGB (1.0 = mọi seed giống hệt nhau)")
        p.add_argument("--max-train-rows", type=int, default=0, help="rút gọn train một lần (0 = giữ hết)")
        p.add_argument("--n-jobs", type=int, default=-1, help="luồng XGB; đặt cụ thể khi chạy song song")
        p.add_argument("--out", required=True)

    s = sp.add_parser("synth")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(fn=cmd_synth)

    o = sp.add_parser("oracle")
    common(o)
    o.add_argument("--targets", nargs="*", help="họ đích cho nhánh (c); mặc định mọi họ qua G1")
    o.add_argument("--rho", type=float, default=0.9)
    o.set_defaults(fn=cmd_oracle)

    f = sp.add_parser("fed")
    common(f)
    f.add_argument("--target", required=True)
    f.add_argument("--K", type=int, default=10)
    f.add_argument("--parts", nargs="+", default=["iid", "by_source"])
    f.add_argument("--m", type=float, nargs="+", default=[0, 0.1, 0.2, 0.3, 0.4])
    f.add_argument("--aggs", nargs="+", default=list(fp.AGGREGATORS))
    f.add_argument("--root-per-class", type=int, default=50)
    f.add_argument("--det-seeds", type=int, nargs="+", default=[42, 43, 44],
                   help="seed có huấn luyện detector (tập chọn đo ở mọi seed)")
    f.set_defaults(fn=cmd_fed)

    r = sp.add_parser("report")
    r.add_argument("--oracle", nargs="*")
    r.add_argument("--fed", nargs="*")
    r.add_argument("--thr", type=float, default=fp.DROP_TARGET_RECALL)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_report)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
