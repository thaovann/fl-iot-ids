#!/usr/bin/env python
"""Chạy một thí nghiệm.

Ví dụ:
  python scripts/run_experiment.py --config configs/ciciomt2024.yaml
  python scripts/run_experiment.py --config configs/smoke.yaml --set seeds=[42] partition.schemes=[iid]
  python scripts/run_experiment.py --config configs/ciciomt2024.yaml --only-windows   # chỉ dựng cache
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fl_ids.config import load_config  # noqa: E402
from fl_ids.experiment import run, summarize  # noqa: E402
from fl_ids.windowing import get_windows, window_counts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="ghi đè dạng khoá.con=giá_trị (YAML)")
    ap.add_argument("--only-windows", action="store_true", help="chỉ dựng/cache cửa sổ rồi dừng")
    ap.add_argument("--rebuild-windows", action="store_true", help="bỏ qua cache cửa sổ")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    ws = get_windows(cfg, force=a.rebuild_windows)
    if a.only_windows:
        print(window_counts(ws).to_string())
        return
    path = run(cfg, ws)
    print(f"\nKết quả: {path}")
    print(summarize(path).round(4).to_string())


if __name__ == "__main__":
    main()
