#!/usr/bin/env python
"""Sinh CSV giả có cùng dạng CICFlowMeter/CICIoMT2024 để chạy thử pipeline (không phải dữ liệu thật).

  python scripts/make_synthetic.py --out data/synthetic
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FEATS = ["Protocol", "Flow Duration", "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet",
         "Fwd Packet Length Mean", "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
         "Flow IAT Std", "Fwd IAT Total", "SYN Flag Count", "ACK Flag Count", "Packet Length Mean",
         "Packet Length Variance", "Average Packet Size", "FWD Init Win Bytes", "Active Mean", "Idle Mean"]

FILES = {  # tên gốc: (số dòng train mỗi phần, số phần train, số dòng test, độ lệch tín hiệu)
    "Benign": (12000, 1, 6000, 0.0),
    "ARP_Spoofing": (3000, 1, 1500, 0.6),
    "MQTT-DDoS-Connect_Flood": (3000, 1, 1500, 0.8),
    "MQTT-Malformed_Data": (2500, 1, 1200, 0.4),
    "Recon-OS_Scan": (2500, 1, 1200, 0.5),
    "Recon-Port_Scan": (2500, 1, 1500, 0.5),
    "TCP_IP-DDoS-ICMP": (2000, 4, 1500, 1.0),
    "TCP_IP-DoS-SYN": (2000, 4, 1500, 0.9),
}


def make(n, shift, rng, t0):
    base = rng.lognormal(0, 1, size=(n, len(FEATS)))
    sig = rng.random(len(FEATS)) < 0.4
    base[:, sig] *= 1 + shift * rng.random(sig.sum())
    df = pd.DataFrame(base, columns=FEATS)
    df["Protocol"] = rng.choice([6, 17] if shift == 0 else [6, 17, 1], size=n)
    df["SYN Flag Count"] = rng.integers(0, 2 + int(3 * shift), size=n)
    df.loc[rng.random(n) < 0.001, "Flow Bytes/s"] = np.inf
    ts = [t0 + timedelta(milliseconds=int(x)) for x in np.cumsum(rng.exponential(20, n))]
    ids = pd.DataFrame({
        "Flow ID": [f"f{i}" for i in range(n)], "Src IP": "192.168.0.2", "Src Port": rng.integers(1024, 65535, n),
        "Dst IP": "192.168.0.10", "Dst Port": rng.choice([80, 443, 1883], n),
        "Timestamp": [t.strftime("%d/%m/%Y %I:%M:%S %p") for t in ts]})
    out = pd.concat([ids, df], axis=1)
    out["Label"] = "No Label"
    return out.sample(frac=1, random_state=0)  # xáo thứ tự dòng: pipeline phải tự sắp theo Timestamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="nhân số dòng")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = Path(a.out)
    for sp in ("train", "test"):
        (root / sp).mkdir(parents=True, exist_ok=True)
    t0 = datetime(2024, 1, 1)
    for name, (ntr, parts, nte, shift) in FILES.items():
        for p in range(1, parts + 1):
            stem = f"{name}{p}" if parts > 1 else name
            make(int(ntr * a.scale), shift, rng, t0).to_csv(root / "train" / f"{stem}_train.pcap.csv", index=False)
        stem = f"{name}1" if parts > 1 else name
        make(int(nte * a.scale), shift, rng, t0).to_csv(root / "test" / f"{stem}_test.pcap.csv", index=False)
    print(f"Đã tạo dữ liệu giả ở {root}")


if __name__ == "__main__":
    main()
