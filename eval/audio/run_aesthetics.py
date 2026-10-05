"""Supplied Audiobox runner, limited to the paper's PQ, CE and CU outputs."""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h3audio_core as core  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", required=True, help="dir with <column>/<stem>.mp4")
    ap.add_argument("--out", required=True, help="parts dir")
    ap.add_argument("--ckpt", required=True, help="audiobox checkpoint.pt")
    ap.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    ap.add_argument("--world", type=int, default=int(os.environ.get("WORLD", 1)))
    ap.add_argument("--columns", default="", help="comma list; default = all in dataset")
    ap.add_argument("--limit", type=int, default=0, help="first N stems only (smoke)")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    torch.set_num_threads(2)
    core.load_dataset(a.dataset)
    cols = [c for c in a.columns.split(",") if c] or None
    units = core.work_units(a.root, cols, a.limit)[a.rank::a.world]
    log(f"rank {a.rank}/{a.world}: {len(units)} clips")

    from audiobox_aesthetics.infer import AesPredictor

    pred = AesPredictor(checkpoint_pth=a.ckpt, batch_size=a.batch)
    log(f"model on {pred.device}")

    w = core.ShardWriter(a.out, a.rank)
    n_fail = 0
    t0 = time.time()
    for i in range(0, len(units), a.batch):
        chunk = units[i:i + a.batch]
        batch, meta = [], []
        for col, stem, path in chunk:
            try:
                x, sr = core.decode_audio(path)
                batch.append({"path": torch.from_numpy(x), "sample_rate": sr})
                meta.append((col, stem, None, None))
            except Exception as e:  # a broken clip is a record, not a crash
                meta.append((col, stem, None, repr(e)))
        scores = pred.forward(batch) if batch else []
        j = 0
        for col, stem, st, err in meta:
            rec = {"col": col, "stem": stem}
            if err is not None:
                rec["error"] = err
                n_fail += 1
            else:
                s = scores[j]
                j += 1
                rec.update({k: round(float(s[k]), 4) for k in ("CE", "CU", "PQ")})
            w(rec)
        if (i // a.batch) % 10 == 0:
            log(f"{w.n}/{len(units)}  {w.n / (time.time() - t0 + 1e-9):.1f} clip/s  fail={n_fail}")
    w.close()
    log(f"RANK_DONE rank={a.rank} n={w.n} fail={n_fail}")


if __name__ == "__main__":
    main()
