"""Progress and remaining-time estimate of the NRF queue (scripts/nrf_repro.sh).   python scripts/nrf_status.py [--out outputs/nrf]

Remaining time = (unfinished base models) x (epochs per base model) x (seconds per epoch) + (unfinished finetune runs) x
(seconds per run). Epochs per base model: mean over finished base models; before any has finished, --epochs_guess.
Seconds per finetune run: mean over finished runs with the same n; before any, --ft_guess seconds x n / 100.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs/nrf")
ap.add_argument("--epochs_guess", type=float, default=25)
ap.add_argument("--ft_guess", type=float, default=240)
a = ap.parse_args()
out = Path(a.out)


def log(p):
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


sec, n_ep, left_known = [], [], 0.0
print("base models")
for s in (1, 2, 5, 7):
    d, L = out / f"base_s{s}", log(out / f"base_s{s}/log.jsonl")
    sec += [r["sec"] for r in L]
    if (d / "done.json").exists():
        n_ep.append(len(L))
        print(f"  s{s}: done, {len(L)} epochs, {json.loads((d / 'done.json').read_text())}")
    elif L:
        r = L[-1]
        print(f"  s{s}: epoch {r['epoch']}, val_corr {r['val_corr']:.4f}, best {r['best']:.4f} at epoch {r['best_ep']}, "
              f"{r['epoch'] - r['best_ep']}/5 epochs without improvement, {np.mean([x['sec'] for x in L]) / 60:.1f} min/epoch (train part)")
    else:
        print(f"  s{s}: not started")
ep_sec = (np.mean(sec) if sec else 290.0) + 55.0                       # + validation on 600 images (0.089 s / image)
ep_total = np.mean(n_ep) if n_ep else a.epochs_guess
base_left = 0.0
for s in (1, 2, 5, 7):
    if not (out / f"base_s{s}/done.json").exists():
        done_ep = len(log(out / f"base_s{s}/log.jsonl"))
        base_left += max(ep_total - done_ep, 5) * ep_sec           # at least the 5 patience epochs remain
print(f"  estimate: {ep_sec / 60:.1f} min/epoch, {ep_total:.0f} epochs/model ({'measured' if n_ep else 'guess'}) -> {base_left / 3600:.1f} h left")

print("finetune runs (3 sources x 3 support sets per cell)")
ft_left = 0.0
for n in (30, 100, 300, 1000):
    dur, done = [], 0
    for H in (1, 2, 5, 7):
        for p in (out / f"ft_h{H}").glob(f"n{n}_rep*_src*"):
            if (p / "done.json").exists():
                done += 1
                L = log(p / "log.jsonl")
                dur.append(sum(r["sec"] for r in L) + len(L) * max(round(0.2 * n), 5) * 0.089 + (n + 1000) * 0.089 + 10)
    per = np.mean(dur) if dur else a.ft_guess * n / 100
    ft_left += (36 - done) * per
    print(f"  n={n}: {done}/36 done, {per / 60:.1f} min/run ({'measured' if dur else 'guess'})")
print(f"  estimate: {ft_left / 3600:.1f} h left")
total = base_left + ft_left
print(f"total left {total / 3600:.1f} h -> finish around {time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time() + total))}"
      + ("" if (out / 'results.csv').exists() else "; results.csv not written yet"))
