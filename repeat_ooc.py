"""Repeat-split interval for the OoC headline (quality 0.818 -> 0.734).

The random split re-seeds StratifiedKFold; the held-out-date split reshuffles which dates go to which fold for each
seed (GroupKFold has no seed and always gives the same assignment). Table, classifier and metric are as in audit.py.

Run: python repeat_ooc.py out/ooc_audit_table.parquet [--seeds 10] [--out out/ooc_repeat.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shortcut_audit as sa  # noqa: E402


def shuffled_group_splits(groups, k, seed):
    g = np.asarray(groups)
    ug = np.random.default_rng(seed).permutation(np.unique(g))
    fold_of = {u: i % k for i, u in enumerate(ug)}
    f = np.array([fold_of[v] for v in g])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "ooc_repeat.json"))
    a = ap.parse_args()
    df = pd.read_parquet(a.table)
    feats = [c for c in df.columns if c.startswith("dino_")]
    X = df[feats].to_numpy(np.float32)
    y = df.quality.astype(str).to_numpy()
    rnd, held = [], []
    for s in range(a.seeds):
        rnd.append(sa._cv_score(X, y, sa.random_splits(y, 5, seed=s)))
        held.append(sa._cv_score(X, y, shuffled_group_splits(df.session, 5, s)))
        print(s, round(rnd[-1], 3), round(held[-1], 3), flush=True)

    def summ(v):
        v = np.asarray(v)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max()),
                "values": v.tolist()}
    res = {"n": int(len(y)), "seeds": a.seeds, "random": summ(rnd), "held_out_session": summ(held),
           "gap": summ(np.array(held) - np.array(rnd))}
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "values"})
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
