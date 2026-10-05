"""Seed repeats of the JUMP headline numbers (seed 0 = out/jump_probes_*.json, seeds 1-4 = out/seed*/).

The seed changes the well sample, the random split and the confound-injection level assignment together (the
grouped GroupKFold split changes because the sample changes).
Run: for s in 1 2 3 4; do AI4S_OUT=out/seed$s python reproduce_jump.py --seed $s; done; python summarize_seeds.py
Output: out/jump_seed_summary.json and a table on screen.
"""
import glob
import json
import os

import numpy as np

from paths import OUT_DIR as W2


def st(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max()),
            "n": int(len(v))}


def main():
    out = {}
    for tag in ("featselect", "harmony"):
        fs = [os.path.join(W2, f"jump_probes_{tag}.json")] + sorted(glob.glob(os.path.join(W2, "seed*", f"jump_probes_{tag}.json")))
        ds = [json.load(open(f)) for f in fs]
        sc = lambda i: [d["split_compare_pos8"][i]["bal_acc"] for d in ds]  # noqa: E731
        r = {"dmso_lab": st([d["nuisance_source_from_dmso"]["bal_acc"] for d in ds]),
             "dmso_lab_null": st([d["nuisance_source_from_dmso"]["null"]["null_mean"] for d in ds]),
             "pos8_random": st(sc(0)), "pos8_plate": st(sc(1)), "pos8_batch": st(sc(2)), "pos8_lab": st(sc(3)),
             "pos8_lab_gap": st(np.array(sc(3)) - np.array(sc(0)))}
        for k in ("inflated_cv", "confounded_on_heldout", "balanced_on_heldout"):
            r["inject_" + k] = st([d["confound_injection_batch"][k] for d in ds])
        out[tag] = r
        print(tag)
        for k, v in r.items():
            print(f"  {k:<32} {v['mean']:.3f} +/- {v['sd']:.3f}  ({v['min']:.3f}-{v['max']:.3f}, n={v['n']})")
    with open(os.path.join(W2, "jump_seed_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
