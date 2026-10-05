"""Check on synthetic data that the probes move in the expected direction (positive and negative controls).

- Features carry only nuisance (lab) signal, no label signal: nuisance_probe high, split_compare near chance.
- Features carry only label signal: nuisance_probe near chance, random and grouped splits both high.
Run: python test_shortcut_audit.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shortcut_audit as sa  # noqa: E402


def make(n=3000, d=20, label_sig=0.0, nuis_sig=0.0, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, 6, n)
    plate = src * 10 + rng.integers(0, 10, n)
    y = rng.integers(0, 4, n)
    X = rng.normal(size=(n, d))
    X[:, 0] += label_sig * y
    X[:, 1] += nuis_sig * src
    return X, y, src, plate


def main():
    X, y, src, plate = make(nuis_sig=2.0)
    nu = sa.nuisance_probe(X, src, groups=plate)
    sc = sa.split_compare(X, y, {"source": src})
    print("nuisance only:", round(nu["bal_acc"], 3), "/ label, random split", round(sc.bal_acc[0], 3))
    assert nu["bal_acc"] > 3 * nu["chance"] and sc.bal_acc[0] < 0.35
    X, y, src, plate = make(label_sig=2.0)
    nu = sa.nuisance_probe(X, src, groups=plate)
    sc = sa.split_compare(X, y, {"source": src})
    print("label only:", round(nu["bal_acc"], 3), "/ label, random and held-out source",
          round(sc.bal_acc[0], 3), round(sc.bal_acc[1], 3))
    assert nu["bal_acc"] < 0.25 and sc.bal_acc[1] > 0.6
    # Confounding: with nuisance signal only and no label signal, the confounded training set scores high inside
    # but chance on the held-out levels.
    rng = np.random.default_rng(1)
    n = 6000
    batch = rng.integers(0, 12, n)
    y = rng.integers(0, 4, n)
    X = rng.normal(size=(n, 10))
    X += 3.0 * rng.normal(size=(12, 10))[batch]  # each batch has its own direction
    ci = sa.confound_injection(X, y, batch)
    print("confounding:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ci.items() if k != "test_levels"})
    assert ci["inflated_cv"] > 0.8 and ci["confounded_on_heldout"] < 0.4
    # Permutation baseline: on samples with no signal the probe score must equal the null (not 1/n_classes)
    X, y, src, plate = make()
    sp = sa.group_splits(src)
    s0 = sa._cv_score(X, y, sp)
    pn = sa.permutation_null(X, y, sp, n_perm=3)
    print("permutation:", round(s0, 3), {k: round(v, 3) for k, v in pn.items()})
    assert abs(s0 - pn["null_mean"]) < 0.05
    print("PASS")


if __name__ == "__main__":
    main()
