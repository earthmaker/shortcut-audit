"""Write a small synthetic feature table for trying audit.py (no downloads needed).

    python examples/make_synthetic.py [--out out/synthetic_audit.csv]

The table has a weak label signal (quality: good/bad) plus a strong imaging-session signal, and sessions are
partly confounded with the label (some sessions are mostly good, others mostly bad). A sound audit should show:
sessions are predictable from the features, the label is partly predictable from the session alone, and the
held-out-session score is lower than the random-split score.
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "out", "synthetic_audit.csv"))
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n_sessions, d = 20, 12
    session = rng.integers(0, n_sessions, a.n)
    p_good = np.where(np.arange(n_sessions) % 2 == 0, 0.8, 0.2)  # confounded sessions
    good = rng.random(a.n) < p_good[session]
    cell = (session % 3).astype(int)
    X = rng.normal(size=(a.n, d))
    X[:, 0] += 0.8 * good                                  # weak label signal
    X += 1.5 * rng.normal(size=(n_sessions, d))[session]   # session-specific offset (the shortcut)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    df["quality"] = np.where(good, "good", "bad")
    df["session"] = [f"S{s:02d}" for s in session]
    df["cell_type"] = [f"C{c}" for c in cell]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print("saved", os.path.abspath(a.out), df.shape)


if __name__ == "__main__":
    main()
