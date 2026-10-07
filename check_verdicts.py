"""Calibration checks for the report's reading rules (not statistical tests; see audit.py).

1. False alarms of the split-gap reading. Features are pure noise (no label and no session information) while the
   label is tied to the session (20 sessions, 80% good in half of them and 20% good in the other half), the
   confounded design the tool is meant for. Pooled out-of-fold predictions on held-out sessions then fall below
   chance, so a held-out "drop" appears without any shortcut. Counts, over seeds, how often the split-gap reading
   fires with the gap rule alone (the rule before the label-signal guard) and with the guard.
2. Two-level nuisance. A chip lot with two levels that the features identify almost perfectly: the nuisance-probe
   reading must be "shortcut capacity present" (with the old rule, twice chance = 1.0 was unreachable).
3. Positive control. The quickstart design (examples/make_synthetic.py: weak label signal, strong session signal,
   sessions confounded with the label): the split-gap reading must still fire.
4. Scope of the guard: weak label signal, no session information. Same confounded design as check 1, but the
   features now carry a weak label signal (label_sig 0.3, 0.5, 0.8 on one feature) and still no session information,
   so the features offer no shortcut. The guard passes (there is a label signal to lose), and the split-gap reading
   can still fire from the pooled-prediction bias alone. Counts how often, over seeds.

Run: python check_verdicts.py [--seeds 20] [--perm 5]      -> out/check_verdicts.json and a summary on screen
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit  # noqa: E402
from paths import OUT_DIR  # noqa: E402


def noise_table(seed, n=1200, n_sessions=20, d=12, label_sig=0.0, session_sig=0.0):
    """Same generator as examples/make_synthetic.py; with both signals at 0 the features are pure noise."""
    rng = np.random.default_rng(seed)
    session = rng.integers(0, n_sessions, n)
    p_good = np.where(np.arange(n_sessions) % 2 == 0, 0.8, 0.2)
    good = rng.random(n) < p_good[session]
    X = rng.normal(size=(n, d))
    X[:, 0] += label_sig * good
    X += session_sig * rng.normal(size=(n_sessions, d))[session]
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    df["quality"] = np.where(good, "good", "bad")
    df["session"] = [f"S{s:02d}" for s in session]
    return df


WEAK_SIGNALS = (0.3, 0.5, 0.8)


def quiet(_msg):
    pass


def false_alarms(seeds, perm, label_sig=0.0):
    rows = []
    for s in range(seeds):
        res = audit.run_audit(noise_table(s, label_sig=label_sig), "quality", ["session"], perm=perm, inject=False, log=quiet)
        rnd, ho = res["split_compare"][0], res["split_compare"][1]
        (_, reading, flagged), = audit.split_flags(res)
        rows.append({"seed": s, "random": rnd["bal_acc"], "random_null": rnd["null"]["null_mean"],
                     "held_out": ho["bal_acc"], "held_out_null": ho["null"]["null_mean"],
                     "gap": ho["gap_vs_random"], "gap_rule_only": bool(ho["gap_vs_random"] <= -audit.GAP_THRESHOLD),
                     "with_guard": bool(flagged)})
    t = pd.DataFrame(rows)
    return {"seeds": seeds, "perm": perm, "label_sig": label_sig,
            "design": "1200 samples, 12 features with no session information "
            f"(label signal {label_sig} on one feature), 20 sessions, P(good) 0.8 / 0.2 alternating by session",
            "random_mean": float(t.random.mean()), "random_null_mean": float(t.random_null.mean()),
            "gap_mean": float(t.gap.mean()),
            "flagged_gaps": [float(g) for g in t.gap[t.with_guard]], "held_out_mean": float(t.held_out.mean()),
            "held_out_min": float(t.held_out.min()),
            "fires_gap_rule_only": int(t.gap_rule_only.sum()), "fires_with_guard": int(t.with_guard.sum()),
            "runs": rows}


def two_level(perm, seed=0, n=1200):
    rng = np.random.default_rng(seed)
    df = noise_table(seed, n=n, label_sig=0.8)
    lot = rng.integers(0, 2, n)
    X = df[[f"f{i}" for i in range(12)]].to_numpy()
    X[:, 1] += 4.0 * lot                                       # the lot is almost perfectly readable
    df[[f"f{i}" for i in range(12)]] = X
    df["chip_lot"] = [f"L{v}" for v in lot]
    res = audit.run_audit(df, "quality", ["chip_lot"], perm=perm, log=quiet)
    r = res["nuisance"]["chip_lot"]
    return {"bal_acc": r["bal_acc"], "chance": r["chance"], "null_max": r["null"]["null_max"],
            "old_rule_2x_chance": bool(r["bal_acc"] >= 2 * r["chance"]), "new_rule": bool(audit.nuisance_flag(r)),
            "confound_injection": res["confound_injection"]["chip_lot"]}


def positive_control(perm, seed=0):
    df = noise_table(seed, label_sig=0.8, session_sig=1.5)
    res = audit.run_audit(df, "quality", ["session"], perm=perm, inject=False, log=quiet)
    (row, reading, flagged), = audit.split_flags(res)
    return {"random": res["split_compare"][0]["bal_acc"], "random_null": res["split_compare"][0]["null"]["null_mean"],
            "held_out": row["bal_acc"], "gap": row["gap_vs_random"], "reading": reading, "flagged": bool(flagged)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--perm", type=int, default=5)
    a = ap.parse_args()
    out = {"false_alarms": false_alarms(a.seeds, a.perm), "two_level_nuisance": two_level(a.perm),
           "positive_control": positive_control(a.perm),
           "weak_signal_no_session": [false_alarms(a.seeds, a.perm, label_sig=v) for v in WEAK_SIGNALS]}
    fa, tl, pc = out["false_alarms"], out["two_level_nuisance"], out["positive_control"]
    print(f"pure noise, label tied to session, {fa['seeds']} seeds: random {fa['random_mean']:.3f}, held-out mean "
          f"{fa['held_out_mean']:.3f} (min {fa['held_out_min']:.3f})")
    print(f"  split-gap reading fires: gap rule alone {fa['fires_gap_rule_only']}/{fa['seeds']}, "
          f"with label-signal guard {fa['fires_with_guard']}/{fa['seeds']}")
    print(f"two-level chip lot: nuisance probe {tl['bal_acc']:.3f} (chance {tl['chance']:.3f}, null max "
          f"{tl['null_max']:.3f}); old rule {tl['old_rule_2x_chance']}, new rule {tl['new_rule']}; "
          f"confound injection: {tl['confound_injection'].get('skipped', 'run')}")
    print(f"positive control: random {pc['random']:.3f} (null {pc['random_null']:.3f}), held-out {pc['held_out']:.3f} "
          f"({pc['gap']:+.3f}): {pc['reading']}")
    print(f"weak label signal, no session information in the features, label tied to session, {a.seeds} seeds:")
    print("  label_sig  random  null   held-out  mean gap  gap rule alone  with guard (false alarms)")
    for w in out["weak_signal_no_session"]:
        print(f"  {w['label_sig']:<9.1f}  {w['random_mean']:.3f}   {w['random_null_mean']:.3f}  {w['held_out_mean']:.3f}"
              f"     {w['gap_mean']:+.3f}    {w['fires_gap_rule_only']:>2}/{w['seeds']}           "
              f"{w['fires_with_guard']:>2}/{w['seeds']}"
              + (f"  (flagged gaps {min(w['flagged_gaps']):+.3f} to {max(w['flagged_gaps']):+.3f})"
                 if w["flagged_gaps"] else ""))
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "check_verdicts.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
