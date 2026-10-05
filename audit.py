"""Shortcut audit entry point — give it one feature table and column names, get a one-page HTML report.

    python audit.py features.parquet --label quality --nuisance session cell_type \
        [--group session] [--control-value DMSO] [--perm 5] [--out report.html]

Input table: one row = one sample (image or well). Feature columns default to every numeric column that is not a
metadata column. csv, parquet and npz are accepted.

Reading rules printed in the report (display thresholds, not statistical tests — all numbers are shown):
  - nuisance probe: >= 2x chance AND above the permutation null max -> "shortcut capacity present"
  - label from nuisance only: >= chance + 0.05 -> "confounded: random-split scores can be inflated"
  - split gap: held-out group >= 0.05 below random split -> "part of the score leans on the shortcut"
Every score is reported next to a permutation null computed with the SAME split, because 1/n_classes is chance
only under random splits; grouped CV with no signal can land below it.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shortcut_audit as sa  # noqa: E402


def load_table(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        return pd.DataFrame({k: list(z[k]) if z[k].ndim > 1 else z[k] for k in z.files})
    return pd.read_csv(path)


def _null(r):
    return r.get("null", {}).get("null_mean", float("nan"))


def verdicts(res):
    out = []
    for name, r in res["nuisance"].items():
        flag = r["bal_acc"] >= 2 * r["chance"] and r["bal_acc"] > r.get("null", {}).get("null_max", -1)
        out.append((f"Nuisance probe: {name}", r["bal_acc"], r["chance"], _null(r),
                    "shortcut capacity present" if flag else "weak"))
    for name, r in res["label_from_nuisance"].items():
        flag = r["bal_acc"] >= r["chance"] + 0.05
        out.append((f"Label from nuisance only: {name}", r["bal_acc"], r["chance"], _null(r),
                    "confounded: random-split scores can be inflated" if flag else "weak confounding"))
    for row in res["split_compare"][1:]:
        flag = row["gap_vs_random"] <= -0.05
        out.append((f"Split gap: {row['split']}", row["bal_acc"], row["chance"], _null(row),
                    f"{row['gap_vs_random']:+.3f} vs random: "
                    + ("part of the score leans on the shortcut" if flag else "small gap")))
    return out


def render(res, path, title):
    rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{v:.3f}</td><td>{c:.3f}</td>"
        f"<td>{'n/a' if np.isnan(nl) else f'{nl:.3f}'}</td><td>{html.escape(t)}</td></tr>"
        for n, v, c, nl, t in verdicts(res))
    sc = pd.DataFrame([{k: v for k, v in r.items() if k != "null"} | {"perm_null": _null(r)}
                       for r in res["split_compare"]]).to_html(index=False, float_format="%.3f", na_rep="")
    doc = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:1em 0}}td,th{{border:1px solid #ccc;padding:6px 8px;text-align:left}}
th{{background:#f3f3f3}}</style>
<h1>{html.escape(title)}</h1>
<p>{res['n']} samples · {res['n_features']} features · label "{html.escape(res['label'])}" ·
nuisance candidates: {html.escape(', '.join(res['nuisance_cols']))}</p>
<h2>Verdicts</h2><table><tr><th>Probe</th><th>Balanced accuracy</th><th>Chance (1/k)</th>
<th>Permutation null (same split)</th><th>Reading</th></tr>{rows}</table>
<h2>Split comparison</h2>{sc}
<p>Every probe uses the same classifier (standardize + logistic regression, C=0.1) and the same metric.
Thresholds are display aids, not statistical tests. Read each score against the permutation null of its own split:
grouped cross-validation with no signal can fall below 1/k.</p>
<details><summary>Raw results (JSON)</summary><pre>{html.escape(json.dumps(res, indent=1, default=float))}</pre></details>
"""
    with open(path, "w") as f:
        f.write(doc)


def main():
    ap = argparse.ArgumentParser(description="Audit whether a classifier can read nuisance variables instead of biology.")
    ap.add_argument("table")
    ap.add_argument("--label", required=True)
    ap.add_argument("--nuisance", nargs="+", required=True)
    ap.add_argument("--group", help="grouping unit for the nuisance probe split (default: none)")
    ap.add_argument("--control-value", help="run the nuisance probe only on samples with this label (e.g. DMSO)")
    ap.add_argument("--features", nargs="*", help="feature columns (default: numeric columns minus metadata)")
    ap.add_argument("--perm", type=int, default=5, help="label permutations per probe for the null (0 = off)")
    ap.add_argument("--out", default="shortcut_audit_report.html")
    ap.add_argument("--title", default="Shortcut audit")
    a = ap.parse_args()
    df = load_table(a.table).dropna(subset=[a.label])
    meta = set([a.label] + a.nuisance + ([a.group] if a.group else []))
    feats = a.features or [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].to_numpy(np.float32)
    ok = np.isfinite(X).all(1)
    df, X = df[ok].reset_index(drop=True), X[ok]
    y = df[a.label].astype(str).to_numpy()
    group = df[a.group].astype(str).to_numpy() if a.group else None
    res = {"n": int(len(df)), "n_features": len(feats), "label": a.label, "nuisance_cols": a.nuisance,
           "nuisance": {}, "label_from_nuisance": {}}
    ctl = (y == a.control_value) if a.control_value else np.ones(len(y), bool)
    for c in a.nuisance:
        nu = df[c].astype(str).to_numpy()
        if len(np.unique(nu[ctl])) > 1:
            g = group[ctl] if (group is not None and a.group != c) else None
            r = sa.nuisance_probe(X[ctl], nu[ctl], groups=g)
            if a.perm:
                sp = sa.group_splits(g) if g is not None else sa.random_splits(nu[ctl])
                r["null"] = sa.permutation_null(X[ctl], nu[ctl], sp, a.perm)
            res["nuisance"][c] = r
        res["label_from_nuisance"][c] = sa.label_from_nuisance(y, df[[c]])
    groups = {c: df[c].astype(str).to_numpy() for c in a.nuisance}
    sc = sa.split_compare(X, y, groups).to_dict(orient="records")
    if a.perm:
        sc[0]["null"] = sa.permutation_null(X, y, sa.random_splits(y), a.perm)
        for row, (c, g) in zip(sc[1:], groups.items()):
            row["null"] = sa.permutation_null(X, y, sa.group_splits(g), a.perm)
    res["split_compare"] = sc
    render(res, a.out, a.title)
    print("report:", a.out)


if __name__ == "__main__":
    main()
