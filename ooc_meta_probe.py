"""Predict the quality label of the OOC Image Dataset (Movčana et al. 2024, Zenodo 10203721; CC BY 4.0 on Zenodo, CC BY-SA per the data paper) from
metadata alone.

How well do imaging date (leading part of imageID), cell type and time predict good/bad without looking at a
single image? This is the "nuisance baseline" an image classifier has to beat. Both a random split and a split
that holds out imaging dates are reported.

Run: python ooc_meta_probe.py
Input: $AI4S_DATA/ooc/OOC_datasheet.xlsx (see paths.py)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shortcut_audit as sa  # noqa: E402
from paths import OOC_XLSX as XLSX  # noqa: E402


def main():
    df = pd.read_excel(XLSX, sheet_name="Main")
    df["session"] = df.imageID.astype(str).str.split("_").str[0]
    df["label"] = df["Decision 1/2 (good/bad)"].map({1: "good", 2: "bad"})
    df = df.dropna(subset=["label"])
    print(f"images {len(df)} · imaging dates {df.session.nunique()} · cell types {df['cell type'].nunique()}")
    print("label fractions", df.label.value_counts(normalize=True).round(3).to_dict())
    by = df.groupby("session").agg(n=("label", "size"), cell=("cell type", "first"),
                                   good=("label", lambda s: (s == "good").mean()))
    print("distribution of good fraction per imaging date:", by.good.describe().round(3).to_dict())
    print("imaging dates with good fraction 0 or 1:", int(((by.good == 0) | (by.good == 1)).sum()), "/", len(by))
    for cols in (["session"], ["cell type"], ["cell type", "time after seeding, h"]):
        r = sa.label_from_nuisance(df.label, df[cols])
        g = sa.label_from_nuisance(df.label, df[cols], groups=df.session)
        print(f"metadata only {cols}: random {r['bal_acc']:.3f} · held-out date {g['bal_acc']:.3f} "
              f"(chance {r['chance']:.2f})")


if __name__ == "__main__":
    main()
