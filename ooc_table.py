"""OOC feature npz + datasheet -> one table that audit.py can read (input for the demo and the README example).

    python ooc_table.py [--feats out/ooc_feats_dinov2-small.npz] [--out out/ooc_audit_table.parquet]
    python audit.py out/ooc_audit_table.parquet --label quality --nuisance session cell_type --group session \
        --title "OOC brightfield quality classifier" --out out/ooc_audit_report.html
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ooc_probe import load  # noqa: E402
from paths import OUT_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=os.path.join(OUT_DIR, "ooc_feats_dinov2-small.npz"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "ooc_audit_table.parquet"))
    a = ap.parse_args()
    df, D, _, _ = load(a.feats)
    t = pd.DataFrame(D, columns=[f"dino_{i}" for i in range(D.shape[1])])
    t["quality"] = df.label.values
    t["session"] = df.session.values
    t["cell_type"] = df["cell type"].astype(str).values
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t.to_parquet(a.out, index=False)
    print("saved", a.out, t.shape)


if __name__ == "__main__":
    main()
