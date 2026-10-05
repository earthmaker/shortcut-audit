"""Run the shortcut-audit probes on JUMP cpg0016 positive-control and DMSO wells (public reproduction).

Run: python reproduce_jump.py [--n-dmso 12000] [--per-class 3000] [--only featselect|harmony]
Input: $AI4S_DATA/jump/{profiles_var_mad_int_featselect,profiles_var_mad_int_featselect_harmony}.parquet + meta/
       (see paths.py and README for download instructions)
Output: out/jump_probes_<variant>.json

Sample: control compounds only (8 positive controls + DMSO). DMSO is abundant, so it is randomly subsampled;
positive controls are capped per class. The same seed picks the SAME wells for both variants (before correction
and after Harmony), because variants are only comparable on identical samples.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shortcut_audit as sa  # noqa: E402
from paths import JUMP_DIR as DATA, OUT_DIR as OUT  # noqa: E402

DMSO = "JCP2022_033924"
# Measured with jump_controls.py: 8 compounds repeated on 92-98% of the 1,876 COMPOUND plates
POS = ["JCP2022_050797", "JCP2022_025848", "JCP2022_012818", "JCP2022_046054",
       "JCP2022_037716", "JCP2022_064022", "JCP2022_085227", "JCP2022_035095"]
KEYS = ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]


def pick_wells(n_dmso, per_class, seed):
    meta = os.path.join(DATA, "meta")
    plate = pd.read_csv(os.path.join(meta, "plate.csv.gz"))
    well = pd.read_csv(os.path.join(meta, "well.csv.gz"))
    w = well.merge(plate[plate.Metadata_PlateType == "COMPOUND"], on=["Metadata_Source", "Metadata_Plate"])
    w = w[w.Metadata_JCP2022.isin(POS + [DMSO])]
    rng = np.random.default_rng(seed)
    parts = []
    for jcp, g in w.groupby("Metadata_JCP2022"):
        n = n_dmso if jcp == DMSO else per_class
        parts.append(g.iloc[rng.permutation(len(g))[:n]])
    return pd.concat(parts, ignore_index=True)


def load_profiles(path, wells):
    pf = pq.ParquetFile(path)
    want = set(map(tuple, wells[KEYS].values))
    feats = [c for c in pf.schema_arrow.names if not c.startswith("Metadata_")]
    keep = []
    for b in pf.iter_batches(batch_size=200_000, columns=KEYS + feats):
        df = b.to_pandas()
        m = [t in want for t in zip(df.Metadata_Source, df.Metadata_Plate, df.Metadata_Well)]
        if any(m):
            keep.append(df[m])
    df = pd.concat(keep, ignore_index=True).merge(wells, on=KEYS)
    X = df[feats].to_numpy(np.float32)
    ok = np.isfinite(X).all(1)
    return df[ok].reset_index(drop=True), X[ok], feats


def run(tag, path, wells, seed):
    df, X, feats = load_profiles(path, wells)
    print(f"[{tag}] wells {len(df)} · features {len(feats)}", flush=True)
    res = {"tag": tag, "n_wells": int(len(df)), "n_features": len(feats)}
    d = (df.Metadata_JCP2022 == DMSO).to_numpy()
    res["nuisance_source_from_dmso"] = sa.nuisance_probe(X[d], df.Metadata_Source[d], groups=df.Metadata_Plate[d])
    src = df.Metadata_Source[d].to_numpy()
    res["nuisance_source_from_dmso"]["null"] = sa.permutation_null(
        X[d], src, sa.group_splits(df.Metadata_Plate[d].to_numpy()), n_perm=5, seed=seed)
    # how far apart the per-lab DMSO means are (mean over features) -- compare before/after correction
    mu = np.vstack([X[d][src == s].mean(0) for s in np.unique(src)])
    res["dmso_between_source_sd"] = float(mu.std(0).mean())
    res["dmso_within_source_sd"] = float(np.mean([X[d][src == s].std(0).mean() for s in np.unique(src)]))
    print(" nuisance (DMSO -> lab)", res["nuisance_source_from_dmso"], flush=True)
    p = ~d
    y = df.Metadata_JCP2022[p].to_numpy()
    res["label_from_nuisance_source"] = sa.label_from_nuisance(y, df.loc[p, ["Metadata_Source"]])
    sc = sa.split_compare(X[p], y, {"plate": df.Metadata_Plate[p].to_numpy(),
                                    "batch": df.Metadata_Batch[p].to_numpy(),
                                    "source": df.Metadata_Source[p].to_numpy()})
    groups = {"plate": df.Metadata_Plate[p].to_numpy(), "batch": df.Metadata_Batch[p].to_numpy(),
              "source": df.Metadata_Source[p].to_numpy()}
    sc["perm_null"] = [sa.permutation_null(X[p], y, sa.random_splits(y), n_perm=3, seed=seed)["null_mean"]] + [
        sa.permutation_null(X[p], y, sa.group_splits(g), n_perm=3, seed=seed)["null_mean"] for g in groups.values()]
    res["split_compare_pos8"] = sc.to_dict(orient="records")
    print(sc.to_string(), flush=True)
    res["confound_injection_batch"] = sa.confound_injection(X[p], y, df.Metadata_Batch[p].to_numpy(), seed=seed)
    print(" confound injection", {k: v for k, v in res["confound_injection_batch"].items() if k != "test_levels"},
          flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dmso", type=int, default=12000)
    ap.add_argument("--per-class", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["featselect", "harmony"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    wells = pick_wells(a.n_dmso, a.per_class, a.seed)
    print("sampled wells", wells.Metadata_JCP2022.value_counts().to_dict(), flush=True)
    files = {"featselect": "profiles_var_mad_int_featselect.parquet",
             "harmony": "profiles_var_mad_int_featselect_harmony.parquet"}
    for tag, f in files.items():
        if a.only and tag != a.only:
            continue
        res = run(tag, os.path.join(DATA, f), wells, a.seed)
        with open(os.path.join(OUT, f"jump_probes_{tag}.json"), "w") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
