"""Measure which compounds repeat on almost every JUMP cpg0016 compound plate and print the control candidates.

Positive-control IDs are counted directly from well.csv instead of being copied from memory or documents: they
are the reference for the probe design, and an error would tilt every later verdict.

Run: python jump_controls.py
Input: $AI4S_DATA/jump/meta/{plate,well,compound}.csv.gz  (see paths.py)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import JUMP_META as META  # noqa: E402


def load():
    plate = pd.read_csv(os.path.join(META, "plate.csv.gz"))
    well = pd.read_csv(os.path.join(META, "well.csv.gz"))
    comp = pd.read_csv(os.path.join(META, "compound.csv.gz"))
    return plate, well, comp


def main():
    plate, well, comp = load()
    cp = plate[plate.Metadata_PlateType == "COMPOUND"]
    w = well.merge(cp, on=["Metadata_Source", "Metadata_Plate"])
    n_plates = w.groupby(["Metadata_Source", "Metadata_Plate"]).ngroups
    print(f"COMPOUND plates {n_plates} · sources {w.Metadata_Source.nunique()} · batches {w.Metadata_Batch.nunique()}")
    per = (w.groupby("Metadata_JCP2022")
             .agg(plates=("Metadata_Plate", lambda s: s.index.size),
                  n_plate=("Metadata_Plate", "nunique"),
                  n_source=("Metadata_Source", "nunique"))
             .sort_values("n_plate", ascending=False))
    per["frac_plate"] = per.n_plate / n_plates
    per = per.join(comp.set_index("Metadata_JCP2022")[["Metadata_InChIKey", "Metadata_SMILES"]])
    print(per.head(15).to_string())


if __name__ == "__main__":
    main()
