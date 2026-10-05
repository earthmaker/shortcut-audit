"""Shortcut audit of the OOC Image Dataset: predict quality (good/bad) and imaging date from image features.

Run: python ooc_probe.py [--feats out/ooc_feats_dinov2-small.npz]
Output: out/ooc_probes.json + tables on screen

Probes
  (a) can imaging date be predicted from the image alone, per cell type (so cell types do not mix) and overall
  (b) quality classification: random split vs. held-out-imaging-date split (DINOv2, classical features, metadata only)
  (c) do the same imaging dates appear on both sides of the distributed split (train/val/test folders), plus the
      score on the distributed split as given
  (d) confound injection: a training set where quality and imaging date are deliberately entangled
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shortcut_audit as sa  # noqa: E402
from paths import OOC_XLSX as XLSX, OUT_DIR  # noqa: E402


def load(feats_path):
    z = np.load(feats_path, allow_pickle=False)
    f = pd.DataFrame({"imageID": z["ids"], "split": z["split"]})
    meta = pd.read_excel(XLSX, sheet_name="Main")
    meta["imageID"] = meta.imageID.astype(str)
    df = f.reset_index().merge(meta, on="imageID", how="inner")
    idx = df["index"].to_numpy()
    df["session"] = df.imageID.str.split("_").str[0]
    df["label"] = df["Decision 1/2 (good/bad)"].map({1: "good", 2: "bad"})
    keep = df.label.notna().to_numpy()
    return df[keep].reset_index(drop=True), z["dino"][idx][keep], z["classic"][idx][keep], len(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=os.path.join(OUT_DIR, "ooc_feats_dinov2-small.npz"))
    a = ap.parse_args()
    df, D, C, n_img = load(a.feats)
    print(f"features for {n_img} images, {len(df)} joined to metadata · imaging dates {df.session.nunique()}")
    res = {"n_images": int(n_img), "n_joined": int(len(df))}

    # (a) imaging-date discrimination: only dates with >= 10 images, per cell type
    rows = []
    for ct, g in [("ALL", df)] + list(df.groupby("cell type")):
        vc = g.session.value_counts()
        ok = g.session.isin(vc[vc >= 10].index).to_numpy()
        if ok.sum() < 50 or g.session[ok].nunique() < 2:
            continue
        ii = g.index[ok]
        for name, X in (("dino", D), ("classic", C)):
            r = sa.nuisance_probe(X[ii], df.session[ii])
            rows.append({"cell_type": ct, "features": name, **r})
    res["session_from_image"] = rows
    print(pd.DataFrame(rows).to_string())

    # (b) quality: random split vs. held-out imaging date
    y = df.label.to_numpy()
    out = {}
    for name, X in (("dino", D), ("classic", C)):
        sc = sa.split_compare(X, y, {"session": df.session.to_numpy()})
        # Permutation baseline on the same splits: the held-out-date split can fall below 0.5 with no signal
        sc["perm_null"] = [sa.permutation_null(X, y, sa.random_splits(y))["null_mean"],
                           sa.permutation_null(X, y, sa.group_splits(df.session.to_numpy()))["null_mean"]]
        out[name] = sc.to_dict(orient="records")
        print(f"[quality · {name}]\n{sc.to_string()}")
    out["meta_session_only_random"] = sa.label_from_nuisance(y, df[["session"]])
    out["meta_session_only_heldout"] = sa.label_from_nuisance(y, df[["session"]], groups=df.session)
    res["quality_split_compare"] = out

    # (c) distributed split
    sp = df.split.str.lower()
    if sp.isin(["train", "test"]).any():
        tr_s = set(df.session[sp == "train"])
        te = (sp == "test").to_numpy()
        shared = df.session[te].isin(tr_s)
        res["official_split"] = {"test_images": int(te.sum()),
                                 "test_images_with_session_in_train": int(shared.sum()),
                                 "test_sessions": int(df.session[te].nunique()),
                                 "test_sessions_seen_in_train": int(df.session[te][shared].nunique())}
        trn = (sp == "train").to_numpy()
        m = sa._clf().fit(D[trn], y[trn])
        p = m.predict_proba(D[te])[:, list(m.classes_).index("good")]
        res["official_split"]["dino_bal_acc"] = float(balanced_accuracy_score(y[te], m.predict(D[te])))
        res["official_split"]["dino_auc"] = float(roc_auc_score(y[te] == "good", p))
        unseen = ~shared.to_numpy()
        if unseen.sum() > 20 and len(set(y[te][unseen])) == 2:
            res["official_split"]["dino_bal_acc_unseen_sessions"] = float(
                balanced_accuracy_score(y[te][unseen], m.predict(D[te][unseen])))
        print("distributed split", res["official_split"])

    # (d) confound injection
    res["confound_injection_session"] = sa.confound_injection(D, y, df.session.to_numpy())
    print("confound injection", {k: v for k, v in res["confound_injection_session"].items() if k != "test_levels"})

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "ooc_probes.json"), "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
