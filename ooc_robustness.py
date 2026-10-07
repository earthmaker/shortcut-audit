"""Robustness checks for the OoC case study (OOC Image Dataset, Movčana et al. 2024, Zenodo 10203721).

These checks add new numbers next to the headline OoC results; they do not recompute or replace them
(ooc_probe.py and repeat_ooc.py remain the source of the headline values). Classifier, metric, splits and seeds
are those of shortcut_audit.py: standardise + logistic regression (C=0.1), balanced accuracy, StratifiedKFold(5,
seed 0) for random splits, GroupKFold(5) for held-out-date splits.

  1. forward in time   cut the imaging dates in calendar order at 60/70/80 %. Inside the earlier period only,
                       estimate the score with (a) held-out-date CV and (b) random CV; then train on the whole
                       earlier period and score the later period. All three cuts are reported. The later period is
                       scored on the cell types that occur in the earlier period (and, for reference, on all of it).
                       The same is repeated on HPMEC alone, cutting on HPMEC's own dates.
  2. robustness        (a) null that keeps every label-date pair and permutes feature rows (held-out-date and
                       random split), (b) class-balanced logistic regression, (c) date probe inside one quality
                       label, (d) date probe inside cell type x day-after-seeding strata, with a same-split null,
                       (e) date-disjoint splits with the test size of the distributed split, 20 draws.
  3. permutations      200 label permutations for the headline probes (date from image, quality random split,
                       quality held-out date); empirical p = (k + 1) / (n + 1), k = permuted scores >= observed.
  4. classic features  without image height and width.

Run: python ooc_robustness.py [--feats ooc_feats_dinov2-small.npz] [--n-perm 200]
Input: the released feature file and $AI4S_DATA/ooc/OOC_datasheet.xlsx (see paths.py)
Output: out/ooc_robustness.json + tables on screen
"""
import argparse
import contextlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shortcut_audit as sa  # noqa: E402
from ooc_probe import load  # noqa: E402
from paths import OUT_DIR  # noqa: E402

CUTS = (0.6, 0.7, 0.8)
# Column order of the classic vector, from ooc_extract.classic(): mean, sd, 5 percentiles, left-right and
# top-bottom gradients, image height, image width, 16-bin histogram (27 columns).
CLASSIC_NAMES = (["mean", "sd", "p1", "p5", "p50", "p95", "p99", "grad_lr", "grad_tb", "height", "width"]
                 + [f"hist_{i:02d}" for i in range(16)])
SIZE_COLS = [CLASSIC_NAMES.index("height"), CLASSIC_NAMES.index("width")]


def fit_score(X, y, tr, te):
    m = sa._clf().fit(X[tr], y[tr])
    return float(balanced_accuracy_score(y[te], m.predict(X[te])))


@contextlib.contextmanager
def balanced_classifier():
    """Swap the library classifier for the same pipeline with class_weight='balanced'."""
    orig = sa._clf
    sa._clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1,
                                                                         class_weight="balanced"))
    try:
        yield
    finally:
        sa._clf = orig


def dates_with_min(s, mask, n_min=10):
    vc = pd.Series(s[mask]).value_counts()
    return mask & np.isin(s, vc[vc >= n_min].index)


def forward(X, y, s, ct, scope):
    """scope: boolean mask of the images that take part (all images, or one cell type)."""
    dates = np.array(sorted(set(s[scope])))
    rows = []
    for q in CUTS:
        cut = dates[int(len(dates) * q)]
        tr = scope & (s < cut)
        late = scope & (s >= cut)
        te = late & np.isin(ct, np.unique(ct[tr]))
        rows.append({"cut_fraction": q, "cut_date": str(cut),
                     "train_dates": int(len(set(s[tr]))), "test_dates": int(len(set(s[te]))),
                     "n_train": int(tr.sum()), "n_test": int(te.sum()), "n_test_all_celltypes": int(late.sum()),
                     "test_good_fraction": float((y[te] == "good").mean()),
                     "train_heldout_date_cv": float(sa._cv_score(X[tr], y[tr], sa.group_splits(s[tr]))),
                     "train_random_cv": float(sa._cv_score(X[tr], y[tr], sa.random_splits(y[tr]))),
                     "later_period": fit_score(X, y, tr, te),
                     "later_period_all_celltypes": fit_score(X, y, tr, late),
                     "test_celltypes_dropped": sorted(set(ct[late]) - set(ct[tr]))})
    return rows


def perm_scores(X, y, splits, n_perm, seed=0):
    """Sample-level label permutations on fixed splits; the same draws as sa.permutation_null(seed=0)."""
    rng = np.random.default_rng(seed)
    return np.array([sa._cv_score(X, rng.permutation(y), splits) for _ in range(n_perm)])


def summ(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=os.path.join(OUT_DIR, "ooc_feats_dinov2-small.npz"))
    ap.add_argument("--n-perm", type=int, default=200, help="permutations for the headline p-values (part 3)")
    ap.add_argument("--n-xperm", type=int, default=10, help="feature-row permutations (part 2a)")
    ap.add_argument("--strata-perm", type=int, default=5, help="null permutations per stratum (part 2d)")
    ap.add_argument("--draws", type=int, default=20, help="date-disjoint draws (part 2e)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "ooc_robustness.json"))
    a = ap.parse_args()

    t_all = time.time()
    timing = {}
    df, D, C, n_img = load(a.feats)
    y = df.label.to_numpy()
    s = df.session.to_numpy()
    ct = df["cell type"].astype(str).to_numpy()
    day = df["day"].to_numpy()
    sp = df.split.str.lower().to_numpy()
    allm = np.ones(len(y), dtype=bool)
    res = {"versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit-learn": sklearn.__version__, "machine": platform.machine(),
                        "platform": platform.platform()},
           "args": vars(a), "n_images": int(n_img), "n_joined": int(len(df)), "dates": int(len(set(s)))}
    print(f"{len(df)} images · {len(set(s))} imaging dates · scikit-learn {sklearn.__version__} · "
          f"Python {platform.python_version()}")

    # 1. forward in time
    t = time.time()
    res["forward_all"] = forward(D, y, s, ct, allm)
    res["forward_hpmec"] = forward(D, y, s, ct, ct == "HPMEC")
    timing["1_forward"] = time.time() - t
    cols = ["cut_fraction", "cut_date", "n_train", "n_test", "train_heldout_date_cv", "later_period",
            "train_random_cv", "later_period_all_celltypes"]
    for k in ("forward_all", "forward_hpmec"):
        print(f"\n[1 {k}] earlier-period estimates vs later-period score (DINOv2)")
        print(pd.DataFrame(res[k])[cols].round(3).to_string(index=False))

    # 2a. null keeping label-date pairs: permute feature rows
    t = time.time()
    gs, rs = sa.group_splits(s), sa.random_splits(y)
    rng = np.random.default_rng(0)
    ng, nr = [], []
    for _ in range(a.n_xperm):
        p = rng.permutation(len(y))
        ng.append(sa._cv_score(D[p], y, gs))
        nr.append(sa._cv_score(D[p], y, rs))
    res["xperm_null"] = {"n_perm": a.n_xperm, "heldout_date": summ(ng), "random": summ(nr)}
    timing["2a_xperm_null"] = time.time() - t

    # 2b. class-balanced logistic regression (removes the training-prevalence intercept shift)
    t = time.time()
    with balanced_classifier():
        br, bg = float(sa._cv_score(D, y, rs)), float(sa._cv_score(D, y, gs))
    res["balanced_lr"] = {"random": br, "heldout_date": bg, "gap": bg - br}
    # dates on which every image is good, or every image is bad, removed
    frac = pd.Series(y == "good").groupby(s).mean()
    pure = frac[(frac == 0) | (frac == 1)].index
    keep = ~np.isin(s, pure)
    nr_, ng_ = (float(sa._cv_score(D[keep], y[keep], sa.random_splits(y[keep]))),
                float(sa._cv_score(D[keep], y[keep], sa.group_splits(s[keep]))))
    res["no_pure_dates"] = {"pure_dates_removed": int(len(pure)), "n": int(keep.sum()),
                            "dates": int(len(set(s[keep]))), "random": nr_, "heldout_date": ng_, "gap": ng_ - nr_}
    timing["2b_balanced_lr_and_pure"] = time.time() - t
    print("\n[2a-b] held-out-date bias checks (DINOv2)")
    print(pd.DataFrame([
        {"check": f"feature-row permutation null ({a.n_xperm}x), held-out date", "random": res["xperm_null"]["random"]["mean"],
         "heldout_date": res["xperm_null"]["heldout_date"]["mean"], "gap": np.nan},
        {"check": "class_weight='balanced'", "random": br, "heldout_date": bg, "gap": bg - br},
        {"check": f"without the {len(pure)} single-label dates", "random": nr_, "heldout_date": ng_, "gap": ng_ - nr_},
    ]).round(3).to_string(index=False))
    print(f"   feature-row null held-out date range {res['xperm_null']['heldout_date']['min']:.3f}"
          f"-{res['xperm_null']['heldout_date']['max']:.3f}")

    # 2c. date probe inside one quality label
    t = time.time()
    rows = []
    for lab in ("good", "bad"):
        ok = dates_with_min(s, y == lab)
        for name, X in (("dino", D), ("classic", C)):
            r = sa.nuisance_probe(X[ok], s[ok])
            rows.append({"label": lab, "features": name, **r})
    res["date_within_label"] = rows
    timing["2c_date_within_label"] = time.time() - t
    print("\n[2c] imaging date from image, one quality label only (dates with >= 10 images of that label)")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # 2d. date probe inside cell type x day-after-seeding strata (>= 3 dates with >= 10 images each)
    t = time.time()
    rows = []
    for (c, d), g in pd.DataFrame({"ct": ct, "day": day, "s": s}).groupby(["ct", "day"]):
        vc = g.s.value_counts()
        keepd = vc[vc >= 10].index
        if len(keepd) < 3:
            continue
        idx = g.index[g.s.isin(keepd)].to_numpy()
        for name, X in (("dino", D), ("classic", C)):
            r = sa.nuisance_probe(X[idx], s[idx])
            nl = sa.permutation_null(X[idx], s[idx], sa.random_splits(s[idx]), n_perm=a.strata_perm)
            rows.append({"cell_type": c, "day": int(d), "features": name, "bal_acc": r["bal_acc"],
                         "chance": r["chance"], "null_mean": nl["null_mean"], "null_max": nl["null_max"],
                         "n": r["n"], "dates": r["classes"]})
    res["date_within_celltype_day"] = rows
    timing["2d_strata"] = time.time() - t
    print(f"\n[2d] imaging date within cell type x day after seeding ({len(rows) // 2} strata, "
          f"null {a.strata_perm} permutations)")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # 2e. date-disjoint splits with the test size of the distributed split
    t = time.time()
    n_test, n_val = int((sp == "test").sum()), int((sp == "val").sum())
    trn, tst = sp == "train", sp == "test"
    distributed = fit_score(D, y, trn, tst)
    dates = np.array(sorted(set(s)))
    rng = np.random.default_rng(0)
    v_all, v_size, sizes = [], [], []
    for _ in range(a.draws):
        perm = rng.permutation(dates)
        te_d, cnt = [], 0
        for dd in perm:
            if cnt >= n_test:
                break
            te_d.append(dd)
            cnt += int((s == dd).sum())
        te = np.isin(s, te_d)
        v_all.append(fit_score(D, y, ~te, te))   # train on every other date (as in the reviewer's check)
        # training size matched to the distributed train folder: also hold out val-sized whole dates
        rest = [dd for dd in perm if dd not in set(te_d)]
        va_d, cnt = [], 0
        for dd in rest:
            if cnt >= n_val:
                break
            va_d.append(dd)
            cnt += int((s == dd).sum())
        tr = ~te & ~np.isin(s, va_d)
        v_size.append(fit_score(D, y, tr, te))
        sizes.append((int(tr.sum()), int(te.sum())))
    res["date_disjoint_distributed_size"] = {
        "draws": a.draws, "distributed_split": distributed, "n_test_distributed": n_test,
        "n_train_distributed": int(trn.sum()),
        "train_all_other_dates": summ(v_all), "train_size_matched": summ(v_size),
        "n_train_range_size_matched": [min(x[0] for x in sizes), max(x[0] for x in sizes)],
        "n_test_range": [min(x[1] for x in sizes), max(x[1] for x in sizes)],
        "values_all_other_dates": v_all, "values_size_matched": v_size}
    timing["2e_date_disjoint"] = time.time() - t
    print(f"\n[2e] date-disjoint splits with the distributed test size ({a.draws} draws, DINOv2)")
    print(pd.DataFrame([
        {"split": "distributed train -> test", "bal_acc_mean": distributed, "sd": np.nan, "min": np.nan, "max": np.nan},
        {"split": "whole dates out, train on all other dates", **{k if k != "mean" else "bal_acc_mean": v
                                                                 for k, v in summ(v_all).items()}},
        {"split": "whole dates out, train size matched", **{k if k != "mean" else "bal_acc_mean": v
                                                           for k, v in summ(v_size).items()}},
    ]).round(3).to_string(index=False))

    # 3. headline permutations
    t = time.time()
    ok = dates_with_min(s, allm)
    heads = [("date_from_image_dino", D[ok], s[ok], sa.random_splits(s[ok])),
             ("quality_random_dino", D, y, rs),
             ("quality_heldout_date_dino", D, y, gs)]
    rows = []
    res["headline_permutations"] = {}
    for name, X, lab, splits in heads:
        t1 = time.time()
        obs = float(sa._cv_score(X, lab, splits))
        null = perm_scores(X, lab, splits, a.n_perm)
        k = int((null >= obs).sum())
        r = {"observed": obs, "n_perm": a.n_perm, "null": summ(null), "null_mean_first5": float(null[:5].mean()),
             "k_ge_observed": k, "p_empirical": (k + 1) / (a.n_perm + 1), "seconds": time.time() - t1,
             "chance": 1 / len(np.unique(lab)), "n": int(len(lab))}
        res["headline_permutations"][name] = r
        rows.append({"probe": name, "observed": obs, "null_mean": r["null"]["mean"], "null_max": r["null"]["max"],
                     "first5_mean": r["null_mean_first5"], "k": k, "p": r["p_empirical"]})
        print(f"   {name} done in {r['seconds']:.0f} s", flush=True)
    timing["3_permutations"] = time.time() - t
    print(f"\n[3] headline probes with {a.n_perm} sample-level label permutations, p = (k+1)/(n+1)")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # 4. classic features without image height and width
    t = time.time()
    keepc = [i for i in range(C.shape[1]) if i not in SIZE_COLS]
    rows = []
    for name, X in (("classic_all_27", C), ("classic_no_height_width_25", C[:, keepc])):
        rows.append({"features": name, "quality_random": float(sa._cv_score(X, y, rs)),
                     "quality_heldout_date": float(sa._cv_score(X, y, gs)),
                     "date_from_image": float(sa._cv_score(X[ok], s[ok], sa.random_splits(s[ok])))})
    size_levels = pd.Series([f"{int(h)}x{int(w)}" for h, w in C[:, SIZE_COLS]]).value_counts().to_dict()
    res["classic_without_size"] = {"dropped_columns": [CLASSIC_NAMES[i] for i in SIZE_COLS],
                                   "dropped_indices": SIZE_COLS, "image_sizes_hxw": size_levels, "rows": rows}
    timing["4_classic_no_size"] = time.time() - t
    print(f"\n[4] classic features without image height/width (dropped columns {SIZE_COLS} = "
          f"{[CLASSIC_NAMES[i] for i in SIZE_COLS]}; image sizes {size_levels})")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    timing["total"] = time.time() - t_all
    res["seconds"] = timing
    print("\nseconds", {k: round(v, 1) for k, v in timing.items()})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
