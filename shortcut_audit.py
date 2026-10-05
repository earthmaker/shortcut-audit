"""Shortcut audit: probes that measure whether a classifier reads nuisance variables
(lab, batch, plate, session, experimenter) instead of the label.

All probes use the same classifier (standardize + logistic regression) and the same metric (balanced accuracy).
Changing the classifier would break comparisons between probes, so it is fixed. Chance (1 / number of classes)
is reported alongside.

Probes
  nuisance_probe       can a nuisance variable be predicted from samples that share one label (e.g. DMSO)?
                       -> shortcut capacity
  label_from_nuisance  how well is the label predicted from one-hot nuisance variables alone, without features?
                       -> confounding baseline
  split_compare        the same task scored with a random split vs. a held-out-group split
                       -> the share of the score that leans on the shortcut
  confound_injection   what happens when label and nuisance are deliberately entangled in the training set:
                       inflation and collapse
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))


def _cv_score(X, y, splits) -> float:
    pred = np.empty(len(y), dtype=object)
    seen = np.zeros(len(y), dtype=bool)
    for tr, te in splits:
        # A class present in the test fold but absent from the training fold cannot be predicted; it counts as wrong.
        m = _clf().fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        seen[te] = True
    return balanced_accuracy_score(y[seen], pred[seen].astype(y.dtype))


def random_splits(y, k=5, seed=0):
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def group_splits(groups, k=5):
    g = np.asarray(groups)
    k = min(k, len(np.unique(g)))
    return list(GroupKFold(k).split(np.zeros(len(g)), groups=g))


def nuisance_probe(X, nuisance, groups=None, k=5) -> dict:
    """Pass only samples that share one label. With `groups`, folds are split by that unit
    (e.g. by plate when predicting the lab)."""
    y = np.asarray(nuisance)
    splits = group_splits(groups, k) if groups is not None else random_splits(y, k)
    return {"bal_acc": _cv_score(X, y, splits), "chance": 1 / len(np.unique(y)), "n": len(y),
            "classes": int(len(np.unique(y)))}


def label_from_nuisance(y, nuisance_df: pd.DataFrame, groups=None, k=5) -> dict:
    """Features are not used. Predict the label from one-hot nuisance columns only."""
    Z = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(nuisance_df.astype(str))
    y = np.asarray(y)
    splits = group_splits(groups, k) if groups is not None else random_splits(y, k)
    return {"bal_acc": _cv_score(Z, y, splits), "chance": 1 / len(np.unique(y)), "n": len(y)}


def split_compare(X, y, group_cols: dict, k=5) -> pd.DataFrame:
    """group_cols = {"plate": plate_ids, "source": source_ids, ...}. The random split is reported as the first row."""
    y = np.asarray(y)
    rows = [{"split": "random", "bal_acc": _cv_score(X, y, random_splits(y, k))}]
    for name, g in group_cols.items():
        rows.append({"split": f"held-out {name}", "bal_acc": _cv_score(X, y, group_splits(g, k))})
    df = pd.DataFrame(rows)
    df["gap_vs_random"] = df.bal_acc - df.bal_acc.iloc[0]
    df["chance"] = 1 / len(np.unique(y))
    return df


def confound_injection(X, y, nuisance, seed=0, test_frac_groups=0.3) -> dict:
    """Build a training set in which each label k is confined to nuisance levels of its own, then score on a
    balanced test set made of held-out nuisance levels.

    This mimics organ-on-chip data where each condition has its own chip lot or run. Returns: cross-validation
    score inside the confounded training set (inflated), the score of a model trained on it and tested on the
    balanced held-out set (real), and, for comparison, the test score of a same-sized unconfounded training set.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    nu = np.asarray(nuisance)
    levels = np.unique(nu)
    labels = np.unique(y)
    rng.shuffle(levels)
    n_test = max(1, int(round(len(levels) * test_frac_groups)))
    test_lv, train_lv = levels[:n_test], levels[n_test:]
    if len(train_lv) < len(labels):
        raise ValueError("fewer nuisance levels than labels: cannot give each label its own levels")
    # Split the training nuisance levels into as many clusters as there are labels and keep, at each level, only
    # the samples of that cluster's label (level -> one label). Inside the training set the level then
    # predicts the label perfectly.
    lv_label = {lv: labels[j % len(labels)] for j, lv in enumerate(train_lv)}
    te = np.isin(nu, test_lv)
    own = np.array([lv_label.get(v) for v in nu], dtype=object)
    conf_idx = np.where(~te & (own == y.astype(object)))[0]
    # Unconfounded control: draw the same number per label at random from all training levels.
    per_lab = pd.Series(y[conf_idx]).value_counts()
    bal_idx = np.concatenate([
        rng.choice(np.where(~te & (y == lab))[0], size=min(int(per_lab.get(lab, 0)), int(((~te) & (y == lab)).sum())),
                   replace=False)
        for lab in labels])
    Xc, yc = X[conf_idx], y[conf_idx]
    inflated = _cv_score(Xc, yc, random_splits(yc))
    m = _clf().fit(Xc, yc)
    real = balanced_accuracy_score(y[te], m.predict(X[te]))
    mb = _clf().fit(X[bal_idx], y[bal_idx])
    control = balanced_accuracy_score(y[te], mb.predict(X[te]))
    return {"inflated_cv": inflated, "confounded_on_heldout": real, "balanced_on_heldout": control,
            "chance": 1 / len(labels), "n_train": int(len(conf_idx)), "n_test": int(te.sum()),
            "test_levels": [str(v) for v in test_lv]}


def permutation_null(X, y, splits, n_perm=5, seed=0) -> dict:
    """Score with labels shuffled on the same splits: the "no signal" baseline for this split.

    1/n_classes is chance only for random splits. With grouped splits the class mix of training and test folds
    diverges, so a score can land below chance even with no signal (synthetic measurement: 0.07 vs 0.167). Read
    probe scores next to this value.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    vals = [_cv_score(X, rng.permutation(y), splits) for _ in range(n_perm)]
    return {"null_mean": float(np.mean(vals)), "null_max": float(np.max(vals)), "n_perm": n_perm}
