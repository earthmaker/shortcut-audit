# Shortcut audit for imaging classifiers

Microscopy and high-content imaging classifiers often score well because they read **imaging session, lab, batch or
plate** rather than biology, and common train/test splits do not separate those variables from the label. This tool
measures that. Give it one feature table (one row per image or well) plus the label column and the nuisance columns,
and it produces a one-page HTML report with: how well each nuisance variable can be predicted from the features
(shortcut capacity), how well the label can be predicted from the nuisance variables alone (confounding baseline),
and the score gap between a random split and a held-out-group split. Every probe uses the same classifier
(standardize + logistic regression, C=0.1) and the same metric (balanced accuracy). The nuisance probe, the
label-from-nuisance baseline and every split are printed next to a permutation null computed on the same split (whole
groups are permuted when the nuisance probe's nuisance is nested in `--group`). Each report also runs a
confound-injection stress test per nuisance column, whenever there are at least as many nuisance levels as labels: it
ties the label to that nuisance in a training set on purpose and shows how far the held-out score falls, next to an
unconfounded control of the same size. The tool re-fits one fixed linear probe on the features you supply; it does not
open a trained model. To audit a specific network, export its penultimate-layer features and audit those. It runs on a
laptop CPU, with no paid services and no private data. The two public datasets used for reproduction are JUMP Cell
Painting (CC0) and the Organ-on-a-Chip Image Dataset (see "Data licenses").

## Layout

| File | Purpose |
|---|---|
| `audit.py` | **Entry script.** Feature table in, HTML report out |
| `shortcut_audit.py` | The probes (importable library) |
| `test_shortcut_audit.py` | Synthetic positive/negative-control tests, including the reading rules |
| `check_verdicts.py` | False-alarm and two-level checks of the report's reading rules (20 seeds) |
| `examples/make_synthetic.py` | Writes a small synthetic table for a no-download quickstart |
| `paths.py` | Data root (`AI4S_DATA`, default `./data`) and output dir (`./out`) |
| `reproduce_jump.py`, `jump_controls.py` | JUMP reproduction |
| `ooc_meta_probe.py`, `ooc_extract.py`, `ooc_probe.py`, `ooc_table.py` | Organ-on-a-Chip (OOC) reproduction |
| `ooc_robustness.py` | OOC robustness checks (forward-in-time check, held-out-date bias checks, date probe within one label and within cell type x day, date-disjoint splits at the distributed test size, 200-permutation nulls, classic features without image size) -> `out/ooc_robustness.json` |

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # the audit, tests and all OoC numbers from the released features (CPU)
pip install -r requirements-extract.txt    # only to extract DINOv2 features from images yourself (torch, transformers<5)
# or: conda env create -f environment.yml && conda activate shortcut-audit   (includes the extraction packages)
```

See "Library versions" below for why scikit-learn is pinned.

## Quickstart on synthetic data (no downloads)

```bash
python test_shortcut_audit.py          # expected last line: PASS
python examples/make_synthetic.py      # writes out/synthetic_audit.csv
python audit.py out/synthetic_audit.csv --label quality --nuisance session cell_type \
    --group session --out out/synthetic_report.html
```

The synthetic table has a weak label signal, a strong session signal and sessions partly confounded with the label.
With scikit-learn 1.7.2 the report shows: nuisance probe session 0.955 ("shortcut capacity present"), label from
session alone 0.798 ("confounded"), random split 0.631 (null 0.494), held-out session 0.447 (-0.183, "part of the score
does not transfer to unseen groups"). The held-out score lies below chance (0.5): when label proportions differ
between sessions, pooled out-of-fold predictions on held-out sessions are biased downwards even without a shortcut.
That is why the split-gap reading is shown only when the random split is at least 0.05 above its own permutation null;
`check_verdicts.py` measures this (pure-noise features, label tied to session, 20 seeds: the gap rule alone fires in
10 of 20 runs, the guarded rule in 0).

## `audit.py` usage

```
python audit.py TABLE --label COL --nuisance COL [COL ...]
                [--group COL] [--control-value VALUE] [--features COL ...]
                [--perm N] [--no-inject] [--out report.html] [--title TEXT]
```

- `TABLE`: csv, parquet or npz. One row per sample. In an npz, each 1-D array is one column and a 2-D array (for
  example an embedding matrix `X`) is expanded to columns `X_0 ... X_d`.
- `--label`: the biological label to be classified.
- `--nuisance`: candidate nuisance columns (session, lab, batch, plate, experimenter ...).
- `--group`: grouping unit for the nuisance-probe split (for example plate when predicting the lab).
- `--control-value`: run the nuisance probe only on samples with this label (for example `DMSO`), so biology is held constant.
- `--features`: feature columns; default is every numeric column that is not named in `--label`/`--nuisance`/`--group`
  and does not start with `Metadata_`. **Numeric metadata columns that you do not name (run number, time, well row)
  become features silently**; the log prints the feature count and the first names, so check it or pass `--features`.
- `--perm`: label permutations per probe for the null baseline (default 5, 0 = off).
- `--no-inject`: skip the confound-injection stress test (it needs at least as many nuisance levels as labels).

Display thresholds in the report (aids, not statistical tests; all numbers are shown):
nuisance probe >= chance + min(chance, 0.10) (twice chance for 10 or more levels, chance + 0.10 for fewer) and above
the permutation-null maximum is "shortcut capacity present"; label-from-nuisance >= chance + 0.05 is "confounded"; if
the random-split score is >= 0.05 above its permutation null, a held-out-group score >= 0.05 below the random split
is "part of the score does not transfer to unseen groups" (shortcut reliance is one component, genuine differences
between groups another), otherwise the row says there is no label signal to lose; a confounded-training CV >= 0.10
above its held-out score is "exploitable".
Progress is printed with elapsed time. On the 3,072-image OoC table (768 features) a full run took 74 s on a Linux
workstation CPU (i9-10940X) and 151-163 s on a 6-core desktop CPU (i5-8500). BLAS threads are capped at 8 by default
(`SHORTCUT_AUDIT_THREADS=n` to change): with one thread per core on a 28-thread machine the same run took 1,764 s.

Run times of every step (2026-10-06, fresh Python 3.11 environment, scikit-learn 1.9.1, Linux workstation CPU
i9-10940X): `test_shortcut_audit.py` 2 s, `ooc_meta_probe.py` 2 s, `ooc_probe.py` 18 s, `ooc_table.py` 2 s, `audit.py`
on the OoC table 74 s, `jump_controls.py` 3 s, `reproduce_jump.py` (both versions) 416 s; on a 6-core desktop CPU
(i5-8500) `audit.py` took 163 s. `ooc_extract.py` took 556 s for the 3,072 OoC images on an NVIDIA RTX 3080 and
reproduced the released features bit for bit. On the i5-8500 (2026-10-07, Python 3.10, scikit-learn 1.7.2) the current
`test_shortcut_audit.py`, which also checks the reading rules, takes about 10 s, and `ooc_robustness.py` took 3,295 s,
of which 3,205 s were the 200-permutation nulls.

## Reproducing the JUMP results

Data: JUMP Cell Painting `cpg0016` (CC0).

- Well/plate/compound metadata (`plate.csv.gz`, `well.csv.gz`, `compound.csv.gz`): the `metadata/` folder of
  <https://github.com/jump-cellpainting/datasets>.
- Well-level profiles `profiles_var_mad_int_featselect.parquet` (before batch correction) and
  `profiles_var_mad_int_featselect_harmony.parquet` (after Harmony), about 2.8 GB each: the
  public `cellpainting-gallery` S3 bucket (anonymous HTTPS, no account needed; the exact URLs below were used
  to produce the numbers in this README on 2026-10-05):

  ```bash
  B=https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/source_all/workspace/profiles_assembled/COMPOUND/v1.0
  curl -C - -O $B/profiles_var_mad_int_featselect.parquet
  curl -C - -O $B/profiles_var_mad_int_featselect_harmony.parquet
  ```

Place the files like this (or set `AI4S_DATA` to another root):

```
data/jump/meta/{plate,well,compound}.csv.gz
data/jump/profiles_var_mad_int_featselect.parquet
data/jump/profiles_var_mad_int_featselect_harmony.parquet
```

```bash
python jump_controls.py                       # counts control compounds that repeat across plates
python reproduce_jump.py                      # both variants; --only featselect | harmony for one
```

Outputs: `out/jump_probes_featselect.json`, `out/jump_probes_harmony.json`.

## Reproducing the OOC results

Data: Organ-on-a-Chip Image Dataset (Movčana et al., *Data* 2024), Zenodo record 10203721
(<https://zenodo.org/records/10203721>; licence: see "Data licenses"). Two files: `OOC_datasheet.xlsx` (120 KB, the
metadata) and `OOC_image_dataset.zip` (**6.7 GB**, 6,710,767,405 bytes, the images).

**CPU path, no image download (a few minutes).** The DINOv2 and classical features extracted from all 3,072 images
are released with this repository (`ooc_feats_dinov2-small.npz`, 9 MB, CC BY-SA 4.0, see "Data licenses"). With it and
the datasheet, every OoC number below except the extraction itself is reproduced on a CPU:

```bash
mkdir -p data/ooc out
curl -L -o data/ooc/OOC_datasheet.xlsx "https://zenodo.org/records/10203721/files/OOC_datasheet.xlsx?download=1"
curl -L -o out/ooc_feats_dinov2-small.npz \
    https://github.com/earthmaker/shortcut-audit/releases/download/v1.0/ooc_feats_dinov2-small.npz
md5sum out/ooc_feats_dinov2-small.npz         # 462310d38de0d64776875bec6b0cf246 (macOS: md5)
python ooc_meta_probe.py                      # metadata only, no images (0.719 / 0.500 ...)
python ooc_probe.py                           # all image probes -> out/ooc_probes.json
python ooc_table.py                           # table for audit.py
python audit.py out/ooc_audit_table.parquet --label quality --nuisance session cell_type --group session \
    --title "OOC brightfield quality classifier" --out out/ooc_audit_report.html
python ooc_robustness.py      # out/ooc_feats_dinov2-small.npz -> out/ooc_robustness.json (~55 min on a 6-core i5 CPU, 200 permutations; --n-perm 3 for a ~2.5 min check)
```

**Full path, from the images.** Download the zip as well and extract the features yourself (needs
`requirements-extract.txt`):

```bash
curl -L -C - -o data/ooc/OOC_image_dataset.zip "https://zenodo.org/records/10203721/files/OOC_image_dataset.zip?download=1"
python ooc_extract.py                         # DINOv2-small + classical features -> out/ooc_feats_dinov2-small.npz
python ooc_extract.py --device cpu            # force CPU if the GPU/MPS path fails
```

`ooc_extract.py` downloads `facebook/dinov2-small` from the Hugging Face Hub on first use (free, no token). `--device`
is `auto` (CUDA, then Apple MPS, then CPU), `cpu`, `cuda` or `mps`; operations missing on MPS fall back to the CPU.
On an RTX 3080 the 3,072 images took 556 s and reproduced the released file bit for bit.

## Expected numbers

Balanced accuracy. These are the values measured when the tool was developed; reruns with the same seeds and data
should match closely, small differences across library versions are possible.

**OOC, metadata only (`ooc_meta_probe.py`, no image used)**

| What is used (no image) | Random split | Held-out imaging date | Chance |
|---|---|---|---|
| Imaging date | **0.719** | 0.500 | 0.50 |
| Cell type | 0.586 | 0.586 | 0.50 |
| Cell type + time | 0.649 | 0.607 | 0.50 |

59 imaging dates; 14 of them are all good or all bad.

**OOC, image features (`ooc_probe.py`; DINOv2-small, 448 px, CLS + mean patch token; 3,072 images, all joined to metadata)**

| Probe | DINOv2 | Classical features (brightness statistics) | Chance / permutation null |
|---|---|---|---|
| Imaging date from image alone (50 dates with >= 10 images) | **0.859** | 0.345 | 0.020 |
| . per cell type (A549 21 dates / HPMEC 24 dates) | 0.885 / 0.883 | 0.601 / 0.504 | 0.048 / 0.042 |
| Quality good/bad, random split | 0.818 | 0.636 | 0.50 / 0.497 |
| . held-out imaging date | **0.734 (-0.084)** | 0.533 (-0.103) | 0.50 / 0.494 |
| Confound injection (imaging date): confounded CV / held-out dates / unconfounded control | 0.904 / **0.649** / 0.737 | | 0.50 |

Distributed train/test split: all 656 test images come from imaging dates that also appear in training (57 of 57
test dates). On that split the same DINOv2 + logistic-regression model gets balanced accuracy 0.798 and AUC 0.885,
higher than the held-out-date score of 0.734. The claim is limited to this: the distributed split does not separate
imaging dates, and a model scored on it can profit from recognizing the day a picture was taken. It is not a claim
about any published model's reported accuracy.

**JUMP, before batch correction (`reproduce_jump.py --only featselect`; 31,760 wells, 737 features)**

| Probe | Value | Chance |
|---|---|---|
| Lab (10) from DMSO wells only, plate-grouped CV (permutation null) | 0.307 (0.102) | 0.100 |
| 8 positive controls from lab alone | 0.149 | 0.125 |
| 8 positive controls, random split | 0.946 | 0.125 |
| . held-out plate / held-out batch / **held-out lab** | 0.946 / 0.934 / **0.855** | |
| Confound injection (batch): confounded CV / held-out batches / unconfounded control | 0.962 / 0.854 / 0.914 | |

**JUMP, before vs. after Harmony (same wells, same seed)**

| Probe | Before | After Harmony |
|---|---|---|
| Lab from DMSO wells only (plate-grouped; permutation null) | 0.307 (0.102) | **0.729** (0.102) |
| Std. dev. of feature means between lab DMSO means | 0.038 | 0.135 |
| 8 positive controls: random / plate / batch / lab held out | 0.946 / 0.946 / 0.934 / 0.855 | 0.941 / 0.942 / 0.927 / 0.863 |
| Confound injection: confounded CV / held-out batches / control | 0.962 / 0.854 / 0.914 | 0.956 / 0.856 / 0.898 |

Reading: the release-pinned recipe configuration for these files (jump-profiling-recipe commit a917fa7,
`inputs/compound.json`) keys Harmony on `Metadata_Source`, i.e. it mixes labs. Even so, lab identity became easier to
read from DMSO wells after correction. Harmony aligns each lab's overall distribution, dominated by compound wells, and
the small DMSO subset can move apart in the process; this is not evidence that Harmony "increased the lab effect" in
general, and we did not re-run Harmony. The standard deviation of lab
means is written by `reproduce_jump.py` (`dmso_between_source_sd`). Numbers in this table are from one Linux run
(scikit-learn 1.9.1, Python 3.11; see "Library versions"). Permutation nulls for every split were 0.123-0.127
(1/k = 0.125), so grouped CV was not biased here.

## Repeat measurements

Single runs above; spread over seeds below (mean +/- SD, Linux, CPU).

| Measurement | Runs | Value |
|---|---|---|
| OOC quality, random split (`repeat_ooc.py`, StratifiedKFold reseeded) | 10 | 0.814 +/- 0.004 |
| OOC quality, held-out imaging date (date-to-fold assignment reshuffled) | 10 | 0.730 +/- 0.007 |
| OOC gap, held-out minus random | 10 | -0.084 +/- 0.010 |
| JUMP lab from DMSO wells, before / after Harmony (`summarize_seeds.py`) | 5 | 0.308 +/- 0.004 / 0.733 +/- 0.005 |
| . permutation null, before / after | 5 | 0.100 / 0.101 |
| JUMP positive controls, lab held out, before / after | 5 | 0.856 +/- 0.003 / 0.861 +/- 0.003 |
| JUMP lab gap (lab held out minus random), before / after | 5 | -0.089 +/- 0.004 / -0.080 +/- 0.004 |

```bash
python repeat_ooc.py out/ooc_audit_table.parquet --seeds 10        # -> out/ooc_repeat.json
for s in 1 2 3 4; do AI4S_OUT=out/seed$s python reproduce_jump.py --seed $s; done
python summarize_seeds.py                                          # -> out/jump_seed_summary.json
```

## Library versions

scikit-learn is pinned to **1.7.2** in `requirements.txt`. From 1.8 on, `GroupKFold` assigns groups to folds
differently, so grouped-split values move while the conclusions do not: with 1.9.1 the synthetic lab probe of the
tests is 0.106 instead of 0.071 (both pass) and the OoC held-out-date score 0.735 instead of 0.734. Which version
produced which numbers in this README:

| Numbers | scikit-learn |
|---|---|
| Synthetic tests, quickstart, `check_verdicts.py`, all OoC tables | 1.7.2 (`requirements.txt`) |
| All JUMP tables and the JUMP seed repeats | 1.9.1 (Linux) — `pip install scikit-learn==1.9.1` to reproduce them |

Measured on the JUMP run (2026-10-07, Intel Mac): with 1.9.1 every JUMP value matched the Linux run to within 0.001
(lab from DMSO 0.307 / 0.729); with 1.7.2 lab from DMSO became 0.303 / 0.731 and the other values moved by at most
0.001. The OoC random-split values and the OoC repeat table (`repeat_ooc.py` makes its own fold assignment) were the
same under both versions.
Other tested versions: Python 3.11, numpy 2.2, pandas 2.3, pyarrow 25; for extraction torch 2.2.2 with
transformers 4.57 (Intel Mac, CPU and MPS) and the Linux workstation run above.

## Data licenses

| Dataset | License | Source |
|---|---|---|
| JUMP Cell Painting `cpg0016` (profiles, metadata) | CC0 1.0 | <https://jump-cellpainting.broadinstitute.org/> |
| Organ-on-a-Chip Image Dataset (Movčana et al. 2024) | The Zenodo record states CC BY 4.0; the data paper (*Data* 9:28, 2024) states CC BY-SA | <https://zenodo.org/records/10203721> |
| Released OoC features `ooc_feats_dinov2-small.npz` (release asset) | **CC BY-SA 4.0** | Derived from the Organ-on-a-Chip Image Dataset |

No images or profiles are redistributed in this repository. The release asset `ooc_feats_dinov2-small.npz` contains
image identifiers, the distributed split folder name, DINOv2-small embeddings (768 values) and 27 classical intensity
features for each of the 3,072 images. It is a derivative of the Organ-on-a-Chip (OOC) Image Dataset by Movčana V,
Strods A, Narbute K, et al. (Zenodo, doi:10.5281/zenodo.10203721; data paper *Data* 9:28, 2024,
doi:10.3390/data9020028), and because the two first-party sources differ, it is distributed under the stricter of the
two, CC BY-SA 4.0 (<https://creativecommons.org/licenses/by-sa/4.0/>); changes: features computed with
`ooc_extract.py`, no images included. `facebook/dinov2-small` is used under its own license (Apache-2.0) on the
Hugging Face Hub.

## License

Code: MIT License (see `LICENSE`). Data keep their own licenses, listed above.
