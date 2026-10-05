# Shortcut audit for imaging classifiers

Microscopy and high-content imaging classifiers often score well because they read **imaging session, lab, batch or
plate** rather than biology, and common train/test splits do not separate those variables from the label. This tool
measures that. Give it one feature table (one row per image or well) plus the label column and the nuisance columns,
and it produces a one-page HTML report with: how well each nuisance variable can be predicted from the features
(shortcut capacity), how well the label can be predicted from the nuisance variables alone (confounding baseline),
and the score gap between a random split and a held-out-group split. Every probe uses the same classifier
(standardize + logistic regression, C=0.1) and the same metric (balanced accuracy). Nuisance-probe and split-comparison
scores are printed next to a permutation null computed on the same split; the label-from-nuisance baseline is read
against chance. A confound-injection test is available as a library function (`shortcut_audit.confound_injection`) and is
used by the reproduction scripts, but is not yet a row of the report. It runs on a laptop CPU, with no paid services and no
private data. The two public datasets used for reproduction are JUMP Cell Painting (CC0) and the Organ-on-a-Chip
Image Dataset (CC-BY-4.0).

## Layout

| File | Purpose |
|---|---|
| `audit.py` | **Entry script.** Feature table in, HTML report out |
| `shortcut_audit.py` | The probes (importable library) |
| `test_shortcut_audit.py` | Synthetic positive/negative-control tests |
| `examples/make_synthetic.py` | Writes a small synthetic table for a no-download quickstart |
| `paths.py` | Data root (`AI4S_DATA`, default `./data`) and output dir (`./out`) |
| `reproduce_jump.py`, `jump_controls.py` | JUMP reproduction |
| `ooc_meta_probe.py`, `ooc_extract.py`, `ooc_probe.py`, `ooc_table.py` | Organ-on-a-Chip (OOC) reproduction |

## Install

```bash
conda env create -f environment.yml && conda activate shortcut-audit
# or
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

`torch` and `transformers` are needed only for `ooc_extract.py` (DINOv2 image features). The audit itself needs
numpy, pandas, scikit-learn, pyarrow and openpyxl. Tested with numpy 2.2, pandas 2.3, scikit-learn 1.7, pyarrow 25.

## Quickstart on synthetic data (no downloads)

```bash
python test_shortcut_audit.py          # expected last line: PASS
python examples/make_synthetic.py      # writes out/synthetic_audit.csv
python audit.py out/synthetic_audit.csv --label quality --nuisance session cell_type \
    --group session --out out/synthetic_report.html
```

The synthetic table has a weak label signal, a strong session signal and sessions partly confounded with the label.
The report should flag session as "shortcut capacity present" and "confounded", and show the held-out-session score
below the random-split score.

## `audit.py` usage

```
python audit.py TABLE --label COL --nuisance COL [COL ...]
                [--group COL] [--control-value VALUE] [--features COL ...]
                [--perm N] [--out report.html] [--title TEXT]
```

- `TABLE`: csv, parquet or npz. One row per sample.
- `--label`: the biological label to be classified.
- `--nuisance`: candidate nuisance columns (session, lab, batch, plate, experimenter ...).
- `--group`: grouping unit for the nuisance-probe split (for example plate when predicting the lab).
- `--control-value`: run the nuisance probe only on samples with this label (for example `DMSO`), so biology is held constant.
- `--features`: feature columns; default is every numeric column that is not metadata.
- `--perm`: label permutations per probe for the null baseline (default 5, 0 = off).

Display thresholds in the report (aids, not statistical tests; all numbers are shown):
nuisance probe >= 2x chance and above the permutation-null maximum is "shortcut capacity present";
label-from-nuisance >= chance + 0.05 is "confounded"; a held-out-group score >= 0.05 below the random split is
"part of the score leans on the shortcut".

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

Data: Organ-on-a-Chip Image Dataset (Movcana et al., *Data* 2024), Zenodo record 10203721, **CC-BY-4.0**:
<https://zenodo.org/records/10203721>. Download `OOC_image_dataset.zip` and `OOC_datasheet.xlsx` into `data/ooc/`.

```bash
python ooc_meta_probe.py                      # metadata only, no images
python ooc_extract.py                         # DINOv2-small + classical features; GPU optional, CPU slower
python ooc_probe.py                           # all image probes -> out/ooc_probes.json
python ooc_table.py                           # table for audit.py
python audit.py out/ooc_audit_table.parquet --label quality --nuisance session cell_type --group session \
    --title "OOC brightfield quality classifier" --out out/ooc_audit_report.html
```

`ooc_extract.py` downloads `facebook/dinov2-small` from the Hugging Face Hub on first use (free, no token).

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
(scikit-learn 1.9.1, Python 3.11); other machines can differ in the third decimal. Permutation nulls for every split were 0.123-0.127
(1/k = 0.125), so grouped CV was not biased here.

## Data licenses

| Dataset | License | Source |
|---|---|---|
| JUMP Cell Painting `cpg0016` (profiles, metadata) | CC0 1.0 | <https://jump-cellpainting.broadinstitute.org/> |
| Organ-on-a-Chip Image Dataset (Movcana et al. 2024) | CC-BY-4.0 | <https://zenodo.org/records/10203721> |

No data is redistributed in this repository. `facebook/dinov2-small` is used under its own license on the Hugging Face Hub.

## License

Code: MIT License (see `LICENSE`). Data keep their own licenses, listed above.
