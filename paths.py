"""Data and output locations shared by all scripts.

AI4S_DATA  data root (default: <repo>/data). Expected layout:
    $AI4S_DATA/jump/meta/{plate,well,compound}.csv.gz
    $AI4S_DATA/jump/profiles_var_mad_int_featselect.parquet
    $AI4S_DATA/jump/profiles_var_mad_int_featselect_harmony.parquet
    $AI4S_DATA/ooc/OOC_image_dataset.zip
    $AI4S_DATA/ooc/OOC_datasheet.xlsx
AI4S_OUT   output folder (default: <repo>/out); used for the seed repeats, e.g. AI4S_OUT=out/seed1.
"""
import os

REPO = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("AI4S_DATA", os.path.join(REPO, "data"))
OUT_DIR = os.environ.get("AI4S_OUT", os.path.join(REPO, "out"))

JUMP_DIR = os.path.join(DATA_ROOT, "jump")
JUMP_META = os.path.join(JUMP_DIR, "meta")
OOC_DIR = os.path.join(DATA_ROOT, "ooc")
OOC_ZIP = os.path.join(OOC_DIR, "OOC_image_dataset.zip")
OOC_XLSX = os.path.join(OOC_DIR, "OOC_datasheet.xlsx")
