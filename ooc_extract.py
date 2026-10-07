"""Extract DINOv2 features and classical features (brightness statistics) from OOC Image Dataset images.

Run: python ooc_extract.py [--model facebook/dinov2-small] [--device auto|cpu|cuda|mps] [--limit N]
Input: $AI4S_DATA/ooc/OOC_image_dataset.zip (read directly from the zip, not unpacked; see paths.py)
Output: out/ooc_feats_<model>.npz with ids, split (distributed split folder name), dino, classic

The classical features are a control for "can session be recognized without deep learning": in brightfield,
brightness, contrast and illumination gradient can differ between imaging sessions.
--device auto (default) picks CUDA, then Apple MPS, then CPU. Use --device cpu if the GPU path fails: on older
PyTorch builds MPS lacks the bicubic interpolation that DINOv2 uses to resize its position embeddings to 448 px
(unsupported MPS operations fall back to the CPU via PYTORCH_ENABLE_MPS_FALLBACK, set below).
"""
import argparse
import io
import os
import sys
import zipfile

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # must be set before torch is imported

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModel  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import OOC_ZIP as ZIP, OUT_DIR as OUT  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def classic(gray):
    x = gray.astype(np.float32).ravel()
    h, w = gray.shape
    # Illumination gradient: difference of left/right and top/bottom half means
    lr = gray[:, : w // 2].mean() - gray[:, w // 2:].mean()
    tb = gray[: h // 2].mean() - gray[h // 2:].mean()
    hist = np.histogram(x, bins=16, range=(0, 255))[0] / x.size
    return np.concatenate([[x.mean(), x.std(), *np.percentile(x, [1, 5, 50, 95, 99]), lr, tb, h, w], hist])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--zip", default=ZIP)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        dev = a.device
    model = AutoModel.from_pretrained(a.model).to(dev).eval()
    zf = zipfile.ZipFile(a.zip)
    names = sorted(n for n in zf.namelist() if n.lower().endswith(EXT) and "__MACOSX" not in n)
    if a.limit:
        names = names[: a.limit]
    print(f"images {len(names)} · device {dev}", flush=True)
    ids, splits, dino, cls = [], [], [], []
    batch, meta = [], []

    def flush():
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(dev)
        with torch.no_grad():
            o = model(pixel_values=x)
        h = o.last_hidden_state
        f = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy()
        dino.extend(f)
        batch.clear()

    for i, n in enumerate(names):
        im = Image.open(io.BytesIO(zf.read(n)))
        g = np.asarray(im.convert("L"))
        cls.append(classic(g))
        rgb = np.asarray(im.convert("RGB").resize((a.size, a.size), Image.BILINEAR), dtype=np.float32) / 255.0
        batch.append((rgb - MEAN) / STD)
        parts = n.split("/")
        ids.append(os.path.splitext(parts[-1])[0])
        splits.append(next((p for p in parts if p.lower() in ("train", "val", "test")), ""))
        if len(batch) == 32:
            flush()
        if i % 500 == 0:
            print(" ", i, n, flush=True)
    if batch:
        flush()
    os.makedirs(a.out, exist_ok=True)
    tag = a.model.split("/")[-1]
    path = os.path.join(a.out, f"ooc_feats_{tag}.npz")
    np.savez_compressed(path, ids=np.array(ids), split=np.array(splits),
                        dino=np.array(dino, dtype=np.float32), classic=np.array(cls, dtype=np.float32))
    print("saved", path, len(ids))


if __name__ == "__main__":
    main()
