"""Stream a labeled real/fake HF dataset -> resized JPEGs + manifest (real=0, fake=1).

Streaming + --n cap means you DON'T download the whole dataset (stops after N images).
Handles image columns that are PIL Images OR raw bytes (e.g. CommunityForensics 'image_data').
Label mapping is EXPLICIT via --fake-values/--real-values (unknown values are skipped, never
mislabeled) or --all-fake.

  python -m src.prep_hf_labeled --dataset bitmind/nano-banana --split train --all-fake --tag bitmind --n 9500
  python -m src.prep_hf_labeled --dataset ComplexDataLab/OpenFake --config core --split train \
      --label-col label --fake-values fake --real-values real --tag openfake --n 8000
  python -m src.prep_hf_labeled --dataset OwensLab/CommunityForensics-Small --split train \
      --image-col image_data --label-col label --fake-values 1 --real-values 0 --tag commfor --n 10000
  python -m src.prep_hf_labeled --dataset saberzl/So-Fake-Set --split train \
      --label-col label --fake-values full_synthetic,tampered --real-values real --tag sofake --n 8000
"""
import argparse, os, csv, random
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
from datasets import load_dataset


def to_pil(val):
    if isinstance(val, Image.Image):
        return val
    if isinstance(val, dict) and val.get("bytes"):
        return Image.open(BytesIO(val["bytes"]))
    if isinstance(val, (bytes, bytearray)):
        return Image.open(BytesIO(bytes(val)))
    raise ValueError(f"unhandled image type {type(val)}")


def save_resized(pil, path, size, q):
    img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if min(h, w) > size:
        s = size / min(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, q])


def pick_image_col(ex, given):
    if given:
        return given
    for c in ("image", "image_data", "img", "jpg", "png"):
        if c in ex:
            return c
    raise SystemExit(f"no image column found; keys = {list(ex.keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--image-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--fake-values", default="", help="comma list of label values meaning FAKE")
    ap.add_argument("--real-values", default="", help="comma list of label values meaning REAL")
    ap.add_argument("--all-fake", action="store_true", help="every image is fake (no label col)")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    fake_vals = {v.strip().lower() for v in a.fake_values.split(",") if v.strip()}
    real_vals = {v.strip().lower() for v in a.real_values.split(",") if v.strip()}

    out = os.path.join(a.out_dir, a.tag)
    os.makedirs(out, exist_ok=True)

    ds = load_dataset(a.dataset, a.config, split=a.split, streaming=True)

    rows, i, kept, img_col, printed = [], 0, {0: 0, 1: 0}, None, False
    for ex in ds:
        if not printed:
            printed = True
            img_col = pick_image_col(ex, a.image_col)
            lv = ex.get(a.label_col) if a.label_col else "(all-fake)"
            print(f"[schema] keys={list(ex.keys())}  image_col={img_col}  first_label={lv!r}", flush=True)

        if a.all_fake:
            label = 1
        else:
            raw = str(ex.get(a.label_col)).strip().lower()
            if raw in fake_vals:
                label = 1
            elif raw in real_vals:
                label = 0
            else:
                continue                                  # unknown -> skip, never mislabel

        try:
            pil = to_pil(ex[img_col])
            path = os.path.join(out, f"{i:06d}_{label}.jpg")
            save_resized(pil, path, a.size, a.quality)
        except Exception:
            continue
        rows.append((path, label))
        kept[label] += 1
        i += 1
        if i % 500 == 0:
            print(f"  {i}/{a.n}  (real {kept[0]}, fake {kept[1]})", flush=True)
        if i >= a.n:
            break

    if not rows:
        raise SystemExit("0 images kept — check --fake-values/--real-values against the printed first_label")

    random.Random(a.seed).shuffle(rows)
    n_val = int(round(len(rows) * a.val_frac))
    for split, rr in [("train", rows[n_val:]), ("val", rows[:n_val])]:
        p = os.path.join(a.out_dir, f"manifest_{a.tag}_{split}.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            w.writerows(rr)
        print(f"  {split}: {len(rr)} -> {p}")
    print(f"done: {a.tag}: {len(rows)} imgs (real {kept[0]}, fake {kept[1]})")


if __name__ == "__main__":
    main()
