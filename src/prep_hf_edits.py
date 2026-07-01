"""Download an instruction-edit HF dataset as a real-vs-edit set.

original image -> negative (label 0), edited image -> positive (label 1). Resized to --size.

IMPORTANT: --orig-col MUST be a REAL photo. Good source: osunlp/MagicBrush (real COCO + DALL-E-2
edits). Do NOT use datasets where the "original" is itself AI-generated (e.g.
timbrooks/instructpix2pix-clip-filtered — both images are Stable-Diffusion synthetic, no real negative).

  python -m src.prep_hf_edits --dataset osunlp/MagicBrush \
      --orig-col source_img --edit-col target_img --n 8000 --tag magicbrush --val-frac 0.1
"""
import argparse, os, csv, random
import numpy as np
import cv2
from datasets import load_dataset


def save_resized(pil_img, path, size, q):
    img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if min(h, w) > size:
        s = size / min(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, q])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset id, e.g. osunlp/MagicBrush")
    ap.add_argument("--orig-col", required=True, help="column with the REAL original image")
    ap.add_argument("--edit-col", required=True, help="column with the edited image")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--tag", required=True, help="source name (dir + manifest tag)")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pos_dir = os.path.join(a.out_dir, a.tag, "pos")
    neg_dir = os.path.join(a.out_dir, a.tag, "neg")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    ds = load_dataset(a.dataset, split=a.split, streaming=True)
    pairs, i = [], 0
    for ex in ds:
        ep = os.path.join(pos_dir, f"{i:06d}.jpg")
        npth = os.path.join(neg_dir, f"{i:06d}.jpg")
        try:
            save_resized(ex[a.edit_col], ep, a.size, a.quality)
            save_resized(ex[a.orig_col], npth, a.size, a.quality)
        except Exception:
            continue
        pairs.append((ep, npth))
        i += 1
        if i % 200 == 0:
            print(f"  {i}/{a.n}", flush=True)
        if i >= a.n:
            break

    random.Random(a.seed).shuffle(pairs)
    n_val = int(round(len(pairs) * a.val_frac))
    for split, pl in [("train", pairs[n_val:]), ("val", pairs[:n_val])]:
        rows = []
        for ep, npth in pl:
            rows += [(ep, 1), (npth, 0)]
        p = os.path.join(a.out_dir, f"manifest_{a.tag}_{split}.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            w.writerows(rows)
        print(f"  {split}: {len(rows)} -> {p}")
    print(f"done: {len(pairs)} pairs from {a.dataset}")


if __name__ == "__main__":
    main()
