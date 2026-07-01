"""Build a Pico-Banana (Nano-Banana) EDIT dataset for training and/or held-out testing.

positives = Nano-Banana edited images (label 1); negatives = their authentic OpenImages
originals (label 0). Images are RESIZED to --size on download (JPEG) so the footprint stays
small: ~30K images @512px ≈ ~3 GB (vs tens of GB full-res). Splits by PAIR (source image),
so an edit and its original never straddle train/val.

  # small held-out test only (default):
  python -m src.prep_picobanana --n 500 --val-frac 1.0
  # a training set (+ small val), resized:
  python -m src.prep_picobanana --n 15000 --val-frac 0.1

Writes: manifest_pico_train.csv, manifest_pico_val.csv (path,label) under --out-dir.
"""
import argparse, os, csv, json, random
import numpy as np
import cv2
import requests

JSONL = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl"
EDIT_BASE = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/"
HEADERS = {"User-Agent": "Mozilla/5.0 (research)"}


def fetch_resized(url, path, size, q, timeout=60):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("decode failed")
    h, w = img.shape[:2]
    if min(h, w) > size:                      # resize short side down to `size`
        s = size / min(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, q])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="number of edit/original PAIRS to keep")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--val-frac", type=float, default=1.0,
                    help="fraction of pairs -> val (1.0 = pure test set; 0.1 = mostly train)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pos_dir = os.path.join(a.out_dir, "picobanana", "pos")
    neg_dir = os.path.join(a.out_dir, "picobanana", "neg")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    pairs, got = [], 0
    r = requests.get(JSONL, stream=True, timeout=60)
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        o = json.loads(line)
        ep = os.path.join(pos_dir, f"{got:06d}.jpg")
        npth = os.path.join(neg_dir, f"{got:06d}.jpg")
        try:
            fetch_resized(EDIT_BASE + o["output_image"], ep, a.size, a.quality)   # edit -> 1
            fetch_resized(o["open_image_input_url"], npth, a.size, a.quality)      # real -> 0
        except Exception:
            continue
        pairs.append((ep, npth))
        got += 1
        if got % 100 == 0:
            print(f"  {got}/{a.n} pairs", flush=True)
        if got >= a.n:
            break
    r.close()

    random.Random(a.seed).shuffle(pairs)
    n_val = int(round(len(pairs) * a.val_frac))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    def write(split, plist):
        rows = []
        for ep, npth in plist:
            rows += [(ep, 1), (npth, 0)]
        path = os.path.join(a.out_dir, f"manifest_pico_{split}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            w.writerows(rows)
        print(f"  {split}: {len(rows)} images -> {path}")

    write("train", train_pairs)
    write("val", val_pairs)
    print(f"done: {got} pairs ({len(train_pairs)} train / {len(val_pairs)} val)")


if __name__ == "__main__":
    main()
