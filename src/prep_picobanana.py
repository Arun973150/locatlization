"""Build a Pico-Banana (Nano-Banana) EDIT dataset for training and/or held-out testing.

positives = Nano-Banana edited images (label 1); negatives = their authentic OpenImages
originals (label 0). Images are RESIZED to --size on download (JPEG) so the footprint stays
small: ~30K images @512px ≈ ~3 GB. Downloads are PARALLEL (network-bound) so 15K pairs take
minutes, not hours. Splits by PAIR (source image) so an edit and its original never straddle
train/val.

  # small held-out test only:
  python -m src.prep_picobanana --n 500 --val-frac 1.0
  # training set (+ 10% held-out val), resized, parallel:
  python -m src.prep_picobanana --n 15000 --val-frac 0.1

Writes: manifest_pico_train.csv, manifest_pico_val.csv (path,label) under --out-dir.
"""
import argparse, os, csv, json, random
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    if min(h, w) > size:
        s = size / min(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, q])


def dl_pair(task):
    idx, o, pos_dir, neg_dir, size, q = task
    ep = os.path.join(pos_dir, f"{idx:06d}.jpg")
    npth = os.path.join(neg_dir, f"{idx:06d}.jpg")
    try:
        fetch_resized(EDIT_BASE + o["output_image"], ep, size, q)   # edit -> 1
        fetch_resized(o["open_image_input_url"], npth, size, q)      # real -> 0
        return (ep, npth)
    except Exception:
        for p in (ep, npth):
            try:
                os.remove(p)
            except OSError:
                pass
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="number of edit/original PAIRS to keep")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--val-frac", type=float, default=1.0,
                    help="fraction of pairs -> val (1.0 = pure test set; 0.1 = mostly train)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pos_dir = os.path.join(a.out_dir, "picobanana", "pos")
    neg_dir = os.path.join(a.out_dir, "picobanana", "neg")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    # read ~2x candidate records to survive ~40% dead Flickr links
    tasks, need = [], a.n * 2 + 100
    r = requests.get(JSONL, stream=True, timeout=60)
    for i, line in enumerate(r.iter_lines(decode_unicode=True)):
        if not line:
            continue
        tasks.append((i, json.loads(line), pos_dir, neg_dir, a.size, a.quality))
        if len(tasks) >= need:
            break
    r.close()

    pairs = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(dl_pair, t) for t in tasks]
        for f in as_completed(futs):
            res = f.result()
            if res:
                pairs.append(res)
                if len(pairs) % 200 == 0:
                    print(f"  {len(pairs)}/{a.n} pairs", flush=True)
            if len(pairs) >= a.n:
                break
        for f in futs:
            f.cancel()
    pairs = pairs[:a.n]

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
    print(f"done: {len(pairs)} pairs ({len(train_pairs)} train / {len(val_pairs)} val)")


if __name__ == "__main__":
    main()
