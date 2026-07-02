"""Build a Pico-Banana (Nano-Banana) EDIT dataset for training and/or held-out testing.

positives = Nano-Banana edited images (label 1); negatives = their authentic OpenImages
originals (label 0). Images are RESIZED to --size on download (JPEG) so the footprint stays
small (~2 imgs/pair * ~100KB). Downloads run in a bounded thread pool that STREAMS records
until it collects --n successful pairs (so a low success rate just means it scans deeper,
instead of running out). Splits by PAIR so an edit and its original never straddle train/val.

  python -m src.prep_picobanana --n 8000 --val-frac 0.1
"""
import argparse, os, csv, json, random
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
import cv2
import requests

JSONL = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl"
EDIT_BASE = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/"
HEADERS = {"User-Agent": "Mozilla/5.0 (research)"}


def fetch_resized(url, path, size, q, timeout=(5, 30)):
    # (connect, read): dead hosts fail fast at connect; large images get up to 30s to read
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
    ap.add_argument("--n", type=int, default=8000, help="number of edit/original PAIRS to keep")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=16, help="fewer = less CDN throttling")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pos_dir = os.path.join(a.out_dir, "picobanana", "pos")
    neg_dir = os.path.join(a.out_dir, "picobanana", "neg")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    r = requests.get(JSONL, stream=True, timeout=60)
    lines = r.iter_lines(decode_unicode=True)
    idx = 0

    def next_task():
        nonlocal idx
        for line in lines:
            if not line:
                continue
            t = (idx, json.loads(line), pos_dir, neg_dir, a.size, a.quality)
            idx += 1
            return t
        return None

    pairs, last_print = [], 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        inflight = set()
        for _ in range(a.workers * 3):                 # prime the pool
            t = next_task()
            if t is None:
                break
            inflight.add(ex.submit(dl_pair, t))
        while inflight and len(pairs) < a.n:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for f in done:
                res = f.result()
                if res:
                    pairs.append(res)
                    if len(pairs) - last_print >= 200:
                        last_print = len(pairs)
                        print(f"  {len(pairs)}/{a.n} pairs (scanned {idx})", flush=True)
                if len(pairs) >= a.n:
                    break
                t = next_task()                        # keep the pool full from the stream
                if t is not None:
                    inflight.add(ex.submit(dl_pair, t))
        for f in inflight:
            f.cancel()
    r.close()
    pairs = pairs[:a.n]

    random.Random(a.seed).shuffle(pairs)
    n_val = int(round(len(pairs) * a.val_frac))
    for split, pl in [("train", pairs[n_val:]), ("val", pairs[:n_val])]:
        rows = []
        for ep, npth in pl:
            rows += [(ep, 1), (npth, 0)]
        p = os.path.join(a.out_dir, f"manifest_pico_{split}.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            w.writerows(rows)
        print(f"  {split}: {len(rows)} images -> {p}")
    print(f"done: {len(pairs)} pairs (scanned {idx} records)")


if __name__ == "__main__":
    main()
