"""Build a Pico-Banana (Nano-Banana) EDIT dataset for training and/or held-out testing.

positives = Nano-Banana edited images (label 1); negatives = their authentic OpenImages
originals (label 0). Resized to --size on download (JPEG) so the footprint stays small.

Beats the CDN rate-limit two ways:
  * --delay throttles each request (stay under the burst limit),
  * --skip resumes deeper in the stream so repeat passes ADD new pairs.
Files are named by global record index, and the manifest is rebuilt from ALL pairs on disk
every run -> passes accumulate. Split by PAIR so an edit + its original never straddle train/val.

  # one throttled run:
  python -m src.prep_picobanana --n 20000 --delay 0.1 --workers 8
  # or accumulate in passes if it still caps:
  python -m src.prep_picobanana --skip 0     --n 8000 --delay 0.1
  python -m src.prep_picobanana --skip 30000 --n 8000 --delay 0.1
  python -m src.prep_picobanana --skip 60000 --n 8000 --delay 0.1
"""
import argparse, os, csv, json, random, time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
import cv2
import requests

JSONL = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl"
EDIT_BASE = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/"
HEADERS = {"User-Agent": "Mozilla/5.0 (research)"}


def fetch_resized(url, path, size, q, timeout=(5, 30)):
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
    idx, o, pos_dir, neg_dir, size, q, delay = task
    if delay:
        time.sleep(delay)                       # throttle to stay under the CDN burst limit
    ep = os.path.join(pos_dir, f"{idx:06d}.jpg")
    npth = os.path.join(neg_dir, f"{idx:06d}.jpg")
    try:
        fetch_resized(EDIT_BASE + o["output_image"], ep, size, q)
        fetch_resized(o["open_image_input_url"], npth, size, q)
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
    ap.add_argument("--n", type=int, default=8000, help="pairs to collect THIS run")
    ap.add_argument("--skip", type=int, default=0, help="skip first N records (resume/accumulate)")
    ap.add_argument("--delay", type=float, default=0.1, help="seconds sleep per pair (throttle)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pos_dir = os.path.join(a.out_dir, "picobanana", "pos")
    neg_dir = os.path.join(a.out_dir, "picobanana", "neg")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    r = requests.get(JSONL, stream=True, timeout=60)
    lines = r.iter_lines(decode_unicode=True)
    if a.skip:                                  # advance past skipped records
        c = 0
        for line in lines:
            if line:
                c += 1
                if c >= a.skip:
                    break
    idx = a.skip

    def next_task():
        nonlocal idx
        for line in lines:
            if not line:
                continue
            t = (idx, json.loads(line), pos_dir, neg_dir, a.size, a.quality, a.delay)
            idx += 1
            return t
        return None

    got, last = 0, 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        inflight = set()
        for _ in range(a.workers * 3):
            t = next_task()
            if t is None:
                break
            inflight.add(ex.submit(dl_pair, t))
        while inflight and got < a.n:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for f in done:
                if f.result():
                    got += 1
                    if got - last >= 200:
                        last = got
                        print(f"  {got}/{a.n} this run (scanned to {idx})", flush=True)
                if got >= a.n:
                    break
                t = next_task()
                if t is not None:
                    inflight.add(ex.submit(dl_pair, t))
        for f in inflight:
            f.cancel()
    r.close()

    # rebuild manifest from ALL pairs on disk (so passes accumulate)
    common = sorted(set(os.listdir(pos_dir)) & set(os.listdir(neg_dir)))
    allp = [(os.path.join(pos_dir, f), os.path.join(neg_dir, f)) for f in common]
    random.Random(a.seed).shuffle(allp)
    n_val = int(round(len(allp) * a.val_frac))
    for split, pl in [("train", allp[n_val:]), ("val", allp[:n_val])]:
        rows = []
        for ep, npth in pl:
            rows += [(ep, 1), (npth, 0)]
        p = os.path.join(a.out_dir, f"manifest_pico_{split}.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            w.writerows(rows)
        print(f"  {split}: {len(rows)} images -> {p}")
    print(f"done: +{got} this run; {len(allp)} total pairs on disk (scanned to record {idx})")


if __name__ == "__main__":
    main()
