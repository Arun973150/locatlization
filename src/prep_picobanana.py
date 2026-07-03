"""Build a Pico-Banana (Nano-Banana) EDIT dataset for training and/or held-out testing.

positives = Nano-Banana edited images (label 1); negatives = their authentic OpenImages
originals (label 0). Resized to --size on download (JPEG) so the footprint stays small.

Robustness/scale:
  * the 178MB record file (sft.jsonl) is downloaded ONCE to data/ and read locally
    (no fragile live streaming -> no ChunkedEncodingError; --skip becomes instant),
  * --delay throttles each request (stay under the CDN burst limit),
  * --skip resumes deeper so repeat passes ADD new pairs; manifest rebuilt from ALL on-disk pairs,
  * fetch_resized retries up to 3x with exponential backoff and 429/Retry-After support,
  * 300 consecutive failures triggers a clean abort with a --skip resume hint.

  python -m src.prep_picobanana --n 20000 --delay 0.5 --workers 4
"""
import argparse, os, csv, json, random, time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
import cv2
import requests

JSONL = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl"
EDIT_BASE = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/"
HEADERS = {"User-Agent": "Mozilla/5.0 (research)"}


def get_jsonl_path(out_dir, retries=4):
    """Download sft.jsonl once to disk (resumable across attempts); return local path."""
    local = os.path.join(out_dir, "sft.jsonl")
    if os.path.exists(local) and os.path.getsize(local) > 150_000_000:
        return local
    tmp = local + ".part"
    for attempt in range(1, retries + 1):
        try:
            print(f"[jsonl] downloading metadata (~178MB), attempt {attempt}...", flush=True)
            with requests.get(JSONL, stream=True, timeout=(5, 120)) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            os.replace(tmp, local)
            print(f"[jsonl] cached -> {local}", flush=True)
            return local
        except Exception as e:
            print(f"[jsonl] attempt {attempt} failed: {e}", flush=True)
            time.sleep(3)
    raise RuntimeError("could not download sft.jsonl after retries")


def fetch_resized(url, path, size, q, timeout=(5, 30)):
    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 0.5 * attempt)))
                continue
            r.raise_for_status()
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("decode failed")
            h, w = img.shape[:2]
            if min(h, w) > size:
                s = size / min(h, w)
                img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, q])
            return
        except Exception:
            if attempt < 3:
                time.sleep(0.5 * attempt)
    raise RuntimeError(f"fetch failed after 3 attempts: {url}")


def dl_pair(task):
    idx, o, pos_dir, neg_dir, size, q, delay = task
    if delay:
        time.sleep(delay)
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
    ap.add_argument("--delay", type=float, default=0.5, help="seconds sleep per pair (throttle)")
    ap.add_argument("--workers", type=int, default=4)
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

    fh = open(get_jsonl_path(a.out_dir), "r", encoding="utf-8")
    for _ in range(a.skip):                     # instant skip in a local file
        fh.readline()
    idx = a.skip

    def next_task():
        nonlocal idx
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            t = (idx, json.loads(raw), pos_dir, neg_dir, a.size, a.quality, a.delay)
            idx += 1
            return t
        return None

    got, last = 0, 0
    fails, streak = 0, 0
    aborted = False

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        inflight = set()
        for _ in range(a.workers * 3):
            t = next_task()
            if t is None:
                break
            inflight.add(ex.submit(dl_pair, t))
        while inflight and got < a.n and not aborted:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for f in done:
                result = f.result()
                if result:
                    got += 1
                    streak = 0
                    if got - last >= 200:
                        last = got
                        print(f"  {got}/{a.n} this run  (scanned {idx}, fails {fails})", flush=True)
                else:
                    fails += 1
                    streak += 1
                    if fails % 100 == 0:
                        print(
                            f"  [warn] {fails} total failures, {streak} consecutive (scanned {idx})",
                            flush=True,
                        )
                    if streak >= 300:
                        print(
                            f"  [abort] {streak} consecutive failures — CDN is likely rate-limiting.\n"
                            f"  Resume later with:  --skip {idx}",
                            flush=True,
                        )
                        aborted = True
                        break
                if got >= a.n or aborted:
                    break
                t = next_task()
                if t is not None:
                    inflight.add(ex.submit(dl_pair, t))
        for f in inflight:
            f.cancel()
    fh.close()

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
