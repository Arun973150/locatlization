"""Optional: shrink an extracted shard to a resized JPEG cache so MORE shards fit a small volume.

Parallelized across all CPU cores. Resizes each image's short side to --size (keeps aspect),
re-encodes JPEG q92, preserves the images/ + labels.csv layout (prep_ntire works unchanged),
then with --delete-raw removes the original. ~50K imgs @512px ≈ 5-6 GB/shard (vs ~20 GB raw).

  python -m src.compact_shard --shard data/ntire/shard_0 --out data/compact/shard_0 --delete-raw
"""
import argparse, os, glob, shutil
from functools import partial
from multiprocessing import Pool
import cv2

cv2.setNumThreads(1)   # one thread per worker process; avoids oversubscription


def resize_short(bgr, size):
    h, w = bgr.shape[:2]
    if min(h, w) <= size:
        return bgr
    s = size / min(h, w)
    return cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def _process(fn, img_src, out_img, size, quality):
    img = cv2.imread(os.path.join(img_src, fn), cv2.IMREAD_COLOR)
    if img is None:
        return 0
    out = resize_short(img, size)
    stem = os.path.splitext(fn)[0]
    cv2.imwrite(os.path.join(out_img, stem + ".jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="extracted shard dir (has images/ + labels.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--delete-raw", action="store_true")
    a = ap.parse_args()

    lab = glob.glob(os.path.join(a.shard, "**", "labels.csv"), recursive=True)
    if not lab:
        raise SystemExit(f"no labels.csv under {a.shard}")
    src_dir = os.path.dirname(lab[0])
    img_src = os.path.join(src_dir, "images")
    img_src = img_src if os.path.isdir(img_src) else src_dir

    out_img = os.path.join(a.out, "images")
    os.makedirs(out_img, exist_ok=True)
    shutil.copy(lab[0], os.path.join(a.out, "labels.csv"))

    files = [f for f in os.listdir(img_src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    work = partial(_process, img_src=img_src, out_img=out_img, size=a.size, quality=a.quality)
    done = 0
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(work, files, chunksize=64), 1):
            done += r
            if k % 10000 == 0:
                print(f"  {k}/{len(files)}", flush=True)
    print(f"[ok] compacted {done}/{len(files)} imgs ({a.workers} workers) -> {a.out}")
    if a.delete_raw:
        shutil.rmtree(a.shard)
        print(f"[freed] removed raw {a.shard}")


if __name__ == "__main__":
    main()
