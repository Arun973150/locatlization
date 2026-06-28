"""
Pico-Banana SSIM mask-derivation probe.

Goal: answer "can we derive usable edit-region masks from Pico-Banana pairs?"
We sample pairs across LOCAL / GLOBAL / GEOMETRIC edit types, derive a candidate
mask from (original, edited), and quantify whether it is localizable or smeared.

No ground-truth masks exist in Pico-Banana, so this is the gating experiment that
decides whether the dataset is a SUPERVISION source or only an EVAL set.
"""
import io, os, sys, json, csv, time
import requests
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

JSONL = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl"
EDIT_BASE = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/"
HEADERS = {"User-Agent": "Mozilla/5.0 (research probe)"}

# expected bucket per edit_type
BUCKET = {
    "Remove an existing object": "LOCAL",
    "Add a new object to the scene": "LOCAL",
    "Replace one object category with another": "LOCAL",
    "Change an object's attribute (e.g., color/material)": "LOCAL",
    "Change overall color tone (warm ↔ cool)": "GLOBAL",
    "Add film grain or vintage filter": "GLOBAL",
    "Strong artistic style transfer (e.g., Van Gogh/anime/etc.)": "GLOBAL",
    "Zoom in": "GEOMETRIC",
    "Outpainting (extend canvas beyond boundaries)": "GEOMETRIC",
}
N_PER_TYPE = 2
MAXDIM = 1024  # downscale originals for speed


def fetch_img(url, timeout=40):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    arr = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("decode failed")
    return img


def sample_records():
    """Stream sft.jsonl, collect up to N_PER_TYPE per target edit_type."""
    want = {k: N_PER_TYPE for k in BUCKET}
    got = {k: [] for k in BUCKET}
    r = requests.get(JSONL, stream=True, timeout=60)
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        o = json.loads(line)
        et = o["edit_type"]
        if et in want and len(got[et]) < want[et]:
            got[et].append(o)
        if all(len(got[k]) >= want[k] for k in want):
            break
    r.close()
    recs = []
    for k in BUCKET:
        recs.extend(got[k])
    return recs


def derive_mask(orig, edited):
    """Return (mask_uint8, heat_uint8, meta) for an aligned pair."""
    h, w = orig.shape[:2]
    edited = cv2.resize(edited, (w, h), interpolation=cv2.INTER_AREA)

    # structural difference
    g1 = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)
    _, smap = ssim(g1, g2, data_range=255, full=True)
    d_ssim = 1.0 - smap  # 0..2, but ~0..1 typical

    # perceptual color difference in LAB
    lab1 = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(edited, cv2.COLOR_BGR2LAB).astype(np.float32)
    d_lab = np.sqrt(((lab1 - lab2) ** 2).sum(2))
    d_lab /= (d_lab.max() + 1e-6)

    comb = 0.5 * np.clip(d_ssim, 0, 1) + 0.5 * d_lab
    comb = cv2.GaussianBlur(comb, (0, 0), sigmaX=2.0)
    heat = (255 * comb / (comb.max() + 1e-6)).astype(np.uint8)

    # Otsu threshold on the combined diff
    thr_in = (255 * comb / (comb.max() + 1e-6)).astype(np.uint8)
    _, mask = cv2.threshold(thr_in, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # drop tiny components
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    total = h * w
    keep = np.zeros_like(mask)
    areas = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a >= 0.001 * total:
            keep[lab == i] = 255
            areas.append(a)
    mask = keep
    areas.sort(reverse=True)

    cov = float(mask.sum() / 255) / total
    n_comp = len(areas)
    largest_frac = (areas[0] / sum(areas)) if areas else 0.0
    meta = dict(coverage=cov, n_comp=n_comp, largest_frac=largest_frac)
    return mask, heat, meta


def verdict(meta, ar_mismatch):
    if ar_mismatch:
        return "GEOM_MISMATCH (align breaks)"
    cov, n_comp, lf = meta["coverage"], meta["n_comp"], meta["largest_frac"]
    if cov < 0.002:
        return "EMPTY (edit too subtle)"
    if cov > 0.45:
        return "DIFFUSE/GLOBAL (unusable)"
    if lf >= 0.5 and n_comp <= 4:
        return "LOCALIZABLE (usable)"
    return "FRAGMENTED (marginal)"


def montage(orig, edited, heat, mask, path):
    h, w = orig.shape[:2]
    edited = cv2.resize(edited, (w, h), interpolation=cv2.INTER_AREA)
    heat_c = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = edited.copy()
    overlay[mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(edited, 0.6, overlay, 0.4, 0)
    row = cv2.hconcat([orig, edited, heat_c, overlay])
    cv2.imwrite(path, row)


def main():
    print("Sampling records across edit types...")
    recs = sample_records()
    print(f"Got {len(recs)} records.\n")

    rows = []
    for i, o in enumerate(recs):
        et = o["edit_type"]
        bucket = BUCKET[et]
        try:
            orig = fetch_img(o["open_image_input_url"])
            edited = fetch_img(EDIT_BASE + o["output_image"])
        except Exception as e:
            print(f"[{i:02d}] {bucket:9s} {et[:34]:34s}  DOWNLOAD FAIL: {e}")
            continue

        # downscale original for speed
        H, W = orig.shape[:2]
        s = MAXDIM / max(H, W)
        if s < 1:
            orig = cv2.resize(orig, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)

        ar_o = orig.shape[1] / orig.shape[0]
        ar_e = edited.shape[1] / edited.shape[0]
        ar_mismatch = abs(ar_o - ar_e) / ar_o > 0.06

        mask, heat, meta = derive_mask(orig, edited)
        v = verdict(meta, ar_mismatch)

        viz = os.path.join(OUT, f"{i:02d}_{bucket}.png")
        montage(orig, edited, heat, mask, viz)

        print(f"[{i:02d}] {bucket:9s} {et[:34]:34s}  cov={meta['coverage']:.3f} "
              f"ncc={meta['n_comp']:2d} largest={meta['largest_frac']:.2f}  -> {v}")
        rows.append(dict(idx=i, bucket=bucket, edit_type=et,
                         cov=round(meta["coverage"], 4), n_comp=meta["n_comp"],
                         largest_frac=round(meta["largest_frac"], 3),
                         ar_mismatch=ar_mismatch, verdict=v))

    with open(os.path.join(OUT, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # summary by bucket
    print("\n=== SUMMARY: verdict by expected bucket ===")
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(int))
    for r in rows:
        usable = r["verdict"].startswith("LOCALIZABLE")
        agg[r["bucket"]]["usable" if usable else "unusable"] += 1
    for b in ["LOCAL", "GLOBAL", "GEOMETRIC"]:
        print(f"  {b:9s}  usable={agg[b]['usable']}  unusable={agg[b]['unusable']}")
    print(f"\nViz + metrics.csv written to: {OUT}")


if __name__ == "__main__":
    main()
