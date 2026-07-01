"""Build an EXTERNAL held-out Nano-Banana test manifest from Pico-Banana.

positives = Nano-Banana edited images (label 1); negatives = their authentic OpenImages
originals (label 0). Nano Banana is a HELD-OUT generator in NTIRE, so this yields a real
generalization number (not seen during training).

  python -m src.prep_picobanana --n 500 --out-dir data
  python -m src.eval --ckpt results/phase1_frozen/best.pt \
      --manifest data/manifest_val.csv --external data/manifest_picobanana.csv

Note: ~40% of Flickr originals are dead links; a pair is kept only if BOTH images download,
so the set stays label-balanced. eval.py normalizes both classes identically (resize + JPEG
re-encode), which removes the PNG-vs-JPG format confound.
"""
import argparse, os, json, csv
import requests

JSONL = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl"
EDIT_BASE = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/"
HEADERS = {"User-Agent": "Mozilla/5.0 (research)"}


def fetch(url, path, timeout=60):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="number of edited/original PAIRS to keep")
    ap.add_argument("--out-dir", default="data")
    a = ap.parse_args()

    pos_dir = os.path.join(a.out_dir, "picobanana", "pos")
    neg_dir = os.path.join(a.out_dir, "picobanana", "neg")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    rows = []
    got = 0
    r = requests.get(JSONL, stream=True, timeout=60)
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        o = json.loads(line)
        ep = os.path.join(pos_dir, f"{got:05d}.png")
        npth = os.path.join(neg_dir, f"{got:05d}.jpg")
        try:
            fetch(EDIT_BASE + o["output_image"], ep)   # Nano-Banana edit -> positive
            fetch(o["open_image_input_url"], npth)      # authentic original -> negative
        except Exception:
            continue
        rows += [(ep, 1), (npth, 0)]
        got += 1
        if got % 50 == 0:
            print(f"  {got}/{a.n} pairs", flush=True)
        if got >= a.n:
            break
    r.close()

    out = os.path.join(a.out_dir, "manifest_picobanana.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)
    print(f"wrote {len(rows)} images ({got} pairs) -> {out}")


if __name__ == "__main__":
    main()
