"""Salvage: build Pico manifests from already-downloaded pos/neg images.

Use this if a prep_picobanana run stalls/was killed before it wrote its manifest — the images
on disk are still good. Pairs are matched by filename (only complete pos+neg pairs are kept),
split by pair into train/val.

  python -m src.pico_manifest_from_disk --val-frac 0.1
"""
import argparse, os, csv, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/picobanana")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pos, neg = os.path.join(a.dir, "pos"), os.path.join(a.dir, "neg")
    common = sorted(set(os.listdir(pos)) & set(os.listdir(neg)))   # complete pairs only
    pairs = [(os.path.join(pos, f), os.path.join(neg, f)) for f in common]
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
        print(f"  {split}: {len(rows)} -> {p}")
    print(f"built from {len(pairs)} on-disk pairs")


if __name__ == "__main__":
    main()
