"""Combine multiple source manifests into one mixed train/val set.

Extensible: add OpenFake / COCO-Inpaint / etc. later by passing more --train / --val manifests.
--cap balances domains so a big source (NTIRE ~150K) doesn't drown a small one (Pico ~30K).
Labels are coerced to int and any bad/missing-label rows are dropped (reported per source).

  python -m src.build_mixed \
    --train data/manifest_train.csv data/manifest_pico_train.csv data/manifest_magicbrush_train.csv \
    --val   data/manifest_val.csv   data/manifest_pico_val.csv   data/manifest_magicbrush_val.csv \
    --cap 40000
"""
import argparse, os
import pandas as pd


def load_clean(path):
    df = pd.read_csv(path)[["path", "label"]].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["path", "label"])
    df["label"] = df["label"].astype(int)
    dropped = before - len(df)
    if dropped:
        print(f"  [warn] {path}: dropped {dropped} rows with bad/missing label")
    return df


def cap_df(df, cap, seed):
    if cap and len(df) > cap:                       # keep classes balanced within the cap
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), cap // 2), random_state=seed))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--cap", type=int, default=0, help="per-source row cap (0=all) to balance domains")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tr = pd.concat([cap_df(load_clean(p), a.cap, a.seed) for p in a.train], ignore_index=True)
    tr = tr.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    va = pd.concat([load_clean(p) for p in a.val], ignore_index=True)

    os.makedirs(a.out_dir, exist_ok=True)
    tr.to_csv(os.path.join(a.out_dir, "manifest_mixed_train.csv"), index=False)
    va.to_csv(os.path.join(a.out_dir, "manifest_mixed_val.csv"), index=False)
    print(f"mixed train {len(tr)}  label counts {tr['label'].value_counts().to_dict()}")
    print(f"mixed val   {len(va)}  label counts {va['label'].value_counts().to_dict()}")
    print("-> data/manifest_mixed_train.csv, data/manifest_mixed_val.csv")


if __name__ == "__main__":
    main()
