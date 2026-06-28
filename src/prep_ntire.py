"""Scan extracted NTIRE shards -> unified manifests (path,label) + stratified val split.

Each shard is `shard_i/.../labels.csv` next to an `images/` folder. We scan recursively
for every labels.csv so the exact nesting of the zip doesn't matter.
labels.csv maps image filename -> label (0=real, 1=AI-generated).
"""
import argparse, os, glob
import pandas as pd
from sklearn.model_selection import train_test_split


def read_labels(csv_path):
    """Robust to header/no-header and to column-name variants."""
    df = pd.read_csv(csv_path)
    lowered = [c.lower() for c in df.columns]
    if not any(c in ("label", "target", "is_fake") for c in lowered):
        # no header -> assume (filename, label)
        df = pd.read_csv(csv_path, header=None, names=["filename", "label"])
    else:
        ren = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ("filename", "image", "name", "img", "file", "image_name"):
                ren[c] = "filename"
            if cl in ("label", "target", "is_fake"):
                ren[c] = "label"
        df = df.rename(columns=ren)
    return df[["filename", "label"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="dir containing extracted shard_*/")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = []
    csvs = glob.glob(os.path.join(a.data_root, "**", "labels.csv"), recursive=True)
    if not csvs:
        raise SystemExit(f"no labels.csv found under {a.data_root}")
    for lab_csv in csvs:
        folder = os.path.dirname(lab_csv)
        img_dir = os.path.join(folder, "images")
        base = img_dir if os.path.isdir(img_dir) else folder
        df = read_labels(lab_csv)
        for fn, label in zip(df["filename"].astype(str), df["label"]):
            path = os.path.join(base, fn)
            if not os.path.splitext(fn)[1]:        # csv stored bare names
                path += ".jpg"
            rows.append((path, int(label)))

    full = pd.DataFrame(rows, columns=["path", "label"]).drop_duplicates("path")
    print("total:", len(full), "| label counts:", full["label"].value_counts().to_dict())

    tr, va = train_test_split(full, test_size=a.val_frac, stratify=full["label"],
                              random_state=a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    tr.to_csv(os.path.join(a.out_dir, "manifest_train.csv"), index=False)
    va.to_csv(os.path.join(a.out_dir, "manifest_val.csv"), index=False)
    print(f"wrote {len(tr)} train / {len(va)} val to {a.out_dir}/")


if __name__ == "__main__":
    main()
