"""Write a tiny fake shard (random JPGs + labels.csv) so the pipeline can be smoke-tested
on CPU without the 114 GB download.

  python -m tests.make_synthetic
  python -m src.prep_ntire --data-root data/ntire/shard_synthetic --out-dir data
  python -m src.train --config configs/phase1_frozen.yaml   # proves the loop runs end-to-end
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

ROOT = "data/ntire/shard_synthetic"
IMG = os.path.join(ROOT, "images")


def main(n=64):
    os.makedirs(IMG, exist_ok=True)
    rows = []
    for i in range(n):
        # crude class signal so AUC isn't exactly 0.5: fakes a touch smoother
        arr = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        label = i % 2
        if label == 1:
            arr = (arr * 0.5 + 64).astype(np.uint8)
        fn = f"img{i:03d}.jpg"
        Image.fromarray(arr).save(os.path.join(IMG, fn), quality=90)
        rows.append((fn, label))
    pd.DataFrame(rows, columns=["filename", "label"]).to_csv(os.path.join(ROOT, "labels.csv"), index=False)
    print(f"wrote {n} images + labels.csv -> {ROOT}")


if __name__ == "__main__":
    main()
