"""Manifest-backed image dataset (path,label CSV)."""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, manifest_csv, transform):
        self.df = pd.read_csv(manifest_csv)
        assert {"path", "label"}.issubset(self.df.columns), "manifest needs path,label"
        self.t = transform

    def __len__(self):
        return len(self.df)

    def labels(self):
        return self.df["label"].astype(int).tolist()

    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            img = Image.open(r["path"]).convert("RGB")
        except Exception:
            # corrupt/missing image -> black frame, keep batch shape stable
            img = Image.new("RGB", (256, 256))
        return self.t(img), float(int(r["label"]))
