"""Manifest-backed image dataset (path,label CSV)."""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, manifest_csv, transform):
        self.df = pd.read_csv(manifest_csv)
        assert {"path", "label"}.issubset(self.df.columns), "manifest needs path,label"
        # coerce labels to int, drop any bad/missing-label rows
        self.df["label"] = pd.to_numeric(self.df["label"], errors="coerce")
        self.df = self.df.dropna(subset=["path", "label"]).reset_index(drop=True)
        self.df["label"] = self.df["label"].astype(int)
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
