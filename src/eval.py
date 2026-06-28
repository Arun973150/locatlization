"""Evaluate a checkpoint: Clean AUC + Robust AUC (per severity) + optional external test.

  python -m src.eval --ckpt results/phase1_frozen/best.pt --manifest data/manifest_val.csv
  # external held-out Nano-Banana set (Pico-Banana built into a path,label manifest):
  python -m src.eval --ckpt ... --manifest data/manifest_val.csv --external data/manifest_picobanana.csv
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.dset import ManifestDataset
from src.transforms_det import DetectionTransform
from src.models import DINOv3Detector


@torch.no_grad()
def score(model, manifest, transform, bs, device, dtype):
    dl = DataLoader(ManifestDataset(manifest, transform), batch_size=bs, shuffle=False,
                    num_workers=8, pin_memory=True)
    ys, ps = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype):
            logit = model(x)
        ps.append(torch.sigmoid(logit.float()).cpu().numpy())
        ys.append(y.numpy())
    return roc_auc_score(np.concatenate(ys), np.concatenate(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--external", default=None, help="optional held-out test manifest (e.g. Pico-Banana)")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    cfg, dc, mc = ck["cfg"], ck["cfg"]["data"], ck["cfg"]["model"]
    device, dtype = "cuda", torch.bfloat16
    bs = cfg["train"]["batch_size"]

    model = DINOv3Detector(mc["backbone"], pooling=mc.get("pooling"),
                           freeze_backbone=True, lora=mc.get("lora"))
    # checkpoint holds only trainable tensors; backbone comes from from_pretrained
    state = ck.get("trainable", ck.get("model"))
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    common = dict(size=dc["image_size"], reencode_jpeg=dc.get("reencode_jpeg", True),
                  jpeg_quality=dc.get("jpeg_quality", 90))
    clean = DetectionTransform(train=False, **common)

    print(f"CLEAN              AUC {score(model, a.manifest, clean, bs, device, dtype):.4f}")
    for sev in ("mild", "moderate", "heavy"):
        t = DetectionTransform(train=False, fixed_severity=sev, **common)
        print(f"ROBUST[{sev:8s}]  AUC {score(model, a.manifest, t, bs, device, dtype):.4f}")
    if a.external:
        print(f"EXTERNAL(unseen)   AUC {score(model, a.external, clean, bs, device, dtype):.4f}")


if __name__ == "__main__":
    main()
