"""Ensemble evaluation: average probabilities from several trained checkpoints (NTIRE late-fusion).

Diverse members (different backbone / pooling / resolution / seed) generalize better than any
single model. Each member is scored with ITS OWN config (resolution etc.), then probabilities
are averaged. Optional horizontal-flip TTA.

  python -m src.ensemble_eval \
      --ckpts results/phase1_frozen/best.pt results/phase2_lora/best.pt results/phase2_lora_cls/best.pt \
      --manifest data/manifest_val.csv --external data/manifest_picobanana.csv --tta
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
def member_probs(ckpt_path, manifest, device, tta):
    ck = torch.load(ckpt_path, map_location="cpu")
    dc, mc = ck["cfg"]["data"], ck["cfg"]["model"]
    model = DINOv3Detector(mc["backbone"], pooling=mc.get("pooling"),
                           freeze_backbone=True, lora=mc.get("lora"))
    model.load_state_dict(ck.get("trainable", ck.get("model")), strict=False)
    model.to(device).eval()
    t = DetectionTransform(size=dc["image_size"], train=False,
                           reencode_jpeg=dc.get("reencode_jpeg", True),
                           jpeg_quality=dc.get("jpeg_quality", 90))
    dl = DataLoader(ManifestDataset(manifest, t), batch_size=64, shuffle=False,
                    num_workers=8, pin_memory=True)
    ys, ps = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(x)
            if tta:
                logit = (logit + model(torch.flip(x, dims=[3]))) / 2
        ps.append(torch.sigmoid(logit.float()).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def report(ckpts, manifest, device, tta, label):
    y0, prob_sum, per = None, None, []
    for c in ckpts:
        y, p = member_probs(c, manifest, device, tta)
        per.append((roc_auc_score(y, p), c))
        prob_sum = p if prob_sum is None else prob_sum + p
        y0 = y
    print(f"\n[{label}] per-model AUC:")
    for a, c in per:
        print(f"    {a:.4f}  {c}")
    print(f"  ENSEMBLE AUC: {roc_auc_score(y0, prob_sum / len(ckpts)):.4f}  (n={len(ckpts)}, tta={tta})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--external", default=None)
    ap.add_argument("--tta", action="store_true", help="horizontal-flip test-time augmentation")
    a = ap.parse_args()
    report(a.ckpts, a.manifest, "cuda", a.tta, "val")
    if a.external:
        report(a.ckpts, a.external, "cuda", a.tta, "external")


if __name__ == "__main__":
    main()
