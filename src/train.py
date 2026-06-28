"""Phase-1+ training: DINOv3 detector, Focal loss, class-balanced sampling, bf16, cosine LR.

Run from repo root:  python -m src.train --config configs/phase1_frozen.yaml
"""
import argparse, os, csv, random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from transformers import get_cosine_schedule_with_warmup

from src.dset import ManifestDataset
from src.transforms_det import DetectionTransform
from src.losses import BinaryFocalLoss
from src.models import DINOv3Detector


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def balanced_sampler(labels):
    labels = np.asarray(labels)
    class_w = 1.0 / np.bincount(labels)
    w = class_w[labels]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)


@torch.no_grad()
def eval_auc(model, loader, device, dtype):
    model.eval(); ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype):
            logit = model(x)
        ps.append(torch.sigmoid(logit.float()).cpu().numpy())
        ys.append(y.numpy())
    return roc_auc_score(np.concatenate(ys), np.concatenate(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))

    set_seed(cfg.get("seed", 0))
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    dtype = torch.bfloat16 if cfg["train"].get("amp_dtype", "bf16") == "bf16" else torch.float16

    dc, tc, mc = cfg["data"], cfg["train"], cfg["model"]
    common = dict(size=dc["image_size"], reencode_jpeg=dc.get("reencode_jpeg", True),
                  jpeg_quality=dc.get("jpeg_quality", 90))
    tr_ds = ManifestDataset(dc["train_manifest"], DetectionTransform(train=True, **common))
    va_ds = ManifestDataset(dc["val_manifest"], DetectionTransform(train=False, **common))

    nw = tc.get("num_workers", 8)
    tr_dl = DataLoader(tr_ds, batch_size=tc["batch_size"], sampler=balanced_sampler(tr_ds.labels()),
                       num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=nw > 0)
    va_dl = DataLoader(va_ds, batch_size=tc["batch_size"], shuffle=False,
                       num_workers=nw, pin_memory=True, persistent_workers=nw > 0)

    model = DINOv3Detector(mc["backbone"], pooling=mc.get("pooling", ["cls", "reg", "mean", "attn"]),
                           freeze_backbone=mc.get("freeze_backbone", True), lora=mc.get("lora")).to(device)

    head_p, bb_p, trainable_names = [], [], set()
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable_names.add(n)
            (bb_p if n.startswith("backbone") else head_p).append(p)
    groups = [{"params": head_p, "lr": tc["lr_head"]}]
    if bb_p:
        groups.append({"params": bb_p, "lr": tc.get("lr_backbone", 1e-4)})
    n_train = sum(p.numel() for g in groups for p in g["params"])
    print(f"trainable params: {n_train/1e6:.2f}M  (backbone groups: {len(bb_p)>0})")

    opt = torch.optim.AdamW(groups, weight_decay=tc.get("weight_decay", 0.05))
    total = len(tr_dl) * tc["epochs"]
    sch = get_cosine_schedule_with_warmup(opt, int(total * tc.get("warmup_ratio", 0.05)), total)
    crit = BinaryFocalLoss(tc.get("focal_gamma", 2.0), tc.get("focal_alpha", 0.5))

    out = os.path.join("results", cfg.get("run_name", "run"))
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.csv"), "w", newline="")
    lw = csv.writer(logf); lw.writerow(["epoch", "train_loss", "val_auc"])

    best = 0.0
    for ep in range(tc["epochs"]):
        model.train(); tot = 0.0; nb = 0
        for x, y in tr_dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=dtype):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step(); sch.step()
            tot += loss.item(); nb += 1
        auc = eval_auc(model, va_dl, device, dtype)
        lw.writerow([ep, tot / max(nb, 1), auc]); logf.flush()
        print(f"epoch {ep}  loss {tot/max(nb,1):.4f}  val_auc {auc:.4f}", flush=True)
        if auc > best:
            best = auc
            # save only trainable tensors (head/attn/LoRA) — frozen backbone reloads from HF
            sd = {k: v for k, v in model.state_dict().items() if k in trainable_names}
            torch.save({"trainable": sd, "cfg": cfg, "auc": auc}, os.path.join(out, "best.pt"))
    print(f"[done] best val AUC {best:.4f}  ->  {out}/best.pt")


if __name__ == "__main__":
    main()
