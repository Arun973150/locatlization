"""Run the trained detector on image(s) -> P(AI-edited/generated).

For crop-mode (native-resolution) models, averages several native 512 crops so the
high-frequency fingerprint is preserved (no whole-image downscale).

  python -m src.predict --ckpt results/phase_native/best.pt --crops 8 img1.jpg img2.png
"""
import argparse
import numpy as np
import cv2
import torch
from PIL import Image

from src.transforms_det import crop_native, canonical_resize, jpeg_reencode, IMAGENET_MEAN, IMAGENET_STD
from src.models import DINOv3Detector

_MEAN = np.array(IMAGENET_MEAN, np.float32)
_STD = np.array(IMAGENET_STD, np.float32)


def to_tensor(bgr, reencode, q):
    if reencode:
        bgr = jpeg_reencode(bgr, q)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).contiguous().float()


def crops_for(pil, size, crop, ncrops):
    bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    if not crop:
        return [canonical_resize(bgr, size)]
    out = [crop_native(bgr, size, center=True)]
    for _ in range(max(0, ncrops - 1)):
        out.append(crop_native(bgr, size, center=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--crops", type=int, default=8, help="native crops to average (crop-mode models)")
    ap.add_argument("--thresh", type=float, default=0.5)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    dc, mc = ck["cfg"]["data"], ck["cfg"]["model"]
    crop = dc.get("crop_native", False)
    size, reencode, q = dc["image_size"], dc.get("reencode_jpeg", True), dc.get("jpeg_quality", 90)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DINOv3Detector(mc["backbone"], pooling=mc.get("pooling"),
                           freeze_backbone=True, lora=mc.get("lora"))
    model.load_state_dict(ck.get("trainable", ck.get("model")), strict=False)
    model.to(device).eval()

    for path in a.images:
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"{path}: could not open ({e})")
            continue
        crops = crops_for(pil, size, crop, a.crops)
        xs = torch.stack([to_tensor(c, reencode, q) for c in crops]).to(device)
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(xs)
            else:
                logits = model(xs)
        p = torch.sigmoid(logits.float()).mean().item()
        verdict = "AI-edited/generated" if p >= a.thresh else "REAL"
        print(f"{path}\n    P(AI) = {p:.4f}  ({len(crops)} crop{'s' if len(crops)>1 else ''})  ->  {verdict}")


if __name__ == "__main__":
    main()
