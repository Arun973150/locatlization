"""Run the trained detector on one or more images -> P(AI-edited/generated).

  python -m src.predict --ckpt results/phase2_lora_mixed/best.pt some_image.jpg another.png

Applies the SAME normalization used in training (resize + JPEG re-encode), so results are
consistent with the eval numbers. Prints a probability in [0,1] and a REAL/AI verdict at 0.5.
"""
import argparse
import torch
from PIL import Image

from src.transforms_det import DetectionTransform
from src.models import DINOv3Detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("images", nargs="+", help="image path(s) to classify")
    ap.add_argument("--thresh", type=float, default=0.5)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    dc, mc = ck["cfg"]["data"], ck["cfg"]["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DINOv3Detector(mc["backbone"], pooling=mc.get("pooling"),
                           freeze_backbone=True, lora=mc.get("lora"))
    model.load_state_dict(ck.get("trainable", ck.get("model")), strict=False)
    model.to(device).eval()

    t = DetectionTransform(size=dc["image_size"], train=False,
                           reencode_jpeg=dc.get("reencode_jpeg", True),
                           jpeg_quality=dc.get("jpeg_quality", 90))

    for path in a.images:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"{path}: could not open ({e})")
            continue
        x = t(img).unsqueeze(0).to(device)
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = model(x)
            else:
                logit = model(x)
        p = torch.sigmoid(logit.float()).item()
        verdict = "AI-edited/generated" if p >= a.thresh else "REAL"
        print(f"{path}\n    P(AI) = {p:.4f}  ->  {verdict}")


if __name__ == "__main__":
    main()
