"""Sanity-check a backbone loads + print token layout / patch size (run BEFORE training).

  python -m scripts.check_backbone                              # default DINOv3 ViT-L
  python -m scripts.check_backbone --config configs/phase1_frozen.yaml
  python -m scripts.check_backbone --backbone facebook/dinov2-with-registers-large
"""
import argparse
import torch
import yaml
from transformers import AutoModel

DEFAULT = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=DEFAULT)
    ap.add_argument("--config", default=None, help="read model.backbone from a YAML config")
    a = ap.parse_args()
    name = yaml.safe_load(open(a.config))["model"]["backbone"] if a.config else a.backbone
    print("backbone:", name)

    m = AutoModel.from_pretrained(name).eval()
    R = int(getattr(m.config, "num_register_tokens", 0) or 0)
    ps = int(getattr(m.config, "patch_size", 16) or 16)
    print(f"hidden_size={m.config.hidden_size}  patch_size={ps}  num_register_tokens={R}")

    size = ps * 16
    with torch.no_grad():
        out = m(pixel_values=torch.randn(1, 3, size, size))
    n = out.last_hidden_state.shape[1]
    print(f"input {size}x{size} -> {n} tokens = 1 CLS + {R} REG + {n - 1 - R} patches")
    print(f"=> set configs image_size to a MULTIPLE OF {ps} (e.g. {ps*16}, {ps*24}, {ps*32})")

    lin = [k for k, mod in m.named_modules() if mod.__class__.__name__ == "Linear"]
    print(f"Linear modules: {len(lin)} (LoRA 'all-linear' targets these); sample: {lin[:5]}")


if __name__ == "__main__":
    main()
