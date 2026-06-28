"""Sanity-check DINOv3 loads and verify token layout + LoRA targets BEFORE training.

  python -m scripts.check_backbone
Confirms: hidden_size, num_register_tokens, last_hidden_state shape, and that
patches = last_hidden_state[:, 1+num_register:]  (so the head slices correctly).
"""
import torch
from transformers import AutoModel

NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def main():
    m = AutoModel.from_pretrained(NAME).eval()
    R = int(getattr(m.config, "num_register_tokens", 4) or 0)
    print("hidden_size       :", m.config.hidden_size)
    print("num_register_tokens:", R)

    x = torch.randn(1, 3, 256, 256)              # 256 = 16*16 patches
    with torch.no_grad():
        out = m(pixel_values=x)
    n = out.last_hidden_state.shape[1]
    print("last_hidden_state :", tuple(out.last_hidden_state.shape))
    print("=> tokens =", n, "= 1 CLS +", R, "REG +", n - 1 - R, "patches")
    print("pooler_output     :",
          tuple(out.pooler_output.shape) if getattr(out, "pooler_output", None) is not None else None)

    lin = [n_ for n_, mod in m.named_modules() if mod.__class__.__name__ == "Linear"]
    print(f"Linear modules    : {len(lin)} (LoRA 'all-linear' targets these)")
    print("sample names      :", lin[:6])


if __name__ == "__main__":
    main()
