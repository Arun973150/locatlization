"""DINOv3 backbone + multi-token/attention pooling head for AI-image detection.

Token layout from DINOv3: [CLS, R register tokens, P patch tokens].
Pooling options: 'cls', 'reg' (mean of register tokens), 'mean' (mean of patches),
'attn' (learned attention pool over patches). The head consumes the concatenation.
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class AttentionPool(nn.Module):
    """Single learned query attends over patch tokens -> one vector. Surfaces local cues
    (subtle/local edits) that CLS averaging dilutes."""

    def __init__(self, dim, heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, tokens):                 # tokens: [B, P, D]
        b = tokens.size(0)
        q = self.query.expand(b, -1, -1)
        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        return out.squeeze(1)                  # [B, D]


class DINOv3Detector(nn.Module):
    def __init__(self, backbone_name, pooling=("cls", "reg", "mean", "attn"),
                 freeze_backbone=True, lora=None, head_hidden=256, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        cfg = self.backbone.config
        self.dim = cfg.hidden_size
        self.num_reg = int(getattr(cfg, "num_register_tokens", 4) or 0)
        self.pooling = tuple(pooling)

        self.lora_enabled = bool(lora and lora.get("enabled"))
        if self.lora_enabled:
            from peft import LoraConfig, get_peft_model
            lc = LoraConfig(
                r=lora.get("r", 16), lora_alpha=lora.get("alpha", 32),
                lora_dropout=lora.get("dropout", 0.05),
                target_modules=lora.get("target_modules", "all-linear"), bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lc)
        elif freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        if "attn" in self.pooling:
            self.attn_pool = AttentionPool(self.dim)
        feat_dim = self.dim * len(self.pooling)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, head_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, pixel_values):
        h = self.backbone(pixel_values=pixel_values).last_hidden_state   # [B, 1+R+P, D]
        cls = h[:, 0]
        reg = h[:, 1:1 + self.num_reg].mean(1) if self.num_reg > 0 else cls
        pat = h[:, 1 + self.num_reg:]
        parts = []
        for p in self.pooling:
            if p == "cls":
                parts.append(cls)
            elif p == "reg":
                parts.append(reg)
            elif p == "mean":
                parts.append(pat.mean(1))
            elif p == "attn":
                parts.append(self.attn_pool(pat))
            else:
                raise ValueError(f"unknown pooling '{p}'")
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)           # [B] logits
