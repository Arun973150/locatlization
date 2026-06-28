# Generalized AI-Image Detector (NTIRE-2026 based) — Exact Project Plan

**Goal.** Open-world detector: *"Is this image AI-generated/edited?"* generalizing across **many generators**.
Generalization is the **objective**. Built on the **NTIRE 2026 Robust AI-Generated Image Detection in the Wild**
dataset, which already provides the multi-generator corpus, the Leave-Generators-Out split, and the robustness protocol.

**Nano-Banana tie-in.** Nano Banana is a **held-out TEST generator** in NTIRE (never in training). So detecting
it becomes a headline unseen-generator result — directly answering the original question, honestly (no training on it).

**Approach.** DINOv3 backbone (NTIRE-winner family) + multi-token/attention pooling + Focal loss + the challenge's
robustness regime, climbed frozen → LoRA → full-FT → 2-expert ensemble. Single A100-40GB, bf16.

---

## 1. Dataset — NTIRE 2026 in-the-wild (primary, pre-built)

| Property | Value |
|---|---|
| Total | **294,500** images — 108,750 real / 185,750 fake |
| Generators | **42** (open + proprietary, 2022–2026) |
| Train | ~100K real + ~177K fake, **20 open-source generators** (Flux Kontext, DeepFloyd-IF, Ovis-Image…), 1–3 gens/real img |
| Splits | toy · train · 1st-Val · **Hard-Val** · **Public-Test** · **Private-Test** (~277K labeled train) |
| Held-out test gens | **Nano Banana**, Qwen-Image, HiDream, Grok Imagine, SeeDream 5 Lite (+ proprietary) |
| Real sources | CC12M, CommonPool, RedCaps (filtered, dedup) |
| Robust track | **36 transforms**, 1–5 consecutive (noise, compression, blur, invisible watermarking, watermark-erasing adversarial) |
| Access | **Codabench competition 12761** — registration/terms required |

**Splits are pre-built → we use NTIRE's own** train / Hard-Val / Public-Test / Private-Test. LOGO is built in
(train = 20 open-source gens; test = unseen newer/proprietary incl. Nano Banana). No manual editor assembly needed.

**Optional external test (origin tie-back):** **Pico-Banana-400K** as an *extra, larger* Nano-Banana test set
(MUST stay OUT of training to keep NTIRE's Nano-Banana-held-out eval honest). Enables a clean ablation:
*train without any Nano Banana → measure Nano-Banana detection*, then *add Pico-Banana to train → measure lift*
(seen vs unseen for one model).

**Access dependency (the one blocker):** data is on Codabench 12761 and likely needs registration + accepting
the challenge terms; the challenge may be closed post-CVPR-2026. Action: register / obtain the download (or a HF
mirror) — I can script the download + prep once a URL/credentials exist, but I cannot pull gated data without access.

---

## 2. Shortcut control — partly handled by NTIRE, still audited

NTIRE already mitigates format shortcuts (real and fake span web sources; robust track applies 36 transforms;
val/test use unpaired uniques to prevent selection bias). We still:
1. **Normalize** every image identically: `decode → resize (short side S, ×16) → crop S×S → re-encode JPEG q≈90`.
2. **Audit on held-out generators** (incl. Nano Banana): report unseen-AUC with vs without normalization. If it
   only looks good *without* normalization, the model rides per-generator pipeline, not artifacts.
The **canonical score is NTIRE Robust-AUC** (their 36-transform regime) — robustness is baked into the metric.

---

## 3. Data pipeline

`prep_ntire.py`: read NTIRE splits → unified manifest `(path, label, generator, split)` → `normalize.py` (§1) →
`data/norm/`. Cache resized images (resize once, augment on the fly). Keep generator labels for per-generator eval.
`prep_picobanana.py` (optional): Pico-Banana → external Nano-Banana test only.

---

## 4. Augmentation — mirror the NTIRE robust regime

Per-image severity bucket (Clean/Mild/Moderate/Heavy): JPEG/WebP recompress (q30–95), down-up resize, blur,
Gaussian/Poisson noise, color jitter, h-flip; emulate the challenge's harder ops (multiple consecutive distortions;
optionally watermarking / watermark-erase). Applied identically to real & fake. Heavy bucket up-weighted over epochs.

---

## 5. Model ladder

**Backbone:** `facebook/dinov3-vitl16-pretrain-lvd1689m` (ViT-L, gated). Tokens `[CLS, 4 REG, patches]` →
patches = `last_hidden_state[:, 5:]`. (ViT-H+/7B = stretch on 80GB.)

1. **Frozen baseline** — MAC pool (CLS⊕4 REG⊕mean-patch) ⊕ attention-pool → MLP `d→256→1`. Audit + lower bound.
2. **LoRA (r=16–32)** on attn/MLP linears + head. Workhorse.
3. **Full fine-tune** ViT-L @512 (grad-checkpoint, bf16) — justified by 20-generator training data; best robust-AUC.
4. **2-expert ensemble** (Ant-Intl style): E1 attention-pool@512, E2 CLS@288 → late-fusion + flip-TTA + SWA.
5. *(opt)* add MetaCLIP2/SigLIP2 expert for backbone diversity.

**Loss:** Focal (γ=2.0, α=0.5) + optional Supervised-Contrastive (cluster generators by real/fake → generalization).
**Optim:** AdamW, cosine+warmup; LR head 1e-3 · LoRA 1e-4 · full-FT 2e-5; label-smooth 0.05.
**Res:** 256 (audit) → 512 (main). **Batch:** 128 frozen-head · 32 FT@512.

---

## 6. Evaluation — NTIRE protocol is the scoreboard

**Primary:** **Clean ROC-AUC** and **Robust ROC-AUC** (NTIRE 36-transform regime) on Public-Test + Private-Test.
**Breakdowns:**
- **Per held-out generator** AUC — Nano Banana, Qwen-Image, HiDream, Grok, SeeDream (the LOGO scorecard).
- **Hard-Val** vs 1st-Val (difficulty gap).
- **#training generators → unseen-AUC** curve (the generalization law).
**Ablations:** frozen→LoRA→full→ensemble; w/ vs w/o §1 normalization; w/ vs w/o Pico-Banana-in-train (Nano-Banana lift).
**Reference:** NTIRE-2026 winners reached **Robust-AUC ≈ 0.972 / Clean ≈ 0.997** (DINOv3 ensembles). Solo-A100
realistic single-model full-FT target ≈ **0.85–0.93 Robust-AUC**; ensemble pushes toward winner range.

---

## 7. Phases, milestones, gates

| Phase | Deliverable | Gate |
|---|---|---|
| 0 NTIRE access + prep + normalize | manifest, normalized splits | data in hand; normalization audit set up |
| 1 Frozen baseline | Clean/Robust-AUC + per-generator LOGO baseline | unseen-AUC survives normalization |
| 2 LoRA / full FT | improved robust-AUC | FT lifts unseen-AUC, not just seen |
| 3 Ensemble + TTA + SWA | best Clean/Robust-AUC | approaches NTIRE leaderboard range |
| 4 Nano-Banana ablation | seen-vs-unseen lift w/ Pico-Banana | quantified per-model generalization |
| 5 Writeup | LOGO scorecard + robustness curves + ablations | — |

---

## 8. Compute / storage (A100-40GB)

- Frozen extraction (ViT-L@512, ~277K): a few hours; cache CLS+pooled (<3 GB).
- LoRA @512: ~30–60 min/epoch. Full-FT ViT-L @512 (bf16, grad-ckpt, batch 32): ~1–2 h/epoch; 3–6 epochs.
- Ensemble = 2× train cost + TTA. Whole ladder ≈ 1–3 days.
- Storage (constrained: ~30 GB container + ~50 GB volume): all 6 shards (~114 GB zip / ~228 GB extracted) do NOT fit.
  Use **1 shard** (peaks ~40 GB during extract, ~20 GB resident; holds all 20 train generators → enough for baseline).
  Scale via `src/compact_shard.py` (512px cache ≈ 4 GB/shard → all 6 ≈ 24 GB). Checkpoints trainable-only (few MB). HF_HOME on the volume.

---

## 9. Risks & mitigations

- **NTIRE data access** (challenge gated/closed) → register on Codabench 12761 / find HF mirror; fallback = assemble
  own multi-generator corpus (prior plan) if unavailable.
- **Per-generator format shortcut** → §1 normalization + audit on held-out generators (Robust-AUC is the honest score).
- **Overfit to "diffusion look"** → 20-generator training + SupCon + per-generator AUC reporting; watch Nano-Banana (GAN-free, proprietary) unseen-AUC.
- **Pico-Banana leakage into NTIRE eval** → keep Pico-Banana strictly external/test-only.
- **Compute creep at 512 full-FT** → start LoRA; full-FT only if robust-AUC plateaus.
- **Licenses** (NTIRE terms, Pico-Banana CC-BY-NC-ND) → research-only, no redistribution.

---

## 10. Repo layout

```
locatlization/
  PLAN.md
  probe/derive_masks.py
  src/
    prep_ntire.py       # read NTIRE splits → manifest (Phase 0)
    prep_picobanana.py  # optional external Nano-Banana test
    normalize.py        # shortcut-control pipeline (§1)
    extract_features.py # frozen DINOv3 cache
    datasets.py augment.py
    models/heads.py     # MAC + attention-pool; ensemble wrapper
    train.py eval.py    # Focal/SupCon; Clean+Robust AUC, per-generator LOGO
  configs/{phase1,phase2,phase3}.yaml
  data/{raw,norm,features}/  results/
```

**Immediate next step:** Phase 0 — get NTIRE data access (Codabench 12761), then `prep_ntire.py` + `normalize.py`,
and run the frozen baseline to get the first **per-held-out-generator AUC (incl. Nano Banana)**.
**Blocker to clear first:** confirm you can download the NTIRE dataset (registration/terms) or point me at a mirror.
