# RunPod setup — Generalized AI-Image Detector

## 1. Instance (storage-constrained: ~30 GB container + ~50 GB volume)
- GPU: **A100** (40 or 80GB). Template: **PyTorch 2.4+ / CUDA 12.x**.
- **Everything big goes on the ~50 GB volume at `/workspace`**: clone the repo there, and
  `export HF_HOME=/workspace/hf_cache` so the 1.2 GB backbone + data never touch the 30 GB container.
- **Budget math:** 1 shard download+extract peaks at ~40 GB (20 GB zip + 20 GB JPGs), drops to ~20 GB
  after the zip auto-deletes → fits 50 GB. Checkpoints are trainable-only (a few MB). **Use 1 shard.**
- Need more data later? `src/compact_shard.py` shrinks shards to ~4 GB each (512px cache) so all 6 fit ~24 GB.
- Use `tmux` so training survives disconnects.

## 2. Code + deps
```bash
cd /workspace
git clone <your-repo> locatlization   # or scp this folder up
cd locatlization
pip install -r requirements.txt
export HF_HOME=/workspace/hf_cache     # cache the 1GB+ backbone on the volume
export TOKENIZERS_PARALLELISM=false
```

## 3. Hugging Face access (two gated things)
```bash
huggingface-cli login                  # paste a token with read access
```
- Accept the **DINOv3 license**: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
- Accept terms on the **dataset**: https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train

## 4. Sanity-check the backbone (do this FIRST — verifies API + token layout for the head)
```bash
python -m scripts.check_backbone
# confirm: hidden_size=1024, num_register_tokens=4, last_hidden_state=[1, 1+4+256, 1024] @256px
```

## 5. Get data (start with ONE shard ≈ 20 GB to iterate fast; all shards share distribution)
```bash
python -m src.download_ntire --shards 0 --out data/ntire
python -m src.prep_ntire --data-root data/ntire --out-dir data
# -> data/manifest_train.csv, data/manifest_val.csv
```

## 6. Smoke test the pipeline WITHOUT real data (optional, runs on CPU in seconds)
```bash
python -m tests.make_synthetic
python -m src.prep_ntire --data-root data/ntire/shard_synthetic --out-dir data
python -m src.train --config configs/phase1_frozen.yaml   # tiny, just proves the loop runs
```

## 7. Phase 1 — frozen baseline (the shortcut-control audit lives here)
```bash
# normalization ON (default config): reencode_jpeg: true
python -m src.train --config configs/phase1_frozen.yaml
python -m src.eval  --ckpt results/phase1_frozen/best.pt --manifest data/manifest_val.csv

# the AUDIT: flip reencode_jpeg to false in a copied config, retrain, compare val AUC.
# big gap (norm << raw) = model was riding compression, not artifacts.
```

## 7b. Save & download the trained model
Training auto-saves `results/<run_name>/best.pt` **and** `last.pt` — *trainable weights only*, so
frozen/LoRA checkpoints are a few MB (full-FT ~1.2 GB). Get it off the pod with whichever is easiest:
- **HF Hub (best — pull it anywhere):**
  `python -m src.upload_model --ckpt results/phase1_frozen/best.pt --repo <you>/nb-detector`
  then later `huggingface-cli download <you>/nb-detector best.pt --local-dir .`
- **JupyterLab:** right-click the `.pt` → Download (fine for small frozen/LoRA ckpts).
- **runpodctl:** `runpodctl send results/phase1_frozen/best.pt` → run the printed `receive` command locally.
- **scp:** `scp root@<pod-ip>:/workspace/locatlization/results/phase1_frozen/best.pt .`

## 8. Scale up
- Add shards: `--shards 0 1 2 3 4 5` (all ~277K, ~114 GB).
- Climb the ladder: `configs/phase2_lora.yaml` → full FT → ensemble (see PLAN.md §5).
- Official score = submit test predictions to **Codabench comp 12761** (Clean + Robust AUC).
- External Nano-Banana test: prep Pico-Banana as a manifest, pass via `--external`.

## Run order recap
One shot: `bash runpod_quickstart.sh`  (DINOv3 frozen baseline; pass another config to override).
Manual: `check_backbone → download_ntire → prep_ntire → train → eval → upload_model`.
