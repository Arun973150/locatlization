#!/usr/bin/env bash
# One-shot: verify backbone -> download 1 shard -> prep -> train.
#   bash runpod_quickstart.sh                               # DINOv3 frozen baseline (default)
#   bash runpod_quickstart.sh configs/phase2_lora.yaml      # LoRA
#   bash runpod_quickstart.sh configs/phase1_dinov2.yaml    # open fallback (no gated access)
set -euo pipefail

export HF_HOME=${HF_HOME:-/workspace/hf_cache}
CONFIG=${1:-configs/phase1_frozen.yaml}
echo "==> config: $CONFIG"

python -m scripts.check_backbone --config "$CONFIG"
python -m src.download_ntire --shards 0
python -m src.prep_ntire --data-root data/ntire --out-dir data
python -m src.train --config "$CONFIG"

RUN=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['run_name'])")
echo "==> done. checkpoint: results/$RUN/best.pt  (+ last.pt)"
echo "    download it: python -m src.upload_model --ckpt results/$RUN/best.pt --repo <you>/nb-detector"
