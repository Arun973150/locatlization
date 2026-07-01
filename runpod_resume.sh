#!/usr/bin/env bash
# Run this ONCE after every pod start/restart. The network volume keeps your repo, data,
# model cache and checkpoints; only the container (pip deps + env) needs restoring.
#   bash runpod_resume.sh
set -e

cd /workspace/locatlization
export HF_HOME=/workspace/hf_cache

echo "== reinstalling deps (container was rebuilt) =="
pip install -q -r requirements.txt

echo "== HF auth =="
hf auth whoami 2>/dev/null || echo "  not logged in -> run: hf auth login"

echo "== what survived on the volume =="
echo "HF_HOME=$HF_HOME"
du -sh data/ntire/* 2>/dev/null || echo "  (no data/ntire yet)"
ls -1 results/*/best.pt 2>/dev/null || echo "  (no checkpoints yet)"
echo "== ready. (remember: export HF_HOME=/workspace/hf_cache in new shells) =="
