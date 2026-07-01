#!/usr/bin/env bash
# Fit ALL 6 shards (~277K images) in a ~100GB volume by compacting to a 512px cache.
# Downloads -> compacts -> deletes-raw ONE shard at a time, so peak disk stays ~65GB.
# End result: data/compact/shard_0..5 (~30GB total) + manifests over it.
#   bash compact_all.sh            # 512px cache (default)
#   bash compact_all.sh 448        # smaller cache if you want more headroom
set -euo pipefail

export HF_HOME=${HF_HOME:-/workspace/hf_cache}
SIZE=${1:-512}

for i in 0 1 2 3 4 5; do
  echo "==> shard $i : download"
  python -m src.download_ntire --shards "$i" --out data/ntire
  echo "==> shard $i : compact to ${SIZE}px + delete raw"
  python -m src.compact_shard --shard "data/ntire/shard_$i" --out "data/compact/shard_$i" \
      --size "$SIZE" --delete-raw
done

echo "==> building manifests over data/compact"
python -m src.prep_ntire --data-root data/compact --out-dir data
echo "==> done. all 6 shards compacted; train on any config (manifests already point here)."
