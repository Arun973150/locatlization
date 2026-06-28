"""Download NTIRE-2026 Robust-AIGen training shards (single-copy, storage-frugal).

Streams the zip straight to the target dir (NO HF cache duplicate), extracts, deletes the
zip immediately. Peak disk during a shard = ~2x shard size (zip + extracted) ≈ 40 GB, then
~20 GB resident. One shard (~50K imgs) holds all 20 training generators and is enough for
the baseline — so `--shards 0` fits a ~50 GB volume.

  python -m src.download_ntire --shards 0 --out data/ntire
"""
import argparse, os, sys, zipfile
import requests
from huggingface_hub import get_token

RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{fn}"


def stream_to_file(url, dest, token):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    part = dest + ".part"
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(part, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/1e9:5.1f} / {total/1e9:.1f} GB ({100*done/total:3.0f}%)",
                          end="", flush=True)
    print()
    os.replace(part, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="deepfakesMSU/NTIRE-RobustAIGenDetection-train")
    ap.add_argument("--shards", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="data/ntire")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    token = get_token()
    if token is None:
        print("WARN: no HF token (run `huggingface-cli login`; dataset is gated).", file=sys.stderr)

    for i in a.shards:
        fn = f"shard_{i}.zip"
        dest = os.path.join(a.out, f"shard_{i}")
        if os.path.isdir(dest) and os.listdir(dest):
            print(f"[skip] {dest} already extracted")
            continue
        zp = os.path.join(a.out, fn)
        print(f"[download] {fn}")
        stream_to_file(RESOLVE.format(repo=a.repo, fn=fn), zp, token)
        print(f"[unzip] -> {dest}")
        with zipfile.ZipFile(zp) as z:
            z.extractall(dest)
        os.remove(zp)                       # free ~20 GB immediately
        print(f"[ok] {dest} (zip removed)")
    print("[done] shards:", a.shards)


if __name__ == "__main__":
    main()
