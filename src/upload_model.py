"""Push a trained checkpoint to a (private) HF model repo so you can download it anywhere.

  hf auth login                               # once
  python -m src.upload_model --ckpt results/phase1_frozen/best.pt --repo <you>/nb-detector

Later, from any machine:
  hf download <you>/nb-detector best.pt --local-dir .
"""
import argparse, os
from huggingface_hub import HfApi, create_repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True, help="e.g. yourname/nb-detector")
    ap.add_argument("--public", action="store_true", help="make the repo public (default private)")
    a = ap.parse_args()

    create_repo(a.repo, repo_type="model", private=not a.public, exist_ok=True)
    api = HfApi()
    api.upload_file(path_or_fileobj=a.ckpt, path_in_repo=os.path.basename(a.ckpt),
                    repo_id=a.repo, repo_type="model")
    log = os.path.join(os.path.dirname(a.ckpt), "log.csv")
    if os.path.exists(log):
        api.upload_file(path_or_fileobj=log, path_in_repo="log.csv",
                        repo_id=a.repo, repo_type="model")
    print(f"[ok] uploaded -> https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
