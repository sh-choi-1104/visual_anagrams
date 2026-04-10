from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

from visual_anagrams.reward import DEFAULT_HF_CACHE_DIR, configure_hf_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SDXL and HPSv3 assets into /data/models.")
    parser.add_argument("--model_root", default="/data/models", type=str)
    parser.add_argument("--hf_cache_dir", default=str(DEFAULT_HF_CACHE_DIR), type=str)
    parser.add_argument("--download_sdxl", action="store_true")
    parser.add_argument("--download_hpsv3_checkpoint", action="store_true")
    parser.add_argument("--download_qwen_backbone", action="store_true")
    parser.add_argument("--clone_hpsv3_repo", action="store_true")
    return parser.parse_args()


def clone_or_update(repo_url: str, target_dir: Path) -> None:
    if target_dir.exists():
        subprocess.run(["git", "-C", str(target_dir), "pull", "--ff-only"], check=True)
        return

    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target_dir)], check=True)


def main() -> None:
    args = parse_args()
    model_root = Path(args.model_root)
    model_root.mkdir(parents=True, exist_ok=True)
    configure_hf_cache(args.hf_cache_dir)

    if args.download_sdxl:
        snapshot_download(
            repo_id="stabilityai/stable-diffusion-xl-base-1.0",
            local_dir=str(model_root / "sdxl-base-1.0"),
            local_dir_use_symlinks=False,
        )

    if args.download_hpsv3_checkpoint:
        snapshot_download(
            repo_id="MizzenAI/HPSv3",
            repo_type="model",
            local_dir=str(model_root / "HPSv3-weights"),
            local_dir_use_symlinks=False,
        )

    if args.download_qwen_backbone:
        snapshot_download(
            repo_id="Qwen/Qwen2-VL-7B-Instruct",
            local_dir=str(model_root / "Qwen2-VL-7B-Instruct"),
            local_dir_use_symlinks=False,
        )

    if args.clone_hpsv3_repo:
        clone_or_update("https://github.com/MizzenAI/HPSv3", model_root / "HPSv3")


if __name__ == "__main__":
    main()
