from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from visual_anagrams.reward import DEFAULT_HF_CACHE_DIR, HPSV2_VERSION_TO_FILENAME, configure_hf_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download minimal SDXL and HPSv2 assets into /data/models.")
    parser.add_argument("--model_root", default="/data/models", type=str)
    parser.add_argument("--hf_cache_dir", default=str(DEFAULT_HF_CACHE_DIR), type=str)
    parser.add_argument("--download_sdxl_minimal", action="store_true")
    parser.add_argument("--download_hpsv2_checkpoint", action="store_true")
    parser.add_argument("--hpsv2_version", default="v2.1", choices=["v2.0", "v2.1"])
    parser.add_argument("--clone_hpsv2_repo", action="store_true")
    return parser.parse_args()


def clone_or_update(repo_url: str, target_dir: Path) -> None:
    if target_dir.exists():
        subprocess.run(["git", "-C", str(target_dir), "pull", "--ff-only"], check=True)
        return

    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target_dir)], check=True)


def download_sdxl_minimal(model_root: Path) -> None:
    allow_patterns = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.fp16.safetensors",
        "tokenizer/*",
        "tokenizer_2/*",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ]
    snapshot_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        local_dir=str(model_root / "sdxl-base-1.0"),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )


def download_hpsv2_checkpoint(model_root: Path, version: str, cache_dir: Path) -> None:
    filename = HPSV2_VERSION_TO_FILENAME[version]
    target_dir = model_root / "HPSv2-weights"
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id="xswu/HPSv2",
        filename=filename,
        cache_dir=str(cache_dir),
    )
    shutil.copy2(downloaded_path, target_dir / filename)


def main() -> None:
    args = parse_args()
    model_root = Path(args.model_root)
    model_root.mkdir(parents=True, exist_ok=True)
    cache_dir = configure_hf_cache(args.hf_cache_dir)

    if args.download_sdxl_minimal:
        download_sdxl_minimal(model_root)

    if args.download_hpsv2_checkpoint:
        download_hpsv2_checkpoint(model_root, version=args.hpsv2_version, cache_dir=cache_dir)

    if args.clone_hpsv2_repo:
        clone_or_update("https://github.com/tgxs002/HPSv2", model_root / "HPSv2-repo")


if __name__ == "__main__":
    main()
