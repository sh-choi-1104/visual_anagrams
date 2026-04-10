from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from huggingface_hub import hf_hub_download
from torchvision.transforms import InterpolationMode


DEFAULT_MODEL_ROOT = Path("/data/models")
DEFAULT_HF_CACHE_DIR = DEFAULT_MODEL_ROOT / ".cache"
HPSV2_VERSION_TO_FILENAME = {
    "v2.0": "HPS_v2_compressed.pt",
    "v2.1": "HPS_v2.1_compressed.pt",
}


def configure_hf_cache(cache_dir: str | Path = DEFAULT_HF_CACHE_DIR) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    hub_dir = cache_dir / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hub_dir))
    return cache_dir


def import_hpsv2_open_clip(repo_path: str | Path):
    repo_path = Path(repo_path)
    package_root = repo_path / "hpsv2"
    src_root = package_root / "src"
    if not src_root.exists():
        raise RuntimeError(
            f"`{repo_path}` does not look like an HPSv2 checkout. Expected `{src_root}` to exist."
        )

    sys.path.insert(0, str(package_root))
    from src.open_clip import create_model_and_transforms, get_tokenizer  # type: ignore
    from src.open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD  # type: ignore

    return create_model_and_transforms, get_tokenizer, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


def center_crop_tensor(images: torch.Tensor, crop_height: int, crop_width: int) -> torch.Tensor:
    height, width = images.shape[-2:]
    top = max((height - crop_height) // 2, 0)
    left = max((width - crop_width) // 2, 0)
    return images[..., top : top + crop_height, left : left + crop_width]


def normalize_images(images: torch.Tensor, mean: tuple[float, float, float], std: tuple[float, float, float]) -> torch.Tensor:
    mean_tensor = images.new_tensor(mean).view(1, 3, 1, 1)
    std_tensor = images.new_tensor(std).view(1, 3, 1, 1)
    return (images - mean_tensor) / std_tensor


@dataclass
class HybridRewardOutput:
    close_scores: torch.Tensor
    far_scores: torch.Tensor
    total_scores: torch.Tensor


class HPSv2RewardModel:
    def __init__(
        self,
        *,
        device: str = "cuda",
        repo_path: str | Path = "/data/models/HPSv2-repo",
        checkpoint_path: str | Path | None = None,
        version: str = "v2.1",
        cache_dir: str | Path = DEFAULT_HF_CACHE_DIR,
    ) -> None:
        if version not in HPSV2_VERSION_TO_FILENAME:
            raise ValueError(f"Unsupported HPSv2 version `{version}`.")

        configure_hf_cache(cache_dir)
        create_model_and_transforms, get_tokenizer, image_mean, image_std = import_hpsv2_open_clip(repo_path)

        model, _, _ = create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )

        if checkpoint_path is None:
            checkpoint_path = hf_hub_download(
                repo_id="xswu/HPSv2",
                filename=HPSV2_VERSION_TO_FILENAME[version],
                cache_dir=str(cache_dir),
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.requires_grad_(False)

        self.model = model.to(device)
        self.device = device
        self.version = version
        self.image_mean = tuple(float(x) for x in image_mean)
        self.image_std = tuple(float(x) for x in image_std)
        self.tokenizer = get_tokenizer("ViT-H-14")

        image_size = model.visual.image_size
        if isinstance(image_size, tuple):
            self.image_size = (int(image_size[0]), int(image_size[1]))
        else:
            self.image_size = (int(image_size), int(image_size))

    def preprocess_tensor(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected `images` to have shape [B, C, H, W], got {tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError(f"Expected RGB images with 3 channels, got {images.shape[1]}")

        images = images.clamp(0, 1)
        images = TF.resize(
            images,
            size=list(self.image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        images = center_crop_tensor(images, crop_height=self.image_size[0], crop_width=self.image_size[1])
        return normalize_images(images, mean=self.image_mean, std=self.image_std)

    def score_images(self, prompts: list[str], images: torch.Tensor) -> torch.Tensor:
        processed_images = self.preprocess_tensor(images.to(device=self.device))
        text = self.tokenizer(prompts).to(device=self.device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=self.device.startswith("cuda")):
            outputs = self.model(processed_images, text)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            logits_per_image = image_features @ text_features.T

        return torch.diagonal(logits_per_image)

    def score_hybrid(
        self,
        *,
        prompt_close: str,
        prompt_far: str,
        image: torch.Tensor,
        far_view: torch.Tensor,
        close_weight: float = 1.0,
        far_weight: float = 1.0,
    ) -> HybridRewardOutput:
        batch_size = image.shape[0]
        prompts = [prompt_close] * batch_size + [prompt_far] * batch_size
        all_images = torch.cat([image, far_view], dim=0)
        scores = self.score_images(prompts=prompts, images=all_images)
        close_scores = scores[:batch_size]
        far_scores = scores[batch_size:]
        total_scores = close_weight * close_scores + far_weight * far_scores
        return HybridRewardOutput(
            close_scores=close_scores,
            far_scores=far_scores,
            total_scores=total_scores,
        )
