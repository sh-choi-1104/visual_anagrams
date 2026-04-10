from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


DEFAULT_MODEL_ROOT = Path("/data/models")
DEFAULT_HF_CACHE_DIR = DEFAULT_MODEL_ROOT / ".cache"


def configure_hf_cache(cache_dir: str | Path = DEFAULT_HF_CACHE_DIR) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    hub_dir = cache_dir / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hub_dir))
    return cache_dir


def import_hpsv3(repo_path: str | Path | None = None):
    try:
        from hpsv3 import HPSv3RewardInferencer  # type: ignore

        return HPSv3RewardInferencer
    except ImportError:
        if repo_path is None:
            raise RuntimeError(
                "Could not import `hpsv3`. Install the package or clone the official repo and pass "
                "`--hpsv3_repo_path /data/models/HPSv3`."
            )

        repo_path = Path(repo_path)
        if not (repo_path / "hpsv3").exists():
            raise RuntimeError(
                f"`{repo_path}` does not look like an HPSv3 checkout. Expected `{repo_path / 'hpsv3'}` to exist."
            )

        sys.path.insert(0, str(repo_path))
        from hpsv3 import HPSv3RewardInferencer  # type: ignore

        return HPSv3RewardInferencer


@dataclass
class HybridRewardOutput:
    close_scores: torch.Tensor
    far_scores: torch.Tensor
    total_scores: torch.Tensor


class HPSv3RewardModel:
    def __init__(
        self,
        *,
        device: str = "cuda",
        repo_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
        cache_dir: str | Path = DEFAULT_HF_CACHE_DIR,
    ) -> None:
        configure_hf_cache(cache_dir)
        inferencer_cls = import_hpsv3(repo_path=repo_path)
        self.inferencer = inferencer_cls(
            config_path=None if config_path is None else str(config_path),
            checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
            device=device,
            differentiable=True,
        )
        self.device = device
        self.inferencer.model.eval()
        for parameter in self.inferencer.model.parameters():
            parameter.requires_grad_(False)

    def score_images(self, prompts: list[str], images: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            image_inputs = [image for image in images]
        else:
            image_inputs = images

        rewards = self.inferencer.reward(prompts=prompts, image_paths=image_inputs)
        if rewards.ndim != 2 or rewards.shape[1] < 1:
            raise RuntimeError(f"Unexpected HPSv3 reward shape: {tuple(rewards.shape)}")
        return rewards[:, 0]

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
