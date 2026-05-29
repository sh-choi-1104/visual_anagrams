from __future__ import annotations

import math
import random

import torch

import visual_anagrams.transformers_compat  # noqa: F401
from peft import LoraConfig


def create_unet_lora_layers(
    unet,
    rank: int = 16,
    alpha: int | None = None,
    adapter_name: str = "default",
) -> torch.nn.ParameterList:
    if rank <= 0:
        raise ValueError(f"`rank` must be positive, got {rank}.")

    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha or rank,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        init_lora_weights="gaussian",
    )
    unet.add_adapter(lora_config, adapter_name=adapter_name)

    trainable_parameters = torch.nn.ParameterList(
        [parameter for parameter in unet.parameters() if parameter.requires_grad]
    )
    if len(trainable_parameters) == 0:
        raise RuntimeError("No trainable LoRA parameters were created on the UNet.")

    return trainable_parameters


def select_train_step_indices(
    total_steps: int,
    num_train_steps: int,
    strategy: str,
    rng: random.Random,
) -> set[int]:
    if num_train_steps <= 0:
        return set()
    if num_train_steps >= total_steps:
        return set(range(total_steps))

    strategy = strategy.lower()
    if strategy == "draft_k":
        start = total_steps - num_train_steps
        return set(range(start, total_steps))

    if strategy == "drtune":
        stride = max(total_steps // num_train_steps, 1)
        max_offset = max(total_steps - stride * (num_train_steps - 1) - 1, 0)
        offset = rng.randint(0, max_offset)
        indices = {min(offset + stride * i, total_steps - 1) for i in range(num_train_steps)}
        return indices

    raise ValueError(f"Unsupported strategy `{strategy}`.")


def select_stop_after_step_index(
    total_steps: int,
    early_stop_max_steps: int,
    rng: random.Random,
) -> int | None:
    if early_stop_max_steps <= 0:
        return None

    early_stop_steps = rng.randint(1, min(total_steps, early_stop_max_steps))
    return total_steps - early_stop_steps


def total_variation_loss(images: torch.Tensor) -> torch.Tensor:
    diff_h = images[:, :, 1:, :] - images[:, :, :-1, :]
    diff_w = images[:, :, :, 1:] - images[:, :, :, :-1]
    return diff_h.abs().mean() + diff_w.abs().mean()


def interpolate_weight(start: float, end: float, progress: float, schedule: str = "linear", exponent: float = 1.0) -> float:
    progress = min(max(progress, 0.0), 1.0)
    if schedule == "linear":
        scaled_progress = progress
    elif schedule == "cosine":
        scaled_progress = 0.5 - 0.5 * math.cos(progress * math.pi)
    else:
        raise ValueError(f"Unsupported schedule `{schedule}`.")

    scaled_progress = scaled_progress**exponent
    return start + (end - start) * scaled_progress


def compute_hybrid_reward_weights(
    *,
    denoising_progress: float,
    close_base_weight: float,
    far_base_weight: float,
    min_weight_scale: float = 0.25,
    schedule: str = "linear",
    exponent: float = 1.0,
    far_early_boost: float = 0.0,
    far_early_boost_exponent: float = 1.0,
) -> tuple[float, float]:
    min_weight_scale = min(max(min_weight_scale, 0.0), 1.0)
    far_early_boost = max(far_early_boost, 0.0)
    far_early_boost_exponent = max(far_early_boost_exponent, 1e-6)
    close_scale = interpolate_weight(
        start=min_weight_scale,
        end=1.0,
        progress=denoising_progress,
        schedule=schedule,
        exponent=exponent,
    )
    far_scale = interpolate_weight(
        start=1.0,
        end=min_weight_scale,
        progress=denoising_progress,
        schedule=schedule,
        exponent=exponent,
    )
    far_early_scale = 1.0 + far_early_boost * ((1.0 - min(max(denoising_progress, 0.0), 1.0)) ** far_early_boost_exponent)
    return close_base_weight * close_scale, far_base_weight * far_scale * far_early_scale
