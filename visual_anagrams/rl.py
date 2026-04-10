from __future__ import annotations

import random

import torch
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor


def create_unet_lora_layers(unet, rank: int = 16) -> AttnProcsLayers:
    lora_attn_procs = {}
    block_out_channels = list(unet.config.block_out_channels)
    reversed_block_out_channels = list(reversed(block_out_channels))

    for name in unet.attn_processors.keys():
        if name.startswith("mid_block"):
            hidden_size = block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = reversed_block_out_channels[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = block_out_channels[block_id]
        else:
            raise ValueError(f"Could not infer hidden size for attention processor `{name}`.")

        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)
    for parameter in lora_layers.parameters():
        parameter.requires_grad_(True)
    return lora_layers


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
