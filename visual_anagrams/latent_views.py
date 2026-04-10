from __future__ import annotations

from dataclasses import dataclass

import torch
import torchvision.transforms.functional as TF


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _scaled_kernel_size(base_kernel_size: int, spatial_size: int, base_spatial_size: int = 128) -> int:
    factor = max(spatial_size // base_spatial_size, 1)
    return _ensure_odd(base_kernel_size * factor)


def gaussian_blur_latents(latents: torch.Tensor, sigma: float, kernel_size: int) -> torch.Tensor:
    if sigma <= 0:
        return latents

    if latents.ndim != 3:
        raise ValueError(f"Expected `latents` to have shape [C, H, W], got {tuple(latents.shape)}")

    scaled_kernel_size = _scaled_kernel_size(kernel_size, latents.shape[-1])
    return TF.gaussian_blur(
        latents,
        kernel_size=[scaled_kernel_size, scaled_kernel_size],
        sigma=[sigma, sigma],
    )


@dataclass
class LatentHybridLowPassView:
    sigma: float = 1.5
    kernel_size: int = 9

    def view(self, latents: torch.Tensor) -> torch.Tensor:
        return latents

    def inverse_view(self, noise_pred: torch.Tensor) -> torch.Tensor:
        return gaussian_blur_latents(noise_pred, sigma=self.sigma, kernel_size=self.kernel_size)


@dataclass
class LatentHybridHighPassView:
    sigma: float = 1.5
    kernel_size: int = 9

    def view(self, latents: torch.Tensor) -> torch.Tensor:
        return latents

    def inverse_view(self, noise_pred: torch.Tensor) -> torch.Tensor:
        low_pass = gaussian_blur_latents(noise_pred, sigma=self.sigma, kernel_size=self.kernel_size)
        return noise_pred - low_pass


def make_latent_hybrid_views(latent_sigma: float, latent_kernel_size: int) -> list[object]:
    return [
        LatentHybridLowPassView(sigma=latent_sigma, kernel_size=latent_kernel_size),
        LatentHybridHighPassView(sigma=latent_sigma, kernel_size=latent_kernel_size),
    ]
