from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from diffusers import DDIMScheduler, EulerDiscreteScheduler, StableDiffusionXLPipeline
from torchvision.utils import save_image
from tqdm.auto import tqdm

from visual_anagrams.latent_views import make_latent_hybrid_views


def get_pipeline_execution_device(pipeline: StableDiffusionXLPipeline) -> torch.device:
    execution_device = getattr(pipeline, "_execution_device", None)
    if execution_device is not None:
        return torch.device(execution_device)

    model_device = getattr(pipeline, "device", None)
    if model_device is not None:
        return torch.device(model_device)

    return torch.device("cpu")


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype `{dtype_name}`. Expected one of {sorted(mapping)}.") from exc


def load_sdxl_pipeline(
    model_path: str | Path,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    scheduler_name: str = "ddim",
    local_files_only: bool = True,
) -> StableDiffusionXLPipeline:
    model_path = Path(model_path)
    if local_files_only and not model_path.exists():
        raise FileNotFoundError(
            f"Could not find SDXL model at `{model_path}`. "
            "Download it first into `/data/models`, or pass `--allow_remote` with a Hugging Face repo id."
        )

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        use_safetensors=True,
        local_files_only=local_files_only,
    )

    scheduler_name = scheduler_name.lower()
    if scheduler_name == "ddim":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    elif scheduler_name == "euler":
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    else:
        raise ValueError(f"Unsupported scheduler `{scheduler_name}`.")

    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(device)

    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.to(dtype=torch.float32)

    return pipeline


def build_prompt(prompt: str, style: str = "") -> str:
    return f"{style} {prompt}".strip()


def ordered_prompts(prompt_close: str, prompt_far: str, style: str = "") -> list[str]:
    # Factorized diffusion conditions low frequencies first and high frequencies second.
    return [build_prompt(prompt_far, style=style), build_prompt(prompt_close, style=style)]


@dataclass
class SDXLPromptConditioning:
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    negative_pooled_prompt_embeds: torch.Tensor
    add_time_ids: torch.Tensor
    negative_add_time_ids: torch.Tensor

    @property
    def cfg_prompt_embeds(self) -> torch.Tensor:
        return torch.cat([self.negative_prompt_embeds, self.prompt_embeds], dim=0)

    @property
    def cfg_pooled_prompt_embeds(self) -> torch.Tensor:
        return torch.cat([self.negative_pooled_prompt_embeds, self.pooled_prompt_embeds], dim=0)

    @property
    def cfg_add_time_ids(self) -> torch.Tensor:
        return torch.cat([self.negative_add_time_ids, self.add_time_ids], dim=0)

    @property
    def num_prompts(self) -> int:
        return self.prompt_embeds.shape[0]


def prepare_sdxl_conditioning(
    pipeline: StableDiffusionXLPipeline,
    prompts: list[str],
    negative_prompt: str = "",
    height: int = 1024,
    width: int = 1024,
) -> SDXLPromptConditioning:
    device = get_pipeline_execution_device(pipeline)

    prompt_embeds = []
    negative_prompt_embeds = []
    pooled_prompt_embeds = []
    negative_pooled_prompt_embeds = []

    for prompt in prompts:
        encoded = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
        )
        prompt_embed, negative_embed, pooled_embed, negative_pooled_embed = encoded
        prompt_embeds.append(prompt_embed)
        negative_prompt_embeds.append(negative_embed)
        pooled_prompt_embeds.append(pooled_embed)
        negative_pooled_prompt_embeds.append(negative_pooled_embed)

    prompt_embeds = torch.cat(prompt_embeds, dim=0)
    negative_prompt_embeds = torch.cat(negative_prompt_embeds, dim=0)
    pooled_prompt_embeds = torch.cat(pooled_prompt_embeds, dim=0)
    negative_pooled_prompt_embeds = torch.cat(negative_pooled_prompt_embeds, dim=0)

    add_time_ids = pipeline._get_add_time_ids(
        original_size=(height, width),
        crops_coords_top_left=(0, 0),
        target_size=(height, width),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=pipeline.text_encoder_2.config.projection_dim,
    ).to(device)
    add_time_ids = add_time_ids.repeat(len(prompts), 1)
    negative_add_time_ids = add_time_ids.clone()

    return SDXLPromptConditioning(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        add_time_ids=add_time_ids,
        negative_add_time_ids=negative_add_time_ids,
    )


def reduce_noise_predictions(noise_pred: torch.Tensor, reduction: str, step_index: int) -> torch.Tensor:
    if reduction == "sum":
        return noise_pred.sum(dim=0, keepdim=True)
    if reduction == "mean":
        return noise_pred.mean(dim=0, keepdim=True)
    if reduction == "alternate":
        prompt_index = step_index % noise_pred.shape[0]
        return noise_pred[prompt_index : prompt_index + 1]
    raise ValueError("Reduction must be one of `sum`, `mean`, or `alternate`.")


def decode_latents(pipeline: StableDiffusionXLPipeline, latents: torch.Tensor) -> torch.Tensor:
    scaling_factor = pipeline.vae.config.scaling_factor
    decoded = pipeline.vae.decode(latents.to(dtype=pipeline.vae.dtype) / scaling_factor, return_dict=False)[0]
    return (decoded / 2 + 0.5).clamp(0, 1)


def simulate_far_view(images: torch.Tensor, resize_factor: float = 0.35, blur_sigma: float = 6.0) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected `images` to have shape [B, C, H, W], got {tuple(images.shape)}")

    height, width = images.shape[-2:]
    min_height = max(16, int(round(height * resize_factor)))
    min_width = max(16, int(round(width * resize_factor)))

    far_view = TF.resize(images, size=[min_height, min_width], antialias=True)
    far_view = TF.resize(far_view, size=[height, width], antialias=True)

    if blur_sigma > 0:
        kernel_size = max(3, int(blur_sigma * 6) | 1)
        far_view = TF.gaussian_blur(
            far_view,
            kernel_size=[kernel_size, kernel_size],
            sigma=[blur_sigma, blur_sigma],
        )

    return far_view.clamp(0, 1)


@dataclass
class LatentHybridSample:
    image: torch.Tensor
    far_view: torch.Tensor
    latents: torch.Tensor
    prompt_order: list[str]


def sample_latent_hybrid(
    pipeline: StableDiffusionXLPipeline,
    conditioning: SDXLPromptConditioning,
    *,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    reduction: str = "sum",
    latent_sigma: float = 1.5,
    latent_kernel_size: int = 9,
    far_resize_factor: float = 0.35,
    far_blur_sigma: float = 6.0,
    generator: torch.Generator | None = None,
    latents: torch.Tensor | None = None,
    train_step_indices: set[int] | None = None,
    detach_unet_input: bool = False,
    detach_untrained_predictions: bool = False,
    stop_after_step_index: int | None = None,
    show_progress: bool = True,
) -> LatentHybridSample:
    device = get_pipeline_execution_device(pipeline)
    views = make_latent_hybrid_views(latent_sigma=latent_sigma, latent_kernel_size=latent_kernel_size)

    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator=generator, eta=0.0)

    latent_height = height // pipeline.vae_scale_factor
    latent_width = width // pipeline.vae_scale_factor
    if latents is None:
        latents = pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=pipeline.unet.config.in_channels,
            height=height,
            width=width,
            dtype=conditioning.prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=None,
        )
    elif tuple(latents.shape[-2:]) != (latent_height, latent_width):
        raise ValueError(
            f"Expected latent size {(latent_height, latent_width)}, got {tuple(latents.shape[-2:])}."
        )

    train_step_indices = set(range(len(timesteps))) if train_step_indices is None else set(train_step_indices)
    progress_bar = tqdm(enumerate(timesteps), total=len(timesteps), disable=not show_progress)

    for step_index, timestep in progress_bar:
        viewed_latents = torch.stack([view.view(latents[0]) for view in views], dim=0)
        unet_input = viewed_latents.detach() if detach_unet_input else viewed_latents
        model_input = torch.cat([unet_input, unet_input], dim=0)
        model_input = pipeline.scheduler.scale_model_input(model_input, timestep)

        noise_pred = pipeline.unet(
            model_input,
            timestep,
            encoder_hidden_states=conditioning.cfg_prompt_embeds,
            added_cond_kwargs={
                "text_embeds": conditioning.cfg_pooled_prompt_embeds,
                "time_ids": conditioning.cfg_add_time_ids,
            },
            return_dict=False,
        )[0]

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_uncond = torch.stack(
            [view.inverse_view(pred) for pred, view in zip(noise_pred_uncond, views)],
            dim=0,
        )
        noise_pred_text = torch.stack(
            [view.inverse_view(pred) for pred, view in zip(noise_pred_text, views)],
            dim=0,
        )

        combined_noise = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        if detach_untrained_predictions and step_index not in train_step_indices:
            combined_noise = combined_noise.detach()

        reduced_noise = reduce_noise_predictions(combined_noise, reduction=reduction, step_index=step_index)
        step_output = pipeline.scheduler.step(
            reduced_noise,
            timestep,
            latents,
            return_dict=True,
            **extra_step_kwargs,
        )
        latents = step_output.prev_sample

        if stop_after_step_index is not None and step_index >= stop_after_step_index:
            pred_original_sample = getattr(step_output, "pred_original_sample", None)
            if pred_original_sample is None:
                raise RuntimeError("Early stop requested, but the scheduler did not expose `pred_original_sample`.")
            latents = pred_original_sample
            break

    image = decode_latents(pipeline, latents)
    far_view = simulate_far_view(image, resize_factor=far_resize_factor, blur_sigma=far_blur_sigma)
    return LatentHybridSample(
        image=image,
        far_view=far_view,
        latents=latents,
        prompt_order=["far", "close"],
    )


def make_generator(seed: int, device: str = "cuda") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def save_hybrid_sample(sample: LatentHybridSample, output_dir: str | Path, prefix: str = "sample") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_image(sample.image, output_dir / f"{prefix}.png")
    save_image(sample.far_view, output_dir / f"{prefix}.far.png")
