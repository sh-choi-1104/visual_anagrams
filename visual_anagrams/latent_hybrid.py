from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms.functional as TF

import visual_anagrams.transformers_compat  # noqa: F401
from diffusers import DDIMScheduler, EulerDiscreteScheduler, StableDiffusionXLPipeline
from torch.utils.checkpoint import checkpoint
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

    # The minimal local snapshot stores only `*.fp16.safetensors` files, so diffusers
    # must be told to load the `fp16` variant explicitly.
    variant = None
    fp16_only_unet = (model_path / "unet" / "diffusion_pytorch_model.fp16.safetensors").exists()
    default_unet = (model_path / "unet" / "diffusion_pytorch_model.safetensors").exists()
    if fp16_only_unet and not default_unet:
        variant = "fp16"

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        use_safetensors=True,
        local_files_only=local_files_only,
        variant=variant,
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
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()

    return pipeline


def build_prompt(prompt: str, style: str = "") -> str:
    return f"{style} {prompt}".strip()


def ordered_prompts(prompt_close: str, prompt_far: str, style: str = "") -> list[str]:
    # Factorized diffusion conditions low frequencies first and high frequencies second.
    return [build_prompt(prompt_far, style=style), build_prompt(prompt_close, style=style)]


def resolve_guidance_scales(
    *,
    guidance_scale: float | None = 7.5,
    guidance_scale_far: float | None = None,
    guidance_scale_close: float | None = None,
) -> tuple[float, float]:
    shared_scale = 7.5 if guidance_scale is None else guidance_scale
    far_scale = shared_scale if guidance_scale_far is None else guidance_scale_far
    close_scale = shared_scale if guidance_scale_close is None else guidance_scale_close
    return far_scale, close_scale


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

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> "SDXLPromptConditioning":
        def move(tensor: torch.Tensor) -> torch.Tensor:
            tensor_dtype = dtype if dtype is not None and torch.is_floating_point(tensor) else None
            return tensor.to(device=device, dtype=tensor_dtype)

        return SDXLPromptConditioning(
            prompt_embeds=move(self.prompt_embeds),
            negative_prompt_embeds=move(self.negative_prompt_embeds),
            pooled_prompt_embeds=move(self.pooled_prompt_embeds),
            negative_pooled_prompt_embeds=move(self.negative_pooled_prompt_embeds),
            add_time_ids=move(self.add_time_ids),
            negative_add_time_ids=move(self.negative_add_time_ids),
        )


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


def decode_latents_raw(pipeline: StableDiffusionXLPipeline, latents: torch.Tensor) -> torch.Tensor:
    scaling_factor = pipeline.vae.config.scaling_factor
    latents = latents.to(dtype=pipeline.vae.dtype) / scaling_factor

    def _decode(vae_latents: torch.Tensor) -> torch.Tensor:
        return pipeline.vae.decode(vae_latents, return_dict=False)[0]

    if torch.is_grad_enabled() and latents.requires_grad:
        return checkpoint(_decode, latents, use_reentrant=False)
    return _decode(latents)


def decode_latents(pipeline: StableDiffusionXLPipeline, latents: torch.Tensor) -> torch.Tensor:
    decoded = decode_latents_raw(pipeline, latents)
    return (decoded / 2 + 0.5).clamp(0, 1)


def encode_images_to_latents(pipeline: StableDiffusionXLPipeline, images: torch.Tensor) -> torch.Tensor:
    images = images.to(dtype=pipeline.vae.dtype)

    def _encode(input_images: torch.Tensor) -> torch.Tensor:
        return pipeline.vae.encode(input_images).latent_dist.mode()

    if torch.is_grad_enabled() and images.requires_grad:
        latents = checkpoint(_encode, images, use_reentrant=False)
    else:
        latents = _encode(images)

    return latents * pipeline.vae.config.scaling_factor


def gaussian_blur_images(images: torch.Tensor, sigma: float, kernel_size: int) -> torch.Tensor:
    if sigma <= 0:
        return images
    return TF.gaussian_blur(
        images,
        kernel_size=[kernel_size, kernel_size],
        sigma=[sigma, sigma],
    )


def compose_rgb_hybrid(
    far_image_raw: torch.Tensor,
    close_image_raw: torch.Tensor,
    *,
    sigma: float,
    kernel_size: int,
) -> torch.Tensor:
    low_pass_far = gaussian_blur_images(far_image_raw, sigma=sigma, kernel_size=kernel_size)
    low_pass_close = gaussian_blur_images(close_image_raw, sigma=sigma, kernel_size=kernel_size)
    high_pass_close = close_image_raw - low_pass_close
    return (low_pass_far + high_pass_close).clamp(-1, 1)


def predict_x0_from_noise(scheduler, *, sample: torch.Tensor, model_output: torch.Tensor, timestep) -> torch.Tensor:
    if isinstance(scheduler, DDIMScheduler):
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        if scheduler.config.prediction_type == "epsilon":
            return (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        if scheduler.config.prediction_type == "sample":
            return model_output
        if scheduler.config.prediction_type == "v_prediction":
            return alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        raise ValueError(f"Unsupported DDIM prediction type `{scheduler.config.prediction_type}`.")

    if isinstance(scheduler, EulerDiscreteScheduler):
        sigma = scheduler.sigmas[scheduler.step_index]
        if scheduler.config.prediction_type in {"sample", "original_sample"}:
            return model_output
        if scheduler.config.prediction_type == "epsilon":
            return sample - sigma * model_output
        if scheduler.config.prediction_type == "v_prediction":
            return model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        raise ValueError(f"Unsupported Euler prediction type `{scheduler.config.prediction_type}`.")

    raise ValueError(f"Unsupported scheduler `{type(scheduler).__name__}` for decoded-x0 composition.")


def noise_from_pred_x0(scheduler, *, sample: torch.Tensor, pred_original_sample: torch.Tensor, timestep) -> torch.Tensor:
    if isinstance(scheduler, DDIMScheduler):
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        if scheduler.config.prediction_type == "epsilon":
            return (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt()
        if scheduler.config.prediction_type == "sample":
            return pred_original_sample
        if scheduler.config.prediction_type == "v_prediction":
            return (alpha_prod_t.sqrt() * sample - pred_original_sample) / beta_prod_t.sqrt()
        raise ValueError(f"Unsupported DDIM prediction type `{scheduler.config.prediction_type}`.")

    if isinstance(scheduler, EulerDiscreteScheduler):
        sigma = scheduler.sigmas[scheduler.step_index]
        if scheduler.config.prediction_type in {"sample", "original_sample"}:
            return pred_original_sample
        if scheduler.config.prediction_type == "epsilon":
            return (sample - pred_original_sample) / sigma
        if scheduler.config.prediction_type == "v_prediction":
            return (pred_original_sample - sample / (sigma**2 + 1)) * (-(sigma**2 + 1) ** 0.5 / sigma)
        raise ValueError(f"Unsupported Euler prediction type `{scheduler.config.prediction_type}`.")

    raise ValueError(f"Unsupported scheduler `{type(scheduler).__name__}` for decoded-x0 composition.")


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
class StepPrediction:
    step_index: int
    denoising_progress: float
    image: torch.Tensor
    far_view: torch.Tensor


@dataclass
class LatentHybridSample:
    image: torch.Tensor
    far_view: torch.Tensor
    latents: torch.Tensor
    prompt_order: list[str]
    step_predictions: list[StepPrediction] | None = None


def sample_latent_hybrid(
    pipeline: StableDiffusionXLPipeline,
    conditioning: SDXLPromptConditioning,
    *,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    guidance_scale_far: float | None = None,
    guidance_scale_close: float | None = None,
    reduction: str = "sum",
    latent_sigma: float = 1.5,
    latent_kernel_size: int = 9,
    composition_space: str = "latent_eps",
    rgb_hybrid_sigma: float = 10.0,
    rgb_hybrid_kernel_size: int = 51,
    far_resize_factor: float = 0.35,
    far_blur_sigma: float = 6.0,
    generator: torch.Generator | None = None,
    latents: torch.Tensor | None = None,
    train_step_indices: set[int] | None = None,
    collect_step_predictions: bool = False,
    step_prediction_indices: set[int] | None = None,
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
    step_prediction_indices = train_step_indices if step_prediction_indices is None else set(step_prediction_indices)
    step_predictions = [] if collect_step_predictions else None
    progress_bar = tqdm(enumerate(timesteps), total=len(timesteps), disable=not show_progress)
    resolved_guidance_scale_far, resolved_guidance_scale_close = resolve_guidance_scales(
        guidance_scale=guidance_scale,
        guidance_scale_far=guidance_scale_far,
        guidance_scale_close=guidance_scale_close,
    )

    for step_index, timestep in progress_bar:
        if composition_space == "latent_eps":
            viewed_latents = torch.stack([view.view(latents[0]) for view in views], dim=0)
        elif composition_space == "decoded_x0_rgb":
            viewed_latents = latents.repeat(conditioning.num_prompts, 1, 1, 1)
        else:
            raise ValueError("`composition_space` must be `latent_eps` or `decoded_x0_rgb`.")

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
        guidance_scales = torch.tensor(
            [resolved_guidance_scale_far, resolved_guidance_scale_close],
            device=noise_pred_uncond.device,
            dtype=noise_pred_uncond.dtype,
        ).view(conditioning.num_prompts, 1, 1, 1)
        if composition_space == "latent_eps":
            noise_pred_uncond = torch.stack(
                [view.inverse_view(pred) for pred, view in zip(noise_pred_uncond, views)],
                dim=0,
            )
            noise_pred_text = torch.stack(
                [view.inverse_view(pred) for pred, view in zip(noise_pred_text, views)],
                dim=0,
            )
            combined_noise = noise_pred_uncond + guidance_scales * (noise_pred_text - noise_pred_uncond)
        else:
            cfg_noise = noise_pred_uncond + guidance_scales * (noise_pred_text - noise_pred_uncond)
            pred_x0_latents = torch.cat(
                [
                    predict_x0_from_noise(
                        pipeline.scheduler,
                        sample=latents,
                        model_output=cfg_noise[prompt_index : prompt_index + 1],
                        timestep=timestep,
                    )
                    for prompt_index in range(conditioning.num_prompts)
                ],
                dim=0,
            )
            pred_x0_images_raw = decode_latents_raw(pipeline, pred_x0_latents)
            target_x0_image = compose_rgb_hybrid(
                far_image_raw=pred_x0_images_raw[0:1],
                close_image_raw=pred_x0_images_raw[1:2],
                sigma=rgb_hybrid_sigma,
                kernel_size=rgb_hybrid_kernel_size,
            )
            target_x0_latent = encode_images_to_latents(pipeline, target_x0_image).to(dtype=latents.dtype)
            combined_noise = noise_from_pred_x0(
                pipeline.scheduler,
                sample=latents,
                pred_original_sample=target_x0_latent,
                timestep=timestep,
            ).to(dtype=latents.dtype)

        if detach_untrained_predictions and step_index not in train_step_indices:
            combined_noise = combined_noise.detach()

        if composition_space == "latent_eps":
            reduced_noise = reduce_noise_predictions(combined_noise, reduction=reduction, step_index=step_index)
        else:
            reduced_noise = combined_noise

        step_output = pipeline.scheduler.step(
            reduced_noise,
            timestep,
            latents,
            return_dict=True,
            **extra_step_kwargs,
        )
        if collect_step_predictions and step_index in step_prediction_indices:
            pred_original_sample = getattr(step_output, "pred_original_sample", None)
            if pred_original_sample is None:
                pred_original_sample = latents
            pred_image = decode_latents(pipeline, pred_original_sample)
            pred_far_view = simulate_far_view(
                pred_image,
                resize_factor=far_resize_factor,
                blur_sigma=far_blur_sigma,
            )
            step_predictions.append(
                StepPrediction(
                    step_index=step_index,
                    denoising_progress=step_index / max(len(timesteps) - 1, 1),
                    image=pred_image,
                    far_view=pred_far_view,
                )
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
        step_predictions=step_predictions,
    )


def make_generator(seed: int, device: str = "cuda") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def save_hybrid_sample(sample: LatentHybridSample, output_dir: str | Path, prefix: str = "sample") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_image(sample.image, output_dir / f"{prefix}.png")
    save_image(sample.far_view, output_dir / f"{prefix}.far.png")
