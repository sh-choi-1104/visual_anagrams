from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from visual_anagrams.latent_hybrid import (
    get_pipeline_execution_device,
    load_sdxl_pipeline,
    ordered_prompts,
    prepare_sdxl_conditioning,
    resolve_dtype,
    sample_latent_hybrid,
    save_hybrid_sample,
)
from visual_anagrams.reward import DEFAULT_HF_CACHE_DIR, configure_hf_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference script for the plain latent-space hybrid baseline, with optional reward-tuned comparison."
    )
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--save_dir", default="results_latent_inference", type=str)
    parser.add_argument("--prompt_close", required=True, type=str, help="Prompt visible from up close.")
    parser.add_argument("--prompt_far", required=True, type=str, help="Prompt visible from far away.")
    parser.add_argument("--style", default="", type=str)
    parser.add_argument("--negative_prompt", default="", type=str)
    parser.add_argument("--sdxl_model_path", default="/data/models/sdxl-base-1.0", type=str)
    parser.add_argument(
        "--compare_lora_path",
        default=None,
        type=str,
        help="Optional reward-tuned LoRA path. If provided, the script saves both baseline and tuned outputs from the same initial latents.",
    )
    parser.add_argument("--scheduler", default="ddim", choices=["ddim", "euler"])
    parser.add_argument("--height", default=1024, type=int)
    parser.add_argument("--width", default=1024, type=int)
    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--num_inference_steps", default=30, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--reduction", default="sum", choices=["sum", "mean", "alternate"])
    parser.add_argument("--latent_sigma", default=1.5, type=float)
    parser.add_argument("--latent_kernel_size", default=9, type=int)
    parser.add_argument("--far_resize_factor", default=0.35, type=float)
    parser.add_argument("--far_blur_sigma", default=6.0, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dtype", default="fp16", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--hf_cache_dir", default=str(DEFAULT_HF_CACHE_DIR), type=str)
    parser.add_argument("--allow_remote", action="store_true")
    return parser.parse_args()


def make_generator(seed: int, device: str) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def prepare_initial_latents(
    pipeline,
    *,
    seed: int,
    num_samples: int,
    height: int,
    width: int,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    device = get_pipeline_execution_device(pipeline)
    latents = []
    for sample_index in range(num_samples):
        generator = make_generator(seed + sample_index, str(device))
        initial_latents = pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=pipeline.unet.config.in_channels,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=None,
        )
        latents.append(initial_latents.detach().cpu())
    return latents


def save_comparison_grid(
    output_dir: Path,
    baseline_image: torch.Tensor,
    baseline_far: torch.Tensor,
    tuned_image: torch.Tensor,
    tuned_far: torch.Tensor,
) -> None:
    comparison = torch.cat([baseline_image, baseline_far, tuned_image, tuned_far], dim=0)
    save_image(comparison, output_dir / "comparison.png", nrow=2)


def run_pass(
    *,
    pipeline,
    conditioning,
    initial_latents: list[torch.Tensor],
    output_root: Path,
    prefix: str,
    args: argparse.Namespace,
) -> list[object]:
    samples = []
    for sample_index, initial_latents_cpu in enumerate(initial_latents):
        seed = args.seed + sample_index
        sample = sample_latent_hybrid(
            pipeline,
            conditioning,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            reduction=args.reduction,
            latent_sigma=args.latent_sigma,
            latent_kernel_size=args.latent_kernel_size,
            far_resize_factor=args.far_resize_factor,
            far_blur_sigma=args.far_blur_sigma,
            generator=make_generator(seed, args.device),
            latents=initial_latents_cpu.to(device=args.device, dtype=conditioning.prompt_embeds.dtype).clone(),
            show_progress=True,
        )
        sample_dir = output_root / f"{seed:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_hybrid_sample(sample, sample_dir, prefix=prefix)
        samples.append(sample)
    return samples


def main() -> None:
    args = parse_args()
    configure_hf_cache(args.hf_cache_dir)

    output_root = Path(args.save_dir) / args.name
    output_root.mkdir(parents=True, exist_ok=True)

    pipeline = load_sdxl_pipeline(
        model_path=args.sdxl_model_path,
        device=args.device,
        torch_dtype=resolve_dtype(args.dtype),
        scheduler_name=args.scheduler,
        local_files_only=not args.allow_remote,
    )
    prompt_list = ordered_prompts(
        prompt_close=args.prompt_close,
        prompt_far=args.prompt_far,
        style=args.style,
    )
    conditioning = prepare_sdxl_conditioning(
        pipeline,
        prompts=prompt_list,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
    )

    initial_latents = prepare_initial_latents(
        pipeline,
        seed=args.seed,
        num_samples=args.num_samples,
        height=args.height,
        width=args.width,
        dtype=conditioning.prompt_embeds.dtype,
    )

    metadata = {
        "prompt_close": args.prompt_close,
        "prompt_far": args.prompt_far,
        "style": args.style,
        "negative_prompt": args.negative_prompt,
        "ordered_prompts": prompt_list,
        "internal_prompt_order": ["far", "close"],
        "scheduler": args.scheduler,
        "height": args.height,
        "width": args.width,
        "num_samples": args.num_samples,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "reduction": args.reduction,
        "latent_sigma": args.latent_sigma,
        "latent_kernel_size": args.latent_kernel_size,
        "far_resize_factor": args.far_resize_factor,
        "far_blur_sigma": args.far_blur_sigma,
        "compare_lora_path": args.compare_lora_path,
        "modes": ["baseline"] if args.compare_lora_path is None else ["baseline", "tuned"],
    }
    with open(output_root / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    baseline_samples = run_pass(
        pipeline=pipeline,
        conditioning=conditioning,
        initial_latents=initial_latents,
        output_root=output_root,
        prefix="baseline",
        args=args,
    )

    if args.compare_lora_path is None:
        return

    pipeline.unet.load_attn_procs(args.compare_lora_path)
    tuned_samples = run_pass(
        pipeline=pipeline,
        conditioning=conditioning,
        initial_latents=initial_latents,
        output_root=output_root,
        prefix="tuned",
        args=args,
    )

    for sample_index, (baseline_sample, tuned_sample) in enumerate(zip(baseline_samples, tuned_samples)):
        seed = args.seed + sample_index
        sample_dir = output_root / f"{seed:04d}"
        save_comparison_grid(
            sample_dir,
            baseline_image=baseline_sample.image,
            baseline_far=baseline_sample.far_view,
            tuned_image=tuned_sample.image,
            tuned_far=tuned_sample.far_view,
        )


if __name__ == "__main__":
    main()
