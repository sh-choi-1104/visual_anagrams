from __future__ import annotations

import argparse
import json
from pathlib import Path

from visual_anagrams.latent_hybrid import (
    load_sdxl_pipeline,
    make_generator,
    ordered_prompts,
    prepare_sdxl_conditioning,
    resolve_dtype,
    sample_latent_hybrid,
    save_hybrid_sample,
)
from visual_anagrams.reward import DEFAULT_HF_CACHE_DIR, configure_hf_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate latent-space hybrid images with SDXL.")
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--save_dir", default="results_latent", type=str)
    parser.add_argument("--prompt_close", required=True, type=str, help="Prompt visible from up close.")
    parser.add_argument("--prompt_far", required=True, type=str, help="Prompt visible from far away.")
    parser.add_argument("--style", default="", type=str)
    parser.add_argument("--negative_prompt", default="", type=str)
    parser.add_argument("--sdxl_model_path", default="/data/models/sdxl-base-1.0", type=str)
    parser.add_argument("--lora_path", default=None, type=str, help="Optional reward-tuned LoRA directory.")
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
    parser.add_argument("--allow_remote", action="store_true", help="Allow Hugging Face download if local path is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_cache(args.hf_cache_dir)

    output_root = Path(args.save_dir) / args.name
    output_root.mkdir(parents=True, exist_ok=True)

    torch_dtype = resolve_dtype(args.dtype)
    pipeline = load_sdxl_pipeline(
        model_path=args.sdxl_model_path,
        device=args.device,
        torch_dtype=torch_dtype,
        scheduler_name=args.scheduler,
        local_files_only=not args.allow_remote,
    )

    if args.lora_path is not None:
        pipeline.unet.load_attn_procs(args.lora_path)

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

    metadata = {
        "prompt_close": args.prompt_close,
        "prompt_far": args.prompt_far,
        "style": args.style,
        "negative_prompt": args.negative_prompt,
        "internal_prompt_order": ["far", "close"],
        "ordered_prompts": prompt_list,
        "scheduler": args.scheduler,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "latent_sigma": args.latent_sigma,
        "latent_kernel_size": args.latent_kernel_size,
        "far_resize_factor": args.far_resize_factor,
        "far_blur_sigma": args.far_blur_sigma,
    }
    with open(output_root / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    for sample_index in range(args.num_samples):
        seed = args.seed + sample_index
        generator = make_generator(seed=seed, device=args.device)
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
            generator=generator,
            show_progress=True,
        )
        save_hybrid_sample(sample, output_root / f"{seed:04d}")


if __name__ == "__main__":
    main()
