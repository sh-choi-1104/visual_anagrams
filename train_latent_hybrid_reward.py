from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from visual_anagrams.latent_hybrid import (
    build_prompt,
    load_sdxl_pipeline,
    make_generator,
    ordered_prompts,
    prepare_sdxl_conditioning,
    resolve_dtype,
    sample_latent_hybrid,
    save_hybrid_sample,
)
from visual_anagrams.reward import DEFAULT_HF_CACHE_DIR, HPSv3RewardModel, configure_hf_cache
from visual_anagrams.rl import (
    create_unet_lora_layers,
    select_stop_after_step_index,
    select_train_step_indices,
    total_variation_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward-tune SDXL latent hybrids with HPSv3.")
    parser.add_argument("--output_dir", default="results_latent_reward", type=str)
    parser.add_argument("--prompt_close", default=None, type=str)
    parser.add_argument("--prompt_far", default=None, type=str)
    parser.add_argument("--style", default="", type=str)
    parser.add_argument("--negative_prompt", default="", type=str)
    parser.add_argument("--prompt_pairs_jsonl", default=None, type=str)
    parser.add_argument("--sdxl_model_path", default="/data/models/sdxl-base-1.0", type=str)
    parser.add_argument("--hpsv3_repo_path", default=None, type=str)
    parser.add_argument("--hpsv3_checkpoint_path", default=None, type=str)
    parser.add_argument("--hpsv3_config_path", default=None, type=str)
    parser.add_argument("--scheduler", default="ddim", choices=["ddim", "euler"])
    parser.add_argument("--height", default=1024, type=int)
    parser.add_argument("--width", default=1024, type=int)
    parser.add_argument("--num_inference_steps", default=30, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--reduction", default="sum", choices=["sum", "mean", "alternate"])
    parser.add_argument("--latent_sigma", default=1.5, type=float)
    parser.add_argument("--latent_kernel_size", default=9, type=int)
    parser.add_argument("--far_resize_factor", default=0.35, type=float)
    parser.add_argument("--far_blur_sigma", default=6.0, type=float)
    parser.add_argument("--algo", default="drtune", choices=["draft_k", "drtune"])
    parser.add_argument("--reward_train_steps", default=5, type=int)
    parser.add_argument("--early_stop_max_steps", default=0, type=int)
    parser.add_argument("--max_iterations", default=200, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--lora_rank", default=16, type=int)
    parser.add_argument("--close_reward_weight", default=1.0, type=float)
    parser.add_argument("--far_reward_weight", default=1.0, type=float)
    parser.add_argument("--tv_weight", default=0.0, type=float)
    parser.add_argument("--save_every", default=25, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dtype", default="fp16", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--hf_cache_dir", default=str(DEFAULT_HF_CACHE_DIR), type=str)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--allow_remote", action="store_true")
    return parser.parse_args()


def load_prompt_pairs(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.prompt_pairs_jsonl is not None:
        pairs = []
        with open(args.prompt_pairs_jsonl, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                pairs.append(
                    {
                        "prompt_close": entry["prompt_close"],
                        "prompt_far": entry["prompt_far"],
                        "style": entry.get("style", args.style),
                        "negative_prompt": entry.get("negative_prompt", args.negative_prompt),
                    }
                )
        if not pairs:
            raise ValueError(f"No prompt pairs were found in `{args.prompt_pairs_jsonl}`.")
        return pairs

    if args.prompt_close is None or args.prompt_far is None:
        raise ValueError("Provide either `--prompt_close/--prompt_far` or `--prompt_pairs_jsonl`.")

    return [
        {
            "prompt_close": args.prompt_close,
            "prompt_far": args.prompt_far,
            "style": args.style,
            "negative_prompt": args.negative_prompt,
        }
    ]


def conditioning_key(pair: dict[str, str], height: int, width: int) -> tuple[str, str, str, str, int, int]:
    return (
        pair["prompt_close"],
        pair["prompt_far"],
        pair.get("style", ""),
        pair.get("negative_prompt", ""),
        height,
        width,
    )


def main() -> None:
    args = parse_args()
    configure_hf_cache(args.hf_cache_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)

    prompt_pairs = load_prompt_pairs(args)
    torch_dtype = resolve_dtype(args.dtype)
    pipeline = load_sdxl_pipeline(
        model_path=args.sdxl_model_path,
        device=args.device,
        torch_dtype=torch_dtype,
        scheduler_name=args.scheduler,
        local_files_only=not args.allow_remote,
    )
    if args.algo == "drtune" and args.early_stop_max_steps > 0 and args.scheduler != "ddim":
        raise ValueError("`--early_stop_max_steps` currently requires `--scheduler ddim`.")

    if args.gradient_checkpointing:
        pipeline.unet.enable_gradient_checkpointing()

    lora_layers = create_unet_lora_layers(pipeline.unet, rank=args.lora_rank)
    pipeline.unet.train()

    optimizer = torch.optim.AdamW(
        lora_layers.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    reward_model = HPSv3RewardModel(
        device=args.device,
        repo_path=args.hpsv3_repo_path,
        checkpoint_path=args.hpsv3_checkpoint_path,
        config_path=args.hpsv3_config_path,
        cache_dir=args.hf_cache_dir,
    )

    rng = random.Random(args.seed)
    conditioning_cache: dict[tuple[str, str, str, str, int, int], object] = {}
    metrics_path = output_dir / "metrics.jsonl"

    best_reward = None
    for iteration in range(args.max_iterations):
        pair = prompt_pairs[iteration % len(prompt_pairs)]
        pair_style = pair.get("style", "")
        pair_negative_prompt = pair.get("negative_prompt", "")

        key = conditioning_key(pair, args.height, args.width)
        if key not in conditioning_cache:
            conditioning_cache[key] = prepare_sdxl_conditioning(
                pipeline,
                prompts=ordered_prompts(
                    prompt_close=pair["prompt_close"],
                    prompt_far=pair["prompt_far"],
                    style=pair_style,
                ),
                negative_prompt=pair_negative_prompt,
                height=args.height,
                width=args.width,
            )
        conditioning = conditioning_cache[key]

        train_step_indices = select_train_step_indices(
            total_steps=args.num_inference_steps,
            num_train_steps=args.reward_train_steps,
            strategy=args.algo,
            rng=rng,
        )
        stop_after_step_index = None
        if args.algo == "drtune":
            stop_after_step_index = select_stop_after_step_index(
                total_steps=args.num_inference_steps,
                early_stop_max_steps=args.early_stop_max_steps,
                rng=rng,
            )

        optimizer.zero_grad(set_to_none=True)
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
            generator=make_generator(seed=args.seed + iteration, device=args.device),
            train_step_indices=train_step_indices,
            detach_unet_input=args.algo == "drtune",
            detach_untrained_predictions=True,
            stop_after_step_index=stop_after_step_index,
            show_progress=False,
        )

        styled_prompt_close = build_prompt(pair["prompt_close"], pair_style)
        styled_prompt_far = build_prompt(pair["prompt_far"], pair_style)
        reward_output = reward_model.score_hybrid(
            prompt_close=styled_prompt_close,
            prompt_far=styled_prompt_far,
            image=sample.image,
            far_view=sample.far_view,
            close_weight=args.close_reward_weight,
            far_weight=args.far_reward_weight,
        )

        reward_loss = -reward_output.total_scores.mean()
        tv_loss = args.tv_weight * total_variation_loss(sample.image)
        loss = reward_loss + tv_loss
        loss.backward()

        grad_norm = clip_grad_norm_(lora_layers.parameters(), args.max_grad_norm)
        optimizer.step()

        metrics = {
            "iteration": iteration,
            "prompt_close": pair["prompt_close"],
            "prompt_far": pair["prompt_far"],
            "close_reward": float(reward_output.close_scores.mean().detach().cpu()),
            "far_reward": float(reward_output.far_scores.mean().detach().cpu()),
            "total_reward": float(reward_output.total_scores.mean().detach().cpu()),
            "reward_loss": float(reward_loss.detach().cpu()),
            "tv_loss": float(tv_loss.detach().cpu()),
            "loss": float(loss.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
            "train_step_indices": sorted(train_step_indices),
            "stop_after_step_index": stop_after_step_index,
        }
        with open(metrics_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        should_save = (iteration + 1) % args.save_every == 0 or iteration == args.max_iterations - 1
        current_reward = metrics["total_reward"]
        if best_reward is None or current_reward > best_reward:
            best_reward = current_reward
            pipeline.unet.save_attn_procs(output_dir / "lora_best")

        if should_save:
            with torch.no_grad():
                preview = sample_latent_hybrid(
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
                    generator=make_generator(seed=args.seed + 100000 + iteration, device=args.device),
                    show_progress=False,
                )
            preview_dir = output_dir / "previews" / f"{iteration:05d}"
            save_hybrid_sample(preview, preview_dir)
            pipeline.unet.save_attn_procs(output_dir / "lora_latest")


if __name__ == "__main__":
    main()
