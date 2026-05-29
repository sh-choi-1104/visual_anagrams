from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
import json
import os
import random
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import save_image

from visual_anagrams.latent_hybrid import (
    build_prompt,
    load_sdxl_pipeline,
    make_generator,
    ordered_prompts,
    prepare_sdxl_conditioning,
    resolve_guidance_scales,
    resolve_dtype,
    sample_latent_hybrid,
    save_hybrid_sample,
)
from visual_anagrams.reward import DEFAULT_HF_CACHE_DIR, HPSv2RewardModel, configure_hf_cache
from visual_anagrams.rl import (
    compute_hybrid_reward_weights,
    create_unet_lora_layers,
    select_stop_after_step_index,
    select_train_step_indices,
    total_variation_loss,
)

DEFAULT_OUTPUT_ROOT = Path("/data/models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward-tune SDXL latent hybrids with HPSv2.")
    parser.add_argument("--output_dir", default="latent_hybrid_reward", type=str)
    parser.add_argument("--prompt_close", default=None, type=str)
    parser.add_argument("--prompt_far", default=None, type=str)
    parser.add_argument("--style", default="", type=str)
    parser.add_argument("--negative_prompt", default="", type=str)
    parser.add_argument("--prompt_pairs_jsonl", default=None, type=str)
    parser.add_argument("--sdxl_model_path", default="/data/models/sdxl-base-1.0", type=str)
    parser.add_argument("--hpsv2_repo_path", default="/data/models/HPSv2-repo", type=str)
    parser.add_argument("--hpsv2_checkpoint_path", default=None, type=str)
    parser.add_argument("--hpsv2_version", default="v2.1", choices=["v2.0", "v2.1"])
    parser.add_argument("--scheduler", default="ddim", choices=["ddim", "euler"])
    parser.add_argument("--height", default=1024, type=int)
    parser.add_argument("--width", default=1024, type=int)
    parser.add_argument("--num_inference_steps", default=30, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--guidance_scale_far", default=None, type=float)
    parser.add_argument("--guidance_scale_close", default=None, type=float)
    parser.add_argument("--reduction", default="sum", choices=["sum", "mean", "alternate"])
    parser.add_argument("--composition_space", default="decoded_x0_rgb", choices=["latent_eps", "decoded_x0_rgb"])
    parser.add_argument("--latent_sigma", default=1.5, type=float)
    parser.add_argument("--latent_kernel_size", default=9, type=int)
    parser.add_argument("--rgb_hybrid_sigma", default=10.0, type=float)
    parser.add_argument("--rgb_hybrid_kernel_size", default=51, type=int)
    parser.add_argument("--far_resize_factor", default=0.35, type=float)
    parser.add_argument("--far_blur_sigma", default=6.0, type=float)
    parser.add_argument("--algo", default="drtune", choices=["draft_k", "drtune"])
    parser.add_argument("--reward_train_steps", default=5, type=int)
    parser.add_argument("--early_stop_max_steps", default=0, type=int)
    parser.add_argument("--max_iterations", default=200, type=int)
    parser.add_argument(
        "--conditioning_cache_size",
        default=0,
        type=int,
        help="Number of prompt-conditionings to cache on CPU. Use 0 to disable caching and minimize VRAM use.",
    )
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--lora_rank", default=16, type=int)
    parser.add_argument("--close_reward_weight", default=1.0, type=float)
    parser.add_argument("--far_reward_weight", default=1.0, type=float)
    parser.add_argument("--step_reward_min_weight_scale", default=0.25, type=float)
    parser.add_argument("--step_reward_schedule", default="linear", choices=["linear", "cosine"])
    parser.add_argument("--step_reward_exponent", default=1.0, type=float)
    parser.add_argument(
        "--far_reward_early_boost",
        default=0.75,
        type=float,
        help="Extra multiplicative boost applied to far-view rewards early in denoising.",
    )
    parser.add_argument(
        "--far_reward_early_boost_exponent",
        default=1.0,
        type=float,
        help="Controls how quickly the early far-view reward boost decays across denoising steps.",
    )
    parser.add_argument("--tv_weight", default=0.0, type=float)
    parser.add_argument("--save_every", default=25, type=int)
    parser.add_argument("--preview_every", default=None, type=int)
    parser.add_argument("--preview_pair_index", default=0, type=int)
    parser.add_argument("--preview_seed", default=2026, type=int)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dtype", default="fp16", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--hf_cache_dir", default=str(DEFAULT_HF_CACHE_DIR), type=str)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--wandb_mode",
        default="auto",
        choices=["auto", "online", "offline", "disabled"],
        help="Use `auto` to go online when `WANDB_API_KEY` is set, otherwise log offline.",
    )
    parser.add_argument("--wandb_project", default="visual-anagrams-latent-hybrid", type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--wandb_run_name", default=None, type=str)
    parser.add_argument("--allow_remote", action="store_true")
    return parser.parse_args()


def resolve_output_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = DEFAULT_OUTPUT_ROOT / output_dir

    try:
        output_dir.relative_to(DEFAULT_OUTPUT_ROOT)
    except ValueError as exc:
        raise ValueError(f"`output_dir` must live under `{DEFAULT_OUTPUT_ROOT}`. Got `{output_dir}`.") from exc

    return output_dir


def resolve_wandb_mode(requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    netrc_path = Path("~/.netrc").expanduser()
    if netrc_path.exists() and "machine api.wandb.ai" in netrc_path.read_text():
        return "online"
    return "offline"


def init_wandb(args: argparse.Namespace, *, output_dir: Path):
    resolved_mode = resolve_wandb_mode(args.wandb_mode)
    if resolved_mode == "disabled":
        return None, resolved_mode

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or output_dir.name,
        dir=str(output_dir / "wandb"),
        mode=resolved_mode,
        config={**vars(args), "resolved_output_dir": str(output_dir)},
    )
    return run, resolved_mode


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
                        "pair_id": entry.get("pair_id"),
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
            "pair_id": "manual_00000",
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


def make_step_reward_schedule_preview(args: argparse.Namespace) -> list[dict[str, float]]:
    preview = []
    for step_index in range(args.num_inference_steps):
        progress = step_index / max(args.num_inference_steps - 1, 1)
        close_weight, far_weight = compute_hybrid_reward_weights(
            denoising_progress=progress,
            close_base_weight=args.close_reward_weight,
            far_base_weight=args.far_reward_weight,
            min_weight_scale=args.step_reward_min_weight_scale,
            schedule=args.step_reward_schedule,
            exponent=args.step_reward_exponent,
            far_early_boost=args.far_reward_early_boost,
            far_early_boost_exponent=args.far_reward_early_boost_exponent,
        )
        preview.append(
            {
                "step_index": step_index,
                "denoising_progress": round(progress, 4),
                "close_weight": round(close_weight, 6),
                "far_weight": round(far_weight, 6),
            }
        )
    return preview


@contextmanager
def temporarily_disable_adapters(module):
    disable_adapters = getattr(module, "disable_adapters", None)
    enable_adapters = getattr(module, "enable_adapters", None)
    if disable_adapters is None or enable_adapters is None:
        yield False
        return

    disable_adapters()
    try:
        yield True
    finally:
        enable_adapters()


def save_preview_comparison(*, baseline_sample, tuned_sample, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    close_grid = torch.cat([baseline_sample.image.detach().cpu(), tuned_sample.image.detach().cpu()], dim=0)
    far_grid = torch.cat([baseline_sample.far_view.detach().cpu(), tuned_sample.far_view.detach().cpu()], dim=0)
    full_grid = torch.cat(
        [
            baseline_sample.image.detach().cpu(),
            baseline_sample.far_view.detach().cpu(),
            tuned_sample.image.detach().cpu(),
            tuned_sample.far_view.detach().cpu(),
        ],
        dim=0,
    )
    save_image(close_grid, output_dir / "comparison.close.png", nrow=2)
    save_image(far_grid, output_dir / "comparison.far.png", nrow=2)
    save_image(full_grid, output_dir / "comparison.png", nrow=2)


def main() -> None:
    args = parse_args()
    if args.preview_every is None:
        args.preview_every = args.save_every
    configure_hf_cache(args.hf_cache_dir)

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run, wandb_mode = init_wandb(args, output_dir=output_dir)
    resolved_guidance_scale_far, resolved_guidance_scale_close = resolve_guidance_scales(
        guidance_scale=args.guidance_scale,
        guidance_scale_far=args.guidance_scale_far,
        guidance_scale_close=args.guidance_scale_close,
    )
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                **vars(args),
                "resolved_output_dir": str(output_dir),
                "wandb_mode_resolved": wandb_mode,
                "resolved_guidance_scale_far": resolved_guidance_scale_far,
                "resolved_guidance_scale_close": resolved_guidance_scale_close,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    with open(output_dir / "reward_schedule_preview.json", "w", encoding="utf-8") as file:
        json.dump(make_step_reward_schedule_preview(args), file, ensure_ascii=False, indent=2)

    prompt_pairs = load_prompt_pairs(args)
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "resolved_guidance_scale_far": resolved_guidance_scale_far,
                "resolved_guidance_scale_close": resolved_guidance_scale_close,
            },
            allow_val_change=True,
        )
    preview_pair = prompt_pairs[args.preview_pair_index % len(prompt_pairs)]
    with open(output_dir / "preview_pair.json", "w", encoding="utf-8") as file:
        json.dump(preview_pair, file, ensure_ascii=False, indent=2)
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "preview_pair_id": preview_pair.get("pair_id"),
                "preview_prompt_close": preview_pair["prompt_close"],
                "preview_prompt_far": preview_pair["prompt_far"],
                "preview_style": preview_pair.get("style", ""),
                "preview_seed": args.preview_seed,
            },
            allow_val_change=True,
        )
    torch_dtype = resolve_dtype(args.dtype)
    try:
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
        reward_model = HPSv2RewardModel(
            device=args.device,
            repo_path=args.hpsv2_repo_path,
            checkpoint_path=args.hpsv2_checkpoint_path,
            version=args.hpsv2_version,
            cache_dir=args.hf_cache_dir,
        )

        rng = random.Random(args.seed)
        conditioning_cache: OrderedDict[tuple[str, str, str, str, int, int], object] = OrderedDict()
        metrics_path = output_dir / "metrics.jsonl"
        pair_order = list(prompt_pairs)
        rng.shuffle(pair_order)

        best_reward = None
        for iteration in range(args.max_iterations):
            if iteration > 0 and iteration % len(pair_order) == 0:
                rng.shuffle(pair_order)
            pair = pair_order[iteration % len(pair_order)]
            pair_style = pair.get("style", "")
            pair_negative_prompt = pair.get("negative_prompt", "")

            if args.conditioning_cache_size > 0:
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
                    ).to(device="cpu")
                    if len(conditioning_cache) > args.conditioning_cache_size:
                        conditioning_cache.popitem(last=False)
                else:
                    conditioning_cache.move_to_end(key)
                conditioning = conditioning_cache[key].to(device=args.device)
            else:
                conditioning = prepare_sdxl_conditioning(
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
                guidance_scale_far=args.guidance_scale_far,
                guidance_scale_close=args.guidance_scale_close,
                reduction=args.reduction,
                latent_sigma=args.latent_sigma,
                latent_kernel_size=args.latent_kernel_size,
                composition_space=args.composition_space,
                rgb_hybrid_sigma=args.rgb_hybrid_sigma,
                rgb_hybrid_kernel_size=args.rgb_hybrid_kernel_size,
                far_resize_factor=args.far_resize_factor,
                far_blur_sigma=args.far_blur_sigma,
                generator=make_generator(seed=args.seed + iteration, device=args.device),
                train_step_indices=train_step_indices,
                collect_step_predictions=True,
                step_prediction_indices=train_step_indices,
                detach_unet_input=args.algo == "drtune",
                detach_untrained_predictions=True,
                stop_after_step_index=stop_after_step_index,
                show_progress=False,
            )

            styled_prompt_close = build_prompt(pair["prompt_close"], pair_style)
            styled_prompt_far = build_prompt(pair["prompt_far"], pair_style)
            reward_outputs = []
            reward_losses = []
            if sample.step_predictions:
                for step_prediction in sample.step_predictions:
                    close_weight, far_weight = compute_hybrid_reward_weights(
                        denoising_progress=step_prediction.denoising_progress,
                        close_base_weight=args.close_reward_weight,
                        far_base_weight=args.far_reward_weight,
                        min_weight_scale=args.step_reward_min_weight_scale,
                        schedule=args.step_reward_schedule,
                        exponent=args.step_reward_exponent,
                        far_early_boost=args.far_reward_early_boost,
                        far_early_boost_exponent=args.far_reward_early_boost_exponent,
                    )
                    reward_output = reward_model.score_hybrid(
                        prompt_close=styled_prompt_close,
                        prompt_far=styled_prompt_far,
                        image=step_prediction.image,
                        far_view=step_prediction.far_view,
                        close_weight=close_weight,
                        far_weight=far_weight,
                    )
                    reward_outputs.append((step_prediction, close_weight, far_weight, reward_output))
                    reward_losses.append(-reward_output.total_scores.mean())
            else:
                close_weight, far_weight = compute_hybrid_reward_weights(
                    denoising_progress=1.0,
                    close_base_weight=args.close_reward_weight,
                    far_base_weight=args.far_reward_weight,
                    min_weight_scale=args.step_reward_min_weight_scale,
                    schedule=args.step_reward_schedule,
                    exponent=args.step_reward_exponent,
                    far_early_boost=args.far_reward_early_boost,
                    far_early_boost_exponent=args.far_reward_early_boost_exponent,
                )
                reward_output = reward_model.score_hybrid(
                    prompt_close=styled_prompt_close,
                    prompt_far=styled_prompt_far,
                    image=sample.image,
                    far_view=sample.far_view,
                    close_weight=close_weight,
                    far_weight=far_weight,
                )
                reward_outputs.append((None, close_weight, far_weight, reward_output))
                reward_losses.append(-reward_output.total_scores.mean())

            reward_loss = torch.stack(reward_losses).mean()
            tv_loss = args.tv_weight * total_variation_loss(sample.image)
            loss = reward_loss + tv_loss
            loss.backward()

            grad_norm = clip_grad_norm_(lora_layers.parameters(), args.max_grad_norm)
            optimizer.step()

            mean_close_reward = sum(
                float(reward_output.close_scores.mean().detach().cpu()) for _, _, _, reward_output in reward_outputs
            ) / len(reward_outputs)
            mean_far_reward = sum(
                float(reward_output.far_scores.mean().detach().cpu()) for _, _, _, reward_output in reward_outputs
            ) / len(reward_outputs)
            mean_total_reward = sum(
                float(reward_output.total_scores.mean().detach().cpu()) for _, _, _, reward_output in reward_outputs
            ) / len(reward_outputs)
            metrics = {
                "iteration": iteration,
                "global_step": iteration + 1,
                "epoch": round((iteration + 1) / len(pair_order), 4),
                "pair_id": pair.get("pair_id"),
                "prompt_close": pair["prompt_close"],
                "prompt_far": pair["prompt_far"],
                "close_reward": mean_close_reward,
                "far_reward": mean_far_reward,
                "total_reward": mean_total_reward,
                "reward_loss": float(reward_loss.detach().cpu()),
                "tv_loss": float(tv_loss.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
                "train_step_indices": sorted(train_step_indices),
                "stop_after_step_index": stop_after_step_index,
                "reward_steps": [
                    {
                        "step_index": None if step_prediction is None else step_prediction.step_index,
                        "denoising_progress": None if step_prediction is None else round(step_prediction.denoising_progress, 4),
                        "close_weight": close_weight,
                        "far_weight": far_weight,
                        "close_reward": float(reward_output.close_scores.mean().detach().cpu()),
                        "far_reward": float(reward_output.far_scores.mean().detach().cpu()),
                        "total_reward": float(reward_output.total_scores.mean().detach().cpu()),
                    }
                    for step_prediction, close_weight, far_weight, reward_output in reward_outputs
                ],
            }
            with open(metrics_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(metrics, ensure_ascii=False) + "\n")

            global_step = iteration + 1
            should_save = global_step % args.save_every == 0 or iteration == args.max_iterations - 1
            should_preview = global_step % args.preview_every == 0 or iteration == args.max_iterations - 1
            should_log = global_step % args.log_every == 0 or global_step == 1 or should_save or should_preview
            current_reward = metrics["total_reward"]
            if best_reward is None or current_reward > best_reward:
                best_reward = current_reward
                pipeline.unet.save_attn_procs(output_dir / "lora_best")

            wandb_payload = {
                "train/close_reward": mean_close_reward,
                "train/far_reward": mean_far_reward,
                "train/total_reward": mean_total_reward,
                "train/reward_loss": metrics["reward_loss"],
                "train/tv_loss": metrics["tv_loss"],
                "train/loss": metrics["loss"],
                "train/grad_norm": metrics["grad_norm"],
                "train/num_reward_steps": len(reward_outputs),
                "train/stop_after_step_index": -1 if stop_after_step_index is None else stop_after_step_index,
            }

            if should_save:
                pipeline.unet.save_attn_procs(output_dir / "lora_latest")

            if should_preview:
                preview_conditioning = prepare_sdxl_conditioning(
                    pipeline,
                    prompts=ordered_prompts(
                        prompt_close=preview_pair["prompt_close"],
                        prompt_far=preview_pair["prompt_far"],
                        style=preview_pair.get("style", ""),
                    ),
                    negative_prompt=preview_pair.get("negative_prompt", ""),
                    height=args.height,
                    width=args.width,
                )
                preview_kwargs = {
                    "height": args.height,
                    "width": args.width,
                    "num_inference_steps": args.num_inference_steps,
                    "guidance_scale": args.guidance_scale,
                    "guidance_scale_far": args.guidance_scale_far,
                    "guidance_scale_close": args.guidance_scale_close,
                    "reduction": args.reduction,
                    "latent_sigma": args.latent_sigma,
                    "latent_kernel_size": args.latent_kernel_size,
                    "composition_space": args.composition_space,
                    "rgb_hybrid_sigma": args.rgb_hybrid_sigma,
                    "rgb_hybrid_kernel_size": args.rgb_hybrid_kernel_size,
                    "far_resize_factor": args.far_resize_factor,
                    "far_blur_sigma": args.far_blur_sigma,
                    "show_progress": False,
                }
                with torch.inference_mode():
                    baseline_preview = None
                    with temporarily_disable_adapters(pipeline.unet) as adapters_disabled:
                        if adapters_disabled:
                            baseline_preview = sample_latent_hybrid(
                                pipeline,
                                preview_conditioning,
                                generator=make_generator(seed=args.preview_seed, device=args.device),
                                **preview_kwargs,
                            )
                    preview = sample_latent_hybrid(
                        pipeline,
                        preview_conditioning,
                        generator=make_generator(seed=args.preview_seed, device=args.device),
                        **preview_kwargs,
                    )
                preview_dir = output_dir / "previews" / f"{global_step:05d}"
                save_hybrid_sample(preview, preview_dir)
                if baseline_preview is not None:
                    save_hybrid_sample(baseline_preview, preview_dir, prefix="baseline")
                    save_preview_comparison(
                        baseline_sample=baseline_preview,
                        tuned_sample=preview,
                        output_dir=preview_dir,
                    )

                if wandb_run is not None:
                    import wandb

                    caption = (
                        f"step={global_step} | pair_id={preview_pair.get('pair_id')} | "
                        f"far={preview_pair['prompt_far']} | close={preview_pair['prompt_close']}"
                    )
                    wandb_payload["preview/close"] = wandb.Image(str(preview_dir / "sample.png"), caption=caption)
                    wandb_payload["preview/far"] = wandb.Image(str(preview_dir / "sample.far.png"), caption=caption)
                    if baseline_preview is not None:
                        comparison_caption = f"{caption} | left=baseline, right=rl"
                        wandb_payload["preview/baseline_close"] = wandb.Image(
                            str(preview_dir / "baseline.png"),
                            caption=caption,
                        )
                        wandb_payload["preview/baseline_far"] = wandb.Image(
                            str(preview_dir / "baseline.far.png"),
                            caption=caption,
                        )
                        wandb_payload["preview/comparison_close"] = wandb.Image(
                            str(preview_dir / "comparison.close.png"),
                            caption=comparison_caption,
                        )
                        wandb_payload["preview/comparison_far"] = wandb.Image(
                            str(preview_dir / "comparison.far.png"),
                            caption=comparison_caption,
                        )
                        wandb_payload["preview/comparison_grid"] = wandb.Image(
                            str(preview_dir / "comparison.png"),
                            caption=f"{caption} | row1=baseline(close,far), row2=rl(close,far)",
                        )

            if should_log:
                print(
                    json.dumps(
                        {
                            "global_step": global_step,
                            "epoch": metrics["epoch"],
                            "pair_id": metrics["pair_id"],
                            "total_reward": metrics["total_reward"],
                            "close_reward": metrics["close_reward"],
                            "far_reward": metrics["far_reward"],
                            "grad_norm": metrics["grad_norm"],
                            "saved_checkpoint": should_save,
                            "saved_preview": should_preview,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            if wandb_run is not None:
                wandb_run.log(wandb_payload, step=global_step)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
