# Latent Hybrid Images with SDXL + HPSv3

This extension adds a latent-space version of the Visual Anagrams / Factorized Diffusion hybrid-image pipeline.

The goal is:

- up close: show `prompt_close`
- from far away: show `prompt_far`

Internally, the code still follows the factorized diffusion idea:

- low-frequency latent component <- `prompt_far`
- high-frequency latent component <- `prompt_close`

Reward tuning is then used to compensate for the artifact issues that often appear when hybrid factorization is moved from pixel space to latent space.

## 1. Put all models under `/data/models`

You can download the required assets with:

```bash
uv run python download_models.py \
  --download_sdxl \
  --download_hpsv3_checkpoint \
  --download_qwen_backbone \
  --clone_hpsv3_repo
```

Expected layout:

- `/data/models/sdxl-base-1.0`
- `/data/models/HPSv3-weights`
- `/data/models/Qwen2-VL-7B-Instruct`
- `/data/models/HPSv3`

If you already have `hpsv3` installed as a package, cloning the repo is optional.

## 2. Baseline latent inference

Plain latent-factorized baseline:

```bash
uv run python inference_latent_hybrid.py \
  --name eagle_cathedral_baseline \
  --prompt_close "a detailed eagle portrait" \
  --prompt_far "a gothic cathedral" \
  --style "an oil painting of" \
  --sdxl_model_path /data/models/sdxl-base-1.0 \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --latent_sigma 1.5 \
  --latent_kernel_size 9
```

If you also want to compare against a reward-tuned LoRA with the exact same initial latent:

```bash
uv run python inference_latent_hybrid.py \
  --name eagle_cathedral_compare \
  --prompt_close "a detailed eagle portrait" \
  --prompt_far "a gothic cathedral" \
  --style "an oil painting of" \
  --sdxl_model_path /data/models/sdxl-base-1.0 \
  --compare_lora_path results_latent_reward/eagle_cathedral_drtune/lora_best
```

For each seed this saves:

- `baseline.png`
- `baseline.far.png`
- `tuned.png` and `tuned.far.png` if `--compare_lora_path` is provided
- `comparison.png` with a side-by-side baseline/tuned grid

## 3. Generate a latent hybrid

```bash
uv run python generate_latent_hybrid.py \
  --name eagle_cathedral \
  --prompt_close "a detailed eagle portrait" \
  --prompt_far "a gothic cathedral" \
  --style "an oil painting of" \
  --sdxl_model_path /data/models/sdxl-base-1.0 \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --latent_sigma 1.5 \
  --latent_kernel_size 9
```

Outputs:

- `sample.png`: the close-view image
- `sample.far.png`: a simulated far-view rendering used for evaluation/debugging

`generate_latent_hybrid.py` is useful when you simply want one output, optionally with a tuned LoRA loaded.

## 4. Reward-tune with HPSv3

`draft_k` tunes the last `K` denoising steps.

```bash
uv run python train_latent_hybrid_reward.py \
  --output_dir results_latent_reward/eagle_cathedral_draftk \
  --prompt_close "a detailed eagle portrait" \
  --prompt_far "a gothic cathedral" \
  --style "an oil painting of" \
  --sdxl_model_path /data/models/sdxl-base-1.0 \
  --hpsv3_repo_path /data/models/HPSv3 \
  --hpsv3_checkpoint_path /data/models/HPSv3-weights/HPSv3.safetensors \
  --algo draft_k \
  --reward_train_steps 5 \
  --max_iterations 200 \
  --learning_rate 1e-4 \
  --lora_rank 16
```

`drtune` adds stop-gradient on the UNet input and trains equally spaced steps:

```bash
uv run python train_latent_hybrid_reward.py \
  --output_dir results_latent_reward/eagle_cathedral_drtune \
  --prompt_close "a detailed eagle portrait" \
  --prompt_far "a gothic cathedral" \
  --style "an oil painting of" \
  --sdxl_model_path /data/models/sdxl-base-1.0 \
  --hpsv3_repo_path /data/models/HPSv3 \
  --hpsv3_checkpoint_path /data/models/HPSv3-weights/HPSv3.safetensors \
  --algo drtune \
  --reward_train_steps 5 \
  --early_stop_max_steps 3 \
  --max_iterations 200 \
  --learning_rate 1e-4 \
  --lora_rank 16
```

Saved artifacts:

- `lora_best/`: best LoRA checkpoint by reward
- `lora_latest/`: latest checkpoint
- `previews/`: sampled previews during tuning
- `metrics.jsonl`: per-step rewards and losses

## 5. Reuse the tuned LoRA

```bash
uv run python generate_latent_hybrid.py \
  --name eagle_cathedral_tuned \
  --prompt_close "a detailed eagle portrait" \
  --prompt_far "a gothic cathedral" \
  --style "an oil painting of" \
  --sdxl_model_path /data/models/sdxl-base-1.0 \
  --lora_path results_latent_reward/eagle_cathedral_drtune/lora_best
```

## Notes

- `prompt_close` is the image you want to see nearby.
- `prompt_far` is the image you want to see after blur / distance.
- The internal factorization order is reversed to match hybrid-image perception.
- HPSv3 support is loaded lazily. If imports fail, either install `hpsv3` or clone the official repository into `/data/models/HPSv3`.
