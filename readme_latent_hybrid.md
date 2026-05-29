# Latent Hybrid Images with SDXL + HPSv2

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
  --download_sdxl_minimal \
  --download_hpsv2_checkpoint \
  --clone_hpsv2_repo
```

Expected layout:

- `/data/models/sdxl-base-1.0`
- `/data/models/HPSv2-weights/HPS_v2.1_compressed.pt`
- `/data/models/HPSv2-repo`

Notes:

- HPSv2 only needs the compressed checkpoint for weights.
- SDXL still needs the diffusers configs/tokenizers/scheduler in addition to the main weight files, so the downloader grabs a minimal diffusers snapshot instead of the whole repository.

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

위 커맨드는 논문에서 말한 latent-space failure case를 보기 위한 `latent_eps` baseline입니다. 실제로는 "섞인 그림"처럼 보일 수 있습니다.

좀 더 hybrid image에 가까운 latent-backbone 실험을 하려면 decoded `x0`의 RGB 공간에서 low/high-pass를 합치는 모드를 쓰면 됩니다:

```bash
uv run python inference_latent_hybrid.py \
  --name eagle_cathedral_decoded_x0 \
  --prompt_close "a detailed eagle portrait" \
  --prompt_far "a gothic cathedral" \
  --style "an oil painting of" \
  --sdxl_model_path /data/models/sdxl-base-1.0 \
  --composition_space decoded_x0_rgb \
  --guidance_scale_far 7.0 \
  --guidance_scale_close 5.0 \
  --rgb_hybrid_sigma 10 \
  --rgb_hybrid_kernel_size 51
```

You can keep a single shared CFG with `--guidance_scale`, or set separate prompt-wise CFG values with:

- `--guidance_scale_far`: CFG for the low-frequency / far prompt
- `--guidance_scale_close`: CFG for the high-frequency / close prompt

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
  --guidance_scale_far 7.0 \
  --guidance_scale_close 5.0 \
  --latent_sigma 1.5 \
  --latent_kernel_size 9
```

Outputs:

- `sample.png`: the close-view image
- `sample.far.png`: a simulated far-view rendering used for evaluation/debugging

`generate_latent_hybrid.py` is useful when you simply want one output, optionally with a tuned LoRA loaded.

## 4. Reward-tune with HPSv2

먼저 synthetic prompt-pair 데이터셋을 만들 수 있습니다:

```bash
uv run python make_synthetic_prompt_pairs.py \
  --output_path data/prompt_pairs_hybrid_10k.jsonl \
  --num_pairs 10000 \
  --seed 0
```

이 데이터셋은:

- `prompt_far`: 멀리서 더 잘 보이도록 low-frequency subject 위주
- `prompt_close`: 가까이서 더 잘 보이도록 high-frequency detail 위주

로 합성됩니다.

`draft_k` tunes the last `K` denoising steps.

```bash
uv run python train_latent_hybrid_reward.py \
  --output_dir results_latent_reward/eagle_cathedral_draftk \
  --prompt_close "a detailed eagle portrait" \
  --prompt_far "a gothic cathedral" \
  --style "an oil painting of" \
  --sdxl_model_path /data/models/sdxl-base-1.0 \
  --hpsv2_repo_path /data/models/HPSv2-repo \
  --hpsv2_checkpoint_path /data/models/HPSv2-weights/HPS_v2.1_compressed.pt \
  --hpsv2_version v2.1 \
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
  --prompt_pairs_jsonl data/prompt_pairs_hybrid_10k.jsonl \
  --sdxl_model_path /data/models/sdxl-base-1.0 \
  --hpsv2_repo_path /data/models/HPSv2-repo \
  --hpsv2_checkpoint_path /data/models/HPSv2-weights/HPS_v2.1_compressed.pt \
  --hpsv2_version v2.1 \
  --algo drtune \
  --composition_space decoded_x0_rgb \
  --reward_train_steps 5 \
  --early_stop_max_steps 3 \
  --max_iterations 200 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --preview_every 500 \
  --far_reward_early_boost 0.75
```

기본 설정에서는 step-aware reward weighting이 켜져 있습니다:

- early denoising step: `far` reward 비중이 더 큼
- late denoising step: `close` reward 비중이 더 큼

관련 인자:

- `--step_reward_min_weight_scale`
- `--step_reward_schedule`
- `--step_reward_exponent`
- `--far_reward_early_boost`
- `--far_reward_early_boost_exponent`

Saved artifacts:

- `lora_best/`: best LoRA checkpoint by reward
- `lora_latest/`: latest checkpoint
- `previews/`: sampled previews during tuning, including `baseline` vs `rl` comparison grids when adapters are available
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
- HPSv2 support is loaded from the official repo checkout at `/data/models/HPSv2-repo`.
- The HPSv2 reward checkpoint is a `.pt` file, not a `.safetensors` file.
