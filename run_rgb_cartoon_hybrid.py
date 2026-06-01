from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from diffusers import FluxImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image

from visual_anagrams.latent_inverse import center_crop_to_aspect, compose_inverse_rgb_hybrid, simulate_far_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cartoonize a reference image with img2img, then compose a RGB-space hybrid from original low frequencies and cartoon high frequencies."
    )
    parser.add_argument("--ref_im_path", default="assets/vgi.png", type=str)
    parser.add_argument("--save_dir", default="results_rgb_cartoon_hybrid/vgi_cartoon_rgb", type=str)
    parser.add_argument("--model_family", choices=["sdxl", "flux", "both"], default="both")
    parser.add_argument("--sdxl_model", default="/data/models/sdxl-base-1.0", type=str)
    parser.add_argument("--flux_model", default="/data/models/FLUX.1-dev", type=str)
    parser.add_argument(
        "--prompt",
        default=(
            "a clean colorful cartoon illustration of the same university building photo, "
            "cel shaded, crisp ink outlines, simplified architectural details, bright natural colors"
        ),
        type=str,
    )
    parser.add_argument(
        "--negative_prompt",
        default="blurry, low quality, deformed architecture, distorted perspective, extra buildings, watermark",
        type=str,
    )
    parser.add_argument("--height", default=512, type=int)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--num_inference_steps", default=28, type=int)
    parser.add_argument("--seed", default=777, type=int)
    parser.add_argument("--sdxl_guidance_scale", default=5.0, type=float)
    parser.add_argument("--flux_guidance_scale", default=3.5, type=float)
    parser.add_argument("--flux_true_cfg_scale", default=1.0, type=float)
    parser.add_argument("--strengths", nargs="+", default=["0.30", "0.42"], help="Img2img strengths to sweep.")
    parser.add_argument("--sigmas", nargs="+", default=["4", "6", "8"], help="RGB hybrid low-pass sigmas to sweep.")
    parser.add_argument("--projection_strength", default=1.0, type=float)
    parser.add_argument("--far_resize_factor", default=0.25, type=float)
    parser.add_argument("--far_blur_sigma", default=2.0, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--max_sequence_length", default=256, type=int)
    return parser.parse_args()


def load_reference(path: str | Path, *, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = center_crop_to_aspect(image, width=width, height=height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def raw_tensor(image: Image.Image, *, device: torch.device) -> torch.Tensor:
    return (TF.to_tensor(image).unsqueeze(0).to(device=device) * 2.0) - 1.0


def kernel_for_sigma(sigma: float) -> int:
    return int(round(sigma * 6)) | 1


def compose_rgb_hybrid(
    *,
    close_image: Image.Image,
    reference_image: Image.Image,
    sigma: float,
    projection_strength: float,
    device: torch.device,
) -> torch.Tensor:
    return compose_inverse_rgb_hybrid(
        raw_tensor(close_image, device=device),
        raw_tensor(reference_image, device=device),
        sigma=sigma,
        kernel_size=kernel_for_sigma(sigma),
        projection_strength=projection_strength,
    )


def save_diag(
    *,
    reference: Image.Image,
    close: Image.Image,
    hybrid_01: torch.Tensor,
    output_dir: Path,
    far_resize_factor: float,
    far_blur_sigma: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    far = simulate_far_view(hybrid_01, resize_factor=far_resize_factor, blur_sigma=far_blur_sigma)
    reference.save(output_dir / "reference.png")
    close.save(output_dir / "cartoon_close.png")
    save_image(hybrid_01, output_dir / "sample.png")
    save_image(far, output_dir / "sample.far.png")
    diag = torch.cat([TF.to_tensor(reference).unsqueeze(0), TF.to_tensor(close).unsqueeze(0), hybrid_01, far], dim=-1)
    diag_path = output_dir / "sample.diagnostic.png"
    save_image(diag.clamp(0, 1), diag_path)
    return diag_path


def load_font(size: int) -> ImageFont.ImageFont:
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def write_contact_sheet(records: list[tuple[str, Path]], output_path: Path) -> None:
    if not records:
        return
    font = load_font(15)
    title_font = load_font(18)
    cell_w, cell_h = 1024, 286
    rows = len(records)
    sheet = Image.new("RGB", (cell_w, cell_h * rows + 38), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "RGB post-hoc cartoon hybrid | reference / cartoon close / hybrid / far", fill=(0, 0, 0), font=title_font)
    for idx, (label, path) in enumerate(records):
        image = Image.open(path).convert("RGB").resize((1024, 256), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (cell_w, cell_h), "white")
        canvas.paste(image, (0, 0))
        ImageDraw.Draw(canvas).text((8, 262), label, fill=(0, 0, 0), font=font)
        sheet.paste(canvas, (0, 38 + idx * cell_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def run_sdxl(args: argparse.Namespace, reference: Image.Image, device: torch.device) -> list[tuple[str, Path]]:
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        args.sdxl_model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    ).to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    records: list[tuple[str, Path]] = []
    for strength_text in args.strengths:
        strength = float(strength_text)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        close = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=reference,
            strength=strength,
            guidance_scale=args.sdxl_guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[0]
        family_dir = Path(args.save_dir) / "sdxl" / f"strength_{strength:g}".replace(".", "p")
        family_dir.mkdir(parents=True, exist_ok=True)
        close.save(family_dir / "cartoon_close.png")
        for sigma_text in args.sigmas:
            sigma = float(sigma_text)
            hybrid_raw = compose_rgb_hybrid(
                close_image=close,
                reference_image=reference,
                sigma=sigma,
                projection_strength=args.projection_strength,
                device=device,
            )
            hybrid_01 = (hybrid_raw / 2 + 0.5).detach().cpu().clamp(0, 1)
            out_dir = family_dir / f"sigma_{sigma:g}".replace(".", "p")
            diag_path = save_diag(
                reference=reference,
                close=close,
                hybrid_01=hybrid_01,
                output_dir=out_dir,
                far_resize_factor=args.far_resize_factor,
                far_blur_sigma=args.far_blur_sigma,
            )
            (out_dir / "metadata.json").write_text(
                json.dumps({"family": "sdxl", "strength": strength, "sigma": sigma, "prompt": args.prompt}, indent=2) + "\n",
                encoding="utf-8",
            )
            records.append((f"SDXL img2img strength {strength:g}, RGB sigma {sigma:g}", diag_path))

    del pipe
    torch.cuda.empty_cache()
    return records


def run_flux(args: argparse.Namespace, reference: Image.Image, device: torch.device) -> list[tuple[str, Path]]:
    pipe = FluxImg2ImgPipeline.from_pretrained(
        args.flux_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    ).to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    records: list[tuple[str, Path]] = []
    for strength_text in args.strengths:
        strength = float(strength_text)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        close = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            true_cfg_scale=args.flux_true_cfg_scale,
            image=reference,
            height=args.height,
            width=args.width,
            strength=strength,
            guidance_scale=args.flux_guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            max_sequence_length=args.max_sequence_length,
        ).images[0]
        family_dir = Path(args.save_dir) / "flux" / f"strength_{strength:g}".replace(".", "p")
        family_dir.mkdir(parents=True, exist_ok=True)
        close.save(family_dir / "cartoon_close.png")
        for sigma_text in args.sigmas:
            sigma = float(sigma_text)
            hybrid_raw = compose_rgb_hybrid(
                close_image=close,
                reference_image=reference,
                sigma=sigma,
                projection_strength=args.projection_strength,
                device=device,
            )
            hybrid_01 = (hybrid_raw / 2 + 0.5).detach().cpu().clamp(0, 1)
            out_dir = family_dir / f"sigma_{sigma:g}".replace(".", "p")
            diag_path = save_diag(
                reference=reference,
                close=close,
                hybrid_01=hybrid_01,
                output_dir=out_dir,
                far_resize_factor=args.far_resize_factor,
                far_blur_sigma=args.far_blur_sigma,
            )
            (out_dir / "metadata.json").write_text(
                json.dumps({"family": "flux", "strength": strength, "sigma": sigma, "prompt": args.prompt}, indent=2) + "\n",
                encoding="utf-8",
            )
            records.append((f"FLUX img2img strength {strength:g}, RGB sigma {sigma:g}", diag_path))

    del pipe
    torch.cuda.empty_cache()
    return records


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    root = Path(args.save_dir)
    root.mkdir(parents=True, exist_ok=True)
    reference = load_reference(args.ref_im_path, width=args.width, height=args.height)
    reference.save(root / "reference.png")

    records: list[tuple[str, Path]] = []
    if args.model_family in {"sdxl", "both"}:
        records.extend(run_sdxl(args, reference, device))
    if args.model_family in {"flux", "both"}:
        records.extend(run_flux(args, reference, device))

    summary_path = root / "summary_rgb_cartoon_hybrid.png"
    write_contact_sheet(records, summary_path)
    (root / "metadata.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
