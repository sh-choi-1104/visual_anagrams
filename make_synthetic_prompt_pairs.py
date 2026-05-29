from __future__ import annotations

import argparse
import json
import random
from itertools import product
from pathlib import Path


STYLES = [
    "an oil painting of",
    "a lithograph of",
    "a woodcut print of",
    "a surreal illustration of",
    "a vintage poster of",
    "a watercolor painting of",
    "an ink drawing of",
    "a pencil illustration of",
    "a stained glass depiction of",
    "a mosaic artwork of",
    "a bronze relief of",
    "a storybook illustration of",
    "a paper-cut artwork of",
    "a mural of",
]

FAR_SUBJECTS = {
    "portrait": [
        "old man portrait",
        "woman's face",
        "child profile",
        "marble mask",
        "crowned queen portrait",
        "serene monk face",
        "skull portrait",
    ],
    "animal": [
        "owl face",
        "lion head",
        "wolf profile",
        "rabbit silhouette",
        "swan",
        "horse profile",
        "phoenix silhouette",
        "butterfly",
    ],
    "architecture": [
        "gothic cathedral facade",
        "lighthouse",
        "temple gate",
        "castle silhouette",
        "pagoda",
        "bell tower",
        "domed palace",
    ],
    "object": [
        "violin",
        "chess king",
        "sailing ship",
        "crescent moon",
        "hourglass",
        "keyhole symbol",
        "spiral shell silhouette",
    ],
}

FAR_DESCRIPTORS = [
    "monumental",
    "centered",
    "iconic",
    "bold",
    "simple",
    "clear",
    "dramatic",
    "symmetrical",
    "high-contrast",
    "poster-like",
]

FAR_COMPOSITIONS = [
    "centered composition",
    "symmetrical layout",
    "bold silhouette",
    "large simple shapes",
    "clean background",
    "frontal view",
]

CLOSE_SUBJECTS = {
    "feather": [
        "eagle feathers",
        "peacock plumage",
        "owl feathers",
        "phoenix feathers",
    ],
    "flora": [
        "rose petals",
        "sunflower petals",
        "ivy vines",
        "wild mushrooms",
        "autumn leaves",
        "coral branches",
    ],
    "material": [
        "stained glass fragments",
        "mosaic tiles",
        "embroidered fabric",
        "ornate lace",
        "folded origami layers",
        "ceramic shards",
    ],
    "mechanical": [
        "watch gears",
        "rusted machinery",
        "brass filigree",
        "circuit traces",
        "mechanical engravings",
    ],
    "texture": [
        "fur patterns",
        "snake scales",
        "frost crystals",
        "smoke swirls",
        "gemstone facets",
        "woven threads",
    ],
}

CLOSE_DESCRIPTORS = [
    "intricate",
    "ornate",
    "ultra-detailed",
    "textured",
    "layered",
    "delicate",
    "dense",
    "filigreed",
    "fractal-like",
    "high-frequency",
]

CLOSE_COMPOSITIONS = [
    "rich micro-details",
    "dense texture",
    "close-up view",
    "fine edges",
    "ornamental detail",
    "busy surface patterns",
]

NEGATIVE_PROMPT = "text, watermark, logo, blurry, low contrast, low quality, distorted anatomy"

FOCUS_STYLES = [
    "an oil painting of",
    "a lithograph of",
    "a woodcut print of",
    "a stained glass depiction of",
]

FOCUS_FAR_PROMPTS = [
    "centered owl face, bold silhouette",
    "iconic rabbit silhouette, large simple shapes",
    "poster-like serene monk face, symmetrical layout",
    "monumental lighthouse, centered composition",
    "high-contrast butterfly, symmetrical layout",
    "simple crescent moon, clean background",
    "iconic violin, centered composition",
    "bold chess king, frontal view",
    "castle silhouette, large simple shapes",
    "temple gate, frontal view",
]

FOCUS_CLOSE_PROMPTS = [
    "intricate eagle feathers, fine edges",
    "high-frequency peacock plumage, dense texture",
    "delicate owl feathers, rich micro-details",
    "textured phoenix feathers, fine edges",
    "ultra-detailed rose petals, busy surface patterns",
    "intricate sunflower petals, dense texture",
    "filigreed ivy vines, ornamental detail",
    "delicate wild mushrooms, busy surface patterns",
    "intricate autumn leaves, rich micro-details",
    "ornate coral branches, dense texture",
    "filigreed stained glass fragments, fine edges",
    "high-frequency mosaic tiles, rich micro-details",
    "embroidered fabric, dense texture",
    "ornate lace, fine edges",
    "high-frequency folded origami layers, fine edges",
    "ceramic shards, busy surface patterns",
    "intricate watch gears, dense texture",
    "ornate brass filigree, ornamental detail",
    "high-frequency circuit traces, fine edges",
    "mechanical engravings, rich micro-details",
    "intricate snake scales, dense texture",
    "frost crystals, fine edges",
    "ornate smoke swirls, close-up view",
    "gemstone facets, busy surface patterns",
    "woven threads, dense texture",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a synthetic prompt-pair dataset for latent hybrid training.")
    parser.add_argument("--output_path", default="data/prompt_pairs_hybrid_10k.jsonl", type=str)
    parser.add_argument("--output_tsv_path", default=None, type=str)
    parser.add_argument("--num_pairs", default=10000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--preset", default="random_v1", choices=["random_v1", "focus_1k"])
    return parser.parse_args()


def pick_subject(rng: random.Random, subject_pool: dict[str, list[str]]) -> tuple[str, str]:
    category = rng.choice(list(subject_pool.keys()))
    subject = rng.choice(subject_pool[category])
    return category, subject


def build_far_prompt(rng: random.Random) -> tuple[str, str]:
    category, subject = pick_subject(rng, FAR_SUBJECTS)
    descriptor = rng.choice(FAR_DESCRIPTORS)
    composition = rng.choice(FAR_COMPOSITIONS)
    prompt = f"{descriptor} {subject}, {composition}"
    return category, prompt


def build_close_prompt(rng: random.Random) -> tuple[str, str]:
    category, subject = pick_subject(rng, CLOSE_SUBJECTS)
    descriptor = rng.choice(CLOSE_DESCRIPTORS)
    composition = rng.choice(CLOSE_COMPOSITIONS)
    prompt = f"{descriptor} {subject}, {composition}"
    return category, prompt


def make_pair(rng: random.Random, pair_id: int) -> dict[str, str]:
    close_category, prompt_close = build_close_prompt(rng)
    far_category, prompt_far = build_far_prompt(rng)
    style = rng.choice(STYLES)
    return {
        "pair_id": f"synthetic_{pair_id:05d}",
        "prompt_close": prompt_close,
        "prompt_far": prompt_far,
        "style": style,
        "negative_prompt": NEGATIVE_PROMPT,
        "close_category": close_category,
        "far_category": far_category,
        "source": "synthetic_hybrid_pairs_v1",
    }


def make_focus_pairs() -> list[dict[str, str]]:
    pairs = []
    for pair_id, (style, prompt_far, prompt_close) in enumerate(
        product(FOCUS_STYLES, FOCUS_FAR_PROMPTS, FOCUS_CLOSE_PROMPTS)
    ):
        pairs.append(
            {
                "pair_id": f"focus_{pair_id:05d}",
                "prompt_close": prompt_close,
                "prompt_far": prompt_far,
                "style": style,
                "negative_prompt": NEGATIVE_PROMPT,
                "close_category": "focus_close",
                "far_category": "focus_far",
                "source": "synthetic_hybrid_pairs_focus_1k_v1",
            }
        )
    return pairs


def write_pairs_jsonl(output_path: Path, pairs: list[dict[str, str]]) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        for pair in pairs:
            file.write(json.dumps(pair, ensure_ascii=False) + "\n")


def write_pairs_tsv(output_path: Path, pairs: list[dict[str, str]]) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("pair_id\tstyle\tprompt_far\tprompt_close\tnegative_prompt\n")
        for pair in pairs:
            row = [
                pair["pair_id"],
                pair["style"],
                pair["prompt_far"],
                pair["prompt_close"],
                pair["negative_prompt"],
            ]
            file.write("\t".join(row) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.preset == "focus_1k":
        pairs = make_focus_pairs()
    else:
        pairs = []
        seen = set()
        while len(pairs) < args.num_pairs:
            pair = make_pair(rng, pair_id=len(pairs))
            dedupe_key = (
                pair["prompt_close"],
                pair["prompt_far"],
                pair["style"],
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            pairs.append(pair)

    write_pairs_jsonl(output_path, pairs)
    if args.output_tsv_path is not None:
        write_pairs_tsv(Path(args.output_tsv_path), pairs)

    summary = {
        "output_path": str(output_path),
        "num_pairs": len(pairs),
        "seed": args.seed,
        "preset": args.preset,
        "styles": len(FOCUS_STYLES if args.preset == "focus_1k" else STYLES),
        "close_categories": ["focus_close"] if args.preset == "focus_1k" else sorted(CLOSE_SUBJECTS.keys()),
        "far_categories": ["focus_far"] if args.preset == "focus_1k" else sorted(FAR_SUBJECTS.keys()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
