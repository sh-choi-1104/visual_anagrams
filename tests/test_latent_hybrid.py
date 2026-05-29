from __future__ import annotations

import random
import unittest

import torch

from visual_anagrams.latent_hybrid import ordered_prompts, resolve_guidance_scales, simulate_far_view
from visual_anagrams.latent_views import LatentHybridHighPassView, LatentHybridLowPassView
from visual_anagrams.rl import compute_hybrid_reward_weights, select_train_step_indices


class LatentHybridTests(unittest.TestCase):
    def test_ordered_prompts_map_far_then_close(self) -> None:
        prompts = ordered_prompts("close fox", "far moon", style="an oil painting of")
        self.assertEqual(
            prompts,
            ["an oil painting of far moon", "an oil painting of close fox"],
        )

    def test_low_and_high_pass_reconstruct_original(self) -> None:
        torch.manual_seed(0)
        tensor = torch.randn(4, 128, 128)
        low = LatentHybridLowPassView(sigma=1.5, kernel_size=9).inverse_view(tensor)
        high = LatentHybridHighPassView(sigma=1.5, kernel_size=9).inverse_view(tensor)
        self.assertTrue(torch.allclose(low + high, tensor, atol=1e-5, rtol=1e-4))

    def test_far_view_preserves_shape(self) -> None:
        image = torch.rand(1, 3, 256, 256)
        far_view = simulate_far_view(image, resize_factor=0.4, blur_sigma=4.0)
        self.assertEqual(far_view.shape, image.shape)
        self.assertGreaterEqual(float(far_view.min()), 0.0)
        self.assertLessEqual(float(far_view.max()), 1.0)

    def test_guidance_scales_default_to_shared_scale(self) -> None:
        guidance_far, guidance_close = resolve_guidance_scales(guidance_scale=6.5)
        self.assertEqual(guidance_far, 6.5)
        self.assertEqual(guidance_close, 6.5)

    def test_guidance_scales_allow_per_prompt_override(self) -> None:
        guidance_far, guidance_close = resolve_guidance_scales(
            guidance_scale=6.5,
            guidance_scale_far=8.0,
            guidance_scale_close=4.0,
        )
        self.assertEqual(guidance_far, 8.0)
        self.assertEqual(guidance_close, 4.0)

    def test_draft_k_selects_last_steps(self) -> None:
        indices = select_train_step_indices(30, 5, "draft_k", random.Random(0))
        self.assertEqual(indices, {25, 26, 27, 28, 29})

    def test_drtune_spreads_steps(self) -> None:
        indices = sorted(select_train_step_indices(30, 5, "drtune", random.Random(0)))
        self.assertEqual(len(indices), 5)
        self.assertGreater(indices[-1] - indices[0], 10)

    def test_step_reward_weights_shift_from_far_to_close(self) -> None:
        early_close, early_far = compute_hybrid_reward_weights(
            denoising_progress=0.0,
            close_base_weight=1.0,
            far_base_weight=1.0,
            min_weight_scale=0.25,
        )
        late_close, late_far = compute_hybrid_reward_weights(
            denoising_progress=1.0,
            close_base_weight=1.0,
            far_base_weight=1.0,
            min_weight_scale=0.25,
        )
        self.assertLess(early_close, early_far)
        self.assertGreater(late_close, late_far)

    def test_far_early_boost_only_affects_early_far_reward(self) -> None:
        base_early_close, base_early_far = compute_hybrid_reward_weights(
            denoising_progress=0.0,
            close_base_weight=1.0,
            far_base_weight=1.0,
            min_weight_scale=0.25,
        )
        boosted_early_close, boosted_early_far = compute_hybrid_reward_weights(
            denoising_progress=0.0,
            close_base_weight=1.0,
            far_base_weight=1.0,
            min_weight_scale=0.25,
            far_early_boost=0.75,
        )
        boosted_late_close, boosted_late_far = compute_hybrid_reward_weights(
            denoising_progress=1.0,
            close_base_weight=1.0,
            far_base_weight=1.0,
            min_weight_scale=0.25,
            far_early_boost=0.75,
        )
        self.assertEqual(base_early_close, boosted_early_close)
        self.assertGreater(boosted_early_far, base_early_far)
        self.assertEqual(boosted_late_close, 1.0)
        self.assertEqual(boosted_late_far, 0.25)


if __name__ == "__main__":
    unittest.main()
