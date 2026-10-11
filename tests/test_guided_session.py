"""End-to-end synthetic listeners, including an unreachable long comparison."""

import unittest

import numpy as np

from phonix.engine import Pair, add_observation, fit_channel, new_channel, pairs_for, refine, select_pair
from phonix.model import Reference
from phonix.profiles import Settings


class GuidedSessionTests(unittest.TestCase):
    def test_guided_session_handles_censored_pairs_and_converges(self):
        settings = Settings(seed=4)
        channel = new_channel(settings)
        reference = Reference()
        rng = np.random.default_rng(4)
        for _ in range(settings.budget):
            refine(channel, settings, reference)
            fit = fit_channel(channel, settings, reference)
            if fit.ready(settings):
                break
            a, b = select_pair(channel, settings, rng, reference)
            self.assertLessEqual(b / a, 2 + 1e-12)
            pair = pairs_for(channel, settings).get((a, b))
            if pair:
                offset = pair.next_offset()
            else:
                i, j = channel["frequencies"].index(a), channel["frequencies"].index(b)
                offset = float(np.clip(fit.matches[j] - fit.matches[i], -settings.offset_limit, settings.offset_limit))
            frequencies = np.array([a, b])
            distortion = 4 * np.exp(-0.5 * (np.log2(frequencies / 6500) / 0.35) ** 2)
            matches = -(reference.target(frequencies) + reference.sensitivity(frequencies) + distortion)
            evidence = offset - (matches[1] - matches[0]) + rng.logistic(0, 1.5)
            answer = "second" if evidence > 0.75 else "first" if evidence < -0.75 else "equal"
            add_observation(channel, settings, a, b, offset, "guided", answer)
        fit = fit_channel(channel, settings, reference)
        self.assertTrue(fit.ready(settings), f"Failed after {channel['inputs']} inputs: {fit}")
        self.assertLess(channel["inputs"], settings.budget)
        frequencies = fit.frequencies
        distortion = 4 * np.exp(-0.5 * (np.log2(frequencies / 6500) / 0.35) ** 2)
        recovered = reference.correction(frequencies, fit.matches)
        error = recovered + distortion
        error -= error.mean()
        self.assertLess(np.sqrt(np.mean(error ** 2)), 1.5)

    def test_boundary_comparison_does_not_replace_unmeasured_coverage(self):
        settings = Settings(initial_points=4, max_points=4)
        channel = new_channel(settings)
        a, b = channel["frequencies"][:2]
        for _ in range(80):
            add_observation(channel, settings, a, b, 17, "guided", "first")
        pair = Pair(a, b, channel["observations"], settings.offset_limit)
        self.assertTrue(pair.boundary)
        fit = fit_channel(channel, settings)
        self.assertFalse(fit.connected)
        self.assertEqual(fit.boundary_pairs, 1)
        self.assertFalse(fit.ready(settings))
        selected = select_pair(channel, settings, np.random.default_rng(1))
        self.assertNotEqual(selected, (a, b))
