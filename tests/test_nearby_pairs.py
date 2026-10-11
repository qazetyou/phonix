"""Local comparison coverage, preservation and limits without live playback."""

import contextlib
import copy
import io
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from phonix.cli import RANGES, run_profile
from phonix.engine import (add_observation, fit_channel, new_channel,
                                     prepare_local_grid, refine, select_pair)
from phonix.profiles import Settings, Store


class NearbyPairTests(unittest.TestCase):
    def test_local_pairs_connect_full_range_with_redundant_checks(self):
        # This test exercises grid additions over the wider supported range.
        settings = Settings(**(RANGES["full"] | dict(min_hz=20, max_hz=16000)))
        channel = new_channel(settings)
        self.assertTrue(prepare_local_grid(channel, settings))
        self.assertFalse(fit_channel(channel, settings).connected)
        self.assertEqual(channel["inputs"], 0)
        rng = np.random.default_rng(37)
        for _ in range(100):
            a, b = select_pair(channel, settings, rng)
            self.assertLessEqual(b / a, 2 + 1e-12)
            # An independently defined smooth listener contour with nearby
            # matches well inside the playback allowance.
            offset = 2 * (np.sin(np.log2(b)) - np.sin(np.log2(a)))
            add_observation(channel, settings, a, b, float(offset), "manual")
        fit = fit_channel(channel, settings)
        self.assertTrue(fit.connected)
        self.assertGreaterEqual(fit.cycles, 2)
        self.assertTrue(np.all(fit.degrees >= 2))
        truth = 2 * np.sin(np.log2(fit.frequencies))
        self.assertLess(np.std(fit.matches - truth), 0.1)

    def test_equally_informative_candidates_prefer_shorter_frequency_gap(self):
        settings = Settings(min_hz=1000, max_hz=2000, initial_points=4, max_points=4)
        channel = {"frequencies": [1000., 1100., 1900., 2000.],
                   "observations": [], "blocked": [], "inputs": 0}
        fit = SimpleNamespace(frequencies=np.array(channel["frequencies"]), matches=np.zeros(4),
                              covariance=np.eye(4), degrees=np.full(4, 2), connected=True)
        with patch("phonix.engine.fit_channel", return_value=fit), \
                patch("phonix.engine.log_weights", return_value=np.full(4, 0.25)):
            self.assertEqual(select_pair(channel, settings, np.random.default_rng(2)), (1900, 2000))

    def test_exact_octave_allowed_and_larger_gap_rejected(self):
        settings = Settings(min_hz=1000, max_hz=3000, initial_points=4, max_points=4)
        channel = {"frequencies": [1000., 1500., 2000., 3000.],
                   "observations": [], "blocked": [], "inputs": 0}
        # Block every local link except the exactly one-octave pair.
        channel["blocked"] = [[1000., 1500.], [1500., 2000.], [1500., 3000.], [2000., 3000.]]
        self.assertEqual(select_pair(channel, settings, np.random.default_rng(0)), (1000, 2000))
        channel["frequencies"][2] = 2000.001
        channel["blocked"] = [[1000., 1500.], [1500., 2000.001], [1500., 3000.], [2000.001, 3000.]]
        before = copy.deepcopy(channel)
        with self.assertRaisesRegex(ValueError, "within one octave"):
            select_pair(channel, settings, np.random.default_rng(0))
        self.assertEqual(channel, before)

    def test_sparse_grid_bridges_preserve_history_without_fabricating_coverage(self):
        settings = Settings(min_hz=20, initial_points=4, max_points=32)
        channel = new_channel(settings)
        original_frequencies = channel["frequencies"].copy()
        a, b = original_frequencies[0], original_frequencies[-1]
        add_observation(channel, settings, a, b, 0, "manual")
        channel["blocked"].append([original_frequencies[0], original_frequencies[1]])
        before = copy.deepcopy(channel)
        self.assertTrue(prepare_local_grid(channel, settings))
        self.assertTrue(set(original_frequencies).issubset(channel["frequencies"]))
        self.assertLessEqual(len(channel["frequencies"]), settings.max_points)
        for key in ("observations", "blocked", "inputs"):
            self.assertEqual(channel[key], before[key])
        self.assertFalse(fit_channel(channel, settings).connected)
        self.assertFalse(prepare_local_grid(channel, settings))
        # Independently demonstrate that the new grid supports actual local
        # comparisons, including cross-checks, without relying on the old link.
        local_only = copy.deepcopy(channel)
        local_only["observations"], local_only["inputs"] = [], 0
        frequencies = local_only["frequencies"]
        for i, a in enumerate(frequencies):
            for b in frequencies[i + 1:]:
                if b / a <= 2 + 1e-12 and [a, b] not in local_only["blocked"]:
                    add_observation(local_only, settings, a, b, 0, "manual")
        fit = fit_channel(local_only, settings)
        self.assertTrue(fit.connected)
        self.assertGreaterEqual(fit.cycles, 2)
        self.assertTrue(np.all(fit.degrees >= 2))

    def test_point_cap_failure_keeps_entire_channel_unchanged(self):
        settings = Settings(min_hz=20, initial_points=4, max_points=5)
        channel = new_channel(settings)
        add_observation(channel, settings, *channel["frequencies"][:2], 0, "manual")
        before = copy.deepcopy(channel)
        with self.assertRaisesRegex(ValueError, "Maximum frequency points is 5"):
            refine(channel, settings)
        self.assertEqual(channel, before)

    def test_interior_chain_links_gain_nearby_cross_checks(self):
        settings = Settings(min_hz=1000, max_hz=8000, initial_points=8, max_points=16)
        channel = {"frequencies": [1000., 1200., 1500., 2400., 4000., 6000., 7000., 8000.],
                   "observations": [], "blocked": [], "inputs": 0}
        self.assertTrue(prepare_local_grid(channel, settings))
        frequencies = channel["frequencies"]
        for i, a in enumerate(frequencies):
            for b in frequencies[i + 1:]:
                if b / a <= 2 + 1e-12:
                    self.assertTrue(any(f != a and f != b
                                        and max(a, f) / min(a, f) <= 2 + 1e-12
                                        and max(b, f) / min(b, f) <= 2 + 1e-12
                                        for f in frequencies), f"No local cross-check for {a}, {b}")

    def test_single_local_cycle_needs_another_cross_check(self):
        settings = Settings(min_hz=1000, max_hz=1800, initial_points=4, max_points=5)
        channel = {"frequencies": [1000., 1200., 1500., 1800.], "observations": [],
                   "blocked": [[1000., 1500.], [1200., 1800.]], "inputs": 0}
        for a, b in ((1000, 1200), (1200, 1500), (1500, 1800), (1000, 1800)):
            add_observation(channel, settings, a, b, 0, "manual")
        self.assertEqual(fit_channel(channel, settings).cycles, 1)
        self.assertTrue(prepare_local_grid(channel, settings))
        new = next(f for f in channel["frequencies"] if f not in (1000, 1200, 1500, 1800))
        for f in (1000, 1200, 1500, 1800):
            add_observation(channel, settings, min(f, new), max(f, new), 0, "manual")
        self.assertGreaterEqual(fit_channel(channel, settings).cycles, 2)

    def test_unreachable_existing_link_is_checked_until_it_can_be_censored(self):
        settings = Settings(min_hz=8000, initial_points=4, max_points=4)
        channel = {"frequencies": [8000., 10000., 12500., 16000.], "observations": [],
                   "blocked": [[8000., 10000.], [8000., 12500.], [10000., 12500.],
                               [10000., 16000.], [12500., 16000.]], "inputs": 0}
        add_observation(channel, settings, 8000, 16000, 17, "guided", "first")
        fit = SimpleNamespace(frequencies=np.array(channel["frequencies"]), matches=np.array([0., 4., 8., 24.]),
                              covariance=np.eye(4), degrees=np.full(4, 2), connected=True,
                              residuals={(8000., 16000.): 8.})
        with patch("phonix.engine.fit_channel", return_value=fit):
            self.assertEqual(select_pair(channel, settings, np.random.default_rng(0)), (8000, 16000))
            for _ in range(5):
                add_observation(channel, settings, 8000, 16000, 17, "guided", "first")
            with self.assertRaisesRegex(ValueError, "within one octave"):
                select_pair(channel, settings, np.random.default_rng(0))

    def test_censored_link_cannot_supply_a_local_cross_check(self):
        settings = Settings()
        channel = new_channel(settings)
        frequencies = channel["frequencies"].copy()
        a, b = frequencies[-3], frequencies[-1]
        for _ in range(6):
            add_observation(channel, settings, a, b, 17, "guided", "first")
        observations = copy.deepcopy(channel["observations"])
        self.assertTrue(prepare_local_grid(channel, settings))
        self.assertTrue(any(frequencies[-2] < f < b for f in channel["frequencies"]))
        self.assertEqual(channel["observations"], observations)
        self.assertEqual(fit_channel(channel, settings).boundary_pairs, 1)
        self.assertFalse(fit_channel(channel, settings).connected)
        self.assertNotEqual(select_pair(channel, settings, np.random.default_rng(0)), (a, b))

    def test_no_fallback_to_a_historical_wide_pair_when_local_pairs_are_blocked(self):
        settings = Settings(**RANGES["full"])
        channel = new_channel(settings)
        a, b = channel["frequencies"][0], channel["frequencies"][-1]
        add_observation(channel, settings, a, b, 0, "manual")
        frequencies = channel["frequencies"]
        channel["blocked"] = [[a, b] for i, a in enumerate(frequencies)
                              for b in frequencies[i + 1:] if b / a <= 2 + 1e-12]
        before = copy.deepcopy(channel)
        with self.assertRaisesRegex(ValueError, "within one octave"):
            select_pair(channel, settings, np.random.default_rng(0))
        self.assertEqual(channel, before)

    def test_resume_saves_bridges_and_plays_only_local_pairs_without_changing_history(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            settings = Settings(**RANGES["full"])
            profile = store.create("existing", settings)
            channel = profile["channels"]["both"]
            original_frequencies = channel["frequencies"].copy()
            add_observation(channel, settings, original_frequencies[0], original_frequencies[-1], 0, "manual")
            store.save(profile)
            observations = copy.deepcopy(channel["observations"])
            with patch("phonix.cli.Player") as player, \
                    patch("builtins.input", side_effect=["", "c", "c", "q"]), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                run_profile(store, profile, 0)
            a, b = player.return_value.play.call_args.args[:2]
            self.assertLessEqual(b / a, 2 + 1e-12)
            saved = store.load("existing")
            self.assertEqual(saved["channels"]["both"]["observations"], observations)
            self.assertEqual(saved["channels"]["both"]["inputs"], 1)
            self.assertTrue(set(original_frequencies).issubset(saved["channels"]["both"]["frequencies"]))
            self.assertEqual(saved["settings"], profile["settings"])
            self.assertIn("maximum 1 octave", output.getvalue())

    def test_short_budget_remains_incomplete_without_changing_tolerance_or_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            settings = Settings(**(RANGES["full"] | dict(budget=4, shape_tolerance=3)))
            profile = store.create("short", settings)
            before = copy.deepcopy(profile["settings"])
            with patch("phonix.cli.Player") as player, \
                    patch("builtins.input", side_effect=["", "c", "c", "e", "e", "e", "e"]), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                run_profile(store, profile, 0)
            self.assertEqual(player.return_value.play.call_count, 4)
            for call in player.return_value.play.call_args_list:
                a, b = call.args[:2]
                self.assertLessEqual(b / a, 2 + 1e-12)
            saved = store.load("short")
            self.assertEqual(saved["settings"], before)
            channel = saved["channels"]["both"]
            self.assertEqual(channel["inputs"], 4)
            self.assertFalse(fit_channel(channel, settings).ready(settings))
            self.assertIn("judgment budget reached", output.getvalue())


if __name__ == "__main__":
    unittest.main()
