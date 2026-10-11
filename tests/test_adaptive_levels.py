"""Frequency-dependent stimulus levels and gain accounting; no live audio."""

import contextlib
import copy
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from phonix.audio import Player, comparison_levels, pair_samples, tone
from phonix.cli import compare, main, run_profile
from phonix.engine import (add_observation, fit_channel, new_channel,
                           reference_base_level, validate_channel)
from phonix.model import Reference
from phonix.profiles import Settings, Store
from phonix.ui import create_profile


class AdaptiveLevelTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(min_hz=60, max_hz=15000, initial_points=17,
                                 max_points=32, tone_leveling="adaptive-v1")
        self.reference = Reference()
        self.channel = new_channel(self.settings)

    def test_starting_levels_follow_prior_around_saved_1k_anchor(self):
        fit = fit_channel(self.channel, self.settings, self.reference)
        self.assertEqual(reference_base_level(1000, fit, self.settings, self.reference), -42)
        for hz in (60, 1000, 15000):
            m = self.reference.expected_matches([1000, hz])
            self.assertAlmostEqual(reference_base_level(hz, fit, self.settings, self.reference),
                                   -42 + m[1] - m[0])
        self.assertGreater(reference_base_level(60, fit, self.settings, self.reference), -42)

    def test_learned_shape_updates_levels_without_changing_gauge(self):
        fit = fit_channel(self.channel, self.settings, self.reference)
        fit.matches = np.log2(fit.frequencies / 1000) * 3
        self.assertAlmostEqual(reference_base_level(60, fit, self.settings, self.reference),
                               -42 + 3 * np.log2(60 / 1000))
        before = reference_base_level(60, fit, self.settings, self.reference)
        fit.matches += 91
        self.assertAlmostEqual(reference_base_level(60, fit, self.settings, self.reference), before)

    def test_ranges_without_1k_use_baseline_anchor_and_remain_gauge_invariant(self):
        for limits in ((2000, 16000), (20, 200)):
            cfg = replace(self.settings, min_hz=limits[0], max_hz=limits[1])
            fit = fit_channel(new_channel(cfg), cfg, self.reference)
            hz = fit.frequencies[0]
            prior = self.reference.expected_matches([1000, hz])
            before = reference_base_level(hz, fit, cfg, self.reference)
            self.assertAlmostEqual(before, cfg.reference_level + prior[1] - prior[0])
            fit.matches += 70
            self.assertAlmostEqual(reference_base_level(hz, fit, cfg, self.reference), before)

    def test_common_shift_preserves_difference_for_all_limits(self):
        for reference, ceiling in ((-60, -54), (-42, -24), (-9, -3)):
            cfg = replace(self.settings, reference_level=reference, max_level=ceiling).validate()
            for desired in (-100, -78, -50, -30, -3, 40):
                for offset in np.linspace(-cfg.offset_limit, cfg.offset_limit, 101):
                    a, b = comparison_levels(offset, cfg, desired)
                    self.assertTrue(-78 <= a <= ceiling and -78 <= b <= ceiling)
                    self.assertAlmostEqual(b - a, offset)
            for bad in (float("nan"), float("inf"), True):
                with self.assertRaises(ValueError):
                    comparison_levels(0, cfg, bad)

    def test_actual_samples_keep_pure_tones_duration_channels_and_order(self):
        for rate in (44100, 48000, 96000):
            cfg = replace(self.settings, sample_rate=rate)
            a, b, offset, desired = 20, 40, -5.25, -10
            levels = comparison_levels(offset, cfg, desired)
            duration = 0.75
            silence = np.zeros((int(rate * .25), 2))
            for channel in ("left", "right", "both"):
                waves = [tone(hz, level, channel, rate, duration, cfg.max_level)
                         for hz, level in zip((a, b), levels)]
                for order in (False, True):
                    first, second = waves[::-1] if order else waves
                    expected = np.vstack([first, silence, second, silence])
                    actual = pair_samples(a, b, offset, order, channel, cfg, desired)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertLessEqual(abs(actual).max(), 10 ** (cfg.max_level / 20))

    def test_player_uses_the_same_calculated_levels_without_live_audio(self):
        with patch("phonix.audio.shutil.which", return_value="pw-play"):
            player = Player(self.settings)
        with patch.object(player, "_play_samples") as output:
            player.play(60, 120, -3, True, "right", base_level=-30)
        np.testing.assert_array_equal(output.call_args.args[0],
                                      pair_samples(60, 120, -3, True, "right", self.settings, -30))

    def test_metadata_is_actual_canonical_gain_even_when_reversed_and_limited(self):
        a, b = self.channel["frequencies"][-2:]
        player = Mock()
        with patch("phonix.cli._input", side_effect=["r", "1"]), \
                contextlib.redirect_stdout(io.StringIO()):
            observation, mode = compare(a, b, "right", self.channel, self.settings, player,
                                        self.reference, np.random.default_rng(3), "guided", order=True)
        self.assertEqual(observation[2], "second")
        self.assertEqual(mode, "guided")
        self.assertEqual(player.play.call_args_list[0], player.play.call_args_list[1])
        call = player.play.call_args
        peaks = comparison_levels(call.args[2], self.settings, call.kwargs["base_level"])
        self.assertEqual((observation[3]["a_dbfs"], observation[3]["b_dbfs"]), peaks)
        self.assertAlmostEqual(peaks[1] - peaks[0], observation[0])

    def test_manual_adjustments_and_help_freeze_prediction_and_store_final_peaks(self):
        a, b = self.channel["frequencies"][:2]
        original = copy.deepcopy(self.channel)
        player = Mock()
        with patch("phonix.cli._input", side_effect=["+", "h", "", "r", "e"]), \
                contextlib.redirect_stdout(io.StringIO()):
            observation, _ = compare(a, b, "both", self.channel, self.settings, player,
                                      self.reference, np.random.default_rng(1), "manual", order=False)
        calls = player.play.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertAlmostEqual(calls[1].args[2] - calls[0].args[2], .5)
        self.assertEqual(calls[1], calls[2])
        self.assertEqual(len({call.kwargs["base_level"] for call in calls}), 1)
        self.assertEqual(self.channel, original)
        add_observation(self.channel, self.settings, a, b, *observation)
        validate_channel(self.channel, self.settings)
        self.assertEqual(self.channel["observations"][0]["playback"], observation[3])

    def test_gain_metadata_does_not_inject_compensation_into_fit_or_eq(self):
        legacy = copy.deepcopy(self.channel)
        for a, b in zip(self.channel["frequencies"], self.channel["frequencies"][1:]):
            for offset in (1, 1.5):
                peaks = comparison_levels(offset, self.settings, -10)
                add_observation(self.channel, self.settings, a, b, offset, "manual", playback={
                    "method": "adaptive-v1", "a_dbfs": peaks[0], "b_dbfs": peaks[1]})
                add_observation(legacy, self.settings, a, b, offset, "manual")
        left, right = [fit_channel(c, self.settings, self.reference) for c in (self.channel, legacy)]
        np.testing.assert_array_equal(left.matches, right.matches)
        np.testing.assert_array_equal(left.covariance, right.covariance)
        np.testing.assert_array_equal(self.reference.correction(left.frequencies, left.matches),
                                      self.reference.correction(right.frequencies, right.matches))

    def test_bad_metadata_rejected_on_append_and_load_without_partial_judgment(self):
        a, b = self.channel["frequencies"][:2]
        good = {"method": "adaptive-v1", "a_dbfs": -40, "b_dbfs": -39}
        variants = [None, {}, good | {"method": "other"}, good | {"a_dbfs": True},
                    good | {"a_dbfs": float("nan")}, good | {"b_dbfs": -20},
                    good | {"a_dbfs": -79}, good | {"b_dbfs": -38}]
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = store.create("invalid", self.settings)
            saved = profile["channels"]["both"]
            add_observation(saved, self.settings, a, b, 1, "manual", playback=good)
            for invalid in variants:
                if invalid is not None:
                    with self.assertRaises(ValueError):
                        add_observation(self.channel, self.settings, a, b, 1, "manual", playback=invalid)
                    self.assertEqual(self.channel["inputs"], 0)
                saved["observations"][0]["playback"] = invalid
                # NaN cannot be written by Store; use finite malformed metadata there.
                if invalid is not None and isinstance(invalid, dict) and invalid.get("a_dbfs") != invalid.get("a_dbfs"):
                    continue
                store.save(profile)
                with self.assertRaises(ValueError):
                    store.load("invalid")
        with self.assertRaises(ValueError):
            replace(self.settings, tone_leveling="unknown").validate()

    def test_new_cli_ui_profiles_opt_in_and_legacy_resume_retains_original_method(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory))
            with patch("phonix.cli.sys.stdin.isatty", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--data-dir", directory, "new", "CLI"]), 0)
                with patch("builtins.input", side_effect=["UI"]), \
                        patch("phonix.ui.select_table", side_effect=lambda title, headers, rows, *args, **kw:
                              next(i for i, row in enumerate(rows) if row[0] == "Create profile")):
                    create_profile(store)
            for name in ("CLI", "UI"):
                self.assertEqual(store.load(name)["settings"]["tone_leveling"], "adaptive-v1")
            cfg = replace(self.settings, tone_leveling="fixed", budget=2)
            profile = store.create("legacy", cfg)
            del profile["settings"]["tone_leveling"]
            channel = profile["channels"]["both"]
            a, b = channel["frequencies"][:2]
            add_observation(channel, cfg, a, b, 1, "manual")
            store.save(profile)
            old = copy.deepcopy(channel["observations"])
            with patch("phonix.cli.Player") as factory, \
                    patch("builtins.input", side_effect=["", "c", "c", "e"]), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_profile(store, store.load("legacy"), 0)
            self.assertFalse(factory.return_value.play.call_args.kwargs)
            resumed = store.load("legacy")["channels"]["both"]
            self.assertEqual(resumed["observations"][:1], old)
            self.assertNotIn("playback", resumed["observations"][-1])

    def test_run_saves_gain_metadata_before_resuming(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = store.create("adaptive", replace(self.settings, budget=1))
            with patch("phonix.cli.Player") as factory, \
                    patch("builtins.input", side_effect=["", "c", "c", "e"]), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_profile(store, profile, 0)
            cfg = factory.call_args.args[0]
            call = factory.return_value.play.call_args
            a_level, b_level = comparison_levels(call.args[2], cfg, call.kwargs["base_level"])
            saved = store.load("adaptive")["channels"]["both"]["observations"][0]
            self.assertEqual(saved["playback"], {"method": "adaptive-v1", "a_dbfs": a_level, "b_dbfs": b_level})
            self.assertAlmostEqual(saved["offset"], b_level - a_level)


if __name__ == "__main__":
    unittest.main()
