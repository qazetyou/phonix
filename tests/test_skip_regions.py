"""Explicitly uncalibrated shapes with measured regional gains; silent audio."""

import contextlib
import copy
from dataclasses import replace
import io
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from phonix.cli import (RANGES, build_preset, choose_local_calibration, compare,
                        describe_quality, profile_summary, run_profile)
from phonix.engine import add_observation, fit_channel, new_channel, refine
from phonix.model import Reference
from phonix.profiles import Settings, Store
from phonix.regional import current_observations, reopen_region, skip_shapes
from phonix.workflow import next_comparison, ready, regions


class SkipRegionTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(**(RANGES["full"] | dict(calibration_method="regional-v2",
                                                       shape_tolerance=3, budget=300, max_level=-18)))
        self.channel = new_channel(self.settings)
        refine(self.channel, self.settings)
        self.reference = Reference()
        baseline = self.reference.expected_matches(self.channel["frequencies"])
        self.truth = {f: float(value + (3 if f < 250 else 0 if f < 2000 else -2))
                      for f, value in zip(self.channel["frequencies"], baseline)}
        self.rng = np.random.default_rng(32)

    def record(self, planned):
        a, b, order = planned
        add_observation(self.channel, self.settings, a, b, self.truth[b] - self.truth[a], "manual")
        self.channel["observations"][-1]["reversed_order"] = bool(order)

    def finish(self):
        for _ in range(150):
            planned = next_comparison(self.channel, self.settings, self.rng)
            if planned is None:
                return
            self.record(planned)
        self.fail("Bounded simulation did not complete")

    def profile(self, store, name="skip"):
        profile = store.create(name, self.settings)
        profile["channels"]["both"] = copy.deepcopy(self.channel)
        store.save(profile)
        return profile

    def test_skip_all_probes_only_boundaries_and_requires_measured_gains(self):
        skip_shapes(self.channel, self.settings)
        self.assertEqual(self.channel["inputs"], 0)
        before = fit_channel(self.channel, self.settings)
        self.assertFalse(before.ready(self.settings))
        self.assertFalse(before.regional["exportable"])
        self.finish()
        fit = fit_channel(self.channel, self.settings)
        self.assertTrue(ready(self.channel, self.settings, fit))
        self.assertFalse(fit.connected)  # Full local graph is still unmeasured.
        self.assertTrue(fit.regional["exportable"])
        self.assertGreater(fit.shape95, self.settings.shape_tolerance)
        self.assertLess(fit.regional["balance_uncertainty_95_db"], self.settings.shape_tolerance)
        membership = {f: n for n, fs in regions(self.channel).items() for f in fs}
        self.assertTrue(all(membership[o["a"]] != membership[o["b"]] for o in self.channel["observations"]))
        correction = self.reference.correction(fit.frequencies, fit.matches)
        for fs in regions(self.channel).values():
            ix = [self.channel["frequencies"].index(f) for f in fs]
            self.assertLess(np.ptp(correction[ix]), 1e-10)
            info = fit.regional["regions"][membership[fs[0]]]
            self.assertIsNone(info["shape_uncertainty_95_db"])
            self.assertFalse(info["shape_checks_met"])

    def test_skip_one_keeps_evidence_but_applies_no_partial_shape_correction(self):
        planned = next_comparison(self.channel, self.settings, self.rng)
        self.record(planned)
        history = copy.deepcopy(self.channel["observations"])
        skip_shapes(self.channel, self.settings, "midrange")
        self.assertEqual(self.channel["observations"], history)
        self.assertEqual(current_observations(self.channel), [])
        self.finish()
        fit = fit_channel(self.channel, self.settings)
        self.assertTrue(ready(self.channel, self.settings, fit))
        self.assertEqual(fit.regional["skipped_regions"], ["midrange"])
        self.assertEqual(self.channel["observations"][:1], history)
        correction = self.reference.correction(fit.frequencies, fit.matches)
        fs = regions(self.channel)["midrange"]
        self.assertLess(np.ptp(correction[[self.channel["frequencies"].index(f) for f in fs]]), 1e-10)
        self.assertTrue(fit.regional["regions"]["bass"]["shape_checks_met"])

    def test_revisit_skipped_region_keeps_other_skips_and_repeats_balance(self):
        skip_shapes(self.channel, self.settings)
        self.finish()
        history = copy.deepcopy(self.channel["observations"])
        reopen_region(self.channel, self.settings, "bass")
        self.assertEqual(set(self.channel["workflow"]["skipped"]), {"midrange", "treble"})
        planned = next_comparison(self.channel, self.settings, self.rng)
        self.assertTrue(all(f < 250 for f in planned[:2]))
        self.assertFalse(ready(self.channel, self.settings))
        self.record(planned)
        self.finish()
        self.assertEqual(self.channel["observations"][:len(history)], history)
        self.assertGreater(self.channel["workflow"]["balance_at"], len(history))
        self.assertTrue(ready(self.channel, self.settings))

    def test_unmeasured_skipped_balance_cannot_export_even_with_incomplete_flag(self):
        skip_shapes(self.channel, self.settings)
        with tempfile.TemporaryDirectory() as directory:
            profile = self.profile(Store(directory))
            with self.assertRaisesRegex(ValueError, "unmeasured priors"):
                build_preset(profile, allow_incomplete=True)

    def test_export_and_status_explicitly_mark_skipped_shapes(self):
        skip_shapes(self.channel, self.settings)
        self.finish()
        with tempfile.TemporaryDirectory() as directory:
            profile = self.profile(Store(directory))
            summary = profile_summary(profile)
            self.assertEqual(summary["status"], "complete")
            self.assertIn("shape skipped", summary["row"][1])
            self.assertTrue(summary["connected"])
            with contextlib.redirect_stdout(io.StringIO()):
                preset, report = build_preset(profile, count=4)
            data = report["channels"]["both"]
            self.assertEqual(data["skipped_shapes_uncalibrated"], ["bass", "midrange", "treble"])
            self.assertIn("no local correction", data["correction_scope"])
            self.assertFalse(data["consistency"]["connected"])
            self.assertIn("equalizer#0", preset["output"])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                describe_quality(self.channel, self.settings, self.reference)
            self.assertIn("Shape uncalibrated", output.getvalue())
            self.assertIn("Conditional balance uncertainty", output.getvalue())

    def test_choice_cancel_and_resume_are_saved_without_judgments(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = self.profile(store)
            original = store.path("skip").read_bytes()
            with patch("phonix.cli.select_table", return_value=None):
                self.assertFalse(choose_local_calibration(store, profile, "both", self.settings))
            self.assertEqual(store.path("skip").read_bytes(), original)
            with patch("phonix.cli.select_table", return_value=1):
                self.assertTrue(choose_local_calibration(store, profile, "both", self.settings))
            saved = store.load("skip")["channels"]["both"]
            self.assertEqual(saved["inputs"], 0)
            self.assertEqual(saved["workflow"]["stage"], "balance")
            expected = next_comparison(copy.deepcopy(saved), self.settings, np.random.default_rng(8))
            actual = next_comparison(store.load("skip")["channels"]["both"], self.settings, np.random.default_rng(8))
            self.assertEqual(actual, expected)

    def test_region_key_requires_confirmation_and_cancellation_replays_nothing(self):
        planned = next_comparison(self.channel, self.settings, self.rng)
        original = copy.deepcopy(self.channel)
        with patch("phonix.cli.Player") as player, patch("phonix.cli._input", side_effect=["k", "k"]), \
                patch("phonix.cli._menu_confirm", side_effect=[False, True]) as confirm, \
                contextlib.redirect_stdout(io.StringIO()):
            result, mode = compare(*planned[:2], "both", self.channel, self.settings, player,
                                   self.reference, self.rng, "guided", order=False)
        self.assertEqual(result, "skip-region")
        self.assertEqual(confirm.call_count, 2)
        self.assertEqual(player.play.call_count, 1)
        self.assertEqual(self.channel, original)

    def test_run_saves_skip_without_recording_judgment_or_blocking_pair(self):
        next_comparison(self.channel, self.settings, self.rng)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = self.profile(store)
            with patch("phonix.cli.Player"), patch("phonix.cli.prepare_volume", return_value=self.settings), \
                    patch("phonix.cli.compare", side_effect=[("skip-region", "guided"), ("quit", "guided")]), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_profile(store, profile, 0)
            saved = store.load("skip")["channels"]["both"]
            self.assertEqual(saved["workflow"]["skipped"], {"midrange": 0})
            self.assertEqual(saved["inputs"], 0)
            self.assertEqual(saved["blocked"], [])

    def test_single_region_and_legacy_reject_skip_atomically(self):
        for cfg in (replace(self.settings, min_hz=2000), replace(self.settings, calibration_method="global-v1")):
            channel = new_channel(cfg)
            before = copy.deepcopy(channel)
            with self.assertRaises(ValueError):
                skip_shapes(channel, cfg)
            self.assertEqual(channel, before)

    def test_malformed_skipped_checkpoints_rejected_and_reset_clears_them(self):
        skip_shapes(self.channel, self.settings)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = self.profile(store)
            for bad in ({"bass": True}, {"bass": 1}, {"unknown": 0}, ["bass"]):
                profile["channels"]["both"]["workflow"]["skipped"] = bad
                store.save(profile)
                with self.assertRaisesRegex(ValueError, "skipped regional"):
                    store.load("skip")
            profile["channels"]["both"] = copy.deepcopy(self.channel)
            store.save(profile)
            self.assertNotIn("workflow", store.reset("skip")["channels"]["both"])


if __name__ == "__main__":
    unittest.main()
