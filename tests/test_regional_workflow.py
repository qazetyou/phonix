"""Regional scheduling, shared gain recovery and resumable silent sessions."""

import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from phonix.cli import RANGES, compare, export_profile, profile_summary, quality, run_profile
from phonix.engine import add_observation, fit_channel, new_channel, pairs_for, prepare_local_grid, refine
from phonix.model import Reference
from phonix.profiles import Settings, Store
from phonix.workflow import (next_comparison, progress, ready, regions,
                                       start_workflow, verified)


def listener(frequency):
    # A deliberately different whole-region level, not just a local bump.
    return 5.0 if frequency < 250 else 0.0 if frequency < 2000 else -4.0


def record(channel, settings, planned, rng):
    a, b, order = planned
    add_observation(channel, settings, a, b, listener(b) - listener(a), "manual")
    channel["observations"][-1]["reversed_order"] = bool(rng.integers(0, 2)) if order is None else order


class RegionalWorkflowTests(unittest.TestCase):
    def full(self):
        # Preserve coverage of legacy/wider profiles as creation defaults change.
        settings = Settings(**(RANGES["full"] | {"min_hz": 20, "max_hz": 16000,
                                                "shape_tolerance": 3, "budget": 150}))
        channel = new_channel(settings)
        prepare_local_grid(channel, settings)
        return settings, channel

    def test_pass_order_local_pairs_and_shared_regional_level_recovery(self):
        settings, channel = self.full()
        rng = np.random.default_rng(51)
        passes, prompts = [], []
        for _ in range(settings.budget):
            planned = next_comparison(channel, settings, rng)
            stage = channel["workflow"]["stage"]
            if not passes or passes[-1] != stage:
                passes.append(stage)
            if planned is None:
                break
            a, b, order = planned
            self.assertLessEqual(b / a, 2 + 1e-12)
            if stage in ("midrange", "bass", "treble"):
                self.assertIn(a, regions(channel)[stage])
                self.assertIn(b, regions(channel)[stage])
            # Readiness may end an extra refinement block before its reverse
            # repeat. Mandatory closing checks must still complete both orders.
            prompts.append((stage, (a, b), order if "refinement_at" not in channel["workflow"] else None))
            record(channel, settings, planned, rng)
        self.assertEqual(passes, ["early", "midrange", "bass", "treble", "balance", "mixed"])
        self.assertTrue(ready(channel, settings))
        self.assertTrue(verified(channel, settings))
        fit = fit_channel(channel, settings)
        truth = np.array([listener(f) for f in fit.frequencies])
        self.assertLess(np.std(fit.matches - truth), 0.1)
        for stage in ("balance", "mixed"):
            keys = {key for actual, key, order in prompts if actual == stage and order is not None}
            for key in keys:
                self.assertEqual({order for actual, pair, order in prompts
                                  if actual == stage and pair == key}, {False, True})
        balance = {key for stage, key, _ in prompts if stage == "balance"}
        self.assertEqual(len(balance), 4)  # Two independent links at each boundary.

    def test_resume_repeats_opposite_order_without_changing_history(self):
        settings, channel = self.full()
        rng = np.random.default_rng(18)
        while True:
            planned = next_comparison(channel, settings, rng)
            record(channel, settings, planned, rng)
            if channel["workflow"]["stage"] == "balance":
                break
        before = copy.deepcopy(channel)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = store.create("resume", settings)
            profile["channels"]["both"] = channel
            store.save(profile)
            resumed = store.load("resume")["channels"]["both"]
            expected = next_comparison(copy.deepcopy(before), settings, np.random.default_rng(99))
            following = next_comparison(resumed, settings, np.random.default_rng(99))
        self.assertEqual(following, expected)
        self.assertEqual(resumed["observations"], before["observations"])
        self.assertEqual(resumed["inputs"], before["inputs"])
        self.assertEqual(resumed["workflow"], before["workflow"])
        for _ in range(4):
            if following[:2] == planned[:2]:
                break
            record(resumed, settings, following, rng)
            following = next_comparison(resumed, settings, rng)
        self.assertEqual(following, (planned[0], planned[1], True))

    def test_noisy_guided_regional_session_preserves_global_precision_checks(self):
        settings = Settings(**(RANGES["full"] | {"shape_tolerance": 3, "budget": 300}))
        channel = new_channel(settings)
        rng = np.random.default_rng(51)
        for _ in range(settings.budget):
            refine(channel, settings)
            planned = next_comparison(channel, settings, rng)
            if planned is None:
                break
            a, b, order = planned
            self.assertLessEqual(b / a, 2 + 1e-12)
            fit = fit_channel(channel, settings)
            pair = pairs_for(channel, settings).get((a, b))
            offset = (pair.next_offset() if pair else float(np.clip(
                fit.matches[channel["frequencies"].index(b)] - fit.matches[channel["frequencies"].index(a)],
                -settings.offset_limit, settings.offset_limit)))
            evidence = offset - (listener(b) - listener(a)) + rng.logistic(0, 1.5)
            answer = "second" if evidence > 0.75 else "first" if evidence < -0.75 else "equal"
            add_observation(channel, settings, a, b, offset, "guided", answer)
            channel["observations"][-1]["reversed_order"] = bool(rng.integers(0, 2)) if order is None else order
        fit = fit_channel(channel, settings)
        self.assertTrue(ready(channel, settings, fit))
        self.assertLess(channel["inputs"], settings.budget)
        self.assertGreaterEqual(fit.cycles, 2)
        self.assertTrue(np.all(fit.degrees >= 2))
        self.assertLessEqual(fit.shape95, settings.shape_tolerance)
        self.assertLessEqual(fit.drift_rms, settings.drift_tolerance)
        self.assertLess(np.std(fit.matches - [listener(f) for f in fit.frequencies]), 3)

    def test_single_region_omits_boundary_and_other_shape_passes(self):
        for range_name in ("treble",):
            settings = Settings(**RANGES[range_name])
            channel = new_channel(settings)
            planned = next_comparison(channel, settings, np.random.default_rng(2))
            self.assertEqual(channel["workflow"]["stage"], "treble")
            self.assertIn("Treble shape", progress(channel, settings))
            self.assertGreaterEqual(planned[0], settings.min_hz)
        settings = Settings(min_hz=300, max_hz=2000, initial_points=7)
        channel = new_channel(settings)
        self.assertEqual(set(regions(channel)), {"midrange"})
        next_comparison(channel, settings, np.random.default_rng(2))
        self.assertEqual(channel["workflow"]["stage"], "midrange")

    def test_old_wide_observation_is_retained_but_not_a_new_boundary_check(self):
        settings, channel = self.full()
        add_observation(channel, settings, 20, 16000, 0, "manual")
        old = copy.deepcopy(channel["observations"])
        planned = next_comparison(channel, settings, np.random.default_rng(2))
        self.assertEqual(channel["workflow"]["stage"], "early")
        self.assertEqual(channel["workflow"]["started_at"], 1)
        self.assertLessEqual(planned[1] / planned[0], 2 + 1e-12)
        self.assertEqual(channel["observations"], old)

    def test_skipped_boundary_uses_alternative_local_link(self):
        settings, channel = self.full()
        rng = np.random.default_rng(3)
        planned = next_comparison(channel, settings, rng)
        channel["blocked"].append(list(planned[:2]))
        replacement = next_comparison(channel, settings, rng)
        self.assertNotEqual(replacement[:2], planned[:2])
        self.assertLessEqual(replacement[1] / replacement[0], 2 + 1e-12)
        self.assertEqual(channel["inputs"], 0)
        self.assertEqual(channel["workflow"]["stage"], "early")

    def test_boundary_level_guard_uses_another_local_boundary_link(self):
        from types import SimpleNamespace

        settings = Settings(min_hz=150, max_hz=450, initial_points=6, max_points=6)
        channel = {"frequencies": [150., 200., 240., 300., 380., 450.],
                   "observations": [], "blocked": [[240., 300.]], "inputs": 0}
        fit = SimpleNamespace(frequencies=np.array(channel["frequencies"]),
                              matches=np.array([0., 0., 0., 20., 0., 0.]),
                              covariance=np.eye(6), degrees=np.full(6, 2), connected=True)
        with patch("phonix.engine.fit_channel", return_value=fit):
            a, b, _ = next_comparison(channel, settings, np.random.default_rng(4))
        self.assertLess(a, 250)
        self.assertGreaterEqual(b, 250)
        self.assertNotEqual((a, b), (200., 300.))  # Closest unblocked pair exceeds the level guard.
        self.assertNotEqual((a, b), (240., 300.))
        self.assertLessEqual(b / a, 2 + 1e-12)

    def test_regional_coverage_does_not_require_unplayable_new_links(self):
        settings = Settings(min_hz=20, max_hz=30, initial_points=4)
        channel = {"frequencies": [20., 23., 27., 30.], "observations": [], "blocked": [], "inputs": 0}
        for a, b in zip(channel["frequencies"], channel["frequencies"][1:]):
            add_observation(channel, settings, a, b, 10, "manual")
        start_workflow(channel)
        channel["workflow"]["stage"] = "bass"
        a, b, _ = next_comparison(channel, settings, np.random.default_rng(4))
        self.assertEqual(channel["workflow"]["stage"], "mixed")
        self.assertIn((a, b), {(20., 23.), (23., 27.), (27., 30.)})
        self.assertLessEqual(b / a, 2 + 1e-12)
        self.assertFalse(ready(channel, settings))

    def test_all_boundary_links_skipped_does_not_fall_back_or_claim_completion(self):
        settings, channel = self.full()
        bass = regions(channel)["bass"]
        mid = regions(channel)["midrange"]
        channel["blocked"] = [[a, b] for a in bass for b in mid if b / a <= 2 + 1e-12]
        with self.assertRaisesRegex(ValueError, "regional pass is incomplete"):
            next_comparison(channel, settings, np.random.default_rng(4))
        self.assertFalse(ready(channel, settings))
        self.assertEqual(channel["inputs"], 0)

    def test_same_order_repetitions_cannot_replace_reverse_check(self):
        settings = Settings(min_hz=3000, max_hz=5000, initial_points=4, max_points=4)
        channel = new_channel(settings)
        start_workflow(channel)
        channel["workflow"]["stage"] = "mixed"
        rng = np.random.default_rng(4)
        first = next_comparison(channel, settings, rng)
        for _ in range(4):
            record(channel, settings, (first[0], first[1], False), rng)
        next_pair = next_comparison(channel, settings, rng)
        self.assertEqual(next_pair, (first[0], first[1], True))
        self.assertFalse(verified(channel, settings))
        record(channel, settings, next_pair, rng)
        self.assertTrue(verified(channel, settings))
        self.assertFalse(ready(channel, settings))  # Verification alone is insufficient.

    def test_censored_pair_in_both_orders_cannot_supply_verification(self):
        settings = Settings(min_hz=3000, max_hz=5000, initial_points=4, max_points=4)
        channel = new_channel(settings)
        start_workflow(channel)
        channel["workflow"]["stage"] = "mixed"
        a, b, _ = next_comparison(channel, settings, np.random.default_rng(4))
        for index in range(6):
            add_observation(channel, settings, a, b, 17, "guided", "first")
            channel["observations"][-1]["reversed_order"] = bool(index % 2)
        self.assertTrue(pairs_for(channel, settings)[a, b].boundary)
        self.assertFalse(verified(channel, settings))
        replacement = next_comparison(channel, settings, np.random.default_rng(4))
        self.assertNotEqual(replacement[:2], (a, b))

    def test_fit_readiness_does_not_bypass_passes_or_export_guard(self):
        settings = Settings(min_hz=3000, max_hz=5000, initial_points=4, max_points=4)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = store.create("coverage", settings)
            channel = profile["channels"]["both"]
            for i, a in enumerate(channel["frequencies"]):
                for b in channel["frequencies"][i + 1:]:
                    add_observation(channel, settings, a, b, 0, "manual")
            self.assertTrue(fit_channel(channel, settings).ready(settings))
            self.assertEqual(profile_summary(profile)["status"], "complete")  # Legacy compatible.
            start_workflow(channel)
            self.assertEqual(profile_summary(profile)["status"], "unfinished")
            metrics = quality(channel, settings, Reference())
            self.assertTrue(metrics["consistency_thresholds_met"])
            self.assertFalse(metrics["calibration_complete"])
            self.assertFalse(metrics["regional_checks_complete"])
            with self.assertRaisesRegex(ValueError, "regional checks"):
                export_profile(profile, Path(directory) / "preset.json")
            with contextlib.redirect_stdout(io.StringIO()):
                report = export_profile(profile, Path(directory) / "preset.json", count=2, allow_incomplete=True)
            self.assertFalse(report["channels"]["both"]["consistency"]["calibration_complete"])

    def test_run_shows_passes_and_preserves_settings_with_shared_fit(self):
        settings, _ = self.full()
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = store.create("run", settings)
            before = copy.deepcopy(profile["settings"])
            def match(a, b, channel_name, channel, settings, player, reference, rng, mode, order=None):
                return (listener(b) - listener(a), "manual", "equal"), mode
            with patch("phonix.cli.Player"), \
                    patch("phonix.cli.prepare_volume", return_value=settings), \
                    patch("phonix.cli.compare", side_effect=match), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                run_profile(store, profile, 0)
            saved = store.load("run")
            self.assertEqual(saved["settings"], before)
            self.assertEqual(profile_summary(saved)["status"], "complete")
            for text in ("Midrange shape", "Bass shape", "Treble shape", "Regional balance", "Grouped verification"):
                self.assertIn(text, output.getvalue())
            self.assertTrue(all(type(o["reversed_order"]) is bool for o in saved["channels"]["both"]["observations"]))

    def test_cancel_or_interrupt_keeps_stage_checkpoint_and_judgments(self):
        settings = Settings(budget=2)
        for result in ("quit", KeyboardInterrupt()):
            with tempfile.TemporaryDirectory() as directory:
                store = Store(directory)
                profile = store.create("cancel", settings)
                before = store.path("cancel").read_bytes()
                effect = result if isinstance(result, BaseException) else lambda *a, **k: (result, "guided")
                with patch("phonix.cli.Player"), \
                        patch("phonix.cli.prepare_volume", return_value=settings), \
                        patch("phonix.cli.compare", side_effect=effect), \
                        contextlib.redirect_stdout(io.StringIO()):
                    run_profile(store, profile, 0)
                self.assertEqual(store.path("cancel").read_bytes(), before)

    def test_cap_exhaustion_retains_incomplete_pass_and_channel_isolation(self):
        settings = Settings(**(RANGES["full"] | {"budget": 1, "channels": "separate"}))
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = store.create("ears", settings)
            with patch("phonix.cli.Player"), \
                    patch("phonix.cli.prepare_volume", return_value=settings), \
                    patch("phonix.cli.compare", return_value=((0, "manual", "equal"), "manual")), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_profile(store, profile, 0)
            saved = store.load("ears")
            self.assertEqual(profile_summary(saved)["status"], "incomplete")
            for channel in saved["channels"].values():
                self.assertEqual(channel["inputs"], 1)
                self.assertEqual(channel["workflow"]["stage"], "early")
                self.assertFalse(verified(channel, settings))
            saved["channels"]["left"]["workflow"]["stage"] = "bass"
            self.assertEqual(saved["channels"]["right"]["workflow"]["stage"], "early")
            self.assertEqual(saved["settings"]["budget"], 1)

    def test_forced_reverse_order_stores_canonical_response(self):
        settings = Settings()
        channel = new_channel(settings)
        a, b = channel["frequencies"][:2]
        with patch("phonix.cli.Player") as player, \
                patch("phonix.cli._input", return_value="1"), \
                contextlib.redirect_stdout(io.StringIO()):
            observation, mode = compare(a, b, "both", channel, settings, player, Reference(),
                                        np.random.default_rng(3), "guided", order=True)
        self.assertEqual(observation[2], "second")
        self.assertEqual(mode, "guided")
        self.assertIs(player.play.call_args.args[3], True)

    def test_invalid_saved_pass_state_and_playback_order_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = store.create("invalid", Settings())
            for state in ({"version": 2, "stage": "early", "started_at": 0},
                          {"version": 1, "stage": "unknown", "started_at": 0},
                          {"version": 1, "stage": "mixed", "started_at": 1},
                          {"version": 1, "stage": "mixed", "started_at": True}):
                profile["channels"]["both"]["workflow"] = state
                store.save(profile)
                with self.assertRaisesRegex(ValueError, "regional pass state"):
                    store.load("invalid")
            channel = profile["channels"]["both"]
            del channel["workflow"]
            add_observation(channel, Settings(), *channel["frequencies"][:2], 0, "manual")
            channel["observations"][-1]["reversed_order"] = 1
            store.save(profile)
            with self.assertRaisesRegex(ValueError, "playback order"):
                store.load("invalid")


if __name__ == "__main__":
    unittest.main()
