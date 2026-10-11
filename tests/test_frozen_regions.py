"""Frozen shape inference, gain-only balance and resumable UI; silent audio."""

import contextlib
import copy
from dataclasses import replace
import io
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from phonix.cli import RANGES, build_preset, menu_revisit_region, run_profile
from phonix.engine import add_observation, fit_channel, new_channel, pairs_for, refine
from phonix.model import Reference
from phonix.profiles import Settings, Store
from phonix.regional import current_observations, reopen_region
from phonix.workflow import next_comparison, ready, regions, verified


def truth(f):
    return (4 if f < 250 else 0 if f < 2000 else -3) + .8 * np.log2(f / 1000)


class FrozenRegionTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(**(RANGES["full"] | dict(calibration_method="regional-v2", shape_tolerance=3, budget=300)))
        self.channel = new_channel(self.settings)
        refine(self.channel, self.settings)
        self.rng = np.random.default_rng(51)

    def record(self, planned, value=None):
        a, b, order = planned
        add_observation(self.channel, self.settings, a, b, truth(b) - truth(a) if value is None else value, "manual")
        self.channel["observations"][-1]["reversed_order"] = bool(self.rng.integers(0, 2)) if order is None else order

    def until_balance(self):
        for _ in range(150):
            planned = next_comparison(self.channel, self.settings, self.rng)
            if self.channel["workflow"]["stage"] == "balance":
                return planned
            self.record(planned)
        self.fail("Shape passes did not finish within bounded simulation")

    def complete(self):
        stages = []
        for _ in range(150):
            planned = next_comparison(self.channel, self.settings, self.rng)
            stage = self.channel["workflow"]["stage"]
            if not stages or stage != stages[-1]:
                stages.append(stage)
            if planned is None:
                return stages
            self.assertLessEqual(planned[1] / planned[0], 2 + 1e-12)
            if stage in regions(self.channel):
                self.assertTrue(set(planned[:2]) <= set(regions(self.channel)[stage]))
            self.record(planned)
        self.fail("Calibration did not finish within bounded simulation")

    def test_shape_passes_then_balance_verify_recover_offsets(self):
        self.assertEqual(self.complete(), ["midrange", "bass", "treble", "balance", "verify"])
        self.assertTrue(ready(self.channel, self.settings))
        self.assertTrue(verified(self.channel, self.settings))
        fit = fit_channel(self.channel, self.settings)
        self.assertLess(np.std(fit.matches - [truth(f) for f in fit.frequencies]), .01)
        self.assertEqual(fit.regional["anchor"], "midrange")
        self.assertEqual(set(self.channel["workflow"]["frozen"]), {"bass", "midrange", "treble"})

    def test_boundary_evidence_preserves_shape_and_its_difference_covariance(self):
        planned = self.until_balance()
        before = fit_channel(self.channel, self.settings)
        snapshots = copy.deepcopy(self.channel["workflow"]["frozen"])
        for _ in range(8):
            self.record(planned, truth(planned[1]) - truth(planned[0]) + 2)
            planned = next_comparison(self.channel, self.settings, self.rng)
        after = fit_channel(self.channel, self.settings)
        for fs in regions(self.channel).values():
            ix = [self.channel["frequencies"].index(f) for f in fs]
            np.testing.assert_allclose(np.diff(before.matches[ix]), np.diff(after.matches[ix]), atol=1e-12)
            for a, b in zip(ix, ix[1:]):
                vector = np.zeros(len(before.frequencies)); vector[a], vector[b] = -1, 1
                self.assertAlmostEqual(vector @ before.covariance @ vector, vector @ after.covariance @ vector, places=10)
        self.assertEqual(self.channel["workflow"]["frozen"], snapshots)
        self.assertGreater(np.std(after.matches - before.matches), .5)
        self.assertFalse(refine(self.channel, self.settings))

    def test_boundary_only_evidence_cannot_export_prior_shapes(self):
        groups = regions(self.channel)
        for a in groups["bass"][-2:]:
            for b in groups["midrange"][:2]:
                add_observation(self.channel, self.settings, a, b, 0, "manual")
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = store.create("prior", self.settings)
            profile["channels"]["both"] = self.channel
            self.assertFalse(fit_channel(self.channel, self.settings).connected)
            with self.assertRaisesRegex(ValueError, "unmeasured priors"):
                build_preset(profile, allow_incomplete=True)

    def test_conflicting_boundary_checks_flag_review_without_reshaping(self):
        planned = self.until_balance()
        frozen = fit_channel(self.channel, self.settings)
        signed = {}
        seen = {}
        for _ in range(100):
            a, b, order = planned
            category = (a < 250, b < 2000)
            if (a, b) not in signed:
                count = seen.get(category, 0)
                signed[a, b] = 8 if count % 2 == 0 else -8
                seen[category] = count + 1
            self.record(planned, signed[a, b])
            planned = next_comparison(self.channel, self.settings, self.rng)
            if planned is None:
                break
        self.assertEqual(self.channel["workflow"]["stage"], "review")
        self.assertFalse(ready(self.channel, self.settings))
        after = fit_channel(self.channel, self.settings)
        for fs in regions(self.channel).values():
            ix = [self.channel["frequencies"].index(f) for f in fs]
            np.testing.assert_allclose(np.diff(frozen.matches[ix]), np.diff(after.matches[ix]), atol=1e-12)

    def test_reopen_requires_fresh_shape_checks_and_new_boundary_epoch(self):
        self.complete()
        history = copy.deepcopy(self.channel["observations"])
        frozen = copy.deepcopy(self.channel["workflow"]["frozen"])
        reopen_region(self.channel, self.settings, "bass")
        self.assertEqual(self.channel["observations"], history)
        self.assertEqual(set(self.channel["workflow"]["frozen"]), {"midrange", "treble"})
        self.assertTrue(all(o["a"] < 250 and o["b"] < 250 or o["a"] >= 250 and o["b"] >= 250
                            for o in current_observations(self.channel)))
        planned = next_comparison(self.channel, self.settings, self.rng)
        self.assertEqual(self.channel["workflow"]["stage"], "bass")
        self.assertIsNotNone(planned[2])
        self.record(planned)
        self.complete()
        self.assertGreater(self.channel["workflow"]["balance_at"], len(history))
        self.assertEqual(self.channel["workflow"]["frozen"]["midrange"], frozen["midrange"])
        self.assertEqual(self.channel["observations"][:len(history)], history)

    def test_tighter_tolerance_requests_review_of_frozen_uncertainty(self):
        self.complete()
        tighter = replace(self.settings, shape_tolerance=1)
        fit = fit_channel(self.channel, tighter)
        self.assertFalse(fit.ready(tighter))
        self.assertTrue(fit.regional["review_regions"])
        self.assertIsNone(next_comparison(self.channel, tighter, self.rng))
        self.assertEqual(self.channel["workflow"]["stage"], "review")

    def test_resume_checkpoints_and_cancel_do_not_mutate_nested_frozen_state(self):
        self.until_balance()
        frozen = self.channel["workflow"]["frozen"]
        at = frozen.pop("treble")
        del self.channel["workflow"]["balance_at"]
        self.channel["workflow"]["stage"] = "treble"
        self.channel["workflow"]["started_at"] = at
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = store.create("cancel", self.settings)
            profile["channels"]["both"] = self.channel
            store.save(profile)
            original = store.path("cancel").read_bytes()
            with patch("phonix.cli.Player"), patch("phonix.cli.prepare_volume", return_value=self.settings), \
                    patch("phonix.cli._input", side_effect=["q"]), contextlib.redirect_stdout(io.StringIO()):
                run_profile(store, store.load("cancel"), 0)
            self.assertEqual(store.path("cancel").read_bytes(), original)
            loaded = store.load("cancel")["channels"]["both"]
            expected = next_comparison(copy.deepcopy(self.channel), self.settings, np.random.default_rng(9))
            actual = next_comparison(loaded, self.settings, np.random.default_rng(9))
            self.assertEqual(actual, expected)

    def test_completion_offers_sound_choice_and_keeps_calibration(self):
        self.complete()
        evidence = copy.deepcopy(self.channel)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = store.create("complete", self.settings)
            profile["channels"]["both"] = self.channel
            store.save(profile)
            with patch("phonix.cli.Player") as player, \
                    patch("phonix.cli.prepare_volume", return_value=self.settings), \
                    patch("phonix.cli.select_table", return_value=2) as picker, \
                    contextlib.redirect_stdout(io.StringIO()):
                run_profile(store, profile, 0)
            self.assertIn("Step 4/4", picker.call_args.args[0])
            self.assertEqual(store.load("complete")["settings"]["sound_profile"], "warm-tilt")
            self.assertEqual(store.load("complete")["channels"]["both"], evidence)
            player.return_value.play.assert_not_called()

    def test_revisit_menu_preserves_other_ear_and_reset_clears_checkpoints(self):
        self.complete()
        cfg = replace(self.settings, channels="separate", sound_profile="v-shaped")
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = store.create("ears", cfg)
            profile["channels"] = {ear: copy.deepcopy(self.channel) for ear in ("left", "right")}
            store.save(profile)
            with patch("phonix.cli.select_table", return_value=0):
                menu_revisit_region(store, profile)
            saved = store.load("ears")
            self.assertIn("reopened_at", saved["channels"]["left"]["workflow"])
            self.assertEqual(saved["channels"]["right"], profile["channels"]["right"])
            with self.assertRaisesRegex(ValueError, "Frozen regional shapes"):
                store.extend("ears")
            reset = store.reset("ears")
            self.assertEqual(reset["settings"], profile["settings"])
            self.assertTrue(all(not c["inputs"] and "workflow" not in c for c in reset["channels"].values()))

    def test_malformed_checkpoints_and_order_rejected(self):
        self.until_balance()
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = store.create("bad", self.settings)
            profile["channels"]["both"] = self.channel
            state = copy.deepcopy(self.channel["workflow"])
            for invalid in (state | {"version": 1}, state | {"frozen": {"bass": True}},
                            state | {"balance_at": -1}, state | {"stage": "verify"},
                            state | {"frozen": {}}, state | {"reopened_at": self.channel["inputs"] + 1}):
                self.channel["workflow"] = invalid
                store.save(profile)
                with self.assertRaises(ValueError):
                    store.load("bad")
            self.channel["workflow"] = state
            self.channel["observations"][0]["reversed_order"] = 1
            store.save(profile)
            with self.assertRaisesRegex(ValueError, "playback order"):
                store.load("bad")

    def test_single_region_and_custom_ranges_have_no_fake_boundary_requirement(self):
        for limits in ((2000, 16000), (240, 280), (20, 1000)):
            with self.subTest(limits=limits):
                self.settings = replace(self.settings, min_hz=limits[0], max_hz=limits[1], initial_points=7)
                self.channel = new_channel(self.settings)
                refine(self.channel, self.settings)
                self.complete()
                self.assertTrue(ready(self.channel, self.settings))

    def test_grid_capacity_failure_is_atomic(self):
        cfg = replace(self.settings, initial_points=4, max_points=4)
        channel = new_channel(cfg)
        original = copy.deepcopy(channel)
        with self.assertRaises(ValueError):
            refine(channel, cfg)
        self.assertEqual(channel, original)

    def test_geometry_rechecks_regions_when_a_singleton_gains_partners(self):
        self.settings = replace(self.settings, min_hz=190, max_hz=600, initial_points=4)
        self.channel = new_channel(self.settings)
        refine(self.channel, self.settings)
        self.assertFalse(refine(self.channel, self.settings))
        self.assertTrue(all(len(fs) >= 4 for fs in regions(self.channel).values()))
        self.complete()
        self.assertTrue(ready(self.channel, self.settings))

    def test_noisy_guided_session_completes_with_frozen_shapes_and_uncertainty(self):
        for _ in range(self.settings.budget):
            planned = next_comparison(self.channel, self.settings, self.rng)
            if planned is None:
                break
            a, b, order = planned
            pair = pairs_for(self.channel, self.settings).get((a, b))
            fit = fit_channel(self.channel, self.settings)
            i, j = [self.channel["frequencies"].index(hz) for hz in (a, b)]
            offset = pair.next_offset() if pair else float(np.clip(fit.matches[j] - fit.matches[i],
                                                                   -self.settings.offset_limit, self.settings.offset_limit))
            evidence = offset - (truth(b) - truth(a)) + self.rng.logistic(0, 1.5)
            answer = "first" if evidence < -.75 else "second" if evidence > .75 else "equal"
            add_observation(self.channel, self.settings, a, b, offset, "guided", answer)
            self.channel["observations"][-1]["reversed_order"] = bool(self.rng.integers(0, 2)) if order is None else order
        self.assertTrue(ready(self.channel, self.settings))
        fit = fit_channel(self.channel, self.settings)
        self.assertLess(fit.shape95, self.settings.shape_tolerance)
        self.assertLess(np.std(fit.matches - [truth(f) for f in fit.frequencies]), 3)


if __name__ == "__main__":
    unittest.main()
