"""Stay in one comparison range through checks and refinement; silent audio."""

import contextlib
import copy
from dataclasses import replace
import io
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from phonix.cli import RANGES, run_profile
from phonix.engine import add_observation, new_channel, refine
from phonix.model import Reference
from phonix.profiles import Settings, Store
from phonix.regional import skip_shapes
from phonix.workflow import (_checks, _geometry, grouped_refinement, next_comparison,
                             progress, ready, regions, start_workflow)


class GroupedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(**(RANGES["full"] | dict(shape_tolerance=3, budget=300)))
        self.channel = new_channel(self.settings)
        refine(self.channel, self.settings)
        self.rng = np.random.default_rng(51)

    def record(self, planned):
        a, b, order = planned
        value = lambda f: .8 * np.log2(f / 1000) + (4 if f < 250 else 0 if f < 2000 else -3)
        add_observation(self.channel, self.settings, a, b, float(value(b) - value(a)), "manual")
        self.channel["observations"][-1]["reversed_order"] = bool(order) if order is not None else bool(self.rng.integers(0, 2))

    def test_shape_balance_and_verification_categories_finish_sequentially(self):
        for method in ("global-v1", "regional-v2"):
            with self.subTest(method=method):
                self.settings = replace(self.settings, calibration_method=method)
                self.channel = new_channel(self.settings)
                refine(self.channel, self.settings)
                checkpoints = {}
                for _ in range(200):
                    planned = next_comparison(self.channel, self.settings, self.rng)
                    if planned is None:
                        break
                    state = self.channel["workflow"]
                    if "refinement_at" not in state:
                        names = checkpoints.setdefault(state["stage"], [])
                        if not names or names[-1] != state["comparison_range"]:
                            names.append(state["comparison_range"])
                    self.record(planned)
                self.assertTrue(ready(self.channel, self.settings))
                self.assertEqual(checkpoints["balance"], ["bass/midrange", "midrange/treble"])
                self.assertEqual(checkpoints["verify" if method == "regional-v2" else "mixed"],
                                 ["midrange", "bass", "treble", "bass/midrange", "midrange/treble"])
                self.assertEqual([n for n in checkpoints if n in regions(self.channel)], ["midrange", "bass", "treble"])

    def test_partial_orders_do_not_rotate_the_next_check_into_another_range(self):
        start_workflow(self.channel)
        self.channel["workflow"]["stage"] = "mixed"
        _, local, bridges, _, _ = _geometry(self.channel, self.settings)
        categories = local | bridges
        pending = _checks(self.channel, categories, 1, True)
        self.assertEqual(pending[0][1], "midrange")
        a, b = sorted(local["midrange"])[0]
        self.record((a, b, False))
        pending = _checks(self.channel, categories, 1, True)
        self.assertEqual(pending[0][1], "midrange")
        self.assertEqual(pending[0][2], {(a, b)})
        self.record((a, b, True))
        self.assertEqual(_checks(self.channel, categories, 1, True)[0][1], "bass")

    def test_skipped_pair_keeps_the_range_active_when_an_alternative_exists(self):
        start_workflow(self.channel)
        self.channel["workflow"]["stage"] = "mixed"
        first = next_comparison(self.channel, self.settings, self.rng)
        self.channel["blocked"].append(list(first[:2]))
        alternative = next_comparison(self.channel, self.settings, self.rng)
        self.assertEqual(self.channel["workflow"]["comparison_range"], "midrange")
        self.assertNotEqual(alternative[:2], first[:2])

    def test_refinement_stays_in_range_until_independent_repeat_checks_finish(self):
        start_workflow(self.channel)
        self.channel["workflow"]["stage"] = "mixed"
        _, local, bridges, _, _ = _geometry(self.channel, self.settings)
        categories = local | bridges
        reference = Reference()
        initial = grouped_refinement(self.channel, self.settings, self.rng, reference, categories)
        name = self.channel["workflow"]["refinement_range"]
        for index in range(4):
            planned = initial if index == 0 else grouped_refinement(self.channel, self.settings, self.rng, reference, categories)
            self.assertEqual(self.channel["workflow"]["comparison_range"], name)
            self.assertEqual(self.channel["workflow"]["refinement_at"], 0)
            self.assertIn(planned[:2], categories[name])
            self.record(planned)
        checking = dict(self.channel, workflow=dict(self.channel["workflow"], started_at=0))
        self.assertEqual(_checks(checking, {name: categories[name]}, 2, True), [])
        grouped_refinement(self.channel, self.settings, self.rng, reference, categories)
        self.assertEqual(self.channel["workflow"]["refinement_at"], 4)

    def test_saved_refinement_resumes_same_range_and_cancellation_keeps_checkpoint(self):
        start_workflow(self.channel)
        self.channel["workflow"]["stage"] = "mixed"
        _, local, bridges, _, _ = _geometry(self.channel, self.settings)
        categories = local | bridges
        first = grouped_refinement(self.channel, self.settings, self.rng, Reference(), categories)
        self.record(first)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = store.create("Grouped", self.settings)
            profile["channels"]["both"] = copy.deepcopy(self.channel)
            store.save(profile)
            loaded = store.load("Grouped")["channels"]["both"]
            following = grouped_refinement(loaded, self.settings, np.random.default_rng(8), Reference(), categories)
            self.assertEqual(following, (first[0], first[1], True))
            original = store.path("Grouped").read_bytes()
            with patch("phonix.cli.Player"), patch("phonix.cli.prepare_volume", return_value=self.settings), \
                    patch("phonix.cli.refine", return_value=False), patch("phonix.cli.ready", return_value=False), \
                    patch("phonix.cli.compare", return_value=("quit", "guided")), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_profile(store, profile, 0)
            self.assertEqual(store.path("Grouped").read_bytes(), original)

    def test_balance_only_never_adds_skipped_local_ranges_to_verification(self):
        self.settings = replace(self.settings, calibration_method="regional-v2")
        skip_shapes(self.channel, self.settings)
        baseline = dict(zip(self.channel["frequencies"], Reference().expected_matches(self.channel["frequencies"])))
        names = {}
        for _ in range(100):
            planned = next_comparison(self.channel, self.settings, self.rng)
            if planned is None:
                break
            a, b, order = planned
            state = self.channel["workflow"]
            sequence = names.setdefault(state["stage"], [])
            if not sequence or sequence[-1] != state["comparison_range"]:
                sequence.append(state["comparison_range"])
            add_observation(self.channel, self.settings, a, b, float(baseline[b] - baseline[a]), "manual")
            self.channel["observations"][-1]["reversed_order"] = bool(order)
        self.assertTrue(ready(self.channel, self.settings))
        self.assertEqual(names["balance"], ["bass/midrange", "midrange/treble"])
        self.assertEqual(names["verify"], ["bass/midrange", "midrange/treble"])
        self.assertIn("boundary", progress(self.channel, self.settings))

    def test_invalid_group_metadata_is_rejected_without_changing_old_sessions(self):
        start_workflow(self.channel)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = store.create("Bad", self.settings)
            profile["channels"]["both"] = self.channel
            original = copy.deepcopy(self.channel["workflow"])
            for invalid in ({"comparison_range": "unknown"}, {"refinement_range": "bass"},
                            {"refinement_at": 0}, {"refinement_range": "bass", "refinement_at": True},
                            {"refinement_range": "bass", "refinement_at": 1}):
                self.channel["workflow"] = original | invalid
                store.save(profile)
                with self.assertRaisesRegex(ValueError, "grouped"):
                    store.load("Bad")
            self.channel["workflow"] = original
            store.save(profile)
            self.assertEqual(store.load("Bad")["channels"]["both"]["workflow"], original)


if __name__ == "__main__":
    unittest.main()
