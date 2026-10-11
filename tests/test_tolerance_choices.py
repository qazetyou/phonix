"""Listening effort choices, saved evidence and stopping safeguards; no audio."""

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from phonix.cli import browse_profiles, choose_tolerance, main, menu_tolerance, profile_summary
from phonix.engine import add_observation, fit_channel
from phonix.profiles import Settings, Store, TOLERANCE_PRESETS


class ToleranceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / "profiles")

    def command(self, *arguments):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(["--data-dir", str(self.store.root), *arguments])

    def test_three_presets_apply_to_every_range_and_explicit_flags_win(self):
        for preset, (shape, drift, budget, _) in TOLERANCE_PRESETS.items():
            for region in ("full", "bass", "midrange", "treble"):
                name = preset + "-" + region
                self.assertEqual(self.command("new", name, "--range", region, "--tolerance", preset), 0)
                cfg = Settings(**self.store.load(name)["settings"])
                self.assertEqual((cfg.shape_tolerance, cfg.drift_tolerance, cfg.budget), (shape, drift, budget))
        self.assertEqual(self.command("new", "custom", "--tolerance", "doubt", "--shape-tolerance", "1.2",
                                      "--drift-tolerance", "0.8", "--budget", "99"), 0)
        cfg = Settings(**self.store.load("custom")["settings"])
        self.assertEqual((cfg.shape_tolerance, cfg.drift_tolerance, cfg.budget), (1.2, 0.8, 99))

    def test_noninteractive_default_does_not_prompt(self):
        with patch("phonix.cli.sys.stdin.isatty", return_value=False), \
                patch("phonix.cli.select_table") as picker:
            self.assertEqual(self.command("new", "default"), 0)
        picker.assert_not_called()
        settings = self.store.load("default")["settings"]
        self.assertEqual((settings["shape_tolerance"], settings["budget"]), (3, 150))

    def test_terminal_creation_choice_cancel_and_explicit_bypass(self):
        with contextlib.redirect_stdout(io.StringIO()), \
                patch("phonix.cli.sys.stdin.isatty", return_value=True), \
                patch("phonix.cli.sys.stdout.isatty", return_value=True), \
                patch.dict("os.environ", {"TERM": "xterm"}), \
                patch("phonix.cli.select_table", side_effect=[0, None]) as picker:
            prefix = ["--data-dir", str(self.store.root), "new"]
            self.assertEqual(main(prefix + ["quick"]), 0)
            self.assertEqual(main(prefix + ["cancelled"]), 0)
            self.assertEqual(main(prefix + ["explicit", "--tolerance", "obsessed"]), 0)
        self.assertEqual(picker.call_count, 2)
        self.assertEqual(self.store.names(), ["explicit", "quick"])
        self.assertEqual(self.store.load("quick")["settings"]["budget"], 80)
        self.assertEqual(self.store.load("explicit")["settings"]["budget"], 300)

    def test_picker_descriptions_default_and_saved_selection(self):
        with patch("phonix.cli.select_table", return_value=2) as picker:
            self.assertEqual(choose_tolerance(), "obsessed")
        self.assertEqual(picker.call_args.args[3], 1)
        self.assertEqual([row[0] for row in picker.call_args.args[2]], ["Casual", "Focused", "Meticulous"])
        self.assertEqual([row[3] for row in picker.call_args.args[2]], [
            "If you are in doubt", "If you are convinced",
            "If you are now obsessed, and want to perfect your sound profile"])
        with patch("phonix.cli.select_table", return_value=None) as picker:
            self.assertIsNone(choose_tolerance(Settings(shape_tolerance=4)))
        self.assertEqual(picker.call_args.args[3], 0)

    def test_change_legacy_profile_preserves_all_evidence_and_other_settings(self):
        settings = Settings(channels="separate", budget=600, bass_db=4, bass_hz=127.5)
        profile = self.store.create("legacy", settings)
        for channel in profile["channels"].values():
            a, b = channel["frequencies"][:2]
            add_observation(channel, settings, a, b, 1, "manual")
            channel["blocked"].append([a, b])
        self.store.save(profile)
        self.assertEqual(self.command("tolerance", "legacy", "--preset", "doubt"), 0)
        loaded = self.store.load("legacy")
        self.assertEqual(loaded["channels"], profile["channels"])
        for key, value in profile["settings"].items():
            if key not in ("budget", "shape_tolerance", "drift_tolerance"):
                self.assertEqual(loaded["settings"][key], value)
        self.assertEqual(loaded["settings"]["budget"], 80)
        self.assertEqual(profile_summary(loaded)["row"][-1], "4")

    def test_smaller_cap_never_discards_answers_and_cancel_is_noop(self):
        settings = Settings(channels="separate", budget=600)
        profile = self.store.create("long", settings)
        channel = profile["channels"]["right"]
        for _ in range(90):
            add_observation(channel, settings, *channel["frequencies"][:2], 0, "manual")
        self.store.save(profile)
        original = self.store.path("long").read_bytes()
        with patch("phonix.cli.select_table", return_value=None):
            menu_tolerance(self.store, profile)
        self.assertEqual(self.store.path("long").read_bytes(), original)
        loaded = self.store.set_tolerance("long", "doubt")
        self.assertEqual(loaded["settings"]["budget"], 90)
        self.assertEqual(loaded["channels"], profile["channels"])
        original = self.store.path("long").read_bytes()
        with self.assertRaisesRegex(ValueError, "Unknown tolerance"):
            self.store.set_tolerance("long", "invalid")
        self.assertEqual(self.store.path("long").read_bytes(), original)

    def test_menu_tolerance_action_preserves_progress(self):
        profile = self.store.create("A", Settings())
        with patch("phonix.cli.select_table", side_effect=[0, 4, 1, 0, None, None, None]):
            browse_profiles(self.store)
        loaded = self.store.load("A")
        self.assertEqual(loaded["settings"]["shape_tolerance"], 4)
        self.assertEqual(loaded["settings"]["budget"], 80)
        self.assertEqual(loaded["channels"], profile["channels"])

    def test_tighter_tolerance_changes_readiness_but_never_weakens_cross_checks(self):
        settings = Settings(initial_points=4, max_points=4)
        profile = self.store.create("A", settings)
        channel = profile["channels"]["both"]
        for index, a in enumerate(channel["frequencies"]):
            for b in channel["frequencies"][index + 1:]:
                add_observation(channel, settings, a, b, 0, "manual")
        fit = replace(fit_channel(channel, settings), shape95=3.5)
        for preset, (shape, drift, budget, _) in TOLERANCE_PRESETS.items():
            cfg = replace(settings, shape_tolerance=shape, drift_tolerance=drift, budget=budget)
            self.assertEqual(bool(fit.ready(cfg)), preset == "doubt")
            for unsafe in (replace(fit, connected=False), replace(fit, cycles=1),
                           replace(fit, degrees=np.array([1, 2, 2, 2])), replace(fit, drift_rms=1.6)):
                self.assertFalse(unsafe.ready(cfg))


if __name__ == "__main__":
    unittest.main()
