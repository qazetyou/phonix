"""16 kHz support, preserved sessions and three bass shelf choices."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from phonix.audio import tone
from phonix.cli import RANGES, browse_profiles, export_profile, main, menu_bass_shelf, profile_summary
from phonix.engine import add_observation, new_channel
from phonix.model import MAX_HZ, Reference
from phonix.peq import Band, bass_shelf, easyeffects_preset, response
from phonix.profiles import BASS_STYLES, Settings, Store
from phonix.ui import choose_range, create_profile


class ExtendedRangeAndBassTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Store(Path(self.directory.name) / "profiles")

    def test_default_midbass_to_15k_and_explicit_treble_keep_reachable_anchors(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--data-dir", str(self.store.root), "new", "full"]), 0)
        settings = Settings(**self.store.load("full")["settings"])
        self.assertEqual((settings.min_hz, settings.max_hz), (60, 15000))
        for name in ("full", "treble"):
            current = Settings(**RANGES[name])
            channel = new_channel(current)
            self.assertEqual(channel["frequencies"][0], current.min_hz)
            self.assertEqual(channel["frequencies"][-1], current.max_hz)
            self.assertIn(12500, channel["frequencies"])
            self.assertEqual(len(channel["frequencies"]), current.initial_points)
            if name == "full":
                self.assertIn(1000, channel["frequencies"])
            matches = Reference().expected_matches(channel["frequencies"][-2:])
            self.assertLess(abs(matches[1] - matches[0]), current.offset_limit)
        sparse = new_channel(Settings(min_hz=20, initial_points=4, max_points=4))
        self.assertIn(1000, sparse["frequencies"])
        self.assertIn(12500, sparse["frequencies"])

    def test_ui_default_range_and_picker_use_midbass_to_15k(self):
        with patch("builtins.input", return_value="default-ui"), \
                patch("phonix.ui.select_table", return_value=7) as picker, \
                contextlib.redirect_stdout(io.StringIO()):
            profile = create_profile(self.store)
        self.assertEqual(picker.call_args.args[2][1][:2], ("Frequency range", "60-15000 Hz"))
        settings = Settings(**profile["settings"])
        self.assertEqual((settings.min_hz, settings.max_hz), (60, 15000))
        self.assertEqual((settings.shape_tolerance, settings.budget), (3, 150))
        self.assertTrue(all(60 <= f <= 15000 for f in profile["channels"]["both"]["frequencies"]))
        with patch("phonix.ui.select_table", return_value=None) as picker:
            self.assertEqual(choose_range(settings), settings)
        self.assertEqual(picker.call_args.args[3], list(RANGES).index("full"))
        self.assertIn(("Full range", "60-15000", "Bass, midrange and treble (recommended)"), picker.call_args.args[2])

    def test_custom_wider_range_and_existing_profile_are_preserved(self):
        settings = Settings(**(RANGES["full"] | dict(min_hz=20, max_hz=16000)))
        profile = self.store.create("existing", settings)
        channel = profile["channels"]["both"]
        add_observation(channel, settings, 20, 16000, 0, "manual")
        self.store.save(profile)
        before = self.store.path("existing").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--data-dir", str(self.store.root), "new", "custom",
                                   "--min-hz", "20", "--max-hz", "16000"]), 0)
            self.assertEqual(main(["--data-dir", str(self.store.root), "new", "default"]), 0)
        custom = self.store.load("custom")
        self.assertEqual((custom["settings"]["min_hz"], custom["settings"]["max_hz"]), (20, 16000))
        self.assertEqual(self.store.path("existing").read_bytes(), before)
        self.assertEqual(self.store.load("existing"), profile)

    def test_reference_extension_is_explicit_and_old_values_are_identical(self):
        reference = Reference()
        old_grid = np.geomspace(20, 15000, 1000)
        old_values = np.interp(np.log(old_grid), np.log(reference.middle[0]), reference.middle[1])
        np.testing.assert_array_equal(reference.sensitivity(old_grid), old_values)
        values = reference.sensitivity([15000, 15500, 16000])
        np.testing.assert_array_equal(values, np.repeat(reference.middle[1, -1], 3))
        self.assertTrue(np.all(np.isfinite(reference.target([15000, 16000]))))
        self.assertNotEqual(reference.target([15000])[0], reference.target([16000])[0])
        with self.assertRaises(ValueError):
            reference.target([MAX_HZ + 1])
        for rate in (44100, 48000, 96000):
            samples = tone(16000, -42, "both", rate=rate)
            self.assertGreater(np.max(abs(samples)), 0)
            self.assertLessEqual(np.max(abs(samples)), 10 ** (-42 / 20))
            preset = easyeffects_preset({"both": [Band(16000, 2, 1)]}, rate)
            self.assertEqual(preset["output"]["equalizer#0"]["left"]["band0"]["frequency"], 16000)

    def test_extend_empty_profile_rebuilds_grid_and_does_not_touch_other_profile(self):
        profile = self.store.create("empty", Settings(min_hz=20, max_hz=12000, initial_points=17, max_points=32, bass_db=4))
        self.store.create("other", Settings(max_hz=12000))
        original = self.store.path("other").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--data-dir", str(self.store.root), "extend", "empty"]), 0)
        expanded = self.store.load("empty")
        self.assertEqual(expanded["settings"]["max_hz"], 16000)
        self.assertEqual(expanded["settings"]["bass_db"], 4)
        self.assertEqual(len(expanded["channels"]["both"]["frequencies"]), 17)
        self.assertEqual(profile_summary(expanded)["status"], "new")
        self.assertEqual(self.store.path("other").read_bytes(), original)

    def test_extend_preserves_both_channels_judgments_and_old_frequencies(self):
        settings = Settings(max_hz=12000, channels="separate", initial_points=4, max_points=4)
        profile = self.store.create("ears", settings)
        for channel in profile["channels"].values():
            for _ in range(2):
                for index, a in enumerate(channel["frequencies"]):
                    for b in channel["frequencies"][index + 1:]:
                        add_observation(channel, settings, a, b, 0, "manual")
        self.assertEqual(profile_summary(profile)["status"], "complete")
        self.store.save(profile)
        original = copy.deepcopy(profile)
        expanded = self.store.extend("ears")
        self.assertEqual(profile_summary(expanded)["status"], "unfinished")
        self.assertEqual(expanded["settings"]["max_points"], 5)
        for name, channel in expanded["channels"].items():
            self.assertEqual(channel["observations"], original["channels"][name]["observations"])
            self.assertEqual(channel["frequencies"], original["channels"][name]["frequencies"] + [16000])
            self.assertEqual(channel["inputs"], 12)
        with self.assertRaisesRegex(ValueError, "unmeasured priors"):
            export_profile(expanded, Path(self.directory.name) / "incomplete.json", allow_incomplete=True)
        before = self.store.path("ears").read_bytes()
        self.store.extend("ears")
        self.assertEqual(self.store.path("ears").read_bytes(), before)
        for upper in (12000, 16001, float("nan")):
            with self.assertRaises(ValueError):
                self.store.extend("ears", upper)
            self.assertEqual(self.store.path("ears").read_bytes(), before)

    def test_extension_rejects_exceeding_point_limit_without_losing_history(self):
        settings = Settings(max_hz=12000, initial_points=32, max_points=32)
        profile = self.store.create("full", settings)
        channel = profile["channels"]["both"]
        add_observation(channel, settings, *channel["frequencies"][:2], 0, "manual")
        self.store.save(profile)
        original = self.store.path("full").read_bytes()
        with self.assertRaises(ValueError):
            self.store.extend("full")
        self.assertEqual(self.store.path("full").read_bytes(), original)

    def test_bass_choices_cli_and_half_gain_points(self):
        for style, (hz, description) in BASS_STYLES.items():
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--data-dir", str(self.store.root), "new", style,
                                       "--bass-db", "4", "--bass-style", style]), 0)
            self.assertEqual(self.store.load(style)["settings"]["bass_hz"], hz)
            self.assertTrue(description)
            self.assertAlmostEqual(response([hz], [bass_shelf(4, hz)], 48000)[0], 2, places=7)
        self.assertGreater(response([100], [bass_shelf(4, 180)], 48000)[0], response([100], [bass_shelf(4, 60)], 48000)[0])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["new", "conflict", "--bass-style", "warm", "--bass-hz", "120"])

    def test_bass_menu_updates_preference_without_changing_judgments_or_gain(self):
        profile = self.store.create("A", Settings(bass_db=4))
        original = copy.deepcopy(profile)
        with patch("phonix.cli.select_table", return_value=None):
            menu_bass_shelf(self.store, profile)
        self.assertEqual(self.store.load("A"), original)
        for selected, hz in enumerate((60, 105, 180)):
            with patch("phonix.cli.select_table", return_value=selected) as picker:
                menu_bass_shelf(self.store, profile)
            self.assertEqual(self.store.load("A")["settings"]["bass_hz"], hz)
            self.assertEqual(profile["channels"], original["channels"])
            self.assertEqual(profile["settings"]["bass_db"], 4)
            self.assertEqual(len(picker.call_args.args[2]), 4)
            self.assertEqual(profile_summary(profile)["row"][6], str(hz))

    def test_manual_bass_shelf_accepts_decimal_and_preserves_session(self):
        settings = Settings(bass_db=4)
        profile = self.store.create("A", settings)
        channel = profile["channels"]["both"]
        add_observation(channel, settings, *channel["frequencies"][:2], 0, "manual")
        self.store.save(profile)
        original = copy.deepcopy(profile)
        with patch("phonix.cli.select_table", return_value=3) as picker, \
                patch("builtins.input", return_value="127.5"):
            menu_bass_shelf(self.store, profile)
        expected = copy.deepcopy(original)
        expected["settings"]["bass_hz"] = 127.5
        self.assertEqual(self.store.load("A"), expected)
        self.assertEqual(picker.call_args.args[2][3][0], "Manual input")
        self.assertEqual(profile_summary(profile)["row"][6], "127.5")
        with patch("phonix.cli.select_table", return_value=None) as picker:
            menu_bass_shelf(self.store, profile)
        self.assertEqual(picker.call_args.args[3], 3)

    def test_manual_bass_shelf_reprompts_for_invalid_values_and_accepts_bounds(self):
        profile = self.store.create("A", Settings(bass_db=-4))
        with patch("phonix.cli.select_table", return_value=3), \
                patch("builtins.input", side_effect=["not a number", "39", "301", "nan", "inf", "-inf", "40"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            menu_bass_shelf(self.store, profile)
        self.assertEqual(output.getvalue().count("Enter a finite frequency"), 6)
        self.assertEqual(self.store.load("A")["settings"]["bass_hz"], 40)
        with patch("phonix.cli.select_table", return_value=3), patch("builtins.input", return_value="300"):
            menu_bass_shelf(self.store, profile)
        self.assertEqual(self.store.load("A")["settings"]["bass_hz"], 300)
        self.assertEqual(profile["settings"]["bass_db"], -4)

    def test_manual_bass_shelf_blank_cancel_eof_and_interrupt_do_not_save(self):
        profile = self.store.create("A", Settings(bass_hz=127.5))
        original = self.store.path("A").read_bytes()
        for answer in ("", "q", " Q "):
            with self.subTest(answer=answer), patch("phonix.cli.select_table", return_value=3), \
                    patch("builtins.input", return_value=answer):
                menu_bass_shelf(self.store, profile)
            self.assertEqual(self.store.path("A").read_bytes(), original)
        with patch("phonix.cli.select_table", return_value=3), patch("builtins.input", side_effect=EOFError):
            menu_bass_shelf(self.store, profile)
        self.assertEqual(self.store.path("A").read_bytes(), original)
        with patch("phonix.cli.select_table", return_value=3), patch("builtins.input", side_effect=KeyboardInterrupt), \
                self.assertRaises(KeyboardInterrupt):
            menu_bass_shelf(self.store, profile)
        self.assertEqual(self.store.path("A").read_bytes(), original)

    def test_export_bass_style_is_an_override_only(self):
        self.store.create("A", Settings(bass_db=4))
        original = self.store.path("A").read_bytes()
        with patch("phonix.cli.export_profile") as export:
            self.assertEqual(main(["--data-dir", str(self.store.root), "export", "A", "--bass-style", "warm",
                                   "--output", "example.json"]), 0)
        self.assertEqual(export.call_args.args[-2:], (180, None))
        self.assertEqual(self.store.path("A").read_bytes(), original)

    def test_browser_can_extend_an_old_profile_and_choose_bass_style(self):
        self.store.create("A", Settings(max_hz=12000))
        with patch("phonix.cli.select_table", side_effect=[0, 4, 5, 2, 2, None, None, None]), \
                patch("phonix.ui.select_table", side_effect=[1, None]):
            browse_profiles(self.store)
        settings = self.store.load("A")["settings"]
        self.assertEqual((settings["max_hz"], settings["bass_hz"]), (16000, 180))
