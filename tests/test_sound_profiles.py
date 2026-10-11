"""Post-calibration voicing stays separate from evidence and neutral fitting."""

import contextlib
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from phonix.cli import (build_preset, export_profile, main, menu_sound_profile,
                        preview_profile)
from phonix.engine import add_observation, fit_channel
from phonix.peq import Band, bass_shelf, response
from phonix.profiles import Settings, Store
from phonix.voicing import SOUND_PROFILES, fit_voiced, voicing_filters


class SoundProfileTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = Store(self.root / "profiles")
        self.settings = Settings(min_hz=100, max_hz=10000, initial_points=4, max_points=4, budget=40)
        self.profile = self.store.create("voices", self.settings)
        channel = self.profile["channels"]["both"]
        for _ in range(2):
            for i, a in enumerate(channel["frequencies"]):
                for b in channel["frequencies"][i + 1:]:
                    add_observation(channel, self.settings, a, b, 0, "manual")
        self.store.save(self.profile)

    def test_voicings_have_explicit_monotonic_shelves_and_neutral_has_none(self):
        self.assertEqual(voicing_filters("neutral"), [])
        grid = np.array([20., 1000., 20000.])
        for name, (_, _, specification) in SOUND_PROFILES.items():
            bands = voicing_filters(name)
            self.assertEqual([(b.kind, b.frequency, b.gain) for b in bands], list(specification))
            self.assertTrue(all(b.purpose == "Sound preference" and b.q == 1 / np.sqrt(2) for b in bands))
            for band in bands:
                curve = response(grid, [band], 48000)
                if band.kind == "Lo-shelf":
                    self.assertAlmostEqual(curve[0], band.gain, delta=.01)
                    self.assertAlmostEqual(curve[-1], 0, delta=.01)
                else:
                    self.assertAlmostEqual(curve[0], 0, delta=.01)
                    self.assertAlmostEqual(curve[-1], band.gain, delta=.01)

    def test_choice_save_cancel_and_legacy_defaults_preserve_evidence(self):
        evidence = copy.deepcopy(self.profile["channels"])
        before = self.store.path("voices").read_bytes()
        with patch("phonix.cli.select_table", return_value=None), contextlib.redirect_stdout(io.StringIO()):
            menu_sound_profile(self.store, self.profile)
        self.assertEqual(self.store.path("voices").read_bytes(), before)
        with patch("phonix.cli.select_table", return_value=1) as picker, contextlib.redirect_stdout(io.StringIO()):
            menu_sound_profile(self.store, self.profile)
        self.assertEqual(self.store.load("voices")["settings"]["sound_profile"], "v-shaped")
        self.assertEqual(self.store.load("voices")["channels"], evidence)
        self.assertEqual([row[0] for row in picker.call_args.args[2]], [v[0] for v in SOUND_PROFILES.values()])
        del self.profile["settings"]["sound_profile"]
        del self.profile["settings"]["calibration_method"]
        self.store.save(self.profile)
        cfg = Settings(**self.store.load("voices")["settings"])
        self.assertEqual((cfg.sound_profile, cfg.calibration_method), ("neutral", "global-v1"))

    def test_preference_reserves_slots_and_does_not_change_fitter_target(self):
        frequencies = np.array(self.profile["channels"]["both"]["frequencies"])
        desired = np.array([1., -1., -2., 2.])
        for name in SOUND_PROFILES:
            cfg = replace(self.settings, sound_profile=name)
            correction = Band(1000, 1, 1)
            with patch("phonix.voicing.fit_with_bass", return_value=([correction], {"bands": []})) as fit:
                bands, metrics = fit_voiced(frequencies, desired, cfg, 12)
            np.testing.assert_array_equal(fit.call_args.args[1], desired)
            self.assertEqual(fit.call_args.args[3], 12 - len(voicing_filters(name)))
            self.assertEqual(bands, [correction, *voicing_filters(name)])
            self.assertEqual(metrics["sound_profile"], name)

    def test_insufficient_slots_and_unknown_names_are_rejected(self):
        for name, count, bass in (("v-shaped", 2, 0), ("warm-tilt", 3, 4), ("bright", 1, 0)):
            with self.assertRaisesRegex(ValueError, "at least"):
                fit_voiced(np.array([100., 1000., 10000.]), np.zeros(3),
                           replace(self.settings, sound_profile=name, bass_db=bass), count)
        for name in ("unknown", None, ["neutral"]):
            with self.assertRaises(ValueError):
                replace(self.settings, sound_profile=name).validate()
            with self.assertRaises(ValueError):
                voicing_filters(name)

    def test_real_export_appends_exact_preferences_reports_target_and_headroom(self):
        self.profile["settings"].update(sound_profile="warm-tilt", bass_db=4)
        original = copy.deepcopy(self.profile)
        channel = self.profile["channels"]["both"]
        before = fit_channel(channel, Settings(**self.profile["settings"]))
        preset, report = build_preset(self.profile, count=6)
        bands = report["channels"]["both"]["peq_fit"]["bands"]
        self.assertEqual([b["purpose"] for b in bands[-3:]], ["Bass preference", "Sound preference", "Sound preference"])
        self.assertEqual([(b["kind"], b["frequency"], b["gain"]) for b in bands[-3:]],
                         [("Lo-shelf", 105., 4.), ("Lo-shelf", 250., 2.), ("Hi-shelf", 2500., -2.)])
        info = report["channels"]["both"]
        np.testing.assert_allclose(info["requested_correction_db"],
                                   np.array(info["model_correction_db"]) + info["bass_preference_curve_db"] + np.array(info["sound_preference_curve_db"]))
        self.assertTrue(report["sound_preference"]["applied_after_calibration"])
        self.assertEqual(report["sound_preference"]["name"], "warm-tilt")
        self.assertEqual(self.profile, original)
        after = fit_channel(channel, Settings(**self.profile["settings"]))
        np.testing.assert_array_equal(before.matches, after.matches)
        rendered = [Band(b["frequency"], b["gain"], b["q"], b["kind"], b["purpose"]) for b in bands]
        grid = np.unique(np.r_[np.geomspace(10, 23999, 2000), [b.frequency for b in rendered]])
        self.assertLessEqual((response(grid, rendered, 48000) + report["input_gain_db"]).max(), -.95)
        self.assertEqual(preset["output"]["equalizer#0"]["num-bands"], 6)

    def test_preview_uses_the_same_preference_filters_as_export(self):
        self.profile["settings"]["sound_profile"] = "v-shaped"
        with patch("phonix.cli.select_table", return_value=None) as table, \
                contextlib.redirect_stdout(io.StringIO()):
            preview_profile(self.profile)
            export_profile(self.profile, self.root / "voice.json", count=12)
        preset = json.loads((self.root / "voice.json").read_text())["output"]["equalizer#0"]
        rows = table.call_args.args[2]
        for i, row in enumerate(rows):
            band = preset["left"][f"band{i}"]
            self.assertEqual(row[2:5], (band["type"], f"{band['frequency']:.1f}", f"{band['gain']:+.2f}"))
        self.assertEqual([r[-1] for r in rows[-2:]], ["Sound preference", "Sound preference"])

    def test_cli_creation_uses_new_workflow_and_validates_preference(self):
        with patch("phonix.cli.sys.stdin.isatty", return_value=False), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--data-dir", str(self.store.root), "new", "CLI", "--sound-profile", "bright"]), 0)
        cfg = Settings(**self.store.load("CLI")["settings"])
        self.assertEqual((cfg.sound_profile, cfg.calibration_method), ("bright", "regional-v2"))


if __name__ == "__main__":
    unittest.main()
