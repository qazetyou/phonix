"""Confirmed profile resets start fresh without touching other saved files."""

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phonix.cli import browse_profiles, menu_export, profile_summary
from phonix.engine import add_observation
from phonix.profiles import Settings, Store


class ProfileResetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.store = Store(self.root / "profiles")

    def add_progress(self, profile):
        settings = Settings(**profile["settings"])
        for channel in profile["channels"].values():
            a, b = channel["frequencies"][:2]
            channel["frequencies"].insert(1, (a * b) ** 0.5)
            add_observation(channel, settings, a, b, 2, "manual")
            channel["blocked"].append(channel["frequencies"][-2:])
            channel["workflow"] = {"version": 1, "stage": "midrange", "started_at": 1}
        self.store.save(profile)

    def make_profile(self, name, status):
        settings = Settings(initial_points=4, max_points=4,
                            budget=1 if status == "incomplete" else 12)
        profile = self.store.create(name, settings)
        channel = profile["channels"]["both"]
        if status == "complete":
            for _ in range(2):
                for index, a in enumerate(channel["frequencies"]):
                    for b in channel["frequencies"][index + 1:]:
                        add_observation(channel, settings, a, b, 0, "manual")
        elif status != "new":
            add_observation(channel, settings, *channel["frequencies"][:2], 0, "manual")
        self.store.save(profile)
        self.assertEqual(profile_summary(profile)["status"], status)
        return profile

    def test_reset_matches_fresh_creation_for_both_channel_modes(self):
        for mode in ("together", "separate"):
            with self.subTest(channels=mode):
                settings = Settings(min_hz=60, max_hz=15000, channels=mode,
                                    mode="manual", budget=150, shape_tolerance=3,
                                    reference_level=-30, max_level=-12,
                                    sample_rate=96000, initial_points=7, max_points=19,
                                    device="physical-output", seed=42, bass_db=4,
                                    bass_hz=180, noise_level=-17.7, noise_ceiling=-4,
                                    filter_mode="bells")
                profile = self.store.create(mode, settings)
                self.add_progress(profile)
                expected = Store(self.root / "fresh").create(mode, settings)
                reset = self.store.reset(mode)
                self.assertEqual(reset, expected)
                self.assertEqual(self.store.load(mode), expected)
                self.assertEqual(reset["settings"], asdict(settings))
                summary = profile_summary(reset)
                self.assertEqual(summary["status"], "new")
                self.assertFalse(summary["connected"])
                self.assertTrue(all(channel["inputs"] == 0 for channel in reset["channels"].values()))
                with patch("phonix.cli.export_profile") as export, \
                        self.assertRaisesRegex(ValueError, "connect all"):
                    menu_export(reset, summary)
                export.assert_not_called()

    def test_reset_legacy_profile_uses_its_compatible_settings_defaults(self):
        profile = self.store.create("legacy", Settings())
        for field in ("bass_db", "bass_hz", "noise_level", "noise_ceiling", "filter_mode"):
            del profile["settings"][field]
        self.store.save(profile)
        expected = Store(self.root / "fresh").create("legacy", Settings(**profile["settings"]))
        self.assertEqual(self.store.reset("legacy"), expected)

    def test_failed_reset_write_preserves_saved_evidence_and_removes_temporary_file(self):
        profile = self.store.create("keep", Settings())
        self.add_progress(profile)
        original = self.store.path("keep").read_bytes()
        with patch("phonix.profiles.Path.replace", side_effect=OSError("write failed")), \
                self.assertRaisesRegex(OSError, "write failed"):
            self.store.reset("keep")
        self.assertEqual(self.store.path("keep").read_bytes(), original)
        self.assertEqual(list(self.store.root.glob(".saving-*")), [])

    def test_reset_rejects_missing_invalid_and_corrupt_profiles_without_changes(self):
        self.store.create("keep", Settings())
        self.store.path("broken").write_text("{", encoding="utf-8")
        before = {path.name: path.read_bytes() for path in self.store.root.iterdir()}
        for name in ("missing", "../keep", "broken"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store.reset(name)
        self.assertEqual({path.name: path.read_bytes() for path in self.store.root.iterdir()}, before)

    def pick_reset(self, confirmation):
        visited = False
        managed = False

        def pick(title, headers, rows, *args):
            nonlocal visited, managed
            if headers == ("Action", "Details") and not visited:
                visited = True
                return next(index for index, row in enumerate(rows) if row[0] == "Manage profile")
            if "> Manage profile" in title and not managed:
                managed = True
                return next(index for index, row in enumerate(rows) if row[0] == "Reset")
            if headers == ("Action",):
                self.assertEqual(rows, [("Cancel",), ("Reset profile",)])
                self.assertIn("Clear calibration", title)
                self.assertIn("keep settings and exports", title)
                return confirmation
            return None

        return pick

    def test_confirmed_reset_reopens_same_profile_as_new_and_preserves_other_files(self):
        original = self.make_profile("Selected", "complete")
        self.store.create("Other", Settings())
        other = self.store.path("Other").read_bytes()
        exports = [self.root / "Selected.json", self.root / "Selected.report.json"]
        for path in exports:
            path.write_text("existing export\n", encoding="utf-8")
        with patch("phonix.cli.select_table", side_effect=self.pick_reset(1)) as picker, \
                patch.object(self.store, "reset", wraps=self.store.reset) as reset, \
                patch("phonix.cli.run_profile") as run, \
                patch("phonix.cli.check_audio") as audio:
            browse_profiles(self.store, open_name="Selected")
        reset.assert_called_once_with("Selected")
        run.assert_not_called()
        audio.assert_not_called()
        self.assertIn("Selected | complete |", picker.call_args_list[0].args[0])
        self.assertIn("Reset Selected?", picker.call_args_list[2].args[0])
        self.assertIn("Selected | new |", picker.call_args_list[3].args[0])
        self.assertEqual(picker.call_args_list[3].args[2][0][0], "Start calibration")
        self.assertEqual(picker.call_args_list[3].args[3], 0)
        self.assertEqual(picker.call_args_list[4].args[0], "Device profiles")
        self.assertEqual(picker.call_args_list[4].args[3], 1)
        self.assertEqual(self.store.load("Selected")["settings"], original["settings"])
        self.assertEqual(self.store.path("Other").read_bytes(), other)
        self.assertTrue(all(path.read_text() == "existing export\n" for path in exports))

    def test_cancel_and_escape_preserve_every_profile_state(self):
        for status in ("new", "unfinished", "incomplete", "complete"):
            self.make_profile(status, status)
            original = self.store.path(status).read_bytes()
            for confirmation in (0, None):
                with self.subTest(status=status, confirmation=confirmation), \
                        patch("phonix.cli.select_table", side_effect=self.pick_reset(confirmation)) as picker, \
                        patch.object(self.store, "reset", wraps=self.store.reset) as reset:
                    browse_profiles(self.store, open_name=status)
                    reset.assert_not_called()
                    self.assertIn("> Manage profile", picker.call_args_list[3].args[0])
                    self.assertIn(f"{status} | {status} |", picker.call_args_list[4].args[0])
                    self.assertEqual(self.store.path(status).read_bytes(), original)
