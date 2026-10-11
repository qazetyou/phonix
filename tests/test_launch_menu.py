"""Menu entry, guided creation and settings changes without live audio."""

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phonix.cli import browse_profiles, main
from phonix.engine import add_observation
from phonix.profiles import Settings, Store
from phonix.ui import (choose_device, create_profile, edit_number,
                                  edit_playback, main_menu)


class LaunchMenuTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / "profiles")

    def test_no_command_opens_menu_in_terminal_and_honors_data_directory(self):
        with contextlib.redirect_stdout(io.StringIO()), \
                patch("phonix.cli.sys.stdin.isatty", return_value=True), \
                patch("phonix.cli.sys.stdout.isatty", return_value=True), \
                patch.dict("os.environ", {"TERM": "xterm"}), \
                patch("phonix.cli.run_ui", side_effect=lambda callback, *args: callback(*args)), \
                patch("phonix.ui.main_menu") as menu:
            self.assertEqual(main(["--data-dir", str(self.store.root)]), 0)
        self.assertEqual(menu.call_args.args[0].root, self.store.root)

    def test_piped_launch_and_explicit_help_still_print_help(self):
        with contextlib.redirect_stdout(io.StringIO()) as output, \
                patch("phonix.cli.sys.stdin.isatty", return_value=False), \
                patch("phonix.ui.main_menu") as menu:
            self.assertEqual(main([]), 0)
            with self.assertRaises(SystemExit) as exit_code:
                main(["--help"])
        self.assertEqual(exit_code.exception.code, 0)
        self.assertIn("usage:", output.getvalue())
        menu.assert_not_called()

    def test_main_menu_dispatches_existing_profiles_and_devices(self):
        self.store.create("A", Settings())
        with patch("phonix.ui.select_table", side_effect=[0, 3, 0, None, None]) as picker, \
                patch("phonix.ui.browse_profiles") as browser, \
                patch("phonix.ui.show_devices") as outputs:
            main_menu(self.store)
        browser.assert_called_once_with(self.store)
        outputs.assert_called_once_with()
        self.assertEqual([row[0] for row in picker.call_args_list[0].args[2]],
                         ["Profiles", "New profile", "Help", "Tools", "Quit"])

    def test_empty_profiles_help_errors_and_cancel_return_to_main_menu(self):
        with patch("phonix.ui.select_table", side_effect=[0, 2, 1, None, 3, 0, None, 4]), \
                patch("phonix.ui.show_devices", side_effect=ValueError("No PipeWire")), \
                patch("builtins.input", return_value=""), \
                contextlib.redirect_stdout(io.StringIO()) as output, \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            main_menu(self.store)
        self.assertIn("New profile", output.getvalue())
        self.assertIn("Up/Down", output.getvalue())
        self.assertIn("No PipeWire", errors.getvalue())
        self.assertEqual(self.store.names(), [])

    def test_creation_opens_saved_profile_actions_without_extra_prompt_or_audio(self):
        self.store.create("A", Settings())
        with patch("phonix.ui.select_table", side_effect=[1, 7, 4]), \
                patch("phonix.cli.select_table", side_effect=[None, None]) as picker, \
                patch("builtins.input", return_value="Z headphones") as text, \
                patch("phonix.cli.Player") as player, \
                contextlib.redirect_stdout(io.StringIO()):
            main_menu(self.store)
        text.assert_called_once()
        player.assert_not_called()
        self.assertEqual(self.store.names(), ["A", "Z headphones"])
        self.assertTrue(picker.call_args_list[0].args[0].startswith("Z headphones | new |"))
        actions = [row[0] for row in picker.call_args_list[0].args[2]]
        self.assertIn("Start calibration", actions)
        self.assertIn("Export to Easy Effects", actions)
        self.assertIn("Manage profile", actions)
        self.assertIn("Settings", actions)
        self.assertEqual(picker.call_args_list[1].args[0], "Device profiles")
        self.assertEqual(picker.call_args_list[1].args[3], 1)

    def test_cancelled_creation_returns_to_main_menu_without_opening_profile(self):
        with patch("phonix.ui.select_table", side_effect=[1, 8, 4]), \
                patch("builtins.input", return_value="draft"), \
                patch("phonix.ui.browse_profiles") as browser:
            main_menu(self.store)
        browser.assert_not_called()
        self.assertEqual(self.store.names(), [])

    def test_guided_creation_sets_choices_and_custom_limits_before_saving(self):
        selections = [1, 3, 2, 4, 1, 5, 1, 0, 1, 3, 0, 1, 2, 6, 0, 11, 7]
        with patch("phonix.ui.select_table", side_effect=selections), \
                patch("phonix.ui.choose_tolerance", return_value="doubt"), \
                patch("phonix.ui.choose_bass_shelf", side_effect=lambda cfg: replace(cfg, bass_hz=127.5)), \
                patch("phonix.ui.devices", return_value=[{"node.name": "physical", "node.description": "Headphones"}]), \
                patch("builtins.input", side_effect=["wizard", "4", "40"]), \
                contextlib.redirect_stdout(io.StringIO()):
            profile = create_profile(self.store)
        settings = Settings(**self.store.load("wizard")["settings"])
        self.assertEqual((settings.min_hz, settings.max_hz), (40, 15000))
        self.assertEqual((settings.shape_tolerance, settings.drift_tolerance, settings.budget), (4, 1.5, 80))
        self.assertEqual((settings.channels, settings.mode, settings.device), ("separate", "manual", "physical"))
        self.assertEqual((settings.bass_db, settings.bass_hz), (4, 127.5))
        self.assertEqual(set(profile["channels"]), {"left", "right"})
        self.assertTrue(all(channel["inputs"] == 0 for channel in profile["channels"].values()))

    def test_creation_cancel_and_duplicate_name_retry_do_not_overwrite(self):
        self.store.create("existing", Settings())
        original = self.store.path("existing").read_bytes()
        with patch("builtins.input", side_effect=["../invalid", "existing", "new"]), \
                patch("phonix.ui.select_table", return_value=8), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertIsNone(create_profile(self.store))
        self.assertIn("already exists", output.getvalue())
        self.assertEqual(self.store.names(), ["existing"])
        self.assertEqual(self.store.path("existing").read_bytes(), original)
        for response in ("q", EOFError()):
            with patch("builtins.input", side_effect=[response]):
                self.assertIsNone(create_profile(self.store))

    def test_numeric_retry_and_cancel_validate_before_mutation(self):
        settings = Settings()
        with patch("builtins.input", side_effect=["nan", "13", "4"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            changed = edit_number(settings, "bass_db")
        self.assertEqual(changed.bass_db, 4)
        self.assertEqual(settings.bass_db, 0)
        self.assertIn("Invalid value", output.getvalue())
        with patch("builtins.input", return_value="q"):
            self.assertEqual(edit_number(settings, "budget"), settings)

    def test_device_failure_still_allows_manual_target_and_cancel(self):
        settings = Settings(device="old")
        with patch("phonix.ui.devices", side_effect=ValueError("offline")), \
                patch("phonix.ui.select_table", return_value=2), \
                patch("builtins.input", return_value="physical"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(choose_device(settings).device, "physical")
        with patch("phonix.ui.devices", return_value=[]), \
                patch("phonix.ui.select_table", return_value=None):
            self.assertEqual(choose_device(settings), settings)

    def test_playback_editor_saves_empty_profile_and_cancel_keeps_file(self):
        profile = self.store.create("A", Settings())
        with patch("phonix.ui.select_table", side_effect=[1, 2, 5]), \
                patch("builtins.input", side_effect=["-30", "-12"]):
            edit_playback(self.store, profile)
        saved = self.store.load("A")
        self.assertEqual((saved["settings"]["reference_level"], saved["settings"]["max_level"]), (-30, -12))
        original = self.store.path("A").read_bytes()
        with patch("phonix.ui.select_table", side_effect=[1, 6]), \
                patch("builtins.input", side_effect=["-31"]):
            edit_playback(self.store, saved)
        self.assertEqual(self.store.path("A").read_bytes(), original)

    def test_playback_editor_locks_tone_settings_after_either_channel_has_evidence(self):
        cfg = Settings(channels="separate")
        profile = self.store.create("ears", cfg)
        channel = profile["channels"]["right"]
        add_observation(channel, cfg, *channel["frequencies"][:2], 0, "manual")
        self.store.save(profile)
        with patch("phonix.ui.select_table", side_effect=[1]) as picker:
            edit_playback(self.store, profile)
        labels = [row[0] for row in picker.call_args_list[0].args[2]]
        self.assertNotIn("Starting tone level (dBFS peak)", labels)
        self.assertNotIn("Sample rate (Hz)", labels)
        loaded = self.store.load("ears")
        self.assertEqual(labels, ["Output device", "Save", "Cancel"])
        self.assertEqual(loaded["settings"]["noise_level"], cfg.noise_level)
        self.assertEqual(loaded["settings"]["reference_level"], cfg.reference_level)
        self.assertEqual(loaded["channels"], profile["channels"])

    def test_profile_report_audio_check_and_playback_editor_dispatch(self):
        profile = self.store.create("A", Settings())
        with patch("phonix.cli.select_table", side_effect=[0, 1, 4, 0, None, None, None]), \
                patch("phonix.ui.select_table", side_effect=[0, 1, None]), \
                patch("phonix.cli.report_profile") as report, \
                patch("phonix.ui.check_audio") as audio, \
                patch("phonix.ui.edit_playback") as editor, \
                patch("builtins.input", return_value=""):
            browse_profiles(self.store)
        report.assert_called_once_with(profile)
        audio.assert_called_once_with(profile)
        editor.assert_called_once_with(self.store, profile)

    def test_demo_dispatch_is_explicit_and_cancel_does_not_create(self):
        with patch("phonix.ui.select_table", side_effect=[3, 1, 1, None, 4]), \
                patch("builtins.input", side_effect=["q", "demo", ""]), \
                patch("phonix.ui.simulate") as demo:
            main_menu(self.store)
        demo.assert_called_once_with(self.store, "demo")


if __name__ == "__main__":
    unittest.main()
