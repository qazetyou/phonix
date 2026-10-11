"""Menu paths, focus and local error recovery without playing audio."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phonix.cli import browse_profiles, menu_profile_settings
from phonix.engine import add_observation
from phonix.profiles import Settings, Store
from phonix.ui import main_menu


def choose_actions(*actions):
    remaining = iter(actions)

    def pick(title, headers, rows, initial=0, **kwargs):
        action = next(remaining)
        if action is None:
            return None
        labels = [row[0] for row in rows]
        if action not in labels:
            raise AssertionError(f"{title}: {action!r} is absent from {labels}")
        return labels.index(action)

    return pick


class MenuNavigationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / "profiles")
        self.profile = self.store.create("Headphones", Settings())

    def test_nested_back_and_escape_restore_focus_without_rewriting_profile(self):
        original = self.store.path("Headphones").read_bytes()
        choices = choose_actions("Settings", "Audio and volume", "Back", "Listening effort",
                                 None, "Back", "Back", None)
        with patch("phonix.cli.select_table", side_effect=choices) as cli_picker, \
                patch("phonix.ui.select_table", side_effect=choices) as ui_picker, \
                patch("phonix.cli.Player") as player:
            browse_profiles(self.store, open_name="Headphones")
        self.assertIn("Settings > Audio and volume", ui_picker.call_args.args[0])
        calls = cli_picker.call_args_list
        self.assertEqual(calls[2].args[3], 0)  # Audio and volume remains highlighted.
        self.assertEqual(calls[4].args[3], 1)  # Listening effort remains highlighted.
        self.assertEqual(calls[5].args[3], 4)  # Settings remains highlighted in the profile.
        player.assert_not_called()
        self.assertEqual(self.store.path("Headphones").read_bytes(), original)

    def test_audio_error_returns_to_audio_menu_and_keeps_settings(self):
        original = self.store.path("Headphones").read_bytes()
        choices = choose_actions("Audio and volume", "Check sound", None, None)
        with patch("phonix.cli.select_table", side_effect=choices), \
                patch("phonix.ui.select_table", side_effect=choices) as picker, \
                patch("phonix.ui.check_audio", side_effect=ValueError("Output unavailable")), \
                patch("phonix.cli._menu_text", return_value="") as pause, \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            menu_profile_settings(self.store, "Headphones")
        self.assertEqual(picker.call_args_list[0].args[0], picker.call_args_list[1].args[0])
        self.assertEqual(picker.call_args_list[1].args[3], 0)
        pause.assert_called_once_with("Enter to return to Audio and volume: ")
        self.assertIn("Output unavailable", errors.getvalue())
        self.assertEqual(self.store.path("Headphones").read_bytes(), original)

    def test_unavailable_export_returns_to_same_profile_with_export_selected(self):
        original = self.store.path("Headphones").read_bytes()
        choices = choose_actions("Export to Easy Effects", None, None)
        with patch("phonix.cli.select_table", side_effect=choices) as picker, \
                patch("phonix.cli._menu_text", return_value="") as pause, \
                patch("phonix.cli.export_profile") as export, \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            browse_profiles(self.store, open_name="Headphones")
        self.assertEqual(picker.call_args_list[0].args[0], picker.call_args_list[1].args[0])
        self.assertEqual(picker.call_args_list[1].args[3], 3)
        pause.assert_called_once_with("Enter to return to the profile menu: ")
        self.assertIn("connect all", errors.getvalue())
        export.assert_not_called()
        self.assertEqual(self.store.path("Headphones").read_bytes(), original)

    def test_bass_amount_and_warmth_are_together_and_preserve_answers(self):
        settings = Settings(**self.profile["settings"])
        channel = self.profile["channels"]["both"]
        add_observation(channel, settings, *channel["frequencies"][:2], 0, "manual")
        self.store.save(self.profile)
        choices = choose_actions("Bass sound", "Bass level", "Bass warmth", "More warmth", "Back", "Back")
        with patch("phonix.cli.select_table", side_effect=choices), \
                patch("phonix.ui.select_table", side_effect=choices), \
                patch("builtins.input", return_value="4"):
            menu_profile_settings(self.store, "Headphones")
        saved = self.store.load("Headphones")
        self.assertEqual((saved["settings"]["bass_db"], saved["settings"]["bass_hz"]), (4, 180))
        self.assertEqual(saved["channels"], self.profile["channels"])

    def test_legacy_settings_browsing_is_read_only_and_extension_is_conditional(self):
        del self.profile["settings"]["filter_mode"]
        self.store.save(self.profile)
        original = self.store.path("Headphones").read_bytes()
        with patch("phonix.cli.select_table", return_value=None) as picker:
            menu_profile_settings(self.store, "Headphones")
        self.assertNotIn("Extend to 16000 Hz", [row[0] for row in picker.call_args.args[2]])
        self.assertEqual(self.store.path("Headphones").read_bytes(), original)
        self.store.create("Full range", Settings(min_hz=60, max_hz=15000))
        with patch("phonix.cli.select_table", return_value=None) as picker:
            menu_profile_settings(self.store, "Full range")
        self.assertIn("Extend to 16000 Hz", [row[0] for row in picker.call_args.args[2]])

    def test_home_help_and_tools_keep_their_selections_on_return(self):
        choices = choose_actions("Help", "Listening controls", None, "Tools", "Back", "Quit")
        with patch("phonix.ui.select_table", side_effect=choices) as picker, \
                patch("builtins.input", return_value=""), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            main_menu(self.store)
        calls = picker.call_args_list
        self.assertEqual(calls[2].args[3], 1)  # The same help topic remains selected.
        self.assertEqual(calls[3].args[3], 2)  # Home highlights Help on return.
        self.assertEqual(calls[5].args[3], 3)  # Home highlights Tools on return.
        self.assertIn("q or Escape returns one menu", output.getvalue())

    def test_failed_reset_stays_in_management_and_preserves_profile(self):
        original = self.store.path("Headphones").read_bytes()
        choices = choose_actions("Manage profile", "Reset", "Reset profile", None, None, None)
        with patch("phonix.cli.select_table", side_effect=choices) as picker, \
                patch.object(self.store, "reset", side_effect=OSError("Cannot save")), \
                patch("phonix.cli._menu_text", return_value="") as pause, \
                contextlib.redirect_stderr(io.StringIO()):
            browse_profiles(self.store, open_name="Headphones")
        self.assertIn("> Manage profile", picker.call_args_list[3].args[0])
        self.assertEqual(picker.call_args_list[3].args[3], 0)
        pause.assert_called_once_with("Enter to return to Manage profile: ")
        self.assertEqual(self.store.path("Headphones").read_bytes(), original)
