"""Profile states, menu actions and keyboard navigation without live audio."""

import contextlib
import curses
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from phonix.cli import browse_profiles, list_profiles, main, menu_export, profile_rows, profile_summary
from phonix.engine import add_observation
from phonix.profiles import Settings, Store
from phonix.terminal import _select_table, select_table, table_lines


class ProfileMenuTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Store(Path(self.directory.name) / "profiles")

    def make_complete(self, name="ready"):
        settings = Settings(initial_points=4, max_points=4, budget=12)
        profile = self.store.create(name, settings)
        channel = profile["channels"]["both"]
        for _ in range(2):
            for index, a in enumerate(channel["frequencies"]):
                for b in channel["frequencies"][index + 1:]:
                    add_observation(channel, settings, a, b, 0, "manual")
        self.store.save(profile)
        return profile

    def test_states_use_observations_consistency_and_remaining_budget(self):
        settings = Settings(budget=1, initial_points=4, max_points=4)
        profile = self.store.create("test", settings)
        self.assertEqual(profile_summary(profile)["status"], "new")
        channel = profile["channels"]["both"]
        a, b = channel["frequencies"][:2]
        add_observation(channel, settings, a, b, 0, "manual")
        self.assertEqual(profile_summary(profile)["status"], "incomplete")
        profile["settings"]["budget"] = 2
        summary = profile_summary(profile)
        self.assertEqual(summary["status"], "unfinished")
        self.assertFalse(summary["connected"])
        self.assertEqual(summary["row"][4], "1 / 2")
        complete = profile_summary(self.make_complete())
        self.assertEqual(complete["status"], "complete")
        self.assertTrue(complete["connected"])

    def test_separate_channels_show_counts_and_both_must_finish(self):
        settings = Settings(channels="separate", budget=1)
        profile = self.store.create("ears", settings)
        right = profile["channels"]["right"]
        add_observation(right, settings, *right["frequencies"][:2], 0, "manual")
        summary = profile_summary(profile)
        self.assertEqual(summary["status"], "unfinished")
        self.assertEqual(summary["row"][3:5], ("Left / Right", "L 0/1; R 1/1"))
        left = profile["channels"]["left"]
        add_observation(left, settings, *left["frequencies"][:2], 0, "manual")
        self.assertEqual(profile_summary(profile)["status"], "incomplete")

    def test_plain_listing_table_and_unreadable_profile_do_not_modify_files(self):
        self.store.create("A", Settings(bass_db=4))
        self.store.path("broken").write_text("{")
        original = self.store.path("A").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()) as output, patch("phonix.cli.select_table") as picker:
            self.assertEqual(main(["--data-dir", str(self.store.root), "list", "--plain"]), 0)
        self.assertIn("Answers / limit", output.getvalue())
        self.assertIn("new", output.getvalue())
        self.assertIn("+4", output.getvalue())
        self.assertIn("unreadable", output.getvalue())
        picker.assert_not_called()
        self.assertEqual(self.store.path("A").read_bytes(), original)

    def test_terminal_listing_opens_browser_and_redirected_output_is_plain(self):
        self.store.create("A", Settings())
        with patch("phonix.cli.sys.stdin.isatty", return_value=True), \
                patch("phonix.cli.sys.stdout.isatty", return_value=True), \
                patch.dict("os.environ", {"TERM": "xterm"}), \
                patch("phonix.cli.browse_profiles") as browser:
            list_profiles(self.store)
            browser.assert_called_once_with(self.store)
        with patch("phonix.cli.sys.stdout.isatty", return_value=False), \
                patch("phonix.cli.browse_profiles") as browser, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            list_profiles(self.store)
            browser.assert_not_called()
        self.assertIn("Profile", output.getvalue())

    def test_empty_listing_returns_without_picker(self):
        with patch("phonix.cli.select_table") as picker, contextlib.redirect_stdout(io.StringIO()) as output:
            browse_profiles(self.store)
        picker.assert_not_called()
        self.assertIn("No saved profiles", output.getvalue())

    def test_direct_open_uses_shared_actions_and_back_highlights_selected_profile(self):
        self.store.create("A", Settings())
        self.store.create("Z", Settings())
        with patch("phonix.cli.select_table", side_effect=[6, None]) as picker:
            browse_profiles(self.store, open_name="Z")
        self.assertTrue(picker.call_args_list[0].args[0].startswith("Z | new |"))
        self.assertEqual(picker.call_args_list[0].args[2][6][0], "Back")
        self.assertEqual(picker.call_args_list[1].args[0], "Device profiles")
        self.assertEqual(picker.call_args_list[1].args[3], 1)

    def test_delete_cancel_keeps_profile_and_confirm_removes_only_selected_file(self):
        self.store.create("A", Settings())
        self.store.create("B", Settings())
        original = self.store.path("A").read_bytes()
        export = Path(self.directory.name) / "B.json"
        export.write_text("existing export\n")
        with patch("phonix.cli.select_table", side_effect=[1, 5, 1, 0, None, None, None]):
            browse_profiles(self.store)
        self.assertEqual(self.store.names(), ["A", "B"])
        with patch("phonix.cli.select_table", side_effect=[1, 5, 1, 1, None]):
            browse_profiles(self.store)
        self.assertEqual(self.store.names(), ["A"])
        self.assertEqual(self.store.path("A").read_bytes(), original)
        self.assertEqual(export.read_text(), "existing export\n")
        with self.assertRaises(ValueError):
            self.store.delete("../outside")
        with self.assertRaisesRegex(ValueError, "No profile"):
            self.store.delete("B")

    def test_start_and_continue_dispatch_then_refresh_saved_progress(self):
        profile = self.store.create("A", Settings())
        def update(store, loaded, extra):
            channel = loaded["channels"]["both"]
            add_observation(channel, Settings(**loaded["settings"]), *channel["frequencies"][:2], 0, "manual")
            store.save(loaded)
        with patch("phonix.cli.select_table", side_effect=[0, 0, 0, None, None]) as picker, \
                patch("phonix.cli.run_profile", side_effect=update) as run, \
                patch("builtins.input", return_value=""):
            browse_profiles(self.store)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.args[2], 0)
        self.assertEqual(picker.call_args_list[1].args[2][0][0], "Start calibration")
        self.assertEqual(picker.call_args_list[2].args[2][0][0], "Continue calibration")
        self.assertEqual(self.store.load("A")["channels"]["both"]["inputs"], 2)

    def test_exhausted_continue_requests_extra_budget_and_cancel_keeps_it(self):
        settings = Settings(budget=1)
        profile = self.store.create("A", settings)
        channel = profile["channels"]["both"]
        add_observation(channel, settings, *channel["frequencies"][:2], 0, "manual")
        self.store.save(profile)
        with patch("phonix.cli.select_table", side_effect=[0, 0, 0, None, None]), \
                patch("phonix.cli.run_profile") as run, \
                patch("builtins.input", side_effect=["q", "24", ""]):
            browse_profiles(self.store)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[2], 24)

    def test_complete_profile_keeps_daily_actions_and_groups_settings_and_management(self):
        self.make_complete()
        with patch("phonix.cli.select_table", side_effect=[0, None, None]) as picker:
            browse_profiles(self.store)
        self.assertEqual([row[0] for row in picker.call_args_list[1].args[2]],
                         ["Progress", "Preview EQ", "Export to Easy Effects", "Settings", "Manage profile", "Back"])

    def test_menu_export_guards_connection_incomplete_and_overwrite(self):
        profile = self.store.create("A", Settings())
        with patch("phonix.cli.export_profile") as export, \
                self.assertRaisesRegex(ValueError, "connect all"):
            menu_export(profile, profile_summary(profile))
        export.assert_not_called()
        summary = {"connected": True, "status": "unfinished"}
        with patch("phonix.cli._menu_confirm", return_value=False), \
                patch("phonix.cli.export_profile") as export:
            menu_export(profile, summary)
        export.assert_not_called()
        output = Path(self.directory.name) / "Custom Preset.json"
        output.with_suffix(".report.json").write_text("existing report\n")
        with patch("phonix.cli._menu_confirm", side_effect=[True, False]), \
                patch("builtins.input", return_value=str(output)), patch("phonix.cli.export_profile") as export:
            menu_export(profile, summary)
        export.assert_not_called()
        with patch("phonix.cli._menu_confirm", return_value=True), \
                patch("builtins.input", return_value=str(output)), patch("phonix.cli.export_profile") as export:
            menu_export(profile, summary)
        export.assert_called_once_with(profile, output, count=12, allow_incomplete=True, overwrite=True)

    def test_complete_menu_export_dispatch_and_return_to_profile(self):
        self.make_complete()
        with patch("phonix.cli.select_table", side_effect=[0, 2, None, None]), \
                patch("phonix.cli.export_profile") as export, \
                patch("builtins.input", side_effect=["", ""]):
            browse_profiles(self.store)
        export.assert_called_once()
        self.assertEqual(export.call_args.args[1], Path("exports/ready.json"))
        self.assertFalse(export.call_args.kwargs["allow_incomplete"])


class TerminalTableTests(unittest.TestCase):
    def screen(self, keys, sizes=None):
        screen = Mock()
        screen.getmaxyx.return_value = (10, 80)
        if sizes:
            screen.getmaxyx.side_effect = sizes
        screen.get_wch.side_effect = keys
        return screen

    def pick(self, screen, rows, initial=0):
        with patch("phonix.terminal.curses.curs_set"):
            return _select_table(screen, "Profiles", ("Name", "Status"), rows, initial)

    def test_table_alignment_and_control_characters(self):
        lines = table_lines(("Name", "Status"), [("A", "new"), ("long name", "unfinished"), ("bad\x1b\n", "unreadable")])
        separators = [line.index("|") for line in [lines[0], *lines[2:]]]
        self.assertEqual(len(set(separators)), 1)
        self.assertNotIn("\x1b", "".join(lines))
        self.assertNotIn("\n", "".join(lines))

    def test_arrow_enter_selection_and_bounds(self):
        rows = [("A", "new"), ("B", "unfinished"), ("C", "complete")]
        screen = self.screen([curses.KEY_DOWN, curses.KEY_DOWN, curses.KEY_DOWN, curses.KEY_UP, "\n"])
        self.assertEqual(self.pick(screen, rows), 1)
        self.assertTrue(any(call.args[-1] == curses.A_REVERSE for call in screen.addnstr.call_args_list))
        self.assertEqual(self.pick(self.screen([curses.KEY_UP, "\n"]), rows), 0)

    def test_scroll_resize_escape_and_interrupt(self):
        rows = [(f"Profile {index}", "unfinished") for index in range(20)]
        self.assertEqual(self.pick(self.screen([curses.KEY_END, "\n"]), rows), 19)
        self.assertIsNone(self.pick(self.screen([curses.KEY_RIGHT, curses.KEY_LEFT, "\x1b"]), rows))
        screen = self.screen([curses.KEY_RESIZE, curses.KEY_DOWN, "\n"], sizes=[(3, 10), (10, 80), (10, 80)])
        self.assertEqual(self.pick(screen, rows), 1)
        with self.assertRaises(KeyboardInterrupt):
            self.pick(self.screen(["\x03"]), rows)

    def test_wrapper_empty_and_terminal_error(self):
        with patch("phonix.terminal.curses.wrapper", side_effect=curses.error("no terminal")) as wrapper:
            self.assertIsNone(select_table("Empty", ("Name",), []))
            wrapper.assert_not_called()
            with self.assertRaisesRegex(ValueError, "list --plain"):
                select_table("Profiles", ("Name",), [("A",)])
