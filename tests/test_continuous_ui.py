"""One terminal session, captured output, editing, resize and restoration."""

import curses
import sys
import unittest
from unittest.mock import Mock, call, patch

from phonix import terminal


class ContinuousUITests(unittest.TestCase):
    def screen(self, keys):
        screen = Mock()
        screen.getmaxyx.return_value = (14, 80)
        screen.get_wch.side_effect = keys
        return screen

    def launch(self, screen, callback):
        def wrapper(function, *args):
            return function(screen, *args)
        with patch("phonix.terminal.curses.wrapper", side_effect=wrapper) as wrapped, \
                patch("phonix.terminal.curses.curs_set"):
            result = terminal.run_ui(callback)
        self.assertEqual(wrapped.call_count, 1)
        self.assertIsNone(terminal._ACTIVE)
        return result

    def test_menu_text_and_output_share_one_screen_and_restore_streams(self):
        screen = self.screen([curses.KEY_DOWN, "\n", "n", "a", "m", "e",
                              curses.KEY_LEFT, curses.KEY_BACKSPACE, "x", "\n", "\x1b"])
        original = sys.stdout, sys.stderr
        def flow():
            print("A report stays in the UI")
            print("Errors stay here too", file=sys.stderr)
            selected = terminal.select_table("Profiles", ("Name",), [("A",), ("B",)])
            value = terminal.read_input("Profile name: ")
            cancelled = terminal.read_input("Optional value: ")
            return selected, value, cancelled
        self.assertEqual(self.launch(screen, flow), (1, "naxe", None))
        self.assertEqual((sys.stdout, sys.stderr), original)
        drawn = "\n".join(call.args[2] for call in screen.addnstr.call_args_list)
        self.assertIn("A report stays in the UI", drawn)
        self.assertIn("Errors stay here too", drawn)
        self.assertIn("Profile name", drawn)

    def test_scroll_resize_long_input_home_end_delete_and_control_sanitizing(self):
        screen = self.screen([curses.KEY_PPAGE, curses.KEY_NPAGE, curses.KEY_RESIZE,
                              "a", "b", curses.KEY_HOME, curses.KEY_DC, curses.KEY_END, "c", "\n"])
        def flow():
            for index in range(40):
                print(f"line {index} \x1b")
            return terminal.read_input("Edit a value: ")
        self.assertEqual(self.launch(screen, flow), "bc")
        drawn = "".join(call.args[2] for call in screen.addnstr.call_args_list)
        self.assertNotIn("\x1b", drawn)
        self.assertIn("line 39", drawn)

    def test_exception_and_interrupt_restore_active_session_and_streams(self):
        original = sys.stdout, sys.stderr
        for error in (KeyboardInterrupt(), ValueError("broken")):
            screen = self.screen([])
            def flow():
                raise error
            with self.assertRaises(type(error)), patch("phonix.terminal.curses.curs_set"):
                terminal._run_ui(screen, flow, ())
            self.assertIsNone(terminal._ACTIVE)
            self.assertEqual((sys.stdout, sys.stderr), original)

    def test_ctrl_c_and_tiny_terminal_escape_return_cleanly(self):
        with self.assertRaises(KeyboardInterrupt):
            self.launch(self.screen(["\x03"]), lambda: terminal.read_input("Name: "))
        self.assertIsNone(terminal._ACTIVE)
        screen = self.screen([curses.KEY_RESIZE, "\x1b"])
        screen.getmaxyx.return_value = (3, 10)
        self.assertIsNone(self.launch(screen, lambda: terminal.read_input("Name: ")))

    def test_cursor_follows_text_editing_after_footer_is_drawn(self):
        screen = self.screen(["a", "b", curses.KEY_LEFT, "x", curses.KEY_HOME,
                              curses.KEY_END, "\n"])
        positions = []
        def refreshed():
            positions.append(screen.move.call_args.args)
            self.assertEqual(screen.mock_calls[-2:], [call.move(*positions[-1]), call.refresh()])
        screen.refresh.side_effect = refreshed
        self.assertEqual(self.launch(screen, lambda: terminal.read_input("Name: ")), "axb")
        self.assertEqual(positions, [(11, 2), (11, 3), (11, 4), (11, 3),
                                     (11, 4), (11, 2), (11, 5)])

    def test_input_cursor_handles_horizontal_scrolling_and_resize(self):
        screen = self.screen([])
        screen.getmaxyx.side_effect = [(9, 20), (3, 10), (15, 40)]
        session = terminal.TerminalSession(screen)
        session.prompt, session.value, session.cursor = "Name: ", "x" * 60, 20
        with patch("phonix.terminal.curses.curs_set") as visibility:
            for _ in range(3):
                session.render()
        self.assertEqual(screen.move.call_args_list, [call(6, 18), call(0, 0), call(12, 22)])
        self.assertEqual(visibility.call_args_list, [call(1), call(0), call(1)])

    def test_cursor_still_moves_when_terminal_cannot_change_visibility(self):
        screen = self.screen([])
        session = terminal.TerminalSession(screen)
        session.prompt, session.value, session.cursor = "Name: ", "abc", 1
        with patch("phonix.terminal.curses.curs_set", side_effect=curses.error("unsupported")):
            session.render()
            screen.move.assert_called_once_with(11, 3)
            screen.move.side_effect = curses.error("resized")
            session.render()
        self.assertEqual(screen.refresh.call_count, 2)

    def test_menu_cursor_tracks_highlighted_row_after_scrolling(self):
        screen = self.screen([curses.KEY_DOWN, curses.KEY_END, curses.KEY_LEFT, "\n"])
        screen.getmaxyx.return_value = (10, 30)
        positions = []
        def refreshed():
            positions.append(screen.move.call_args.args)
            self.assertEqual(screen.mock_calls[-2:], [call.move(*positions[-1]), call.refresh()])
        screen.refresh.side_effect = refreshed
        rows = [(str(index),) for index in range(10)]
        with patch("phonix.terminal.curses.curs_set") as visibility:
            self.assertEqual(terminal._select_table(screen, "Profiles", ("Name",), rows, 0), 9)
        self.assertEqual(positions, [(3, 0), (4, 0), (6, 0), (6, 0)])
        self.assertTrue(all(arguments == call(0) for arguments in visibility.call_args_list))

    def test_selected_description_is_visible_without_scrolling_wide_columns(self):
        screen = self.screen([curses.KEY_DOWN, "\n"])
        screen.getmaxyx.return_value = (10, 40)
        rows = [("Output", "physical-output-name" * 3, "Where tones will play"),
                ("Listening effort", "150 answers/channel", "May finish before this limit")]
        with patch("phonix.terminal.curses.curs_set"):
            self.assertEqual(terminal._select_table(screen, "Setup", ("Setting", "Value", "Details"),
                                                   rows, 0, -1), 1)
        footer = [entry.args[2] for entry in screen.addnstr.call_args_list if entry.args[0] == 7]
        self.assertEqual(footer, ["Where tones will play", "May finish before this limit"])

    def test_escape_delay_is_shortened_and_restored_across_nested_ui(self):
        screen = self.screen(["\x1b"])
        def flow():
            self.assertEqual(curses.get_escdelay(), 50)
            return terminal.run_ui(lambda: terminal.read_input("Name: "))
        original = curses.get_escdelay()
        self.assertIsNone(self.launch(screen, flow))
        self.assertEqual(curses.get_escdelay(), original)

    def test_escape_delay_restored_on_error_or_interrupt(self):
        original = curses.get_escdelay()
        for error in (ValueError("broken"), KeyboardInterrupt()):
            def flow():
                self.assertEqual(curses.get_escdelay(), 50)
                raise error
            with self.assertRaises(type(error)):
                self.launch(self.screen([]), flow)
            self.assertEqual(curses.get_escdelay(), original)

    def test_standalone_menu_shortens_delay_and_restores_it_on_back(self):
        screen = self.screen(["\x1b"])
        def wrapper(callback, *args):
            return callback(screen, *args)
        def read_key():
            self.assertEqual(curses.get_escdelay(), 50)
            return "\x1b"
        screen.get_wch.side_effect = read_key
        original = curses.get_escdelay()
        with patch("phonix.terminal.curses.wrapper", side_effect=wrapper), \
                patch("phonix.terminal.curses.curs_set"):
            self.assertIsNone(terminal.select_table("Profiles", ("Name",), [("A",)]))
        self.assertEqual(curses.get_escdelay(), original)


if __name__ == "__main__":
    unittest.main()
