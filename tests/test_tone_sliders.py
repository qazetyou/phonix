"""Silent arrow-key matching, replay, and comparison-relative slider checks."""

import copy
import curses
import unittest
from unittest.mock import Mock, patch

import numpy as np

from phonix import terminal
from phonix.audio import comparison_levels
from phonix.cli import compare
from phonix.engine import new_channel
from phonix.model import Reference
from phonix.profiles import Settings


class ToneSliderTests(unittest.TestCase):
    def launch(self, keys, settings=None, order=False, callback=None):
        settings = settings or Settings(min_hz=500, max_hz=2000, tone_leveling="adaptive-v1")
        channel = new_channel(settings)
        a, b = channel["frequencies"][-2:]
        screen = Mock()
        screen.getmaxyx.return_value = (24, 80)
        screen.get_wch.side_effect = keys
        painting, frames = {}, []
        screen.erase.side_effect = painting.clear
        screen.addnstr.side_effect = lambda row, col, text, width, style=0: painting.update({row: text[:width]})
        screen.refresh.side_effect = lambda: frames.append("\n".join(text for _, text in sorted(painting.items())))
        player = Mock()
        original = copy.deepcopy(channel)
        def task():
            terminal.show_screen("Step 3/4 | Calibrate", [],
                                 status=("Current passage: Treble shape", "Both | Saved answers: 0 / 150 (maximum)"))
            def pair():
                return compare(a, b, "both", channel, settings, player, Reference(),
                               np.random.default_rng(1), "guided", order=order)
            return callback(pair) if callback else pair()
        with patch("phonix.terminal.curses.curs_set"):
            result = terminal._run_ui(screen, task, ())
        self.assertEqual(channel, original)  # Only accepted observations may change evidence.
        self.assertIsNone(terminal._ACTIVE)
        return result, frames, player, settings

    def test_arrows_change_half_db_without_playing_then_enter_replays(self):
        (observation, mode), frames, player, settings = self.launch(
            [curses.KEY_UP, curses.KEY_UP, curses.KEY_DOWN, "\n", "e", "\n"])
        self.assertEqual(player.play.call_count, 2)
        first, last = player.play.call_args_list
        self.assertAlmostEqual(last.args[2] - first.args[2], .5)
        self.assertEqual(mode, "manual")
        self.assertEqual(observation[1:3], ("manual", "equal"))
        self.assertAlmostEqual(observation[0], last.args[2])
        peaks = comparison_levels(observation[0], settings, last.kwargs["base_level"])
        self.assertEqual((observation[3]["a_dbfs"], observation[3]["b_dbfs"]), peaks)
        self.assertTrue(any("+1.0 dB since start" in frame for frame in frames))
        self.assertTrue(any("+0.5 dB since start" in frame for frame in frames))
        self.assertTrue(any("Changes pending" in frame for frame in frames))
        self.assertNotIn("Changes pending", frames[-1])

    def test_pending_changes_require_listening_before_acceptance(self):
        (observation, mode), frames, player, _ = self.launch(
            [curses.KEY_DOWN, "e", "\n", "\n", "e", "\n"])
        self.assertEqual(player.play.call_count, 2)
        self.assertTrue(any("before recording an answer" in frame for frame in frames))
        self.assertEqual(observation[1], "manual")
        self.assertEqual(mode, "manual")

    def test_cancel_discards_unplayed_changes(self):
        result, _, player, _ = self.launch([curses.KEY_UP, "\x1b"])
        self.assertEqual(result, ("quit", "manual"))
        player.play.assert_called_once()

    def test_reversed_order_marks_first_tone_but_retains_canonical_offset(self):
        (observation, _), frames, player, _ = self.launch(
            [curses.KEY_UP, "\n", "e", "\n"], order=True)
        a, b = player.play.call_args.args[:2]
        titles = next(frame for frame in frames if "ADJUST" in frame)
        frequency_line = next(line for line in titles.splitlines() if f"> {b:g} Hz" in line)
        self.assertLess(frequency_line.index(f"> {b:g} Hz"), frequency_line.index(f"{a:g} Hz"))
        self.assertTrue(player.play.call_args.args[3])
        self.assertAlmostEqual(observation[0] - player.play.call_args_list[0].args[2], .5)

    def test_both_peaks_remain_within_limits_and_changes_use_actual_peaks(self):
        settings = Settings(reference_level=-9, max_level=-3, tone_leveling="adaptive-v1")
        (observation, _), frames, player, _ = self.launch(
            [*[curses.KEY_UP] * 40, "\n", "e", "\n"], settings=settings)
        self.assertEqual(observation[0], settings.offset_limit)
        self.assertEqual(player.play.call_count, 2)
        initial, final = [comparison_levels(call.args[2], settings, call.kwargs["base_level"])
                          for call in player.play.call_args_list]
        for peak in final:
            self.assertLessEqual(peak, settings.max_level)
            self.assertGreaterEqual(peak, -78)
        for before, after in zip(initial, final):
            self.assertIn(f"{after - before:+.1f} dB since start", frames[-1])
        self.assertTrue(any("Level limit reached" in frame for frame in frames))
        self.assertIn("Pair shifted together", frames[-1])

    def test_help_restores_sliders_and_does_not_replay(self):
        (_, mode), frames, player, _ = self.launch(
            [curses.KEY_UP, "h", "\n", "\n", "\n", "e", "\n"])
        self.assertEqual(mode, "manual")
        self.assertEqual(player.play.call_count, 2)
        self.assertTrue(any("Calibration help | comparison" in frame for frame in frames))
        self.assertTrue(any("ADJUST" in frame and "+0.5 dB since start" in frame for frame in frames))
        self.assertIn("Up/Down: +/-0.5 dB", frames[-1])

    def test_starting_changes_reset_for_each_pair(self):
        result, frames, player, _ = self.launch(
            [curses.KEY_UP, "\n", "e", "\n", curses.KEY_DOWN, "\n", "e", "\n"],
            callback=lambda pair: (pair(), pair()))
        self.assertEqual(player.play.call_count, 4)
        self.assertTrue(any("-0.5 dB since start" in frame for frame in frames))
        first, second = (item[0] for item in result)
        self.assertAlmostEqual(first[0] - second[0], 1)

    def test_sliders_and_controls_fit_standard_terminal(self):
        _, frames, _, _ = self.launch(["q", "\n"])
        for frame in frames:
            if "ADJUST" not in frame:
                continue
            self.assertIn("Reference", frame)
            self.assertIn("[O]", frame)
            self.assertIn("NEXT ACTION", frame)
            self.assertIn("OTHER CONTROLS", frame)
            self.assertIn("Up/Down: +/-0.5 dB", frame)
            self.assertIn("Enter: play pair", frame)
            self.assertIn("Current passage: Treble shape", frame)

    def test_shared_scale_moves_marker_at_half_db_and_expands(self):
        before = terminal.tone_slider_lines((1000, 1500), (-42, -40), (0, 0), 1)
        after = terminal.tone_slider_lines((1000, 1500), (-42, -39.5), (0, .5), 1)
        self.assertNotEqual(before[2:7], after[2:7])
        expanded = terminal.tone_slider_lines((1000, 1500), (-42, -28), (0, 12), 1)
        self.assertIn("+12", expanded[2])
        self.assertIn("+12.0 dB since start", expanded[-2])

    def test_tick_decimals_and_expanded_scale_keep_both_rails_aligned(self):
        for changes in ((0, 0), (0, .5), (-.5, 1.5), (0, 12), (-24, 24)):
            with self.subTest(changes=changes):
                lines = terminal.tone_slider_lines((1000, 1500), (-42, -40), changes, 1)
                for line in lines[2:7]:
                    for cell in (line[:29], line[29:]):
                        symbol = next(value for value in ("[O]", "-+-", " | ") if value in cell)
                        self.assertEqual(cell.index(symbol) + 1, 14)

    def test_plain_input_still_supports_manual_commands_and_empty_enter_replay(self):
        settings = Settings()
        channel = new_channel(settings)
        player = Mock()
        with patch("phonix.cli._input", side_effect=["+", "", "e"]), patch("builtins.print"):
            observation, _ = compare(*channel["frequencies"][-2:], "both", channel, settings,
                                     player, Reference(), np.random.default_rng(1), "manual", order=False)
        self.assertEqual(player.play.call_count, 3)  # Text fallback keeps automatic +/- replay.
        self.assertEqual(observation[1], "manual")

    def test_resize_and_tiny_terminal_keep_escape_available(self):
        screen = Mock()
        screen.getmaxyx.return_value = (4, 10)
        screen.get_wch.side_effect = [curses.KEY_RESIZE, curses.KEY_UP, "\x1b"]
        with patch("phonix.terminal.curses.curs_set"):
            def task():
                terminal.show_screen("Step 3/4", [("Tones", ("Test",))], comparison=True)
                return terminal.read_input("Your choice > ")
            self.assertIsNone(terminal._run_ui(screen, task, ()))


if __name__ == "__main__":
    unittest.main()
