"""Readable step screens and help preserve listening state; no live audio."""

import contextlib
import curses
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from phonix import terminal
from phonix.cli import compare, prepare_reference, run_profile
from phonix.engine import new_channel
from phonix.model import Reference
from phonix.music import music_reference_levels
from phonix.profiles import Settings, Store
import numpy as np


class CalibrationScreenTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name))

    def launch(self, callback, commands):
        screen = Mock()
        screen.getmaxyx.return_value = (24, 80)
        screen.get_wch.side_effect = [key for command in commands for key in [*command, "\n"]]
        painting, frames = {}, []
        screen.erase.side_effect = painting.clear
        def draw(row, column, text, width, style=0):
            painting[row] = (text[:width], style)
        screen.addnstr.side_effect = draw
        screen.refresh.side_effect = lambda: frames.append(dict(painting))
        def wrapper(function, *args):
            return function(screen, *args)
        with patch("phonix.terminal.curses.wrapper", side_effect=wrapper), \
                patch("phonix.terminal.curses.curs_set"):
            terminal.run_ui(callback)
        return frames

    def text(self, frame):
        return "\n".join(text for row, (text, _) in sorted(frame.items()))

    def test_three_steps_help_and_replay_keep_only_current_task_visible(self):
        profile = self.store.create("Screens", Settings(budget=1))
        commands = ["h", "", "", "h", "", "c", "h", "", "c", "h", "", "r", "e"]
        with patch("phonix.cli.Player") as player:
            frames = self.launch(lambda: run_profile(self.store, profile, 0), commands)
        methods = [entry[0] for entry in player.return_value.method_calls]
        self.assertEqual(methods, ["play_music", "play_tone", "play", "play"])
        self.assertEqual(self.store.load("Screens")["channels"]["both"]["inputs"], 1)
        for step in ("Step 1/3", "Step 2/3", "Step 3/3"):
            visible = [frame for frame in frames if step in frame[0][0]]
            self.assertTrue(visible)
            heading = "TONES: FIRST, THEN SECOND" if step == "Step 3/3" else "GOAL"
            self.assertTrue(any(heading in self.text(frame) and "NEXT ACTION" in self.text(frame) for frame in visible))
        references = [self.text(frame) for frame in frames if "Step 2/3" in frame[0][0]]
        self.assertTrue(all("Tone level: -42 dBFS peak" in text for text in references))
        self.assertTrue(any("confirm this tone is clearly audible and comfortable" in text for text in references))
        self.assertTrue(all("Equal checks:" not in text and "MUSIC PREVIEW" not in text for text in references))
        player.return_value.play_reference_match.assert_not_called()
        comparisons = [frame for frame in frames if "Step 3/3" in frame[0][0]]
        for frame in comparisons:
            self.assertIn("Current passage: Treble shape", frame[1][0])
            self.assertIn("Saved answers: 0 / 1 (maximum)", frame[2][0])
            self.assertEqual(frame[1][1], curses.A_BOLD)
            self.assertNotIn("Equal checks", self.text(frame))
            self.assertNotIn("MUSIC PREVIEW", self.text(frame))
        summary = self.text(frames[-1])
        self.assertIn("Profile status: incomplete", summary)
        self.assertIn("1 / 1 saved answers", summary)
        self.assertNotIn("matched level*", summary)  # Detailed results live under Progress.
        help_frames = [frame for frame in frames if "Calibration help | comparison" in frame[0][0]]
        self.assertTrue(help_frames)
        self.assertTrue(all("LISTENING CONTROLS" in self.text(frame) and "Guided: 1 first louder" in self.text(frame)
                            for frame in help_frames))

    def test_reference_level_indicator_follows_draft_adjustments(self):
        settings = Settings()
        player = Mock(settings=settings)
        frames = self.launch(lambda: prepare_reference(player, settings), ["+", "h", "", "q"])
        captions = [frame[1][0] for frame in frames if "Step 2/3" in frame[0][0]]
        self.assertTrue(any("Tone level: -42 dBFS peak" in text for text in captions))
        self.assertTrue(any("Tone level: -41 dBFS peak" in text for text in captions))
        self.assertEqual(player.play_tone.call_count, 2)
        self.assertEqual(player.settings, settings)

    def test_reference_controls_remain_visible_at_80_by_24_for_new_and_resumed_profiles(self):
        settings = Settings(**music_reference_levels())
        for resuming in (False, True):
            with self.subTest(resuming=resuming):
                player = Mock(settings=settings)
                frames = self.launch(lambda: prepare_reference(player, settings, resuming=resuming), ["q"])
                text = self.text(frames[-1])
                self.assertIn("OTHER CONTROLS", text)
                self.assertIn("r: replay", text)
                self.assertIn("h: help", text)
                self.assertIn("q:", text)
                self.assertIn("confirm this tone is clearly audible and comfortable", text)

    def test_adaptive_comparison_controls_and_shift_notice_fit_80_by_24(self):
        for mode in ("manual", "guided"):
            settings = Settings(reference_level=-9, max_level=-3, tone_leveling="adaptive-v1")
            channel = new_channel(settings)
            a, b = channel["frequencies"][-2:]
            player = Mock()
            frames = self.launch(lambda: compare(a, b, "both", channel, settings, player,
                                                Reference(), np.random.default_rng(1), mode), ["q"])
            text = self.text(frames[-1])
            self.assertIn("Pair shifted together", text)
            self.assertIn("dBFS", text)
            self.assertIn("ADJUST", text)
            self.assertIn("NEXT ACTION", text)
            self.assertIn("OTHER CONTROLS", text)
            self.assertIn("r: replay | s: skip pair | h: help | q: save and return", text)

    def test_manual_mode_updates_controls_without_replaying_or_saving_help(self):
        profile = self.store.create("Manual", Settings(budget=1))
        with patch("phonix.cli.Player") as player:
            frames = self.launch(lambda: run_profile(self.store, profile, 0),
                                 ["", "c", "c", "m", "h", "", "e"])
        player.return_value.play.assert_called_once()
        self.assertTrue(any("ADJUST" in self.text(frame) and "+/-0.5 dB" in self.text(frame)
                            for frame in frames if "Step 3/3" in frame[0][0]))
        observation = self.store.load("Manual")["channels"]["both"]["observations"][0]
        self.assertEqual(observation["kind"], "manual")

    def test_setup_cancel_still_records_no_answers_or_settings_changes(self):
        profile = self.store.create("Cancel", Settings())
        original = self.store.path("Cancel").read_bytes()
        with patch("phonix.cli.Player") as player:
            frames = self.launch(lambda: run_profile(self.store, profile, 0), ["h", "", "q"])
        player.return_value.play_music.assert_not_called()
        self.assertIn("Setup cancelled", self.text(frames[-1]))
        self.assertEqual(self.store.path("Cancel").read_bytes(), original)

    def test_passage_stays_pinned_when_scrolling_and_resizing(self):
        screen = Mock()
        screen.getmaxyx.return_value = (24, 80)
        session = terminal.TerminalSession(screen)
        with patch("phonix.terminal.curses.curs_set"):
            session.show_screen("Step 3/3 | Calibrate", [("Goal", [f"line {index}" for index in range(40)])],
                                ("Current passage: Regional balance", "Left ear | Saved answers: 4 / 150 (maximum)"))
            session.scroll = 15
            screen.getmaxyx.return_value = (14, 50)
            screen.addnstr.reset_mock()
            session.render()
        pinned = {entry.args[0]: entry.args[2] for entry in screen.addnstr.call_args_list if entry.args[0] < 3}
        self.assertIn("Step 3/3", pinned[0])
        self.assertEqual(pinned[1], "Current passage: Regional balance")
        self.assertIn("Left ear", pinned[2])

    def test_help_restores_screen_even_when_interrupted(self):
        screen = Mock()
        screen.getmaxyx.return_value = (24, 80)
        session = terminal.TerminalSession(screen)
        with patch("phonix.terminal.curses.curs_set"), \
                patch.object(terminal, "_ACTIVE", session):
            terminal.show_screen("Step 3/3", [("Goal", ("Original task",))], status=("Current passage: Bass shape",))
            original = (session.title, session.status, session.lines.copy(), session.section_headings.copy())
            with self.assertRaises(KeyboardInterrupt):
                with terminal.preserve_screen():
                    terminal.show_screen("Help", [("Details", ("Temporary help",))], status=())
                    raise KeyboardInterrupt
            self.assertEqual((session.title, session.status, session.lines, session.section_headings), original)

    def test_redirected_output_has_marked_sections_without_terminal_escape_codes(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            terminal.show_screen("Step 1/3", [("Goal", ("Set music volume",)), ("Next action", ("Play music",))],
                                 status=("Volume > Reference > Calibration",))
        text = output.getvalue()
        self.assertIn("\nGOAL\nSet music volume\n\nNEXT ACTION\n", text)
        self.assertNotIn("\x1b", text)
