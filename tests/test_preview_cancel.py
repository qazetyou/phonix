"""Interruptible music and terminal cleanup without live audio."""

import curses
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np

from phonix import terminal
from phonix.audio import Player
from phonix.cli import prepare_volume, run_profile
from phonix.profiles import Settings, Store


class PreviewTransportTests(unittest.TestCase):
    def process(self, replies, returncode=0):
        process = MagicMock()
        process.__enter__.return_value = process
        process.communicate.side_effect = replies
        process.returncode = returncode
        return process

    def play(self, process, cancel):
        with patch("phonix.audio.shutil.which", return_value="pw-play"), \
                patch("phonix.audio.subprocess.Popen", return_value=process) as launch:
            result = Player(Settings())._play_samples(np.zeros((48, 2)), cancel=cancel)
        self.assertFalse(Path(launch.call_args.args[0][-1]).exists())
        return result

    def test_successful_playback_returns_true_without_cancel_or_termination(self):
        process = self.process([(None, b"")])
        cancel = Mock(return_value=False)
        self.assertTrue(self.play(process, cancel))
        cancel.assert_not_called()
        process.terminate.assert_not_called()
        process.communicate.assert_called_once_with(timeout=0.05)

    def test_stop_terminates_reaps_and_removes_wav_without_reporting_signal_error(self):
        process = self.process([subprocess.TimeoutExpired("pw-play", .05), (None, b"")], -15)
        self.assertFalse(self.play(process, Mock(return_value=True)))
        process.terminate.assert_called_once()
        process.kill.assert_not_called()
        self.assertEqual(process.communicate.call_args_list, [call(timeout=.05), call(timeout=2)])

    def test_stalled_termination_kills_and_reaps(self):
        process = self.process([subprocess.TimeoutExpired("pw-play", .05),
                                subprocess.TimeoutExpired("pw-play", 2), (None, b"")], -9)
        self.assertFalse(self.play(process, Mock(return_value=True)))
        process.kill.assert_called_once()
        self.assertEqual(process.communicate.call_args_list[-1], call())

    def test_no_stop_keeps_polling_and_returns_on_natural_completion(self):
        process = self.process([subprocess.TimeoutExpired("pw-play", .05), (None, b"")])
        cancel = Mock(return_value=False)
        self.assertTrue(self.play(process, cancel))
        cancel.assert_called_once()
        process.terminate.assert_not_called()

    def test_callback_error_and_ctrl_c_reap_before_propagating(self):
        for error in (RuntimeError("input failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                process = self.process([subprocess.TimeoutExpired("pw-play", .05), (None, b"")])
                with self.assertRaises(type(error)):
                    self.play(process, Mock(side_effect=error))
                process.terminate.assert_called_once()
                self.assertEqual(process.communicate.call_args_list[-1], call(timeout=2))

    def test_total_timeout_still_applies_to_cancellable_playback(self):
        process = self.process([subprocess.TimeoutExpired("pw-play", .05), (None, b"")])
        with patch("phonix.audio.time.monotonic", side_effect=[0, 0, 16]), \
                self.assertRaises(subprocess.TimeoutExpired):
            self.play(process, Mock(return_value=False))
        process.terminate.assert_called_once()

    def test_playback_failures_are_not_mistaken_for_user_cancellation(self):
        process = self.process([(None, b"No such output")], 1)
        cancel = Mock(return_value=True)
        with self.assertRaisesRegex(ValueError, "No such output"):
            self.play(process, cancel)
        cancel.assert_not_called()


class PreviewControlTests(unittest.TestCase):
    def screen(self, keys):
        screen = Mock()
        screen.getmaxyx.return_value = (24, 80)
        screen.get_wch.side_effect = keys
        return screen

    def launch(self, screen, callback):
        def wrapper(function, *args):
            return function(screen, *args)
        with patch("phonix.terminal.curses.wrapper", side_effect=wrapper), \
                patch("phonix.terminal.curses.curs_set"):
            return terminal.run_ui(callback)

    def test_stop_keys_are_immediate_and_timeout_restored_after_every_poll(self):
        keys = ["\n", "\r", curses.KEY_ENTER, "\x1b", "s", "q", "x", curses.error()]
        screen = self.screen(keys)
        session = terminal.TerminalSession(screen)
        results = [session.poll_preview_stop() for _ in keys]
        self.assertEqual(results, [True] * 6 + [False, False])
        self.assertEqual(screen.timeout.call_args_list, [call(0), call(-1)] * len(keys))

    def test_resize_and_ctrl_c_restore_polling_mode(self):
        screen = self.screen([curses.KEY_RESIZE, "\x03"])
        session = terminal.TerminalSession(screen)
        with patch("phonix.terminal.curses.curs_set"):
            self.assertFalse(session.poll_preview_stop())
            with self.assertRaises(KeyboardInterrupt):
                session.poll_preview_stop()
        screen.refresh.assert_called_once()
        self.assertEqual(screen.timeout.call_args_list[-1], call(-1))

    def test_preview_footer_and_active_state_restore_on_failure(self):
        screen = self.screen([])
        session = terminal.TerminalSession(screen)
        with patch("phonix.terminal._ACTIVE", session), patch("phonix.terminal.curses.curs_set"):
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                with terminal.music_preview_controls() as cancel:
                    self.assertTrue(session.preview_active)
                    self.assertEqual(cancel, session.poll_preview_stop)
                    raise RuntimeError("stopped")
            self.assertFalse(session.preview_active)
        text = "\n".join(c.args[2] for c in screen.addnstr.call_args_list)
        self.assertIn("Enter, Esc or s: stop preview", text)
        self.assertIn("Enter: submit", text)

    def test_plain_terminal_stop_and_mode_restoration(self):
        previous = ["saved terminal mode"]
        with patch("phonix.terminal._ACTIVE", None), patch("phonix.terminal.sys.stdin") as stdin, \
                patch("termios.tcgetattr", return_value=previous), patch("tty.setcbreak") as cbreak, \
                patch("termios.tcsetattr") as restore, \
                patch("select.select", side_effect=[([], [], []), ([101], [], [])]), \
                patch("os.read", return_value=b"\x1b"):
            stdin.fileno.return_value = 101
            stdin.isatty.return_value = True
            with terminal.music_preview_controls() as cancel:
                self.assertFalse(cancel())
                self.assertTrue(cancel())
            cbreak.assert_called_once_with(101)
            self.assertEqual(restore.call_args.args[0], 101)
            self.assertEqual(restore.call_args.args[2], previous)

    def test_redirected_input_retains_blocking_transport(self):
        with patch("phonix.terminal._ACTIVE", None), patch("phonix.terminal.sys.stdin") as stdin:
            stdin.isatty.return_value = False
            with terminal.music_preview_controls() as cancel:
                self.assertIsNone(cancel)

    def test_early_stop_replay_and_return_preserve_profile(self):
        screen = self.screen(["\n", "s", "r", "\n", "\x1b", "q", "\n"])
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = store.create("Preview", Settings())
            original = store.path("Preview").read_bytes()
            def stop(cancel):
                self.assertTrue(cancel())
                return False
            with patch("phonix.cli.Player") as player:
                player.return_value.play_music.side_effect = stop
                self.launch(screen, lambda: run_profile(store, profile, 0))
            self.assertEqual(player.return_value.play_music.call_count, 2)
            player.return_value.play_tone.assert_not_called()
            self.assertEqual(store.path("Preview").read_bytes(), original)
        self.assertIsNone(terminal._ACTIVE)
        text = "\n".join(c.args[2] for c in screen.addnstr.call_args_list)
        self.assertIn("Music preview stopped early", text)

    def test_stopped_preview_can_continue_to_reference_check(self):
        cfg = Settings()
        screen = self.screen(["\n", "s", "c", "\n"])
        player = Mock()
        def stop(cancel):
            self.assertTrue(cancel())
            return False
        player.play_music.side_effect = stop
        with patch("phonix.cli.prepare_reference", return_value=cfg) as reference:
            result = self.launch(screen, lambda: prepare_volume(player, cfg))
        self.assertEqual(result, cfg)
        reference.assert_called_once()


if __name__ == "__main__":
    unittest.main()
