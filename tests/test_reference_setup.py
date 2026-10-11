"""Accepted tone setup, cancellation, bounds and resume without live audio."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from phonix.audio import pair_samples
from phonix.cli import prepare_reference, run_profile
from phonix.engine import add_observation
from phonix.profiles import Settings, Store


class ReferenceSetupTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name))

    def test_adjustment_is_saved_before_comparisons_and_preserves_offset_allowance(self):
        before = Settings(budget=1)
        profile = self.store.create("A", before)
        played = []
        with patch("phonix.cli.Player") as factory, \
                patch("builtins.input", side_effect=["", "c", "++", "+", "r", "c", "e"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            player = factory.return_value
            player.play_tone.side_effect = lambda: played.append(player.settings.reference_level)
            player.play.side_effect = lambda *args: self.assertEqual(
                self.store.load("A")["settings"]["reference_level"], -38)
            run_profile(self.store, profile, 0)
        self.assertEqual(played, [-42, -39, -38, -38])
        after = Settings(**self.store.load("A")["settings"])
        self.assertEqual((after.reference_level, after.max_level), (-38, -20))
        self.assertEqual(after.offset_limit, before.offset_limit)
        self.assertEqual((after.noise_level, after.noise_ceiling), (before.noise_level, before.noise_ceiling))
        self.assertEqual(profile["channels"]["both"]["inputs"], 1)
        self.assertIn("Step 2/3 | Check the reference tone", output.getvalue())
        self.assertIn("clearly audible and comfortable", output.getvalue())
        player.play_reference_match.assert_not_called()
        player.play_pink_noise.assert_not_called()
        np.testing.assert_allclose(pair_samples(1000, 2000, 18, False, "both", after),
                                   pair_samples(1000, 2000, 18, False, "both", before) * 10 ** (4 / 20), atol=1e-12)

    def test_cancel_or_interrupt_after_adjustment_keeps_profile_and_plays_no_pairs(self):
        for name, last in (("cancel", "q"), ("interrupt", KeyboardInterrupt()), ("eof", EOFError())):
            profile = self.store.create(name, Settings())
            original = self.store.path(name).read_bytes()
            with patch("phonix.cli.Player") as factory, \
                    patch("builtins.input", side_effect=["", "c", "++", last]), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_profile(self.store, profile, 0)
            factory.return_value.play.assert_not_called()
            self.assertEqual(self.store.path(name).read_bytes(), original)

    def test_resume_replays_saved_level_and_rejects_changes_after_either_ear_has_data(self):
        cfg = Settings(channels="separate", budget=1)
        profile = self.store.create("ears", cfg)
        channel = profile["channels"]["right"]
        add_observation(channel, cfg, *channel["frequencies"][:2], 0, "manual")
        self.store.save(profile)
        original = self.store.path("ears").read_bytes()
        levels = []
        with patch("phonix.cli.Player") as factory, \
                patch("builtins.input", side_effect=["", "c", "++", "r", "c", "q"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            player = factory.return_value
            player.play_tone.side_effect = lambda: levels.append(player.settings.reference_level)
            run_profile(self.store, profile, 0)
        self.assertEqual(levels, [-42, -42])
        self.assertIn("lock the tone level", output.getvalue())
        self.assertEqual(self.store.path("ears").read_bytes(), original)

    def test_failed_adjustment_playback_discards_draft_before_any_judgment(self):
        profile = self.store.create("A", Settings())
        original = self.store.path("A").read_bytes()
        with patch("phonix.cli.Player") as factory, \
                patch("builtins.input", side_effect=["", "c", "++"]), \
                contextlib.redirect_stdout(io.StringIO()):
            factory.return_value.play_tone.side_effect = [None, ValueError("Playback failed")]
            with self.assertRaisesRegex(ValueError, "Playback failed"):
                run_profile(self.store, profile, 0)
        factory.return_value.play.assert_not_called()
        self.assertEqual(self.store.path("A").read_bytes(), original)
        self.assertEqual(factory.return_value.settings.reference_level, -42)

    def test_bounds_preserve_allowance_when_possible_and_reduce_it_at_peak_cap(self):
        for settings, keys, expected, allowance in (
            (Settings(), ["++"] * 12, -9, 6),
            (Settings(), ["--"] * 8, -60, 18),
            (Settings(reference_level=-42, max_level=-12), ["++"] * 4, -30, 24),
        ):
            player = Mock(settings=settings)
            levels = []
            player.play_tone.side_effect = lambda: levels.append(player.settings)
            with patch("builtins.input", side_effect=keys + ["c"]), contextlib.redirect_stdout(io.StringIO()):
                result = prepare_reference(player, settings)
            self.assertEqual(result.reference_level, expected)
            self.assertEqual(result.offset_limit, allowance)
            self.assertTrue(all(item.max_level <= -3 for item in levels))
            self.assertTrue(all(item.reference_level >= -60 for item in levels))
            self.assertEqual(player.settings, settings)

    def test_confirming_setup_alone_does_not_record_judgments_or_save_noise_override(self):
        profile = self.store.create("A", Settings())
        with patch("phonix.cli.Player"), \
                patch("builtins.input", side_effect=["", "c", "+", "c", "q"]), \
                contextlib.redirect_stdout(io.StringIO()):
            run_profile(self.store, profile, 0, noise_level=-24, noise_ceiling=-8)
        loaded = self.store.load("A")
        self.assertEqual(loaded["channels"]["both"]["inputs"], 0)
        self.assertEqual(loaded["settings"]["reference_level"], -41)
        self.assertEqual((loaded["settings"]["noise_level"], loaded["settings"]["noise_ceiling"]), (-20, -6))

    def test_default_acceptance_needs_one_tone_and_no_equal_judgments(self):
        for command in ("", "c"):
            cfg = Settings()
            player = Mock(settings=cfg)
            with self.subTest(command=command), patch("builtins.input", side_effect=[command]), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(prepare_reference(player, cfg), cfg)
            player.play_tone.assert_called_once_with()
            player.play_reference_match.assert_not_called()
            self.assertIn("not barely detectable", output.getvalue())
            self.assertNotIn("Equal checks:", output.getvalue())

    def test_legacy_answers_never_adjust_accept_or_replay_the_tone(self):
        cfg = Settings()
        player = Mock(settings=cfg)
        with patch("builtins.input", side_effect=["1", "2", "e", "u", "invalid", "q"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertIsNone(prepare_reference(player, cfg))
        player.play_tone.assert_called_once_with()
        self.assertEqual(player.settings, cfg)
        self.assertIn("Confirm with Enter/c only", output.getvalue())

    def test_help_and_replay_keep_draft_level_without_automatic_gain(self):
        cfg = Settings()
        player = Mock(settings=cfg)
        heard = []
        player.play_tone.side_effect = lambda: heard.append(player.settings.reference_level)
        with patch("builtins.input", side_effect=["+", "h", "", "r", "c"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            selected = prepare_reference(player, cfg)
        self.assertEqual(heard, [-42, -41, -41])
        self.assertEqual(selected.reference_level, -41)
        self.assertEqual(player.settings, cfg)
        self.assertIn("not a hearing-threshold test", output.getvalue())

    def test_bound_feedback_never_raises_gain_or_saves_without_confirmation(self):
        cfg = Settings(reference_level=-9, max_level=-3)
        profile = self.store.create("limit", cfg)
        original = self.store.path("limit").read_bytes()
        with patch("phonix.cli.Player") as factory, \
                patch("builtins.input", side_effect=["", "c", "+", "q"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            run_profile(self.store, profile, 0)
        self.assertIn("Tone adjustment limit reached", output.getvalue())
        factory.return_value.play_tone.assert_called_once_with()
        self.assertEqual(self.store.path("limit").read_bytes(), original)
        factory.return_value.play.assert_not_called()

    def test_high_reference_is_saved_before_pairs_and_new_allowance_is_used(self):
        cfg = Settings(budget=1)
        profile = self.store.create("high", cfg)
        with patch("phonix.cli.Player") as factory, \
                patch("builtins.input", side_effect=["", "c"] + ["++"] * 9 + ["c", "e"]), \
                contextlib.redirect_stdout(io.StringIO()):
            player = factory.return_value
            def inspect_pair(*args):
                saved = Settings(**self.store.load("high")["settings"])
                self.assertEqual(saved.reference_level, -15)
                self.assertEqual(saved.max_level, -3)
                self.assertEqual(player.settings, saved)
                self.assertEqual(saved.offset_limit, 12)
            player.play.side_effect = inspect_pair
            run_profile(self.store, profile, 0)
        self.assertEqual(self.store.load("high")["channels"]["both"]["inputs"], 1)

    def test_cancelling_after_replay_still_discards_adjustment(self):
        profile = self.store.create("checked", Settings())
        original = self.store.path("checked").read_bytes()
        with patch("phonix.cli.Player") as factory, \
                patch("builtins.input", side_effect=["", "c", "++", "r", "q"]), \
                contextlib.redirect_stdout(io.StringIO()):
            run_profile(self.store, profile, 0)
        self.assertEqual(self.store.path("checked").read_bytes(), original)
        factory.return_value.play.assert_not_called()

    def test_initial_playback_failure_cannot_confirm_or_record_judgments(self):
        profile = self.store.create("failed", Settings())
        original = self.store.path("failed").read_bytes()
        with patch("phonix.cli.Player") as factory, \
                patch("builtins.input", side_effect=["", "c"]), \
                contextlib.redirect_stdout(io.StringIO()):
            factory.return_value.play_tone.side_effect = ValueError("Playback failed")
            with self.assertRaisesRegex(ValueError, "Playback failed"):
                run_profile(self.store, profile, 0)
        self.assertEqual(self.store.path("failed").read_bytes(), original)
        factory.return_value.play.assert_not_called()

    def test_saved_setup_without_judgments_is_reused_on_next_run(self):
        profile = self.store.create("prepared", Settings(reference_level=-38, max_level=-20))
        original = self.store.path("prepared").read_bytes()
        with patch("phonix.cli.Player") as factory, \
                patch("builtins.input", side_effect=["", "c", "c", "q"]), \
                contextlib.redirect_stdout(io.StringIO()):
            run_profile(self.store, profile, 0)
        factory.return_value.play_tone.assert_called_once_with()
        self.assertEqual(factory.return_value.settings.reference_level, -38)
        self.assertEqual(self.store.path("prepared").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
