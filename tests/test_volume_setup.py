"""Pink-noise spectrum and session preparation checks without live playback."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from phonix.audio import Player, pair_samples, pink_noise
from phonix.cli import main, run_profile
from phonix.engine import add_observation
from phonix.profiles import Settings, Store


class PinkNoiseTests(unittest.TestCase):
    def test_noise_rms_ramps_stereo_and_peak_bounds_at_supported_rates(self):
        for rate in (44100, 48000, 96000):
            with self.subTest(rate=rate):
                samples = pink_noise(-20, rate=rate)
                self.assertEqual(samples.shape, (rate * 5, 2))
                np.testing.assert_array_equal(samples[:, 0], samples[:, 1])
                np.testing.assert_array_equal(samples[[0, -1]], 0)
                self.assertTrue(np.all(np.isfinite(samples)))
                self.assertAlmostEqual(float(np.sqrt(np.mean(samples ** 2))), 10 ** (-20 / 20), places=12)
                self.assertLessEqual(float(np.max(abs(samples))), 10 ** (-6 / 20))

    def test_pink_noise_has_similar_energy_per_octave(self):
        samples = pink_noise(-20)[:, 0]
        frequencies = np.fft.rfftfreq(len(samples), 1 / 48000)
        spectrum = abs(np.fft.rfft(samples)) ** 2
        bands = [(125, 250), (250, 500), (500, 1000), (1000, 2000),
                 (2000, 4000), (4000, 8000), (8000, 16000)]
        powers = np.array([spectrum[(frequencies >= low) & (frequencies < high)].sum() for low, high in bands])
        relative_db = 10 * np.log10(powers / powers.mean())
        self.assertLess(float(np.max(abs(relative_db))), 1.0)

    def test_ceiling_attenuates_noise_without_clipping_or_extra_gain(self):
        quiet = pink_noise(-42)
        limited = pink_noise(-12, ceiling=-6)
        self.assertLessEqual(float(np.max(abs(limited))), 10 ** (-6 / 20) + 1e-12)
        self.assertAlmostEqual(float(np.max(abs(limited))), 10 ** (-6 / 20), places=12)
        self.assertLess(float(np.sqrt(np.mean(limited ** 2))), 10 ** (-12 / 20))
        # Whole-signal scaling retains the spectrum; samples are not hard clipped.
        index = np.argmax(abs(quiet[:, 0]))
        ratio = limited[index, 0] / quiet[index, 0]
        np.testing.assert_allclose(limited, quiet * ratio, atol=1e-12)

    def test_replay_is_repeatable_and_invalid_parameters_rejected(self):
        np.testing.assert_array_equal(pink_noise(-42), pink_noise(-42))
        cases = [{"level_dbfs": -6}, {"level_dbfs": np.nan}, {"level_dbfs": -90},
                 {"level_dbfs": -42, "rate": 8000}, {"level_dbfs": -42, "duration": 0},
                 {"level_dbfs": -42, "duration": 11}, {"level_dbfs": -42, "duration": np.inf},
                 {"level_dbfs": -42, "ceiling": -60}]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                pink_noise(**values)

    def test_player_uses_current_configuration_and_existing_transport(self):
        settings = Settings(sample_rate=44100, reference_level=-36, max_level=-18, device="speaker-node",
                            noise_level=-24, noise_ceiling=-8)
        with patch("phonix.audio.shutil.which", return_value="pw-play"), \
                patch.object(Player, "_play_samples") as transport:
            player = Player(settings)
            player.play_pink_noise()
        samples = transport.call_args.args[0]
        np.testing.assert_array_equal(samples, pink_noise(-24, rate=44100, ceiling=-8))
        self.assertEqual(player.settings.device, "speaker-node")

    def test_music_volume_reference_is_independent_of_tone_attenuation(self):
        with patch("phonix.audio.shutil.which", return_value="pw-play"), \
                patch.object(Player, "_play_samples") as transport:
            Player(Settings()).play_music()
            first = transport.call_args.args[0]
            Player(Settings(reference_level=-60, max_level=-54, noise_level=-30, noise_ceiling=-12)).play_music()
            second = transport.call_args.args[0]
        np.testing.assert_array_equal(first, second)
        self.assertGreater(float(np.sqrt(np.mean(first ** 2))), 0)


class VolumeSetupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Store(Path(self.directory.name))

    def test_music_replay_precedes_first_comparison_and_records_no_judgments(self):
        profile = self.store.create("A", Settings(budget=1))
        with patch("phonix.cli.Player") as player, \
                patch("builtins.input", side_effect=["", "r", "c", "c", "e"]), \
                contextlib.redirect_stdout(io.StringIO()):
            run_profile(self.store, profile, 0)
        methods = [call[0] for call in player.return_value.method_calls]
        self.assertEqual(methods, ["play_music", "play_music", "play_tone", "play"])
        self.assertEqual(self.store.load("A")["channels"]["both"]["inputs"], 1)

    def test_quit_before_or_after_music_keeps_session_unchanged(self):
        for name, answers, noise_count in (("before", ["q"], 0), ("after", ["", "q"], 1)):
            with self.subTest(name=name):
                profile = self.store.create(name, Settings(budget=1))
                original = self.store.path(name).read_bytes()
                with patch("phonix.cli.Player") as player, \
                        patch("builtins.input", side_effect=answers), \
                        contextlib.redirect_stdout(io.StringIO()):
                    run_profile(self.store, profile, 0)
                self.assertEqual(player.return_value.play_music.call_count, noise_count)
                player.return_value.play.assert_not_called()
                self.assertEqual(self.store.path(name).read_bytes(), original)

    def test_music_interrupt_preserves_existing_judgments_and_resume_reminder(self):
        settings = Settings(budget=3)
        profile = self.store.create("A", settings)
        channel = profile["channels"]["both"]
        a, b = channel["frequencies"][:2]
        add_observation(channel, settings, a, b, 1, "manual")
        self.store.save(profile)
        with patch("phonix.cli.Player") as player, \
                patch("builtins.input", side_effect=[""]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            player.return_value.play_music.side_effect = KeyboardInterrupt()
            run_profile(self.store, profile, 0)
        self.assertIn("same volume used for your earlier judgments", output.getvalue())
        player.return_value.play.assert_not_called()
        self.assertEqual(self.store.load("A"), profile)

    def test_cannot_start_comparisons_without_playing_music_first(self):
        profile = self.store.create("A", Settings(budget=1))
        with patch("phonix.cli.Player") as player, \
                patch("builtins.input", side_effect=["c", "", "q"]), \
                contextlib.redirect_stdout(io.StringIO()):
            run_profile(self.store, profile, 0)
        player.return_value.play_music.assert_called_once()
        player.return_value.play.assert_not_called()
        self.assertEqual(self.store.load("A")["channels"]["both"]["inputs"], 0)

    def test_preview_override_does_not_modify_tones_budget_or_saved_profile(self):
        settings = Settings(budget=1)
        profile = self.store.create("A", settings)
        original = self.store.path("A").read_bytes()
        with patch("phonix.cli.Player") as player, \
                patch("builtins.input", side_effect=["", "q"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            run_profile(self.store, profile, 0, noise_level=-24, noise_ceiling=-8)
        playback = player.call_args.args[0]
        self.assertEqual((playback.noise_level, playback.noise_ceiling), (-24, -8))
        self.assertEqual((playback.reference_level, playback.max_level, playback.budget),
                         (settings.reference_level, settings.max_level, settings.budget))
        self.assertIn("Lower hardware/system volume before playing", output.getvalue())
        self.assertEqual(self.store.path("A").read_bytes(), original)

    def test_noise_options_cli_persistence_legacy_loading_and_validation(self):
        directory = str(self.store.root)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--data-dir", directory, "new", "A", "--noise-level", "-24", "--noise-ceiling", "-8"]), 0)
        profile = self.store.load("A")
        self.assertEqual(profile["settings"]["noise_level"], -24)
        self.assertEqual(profile["settings"]["noise_ceiling"], -8)
        original = self.store.path("A").read_bytes()
        with patch("phonix.cli.run_profile") as run:
            self.assertEqual(main(["--data-dir", directory, "run", "A", "--noise-level", "-22", "--noise-ceiling", "-5"]), 0)
            self.assertEqual(run.call_args.args[3:5], (-22, -5))
        self.assertEqual(self.store.path("A").read_bytes(), original)
        del profile["settings"]["noise_level"]
        del profile["settings"]["noise_ceiling"]
        self.store.save(profile)
        loaded = self.store.load("A")
        legacy_settings = Settings(**loaded["settings"]).validate()
        self.assertEqual((legacy_settings.noise_level, legacy_settings.noise_ceiling), (-20, -6))
        self.assertEqual(legacy_settings.reference_level, profile["settings"]["reference_level"])
        for values in ({"noise_level": np.nan}, {"noise_level": -6}, {"noise_ceiling": 0},
                       {"noise_ceiling": -30}, {"noise_level": -12, "noise_ceiling": -24}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Settings(**values).validate()

    def test_invalid_preview_override_cannot_play_or_modify_judgments(self):
        profile = self.store.create("A", Settings())
        original = self.store.path("A").read_bytes()
        with patch("phonix.cli.Player") as player, \
                self.assertRaisesRegex(ValueError, "Noise RMS level"):
            run_profile(self.store, profile, 1, noise_level=-6)
        player.assert_not_called()
        self.assertEqual(self.store.path("A").read_bytes(), original)

    def test_tone_check_after_music_is_replayable_and_records_no_judgments(self):
        profile = self.store.create("A", Settings(budget=1))
        original = self.store.path("A").read_bytes()
        with patch("phonix.cli.Player") as player, \
                patch("builtins.input", side_effect=["", "t", "t", "r", "q"]), \
                contextlib.redirect_stdout(io.StringIO()):
            run_profile(self.store, profile, 0)
        self.assertEqual([call[0] for call in player.return_value.method_calls],
                         ["play_music", "play_tone", "play_tone", "play_music"])
        self.assertEqual(self.store.path("A").read_bytes(), original)

    def test_empty_profile_tone_levels_persist_with_same_comparison_range(self):
        before = Settings()
        profile = self.store.create("A", before)
        with patch("phonix.cli.Player") as player, \
                patch("builtins.input", side_effect=["", "t", "q"]), \
                contextlib.redirect_stdout(io.StringIO()):
            run_profile(self.store, profile, 0, reference_level=-30, max_level=-12)
        playback = player.call_args.args[0]
        saved = Settings(**self.store.load("A")["settings"])
        self.assertEqual((saved.reference_level, saved.max_level), (-30, -12))
        self.assertEqual(saved.offset_limit, before.offset_limit)
        self.assertEqual((saved.noise_level, saved.noise_ceiling), (before.noise_level, before.noise_ceiling))
        np.testing.assert_allclose(pair_samples(1000, 2000, 18, False, "both", playback),
                                   pair_samples(1000, 2000, 18, False, "both", before) * 10 ** (12 / 20), atol=1e-12)
        player.return_value.play.assert_not_called()
        self.assertEqual(self.store.load("A")["channels"]["both"]["inputs"], 0)

    def test_tone_level_changes_rejected_after_either_channel_has_judgments(self):
        for name, changed in (("reference", {"reference_level": -36}), ("ceiling", {"max_level": -18})):
            with self.subTest(name=name):
                settings = Settings(channels="separate")
                profile = self.store.create(name, settings)
                channel = profile["channels"]["right"]
                a, b = channel["frequencies"][:2]
                add_observation(channel, settings, a, b, 1, "manual")
                self.store.save(profile)
                original = self.store.path(name).read_bytes()
                with patch("phonix.cli.Player") as player, \
                        self.assertRaisesRegex(ValueError, "after saved judgments"):
                    run_profile(self.store, profile, 1, **changed)
                player.assert_not_called()
                self.assertEqual(self.store.path(name).read_bytes(), original)
                # Explicitly repeating the saved levels is valid on resume.
                with patch("phonix.cli.Player"), patch("builtins.input", return_value="q"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    run_profile(self.store, profile, 0, reference_level=-42, max_level=-24)
                self.assertEqual(self.store.path(name).read_bytes(), original)

    def test_tone_level_cli_flags_and_invalid_settings_do_not_rewrite_profile(self):
        self.store.create("A", Settings())
        with patch("phonix.cli.Player") as player, patch("builtins.input", return_value="q"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--data-dir", str(self.store.root), "run", "A",
                                   "--reference-level", "-30", "--max-level", "-12"]), 0)
        self.assertEqual(player.call_args.args[0].reference_level, -30)
        original = self.store.path("A").read_bytes()
        with patch("phonix.cli.Player") as player, self.assertRaises(ValueError):
            run_profile(self.store, self.store.load("A"), 1, reference_level=-8, max_level=-2)
        player.assert_not_called()
        self.assertEqual(self.store.path("A").read_bytes(), original)

    def test_tone_check_interrupt_keeps_saved_judgments(self):
        settings = Settings()
        profile = self.store.create("A", settings)
        channel = profile["channels"]["both"]
        a, b = channel["frequencies"][:2]
        add_observation(channel, settings, a, b, 1, "manual")
        self.store.save(profile)
        with patch("phonix.cli.Player") as player, \
                patch("builtins.input", side_effect=["", "t"]), \
                contextlib.redirect_stdout(io.StringIO()):
            player.return_value.play_tone.side_effect = KeyboardInterrupt()
            run_profile(self.store, profile, 0)
        player.return_value.play.assert_not_called()
        self.assertEqual(self.store.load("A"), profile)
