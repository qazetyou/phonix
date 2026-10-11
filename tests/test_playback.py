"""Playback transport and troubleshooting tests without real audio."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import wave

import numpy as np

from phonix.audio import (PLAYBACK_LEAD_IN_SECONDS, Player, pair_samples,
                                    pink_noise, reference_band_noise, reference_match_samples, tone)
from phonix.cli import check_audio, main
from phonix.engine import add_observation
from phonix.profiles import Settings, Store


class PlaybackTests(unittest.TestCase):
    def test_check_writes_nonzero_ramped_stereo_wav_and_passes_selected_target(self):
        process = MagicMock()
        process.__enter__.return_value = process
        process.communicate.return_value = (None, b"")
        process.returncode = 0
        captured = {}

        def inspect_command(command, **kwargs):
            target_index = command.index("--target")
            self.assertEqual(command[target_index + 1], "speaker-node")
            properties = dict(part.split("=", 1) for part in command[command.index("--properties") + 1].split())
            self.assertEqual(properties["node.dont-move"], "true")
            self.assertEqual(properties["node.dont-fallback"], "true")
            self.assertEqual(properties["application.name"], "phonix")
            with wave.open(command[-1], "rb") as stream:
                self.assertEqual((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()),
                                 (2, 2, 48000, 72000))
                captured["samples"] = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2)
            captured["path"] = Path(command[-1])
            return process

        with patch("phonix.audio.shutil.which", return_value="/usr/bin/pw-play"), \
                patch("phonix.audio.subprocess.Popen", side_effect=inspect_command):
            Player(Settings(device="speaker-node")).play_tone()
        samples = captured["samples"]
        self.assertGreater(np.max(abs(samples)), 250)
        self.assertLessEqual(np.max(abs(samples)), round(32767 * 10 ** (-42 / 20)))
        np.testing.assert_array_equal(samples[:, 0], samples[:, 1])
        np.testing.assert_array_equal(samples[:24000], 0)
        np.testing.assert_array_equal(samples[[0, -1]], 0)
        self.assertFalse(captured["path"].exists())

    def test_every_stream_and_replay_has_silent_lead_in_without_changing_audio(self):
        process = MagicMock()
        process.__enter__.return_value = process
        process.communicate.return_value = (None, b"")
        process.returncode = 0
        for rate in (44100, 48000, 96000):
            cfg = Settings(sample_rate=rate)
            cases = [("tone", channel) for channel in ("both", "left", "right")]
            cases += [("pair", False), ("pair", True), ("noise", None)]
            cases += [("match", False), ("match", True)]
            for kind, choice in cases:
                with self.subTest(rate=rate, kind=kind, choice=choice):
                    if kind == "tone":
                        expected = tone(1000, cfg.reference_level, choice, rate, duration=1,
                                        ceiling=cfg.max_level)
                    elif kind == "pair":
                        expected = pair_samples(20, 1000, 3, choice, "left", cfg)
                    elif kind == "match":
                        expected = reference_match_samples(cfg, choice)
                    else:
                        expected = pink_noise(cfg.noise_level, rate, ceiling=cfg.noise_ceiling)
                    lead_frames = int(rate * PLAYBACK_LEAD_IN_SECONDS)

                    def inspect_command(command, **kwargs):
                        with wave.open(command[-1], "rb") as stream:
                            self.assertEqual(stream.getnframes(), lead_frames + len(expected))
                            self.assertEqual(stream.getframerate(), rate)
                            pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2)
                        np.testing.assert_array_equal(pcm[:lead_frames], 0)
                        np.testing.assert_array_equal(pcm[lead_frames:], np.rint(expected * 32767).astype("<i2"))
                        return process

                    with patch("phonix.audio.shutil.which", return_value="/usr/bin/pw-play"), \
                            patch("phonix.audio.subprocess.Popen", side_effect=inspect_command) as launch:
                        player = Player(cfg)
                        for _ in range(2):
                            if kind == "tone":
                                player.play_tone(channel=choice)
                            elif kind == "pair":
                                player.play(20, 1000, 3, choice, "left")
                            elif kind == "match":
                                player.play_reference_match(choice)
                            else:
                                player.play_pink_noise()
                        self.assertEqual(launch.call_count, 2)

    def test_reference_matching_keeps_preview_gain_equal_durations_fades_and_order(self):
        for rate in (44100, 48000, 96000):
            with self.subTest(rate=rate):
                cfg = Settings(sample_rate=rate, reference_level=-15, max_level=-3,
                               noise_level=-12, noise_ceiling=-6)
                preview = pink_noise(cfg.noise_level, rate, ceiling=cfg.noise_ceiling)
                count, gap = 2 * rate, int(rate * 0.25)
                direct = reference_match_samples(cfg, True)
                reverse = reference_match_samples(cfg, False)
                self.assertEqual(direct.shape, (2 * count + 2 * gap, 2))
                np.testing.assert_array_equal(direct[:count], reference_band_noise(cfg))
                self.assertLess(np.mean(direct[:count] ** 2), np.mean(preview ** 2))
                np.testing.assert_array_equal(direct[:count], reverse[count + gap:2 * count + gap])
                np.testing.assert_array_equal(direct[count + gap:2 * count + gap], reverse[:count])
                np.testing.assert_array_equal(reverse[:count], tone(1000, -15, "both", rate, duration=2, ceiling=-3))
                np.testing.assert_array_equal(direct[count:count + gap], 0)
                np.testing.assert_array_equal(direct[2 * count + gap:], 0)
                np.testing.assert_array_equal(direct[[0, count - 1, count + gap, 2 * count + gap - 1]], 0)
                self.assertLessEqual(abs(direct[:count]).max(), 10 ** (cfg.noise_ceiling / 20))
                self.assertLessEqual(abs(reverse[:count]).max(), 10 ** (cfg.reference_level / 20))

    def test_reference_band_preserves_in_band_gain_and_rejects_other_frequencies(self):
        for rate in (44100, 48000, 96000):
            with self.subTest(rate=rate):
                cfg = Settings(sample_rate=rate)
                t = np.arange(5 * rate) / rate
                # Known independent input: three pass-band and two rejected tones.
                source = sum(0.01 * np.sin(2 * np.pi * f * t)
                             for f in (500, 950, 1000, 1050, 1500))
                preview = np.column_stack((source, source))
                expected = sum(tone(f, -40, "both", rate, duration=2)
                               for f in (950, 1000, 1050))
                with patch("phonix.audio.pink_noise", return_value=preview) as generate:
                    actual = reference_band_noise(cfg)
                generate.assert_called_once_with(cfg.noise_level, rate, ceiling=cfg.noise_ceiling)
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)

    def test_reference_band_is_quieter_and_concentrated_around_one_khz(self):
        for rate in (44100, 48000, 96000):
            with self.subTest(rate=rate):
                cfg = Settings(sample_rate=rate, noise_level=-17.7, noise_ceiling=-4)
                band = reference_band_noise(cfg)
                preview = pink_noise(cfg.noise_level, rate, ceiling=cfg.noise_ceiling)
                self.assertTrue(np.isfinite(band).all())
                np.testing.assert_array_equal(band[:, 0], band[:, 1])
                np.testing.assert_array_equal(band[[0, -1]], 0)
                self.assertLess(np.mean(band ** 2), 0.05 * np.mean(preview ** 2))
                frequencies = np.fft.rfftfreq(len(band), 1 / rate)
                energy = abs(np.fft.rfft(band[:, 0])) ** 2
                # Time fades cause small spectral leakage outside the filter band.
                outside = (frequencies < 900) | (frequencies > 1100)
                self.assertLess(energy[outside].sum(), 0.001 * energy.sum())

    def test_reference_band_follows_preview_gain_including_peak_attenuation(self):
        for rate in (44100, 48000, 96000):
            with self.subTest(rate=rate):
                quiet = reference_band_noise(Settings(sample_rate=rate, noise_level=-30, noise_ceiling=-4))
                louder = reference_band_noise(Settings(sample_rate=rate, noise_level=-24, noise_ceiling=-4))
                np.testing.assert_allclose(louder, quiet * 10 ** (6 / 20), atol=1e-12, rtol=0)
                limited = reference_band_noise(Settings(sample_rate=rate, noise_level=-12, noise_ceiling=-6))
                lower_ceiling = reference_band_noise(Settings(sample_rate=rate, noise_level=-12, noise_ceiling=-12))
                np.testing.assert_allclose(lower_ceiling, limited * 10 ** (-6 / 20), atol=1e-12, rtol=0)
                changed_tone = reference_band_noise(Settings(sample_rate=rate, noise_level=-30, noise_ceiling=-4,
                                                             reference_level=-15, max_level=-3))
                np.testing.assert_array_equal(changed_tone, quiet)

    def test_high_reference_comparison_extremes_obey_peak_ceiling(self):
        cfg = Settings(reference_level=-14.5, max_level=-3)
        for offset in (-cfg.offset_limit, cfg.offset_limit):
            for order in (False, True):
                samples = pair_samples(20, 16000, offset, order, "both", cfg)
                self.assertLessEqual(abs(samples).max(), 10 ** (-3 / 20))
        for values in ((-8, -3), (-9, -2), (-14, -9)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Settings(reference_level=values[0], max_level=values[1]).validate()

    def test_comparison_on_default_output_also_disables_stream_rerouting(self):
        process = MagicMock()
        process.__enter__.return_value = process
        process.communicate.return_value = (None, b"")
        process.returncode = 0
        with patch("phonix.audio.shutil.which", return_value="/usr/bin/pw-play"), \
                patch("phonix.audio.subprocess.Popen", return_value=process) as launch:
            Player(Settings()).play(2000, 4000, 0, False, "both")
        command = launch.call_args.args[0]
        self.assertNotIn("--target", command)
        properties = dict(part.split("=", 1) for part in command[command.index("--properties") + 1].split())
        self.assertEqual(properties["node.dont-move"], "true")
        self.assertEqual(properties["node.dont-fallback"], "true")
        self.assertEqual(properties["node.name"], "phonix")

    def test_player_failure_is_reported_and_check_does_not_accept_it(self):
        process = MagicMock()
        process.__enter__.return_value = process
        process.communicate.return_value = (None, b"Output device failed")
        process.returncode = 3
        with patch("phonix.audio.shutil.which", return_value="/usr/bin/pw-play"), \
                patch("phonix.audio.subprocess.Popen", return_value=process), \
                self.assertRaisesRegex(ValueError, "Output device failed"):
            Player(Settings()).play_tone()

    def test_audio_check_replay_and_overrides_preserve_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            settings = Settings()
            profile = store.create("test", settings)
            channel = profile["channels"]["both"]
            a, b = channel["frequencies"][:2]
            add_observation(channel, settings, a, b, 1, "manual")
            store.save(profile)
            before = store.path("test").read_bytes()
            sinks = [{"object.serial": 54, "node.name": "speaker-node", "node.description": "Speaker"}]
            with patch("phonix.cli.devices", return_value=sinks), \
                    patch("phonix.cli.Player") as player, \
                    patch("builtins.input", side_effect=["", "r", "q"]), \
                    contextlib.redirect_stdout(io.StringIO()):
                check_audio(profile, device="speaker-node", level=-36)
            checked_settings = player.call_args.args[0]
            self.assertEqual(checked_settings.device, "speaker-node")
            self.assertEqual(checked_settings.reference_level, -36)
            self.assertEqual(player.return_value.play_tone.call_count, 2)
            self.assertEqual(store.path("test").read_bytes(), before)
            self.assertEqual(profile, store.load("test"))

    def test_check_quit_and_invalid_output_never_play(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Store(directory).create("test", Settings())
            with patch("phonix.cli.devices", return_value=[]), \
                    patch("phonix.cli.Player") as player, \
                    patch("builtins.input", return_value="q"), \
                    contextlib.redirect_stdout(io.StringIO()):
                check_audio(profile)
                player.return_value.play_tone.assert_not_called()
                with self.assertRaisesRegex(ValueError, "not an available"):
                    check_audio(profile, device="missing")

    def test_cli_check_uses_profile_without_saving(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            store.create("test", Settings())
            before = store.path("test").read_bytes()
            with patch("phonix.cli.check_audio") as check:
                self.assertEqual(main(["--data-dir", directory, "audio-check", "test", "--level", "-36"]), 0)
            self.assertEqual(check.call_args.args[1:], (None, -36))
            self.assertEqual(store.path("test").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
