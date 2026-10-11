"""Prepared music content, transport bounds and failures; no live audio."""

import hashlib
from importlib.resources import files
import io
from pathlib import Path
import re
import shutil
import subprocess
import unittest
from unittest.mock import MagicMock, patch
import wave

import numpy as np

from phonix.audio import PLAYBACK_LEAD_IN_SECONDS, Player
from phonix.music import MUSIC_DURATION_SECONDS, MUSIC_SAMPLE_RATE, music_samples, preview_info
from phonix.profiles import Settings


class BuiltinMusicTests(unittest.TestCase):
    def tearDown(self):
        music_samples.cache_clear()
        preview_info.cache_clear()

    def test_excerpt_is_complete_stereo_repeatable_and_has_faded_edges(self):
        samples = music_samples()
        self.assertEqual(samples.shape, (MUSIC_DURATION_SECONDS * MUSIC_SAMPLE_RATE, 2))
        self.assertTrue(np.isfinite(samples).all())
        self.assertGreater(np.max(abs(samples)), 0.5)
        self.assertLessEqual(np.max(abs(samples)), 10 ** (-1 / 20) + 4 / 32767)
        self.assertLess(np.max(abs(samples[:48])), 0.01)
        self.assertLess(np.max(abs(samples[-48:])), 0.01)
        self.assertGreater(np.sqrt(np.mean((samples[:, 0] - samples[:, 1]) ** 2)), 0.001)
        self.assertFalse(samples.flags.writeable)
        self.assertIs(music_samples(), samples)

    def test_transport_uses_fixed_asset_rate_routes_to_output_and_allows_full_duration(self):
        for rate in (44100, 48000, 96000):
            process = MagicMock()
            process.__enter__.return_value = process
            process.communicate.return_value = (None, b"")
            process.returncode = 0
            def inspect(command, **kwargs):
                self.assertEqual(command[command.index("--target") + 1], "physical-output")
                self.assertIn("node.dont-move=true", command[command.index("--properties") + 1])
                with wave.open(command[-1], "rb") as stream:
                    self.assertEqual(stream.getframerate(), MUSIC_SAMPLE_RATE)
                    self.assertEqual(stream.getnframes(), int(MUSIC_SAMPLE_RATE * (MUSIC_DURATION_SECONDS + PLAYBACK_LEAD_IN_SECONDS)))
                    pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2)
                lead_in = int(MUSIC_SAMPLE_RATE * PLAYBACK_LEAD_IN_SECONDS)
                self.assertFalse(pcm[:lead_in].any())
                np.testing.assert_array_equal(pcm[lead_in:], np.rint(music_samples() * 32767).astype("<i2"))
                return process
            with self.subTest(rate=rate), patch("phonix.audio.shutil.which", return_value="pw-play"), \
                    patch("phonix.audio.subprocess.Popen", side_effect=inspect):
                Player(Settings(sample_rate=rate, device="physical-output", reference_level=-60,
                                max_level=-54, noise_level=-30, noise_ceiling=-12)).play_music()
            self.assertGreater(process.communicate.call_args.kwargs["timeout"], MUSIC_DURATION_SECONDS + PLAYBACK_LEAD_IN_SECONDS)

    def test_checksum_failure_is_reported_without_playback(self):
        music_samples.cache_clear()
        info = preview_info()
        with patch("phonix.music.files") as resource, \
                patch("phonix.music.preview_info", return_value=info), \
                patch("phonix.audio.shutil.which", return_value="pw-play"), \
                patch("phonix.audio.subprocess.Popen") as launch:
            resource.return_value.joinpath.return_value.read_bytes.return_value = b"corrupted music"
            with self.assertRaisesRegex(ValueError, "checksum failed"):
                Player(Settings()).play_music()
        launch.assert_not_called()

    def test_missing_music_reports_an_actionable_error(self):
        music_samples.cache_clear()
        with patch("phonix.music.files") as resource:
            resource.return_value.joinpath.return_value.read_bytes.side_effect = FileNotFoundError()
            with self.assertRaisesRegex(ValueError, "missing; reinstall"):
                music_samples()

    def test_invalid_format_and_truncation_are_rejected(self):
        content = files("phonix").joinpath("data/music_preview.wav").read_bytes()
        short = content[:-100]
        invalid = io.BytesIO()
        with wave.open(invalid, "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(MUSIC_SAMPLE_RATE)
            stream.writeframes(b"\0\0" * 100)
        for payload, phrase in ((short, "truncated"), (invalid.getvalue(), "format")):
            music_samples.cache_clear()
            with self.subTest(phrase=phrase), patch("phonix.music.files") as resource, \
                    patch("phonix.music.preview_info", return_value={"sha256": hashlib.sha256(payload).hexdigest()}):
                resource.return_value.joinpath.return_value.read_bytes.return_value = payload
                with self.assertRaisesRegex(ValueError, phrase):
                    music_samples()

    def test_invalid_metadata_reports_an_actionable_error(self):
        for value in ("[]", "{}", "invalid JSON"):
            preview_info.cache_clear()
            with self.subTest(value=value), patch("phonix.music.files") as resource:
                resource.return_value.joinpath.return_value.read_text.return_value = value
                with self.assertRaisesRegex(ValueError, "information is missing or invalid"):
                    preview_info()

    @unittest.skipUnless(shutil.which("ffmpeg"), "Optional offline loudness verification needs FFmpeg")
    def test_bundled_pcm_matches_documented_loudness_and_true_peak(self):
        path = Path(__file__).resolve().parents[1] / "phonix/data/music_preview.wav"
        result = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                                 "-af", "ebur128=peak=true", "-f", "null", "-"],
                                capture_output=True, text=True, check=True, timeout=30)
        summary = result.stderr[result.stderr.rfind("Summary:"):]
        loudness = float(re.search(r"I:\s+([-\d.]+) LUFS", summary).group(1))
        peak = float(re.search(r"Peak:\s+([-\d.]+) dBFS", summary).group(1))
        self.assertAlmostEqual(loudness, -14, delta=0.2)
        self.assertEqual(loudness, preview_info()["measured_lufs"])
        self.assertLessEqual(peak, -1.0)
        self.assertEqual(peak, preview_info()["measured_true_peak_dbfs"])
