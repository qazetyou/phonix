"""Legacy noise compatibility and music/tone setup persistence; no live audio."""

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from phonix.audio import pink_noise
from phonix.cli import main, prepare_volume
from phonix.engine import add_observation
from phonix.music import music_reference_levels
from phonix.profiles import Settings, Store, REFERENCE_NOISE_PRESET
from phonix.ui import create_profile, edit_playback


class MusicReferenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / "profiles")

    def assert_music_reference(self, settings):
        for key, value in REFERENCE_NOISE_PRESET.items():
            self.assertEqual(settings[key], value)

    def test_new_cli_and_ui_profiles_use_same_reference_noise_defaults(self):
        with contextlib.redirect_stdout(io.StringIO()), \
                patch("phonix.cli.sys.stdin.isatty", return_value=False):
            self.assertEqual(main(["--data-dir", str(self.store.root), "new", "cli"]), 0)
        with patch("builtins.input", return_value="ui"), \
                patch("phonix.ui.select_table", side_effect=lambda title, headers, rows, *args, **kwargs:
                      next(index for index, row in enumerate(rows) if row[0] == "Create profile")), \
                contextlib.redirect_stdout(io.StringIO()):
            create_profile(self.store)
        for name in ("cli", "ui"):
            self.assert_music_reference(self.store.load(name)["settings"])

    def test_explicit_noise_flags_override_each_preset_field_independently(self):
        for name, flags, expected in (
                ("rms", ["--noise-level", "-24"], (-24, -4)),
                ("ceiling", ["--noise-ceiling", "-8"], (-17.7, -8)),
                ("both", ["--noise-level", "-24", "--noise-ceiling", "-8"], (-24, -8))):
            with self.subTest(name=name), contextlib.redirect_stdout(io.StringIO()), \
                    patch("phonix.cli.sys.stdin.isatty", return_value=False):
                self.assertEqual(main(["--data-dir", str(self.store.root), "new", name, *flags]), 0)
                cfg = Settings(**self.store.load(name)["settings"])
                self.assertEqual((cfg.noise_level, cfg.noise_ceiling), expected)

    def test_resumed_editor_keeps_legacy_noise_and_existing_evidence(self):
        cfg = Settings(channels="separate", noise_level=-24, noise_ceiling=-8)
        profile = self.store.create("ears", cfg)
        channel = profile["channels"]["right"]
        add_observation(channel, cfg, *channel["frequencies"][:2], 0, "manual")
        self.store.save(profile)
        original = self.store.path("ears").read_bytes()
        with patch("phonix.ui.select_table", side_effect=[2]):
            edit_playback(self.store, profile)
        self.assertEqual(self.store.path("ears").read_bytes(), original)
        with patch("phonix.ui.select_table", side_effect=[1]) as picker:
            edit_playback(self.store, profile)
        self.assertEqual(self.store.load("ears"), profile)
        labels = [row[0] for row in picker.call_args.args[2]]
        self.assertEqual(labels, ["Output device", "Save", "Cancel"])

    def test_default_tone_preset_cancel_and_save_leave_music_and_noise_unchanged(self):
        cfg = Settings(reference_level=-30, max_level=-12, noise_level=-24, noise_ceiling=-8)
        profile = self.store.create("empty", cfg)
        original = self.store.path("empty").read_bytes()
        with patch("phonix.ui.select_table", side_effect=[4, 6]):
            edit_playback(self.store, profile)
        self.assertEqual(self.store.path("empty").read_bytes(), original)
        with patch("phonix.ui.select_table", side_effect=[4, 5]) as picker:
            edit_playback(self.store, profile)
        saved = Settings(**self.store.load("empty")["settings"])
        self.assertEqual(saved, replace(cfg, **music_reference_levels()))
        labels = [row[0] for row in picker.call_args.args[2]]
        self.assertNotIn("Reference noise preset", labels)
        self.assertNotIn("Reference-noise level (dBFS RMS)", labels)

    def test_preview_explains_estimate_and_player_volume_without_changing_settings(self):
        for cfg in (Settings(**REFERENCE_NOISE_PRESET), Settings()):
            with self.subTest(settings=cfg), patch("builtins.input", return_value="q"), \
                    patch("phonix.cli.Player") as player, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertIsNone(prepare_volume(player.return_value, cfg))
                player.return_value.play_music.assert_not_called()
                self.assertIn("About -14 LUFS", output.getvalue())
                self.assertIn("Vibe Ace by Kevin MacLeod | 25 seconds", output.getvalue())
                self.assertNotIn("Custom/legacy", output.getvalue())
                self.assertIn("100%", output.getvalue())

    def test_preset_reaches_requested_rms_without_peak_attenuation_at_all_rates(self):
        for rate in (44100, 48000, 96000):
            with self.subTest(rate=rate):
                samples = pink_noise(REFERENCE_NOISE_PRESET["noise_level"], rate=rate,
                                     ceiling=REFERENCE_NOISE_PRESET["noise_ceiling"])
                rms = 20 * np.log10(np.sqrt(np.mean(samples ** 2)))
                self.assertAlmostEqual(rms, -17.7, places=10)
                self.assertLessEqual(abs(samples).max(), 10 ** (-4 / 20))

    @unittest.skipUnless(shutil.which("ffmpeg"), "Optional offline loudness check needs FFmpeg")
    def test_reference_noise_parent_signal_keeps_its_original_loudness(self):
        with tempfile.TemporaryDirectory() as root:
            for rate in (44100, 48000, 96000):
                with self.subTest(rate=rate):
                    samples = pink_noise(REFERENCE_NOISE_PRESET["noise_level"], rate=rate,
                                         ceiling=REFERENCE_NOISE_PRESET["noise_ceiling"])
                    path = Path(root) / "noise.wav"
                    with wave.open(str(path), "wb") as stream:
                        stream.setnchannels(2)
                        stream.setsampwidth(2)
                        stream.setframerate(rate)
                        stream.writeframes(np.rint(samples * 32767).astype("<i2").tobytes())
                    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                                             "-af", "ebur128=peak=true", "-f", "null", "-"],
                                            capture_output=True, text=True, check=True, timeout=15)
                    summary = result.stderr[result.stderr.rfind("Summary:"):]
                    loudness = float(re.search(r"I:\s+([-\d.]+) LUFS", summary).group(1))
                    peak = float(re.search(r"Peak:\s+([-\d.]+) dBFS", summary).group(1))
                    self.assertAlmostEqual(loudness, -14, delta=0.2)
                    self.assertLess(peak, -3)


if __name__ == "__main__":
    unittest.main()
