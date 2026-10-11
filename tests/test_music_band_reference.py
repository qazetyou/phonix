"""Music-energy defaults and preserved sessions; no live audio."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from phonix.audio import tone
from phonix.cli import main, run_profile
from phonix.music import music_reference_levels, music_samples, preview_info
from phonix.profiles import Settings, Store
from phonix.ui import create_profile
from tools.prepare_music_preview import reference_from_samples


class MusicBandReferenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name))

    def tearDown(self):
        music_samples.cache_clear()
        preview_info.cache_clear()

    def test_analysis_recovers_known_band_energy_without_stereo_cancellation(self):
        rate = 48000
        seconds = np.arange(rate) / rate
        carrier = np.sin(2 * np.pi * 1000 * seconds)
        outside = 0.2 * np.sin(2 * np.pi * 1500 * seconds) + 0.1
        samples = np.column_stack((0.01 * carrier + outside, -0.02 * carrier + outside))
        result = reference_from_samples(samples, rate)
        expected_peak = 20 * np.log10(np.sqrt((0.01**2 + 0.02**2) / 2))
        self.assertAlmostEqual(result['tone_peak_dbfs'], expected_peak, places=8)
        self.assertAlmostEqual(result['band_rms_dbfs'], expected_peak - 10 * np.log10(2), places=8)
        # Doubling the full duration with silence halves mean energy; no gating.
        longer = reference_from_samples(np.vstack((samples, np.zeros_like(samples))), rate)
        self.assertAlmostEqual(longer['band_rms_dbfs'] - result['band_rms_dbfs'], -10 * np.log10(2), delta=0.03)

    def test_missing_or_invalid_band_energy_is_rejected(self):
        for samples in (np.zeros((4800, 2)), np.ones((4800, 1)), np.full((4800, 2), np.nan)):
            with self.subTest(shape=samples.shape), self.assertRaises(ValueError):
                reference_from_samples(samples, 48000)

    def test_bundled_metadata_matches_time_domain_filtered_energy(self):
        samples = music_samples()
        n = len(samples)
        frequencies = np.fft.rfftfreq(n, 1 / 48000)
        power = 0.0
        for channel in range(2):
            spectrum = np.fft.rfft(samples[:, channel])
            spectrum[(frequencies < 935) | (frequencies > 1065)] = 0
            filtered = np.fft.irfft(spectrum, n=n)
            power += float(np.mean(filtered**2)) / 2
        recorded = preview_info()['tone_reference']
        self.assertAlmostEqual(recorded['band_rms_dbfs'], 10 * np.log10(power), places=8)
        self.assertAlmostEqual(recorded['tone_peak_dbfs'], 10 * np.log10(2 * power), places=8)
        levels = music_reference_levels()
        self.assertAlmostEqual(levels['reference_level'], recorded['tone_peak_dbfs'], delta=0.05)
        for rate in (44100, 48000, 96000):
            played = tone(1000, levels['reference_level'], 'both', rate, duration=1, ceiling=levels['max_level'])
            rms = 10 * np.log10(np.mean(played[:, 0]**2))
            self.assertAlmostEqual(rms, recorded['band_rms_dbfs'], delta=0.15)

    def test_cli_and_ui_new_profiles_share_defaults_without_loading_pcm(self):
        with patch('phonix.music.music_samples', side_effect=AssertionError('Runtime analysis')), \
                patch('phonix.cli.sys.stdin.isatty', return_value=False), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--data-dir', str(self.store.root), 'new', 'CLI']), 0)
            with patch('builtins.input', side_effect=['UI']), \
                    patch('phonix.ui.select_table', side_effect=lambda title, headers, rows, *args, **kwargs:
                          next(i for i, row in enumerate(rows) if row[0] == 'Create profile')):
                create_profile(self.store)
        for name in ('CLI', 'UI'):
            cfg = Settings(**self.store.load(name)['settings'])
            self.assertEqual((cfg.reference_level, cfg.max_level), (-38.5, -20.5))
            self.assertEqual(cfg.offset_limit, 18)
            self.assertEqual(cfg.calibration_method, 'regional-v2')

    def test_explicit_cli_levels_override_defaults_and_keep_bounds(self):
        for name, flags, expected in (
            ('reference', ['--reference-level', '-30'], (-30, -12)),
            ('ceiling', ['--max-level', '-12'], (-38.5, -12)),
            ('both', ['--reference-level', '-50', '--max-level', '-32'], (-50, -32)),
        ):
            with self.subTest(name=name), patch('phonix.cli.sys.stdin.isatty', return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--data-dir', str(self.store.root), 'new', name, *flags]), 0)
            cfg = Settings(**self.store.load(name)['settings'])
            self.assertEqual((cfg.reference_level, cfg.max_level), expected)

    def test_existing_profile_keeps_previous_level_without_reapplying_music_default(self):
        profile = self.store.create('existing', Settings(reference_level=-42, max_level=-24))
        original = self.store.path('existing').read_bytes()
        with patch('phonix.cli.Player') as factory, patch('builtins.input', side_effect=['', 'c', 'c', 'q']), \
                contextlib.redirect_stdout(io.StringIO()):
            run_profile(self.store, profile, 0)
        self.assertEqual(factory.return_value.settings.reference_level, -42)
        self.assertEqual(self.store.path('existing').read_bytes(), original)
        del profile['settings']['reference_level']
        del profile['settings']['max_level']
        self.store.save(profile)
        legacy = Settings(**self.store.load('existing')['settings'])
        self.assertEqual((legacy.reference_level, legacy.max_level), (-42, -24))

    def test_music_default_can_be_adjusted_and_is_saved_before_comparisons(self):
        profile = self.store.create('adjusted', Settings(budget=1, **music_reference_levels()))
        with patch('phonix.cli.Player') as factory, \
                patch('builtins.input', side_effect=['', 'c', '+', 'c', 'e']), \
                contextlib.redirect_stdout(io.StringIO()):
            player = factory.return_value
            player.play.side_effect = lambda *args: self.assertEqual(
                self.store.load('adjusted')['settings']['reference_level'], -37.5)
            run_profile(self.store, profile, 0)
        saved = self.store.load('adjusted')
        self.assertEqual(saved['settings']['max_level'], -19.5)
        self.assertEqual(saved['channels']['both']['inputs'], 1)

    def test_invalid_reference_metadata_is_rejected_before_using_a_default(self):
        original = copy.deepcopy(preview_info())
        malformed = [None, {}, dict(original['tone_reference'], tone_peak_dbfs=float('nan')),
                     dict(original['tone_reference'], band_rms_dbfs='invalid'),
                     dict(original['tone_reference'], tone_peak_dbfs=-1),
                     dict(original['tone_reference'], band_low_hz=20)]
        for reference in malformed:
            preview_info.cache_clear()
            metadata = dict(original, tone_reference=reference)
            with self.subTest(reference=reference), patch('phonix.music.files') as resource:
                resource.return_value.joinpath.return_value.read_text.return_value = json.dumps(metadata)
                with self.assertRaisesRegex(ValueError, 'information is missing or invalid'):
                    music_reference_levels()
