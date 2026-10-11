"""Silent range, bass-preference, compatibility and full-session checks."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from phonix.audio import pair_samples
from phonix.cli import RANGES, export_profile, main, simulate
from phonix.engine import add_observation, fit_channel, new_channel, pairs_for, refine, select_pair
from phonix.model import Reference, log_weights
from phonix.peq import Band, bass_shelf, easyeffects_preset, fit_with_bass, response
from phonix.profiles import Settings, Store


class BassFilterTests(unittest.TestCase):
    def test_shelf_endpoints_half_gain_and_monotonicity(self):
        for rate in (44100, 48000, 96000):
            for gain in (-12, -4, 0, 4, 12):
                with self.subTest(rate=rate, gain=gain):
                    shelf = bass_shelf(gain)
                    values = response([0, 105, rate / 2], [shelf], rate)
                    np.testing.assert_allclose(values, [gain, gain / 2, 0], atol=1e-7)
                    rendered = response(np.geomspace(5, rate * 0.49, 600), [shelf], rate)
                    self.assertTrue(np.all(np.diff(rendered) * np.sign(gain) <= 1e-7))

    def test_preference_is_exact_and_does_not_change_calibration_fit(self):
        frequencies = np.geomspace(20, 12000, 32)
        correction = response(frequencies, [Band(300, -3, 1), Band(6000, 2, 2)], 48000)
        neutral, neutral_metrics = fit_with_bass(frequencies, correction, count=3)
        boosted, metrics = fit_with_bass(frequencies, correction, count=4, bass_db=4)
        grid = np.geomspace(10, 20000, 1000)
        np.testing.assert_allclose(response(grid, boosted, 48000) - response(grid, neutral, 48000),
                                   response(grid, [bass_shelf(4)], 48000), atol=1e-9)
        self.assertEqual(metrics["shape_rmse_db"], neutral_metrics["shape_rmse_db"])
        self.assertEqual(len(boosted), 4)
        self.assertEqual(boosted[-1].kind, "Lo-shelf")

    def test_native_shelf_and_combined_stereo_headroom(self):
        left = [Band(60, 6, 1), bass_shelf(12)]
        right = [bass_shelf(-4)]
        eq = easyeffects_preset({"left": left, "right": right}, 48000)["output"]["equalizer#0"]
        self.assertEqual(eq["left"]["band1"]["type"], "Lo-shelf")
        self.assertEqual(eq["left"]["band1"]["gain"], 12)
        self.assertEqual(eq["right"]["band1"]["type"], "Off")
        grid = np.geomspace(10, 23000, 2000)
        for bands in (left, right):
            self.assertLessEqual(float(response(grid, bands, 48000).max()) + eq["input-gain"], -0.99)

    def test_full_range_tail_report_includes_shelf_below_20_hz(self):
        _, metrics = fit_with_bass(np.geomspace(20, 12000, 20), np.zeros(20), count=2, bass_db=4)
        self.assertGreater(metrics["outside_range_peak_db"], 3.99)

    def test_invalid_preference_and_insufficient_band_count(self):
        for gain, hz in ((13, 105), (-13, 105), (np.nan, 105), (4, 0), (4, np.inf)):
            with self.assertRaises(ValueError):
                bass_shelf(gain, hz)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            fit_with_bass([20, 200, 2000, 12000], np.zeros(4), count=1, bass_db=4)
        with self.assertRaisesRegex(ValueError, "1–32"):
            fit_with_bass([20, 200, 2000, 12000], np.zeros(4), count=33, bass_db=4)


class RangeTests(unittest.TestCase):
    def test_new_cli_ranges_and_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            for name in RANGES:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["--data-dir", directory, "new", name, "--range", name, "--bass-db", "4"]), 0)
                settings = Settings(**store.load(name)["settings"])
                self.assertEqual(settings.min_hz, RANGES[name]["min_hz"])
                self.assertEqual(settings.max_hz, RANGES[name]["max_hz"])
                self.assertEqual(settings.bass_db, 4)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--data-dir", directory, "new", "default"]), 0)
                self.assertEqual(main(["--data-dir", directory, "new", "custom", "--range", "full",
                                       "--min-hz", "40", "--max-hz", "10000", "--budget", "100",
                                       "--initial-points", "8", "--max-points", "16", "--bass-hz", "120"]), 0)
            self.assertEqual(store.load("default")["settings"]["min_hz"], 60)
            self.assertEqual(store.load("default")["settings"]["max_hz"], 15000)
            custom = store.load("custom")["settings"]
            self.assertEqual((custom["min_hz"], custom["max_hz"], custom["budget"], custom["initial_points"], custom["max_points"], custom["bass_hz"]),
                             (40, 10000, 100, 8, 16, 120))

    def test_wide_range_anchor_and_preference_independent_sampling(self):
        for name in ("bass", "midrange", "full"):
            settings = Settings(**RANGES[name])
            original = new_channel(settings)
            self.assertIn(1000, original["frequencies"])
            self.assertEqual(len(original["frequencies"]), settings.initial_points)
            preferred = new_channel(replace(settings, bass_db=4))
            self.assertEqual(original, preferred)
            self.assertEqual(select_pair(original, settings, np.random.default_rng(4)),
                             select_pair(preferred, replace(settings, bass_db=4), np.random.default_rng(4)))

    def test_low_bass_pairs_have_equal_duration_enough_cycles_and_no_extra_gain(self):
        settings = Settings(**RANGES["full"], bass_db=4)
        samples = pair_samples(20, 1000, 0, False, "left", settings)
        length = int(0.75 * settings.sample_rate)
        gap = int(0.25 * settings.sample_rate)
        self.assertEqual(len(samples), 2 * (length + gap))
        self.assertTrue(np.all(samples[:, 1] == 0))
        self.assertTrue(np.all(samples[length:length + gap] == 0))
        maximum = 10 ** (settings.reference_level / 20)
        self.assertLessEqual(np.max(abs(samples)), maximum + 1e-10)
        reversed_samples = pair_samples(20, 1000, 0, True, "left", settings)
        np.testing.assert_array_equal(samples[:length], reversed_samples[length + gap:2 * length + gap])

    def test_legacy_profile_retains_judgments_model_and_neutral_preference(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            profile = store.create("legacy", Settings())
            channel = profile["channels"]["both"]
            a, b = channel["frequencies"][:2]
            add_observation(channel, Settings(), a, b, 2, "manual")
            del profile["settings"]["bass_db"]
            del profile["settings"]["bass_hz"]
            store.save(profile)
            loaded = store.load("legacy")
            self.assertEqual(loaded["channels"], profile["channels"])
            self.assertEqual(loaded["model"], profile["model"])
            self.assertEqual(Settings(**loaded["settings"]).bass_db, 0)
            self.assertEqual(Settings(**loaded["settings"]).min_hz, 2000)

    def test_preference_settings_validation(self):
        for values in ({"bass_db": 13}, {"bass_db": np.nan}, {"bass_db": True},
                       {"bass_hz": 39}, {"bass_hz": 301}, {"bass_hz": np.inf}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Settings(**values).validate()

    def test_redundant_bass_coverage_refines_instead_of_raising_levels(self):
        # Keep exercising the supported sub-bass edge beyond the default range.
        settings = Settings(**(RANGES["full"] | dict(min_hz=20, max_hz=16000)))
        channel = new_channel(settings)
        reference = Reference()
        frequencies = np.array(channel["frequencies"])
        rolloff = -3 * np.maximum(0, np.log2(120 / frequencies))
        matching = -(reference.target(frequencies) + reference.sensitivity(frequencies) + rolloff)
        for _ in range(4):
            for i, a in enumerate(frequencies):
                for j, b in enumerate(frequencies[i + 1:], i + 1):
                    offset = float(matching[j] - matching[i])
                    if abs(offset) < settings.offset_limit * 0.90:
                        add_observation(channel, settings, a, b, offset, "manual")
        fit = fit_channel(channel, settings)
        self.assertTrue(fit.connected)
        self.assertEqual(fit.degrees[0], 1)
        old_observations = channel["observations"].copy()
        self.assertTrue(refine(channel, settings))
        self.assertTrue(any(20 < f < frequencies[1] for f in channel["frequencies"]))
        self.assertEqual(channel["observations"], old_observations)
        self.assertEqual(settings.max_level, -24)

    def test_simulated_bass_midrange_and_full_sessions_converge_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "profiles")
            for name in ("bass", "midrange", "full"):
                with self.subTest(name=name), contextlib.redirect_stdout(io.StringIO()):
                    output = Path(directory) / (name + ".json")
                    simulate(store, name, output, range_name=name, bass_db=4)
                    profile = store.load(name)
                    settings = Settings(**profile["settings"])
                    fit = fit_channel(profile["channels"]["both"], settings)
                    self.assertTrue(fit.ready(settings))
                    self.assertLess(profile["channels"]["both"]["inputs"], settings.budget)
                    frequencies = fit.frequencies
                    distortion = (4 * np.exp(-0.5 * (np.log2(frequencies / 6500) / 0.35) ** 2)
                                  - 3 * np.maximum(0, np.log2(120 / frequencies)))
                    error = Reference().correction(frequencies, fit.matches) + distortion
                    error -= log_weights(frequencies) @ error
                    self.assertLess(float(np.sqrt(log_weights(frequencies) @ error ** 2)), 1)
                    eq = json.loads(output.read_text())["output"]["equalizer#0"]
                    self.assertEqual(eq["num-bands"], 12)
                    self.assertEqual(eq["left"]["band11"]["type"], "Lo-shelf")

    def test_noisy_guided_full_session_connects_bass_mids_and_treble(self):
        # Local comparisons need more evidence to meet the same strict 1.5 dB
        # legacy uncertainty target. Extend only this synthetic fixture's cap;
        # real profiles retain their chosen tolerance and judgment budget.
        settings = Settings(**(RANGES["full"] | dict(budget=1000)), seed=41)
        channel = new_channel(settings)
        reference = Reference()
        rng = np.random.default_rng(41)

        def distortion(frequencies):
            return (4 * np.exp(-0.5 * (np.log2(frequencies / 6500) / 0.35) ** 2)
                    - 3 * np.maximum(0, np.log2(120 / frequencies)))

        for _ in range(settings.budget):
            refine(channel, settings, reference)
            fit = fit_channel(channel, settings, reference)
            if fit.ready(settings):
                break
            a, b = select_pair(channel, settings, rng, reference)
            self.assertLessEqual(b / a, 2 + 1e-12)
            pair = pairs_for(channel, settings).get((a, b))
            if pair:
                offset = pair.next_offset()
            else:
                i, j = channel["frequencies"].index(a), channel["frequencies"].index(b)
                offset = float(np.clip(fit.matches[j] - fit.matches[i], -settings.offset_limit, settings.offset_limit))
            frequencies = np.array([a, b])
            matches = -(reference.target(frequencies) + reference.sensitivity(frequencies) + distortion(frequencies))
            evidence = offset - (matches[1] - matches[0]) + rng.logistic(0, 1.5)
            answer = "second" if evidence > 0.75 else "first" if evidence < -0.75 else "equal"
            add_observation(channel, settings, a, b, offset, "guided", answer)
        fit = fit_channel(channel, settings, reference)
        self.assertTrue(fit.ready(settings), f"Did not converge in {channel['inputs']} judgments")
        self.assertTrue(np.all(fit.degrees >= 2))
        self.assertIn(1000, fit.frequencies)
        error = reference.correction(fit.frequencies, fit.matches) + distortion(fit.frequencies)
        error -= log_weights(fit.frequencies) @ error
        # Conditional 95% intervals are not deterministic error limits. Check
        # independent recovery separately from the graph's stopping thresholds.
        self.assertLess(float(np.sqrt(log_weights(fit.frequencies) @ error ** 2)), 2.0)

    def test_export_override_keeps_saved_calibration_and_reports_effective_target(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "profiles")
            settings = Settings(initial_points=4, max_points=4)
            profile = store.create("A", settings)
            channel = profile["channels"]["both"]
            for _ in range(3):
                for i, a in enumerate(channel["frequencies"]):
                    for b in channel["frequencies"][i + 1:]:
                        add_observation(channel, settings, a, b, 0, "manual")
            store.save(profile)
            original = json.dumps(profile, sort_keys=True)
            with contextlib.redirect_stdout(io.StringIO()):
                report = export_profile(profile, Path(directory) / "boosted.json", count=3, bass_db=4, bass_hz=120)
            self.assertEqual(json.dumps(store.load("A"), sort_keys=True), original)
            self.assertEqual(json.dumps(profile, sort_keys=True), original)
            self.assertEqual(report["settings"]["bass_db"], 4)
            self.assertTrue(report["bass_preference"]["bass_region_unmeasured"])
            values = report["channels"]["both"]
            np.testing.assert_allclose(np.array(values["requested_correction_db"]) - values["model_correction_db"],
                                       response(values["frequency_hz"], [bass_shelf(4, 120)], settings.sample_rate), atol=1e-12)
            self.assertEqual(values["peq_fit"]["bands"][-1]["kind"], "Lo-shelf")


if __name__ == "__main__":
    unittest.main()
