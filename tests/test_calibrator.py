import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from phonix.audio import pair_samples, tone
from phonix.cli import export_profile, main, run_profile
from phonix.engine import Pair, add_observation, fit_channel, new_channel, prepare_local_grid, probabilities, refine, select_pair
from phonix.model import Reference, integrate, log_weights
from phonix.peq import Band, easyeffects_preset, fit_peq, response
from phonix.profiles import Settings, Store, atomic_json


def all_pairs(channel, settings, values, repeats=2, noise=0, seed=3):
    rng = np.random.default_rng(seed)
    frequencies = channel["frequencies"]
    for _ in range(repeats):
        for i, a in enumerate(frequencies):
            for j in range(i + 1, len(frequencies)):
                b = frequencies[j]
                add_observation(channel, settings, a, b, float(values[j] - values[i] + rng.normal(0, noise)), "manual")


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(initial_points=7, max_points=7)
        self.reference = Reference()

    def test_matches_recover_shape_independent_of_absolute_level(self):
        channel = new_channel(self.settings)
        truth = np.array([5.0, 1.0, -2.0, -3.0, 2.0, 4.0, 6.0]) + 80
        all_pairs(channel, self.settings, truth, repeats=2, noise=0.2)
        fit = fit_channel(channel, self.settings)
        self.assertTrue(fit.connected)
        self.assertTrue(fit.ready(self.settings))
        expected = truth - log_weights(fit.frequencies) @ truth
        self.assertLess(np.sqrt(np.mean((fit.matches - expected) ** 2)), 0.2)
        weights = log_weights(fit.frequencies)
        self.assertAlmostEqual(float(weights @ fit.matches), 0, places=10)
        self.assertAlmostEqual(float(weights @ fit.covariance @ weights), 0, places=10)

    def test_chain_alone_does_not_claim_drift_checked(self):
        channel = new_channel(self.settings)
        for a, b in zip(channel["frequencies"][:-1], channel["frequencies"][1:]):
            add_observation(channel, self.settings, a, b, 1.0, "manual")
        fit = fit_channel(channel, self.settings)
        self.assertTrue(fit.connected)
        self.assertEqual(fit.cycles, 0)
        self.assertIsNone(fit.drift_rms)
        self.assertFalse(fit.ready(self.settings))

    def test_cross_checks_reduce_accumulated_chain_error(self):
        channel = new_channel(self.settings)
        frequencies = channel["frequencies"]
        truth = np.arange(len(frequencies), dtype=float)
        for a, b in zip(frequencies[:-1], frequencies[1:]):
            add_observation(channel, self.settings, a, b, 1.6, "manual")
        before = fit_channel(channel, self.settings)
        before_error = abs((before.matches[-1] - before.matches[0]) - 6)
        all_pairs(channel, self.settings, truth, repeats=3, noise=0.2)
        after = fit_channel(channel, self.settings)
        after_error = abs((after.matches[-1] - after.matches[0]) - 6)
        self.assertLess(after_error, before_error / 4)
        self.assertGreater(after.cycles, 0)

    def test_contradictory_cycle_is_reported(self):
        settings = Settings(initial_points=4, max_points=4, drift_tolerance=0.1)
        channel = new_channel(settings)
        all_pairs(channel, settings, np.zeros(4), repeats=2)
        a, b = channel["frequencies"][0], channel["frequencies"][-1]
        channel["observations"] = [o for o in channel["observations"] if (o["a"], o["b"]) != (a, b)]
        channel["inputs"] = len(channel["observations"])
        for _ in range(2):
            add_observation(channel, settings, a, b, 8, "manual")
        fit = fit_channel(channel, settings)
        self.assertGreater(fit.drift_rms, settings.drift_tolerance)
        self.assertFalse(fit.ready(settings))

    def test_repeated_inconsistent_manual_matches_retain_uncertainty(self):
        observations = [{"kind": "manual", "offset": offset, "answer": "equal"} for offset in (-6, 6)]
        pair = Pair(2000, 4000, observations, 18)
        self.assertGreater(pair.variance, 30)

    def test_guided_matching_recovers_noisy_equality_point(self):
        rng = np.random.default_rng(51)
        observations = []
        truth = 4.25
        for _ in range(140):
            pair = Pair(2000, 4000, observations, 18)
            offset = pair.next_offset()
            # Independently simulate a listener whose internal comparison is
            # logistic noise around a known equality point.
            evidence = offset - truth + rng.logistic(0, 1.5)
            answer = "second" if evidence > 0.75 else "first" if evidence < -0.75 else "equal"
            observations.append({"kind": "guided", "offset": offset, "answer": answer})
        pair = Pair(2000, 4000, observations, 18)
        self.assertLess(abs(pair.mean - truth), 0.8)
        self.assertLess(pair.variance, 0.5)

    def test_probability_polarity_and_lapse_floor(self):
        p = probabilities(np.array([0.0]), 12)
        self.assertGreater(p[2, 0], 0.9)
        np.testing.assert_allclose(p.sum(axis=0), 1)
        self.assertGreater(p.min(), 0)

    def test_adaptive_selection_connects_coverage_and_adds_cycles(self):
        channel = new_channel(self.settings)
        truth = np.array([0, 2, -2, 0, 1, -1, 0], dtype=float)
        rng = np.random.default_rng(22)
        for _ in range(35):
            a, b = select_pair(channel, self.settings, rng)
            i, j = channel["frequencies"].index(a), channel["frequencies"].index(b)
            add_observation(channel, self.settings, a, b, float(truth[j] - truth[i] + rng.normal(0, 0.5)), "manual")
        fit = fit_channel(channel, self.settings)
        self.assertTrue(fit.connected)
        self.assertGreaterEqual(fit.cycles, 2)
        self.assertTrue(np.all(fit.degrees >= 2))
        self.assertLess(fit.shape95, 1.5)
        expected = truth - log_weights(fit.frequencies) @ truth
        self.assertLess(np.sqrt(np.mean((fit.matches - expected) ** 2)), 0.5)

    def test_refinement_retains_observations_and_finds_contour_detail(self):
        settings = Settings(max_hz=12000, initial_points=4, max_points=9)
        channel = new_channel(settings)
        prepare_local_grid(channel, settings)
        count = len(channel["frequencies"])
        matches = self.reference.expected_matches(channel["frequencies"])
        all_pairs(channel, settings, matches, repeats=3)
        old = channel["observations"].copy()
        self.assertTrue(refine(channel, settings))
        self.assertEqual(old, channel["observations"])
        self.assertEqual(len(channel["frequencies"]), count + 1)
        self.assertFalse(fit_channel(channel, settings).connected)

    def test_out_of_bounds_match_is_not_called_converged(self):
        observations = [{"kind": "guided", "offset": 17.0, "answer": "first"}] * 80
        pair = Pair(2000, 4000, observations, 18)
        self.assertTrue(pair.boundary)


class ModelAndFilterTests(unittest.TestCase):
    def setUp(self):
        self.reference = Reference()

    def test_df_response_maps_to_zero_eq_under_shared_proxy(self):
        frequencies = np.geomspace(2000, 12000, 21)
        matching = -(self.reference.target(frequencies) + self.reference.sensitivity(frequencies)) + 39
        np.testing.assert_allclose(self.reference.correction(frequencies, matching), 0, atol=1e-10)

    def test_same_listener_across_devices_cancels_shared_model_error(self):
        frequencies = np.geomspace(2000, 12000, 21)
        target = self.reference.target(frequencies)
        own_hearing = self.reference.sensitivity(frequencies) + 3 * np.sin(np.log(frequencies / 2000))
        devices = [target + np.sin(np.log(frequencies)), target + 4 * np.cos(np.log(frequencies / 3000))]
        corrected = []
        for device in devices:
            matches = -(device + own_hearing)
            correction = self.reference.correction(frequencies, matches)
            perceived = device + own_hearing + correction
            perceived -= integrate(perceived, np.log(frequencies)) / np.log(frequencies[-1] / frequencies[0])
            corrected.append(perceived)
        np.testing.assert_allclose(corrected[0], corrected[1], atol=1e-10)

    def test_reference_bounds_and_normalization(self):
        self.assertAlmostEqual(float(self.reference.sensitivity([1000])[0]), 0, places=6)
        for frequency in (0, 16001, np.nan):
            with self.assertRaises(ValueError):
                self.reference.target([frequency])

    def test_bell_gain_matches_center_at_multiple_rates(self):
        for rate in (44100, 48000, 96000):
            gain = response(np.array([6000.0]), [Band(6000, 4, 2)], rate)
            self.assertAlmostEqual(gain[0], 4, places=9)

    def test_known_filter_curve_recovered_by_bounded_fit(self):
        frequencies = np.geomspace(2000, 12000, 32)
        known = [Band(4200, -3, 2), Band(8000, 4, 3)]
        target = response(frequencies, known, 48000)
        bands, metrics = fit_peq(frequencies, target, count=4)
        self.assertLess(metrics["shape_rmse_db"], 0.35)
        self.assertLess(metrics["accumulated_contour_error_db"], 3)
        self.assertTrue(all(-6 <= band.gain <= 6 and 0.4 <= band.q <= 8 for band in bands))

    def test_zero_target_does_not_create_filters(self):
        frequencies = np.geomspace(2000, 12000, 7)
        bands, metrics = fit_peq(frequencies, np.zeros(7), count=4)
        self.assertLess(metrics["shape_rmse_db"], 1e-10)
        self.assertTrue(all(abs(band.gain) < 1e-10 for band in bands))

    def test_headroom_protects_both_channels(self):
        bands = {"left": [Band(6000, 6, 2), Band(6500, 6, 2)], "right": [Band(4000, 3, 1)]}
        preset = easyeffects_preset(bands, 48000)
        equalizer = preset["output"]["equalizer#0"]
        self.assertTrue(equalizer["split-channels"])
        self.assertEqual(equalizer["right"]["band1"]["type"], "Off")
        grid = np.geomspace(20, 23999, 20000)
        for channel in bands.values():
            self.assertLess((response(grid, channel, 48000) + equalizer["input-gain"]).max(), -0.9)

    def test_export_contract_matches_easyeffects_828_fields(self):
        preset = easyeffects_preset({"both": [Band(4000, -2, 1.5)]}, 48000)
        output = preset["output"]
        self.assertEqual(output["plugins_order"], ["equalizer#0"])
        equalizer = output["equalizer#0"]
        # Field contract checked against upstream v8.2.8 equalizer_preset.cpp
        # and easyeffects_db_equalizer(_channel).kcfg, not GUI heuristics.
        required = {"bypass", "input-gain", "output-gain", "mode", "decramp", "split-channels", "balance", "pitch-left", "pitch-right", "num-bands", "left", "right"}
        self.assertEqual(set(equalizer), required)
        self.assertEqual(equalizer["mode"], "IIR")
        self.assertEqual(equalizer["decramp"], "Off")
        self.assertEqual(equalizer["left"], equalizer["right"])
        band = equalizer["left"]["band0"]
        self.assertEqual(set(band), {"type", "mode", "slope", "solo", "mute", "gain", "frequency", "q", "width"})
        self.assertEqual(band["mode"], "APO (DR)")
        self.assertEqual(band["slope"], "x1")
        self.assertFalse(equalizer["split-channels"])


class AudioTests(unittest.TestCase):
    def test_ramps_peak_limit_and_channel_isolation(self):
        samples = tone(4000, -24, "left")
        self.assertLessEqual(abs(samples).max(), 10 ** (-24 / 20) + 1e-12)
        self.assertEqual(samples[0, 0], 0)
        self.assertEqual(samples[-1, 0], 0)
        np.testing.assert_array_equal(samples[:, 1], 0)

    def test_pair_random_order_preserves_tone_levels(self):
        settings = Settings()
        direct = pair_samples(2000, 4000, 6, False, "both", settings)
        reversed_pair = pair_samples(2000, 4000, 6, True, "both", settings)
        count = int(0.45 * settings.sample_rate)
        gap = int(0.25 * settings.sample_rate)
        np.testing.assert_array_equal(direct[:count], reversed_pair[count + gap:2 * count + gap])
        self.assertLessEqual(abs(reversed_pair).max(), 10 ** (settings.max_level / 20))

    def test_invalid_and_excessive_playback_rejected(self):
        for hz, level in [(4000, -6), (24000, -42), (np.nan, -42), (4000, np.nan)]:
            with self.assertRaises(ValueError):
                tone(hz, level, "both")
        with self.assertRaises(ValueError):
            pair_samples(2000, 4000, 30, False, "both", Settings())


class PersistenceAndCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.store = Store(self.root / "profiles")

    def test_multiple_profiles_resume_without_losing_observations(self):
        for name in ("Headphones A", "Headphones B"):
            self.store.create(name, Settings())
        profile = self.store.load("Headphones A")
        channel = profile["channels"]["both"]
        a, b = channel["frequencies"][:2]
        add_observation(channel, Settings(), a, b, 2, "manual")
        self.store.save(profile)
        loaded = self.store.load("Headphones A")
        self.assertEqual(loaded["channels"]["both"]["observations"], channel["observations"])
        self.assertEqual(self.store.load("Headphones B")["channels"]["both"]["inputs"], 0)
        self.assertEqual(len(self.store.names()), 2)

    def test_duplicate_and_path_traversal_profiles_rejected(self):
        self.store.create("A", Settings())
        for name in ("A", "../outside", "/tmp/outside", ""):
            with self.assertRaises(ValueError):
                self.store.create(name, Settings())

    def test_corrupt_or_wrong_model_profile_rejected(self):
        profile = self.store.create("A", Settings())
        profile["model"] = "other-model"
        self.store.save(profile)
        with self.assertRaises(ValueError):
            self.store.load("A")
        self.store.path("A").write_text("{")
        with self.assertRaises(ValueError):
            self.store.load("A")

    def test_atomic_save_failure_preserves_previous_file(self):
        path = self.root / "saved.json"
        atomic_json(path, {"value": 2})
        with self.assertRaises(ValueError):
            atomic_json(path, {"value": np.nan})
        self.assertEqual(json.loads(path.read_text()), {"value": 2})
        self.assertEqual(list(self.root.glob(".saving-*")), [])

    def test_invalid_settings(self):
        cases = [{"min_hz": 13000, "max_hz": 12000}, {"max_hz": 24000}, {"max_hz": np.nan},
                 {"sample_rate": 8000}, {"budget": 0}, {"seed": -1}, {"max_level": 0},
                 {"max_points": 3}, {"shape_tolerance": 0}, {"channels": "bad"}]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Settings(**values).validate()

    def test_cli_error_is_readable_and_does_not_create_profile(self):
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            code = main(["--data-dir", str(self.root), "new", "A", "--min-hz", "16000"])
        self.assertEqual(code, 1)
        self.assertIn("Frequency limits", errors.getvalue())
        self.assertFalse((self.root / "A.json").exists())

    def test_disconnected_export_rejected_even_with_override(self):
        profile = self.store.create("A", Settings())
        with self.assertRaisesRegex(ValueError, "unmeasured priors"):
            export_profile(profile, self.root / "A.json", allow_incomplete=True)
        self.assertFalse((self.root / "A.json").exists())

    def test_export_requires_explicit_override_if_not_converged(self):
        settings = Settings(initial_points=4, max_points=4)
        profile = self.store.create("A", settings)
        channel = profile["channels"]["both"]
        for a, b in zip(channel["frequencies"][:-1], channel["frequencies"][1:]):
            add_observation(channel, settings, a, b, 0, "manual")
        with self.assertRaisesRegex(ValueError, "thresholds"):
            export_profile(profile, self.root / "A.json")
        with contextlib.redirect_stdout(io.StringIO()):
            export_profile(profile, self.root / "A.json", allow_incomplete=True, count=2)
        report = json.loads((self.root / "A.report.json").read_text())
        self.assertFalse(report["channels"]["both"]["consistency"]["consistency_thresholds_met"])
        with self.assertRaisesRegex(ValueError, "already exist"):
            export_profile(profile, self.root / "A.json", allow_incomplete=True)

    def test_run_saves_both_channel_judgments_with_no_real_audio(self):
        profile = self.store.create("A", Settings(channels="separate", budget=1))
        with patch("phonix.cli.Player") as player, patch("builtins.input", side_effect=["", "", "", "e", "e"]), contextlib.redirect_stdout(io.StringIO()):
            run_profile(self.store, profile, 0)
        self.assertEqual(player.return_value.play_music.call_count, 1)
        self.assertEqual(player.return_value.play.call_count, 2)
        loaded = self.store.load("A")
        self.assertEqual(loaded["channels"]["left"]["inputs"], 1)
        self.assertEqual(loaded["channels"]["right"]["inputs"], 1)

    def test_keyboard_interrupt_stops_and_keeps_saved_data(self):
        profile = self.store.create("A", Settings(budget=3))
        with patch("phonix.cli.Player") as player, patch("builtins.input", side_effect=["", "", "", "e"]), contextlib.redirect_stdout(io.StringIO()):
            player.return_value.play.side_effect = [None, KeyboardInterrupt()]
            run_profile(self.store, profile, 0)
        self.assertEqual(self.store.load("A")["channels"]["both"]["inputs"], 1)


if __name__ == "__main__":
    unittest.main()
