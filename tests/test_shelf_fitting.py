"""Shelf mathematics, bounded automatic fitting and profile/export compatibility."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from phonix.cli import build_preset, main, menu_filter_mode
from phonix.engine import add_observation, new_channel, select_pair
from phonix.peq import Band, bass_shelf, easyeffects_preset, fit_peq, fit_with_bass, response
from phonix.profiles import Settings, Store


class ShelfResponseTests(unittest.TestCase):
    def test_high_shelf_endpoints_midpoint_monotonicity_and_inverse(self):
        for rate in (44100, 48000, 96000):
            grid = np.geomspace(10, rate * 0.4999, 500)
            for frequency in (2000, 6000, 15000):
                for gain in (-6, 0, 6):
                    with self.subTest(rate=rate, frequency=frequency, gain=gain):
                        shelf = Band(frequency, gain, 1 / np.sqrt(2), "Hi-shelf")
                        np.testing.assert_allclose(response([0, frequency, rate / 2], [shelf], rate),
                                                   [0, gain / 2, gain], atol=1e-8)
                        rendered = response(grid, [shelf], rate)
                        self.assertTrue(np.all(np.diff(rendered) * np.sign(gain) >= -1e-8))
                        inverse = Band(frequency, -gain, shelf.q, shelf.kind)
                        np.testing.assert_allclose(response(grid, [shelf, inverse], rate), 0, atol=1e-8)

    def test_shelves_match_independent_bilinear_analog_transfer(self):
        for rate in (44100, 48000, 96000):
            frequencies = np.geomspace(20, rate * 0.49, 150)
            for kind in ("Lo-shelf", "Hi-shelf"):
                for gain in (-6, 6):
                    band = Band(3500, gain, 1 / np.sqrt(2), kind)
                    amplitude = 10 ** (gain / 40)
                    s = 1j * np.tan(np.pi * frequencies / rate) / np.tan(np.pi * band.frequency / rate)
                    middle = np.sqrt(amplitude) / band.q * s
                    analog = (amplitude * (s ** 2 + middle + amplitude) / (amplitude * s ** 2 + middle + 1)
                              if kind == "Lo-shelf" else
                              amplitude * (amplitude * s ** 2 + middle + 1) / (s ** 2 + middle + amplitude))
                    np.testing.assert_allclose(response(frequencies, [band], rate),
                                               20 * np.log10(abs(analog)), atol=1e-8)

    def test_native_high_shelf_stereo_padding_and_full_band_headroom(self):
        left = [Band(4000, 6, 1 / np.sqrt(2), "Hi-shelf"), bass_shelf(4)]
        right = [Band(3000, -6, 1 / np.sqrt(2), "Hi-shelf")]
        for rate in (44100, 48000, 96000):
            eq = easyeffects_preset({"left": left, "right": right}, rate)["output"]["equalizer#0"]
            self.assertEqual(eq["left"]["band0"]["type"], "Hi-shelf")
            self.assertEqual(eq["right"]["band1"]["type"], "Off")
            self.assertNotIn("purpose", eq["left"]["band0"])
            grid = np.unique(np.r_[np.geomspace(10, rate * 0.499999, 4000), rate / 2])
            for bands in (left, right):
                self.assertLessEqual(float(response(grid, bands, rate).max()) + eq["input-gain"], -0.99)
        with self.assertRaisesRegex(ValueError, "outside export bounds"):
            easyeffects_preset({"both": [Band(4000, 7, 0.707, "Hi-shelf")]}, 48000)


class AutomaticFitTests(unittest.TestCase):
    def setUp(self):
        self.frequencies = np.geomspace(60, 15000, 32)

    def test_broad_offsets_select_correct_shelf_and_improve_curve_contour_fit(self):
        for kind, midpoint, gain in (("Lo-shelf", 170, 4), ("Hi-shelf", 4000, -4)):
            with self.subTest(kind=kind):
                target = response(self.frequencies, [Band(midpoint, gain, 1 / np.sqrt(2), kind)], 48000)
                bells, baseline = fit_peq(self.frequencies, target, count=2, filter_mode="bells")
                automatic, metrics = fit_peq(self.frequencies, target, count=2)
                self.assertTrue(all(band.kind == "Bell" for band in bells))
                self.assertIn(kind, [band.kind for band in automatic])
                self.assertEqual(len(automatic), 2)
                self.assertLess(metrics["shape_rmse_db"], baseline["shape_rmse_db"])
                self.assertLess(metrics["accumulated_contour_error_db"], baseline["accumulated_contour_error_db"])
                self.assertLess(metrics["curve_contour_objective"], metrics["bell_curve_contour_objective"])
                self.assertLessEqual(metrics["outside_range_peak_db"], metrics["correction_outside_limit_db"] + 1e-6)
                for band in automatic:
                    self.assertLessEqual(abs(band.gain), 6)
                    self.assertEqual(band.purpose, "Correction")
                    if band.kind != "Bell":
                        self.assertAlmostEqual(band.q, 1 / np.sqrt(2))
                        self.assertGreaterEqual(band.frequency, self.frequencies[1])
                        self.assertLessEqual(band.frequency, self.frequencies[-2])

    def test_local_features_keep_bells(self):
        target = response(self.frequencies, [Band(250, -3, 1), Band(5500, 3, 2)], 48000)
        bands, metrics = fit_peq(self.frequencies, target, count=2)
        self.assertEqual(metrics["selected_topology"], "bells")
        self.assertTrue(all(band.kind == "Bell" for band in bands))
        self.assertLess(metrics["shape_rmse_db"], 0.05)

    def test_residual_initialization_reduces_four_band_local_minimum(self):
        target = response(self.frequencies, [Band(250, -3, 1), Band(5500, 3, 2)], 48000)
        _, metrics = fit_peq(self.frequencies, target, count=4, filter_mode="bells")
        self.assertLess(metrics["shape_rmse_db"], 0.05)
        self.assertLessEqual(metrics["fit_objective"], metrics["candidates"][0]["objective"])

    def test_both_shelves_are_available_without_forcing_them(self):
        target = response(self.frequencies, [Band(170, 4, 2 ** -0.5, "Lo-shelf"),
                                             Band(4000, -3, 2 ** -0.5, "Hi-shelf")], 48000)
        bands, metrics = fit_peq(self.frequencies, target, count=2)
        self.assertEqual({band.kind for band in bands}, {"Lo-shelf", "Hi-shelf"})
        self.assertLess(metrics["shape_rmse_db"], 0.03)

    def test_single_band_and_zero_target(self):
        for kind, midpoint in (("Lo-shelf", 170), ("Hi-shelf", 4000)):
            target = response(self.frequencies, [Band(midpoint, 4, 2 ** -0.5, kind)], 48000)
            bands, metrics = fit_peq(self.frequencies, target, count=1)
            self.assertEqual(len(bands), 1)
            self.assertEqual(bands[0].kind, kind)
        bands, metrics = fit_peq(self.frequencies, np.ones(32) * 3, count=4)
        self.assertEqual(metrics["selected_topology"], "bells")
        self.assertEqual(len(metrics["candidates"]), 1)
        self.assertTrue(all(abs(band.gain) < 1e-10 for band in bands))

    def test_ranges_only_offer_shelves_with_supported_transition_coverage(self):
        for limits, absent in (((2000, 16000), "Lo-shelf"), ((20, 1000), "Hi-shelf")):
            frequencies = np.geomspace(*limits, 12)
            _, metrics = fit_peq(frequencies, np.linspace(-2, 2, 12), count=2)
            self.assertTrue(all(absent not in candidate["topology"] for candidate in metrics["candidates"]))

    def test_exact_upper_bound_is_exportable(self):
        bands, _ = fit_peq(np.geomspace(2000, 16000, 7), np.zeros(7), count=4)
        self.assertLessEqual(max(band.frequency for band in bands), 16000)
        easyeffects_preset({"both": bands}, 48000)

    def test_bass_only_range_can_fit_shelf_above_its_geometric_midpoint(self):
        frequencies = np.geomspace(20, 1000, 12)
        target = response(frequencies, [Band(250, 4, 2 ** -0.5, "Lo-shelf")], 48000)
        bands, metrics = fit_peq(frequencies, target, count=1)
        self.assertEqual(bands[0].kind, "Lo-shelf")
        self.assertGreater(bands[0].frequency, np.sqrt(20 * 1000))
        self.assertLess(metrics["shape_rmse_db"], 0.03)

    def test_failed_optional_candidates_keep_uniform_bell_fallback(self):
        target = np.linspace(-2, 2, 32)
        with patch("phonix.peq._residual_seed", side_effect=np.linalg.LinAlgError("failure")):
            bands, metrics = fit_peq(self.frequencies, target, count=2)
        self.assertEqual(metrics["selected_topology"], "bells")
        self.assertTrue(all(band.kind == "Bell" for band in bands))
        self.assertTrue(any("error" in item for item in metrics["candidates"]))

    def test_excessive_headroom_keeps_bell_fallback(self):
        target = response(self.frequencies, [Band(4000, -4, 2 ** -0.5, "Hi-shelf")], 48000)
        with patch("phonix.peq.headroom", return_value=-37):
            bands, metrics = fit_peq(self.frequencies, target, count=2)
        self.assertTrue(all(band.kind == "Bell" for band in bands))
        self.assertEqual(metrics["selected_topology"], "bells")

    def test_excessive_unmeasured_tails_reject_otherwise_better_shelves(self):
        target = response(self.frequencies, [Band(170, 4, 2 ** -0.5, "Lo-shelf")], 48000)
        def excessive_tail(frequencies, bands, rate):
            if len(frequencies) == 80 and any(band.kind != "Bell" for band in bands):
                return np.full(80, 9.0)
            return response(frequencies, bands, rate)
        with patch("phonix.peq.response", side_effect=excessive_tail):
            bands, metrics = fit_peq(self.frequencies, target, count=2)
        self.assertTrue(all(band.kind == "Bell" for band in bands))
        self.assertEqual(metrics["selected_topology"], "bells")
        self.assertTrue(any(candidate["outside_range_peak_db"] > metrics["correction_outside_limit_db"]
                            and candidate["curve_contour_objective"] < metrics["bell_curve_contour_objective"]
                            for candidate in metrics["candidates"]))

    def test_invalid_mode_rate_and_count_are_rejected(self):
        for kwargs in ({"filter_mode": "unknown"}, {"sample_rate": np.nan}, {"sample_rate": 0},
                       {"count": 0}, {"count": 33}, {"count": 1.5}, {"count": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit_peq(self.frequencies, np.zeros(32), **kwargs)
        for count in (True, 1.5, 0, 33):
            with self.subTest(count=count), self.assertRaises(ValueError):
                fit_with_bass(self.frequencies, np.zeros(32), count=count)

    def test_bass_preference_is_exact_and_keeps_correction_selection(self):
        target = response(self.frequencies, [Band(4000, -4, 2 ** -0.5, "Hi-shelf")], 48000)
        neutral, original = fit_with_bass(self.frequencies, target, count=2)
        preferred, metrics = fit_with_bass(self.frequencies, target, count=3, bass_db=4)
        self.assertEqual(preferred[:-1], neutral)
        self.assertEqual(metrics["selected_topology"], original["selected_topology"])
        self.assertEqual(metrics["shape_rmse_db"], original["shape_rmse_db"])
        self.assertEqual(preferred[-1].purpose, "Bass preference")
        grid = np.geomspace(10, 23000, 600)
        np.testing.assert_allclose(response(grid, preferred, 48000) - response(grid, neutral, 48000),
                                   response(grid, [bass_shelf(4)], 48000), atol=1e-9)


class FilterModeProfileTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = Store(self.root / "profiles")

    def connected_profile(self):
        settings = Settings(initial_points=4, max_points=4)
        profile = self.store.create("device", settings)
        channel = profile["channels"]["both"]
        for a, b in zip(channel["frequencies"], channel["frequencies"][1:]):
            add_observation(channel, settings, a, b, 0, "manual")
        self.store.save(profile)
        return profile

    def test_menu_changes_mode_without_discarding_judgments_and_cancel_keeps_file(self):
        profile = self.connected_profile()
        evidence = copy.deepcopy(profile["channels"])
        with patch("phonix.cli.select_table", return_value=1) as picker:
            menu_filter_mode(self.store, profile)
        self.assertEqual(profile["settings"]["filter_mode"], "bells")
        self.assertEqual(self.store.load("device")["channels"], evidence)
        before = self.store.path("device").read_bytes()
        with patch("phonix.cli.select_table", return_value=None):
            menu_filter_mode(self.store, profile)
        self.assertEqual(self.store.path("device").read_bytes(), before)

    def test_legacy_profiles_default_to_auto_without_rewriting_saved_data(self):
        profile = self.connected_profile()
        del profile["settings"]["filter_mode"]
        self.store.save(profile)
        before = self.store.path("device").read_bytes()
        loaded = self.store.load("device")
        self.assertEqual(Settings(**loaded["settings"]).filter_mode, "auto")
        _, report = build_preset(loaded, count=2, allow_incomplete=True)
        self.assertEqual(report["settings"]["filter_mode"], "auto")
        self.assertEqual(self.store.path("device").read_bytes(), before)

    def test_cli_mode_and_export_override_are_saved_only_where_requested(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--data-dir", str(self.store.root), "new", "bells", "--filter-mode", "bells"]), 0)
        self.assertEqual(self.store.load("bells")["settings"]["filter_mode"], "bells")
        profile = self.connected_profile()
        before = self.store.path("device").read_bytes()
        target = self.root / "output.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--data-dir", str(self.store.root), "export", "device", "--bands", "2",
                                   "--allow-incomplete", "--output", str(target), "--filter-mode", "bells"]), 0)
        report = json.loads(target.with_suffix(".report.json").read_text())
        self.assertEqual(report["settings"]["filter_mode"], "bells")
        self.assertEqual(report["channels"]["both"]["peq_fit"]["selected_topology"], "bells")
        self.assertEqual(self.store.path("device").read_bytes(), before)

    def test_filter_mode_has_no_effect_on_probe_schedule_or_model(self):
        auto, bells = Settings(), Settings(filter_mode="bells")
        first, second = new_channel(auto), new_channel(bells)
        self.assertEqual(first, second)
        self.assertEqual(select_pair(first, auto, np.random.default_rng(7)),
                         select_pair(second, bells, np.random.default_rng(7)))
        with self.assertRaisesRegex(ValueError, "Filter mode"):
            Settings(filter_mode="other").validate()


if __name__ == "__main__":
    unittest.main()
