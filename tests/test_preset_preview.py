"""Read-only preset previews share exported filters and stay in the profile UI."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phonix.cli import browse_profiles, build_preset, export_profile, preview_profile
from phonix.engine import add_observation
from phonix.profiles import Settings, Store


class PresetPreviewTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = Store(self.root / "profiles")

    def profile(self, name="test", channels="together", complete=True, bass_db=0):
        settings = Settings(initial_points=4, max_points=4, budget=12,
                            channels=channels, bass_db=bass_db)
        profile = self.store.create(name, settings)
        for channel in profile["channels"].values():
            frequencies = channel["frequencies"]
            if complete:
                for _ in range(2):
                    for index, a in enumerate(frequencies):
                        for b in frequencies[index + 1:]:
                            add_observation(channel, settings, a, b, 0, "manual")
            else:
                for index, (a, b) in enumerate(zip(frequencies, frequencies[1:])):
                    add_observation(channel, settings, a, b, index - 1, "manual")
        self.store.save(profile)
        return profile

    def preview(self, profile):
        with contextlib.redirect_stdout(io.StringIO()) as output, \
                patch("phonix.cli.select_table", return_value=None) as table, \
                patch("phonix.cli.Player") as player, \
                patch("phonix.cli.atomic_json") as save:
            preview_profile(profile)
        player.assert_not_called()
        save.assert_not_called()
        return table, output.getvalue()

    def test_preview_matches_exported_filters_including_shelf_and_headroom(self):
        profile = self.profile(bass_db=4)
        original = copy.deepcopy(profile)
        original_file = self.store.path("test").read_bytes()
        table, _ = self.preview(profile)
        target = self.root / "preset.json"
        with contextlib.redirect_stdout(io.StringIO()):
            export_profile(profile, target, count=12)
        equalizer = json.loads(target.read_text())["output"]["equalizer#0"]
        title, headers, rows = table.call_args.args
        self.assertIn(f"preamp {equalizer['input-gain']:.2f} dB", title)
        self.assertIn("complete", title)
        self.assertEqual(headers, ("Channel", "Filter", "Type", "Frequency (Hz)", "Gain (dB)", "Q", "Purpose"))
        self.assertEqual(len(rows), 12)
        for index, row in enumerate(rows):
            band = equalizer["left"][f"band{index}"]
            self.assertEqual(row, ("Both", str(index + 1), band["type"],
                                   f"{band['frequency']:.1f}", f"{band['gain']:+.2f}", f"{band['q']:.3f}",
                                   "Bass preference" if index == 11 else "Correction"))
        self.assertEqual(rows[-1][2:5], ("Lo-shelf", "105.0", "+4.00"))
        self.assertEqual(profile, original)
        self.assertEqual(self.store.path("test").read_bytes(), original_file)

    def test_connected_unfinished_preview_is_provisional_but_export_still_requires_override(self):
        profile = self.profile(complete=False)
        table, output = self.preview(profile)
        self.assertIn("provisional", table.call_args.args[0])
        self.assertIn("unfinished", output)
        self.assertIn("do not measure personal diffuse-field accuracy", output)
        with self.assertRaisesRegex(ValueError, "checks are not met"):
            build_preset(profile)
        preset, _ = build_preset(profile, count=2, allow_incomplete=True)
        self.assertEqual(preset["output"]["equalizer#0"]["num-bands"], 2)

    def test_separate_channels_have_their_own_filter_rows(self):
        profile = self.profile(channels="separate", complete=False)
        settings = Settings(**profile["settings"])
        right = profile["channels"]["right"]
        add_observation(right, settings, *right["frequencies"][:2], 3, "manual")
        table, _ = self.preview(profile)
        rows = table.call_args.args[2]
        left = [row for row in rows if row[0] == "Left"]
        right = [row for row in rows if row[0] == "Right"]
        self.assertEqual(len(left), 12)
        self.assertEqual(len(right), 12)
        self.assertNotEqual([row[3:] for row in left], [row[3:] for row in right])

    def test_new_and_disconnected_profiles_explain_missing_coverage_without_fitting(self):
        profile = self.store.create("empty", Settings(initial_points=4, max_points=4))
        for observed in (False, True):
            if observed:
                channel = profile["channels"]["both"]
                add_observation(channel, Settings(**profile["settings"]),
                                *channel["frequencies"][:2], 0, "manual")
            with patch("phonix.cli.build_preset") as build, \
                    patch("phonix.cli._menu_text", return_value=None) as back:
                table, output = self.preview(profile)
            build.assert_not_called()
            table.assert_not_called()
            back.assert_called_once()
            self.assertIn("Continue calibration", output)

    def test_one_unconnected_ear_prevents_preview_of_partial_stereo_preset(self):
        profile = self.profile(channels="separate")
        profile["channels"]["right"] = self.store.create(
            "new", Settings(channels="separate", initial_points=4, max_points=4))["channels"]["right"]
        with patch("phonix.cli.build_preset") as build, \
                patch("phonix.cli._menu_text", return_value=""):
            table, _ = self.preview(profile)
        build.assert_not_called()
        table.assert_not_called()

    def test_preview_action_returns_to_selected_profile(self):
        self.profile()
        def choose(title, headers, rows, initial=0):
            if title == "Device profiles":
                return None
            if headers[0] == "Channel":
                return 0  # Enter leaves the preview without editing a filter.
            if not hasattr(choose, "opened"):
                choose.opened = True
                return next(index for index, row in enumerate(rows) if row[0] == "Preview EQ")
            return None
        with patch("phonix.cli.select_table", side_effect=choose) as table, \
                contextlib.redirect_stdout(io.StringIO()):
            browse_profiles(self.store, open_name="test")
        self.assertEqual(table.call_count, 4)
        self.assertEqual(table.call_args_list[0].args[0], table.call_args_list[2].args[0])

    def test_disconnected_preview_in_menu_returns_to_same_profile(self):
        self.store.create("empty", Settings())
        with patch("phonix.cli.select_table", side_effect=[2, None, None]) as table, \
                patch("phonix.cli._menu_text", return_value=None), \
                contextlib.redirect_stdout(io.StringIO()):
            browse_profiles(self.store, open_name="empty")
        self.assertEqual(table.call_args_list[0].args[0], table.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
