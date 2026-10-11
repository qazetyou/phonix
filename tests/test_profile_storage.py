"""The project rename keeps existing profiles accessible without migrating data."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phonix.cli import default_directory, main
from phonix.engine import add_observation
from phonix.profiles import Settings, Store


class ProfileStorageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data_home = Path(directory.name)
        environment = patch.dict("os.environ", {"XDG_DATA_HOME": str(self.data_home)})
        environment.start()
        self.addCleanup(environment.stop)

    def test_fresh_install_uses_phonix_without_creating_a_directory(self):
        expected = self.data_home / "phonix/profiles"
        self.assertEqual(default_directory(), expected)
        self.assertFalse(expected.exists())

    def test_existing_profiles_keep_their_saved_evidence_and_file_contents(self):
        store = Store(self.data_home / "treble-calibrator/profiles")
        settings = Settings()
        profile = store.create("Existing headphones", settings)
        channel = profile["channels"]["both"]
        a, b = channel["frequencies"][:2]
        add_observation(channel, settings, a, b, 1.0, "manual")
        store.save(profile)
        original = store.path(profile["name"]).read_bytes()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["list", "--plain"]), 0)
        self.assertIn(profile["name"], output.getvalue())
        self.assertEqual(default_directory(), store.root)
        self.assertEqual(store.load(profile["name"])["channels"]["both"]["inputs"], 1)
        self.assertEqual(store.path(profile["name"]).read_bytes(), original)
        self.assertFalse((self.data_home / "phonix").exists())

    def test_phonix_directory_takes_precedence_when_both_exist(self):
        legacy = Store(self.data_home / "treble-calibrator/profiles")
        current = Store(self.data_home / "phonix/profiles")
        legacy.create("Legacy", Settings())
        current.create("Current", Settings())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["list", "--plain"]), 0)
        self.assertEqual(default_directory(), current.root)
        self.assertIn("Current", output.getvalue())
        self.assertNotIn("Legacy", output.getvalue())
        self.assertTrue(legacy.path("Legacy").exists())

    def test_explicit_directory_still_overrides_automatic_selection(self):
        Store(self.data_home / "treble-calibrator/profiles").create("Legacy", Settings())
        explicit = Store(self.data_home / "chosen")
        explicit.create("Chosen", Settings())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--data-dir", str(explicit.root), "list", "--plain"]), 0)
        self.assertIn("Chosen", output.getvalue())
        self.assertNotIn("Legacy", output.getvalue())

    def test_unset_xdg_data_home_uses_home_local_share(self):
        with patch.dict("os.environ", {}, clear=True), patch("phonix.cli.Path.home", return_value=self.data_home):
            self.assertEqual(default_directory(), self.data_home / ".local/share/phonix/profiles")
