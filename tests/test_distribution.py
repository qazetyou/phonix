"""Version reporting and safe source staging, without building or playing audio."""

import contextlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from phonix import __version__
from phonix.cli import main
from tools.build_release import unpack_source


class DistributionTests(unittest.TestCase):
    def test_version_exits_before_creating_profiles_or_starting_audio(self):
        with contextlib.redirect_stdout(io.StringIO()) as output, \
                patch("phonix.cli.Store") as store, patch("phonix.cli.Player") as player:
            with self.assertRaises(SystemExit) as exit_code:
                main(["--version"])
        self.assertEqual(exit_code.exception.code, 0)
        self.assertEqual(output.getvalue(), f"phonix {__version__}\n")
        store.assert_not_called()
        player.assert_not_called()

    def archive(self, directory, entries):
        path = directory / "source.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name, kind in entries:
                member = tarfile.TarInfo(name)
                member.type = kind
                if kind == tarfile.REGTYPE:
                    member.size = 4
                    archive.addfile(member, io.BytesIO(b"test"))
                else:
                    member.linkname = "outside"
                    archive.addfile(member)
        return path

    def test_source_staging_preserves_files_without_requiring_python_312(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = self.archive(directory, [("phonix-0.1.0/pyproject.toml", tarfile.REGTYPE),
                                                ("phonix-0.1.0/docs/install.md", tarfile.REGTYPE)])
            destination = directory / "staging"
            destination.mkdir()
            root = unpack_source(archive, destination)
            self.assertEqual(root.name, "phonix-0.1.0")
            self.assertEqual((root / "docs/install.md").read_text(), "test")

    def test_source_staging_rejects_paths_and_links_before_writing_any_files(self):
        for name, kind in (("../outside", tarfile.REGTYPE), ("/outside", tarfile.REGTYPE),
                           ("project/link", tarfile.SYMTYPE), ("project/link", tarfile.LNKTYPE)):
            with self.subTest(name=name, kind=kind), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                archive = self.archive(directory, [("project/pyproject.toml", tarfile.REGTYPE), (name, kind)])
                destination = directory / "staging"
                destination.mkdir()
                with self.assertRaises(ValueError):
                    unpack_source(archive, destination)
                self.assertEqual(list(destination.iterdir()), [])

    def test_source_staging_rejects_missing_project_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = self.archive(directory, [("project/README.md", tarfile.REGTYPE)])
            destination = directory / "staging"
            destination.mkdir()
            with self.assertRaisesRegex(ValueError, "one project directory"):
                unpack_source(archive, destination)


if __name__ == "__main__":
    unittest.main()
