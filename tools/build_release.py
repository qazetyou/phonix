"""Build a shareable Linux release using the locally installed setuptools."""

import argparse
from email.parser import BytesParser
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


def unpack_source(archive_path, destination):
    """Extract regular source files into a fresh staging directory."""
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unexpected source-archive entry: {member.name}")
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    roots = list(destination.iterdir())
    if len(roots) != 1 or not (roots[0] / "pyproject.toml").is_file():
        raise ValueError("Source archive must contain one project directory")
    return roots[0]


def checksum_line(path):
    return f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"


def build_release(root, destination):
    # Import only for development builds; setuptools is not a runtime dependency.
    from setuptools.build_meta import build_sdist

    destination.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(root)
        source = destination / build_sdist(str(destination))
    finally:
        os.chdir(previous)

    with tempfile.TemporaryDirectory(prefix="phonix-release-") as temporary:
        stage = Path(temporary)
        extracted = stage / "source"
        extracted.mkdir()
        project = unpack_source(source, extracted)
        wheel_output = stage / "wheels"
        wheel_output.mkdir()
        subprocess.run([sys.executable, "-c",
                        "import sys; from setuptools.build_meta import build_wheel; build_wheel(sys.argv[1])",
                        str(wheel_output)], cwd=project, check=True)
        wheels = list(wheel_output.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("Expected one built wheel")
        wheel = destination / wheels[0].name
        shutil.copyfile(wheels[0], wheel)

    with zipfile.ZipFile(wheel) as archive:
        metadata_path = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_path))
    if metadata["Name"] != "phonix":
        raise ValueError("Unexpected wheel project name")
    version = metadata["Version"]
    files = [source, wheel]
    for original, name in (("docs/install.md", "INSTALL.md"), ("CHANGELOG.md", "CHANGELOG.md"),
                           ("LICENSE", "LICENSE"), ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md")):
        target = destination / name
        shutil.copyfile(root / original, target)
        files.append(target)
    checksums = destination / "SHA256SUMS"
    checksums.write_text("".join(checksum_line(path) for path in files), encoding="utf-8")
    files.append(checksums)
    bundle = destination / f"phonix-{version}-linux-release.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"phonix-{version}-linux/{path.name}")
    bundle.with_suffix(".zip.sha256").write_text(checksum_line(bundle), encoding="utf-8")
    return bundle


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "dist")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination == root:
        parser.error("Choose an output directory separate from the source root")
    bundle = build_release(root, destination)
    print(f"Delivery ready: {bundle}")


if __name__ == "__main__":
    main()
