# phonix

A Linux terminal app for listening-based headphone calibration and native
Easy Effects parametric EQ export. Compare tones, save a device profile, and
export a preset with optional bass and sound preferences.

The correction is experimental: it uses the B&K 5128 diffuse-field target and
a fixed population hearing approximation. It does not measure your individual
diffuse-field response. See the [model and limitations](docs/model.md).

## Features

- Bass, midrange, treble or full-range calibration, with guided or manual matching.
- Both ears together or separate left/right calibration.
- Named profiles, automatic progress saving and resumable sessions.
- Optional bass shelves and sound preferences, with native Easy Effects EQ export.

## Quick start

Requires Linux, Python 3.11+, NumPy 1.24+ and PipeWire's `pw-play`. Pip installs
NumPy if needed. Easy Effects is needed to apply exported presets.

From the project directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/phonix
```

Choose **New profile**, select your physical output and listening settings, then
choose **Start calibration**. Use the music preview to set a comfortable volume,
confirm the reference tone, and follow the comparisons. Keep volume and headphone
fit consistent, bypass existing EQ, and skip inaudible or distorted tones.
Accepted judgments are saved automatically; return to **Profiles** to resume.

After calibration, choose your sound preference, then **Export to Easy Effects**.
Import the preset JSON through Easy Effects' **Presets** menu. Phonix also writes
a separate `.report.json` with results; import the preset itself. Calibration
does not change system volume or activate EQ.

For command-line options, run `.venv/bin/phonix --help`.

## Documentation

- [User and developer guide](docs/usage.md): controls, calibration workflow,
  profile settings, CLI examples, troubleshooting and export details.
- [Installation guide](docs/install.md): release packages, prerequisites,
  upgrades and removal.
- [Model and error definitions](docs/model.md): assumptions, sampling,
  consistency checks and numerical EQ fitting.
- [Release notes](CHANGELOG.md).

## Development

From the project directory, install in editable mode and run the silent tests:

```bash
.venv/bin/python -m pip install -e .
python3 -m unittest discover -s tests -v
```

Build a release with `.venv/bin/python tools/build_release.py` using setuptools
77+. See the [development and release details](docs/usage.md#installation-and-prerequisites).

## License

Source code and documentation use the [MIT license](LICENSE). Bundled data and
music retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
