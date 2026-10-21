# Installing Phonix 0.1.0 on Linux

## Prerequisites

- Python 3.11 or newer, with `venv` and `pip` available.
- A running PipeWire audio session and `pw-play` on `PATH`. `pw-dump` is
  needed only to list output devices. These tools come from your Linux
  distribution's PipeWire packages.
- NumPy 1.24 or newer. The normal installation command installs it if needed.
- A terminal for the interactive UI. An 80-column, 24-row window is recommended.
- Easy Effects to apply the exported PEQ preset. It is not required to start
  Phonix or run the silent demo.

## From the release bundle

Extract `phonix-0.1.0-linux-release.zip` and enter its `phonix-0.1.0-linux`
directory. Check the packaged files before installing:

```bash
sha256sum -c SHA256SUMS
python3 -m venv .venv
.venv/bin/python -m pip install ./phonix-0.1.0-py3-none-any.whl
.venv/bin/phonix --version
.venv/bin/phonix --help
.venv/bin/phonix
```

`--version` should report `phonix 0.1.0`. Pip may download NumPy; the wheel
contains Phonix and its reference assets, rather than a bundled Python runtime
or NumPy. No compilation of Phonix is needed. Checksums detect file changes;
they are not a signed publisher identity.

If your machine already has a compatible system NumPy, an offline alternative
is:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-index --no-deps ./phonix-0.1.0-py3-none-any.whl
.venv/bin/phonix --help
```

Use the offline command only when NumPy is already available in that environment.

## From the source archive

```bash
tar -xzf phonix-0.1.0.tar.gz
cd phonix-0.1.0
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/phonix
```

Source installation may download the setuptools build dependency as well as
NumPy. The source archive includes the README, the full user and developer guide
in `docs/usage.md`, model documentation, tests, asset notices and release-building
tool. For development, use `pip install -e .`
instead of `pip install .`.

## First session

Choose **New profile**, configure the device, range and listening effort, then
create the profile and choose **Start calibration**. Set your usual hardware
volume with the music excerpt; Enter, Esc or `s` stops the preview early. Confirm
the reference tone is audible and comfortable. Keep the system volume and
headphone fit fixed during calibration.

In tone comparisons, Up/Down adjusts the marked tone in 0.5 dB steps without
playing it. Empty Enter replays; `e` followed by Enter accepts an equal-loudness
match. Guided `1`/`2` judgments remain available. `h` opens help and `q`/Esc
returns after saving already accepted judgments. Skip inaudible tones.

When calibration completes, choose Neutral/reference or an optional sound
preference in Step 4. Then use **Preview EQ** and **Export to Easy Effects** from
the profile menu. Import
the exported preset JSON in Easy Effects, rather than its companion report.
Calibration does not activate EQ or change system volume. Bypass existing EQ
while collecting calibration answers. The result is experimental and does not
measure your individual diffuse-field response.

## Storage, upgrades and removal

Profiles are normally stored in `~/.local/share/phonix/profiles` (or under
`$XDG_DATA_HOME`). Legacy `treble-calibrator` profiles are reused when the new
directory does not exist. `phonix --data-dir /path/to/profiles` selects explicit
storage. Back up your profile directory before upgrades; retain your listening
level when resuming calibration.

Upgrade with `.venv/bin/python -m pip install --upgrade /path/to/new-wheel.whl`.
Uninstall with `.venv/bin/python -m pip uninstall phonix`. Uninstalling the
program preserves saved profiles and exported presets.

## Troubleshooting

- **Command not found:** run `.venv/bin/phonix` from the installation directory,
  or activate the environment with `source .venv/bin/activate`.
- **No NumPy:** install it in the same environment with
  `.venv/bin/python -m pip install numpy`.
- **No `pw-play`:** install your distribution's PipeWire command-line tools and
  make sure your desktop audio session is running.
- **Silent tones:** use the profile's audio check and select the physical output.
  Inspect stream mute/routing; avoid an Easy Effects virtual sink for calibration.
  A successful player exit does not prove that sound was audible.
- **Crowded terminal:** enlarge the window. Page Up/Page Down scroll comparison
  text; Up/Down adjust the tone at that step.

The original program uses the MIT license. Bundled data and music retain the
licenses listed in `THIRD_PARTY_NOTICES.md` and the source archive's data notices.
