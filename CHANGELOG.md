# Release notes

## 0.1.0 — 2026-10-11

First experimental Linux release. Python 3.11+, NumPy and PipeWire's `pw-play`
are required. Easy Effects applies the exported EQ presets.

- Continuous terminal UI with profile creation, browsing, resume, reset,
  progress, preset preview and export.
- Listening-based bass, midrange and treble calibration with a fixed B&K 5128
  diffuse-field target and documented sensitivity approximation.
- Nearby comparisons grouped by range, separate regional balance, optional
  local-calibration skipping, and post-calibration sound preferences.
- Guided judgments or manual matching with two aligned vertical sliders,
  immediate 0.5 dB arrow adjustments and explicit Enter replay.
- Casual, Focused and Meticulous listening-effort choices, keeping the
  existing descriptions, tolerances and maximum judgment counts.
- Built-in credited music excerpt with early cancellation, reference-tone
  setup, and frequency-dependent starting tone levels.
- Bounded bell, low-shelf and high-shelf fitting with native Easy Effects JSON
  export, channel support and preamp headroom.
- MIT source license, third-party notices, installable wheel, source archive,
  release bundle and SHA-256 checksums. `phonix --version` reports the release.

The inferred correction remains a listening-only approximation, not measured
personal diffuse-field accuracy. Inaudible or disconnected frequencies may
prevent export. Absolute listening level and individual anatomy are not
independently measured. See `docs/model.md` in the source archive.

Package installation and silent tests are verified on Python 3.14.7 / NumPy
2.4.6. The declared Python 3.11+ / NumPy 1.24+ minimums have not been tested
as a complete version matrix. Export follows the Easy Effects 8.2.8 schema;
audibility and importing into a recipient's Easy Effects installation need
a local listening/UI check. Other operating systems are not supported yet.
