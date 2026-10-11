# Third-party notices

Phonix's original source code and documentation use the [MIT license](LICENSE).
The bundled data and recording retain their own licenses:

| Asset | Credit | License | Bundled notice |
| --- | --- | --- | --- |
| `phonix/data/df5128.csv` | AutoEq; Jaakko Pasanen | MIT | `phonix/data/LICENSE-AutoEq.txt` |
| `phonix/data/middle_ear.csv` | LoudnessModel; Malcolm Slaney | BSD 2-Clause | `phonix/data/LICENSE-LoudnessModel.txt` |
| `phonix/data/music_preview.wav` | **Vibe Ace**, Kevin MacLeod (incompetech.com) | Creative Commons Attribution 3.0 Unported | `phonix/data/LICENSE-Music.txt` |

The music excerpt starts at source second 30 and lasts 25 seconds. Changes
include excerpting, resampling to 48 kHz stereo PCM16, fades, and loudness
normalization with true-peak limiting. The recording's license is
<https://creativecommons.org/licenses/by/3.0/>. No endorsement is implied.

Source commits, derivation and checksums for the calibration data are recorded
in `phonix/data/provenance.json`. Music sources, modification details and
checksums are recorded in `phonix/data/music_preview.json` and its notice.
All three notices are included in the source distribution and wheel, along
with the assets they cover. Wheel license metadata describes the combination
of licenses; it does not relicense the bundled music or data under MIT.

NumPy and setuptools are external dependencies and retain their own licenses.
PipeWire and Easy Effects are separately installed programs, not bundled here.
