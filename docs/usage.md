# Phonix user and developer guide

Detailed usage, configuration, compatibility notes and development commands.
For a brief overview, see the [project README](../README.md).
Run the examples below from the project directory, rather than `docs/`.

## Contents

- [Overview and features](#overview-and-features)
- [Quick start and listening controls](#quick-start)
- [Troubleshoot silent playback](#check-silent-playback)
- [Customize a profile](#customize-a-profile)
- [Skip local calibration](#skip-local-calibration)
- [Choose a sound profile](#choose-a-sound-profile-after-calibration)
- [Choose bass level](#choose-bass-level)
- [Browse and manage profiles](#browse-and-manage-profiles)
- [Export to Easy Effects](#export-to-easy-effects)
- [Installation, development setup and storage](#installation-and-prerequisites)
- [Verification and music preparation](#verification)
- [Build a release](#building-a-delivery)
- [License](#license)

## Overview and features

A Linux terminal program for listening-based bass, midrange and treble calibration and native Easy Effects PEQ export. It uses the B&K 5128 diffuse-field curve and one fixed, documented population sensitivity approximation across device profiles.

The result is an experimental, model-based correction. Equal-loudness probes measure the combined playback device and listener; they do not independently recover your anatomy or verify your personal diffuse-field response. Consistency and numerical EQ fitting errors are reported separately. See [the model and error definitions](model.md).

## What is implemented

- Bass, midrange, treble and full-range sessions; new profiles default to 60–15000 Hz.
- Configurable lower and upper frequency limits and bass preference shelf.
- Guided louder/equal comparisons and manual adjustment, switchable during a session.
- A replayable built-in music excerpt to set your usual listening volume.
- Adaptive pair selection, randomized tone order, cross-checks and frequency refinement.
- Separate left/right calibration or both channels together.
- Named device profiles, a table browser with arrow-key navigation, and automatic saving after each accepted judgment.
- A main menu on launch and guided profile setup, including advanced settings.
- Resuming an unfinished session with the same `run` command.
- Bounded parametric EQ fitting, input attenuation for headroom, and native Easy Effects 8.2.8 JSON export.
- A silent demonstration and automated tests; those commands play no live audio.

There is one Python runtime dependency: NumPy. Audio uses PipeWire's existing `pw-play`; no Python audio package, GUI framework, SciPy or machine-learning runtime is required. Calibration does not change system audio settings or activate a preset.

## Quick start

Linux requires Python 3.11+, NumPy, and PipeWire's `pw-play`. Easy Effects applies
exported presets. For a release wheel or source archive, follow the
[installation guide](install.md). From a source checkout or extracted source
directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/phonix --version
.venv/bin/phonix
```

Launching without arguments in a terminal opens **Home**: **Profiles**, **New profile**, **Help**, **Tools**, and **Quit**. Use Up/Down and Enter, or `q`/Escape to return one menu. Menus keep your selected row when you return. Tools contains output-device information and the silent demo; Help groups getting started, listening controls and understanding results.

New profile asks for a name and opens a setup table with output device, frequency range, listening effort, bass sound, ears and comparison method. Everyday choices use labels such as **Both ears together** and **Adjust tones yourself**. **Advanced settings** contains custom frequency limits, answer caps, tolerances, playback levels and sampling settings. Setup and settings tables also show the highlighted option's description below the table, so it remains readable when columns extend beyond the terminal width. Select **Create profile** to save and immediately open its action menu. Back returns to the profile list with that profile highlighted. Cancelling creation discards the draft and returns to Home. Creating or browsing profiles does not play audio.

The application keeps one full-screen terminal UI open until you quit. Names, numeric values and export paths are entered in its text field; setup, calibration prompts, audio checks, reports, help and errors stay inside that screen. Type a response and press Enter, or use Escape to cancel/back. Up/Down and Page Up/Page Down scroll the screen's text; during tone comparisons, Up/Down adjust the marked tone instead, and Page Up/Page Down still scroll. Left/Right, Home/End, Backspace and Delete edit the text field. Calibration replaces instructions with the current task instead of accumulating previous comparisons. Its step and passage indicators stay visible while scrolling. Small terminals ask you to enlarge the window. The terminal is restored on exit.

The text cursor follows your editing position in the input field, including when long text scrolls. Menus hide the text cursor and indicate selection with the highlighted row and `>` marker.

Escape uses a 50 ms key-sequence wait so going back responds quickly while arrow keys remain available. The previous terminal-library delay is restored when the UI closes.

Open Profiles and select a profile to start/resume calibration, export an Easy Effects preset, delete it, change its bass shelf or tolerance, view a report, check audio, or edit playback settings. Playback changes have Save/Cancel; tone levels and sample rate are locked once either channel has saved judgments. Use the same equipment and listening volume when resuming. Selecting an output for the program does not switch the system's default output.

To use the bare `phonix` command in this shell, activate the project environment first:

```bash
source .venv/bin/activate
phonix
```

The existing commands remain available for scripting. `--help` always prints command help; launching without arguments when input/output is redirected also prints help. For example:

```bash
.venv/bin/phonix new my-headphones --range full --bass-db 4
.venv/bin/phonix run my-headphones
```

Start/Continue in Profiles (or the `run` command) presents separate screens,
with bold section headings for the current sounds and next action:

1. **Music-volume preview:** Press Enter to play a 25-second built-in excerpt of **Vibe Ace by Kevin MacLeod**. Adjust hardware/system volume to resemble your usual comfortable music-listening level. While it plays, **Enter, Escape or `s`** stops it early and returns to volume setup; stop keys work immediately, without another Enter. After playback finishes or stops, `r` replays the excerpt from its beginning, Enter or `c` continues to tone setup, `t` offers an optional early tone preview, and `q` returns. Stopping the excerpt does not record an answer or change profile settings.
2. **Reference-tone check:** Keep hardware/system volume fixed. A short 1 kHz tone plays in both channels. New profiles start from the music excerpt’s average energy around 1 kHz. Use the saved/default starting level unless it needs adjustment: `+`/`-` changes 1 dB and `++`/`--` changes 3 dB, replaying the tone at the new level. Confirm with Enter or `c` only when the tone is **clearly audible and comfortable**, rather than barely detectable. `r` replays; `q`/Escape cancels without saving draft adjustments. There is no noise comparison, automatic gain increase or three-equal-check requirement. The confirmed level is saved before calibration begins. If either channel has judgments, the saved level is locked and the check offers only confirmation, replay, help or return.
3. **Calibration comparisons:** The header identifies the current passage, ear and saved-answer count. The limit is a maximum, rather than a required number of tests. Two vertical sliders show the tone order, adjustable frequency and changes since the pair began, with controls for adjustment, replay and accepting a match. The reference-tone check shows its current level and whether it is locked.
4. **Choose your sound:** After a regional session meets its completion checks,
   choose Neutral/reference or an optional sound preference. Neutral/reference
   adds no voicing filters. Legacy shared-fit profiles retain three steps.

Enter `h` during any step to read detailed help, including digital levels. Returning from Help restores the same task without replaying audio or recording an answer. After calibration, a short summary shows what was saved and what to do next; **Progress** in the profile menu retains the detailed report.

The tone ceiling follows adjustments, capped at -3 dBFS peak. The reference range is -60 to -9 dBFS, always leaving at least 6 dB of upward comparison headroom. The original comparison allowance (default ±18 dB) is preserved where possible; at higher references it shrinks to respect the ceiling, and Help shows the available allowance. If the tone remains unusable at an adjustment limit, return to check the output and playback configuration. Changes require an explicit manual adjustment. Both preparation steps record no calibration judgments. On resume, restore the earlier hardware volume and confirm the saved reference tone. If that level is unusable, create a new profile rather than mixing listening levels in the existing session.

All profiles now use the same **built-in music excerpt**, assuming music-player volume at 100%. Its prepared stereo PCM measures **-14.1 LUFS** and **-1.0 dBTP** with offline FFmpeg analysis. It is a fixed reference for normalized music, rather than the measured average of your recordings. Tracks, player normalization and processing can change actual playback loudness.

The preview is bundled as a 25-second, 48 kHz stereo PCM16 WAV with short fades (about 4.8 MB). It was loudness-normalized once with true-peak limiting during preparation. Playback uses the finished file unchanged, with no runtime normalization, download or decoder dependency. Python's standard library and NumPy load it, and the existing PipeWire transport plays it on the profile's output. Its native 48 kHz rate is independent of the calibration tone rate; PipeWire handles output resampling as needed. System/hardware volume controls the actual listening level; the program does not change that volume. Lower it before starting the preview, then raise it to your usual comfortable music level.

Music credit: **Vibe Ace by Kevin MacLeod (incompetech.com)**, licensed under [Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/), obtained from [librosa's credited recording](https://github.com/librosa/data/blob/main/audio/Kevin_MacLeod_-_Vibe_Ace.txt). The excerpt covers source seconds 30–55; changes include excerpting, fades, resampling and offline loudness normalization with peak limiting. Full attribution is bundled in `phonix/data/LICENSE-Music.txt`; source and output checksums, preparation details and measurements are in `phonix/data/music_preview.json`. Calibration Help also shows the music credit. The music license is independent of the source-code and model-data licenses.

**Playback settings → Default tone levels → Save** restores the music-based -38.5 dBFS reference and -20.5 dBFS ceiling before calibration has begun. The editor hides tone levels and sample rate once either channel has judgments; the chosen output can still be changed. The unused noise controls have been removed from the menus. Existing noise fields and old `--noise-level`/`--noise-ceiling` arguments remain accepted for compatibility, but they affect neither music nor the reference-tone check. Saved tone levels and calibration evidence are retained. When resuming a session begun with the old pink-noise preview or noise/tone match, restore the system volume used for those earlier judgments instead of choosing a new volume or reference level.

This is a subjective listening reference, not a measurement or normalization of the loudness of your music. `run` resumes previously accepted judgments. Use a physical output and bypass processing for calibration. The amplitude ceilings are digital dBFS, not acoustic sound-pressure guarantees.

For new CLI/UI profiles, the reference starts at **-38.5 dBFS peak**, derived offline from the excerpt's **935–1065 Hz** band. Its full 25 seconds, including fades, have -41.49 dBFS RMS in that band when averaging the left and right channel powers. A steady sine with the same RMS has -38.48 dBFS peak; the default rounds this to -38.5 dBFS. The short tone's playback fades slightly reduce its time-mean RMS. This is a digital energy reference, not an equal-loudness claim or an acoustic-level estimate. Existing profiles keep their saved levels, and missing legacy tone fields still use -42/-24 dBFS. Defaults are read from bundled metadata, with no runtime FFT or music decoding during profile creation.

The check establishes a usable operating level; it is not a hearing-threshold test or a measurement of acoustic diffuse-field accuracy. Keep the default unless an adjustment is necessary, and retain the same level when resuming. This simpler workflow does not demonstrate improved accuracy or equal acoustic levels across devices. The neutral correction still comes from frequency comparisons interpreted using the DF target and fixed hearing model. **Settings → Audio and volume → Playback settings** can also edit starting levels for a profile with no judgments. For scripting, the equivalent initial override remains:

```bash
.venv/bin/phonix run my-headphones --reference-level -30 --max-level -12
```

This saves the requested starting tone levels while preserving the default ±18 dB comparison allowance. With `new`, specifying only `--reference-level` derives a ceiling 18 dB above it, capped at -3 dBFS; an explicit `--max-level` overrides that ceiling. With `run`, omitted levels keep the profile’s existing settings. The two-step preparation still runs and asks you to confirm the tone is audible and comfortable; music playback stays fixed. Tone levels cannot change after any channel has saved judgments. A configured tone ceiling must be at least 6 dB above its reference and no higher than -3 dBFS.

The example requests a +4 dB deep-bass shelf relative to unmodified diffuse field. Omit `--bass-db` or set it to `0` for the baseline target. The shelf preference is applied when exporting; calibration tones and inferred neutral correction are independent of that choice.

During UI comparisons, two vertical sliders show the frequencies in playback order, with `>` and **ADJUST** marking the adjustable tone (the higher frequency, even when it plays first). Up/Down immediately change its relative level by **0.5 dB** and select manual matching. The screen redraws after each change; arrows do not play sound. Press **Enter with an empty command field** to play the modified pair, then type `e` and Enter to save an equal-loudness match. Unheard changes cannot be saved. Each slider shows its actual digital peak and change since this pair began; the relative adjustment is also shown. Both rails share a scale that expands for larger changes. Playback limits can shift both tones together, which is reflected in the sliders. These are digital level indicators, not measured acoustic loudness.

Guided choices remain available: `1` for first louder, `2` for second louder, `e` for equal, or `u` if unsure. `m`/`g` switch manual/guided mode, `r` replays, `s` skips an unusable pair, `h` opens help and `q`/Escape saves accepted judgments and returns. Letter commands are followed by Enter. Manual text shortcuts `+`/`-` change 0.5 dB and `++`/`--` change 2 dB; in the UI they also wait for Enter to replay. Outside the full-screen UI, those text shortcuts retain automatic replay and empty Enter also replays. Replay, adjustment and help do not count as answers. Ctrl-C stops playback and preserves already accepted judgments.

**Frequency-dependent tone levels:** New CLI/UI profiles place both comparison tones around an estimated common perceived loudness, anchored to the saved 1 kHz reference. The fixed hearing/DF estimate supplies the initial shape; accepted answers refine it. For example, bass may start at a higher digital level than mids. The screen shows actual levels in playback order. These are starting estimates, not a guarantee of equal loudness or a new music-level measurement at every frequency.

If either tone would exceed the saved ceiling or -78 dBFS floor, both shift together to preserve the requested level difference. Manual controls adjust that difference; the other tone can also move when a bound requires a common shift. Replay and help keep the same levels. Each accepted answer records both actual digital peaks, and the fit uses their B-minus-A difference. Starting compensation is not added separately to the exported EQ. Pure waveforms, equal pair durations, fades and nearby-frequency comparisons remain unchanged.

The playback method is saved as `tone_leveling=adaptive-v1`. Existing profiles (including incomplete sessions) keep their original fixed-reference method and all observations; create a new profile to use the new method. Reset retains the saved method with the other settings. A range without 1 kHz uses the fixed baseline to anchor its level, without claiming to measure the missing midrange. The hearing model still assumes level independence; this improves how probes are presented without establishing acoustic DF accuracy.

**Nearby comparisons:** New calibration pairs are limited to **one octave**, meaning the higher frequency is at most twice the lower frequency (for example, 40 vs 80 Hz or 8 vs 16 kHz). The adaptive selector favors shorter gaps while choosing useful cross-checks. It links bass, mids and treble through overlapping local comparisons; it never schedules a 20 Hz vs 16 kHz comparison. The current pair's gap is shown in octaves.

At the start of calibration and as needed later, intermediate points are added to connect the local comparisons and provide a nearby third frequency for cross-checking each possible link. The grid also needs at least two independent cross-check cycles and two partners per frequency. These additions stay within the profile's Maximum frequency points setting. If that cap cannot support the local grid, calibration reports the limitation without discarding frequencies or judgments. When creating a profile, **Advanced settings → Maximum frequency points** controls the cap (up to 32); a narrower range can also reduce the required points. Old judgments, including distant comparisons, remain available to the fit, but future pairs obey the one-octave limit. Closer comparisons can require more judgments to constrain the overall curve; the selected tolerance and judgment cap are unchanged. Inaudible bass still needs to be skipped and remains unmeasured unless other usable comparisons establish its coverage.

## Check silent playback

From the project directory, test a familiar midrange tone without recording judgments:

```bash
.venv/bin/phonix audio-check my-headphones
```

Press Enter to play a ramped, one-second 1 kHz tone in both channels. `r` replays it; `q` returns. The default level and device come from the profile. Player completion means the playback process succeeded; it does not establish that the tone was audible.

Every playback stream starts with 500 ms of digital silence before its first sound, including tone pairs, previews, audio checks and replays. This gives the output time to settle and helps prevent the first tone's beginning from being cut off during startup. The silence is played through the same stream; tone levels, durations and the gap between paired tones are unchanged. Whether 500 ms is sufficient for your equipment needs a listening check.

To separate default routing from a particular output, list sinks with `devices`, then copy its node name or serial into the check command. An optional level override applies only to the check:

```bash
.venv/bin/phonix devices
.venv/bin/phonix audio-check my-headphones --device YOUR_NODE_NAME --level -36
```

The check leaves system volume, profile settings and listening judgments untouched. Levels must remain within the profile's playback bounds; it never increases volume automatically. The music-based default is -38.5 dBFS peak; older profiles may use -42 dBFS. Either can be difficult to hear on laptop speakers depending on system volume. If only a higher-level check is audible, set suitable reference and ceiling settings using `run` on a profile with no judgments, or create a new profile (for example `--reference-level -36 --max-level -18` preserves the default 18 dB comparison allowance); establish a comfortable volume before collecting judgments. If a directly targeted check remains silent while ordinary audio works, inspect the player stream's mute and routing settings during playback. [WirePlumber's `wpctl` documentation](https://pipewire.pages.freedesktop.org/wireplumber/man/wpctl.html) describes `status`, `inspect` and `get-volume` for read-only checks.

Easy Effects can redirect playback streams into its virtual sink even when `pw-play --target` names a physical output. This was reproduced on this machine: the player linked to `easyeffects_sink` while the speaker stayed suspended. Calibration and audio-check streams now have their own `phonix` identity and use PipeWire's `node.dont-move=true` and `node.dont-fallback=true` properties. With Easy Effects open, silent live routing checks verified direct speaker links for both an explicit speaker target and the current default output. See [WirePlumber's linking policy](https://pipewire.pages.freedesktop.org/wireplumber/policies/linking.html) for the property semantics.

Choose a physical output for calibration. If the system default itself is the Easy Effects virtual sink, use the physical sink's name with `--device` when creating the profile or running the check. The properties keep each stream on its chosen target; they do not choose a different system default or disable other applications' processing.

## Customize a profile

New CLI/UI profiles use a two-stage regional calibration. The default range remains 60–15000 Hz, partitioned into bass below 250 Hz, midrange from 250 Hz to 2 kHz, and treble above 2 kHz. Custom limits clip those regions; a lone endpoint joins its neighboring region. The program learns midrange, bass and treble shapes separately, checks their consistency, then freezes each shape. Pure-tone comparisons across nearby region boundaries determine constant regional gains, with midrange as the anchor when available. No pair spans more than one octave. Finally, repeated shape/boundary checks verify the combined result.

Comparisons stay grouped by range throughout calibration, including verification: midrange, bass, then treble, omitting skipped or absent regions. Each range's required checks (including reversed tone order) finish before the next range begins. Regional balance runs separately, completing the bass/midrange boundary before the midrange/treble boundary. The active range or boundary appears beside the current passage. Extra refinement, when needed, also completes a block of independent repeats within one range/boundary before moving on; it may require another grouped pass if uncertainty remains. Skipping an individual pair does not switch ranges, and resuming retains the active refinement checkpoint. These changes alter scheduling only; the DF target, inference and completion tolerances stay fixed.

During balance, accepted answers cannot reshape a frozen region. Its uncertainty remains part of the balance estimate. If strong repeated contradictions appear, Progress identifies the regions needing review; **Settings → Revisit regional shape** reopens a selected region while keeping all answers and the other frozen shapes. Fresh local checks are required, followed by fresh balance and verification. Old boundary answers remain saved for audit but do not determine the new balance after reopening. A tighter listening tolerance can also require reopening a previously frozen shape.

The local shape uncertainty threshold is the chosen shape tolerance divided by √2, reserving uncertainty for regional balance. Each region needs connected observations, at least two independent cycles, two links per frequency, and acceptable disagreement before freezing. Geometry is prepared before the first freeze; the grid then stays fixed. Restrictive point caps, inaudible comparisons, disagreement or exhausted answer budgets can leave a session incomplete. Sparse custom ranges may need additional points to support regional checks. An extended range requires a new profile once any shape is frozen.

Freezing and balancing refer to the sampled matching curve. Between adjacent region endpoints, the export fitter interpolates in log frequency and approximates that continuous curve with PEQ; it does not impose hard gain steps at 250 Hz or 2 kHz. Exact sampled shapes and smooth exported filter responses can differ within the reported fitting error. This is still listening-only, model-based DF calibration, not an acoustic measurement.

Existing profiles keep their earlier shared-fit workflow and evidence. New profiles save `calibration_method=regional-v2`; missing legacy fields and low-level `Settings()` use `global-v1`. Reset retains the saved method and preferences. Create a new profile to adopt the new workflow.

### Skip local calibration

Before a channel starts, **Local calibration → Skip all local passes** takes you straight to regional balance. During any local shape pass, **k → Skip this region** skips that region after confirmation. **s** still skips only the current pair. Skipping does not add a judgment, erase answers, or block a pair. Choosing full local calibration or balance-only is saved, so resuming does not ask again.

Skipped regions keep their original sound shape: the requested calibration correction is constant within each skipped region, and only its overall gain is determined from nearby boundary comparisons. Partial local answers remain in the history but do not supply a local correction after skipping. Verification repeats boundary checks and checks only the shapes you actually calibrated. At least two regions are needed for balance-only mode; a single-region range has no relative regional levels to measure.

Profiles show **shape skipped** in their status. Progress, Preview EQ and export reports identify the **shape uncalibrated** regions. Completion then means the retained local shapes and the conditional regional balance checks passed, not that the skipped shapes meet the DF target. Balance uncertainty assumes no local correction in skipped regions; it cannot quantify their unknown response. Export requires measured links connecting regional gains and measured shapes for every non-skipped region, even with `--allow-incomplete`. It never exports a guessed fine-frequency correction for skipped regions. The PEQ fitter smooths/approximates the requested regional gains, so the rendered response can vary within the reported fit error. Bass and sound preferences remain separate, deliberate additions.

Use **Settings → Revisit regional shape** to calibrate a skipped region later. Its saved answers are retained, the other regions stay settled, and new balance/verification checks are required afterward. Reset clears all skip choices and answers.

### Choose a sound profile after calibration

After regional balance and verification finish for every channel, **Step 4 → Choose your sound** offers the following preferences. You can change them later under **Settings → Sound profile**, without repeating calibration.

| Sound profile | Added filters relative to the reference | Character |
| --- | --- | --- |
| Neutral/reference | None | Keep the model-based DF calibration |
| V-shaped | +3 dB low shelf at 180 Hz; +3 dB high shelf at 4 kHz | More bass weight and treble sparkle |
| Warm tilt | +2 dB low shelf at 250 Hz; −2 dB high shelf at 2.5 kHz | More body and a gentler top end |
| Bright | +2 dB high shelf at 3 kHz | A modest brightness lift |

These are fixed taste presets, not measured neutral targets. All shelves use Q=1/√2 with smooth transitions. **Neutral/reference adds no voicing**; the separate Bass sound setting still applies if nonzero. For an unmodified DF reference, choose Neutral/reference and set Bass level to 0 dB. Calibration stimuli, observations, inference, frozen shapes and regional gains never depend on the selected preference.

Preferences appear as **Sound preference** filters in Preview EQ and in the Easy Effects preset. They consume one or two of the requested total filter slots; a separate nonzero bass shelf consumes another, and at least one correction slot must remain. Export fits the neutral correction with the remaining slots, then appends the exact preference filters and computes headroom for all filters. Preferences can affect frequencies outside the calibrated range; this does not supply measurements there. The report separates neutral correction, extra bass, sound voicing and their combined requested curve.

CLI creation also accepts `--sound-profile neutral`, `v-shaped`, `warm-tilt` or `bright`. The post-calibration menu opens with the saved choice highlighted. Cancel keeps that choice. To save several variants from the same device calibration, change Sound profile and export each to a different filename. No audio audition of the EQ is performed inside this selector; import the exported preset into Easy Effects to listen to the result.

Choose a session with `--range`:

| Range | Frequency limits | Initial / maximum points |
| --- | --- | --- |
| `full` (default) | 60–15000 Hz, mid-bass through treble | 17 / 32 |
| `bass` | 20–1000 Hz, including a midrange reference | 11 / 21 |
| `midrange` | 200–2000 Hz | 9 / 21 |
| `treble` | 2000–16000 Hz | 7 / 17 |

New CLI and UI profiles default to 60 Hz–15 kHz. Existing profiles retain their saved ranges. Advanced settings or `--min-hz`/`--max-hz` can select custom limits within the supported 20–16000 Hz range; for example, `new my-headphones --min-hz 20 --max-hz 16000` includes sub-bass and the highest supported treble. The built-in music preview is independent of the chosen tone range.

Choose one of three listening tolerances with Up/Down and Enter when creating a profile in a terminal, or supply `--tolerance`:

| Choice | Command-line value | Shape tolerance | Judgment cap per channel | Description |
| --- | --- | --- | --- | --- |
| Casual | `doubt` | 4 dB | 80 | If you are in doubt |
| Focused (default) | `convinced` | 3 dB | 150 | If you are convinced |
| Meticulous | `obsessed` | 2 dB | 300 | If you are now obsessed, and want to perfect your sound profile |

These tolerances bound the estimated 95% shape uncertainty under the listening model; they are not measured error against the DF target. Every choice keeps the same 1.5 dB cross-check disagreement limit, two independent comparison cycles, and two usable links per sampled frequency. Smaller tolerances usually need more comparisons and can add more contour detail. The caps apply to every range; sessions can finish earlier or remain incomplete at the cap. Separate left/right calibration has a cap for each ear. Manual matches can take several adjustments for one accepted judgment.

```bash
.venv/bin/phonix new my-headphones --tolerance convinced
```

Existing profiles keep their saved limits until you select **Settings → Listening effort** in the profile menu, or run:

```bash
.venv/bin/phonix tolerance my-headphones --preset doubt
```

Changing this choice preserves frequencies, judgments, skipped pairs, playback levels and bass preferences. It replaces the stopping tolerances and total judgment cap. If more judgments than the new cap are already saved, the cap is kept at the used count; no evidence is discarded. A profile's complete/incomplete status is recalculated using its selected tolerance. Further judgments can still be added explicitly. Without a terminal, profile creation defaults to `convinced` and does not prompt.

Full-range sessions connect all regions in one comparison graph. A 1 kHz point provides a common midrange reference whenever the range includes it. A 12.5 kHz point supports upper-treble comparisons when the range extends above it. Separate narrow-range profiles determine shape within their own ranges; their overall gains cannot be combined into a full-range calibration without comparisons linking those regions.

Legacy shared-fit profiles follow these regional passes within one profile:

1. **Early boundary checks:** sample nearby links across the bass/midrange and midrange/treble boundaries.
2. **Midrange, then bass, then treble shape:** establish local coverage and redundant comparisons in each region. These are preliminary passes; they do not each spend the whole budget reaching the final tolerance.
3. **Regional balance:** revisit two distinct links at each boundary where available, checking each in both playback orders.
4. **Grouped verification and refinement:** finish a local pair's checks in each region and then each boundary in both orders, then reduce uncertainty and disagreement with further grouped comparisons.

The scheduling boundaries are 250 Hz and 2 kHz. A lone endpoint joins its neighboring region, and regions outside the selected range are omitted. Every pair remains within one octave, including balance checks; the boundaries do not limit EQ filter locations. All observations enter the same fit, preserving relative regional levels. There is no separate regional normalization or requirement to make whole noise bands equally loud. The DF target and optional bass shelf are unchanged.

The UI and **Progress** show the current pass. Each ear saves its own pass checkpoint when calibrated separately. Resuming an unfinished older profile keeps its entire history and adopts these passes; old judgments can supply local coverage but do not replace fresh closing checks. Existing completed profiles retain their status. Cancellation records no judgment or pass advancement. All passes use the existing judgment cap; reaching it before verification and consistency checks finish leaves the profile incomplete.

Explicit frequency/point flags override the range; `--budget`, `--shape-tolerance` and `--drift-tolerance` override the tolerance choice:

```bash
.venv/bin/phonix new another-device \
  --range treble \
  --min-hz 2500 --max-hz 14000 \
  --channels separate --mode manual \
  --budget 240 --shape-tolerance 1.5 --drift-tolerance 1.5
```

Supported frequency limits are 20–16000 Hz. The population source marks its response above 12.5 kHz as unvalidated extrapolation; profiles and reports retain this qualification. From 15 to 16 kHz, the sensitivity proxy explicitly holds its bundled 15 kHz endpoint constant; the DF target continues to use its source data. This adds no validated hearing data and leaves the baseline through 15 kHz unchanged. Probes can play through other equipment, but the target and sensitivity approximation are intended for headphones.

Bass and midrange use the existing fixed linear middle-ear proxy. It omits level-dependent loudness, hearing thresholds and temporal integration, so enabling low-frequency probing does not validate the acoustic accuracy of its target estimate. [Research on loudness models](https://journals.sagepub.com/doi/10.1177/2331216514550620) explains why level and threshold behavior matter. Keep headphone position and seal consistent: leaks can substantially reduce bass response ([Audio Precision](https://www.audioprecision.com/blog/headphone-testing-tips)). Skip inaudible or distorted comparisons. Bass pairs allow at least 15 cycles of the lower tone, with the same duration for both tones, without increasing their digital levels. Adaptive refinement can add nearby frequencies when wider comparisons cannot support independent cross-checks within the level limits.

Existing treble profiles retain their saved frequencies, observations and hearing baseline. Profiles created before bass preferences were added default to a 0 dB shelf.

To extend a legacy profile, or a new regional profile before any shape is frozen, select **Settings → Extend to 16000 Hz** in its menu, or run:

```bash
.venv/bin/phonix extend my-headphones
```

An empty profile gets a new starting grid. A session with saved judgments or skipped pairs keeps its entire grid and history, adding the new upper endpoint. The maximum point allowance increases if needed, up to the supported 32 points; extension is rejected if an existing 32-point session has no space. Added frequencies require new comparisons before export, including for previously complete sessions. If its judgment budget is exhausted, Continue offers additional judgments. Saved profiles retain their previous limits until explicitly extended; this command can also set a higher upper limit within 16 kHz using `--max-hz`.

## Choose bass level

`--bass-db 4` adds 4 dB at deep-bass frequencies relative to the calibrated DF baseline, tapering to 0 dB in the midrange. Here, "flat" means no added shelf; the DF reference itself is not a flat acoustic response. The bass setting accepts -12 to +12 dB. `--bass-hz` sets the shelf's half-gain frequency, defaulting to 105 Hz and configurable from 40 to 300 Hz. Thus a +4 dB, 105 Hz shelf adds +2 dB at 105 Hz and approximately +4 dB well below it. It is a smooth RBJ low shelf with Q = 1/√2.

The profile's **Settings → Bass sound** menu groups bass level and bass warmth. **Bass warmth** offers **Deep rumble**, **Balanced bass**, **More warmth**, and **Manual input**. Warmth chooses the shelf's midpoint and has no effect when bass level is 0 dB. Back/Escape keeps your choices; other calibration settings and judgments remain unchanged. The three presets are also available with `--bass-style` on `new` or `export`:

| Style | Shelf midpoint | Sound with positive boost |
| --- | --- | --- |
| `sub-bass` | 60 Hz | Deep rumble, with less added upper-bass warmth. |
| `balanced` (default) | 105 Hz | Bass weight and punch. |
| `warm` | 180 Hz | Broader bass lift, with more body and warmth. |
| Manual input | 40–300 Hz | Choose your own shelf transition frequency. |

These descriptions indicate the emphasis of a boost; a negative gain reduces those regions instead. A shelf has a transition midpoint, rather than a bell filter's peak center. All options retain the same deep-bass gain set by `--bass-db`; `0` dB leaves every style neutral. The menu saves the chosen midpoint without changing judgments or gain. Manual input accepts decimals, such as `127.5`; invalid values prompt again. Enter keeps the current midpoint, and `q`, Escape in the selection menu, or Ctrl-C cancels without saving a change. A custom midpoint highlights Manual input when reopening the menu. `--bass-hz` remains available for a custom midpoint on the command line and cannot be combined with `--bass-style`.

```bash
.venv/bin/phonix new warm-headphones --bass-db 4 --bass-style warm
.venv/bin/phonix export my-headphones --bass-style sub-bass --output exports/my-headphones-sub.json
```

For example:

```bash
.venv/bin/phonix new bass-headphones --range bass --bass-db 4 --bass-hz 105
.venv/bin/phonix new mid-headphones --range midrange
```

You can override the preference on export without changing or repeating a listening session:

```bash
.venv/bin/phonix export my-headphones \
  --bass-db 4 --bass-hz 105 --output exports/my-headphones-bass4.json
```

Adding a shelf to a treble-only profile changes bass preference but supplies no measurement of that device's bass response. The export report identifies this case. Input attenuation preserves the relative bass lift while providing headroom for the combined correction and shelf.

`--channels together` sends each tone to both channels and exports identical left/right EQ. `--channels separate` calibrates each independently and exports split-channel EQ. Each channel's overall gain is treated separately; this does not perform a between-ear loudness-balance test.

`--budget` limits accepted judgments **per channel**, not a promised convergence count. Replays, manual adjustment steps and unsure responses do not consume that budget. The session stops earlier if its model-based consistency thresholds are met. To add more judgments or inspect results:

```bash
.venv/bin/phonix run my-headphones --extra-judgments 120
.venv/bin/phonix report my-headphones
.venv/bin/phonix list
```

To choose an output, list PipeWire sinks and use a node name or serial:

```bash
.venv/bin/phonix devices
.venv/bin/phonix new wired-headphones --device YOUR_NODE_NAME
```

Without `--device`, playback follows the current default output. Keep the physical device and listening conditions consistent with the profile. The saved calibration configuration and hearing-model version remain fixed when resuming. A skipped pair stays skipped in that profile; other pairs can establish coverage.

## Browse and manage profiles

From the project directory:

```bash
.venv/bin/phonix list
```

In a terminal, this opens a table with profile name, status, frequency range, channels, judgments used versus budget, bass gain, shelf midpoint and shape tolerance. Up/Down selects a row and Enter opens its actions. Use Left/Right to scroll columns on narrow terminals, and `q` or Escape to return. Selecting a profile does not change the system output or activate its EQ.

| Status | Meaning |
| --- | --- |
| `new` | No calibration judgments saved. |
| `unfinished` | Calibration started; an unresolved channel still has judgment budget. |
| `incomplete` | Unresolved channels have exhausted their budget. Continue offers additional judgments. |
| `complete` | Every channel meets the listening consistency checks and has finished closing regional checks when using regional passes. This does not certify measured acoustic accuracy. |
| `unreadable` | The saved file cannot be loaded or evaluated; other profiles remain available. |

`Answers / limit` counts accepted comparisons, not replays or preview sounds. The limit is a maximum; sessions may finish earlier. `Both` means each comparison played through both channels together. For separate calibration, the table shows left and right counts independently, such as `L 12/150; R 3/150`.

The profile menu keeps **Start/Continue calibration**, **Progress**, **Preview EQ**, **Export to Easy Effects**, **Settings**, **Manage profile**, and **Back** directly accessible. Complete profiles omit Start/Continue until coverage or a stricter tolerance needs further calibration. Calibration uses the existing volume setup and saves progress as usual.

| Menu | Options |
| --- | --- |
| Settings | Audio and volume, Listening effort, Bass sound, EQ filter style, Back |
| Audio and volume | Check sound, Playback settings, Back |
| Manage profile | Reset, Delete, Back |

Settings also offers **Extend to 16000 Hz** for profiles ending between 12 and 16 kHz. Menu titles show your current location; Back/Escape returns to the parent menu. Errors return to the menu where the action was chosen. **Export to Easy Effects** asks for a path, defaulting to `exports/PROFILE.json`, and writes a 12-band preset plus its report. Disconnected coverage cannot be exported; connected estimates that have not met consistency checks require explicit confirmation. Replacing an export also requires confirmation. **Manage profile → Delete** defaults to Cancel and removes only the selected profile and its judgments; previously exported presets and reports are kept.

Choose **Manage profile → Reset** to restart calibration under the same profile name and saved settings. The confirmation defaults to **Cancel**; select **Reset profile** to clear every channel's judgments, skipped pairs, added frequency points and regional progress. This is equivalent to creating a fresh profile with that name and configuration. The profile returns to **new**, with **Start calibration** highlighted, and its menu stays open. Existing exported presets/reports and other profiles are kept. Escape cancels without changing the profile.

For a text table without the menu:

```bash
.venv/bin/phonix list --plain
```

Redirected output and terminals with `TERM=dumb` automatically use this text view. The menu uses Python's built-in `curses`; no additional Python package is required.

Each profile's **Preview EQ** opens a scrollable table of the same 12-band preset used by menu export: channel, filter number, type, frequency, gain, Q and purpose (**Correction** or **Bass preference**). The title shows input gain (preamp attenuation for headroom) and labels unfinished estimates **provisional**. Both-channel profiles show one shared filter set; separate-channel profiles show Left and Right sets. Up/Down and Page Up/Page Down navigate filters, Left/Right scroll columns, and Enter or Escape returns to the profile menu. Preview writes no files and plays no audio. Profiles whose comparisons do not yet connect all sampled frequencies explain that more calibration is needed instead of showing filters based on unmeasured priors.

Choose **Settings → EQ filter style** for **Auto** (default) or **Bells only**. Auto compares correction bells against combinations with one low shelf, one high shelf, or both, at the same total band count. It selects shelves only for a useful improvement in the combined curve and contour approximation, with limits on unmeasured tails and required attenuation. Residual-based starting guesses also improve bell fitting. Bells only limits the neutral correction to bells; your separately chosen bass preference still uses its exact low shelf. Changing fitting mode preserves all calibration judgments and does not require additional listening. Existing profiles without a saved mode use Auto without rewriting their files.

Auto evaluates multiple fits when generating a preview or export, which can take several seconds. Fitting uses NumPy and adds no runtime dependency or work to tone playback.

## Export to Easy Effects

```bash
.venv/bin/phonix export my-headphones \
  --output exports/my-headphones.json --bands 12
```

This writes the preset and `exports/my-headphones.report.json`. The report contains listening uncertainty, cross-check disagreement, PEQ shape RMS error, accumulated contour mismatch, the neutral correction and preference curves, sample rate, data provenance and headroom attenuation. Correction bell gains are bounded to ±6 dB and Q to 0.4–8; the preference shelf allows ±12 dB. Input attenuation includes 1 dB of margin beyond the estimated combined filter boost.

`--bands` is the total export count, defaulting to 12. A nonzero shelf uses one slot, leaving 11 correction bells by default; at least 2 total slots are required in that case. With a 0 dB preference, every slot is available for correction. The shelf is added after fitting the neutral correction, so it is preserved exactly rather than approximated by bells or absorbed into the overall gain offset.

In Easy Effects, select output processing and import `my-headphones.json` through **Presets**. Import the preset file, not the separate `.report.json`. The preset contains only Equalizer with the appropriate channel settings; retain your existing preset if you want to return to its effects. Use the same processing sample rate as the profile's `--sample-rate` (48000 Hz by default) for the closest numerical agreement. [Easy Effects preset documentation](https://wwmm.github.io/easyeffects/user_interface/userpresets.html).

Export normally requires complete frequency coverage and satisfied listening consistency thresholds. `--allow-incomplete` permits a connected but not converged estimate, with its uncertainty retained in the report. It never permits exporting unmeasured, disconnected frequencies. Filter error is reported even if it exceeds the shape tolerance; increasing `--bands` may improve the fit. `--overwrite` is required to replace existing export files.

For scripting, `new --filter-mode auto|bells` saves the fitting mode. `export --filter-mode auto|bells` overrides only that export. Reports record the selected filter combination, candidate scores and each filter's purpose. Correction shelves have fixed Q = 1/√2 and gains limited to ±6 dB; the independent bass preference keeps its -12 to +12 dB range. Shelves affect frequencies beyond the calibrated range: the report includes both correction-only and total outside-range peaks, and headroom includes all filters. Better numerical approximation is not a measurement of personal diffuse-field accuracy.

PEQ filters have tails outside the selected frequency range. Correction tails are penalized during fitting. The report's outside-range peak includes the intentional bass shelf as well; the range is not a sharp processing boundary.

## Installation and prerequisites

Python 3.11+, NumPy 1.24+, and `pw-play` from PipeWire's command-line tools are required. `pw-dump` is needed only for `devices`. Easy Effects applies the exported preset.

For a normal installation, use the Quick start above or the
[wheel/source installation guide](install.md). For development, install
with `.venv/bin/python -m pip install -e .` so source edits take effect immediately.
When a compatible NumPy is already installed system-wide, the offline development
setup verified here is:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-index --no-deps --no-build-isolation -e .
.venv/bin/phonix --help
```

`--system-site-packages` makes existing NumPy available in the environment.
`--no-deps` skips runtime dependency downloads and `--no-build-isolation` uses
the environment's installed setuptools (77+ required). `--no-index` prevents
package downloads. Use the normal installation when those dependencies are
missing. `-e` installs in editable mode.

From the project directory, `python3 -m phonix` also works directly when NumPy is available. It requires no build step.

Profiles default to `$XDG_DATA_HOME/phonix/profiles`, normally `~/.local/share/phonix/profiles`. If that directory does not exist and the previous `treble-calibrator/profiles` directory does, phonix reuses the previous directory to keep saved profiles accessible. No profiles are moved or rewritten by the rename. If both directories exist, phonix uses its own directory; `--data-dir` can select either location explicitly. For project-local storage, put the global option before the subcommand:

```bash
.venv/bin/phonix --data-dir profiles new local-headphones
.venv/bin/phonix --data-dir profiles run local-headphones
```

The `profiles/`, `exports/` and `.venv/` directories are ignored by Git.

## Verification

From this project directory:

```bash
python3 -m unittest discover -s tests -v
python3 -m phonix --data-dir profiles demo \
  --name another-silent-demo --range full --bass-db 4 \
  --output exports/another-silent-demo.json
```

The tests cover noisy guided/manual estimates, all session ranges, low-bass cross-check refinement, drift, contradictory comparisons, unreachable matches, channel isolation, amplitude bounds, legacy profile recovery, invalid inputs, PEQ recovery, shelf response and the Easy Effects export field contract. Music tests verify asset integrity, native-rate playback, the full playback timeout, replay, cancellation and unchanged session evidence. Reference-energy tests check known synthetic tones, stereo power averaging without phase cancellation, full-duration averaging, bundled band energy, new-profile defaults and explicit overrides. All transport is mocked; optional FFmpeg checks analyze files silently. The demo models a synthetic headphone with a known treble peak and bass roll-off. Its preset is **not a calibration for your real headphones**.

To rebuild the music asset for development, obtain the exact credited source recording linked above and run this from the project directory, with FFmpeg installed:

```bash
.venv/bin/python tools/prepare_music_preview.py /path/to/Kevin_MacLeod_-_Vibe_Ace.hq.ogg
```

The helper verifies the source checksum, prepares the excerpt and updates its measurements and metadata, including the precomputed tone reference from the final PCM16 excerpt. `--output-dir` can write a review copy elsewhere instead of replacing the bundled files. Neither this helper nor FFmpeg is required to use phonix.

Verified here: Python 3.14.7, NumPy 2.4.6, editable installation, silent simulation, the Easy Effects 8.2.8 source schema, and silent live PipeWire routing with Easy Effects open. Audibility and importing a preset into the running Easy Effects UI still require a listening/UI check.

## Building a delivery

From the source directory, using an environment with setuptools 77+ installed:

```bash
.venv/bin/python tools/build_release.py
```

This builds the source archive, builds the wheel from an extracted copy of that
archive, and writes both to `dist/`. It also creates a Linux release ZIP with
installation instructions, release notes, license notices and `SHA256SUMS`.
The ZIP's checksum is written beside it. `--output-dir /path/to/output` changes
the output directory. Building uses local setuptools and does not download
dependencies. The generated files, virtual environment and saved profiles are
excluded from the source archive; build tools and tests are included.

The wheel works with a compatible Python installation; it is not a standalone
executable and does not bundle NumPy or PipeWire. Python/NumPy minimum-version
coverage is not yet verified. [Release notes](../CHANGELOG.md) record the tested
environment and remaining listening/UI checks.

## License

Original source code and documentation use the [MIT license](../LICENSE).
Bundled reference data and music retain their separate licenses and attribution;
see [third-party notices](../THIRD_PARTY_NOTICES.md). These notices accompany both
the wheel and the source archive.
