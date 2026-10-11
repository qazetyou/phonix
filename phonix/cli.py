"""Terminal menus and scriptable commands for listening calibration."""

import argparse
import copy
from dataclasses import asdict, replace
import os
from pathlib import Path
import subprocess
import sys
import shlex

import numpy as np

from . import __version__
from .audio import Player, comparison_levels, devices
from .engine import Pair, add_observation, fit_channel, pairs_for, reference_base_level, refine
from .model import MAX_HZ, Reference, provenance
from .music import MUSIC_DURATION_SECONDS, music_reference_levels, preview_info
from .peq import bass_shelf, easyeffects_preset, response
from .profiles import (BASS_STYLES, REFERENCE_MAX, TONE_CEILING_MAX, TOLERANCE_PRESETS,
                       REFERENCE_NOISE_PRESET, Settings, Store, atomic_json)
from .terminal import (comparison_ui, music_preview_controls, preserve_screen, read_input,
                       run_ui, select_table, show_screen, table_lines, tone_slider_lines)
from .workflow import next_comparison, progress, ready, start_workflow, verified
from .voicing import SOUND_PROFILES, fit_voiced, voicing_filters


# Bass includes a midrange reference so its relative level is observable.
RANGES = {
    "bass": dict(min_hz=20.0, max_hz=1000.0, initial_points=11, max_points=21, budget=400),
    "midrange": dict(min_hz=200.0, max_hz=2000.0, initial_points=9, max_points=21, budget=300),
    "treble": dict(min_hz=2000.0, max_hz=16000.0, initial_points=7, max_points=17, budget=240),
    "full": dict(min_hz=60.0, max_hz=15000.0, initial_points=17, max_points=32, budget=600),
}


def default_directory():
    data_home = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    current = data_home / "phonix/profiles"
    legacy = data_home / "treble-calibrator/profiles"
    # Reuse existing evidence without moving or rewriting the user's profiles.
    return legacy if not current.exists() and legacy.is_dir() else current


def parser():
    command = argparse.ArgumentParser(prog="phonix", description="Listening-based bass, midrange and treble calibration with approximate B&K 5128 DF PEQ export")
    command.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    command.add_argument("--data-dir", type=Path, default=default_directory(), help="Profile storage directory")
    commands = command.add_subparsers(dest="command")
    create = commands.add_parser("new", help="Create a named device profile")
    create.add_argument("name")
    create.add_argument("--range", choices=list(RANGES), default="full", help="Session range (default: full, 60–15000 Hz, mid-bass through treble). Bass includes a 1 kHz reference")
    create.add_argument("--tolerance", choices=list(TOLERANCE_PRESETS), help="doubt: 4 dB / 80 judgments; convinced: 3 dB / 150 (default); obsessed: 2 dB / 300. Interactive terminals offer a choice")
    defaults = Settings()
    for field, kind in (("min_hz", float), ("max_hz", float), ("budget", int), ("shape_tolerance", float),
                        ("drift_tolerance", float), ("reference_level", float), ("max_level", float),
                        ("sample_rate", int), ("initial_points", int), ("max_points", int), ("seed", int)):
        create.add_argument("--" + field.replace("_", "-"), type=kind,
                            default=None if field in RANGES["full"] or field in ("shape_tolerance", "drift_tolerance", "reference_level", "max_level") else getattr(defaults, field),
                            help="Overrides the range or tolerance preset" if field in RANGES["full"] or field in ("shape_tolerance", "drift_tolerance") else None)
    create.add_argument("--bass-db", type=float, default=defaults.bass_db, help="Bass shelf relative to unmodified DF, -12 to +12 dB (default: 0)")
    bass = create.add_mutually_exclusive_group()
    bass.add_argument("--bass-hz", type=float, default=defaults.bass_hz, help="Custom bass shelf half-gain frequency, 40–300 Hz (default: 105)")
    bass.add_argument("--bass-style", choices=list(BASS_STYLES), help="sub-bass: 60 Hz, deep rumble; balanced: 105 Hz, weight/punch; warm: 180 Hz, body/warmth (descriptions assume boost)")
    create.add_argument("--noise-level", type=float, help=argparse.SUPPRESS)
    create.add_argument("--noise-ceiling", type=float, help=argparse.SUPPRESS)
    create.add_argument("--channels", choices=["together", "separate"], default=defaults.channels)
    create.add_argument("--mode", choices=["guided", "manual"], default=defaults.mode, help="Starting mode; switch any time during calibration")
    create.add_argument("--filter-mode", choices=["auto", "bells"], default=defaults.filter_mode,
                        help="PEQ fitting: auto selects useful shelves; bells uses only correction bells")
    create.add_argument("--sound-profile", choices=list(SOUND_PROFILES), default="neutral",
                        help="Post-calibration voicing; neutral adds no voicing (custom bass is separate)")
    create.add_argument("--device", help="PipeWire sink node name or serial; defaults to your current output")
    listing = commands.add_parser("list", help="Browse saved profiles in an arrow-key table")
    listing.add_argument("--plain", action="store_true", help="Print the profile table without an interactive menu")
    extend = commands.add_parser("extend", help="Extend a profile's upper range while preserving saved judgments")
    extend.add_argument("name")
    extend.add_argument("--max-hz", type=float, default=MAX_HZ, help="New upper frequency limit (default: 16000 Hz)")
    tolerance = commands.add_parser("tolerance", help="Choose a stopping tolerance and judgment cap for an existing profile; preserve progress")
    tolerance.add_argument("name")
    tolerance.add_argument("--preset", choices=list(TOLERANCE_PRESETS), required=True)
    commands.add_parser("devices", help="List available PipeWire output devices")
    run = commands.add_parser("run", help="Start or resume a calibration session (plays tones)")
    run.add_argument("name")
    run.add_argument("--extra-judgments", type=int, default=0, help="Increase the total per-channel judgment budget")
    run.add_argument("--noise-level", type=float, help=argparse.SUPPRESS)
    run.add_argument("--noise-ceiling", type=float, help=argparse.SUPPRESS)
    run.add_argument("--reference-level", type=float, help="Save a starting tone peak level (-60 to -9 dBFS); only changeable before any judgments")
    run.add_argument("--max-level", type=float, help="Save the starting tone peak ceiling (no higher than -3 dBFS); only changeable before any judgments")
    check = commands.add_parser("audio-check", help="Test a 1 kHz tone without recording calibration judgments")
    check.add_argument("name", help="Use this profile's playback configuration")
    check.add_argument("--device", help="Use a PipeWire sink name or serial for this check only")
    check.add_argument("--level", type=float, help="Override the check's tone peak in dBFS (-60 to -9, within the profile ceiling)")
    report = commands.add_parser("report", help="Show consistency and model-based correction estimates")
    report.add_argument("name")
    export = commands.add_parser("export", help="Fit and export a native Easy Effects PEQ preset")
    export.add_argument("name")
    export.add_argument("--output", type=Path, required=True, help="Preset JSON path; a separate .report.json is also saved")
    export.add_argument("--bands", type=int, default=12, help="Total bands including any bass shelf (default: 12)")
    export.add_argument("--filter-mode", choices=["auto", "bells"], help="Override this export's fitting mode without changing the profile")
    export.add_argument("--bass-db", type=float, help="Override this export's bass level without changing the session")
    bass = export.add_mutually_exclusive_group()
    bass.add_argument("--bass-hz", type=float, help="Override this export's bass shelf frequency")
    bass.add_argument("--bass-style", choices=list(BASS_STYLES), help="Shelf midpoint: sub-bass 60 Hz, balanced 105 Hz, warm 180 Hz")
    export.add_argument("--allow-incomplete", action="store_true", help="Export a connected estimate that has not met consistency thresholds")
    export.add_argument("--overwrite", action="store_true", help="Allow replacing the specified export files")
    demo = commands.add_parser("demo", help="Create and test a simulated profile without playing audio")
    demo.add_argument("--name", default="silent-demo")
    demo.add_argument("--output", type=Path)
    demo.add_argument("--range", choices=list(RANGES), default="treble")
    demo.add_argument("--bass-db", type=float, default=0.0)
    return command


def quality(channel, settings, reference):
    fit = fit_channel(channel, settings, reference)
    return {"judgments": channel["inputs"], "frequencies": len(fit.frequencies),
            "connected": fit.connected, "independent_cross_checks": fit.cycles,
            "shape_uncertainty_95_db": fit.shape95, "max_point_uncertainty_95_db": fit.max95,
            "cross_check_rms_db": fit.drift_rms, "level_boundary_pairs": fit.boundary_pairs,
            "consistency_thresholds_met": bool(fit.ready(settings)),
            "regional_pass": progress(channel, settings), "regional_checks_complete": verified(channel, settings),
            "calibration_complete": bool(ready(channel, settings, fit)),
            "regional_fit": fit.regional}


def describe_quality(channel, settings, reference):
    metrics = quality(channel, settings, reference)
    drift = "unavailable" if metrics["cross_check_rms_db"] is None else f"{metrics['cross_check_rms_db']:.2f} dB"
    status = "consistent" if metrics["calibration_complete"] else "incomplete"
    print(f"  {metrics['judgments']} judgments used; cap {settings.budget} per channel; {metrics['frequencies']} frequencies; {status}")
    skipped = (metrics["regional_fit"] or {}).get("skipped_regions", [])
    if skipped:
        print("  Shape uncalibrated: " + ", ".join(skipped) + "; no local correction applied there.")
        print(f"  Conditional balance uncertainty: {metrics['regional_fit']['balance_uncertainty_95_db']:.2f} dB; tolerance {settings.shape_tolerance:g} dB")
        print("  This checks regional gains assuming skipped shapes are unchanged; it does not verify those shapes.")
    else:
        print(f"  Estimated shape uncertainty: {metrics['shape_uncertainty_95_db']:.2f} dB (95% model interval); tolerance {settings.shape_tolerance:g} dB")
    print(f"  Cross-check disagreement: {drift}; tolerance {settings.drift_tolerance:g} dB; independent checks: {metrics['independent_cross_checks']}")
    print(f"  Pass: {metrics['regional_pass']}")
    if metrics["regional_fit"]:
        regional = metrics["regional_fit"]
        for name, info in regional["regions"].items():
            if info.get("skipped"):
                print(f"  {name.capitalize()} shape: uncalibrated (skipped); original sound shape retained.")
                continue
            state = "frozen" if info["frozen_at"] is not None else "still learning"
            print(f"  {name.capitalize()} shape: {state}; {info['shape_uncertainty_95_db']:.2f} dB estimated uncertainty")
        print("  Regional balance: " + ("measured links connected" if regional["balance_connected"] else "awaiting connected boundary comparisons"))
        if regional["review_regions"]:
            print("  Review needed: " + ", ".join(regional["review_regions"]) + ". Open Settings > Revisit regional shape.")
    if metrics["level_boundary_pairs"]:
        print("  Boundary-limited comparisons are excluded; alternative links determine coverage.")


PROFILE_HEADERS = ("Profile", "Status", "Range (Hz)", "Channels", "Answers / limit", "Bass (dB)", "Bass warmth (Hz)", "Tolerance (dB)")


def profile_summary(profile, reference=None):
    settings = Settings(**profile["settings"]).validate()
    reference = reference or Reference()
    fits = {name: fit_channel(channel, settings, reference) for name, channel in profile["channels"].items()}
    if not any(channel["inputs"] for channel in profile["channels"].values()):
        status = "new"
    elif all(ready(profile["channels"][name], settings, fit) for name, fit in fits.items()):
        status = "complete"
    elif any(not ready(channel, settings, fits[name]) and channel["inputs"] < settings.budget
             for name, channel in profile["channels"].items()):
        status = "unfinished"
    else:
        status = "incomplete"
    counts = profile["channels"]
    judgments = (f"{counts['both']['inputs']} / {settings.budget}" if settings.channels == "together" else
                 f"L {counts['left']['inputs']}/{settings.budget}; R {counts['right']['inputs']}/{settings.budget}")
    skipped = any((fit.regional or {}).get("skipped_regions") for fit in fits.values())
    display_status = status + " (shape skipped)" if skipped else status
    return {"status": status, "display_status": display_status,
            "connected": all(fit.connected or (fit.regional or {}).get("exportable", False) for fit in fits.values()),
            "row": (profile["name"], display_status, f"{settings.min_hz:g}-{settings.max_hz:g}",
                    "Both" if settings.channels == "together" else "Left / Right", judgments, f"{settings.bass_db:+g}", f"{settings.bass_hz:g}", f"{settings.shape_tolerance:g}")}


def profile_rows(store):
    rows = []
    reference = Reference()
    for name in store.names():
        try:
            rows.append(profile_summary(store.load(name), reference)["row"])
        except (ValueError, OSError, np.linalg.LinAlgError):
            rows.append((name, "unreadable", "-", "-", "-", "-", "-", "-"))
    return rows


def _menu_confirm(title, action):
    return select_table(title, ("Action",), [("Cancel",), (action,)]) == 1


def _menu_text(prompt):
    try:
        value = read_input(prompt)
        return None if value is None else value.strip()
    except EOFError:
        return None


def _menu_error(error, destination):
    print(f"Error: {error}", file=sys.stderr)
    _menu_text(f"Enter to return to {destination}: ")


def menu_export(profile, summary):
    if not summary["connected"]:
        raise ValueError("Export unavailable: comparisons do not connect all sampled frequencies or the chosen regional balance. Continue calibration first")
    incomplete = summary["status"] != "complete"
    if incomplete and not _menu_confirm("Consistency checks are unfinished; uncertainty will be retained.",
                                        "Export incomplete estimate"):
        return
    default = Path("exports") / (profile["name"] + ".json")
    entered = _menu_text(f"Easy Effects preset path [{default}] (q cancels): ")
    if entered is None or entered.lower() == "q":
        return
    output = Path(entered).expanduser() if entered else default
    overwrite = output.exists() or output.with_suffix(".report.json").exists()
    if overwrite and not _menu_confirm(f"Existing export: {output}", "Replace preset and report"):
        return
    export_profile(profile, output, count=12, allow_incomplete=incomplete, overwrite=overwrite)


def preview_profile(profile):
    """Show the same 12-band preset as menu export without writing files."""
    summary = profile_summary(profile)
    if not summary["connected"]:
        print("EQ preset preview unavailable: comparisons do not connect all sampled frequencies or the chosen regional balance.")
        print("Continue calibration before previewing PEQ filters.")
        _menu_text("Enter to return to the profile menu: ")
        return
    print("Calculating the EQ preview; fitting can take several seconds...")
    preset, report = build_preset(profile, count=12, allow_incomplete=True)
    state = "complete" if summary["status"] == "complete" else "provisional"
    equalizer = preset["output"]["equalizer#0"]
    rows = []
    for channel_name in report["channels"]:
        channel = "left" if channel_name == "both" else channel_name
        descriptions = report["channels"][channel_name]["peq_fit"]["bands"]
        for index in range(equalizer["num-bands"]):
            band = equalizer[channel][f"band{index}"]
            purpose = descriptions[index]["purpose"] if index < len(descriptions) else "Unused"
            rows.append((channel_name.capitalize(), str(index + 1), band["type"],
                         f"{band['frequency']:.1f}", f"{band['gain']:+.2f}", f"{band['q']:.3f}", purpose))
    print(f"EQ preset preview: {profile['name']} | {state}")
    print(f"Sound profile: {SOUND_PROFILES[report['settings']['sound_profile']][0]} (post-calibration preference)")
    for name, channel in report["channels"].items():
        if channel["skipped_shapes_uncalibrated"]:
            print(f"{name.capitalize()}: shape uncalibrated in " + ", ".join(channel["skipped_shapes_uncalibrated"]) + "; regional gain only.")
    if state == "provisional":
        print("Consistency thresholds or regional checks are unfinished; these filters may change.")
    print(report["interpretation"])
    shape_note = " | shape uncalibrated (skipped)" if any(c["skipped_shapes_uncalibrated"] for c in report["channels"].values()) else ""
    select_table(f"EQ preview | {state}{shape_note} | preamp {report['input_gain_db']:.2f} dB | {profile['name']}",
                 ("Channel", "Filter", "Type", "Frequency (Hz)", "Gain (dB)", "Q", "Purpose"), rows)


def menu_filter_mode(store, profile):
    settings = Settings(**profile["settings"])
    modes = ("auto", "bells")
    rows = [("Auto", "Choose peaks, dips and broad shelves to fit your correction"),
            ("Bells only", "Use peaks and dips; keep your separate bass preference")]
    selected = select_table("EQ filter style | affects preview and export; keeps calibration",
                            ("Mode", "Details"), rows, modes.index(settings.filter_mode))
    if selected is not None:
        profile["settings"] = asdict(replace(settings, filter_mode=modes[selected]).validate())
        store.save(profile)


def choose_bass_shelf(settings):
    labels = {"sub-bass": "Deep rumble", "balanced": "Balanced bass", "warm": "More warmth"}
    choices = [(labels[name], f"{hz:g}", description) for name, (hz, description) in BASS_STYLES.items()]
    initial = next((index for index, (_, hz, _) in enumerate(choices) if float(hz) == settings.bass_hz), len(choices))
    choices.append(("Manual input", "40-300", "Choose your own shelf transition frequency"))
    selected = select_table(f"Bass warmth | {settings.bass_db:+g} dB | descriptions apply when bass is boosted",
                            ("Style", "Midpoint (Hz)", "Sound"), choices, initial, description_column=-1)
    if selected is None:
        return None
    if selected == len(BASS_STYLES):
        while True:
            entered = _menu_text(f"Bass shelf midpoint in Hz, 40-300 [{settings.bass_hz:g}] (q cancels): ")
            if entered is None or not entered or entered.lower() == "q":
                return None
            try:
                settings = replace(settings, bass_hz=float(entered)).validate()
                break
            except ValueError:
                print("Enter a finite frequency from 40 to 300 Hz, or q to cancel.")
    else:
        settings = replace(settings, bass_hz=float(choices[selected][1])).validate()
    return settings


def menu_bass_shelf(store, profile):
    settings = choose_bass_shelf(Settings(**profile["settings"]))
    if settings is None:
        return
    profile["settings"] = asdict(settings)
    store.save(profile)


def choose_tolerance(settings=None):
    """Arrow-key choice shared by creation and existing-profile actions."""
    names = list(TOLERANCE_PRESETS)
    initial = names.index("convinced")
    if settings is not None:
        initial = next((index for index, (shape, drift, _, _) in enumerate(TOLERANCE_PRESETS.values())
                        if (shape, drift) == (settings.shape_tolerance, settings.drift_tolerance)), initial)
    labels = {"doubt": "Casual", "convinced": "Focused", "obsessed": "Meticulous"}
    rows = [(labels[name], f"{shape:g}", str(budget), description)
            for name, (shape, _, budget, description) in TOLERANCE_PRESETS.items()]
    selected = select_table("Listening effort | may finish before the limit; estimates listening consistency",
                            ("Choice", "Tolerance (dB)", "Maximum / channel", "For you"), rows, initial, description_column=-1)
    return None if selected is None else names[selected]


def menu_tolerance(store, profile):
    preset = choose_tolerance(Settings(**profile["settings"]))
    if preset is not None:
        store.set_tolerance(profile["name"], preset)


def menu_profile_settings(store, name):
    """Keep less frequent choices together and return to this menu after edits."""
    from .ui import audio_menu, edit_bass_sound

    selected = 0
    while True:
        profile = store.load(name)
        settings = Settings(**profile["settings"])
        actions = [("Audio and volume", "Choose output, check sound and edit playback levels"),
                   ("Listening effort", f"{settings.shape_tolerance:g} dB tolerance; at most {settings.budget} answers/channel"),
                   ("Bass sound", f"{settings.bass_db:+g} dB; warmth at {settings.bass_hz:g} Hz"),
                   ("EQ filter style", "Auto: peaks, dips and shelves" if settings.filter_mode == "auto" else "Bells only: peaks and dips")]
        actions.append(("Sound profile", SOUND_PROFILES[settings.sound_profile][0] + "; applied after calibration"))
        if settings.calibration_method == "regional-v2":
            actions.append(("Revisit regional shape", "Reopen a frozen shape; retain answers and repeat balance"))
        if 12000 <= settings.max_hz < MAX_HZ:
            actions.append(("Extend to 16000 Hz", "Include higher treble; keeps your saved answers"))
        actions.append(("Back", "Return to this profile"))
        selected = select_table(f"{name} > Settings", ("Action", "Details"), actions, selected)
        if selected is None or actions[selected][0] == "Back":
            return
        try:
            action = actions[selected][0]
            if action == "Audio and volume":
                audio_menu(store, profile)
            elif action == "Listening effort":
                menu_tolerance(store, profile)
            elif action == "Bass sound":
                changed = edit_bass_sound(settings)
                if changed != settings:
                    profile["settings"] = asdict(changed)
                    store.save(profile)
            elif action == "EQ filter style":
                menu_filter_mode(store, profile)
            elif action == "Sound profile":
                menu_sound_profile(store, profile)
            elif action == "Revisit regional shape":
                menu_revisit_region(store, profile)
            else:
                store.extend(name)
        except (ValueError, OSError, subprocess.TimeoutExpired, np.linalg.LinAlgError) as error:
            _menu_error(error, "Settings")


def menu_sound_profile(store, profile, after_calibration=False):
    settings = Settings(**profile["settings"])
    names = list(SOUND_PROFILES)
    rows = [(label, description) for label, description, _ in SOUND_PROFILES.values()]
    title = ("Step 4/4 | Choose your sound" if after_calibration else f"{profile['name']} > Sound profile")
    title += f" | extra bass {settings.bass_db:+g} dB"
    if any(c.get("workflow", {}).get("skipped") for c in profile["channels"].values()):
        title += " | shape skipped"
        rows[0] = ("Neutral/reference", "No added voicing; skipped shapes remain uncalibrated")
    print("Neutral/reference adds no voicing to the DF estimate. Your separate bass setting still applies.")
    print("Preferences change preview/export only; calibration evidence and regional shapes are retained.")
    selected = select_table(title, ("Sound", "What it adds"), rows,
                            names.index(settings.sound_profile), description_column=-1)
    if selected is not None and names[selected] != settings.sound_profile:
        changed = replace(settings, sound_profile=names[selected]).validate()
        updated = dict(profile, settings=asdict(changed))
        store.save(updated)
        profile["settings"] = updated["settings"]


def menu_revisit_region(store, profile):
    from .regional import reopen_region

    settings = Settings(**profile["settings"])
    choices = [(ear, name) for ear, channel in profile["channels"].items()
               for name in channel.get("workflow", {}).get("frozen", {})]
    if not choices:
        print("No shapes have been frozen yet. Continue calibration to complete the shape passes.")
        _menu_text("Enter to return to Settings: ")
        return
    selected = select_table("Revisit a regional shape | saved answers are kept",
                            ("Channel", "Region", "Status"),
                            [(ear.capitalize(), name.capitalize(),
                              "Review flagged" if name in profile["channels"][ear].get("workflow", {}).get("review_regions", []) else
                              "Shape uncalibrated (skipped)" if name in profile["channels"][ear].get("workflow", {}).get("skipped", {}) else "Frozen")
                             for ear, name in choices])
    if selected is not None:
        ear, name = choices[selected]
        updated = copy.deepcopy(profile)
        reopen_region(updated["channels"][ear], settings, name)
        store.save(updated)


def menu_manage_profile(store, name):
    selected = 0
    actions = [("Reset", "Start over with this name and settings; clears calibration"),
               ("Delete", "Remove this profile and its calibration"),
               ("Back", "Return to this profile")]
    while True:
        selected = select_table(f"{name} > Manage profile | exported files are kept",
                                ("Action", "Details"), actions, selected)
        if selected is None or actions[selected][0] == "Back":
            return None
        try:
            if actions[selected][0] == "Reset":
                if _menu_confirm(f"Reset {name}? Clear calibration; keep settings and exports.", "Reset profile"):
                    store.reset(name)
                    return "reset"
            elif _menu_confirm(f"Delete {name} and all saved judgments? Exports are kept.", "Delete profile"):
                store.delete(name)
                return "delete"
        except (ValueError, OSError) as error:
            _menu_error(error, "Manage profile")


def browse_profiles(store, open_name=None):
    """Browse profiles, optionally opening a newly created profile first."""
    selected_name = open_name
    while True:
        if open_name is not None:
            selected_name = open_name
            open_name = None
        else:
            rows = profile_rows(store)
            if not rows:
                print("No saved profiles. Choose New profile from Home.")
                return
            initial = next((index for index, row in enumerate(rows) if row[0] == selected_name), 0)
            selected = select_table("Device profiles", PROFILE_HEADERS, rows, initial)
            if selected is None:
                return
            selected_name = rows[selected][0]
        selected_action = None
        while True:
            action = None
            try:
                profile = store.load(selected_name)
                summary = profile_summary(profile)
                status = summary["status"]
                actions = []
                if status != "complete":
                    actions.append(("Start calibration" if status == "new" else "Continue calibration",
                                    "Add more answers to finish" if status == "incomplete" else "Listen to tones; progress saves automatically"))
                actions.extend([("Progress", "See completed checks and what still needs work"),
                                ("Preview EQ", "View your filters before exporting"),
                                ("Export to Easy Effects", "Save an EQ preset to use in Easy Effects"),
                                ("Settings", "Audio, listening effort, bass and EQ filter style"),
                                ("Manage profile", "Reset calibration or delete this profile")])
                actions.append(("Back", "Return to profiles"))
                settings = profile["settings"]
                title = (f"{selected_name} | {summary['display_status']} | {settings['min_hz']:g}-{settings['max_hz']:g} Hz | "
                         f"bass {settings.get('bass_db', 0):+g} dB @ {settings.get('bass_hz', 105):g} Hz")
                initial = next((index for index, row in enumerate(actions) if row[0] == selected_action), 0)
                action_index = select_table(title, ("Action", "Details"), actions, initial)
                if action_index is None or actions[action_index][0] == "Back":
                    break
                action = actions[action_index][0]
                selected_action = action
                if action == "Manage profile":
                    outcome = menu_manage_profile(store, selected_name)
                    if outcome == "delete":
                        break
                    if outcome == "reset":
                        selected_action = "Start calibration"
                elif action == "Export to Easy Effects":
                    menu_export(profile, summary)
                    _menu_text("Enter to return to the profile menu: ")
                elif action == "Progress":
                    report_profile(profile)
                    _menu_text("Enter to return to the profile menu: ")
                elif action == "Preview EQ":
                    preview_profile(profile)
                elif action == "Settings":
                    menu_profile_settings(store, selected_name)
                else:
                    extra = 0
                    if status == "incomplete":
                        entered = _menu_text("Raise the answer limit by how many per channel? [120] (q cancels): ")
                        if entered is None or entered.lower() == "q":
                            continue
                        try:
                            extra = int(entered or "120")
                        except ValueError as error:
                            raise ValueError("Additional budget must be a positive integer") from error
                        if extra <= 0:
                            raise ValueError("Additional budget must be a positive integer")
                    run_profile(store, profile, extra)
                    _menu_text("Enter to return to the profile menu: ")
            except (ValueError, OSError, subprocess.TimeoutExpired, np.linalg.LinAlgError) as error:
                _menu_error(error, "the profile menu" if action is not None else "profiles")
                if action is None:
                    break


def list_profiles(store, plain=False):
    if not plain and sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM", "dumb") != "dumb":
        browse_profiles(store)
        return
    rows = profile_rows(store)
    if rows:
        print("\n".join(table_lines(PROFILE_HEADERS, rows)))
        print("Answers / limit counts saved comparisons per channel; previews and replays do not count. You may finish before the limit.")
        print("new: no judgments | unfinished: in progress | incomplete: budget exhausted | complete: consistency checks met")
    else:
        print("No saved profiles. Create one with the new command.")


def report_profile(profile):
    settings = Settings(**profile["settings"]).validate()
    reference = Reference()
    print(f"Sound profile: {SOUND_PROFILES[settings.sound_profile][0]}; applied after the reference calibration")
    print(f"Profile: {profile['name']} | {settings.min_hz:g}–{settings.max_hz:g} Hz | {settings.sample_rate} Hz sample rate")
    print("Target: B&K 5128 DF; fixed middle-ear sensitivity approximation.")
    print(f"Stopping tolerance: shape {settings.shape_tolerance:g} dB; cross-check disagreement {settings.drift_tolerance:g} dB; cap {settings.budget} judgments per channel.")
    print(f"Bass preference: {settings.bass_db:+g} dB shelf, half gain at {settings.bass_hz:g} Hz (0 dB = unmodified DF).")
    print("Consistency intervals exclude hearing-model error and unsampled features.")
    for channel_name, channel in profile["channels"].items():
        print(f"\n{channel_name}:")
        describe_quality(channel, settings, reference)
        fit = fit_channel(channel, settings, reference)
        if not fit.connected:
            print("  Coverage is disconnected; no correction estimate can be exported yet.")
            continue
        correction = reference.correction(fit.frequencies, fit.matches)
        correction += response(fit.frequencies, [bass_shelf(settings.bass_db, settings.bass_hz)], settings.sample_rate)
        print("     Hz     matched level*   correction*    uncertainty")
        for hz, matched, gain, variance in zip(fit.frequencies, fit.matches, correction, np.diag(fit.covariance)):
            print(f"  {hz:7.0f}    {matched:+7.2f} dB    {gain:+7.2f} dB    ±{1.96 * np.sqrt(max(0, variance)):.2f} dB")
    print("\n* Shape relative to overall level; correction is model-based, not an acoustic measurement.")
    if settings.max_hz > 12500:
        print("The hearing proxy above 12.5 kHz uses unvalidated extrapolated source data.")
    if settings.min_hz < 2000:
        print("Bass/midrange use the same linear proxy; level-dependent loudness is not modeled.")


def _input(prompt):
    value = _menu_text(prompt)
    return "q" if value is None else value.lower()


def _calibration_help(topic, settings):
    music = preview_info()
    band = music["tone_reference"]
    music_peak = music_reference_levels()["reference_level"]
    tone_guidance = (("Both tones start from the matching curve, anchored to the saved 1 kHz level.",
                      "The fixed hearing estimate starts the curve; saved answers refine it.",
                      "At a playback limit, the pair shifts together; the difference is preserved.",
                      "Manual controls change the difference; a limit may also move the other tone.",
                      "Starting gains are accounted for in the fit, not added to exported EQ.")
                     if settings.tone_leveling == "adaptive-v1" else
                     ("This profile retains its original fixed-reference playback.",
                      "Tone A stays at the saved level; tone B changes by the comparison offset.",
                      "Create a new profile to use frequency-dependent starting levels."))
    passages = (("Shape passes: learn midrange, bass and treble separately, then freeze each shape.",
                 "Regional balance: nearby cross-boundary tones adjust only regional gains.",
                 "Verification: repeat checks without reshaping frozen regions.",
                 "Finish one range's checks before moving on; boundary balance has its own blocks.",
                 "Conflicts require a regional review; saved answers are kept.",
                 "Choose a sound preference after calibration, or later in Settings.")
                if settings.calibration_method == "regional-v2" else
                ("Early boundary checks: establish links between frequency regions.",
                 "Midrange / bass / treble shape: compare within each available region.",
                 "Regional balance: cross-check links between those regions.",
                 "Grouped verification: finish each range, then refine the shared curve."))
    guidance = {
        "volume": (
            ("What to do", ("Play the built-in music excerpt and set hardware/system volume to your usual level.",
                            "While music plays, Enter, Esc or s stops it and returns to volume setup.",
                            "Lower volume before starting. Bypass EQ for the calibration tones.",
                            "Keep headphone position and seal fixed; use a quiet environment.")),
            ("What the preview means", (f"The same {MUSIC_DURATION_SECONDS}-second excerpt plays from the beginning on every replay.",
                                         "About -14 LUFS, assuming music-player volume at 100%.",
                                         "A fixed reference; your recordings may sound louder or quieter.",
                                         "The program does not change system volume.")),
            ("Music credit", (f"{music['title']} by {music['artist']} (incompetech.com).",
                              f"Creative Commons Attribution 3.0: {music['license_url']}",
                              "Excerpted, faded and loudness-normalized once with peak limiting.",
                              "Complete attribution and preparation details: data/LICENSE-Music.txt."))),
        "reference": (
            ("What to check", ("A short 1 kHz tone should be clearly audible and comfortable.",
                               "Keep system volume fixed after setting it with the music excerpt.",
                               "Use the starting level unless you need an adjustment; this is not a hearing-threshold test.")),
            ("Music-based starting level", (f"New profiles start at {music_peak:g} dBFS peak from the excerpt's 935–1065 Hz band.",
                                           f"Band energy: {band['band_rms_dbfs']:.1f} dBFS RMS, averaged over both channels and the full excerpt.",
                                           "Equivalent steady sine energy; playback fades reduce its average slightly.",
                                           "This is not an equal-loudness or acoustic-level measurement.",
                                           "Existing profiles and manual adjustments keep their selected levels.")),
            ("How to finish", ("Enter/c confirms that this level is usable and begins calibration.",
                               "r replays; h opens help; q cancels draft changes.",
                               "No noise match or equal-loudness judgments are required in this step.")),
            ("Level controls", ("+/- changes 1 dB; ++/-- changes 3 dB before any saved judgments.",
                                "Changes replay the tone and must be confirmed before saving.",
                                "Saved judgments lock the reference level on every resume."))),
        "comparison": (
            ("Listening controls", ("Guided: 1 first louder, 2 second louder, e equal, u unsure.",
                                    "UI: Up/Down adjusts the marked tone by 0.5 dB and selects manual mode.",
                                    "Enter plays the pair after changes; e saves a match after listening.",
                                    "Sliders show digital level changes since this pair began, not acoustic loudness.",
                                    "Manual: +/- changes 0.5 dB; ++/-- changes 2 dB; e accepts.",
                                    "m/g switches to manual/guided without recording an answer.",
                                    "r replays; s skips an inaudible or unclear pair; q saves and returns.")),
            ("Tone levels", tone_guidance),
            ("Passages", passages),
            ("Local calibration", (("k: skip the active region's shape; keep only its regional gain adjustment.",
                                    "Skipped shapes stay uncalibrated. Revisit them later in Settings.")
                                   if settings.calibration_method == "regional-v2" else
                                   ("This legacy profile keeps its original local calibration workflow.",))),
            ("Progress", ("Saved answers / limit is a maximum, not a required test count.",
                          "Replay, help and unsure responses do not count as answers.",
                          "Keep system volume fixed. Skip inaudible bass rather than guessing."))),
    }
    levels = ("Digital levels", (f"1 kHz reference: {settings.reference_level:g} dBFS peak; ceiling: {settings.max_level:g} dBFS.",
                                f"Comparison adjustment allowance: +/-{settings.offset_limit:g} dB.",
                                f"Music excerpt: {music['measured_lufs']:g} LUFS, {music['measured_true_peak_dbfs']:g} dBTP; fixed.",
                                "dBFS is digital signal level, not loudness at your ears."))
    with preserve_screen():
        show_screen(f"Calibration help | {topic}", (*guidance[topic], levels), status=())
        _input("Enter or Esc to return to the current step > ")


def check_audio(profile, device=None, level=None):
    settings = Settings(**profile["settings"]).validate()
    # Check overrides never rewrite a profile's settings or observations.
    settings = replace(settings, device=settings.device if device is None else device,
                       reference_level=settings.reference_level if level is None else level).validate()
    sinks = devices()
    if settings.device:
        matches = [sink for sink in sinks if str(sink.get("object.serial")) == settings.device
                   or sink.get("node.name") == settings.device]
        if not matches:
            raise ValueError("Selected output is not an available PipeWire sink. Review Output devices and Playback settings")
        selected = matches[0]
        print(f"Output: {selected.get('node.description', selected.get('node.name'))}")
    else:
        print("Output: current system default (choose a particular output in Playback settings).")
    print(f"Audio check for {profile['name']}: 1000 Hz, 1 second, both channels, {settings.reference_level:g} dBFS peak.")
    print("Start at a comfortable hardware volume. This check does not change system volume or saved calibration.")
    player = Player(settings)
    while True:
        answer = _input("Enter or r: play/replay the check tone | q: quit > ")
        if answer in ("q", "quit"):
            print("Audio check ended; no calibration judgments recorded.")
            return
        if answer in ("", "r", "replay"):
            player.play_tone()
            print("Player completed. If silent, check output routing and mute settings; completion does not prove audibility.")
        else:
            print("Enter plays the tone; q returns without playing audio.")


def compare(a, b, channel_name, channel, settings, player, reference, rng, mode, order=None):
    fit = fit_channel(channel, settings, reference)
    indices = {float(hz): i for i, hz in enumerate(fit.frequencies)}
    pair = pairs_for(channel, settings).get((a, b))
    if pair is None:
        pair = Pair(a, b, [], settings.offset_limit)
        warm = fit.matches[indices[b]] - fit.matches[indices[a]]
        offset = float(np.clip(warm + rng.uniform(-1.5, 1.5), -settings.offset_limit, settings.offset_limit))
    else:
        offset = pair.next_offset() if mode == "guided" else float(np.clip(pair.mean + rng.uniform(-1.5, 1.5), -settings.offset_limit, settings.offset_limit))
    # Freeze the prediction during this prompt. Replays and help cannot learn
    # from unaccepted answers or change the pair's operating level.
    base_level = (reference_base_level(a, fit, settings, reference)
                  if settings.tone_leveling == "adaptive-v1" else None)
    reversed_order = bool(rng.integers(0, 2)) if order is None else order
    first_hz, second_hz = (b, a) if reversed_order else (a, b)
    interactive = comparison_ui()
    starting_offset = offset
    starting_levels = comparison_levels(offset, settings, base_level)
    heard_offset = None
    played = False
    notice = ""
    local_stage = (channel.get("workflow", {}).get("stage")
                   if settings.calibration_method == "regional-v2" else None)
    can_skip_region = local_stage in ("bass", "midrange", "treble")
    def accepted(kind, response):
        result = (offset, kind, response)
        if base_level is not None:
            a_level, b_level = comparison_levels(offset, settings, base_level)
            result += ({"method": settings.tone_leveling, "a_dbfs": a_level, "b_dbfs": b_level},)
        return result, mode

    while True:
        playback_info = ()
        if base_level is not None:
            a_level, b_level = comparison_levels(offset, settings, base_level)
            first_level, second_level = (b_level, a_level) if reversed_order else (a_level, b_level)
            playback_info = (f"Levels follow your matching curve: {first_level:.1f} / {second_level:.1f} dBFS peak.",)
            if abs(a_level - base_level) > 1e-6:
                playback_info += ("Pair shifted together to stay within playback limits.",)
        if mode == "manual":
            goal = "Adjust the indicated tone until both sounds are equally loud."
            next_action = ("+ / -: change 0.5 dB | ++ / --: change 2 dB | e: accept equal",)
            level = (f"Adjustable tone: {b:.0f} Hz ({'first' if reversed_order else 'second'}), offset {offset:+.1f} dB",)
        else:
            goal = "Listen to both tones, then choose which sounded louder."
            next_action = ("1: first louder | 2: second louder | e: equal | u: unsure",)
            level = ()
        sections = [("Goal", (goal,)),
                    ("Playing", (f"First: {first_hz:.0f} Hz | Second: {second_hz:.0f} Hz",
                                 f"Frequency gap: {np.log2(b / a):.2f} octaves (maximum 1 octave).", *playback_info, *level)),
                    ("Next action", next_action),
                    ("Other controls", ("r: replay | s: skip pair | h: help | q: save and return",
                                        *(("k: skip this region's local calibration",) if can_skip_region else ())))]
        if notice:
            sections.append(("Feedback", (notice,)))
        if interactive:
            peaks = comparison_levels(offset, settings, base_level)
            changes = tuple(current - start for current, start in zip(peaks, starting_levels))
            levels_in_order = peaks[::-1] if reversed_order else peaks
            changes_in_order = changes[::-1] if reversed_order else changes
            sliders = tone_slider_lines((first_hz, second_hz), levels_in_order,
                                        changes_in_order, 0 if reversed_order else 1)
            adjustment = f"Relative adjustment: {offset - starting_offset:+.1f} dB"
            sliders[-1] = adjustment + " | Changes since start; rails auto-scale."
            if notice:
                sliders[-1] = notice
            elif played and heard_offset is not None and abs(offset - heard_offset) > 1e-9:
                sliders[-1] = adjustment + " | Changes pending: Enter to hear them."
            elif base_level is not None and abs(peaks[0] - base_level) > 1e-6:
                sliders[-1] = adjustment + " | Pair shifted together within limits."
            controls = ("Manual: e accepts equal loudness after listening." if mode == "manual" else
                        "Guided: 1 first louder | 2 second louder | e equal | u unsure")
            other = "r: replay | s: skip pair | h: help | q: save and return"
            if can_skip_region:
                other = "r replay | s skip pair | h help | q return | k skip region"
            sections = [(f"Tones: first, then second | Gap {np.log2(b / a):.2f} octaves", sliders),
                        ("Next action", (controls,)), ("Other controls", (other,))]
        show_screen(None, sections, comparison=interactive)
        if not played:
            if base_level is None:
                player.play(a, b, offset, reversed_order, channel_name)
            else:
                player.play(a, b, offset, reversed_order, channel_name, base_level=base_level)
            played = True
            heard_offset = offset
        answer = _input("Your choice > ")
        notice = ""
        if answer in ("h", "help", "?"):
            _calibration_help("comparison", settings)
            continue
        if answer == "k" and can_skip_region:
            if _menu_confirm(f"Skip {local_stage} shape? Keep its original sound; balance its level only.", "Skip this region"):
                return "skip-region", mode
            continue
        if answer in ("up", "down"):
            # A user-selected offset is a manual match, not an adaptive
            # guided probe. Wait for explicit replay before accepting it.
            mode = "manual"
            updated = float(np.clip(offset + (0.5 if answer == "up" else -0.5),
                                    -settings.offset_limit, settings.offset_limit))
            if updated == offset:
                notice = "Level limit reached. Skip if the tones cannot be matched."
            offset = updated
            continue
        if interactive and answer in ("1", "2", "e", "equal", "=") and abs(offset - heard_offset) > 1e-9:
            notice = "Press Enter to hear the changed pair before recording an answer."
            continue
        if mode == "manual":
            if answer in ("+", "-", "++", "--"):
                change = {"+": 0.5, "-": -0.5, "++": 2.0, "--": -2.0}[answer]
                updated = float(np.clip(offset + change, -settings.offset_limit, settings.offset_limit))
                if updated == offset:
                    notice = "Level limit reached. Skip if the tones cannot be matched."
                offset = updated
                if not interactive:
                    played = False
                continue
            if answer in ("e", "equal", "="):
                return accepted("manual", "equal")
        else:
            if answer in ("1", "2", "e", "equal", "="):
                response = "equal" if answer in ("e", "equal", "=") else ("first" if answer == "1" else "second")
                if reversed_order and response != "equal":
                    response = "second" if response == "first" else "first"
                return accepted("guided", response)
            if answer in ("u", "unsure"):
                notice = "No answer recorded. Replay or skip if this comparison is unclear."
                continue
        if answer in ("q", "quit"):
            return "quit", mode
        if answer in ("s", "skip"):
            return "skip", mode
        if answer in ("", "r", "replay"):
            played = False
        elif answer in ("m", "g"):
            mode = "manual" if answer == "m" else "guided"
        else:
            notice = "Use one of the displayed commands, or h for help."


def prepare_reference(player, settings, resuming=False, profile_name=None):
    """Confirm an audible, comfortable reference; save only on acceptance."""
    steps = 4 if settings.calibration_method == "regional-v2" else 3
    title = f"Step 2/{steps} | Check the reference tone" + (f" | {profile_name}" if profile_name else "")
    selected = settings
    allowance = settings.offset_limit
    lower, upper = max(-60.0, -78.0 + allowance), REFERENCE_MAX
    played, notice = False, ""

    def reference_screen():
        sections = [("Goal", ("Keep hardware/system volume at the level set with the music.",
                              "The tone should be clearly audible and comfortable, not barely detectable.")),
                    ("Playing", ("Short 1 kHz tone | Both channels",
                                 f"Music-based default: {music_reference_levels()['reference_level']:g} dBFS peak; saved levels are kept.")),
                    ("Next action", ("Enter/c: confirm this tone is clearly audible and comfortable.",
                                     "Only confirm if you can judge it reliably; q returns.")),
                    ("Other controls", (("r: replay | h: help | q: return" if resuming else
                                         "+/-: adjust 1 dB | ++/--: adjust 3 dB | r: replay | h: help | q: cancel"),))]
        if resuming:
            sections.insert(1, ("Saved reference", ("Saved judgments lock the tone level; adjustments are unavailable.",
                                                   "Restore earlier system volume; use a new profile if unusable.")))
        if notice:
            sections.append(("Feedback", (notice,)))
        show_screen(title, sections, status=(f"Tone level: {selected.reference_level:g} dBFS peak | " +
                                            ("Saved level locked" if resuming else "Adjust only if needed"),))

    try:
        while True:
            reference_screen()
            if not played:
                player.settings = selected
                player.play_tone()
                played = True
            answer = _input("Audible and comfortable? Enter/c to confirm > ")
            notice = ""
            if answer in ("q", "quit"):
                return None
            if answer in ("", "c", "continue"):
                return selected
            if answer in ("r", "t", "replay"):
                played = False
            elif answer in ("h", "help", "?"):
                _calibration_help("reference", selected)
            elif answer in ("+", "-", "++", "--"):
                if resuming:
                    notice = "Level changes are unavailable after saved judgments; use r, Enter/c, or q."
                    continue
                delta = {"+": 1.0, "-": -1.0, "++": 3.0, "--": -3.0}[answer]
                level = float(np.clip(selected.reference_level + delta, lower, upper))
                if level == selected.reference_level:
                    notice = "Tone adjustment limit reached. Return if this level is not usable."
                    continue
                selected = replace(selected, reference_level=level,
                                   max_level=min(TONE_CEILING_MAX, level + allowance)).validate()
                played = False
            else:
                notice = "Confirm with Enter/c only when the tone is clearly audible and comfortable."
    finally:
        # The caller commits confirmed settings. Cancel/errors discard all drafts.
        player.settings = settings


def prepare_volume(player, settings, resuming=False, profile_name=None):
    steps = 4 if settings.calibration_method == "regional-v2" else 3
    title = f"Step 1/{steps} | Set music volume" + (f" | {profile_name}" if profile_name else "")
    music = preview_info()

    def volume_screen(played=False, notice="", playing=False):
        goal = ("Restore the same volume used for your earlier judgments." if resuming else
                "Set hardware/system volume to your usual comfortable music level.")
        sections = [("Goal", (goal, "Keep headphone position fixed; bypass EQ for the test tones.")),
                    ("Music preview", (f"{music['title']} by {music['artist']} | {MUSIC_DURATION_SECONDS} seconds",
                                       "About -14 LUFS, with music-player volume assumed at 100%.",
                                       "A fixed reference; your recordings may be louder or quieter.")),
                    ("Next action", (("Enter, Esc or s: stop the preview and return to volume setup" if playing else
                                      "Enter/c: keep this volume and check the reference tone" if played else
                                      "Lower hardware/system volume before playing. Enter plays the music excerpt."),)),
                    ("Other controls", (("Stop keys work immediately; no Enter needed after s or Esc." if playing else
                                         "r: replay | t: preview tone | h: help | q: return" if played else
                                         "h: help | q: return without playing"),))]
        if notice:
            sections.append(("Feedback", (notice,)))
        path = "Volume > Reference > Shapes and balance > Sound profile" if steps == 4 else "Step 1: Volume  >  Step 2: Reference  >  Step 3: Calibration"
        show_screen(title, sections, status=(path,))

    volume_screen()
    while True:
        answer = _input(f"Enter to play {MUSIC_DURATION_SECONDS} seconds of music > ")
        if answer in ("q", "quit"):
            return None
        if answer in ("h", "help", "?"):
            _calibration_help("volume", settings)
            volume_screen()
            continue
        if answer in ("", "y", "yes", "r", "replay"):
            break
        volume_screen(notice="Enter plays the volume preview; q returns without playing audio.")
    while True:
        volume_screen(played=True, playing=True)
        with music_preview_controls() as cancel:
            completed = player.play_music() if cancel is None else player.play_music(cancel=cancel)
        volume_screen(played=True, notice="Music preview stopped early. Replay or continue when ready." if completed is False else
                      "Music preview finished. Replay or continue when ready.")
        while True:
            answer = _input("Replay, preview a tone, or continue > ")
            if answer in ("q", "quit"):
                return None
            if answer in ("", "c", "continue"):
                return prepare_reference(player, settings, resuming=resuming, profile_name=profile_name)
            if answer in ("r", "replay"):
                break
            if answer in ("h", "help", "?"):
                _calibration_help("volume", settings)
                volume_screen(played=True)
                continue
            if answer in ("t", "tone"):
                player.play_tone()
                volume_screen(played=True, notice="Starting tone previewed. Step 2 checks that it is clearly audible and comfortable.")
                continue
            volume_screen(played=True, notice="Use r to adjust volume, t to preview a tone, or Enter to continue.")


def _session_summary(profile, message="Session saved. You can resume later."):
    summary = profile_summary(profile)
    settings = Settings(**profile["settings"])
    counts = tuple(f"{name.capitalize()}: {channel['inputs']} / {settings.budget} saved answers (maximum)"
                   for name, channel in profile["channels"].items())
    next_action = ("Preview EQ or export from the profile menu." if summary["status"] == "complete" else
                   "Choose Continue calibration to add more answers." if summary["status"] != "new" else
                   "Choose Start calibration when you are ready.")
    if any(channel.get("workflow", {}).get("stage") == "review" for channel in profile["channels"].values()):
        next_action = "Open Settings > Revisit regional shape; saved answers are kept."
    show_screen(f"Calibration summary | {profile['name']}",
                (("Result", (message,)), ("Saved progress", counts),
                 ("Next action", (next_action, "Open Progress in the profile menu for detailed results."))),
                status=(f"Profile status: {summary['display_status']}",))


def choose_local_calibration(store, profile, channel_name, settings):
    """Choose full shape probing or balance-only before a channel starts."""
    channel = profile["channels"][channel_name]
    from .regional import skip_shapes
    from .workflow import regions

    rows = [("Calibrate local shapes", "Learn each region, then balance their levels"),
            ("Skip all local passes", "Keep original shapes; measure regional balance only"),
            ("Save and return", "Decide later; no answers are added")]
    selected = select_table(f"Step 3/4 | Local calibration | {channel_name.capitalize()}",
                            ("Option", "What happens"), rows, description_column=-1)
    if selected is None or selected == 2:
        return False
    if selected == 1:
        if len(regions(channel)) < 2:
            raise ValueError("This range has only one region. There is no regional balance to perform; keep local calibration or use a wider range")
        skip_shapes(channel, settings)
    else:
        # Persist the decision even before the first accepted judgment so
        # resuming does not ask again. Comparison planning stays transactional.
        first = next(n for n in ("midrange", "bass", "treble") if n in regions(channel))
        channel["workflow"] = {"version": 2, "stage": first, "started_at": channel["inputs"], "frozen": {}}
    store.save(profile)
    return True


def run_profile(store, profile, extra, noise_level=None, noise_ceiling=None, reference_level=None, max_level=None):
    saved_settings = Settings(**profile["settings"]).validate()
    settings = replace(saved_settings,
                       reference_level=saved_settings.reference_level if reference_level is None else reference_level,
                       max_level=saved_settings.max_level if max_level is None else max_level).validate()
    tone_levels_changed = (settings.reference_level, settings.max_level) != (saved_settings.reference_level, saved_settings.max_level)
    resuming = any(channel["inputs"] for channel in profile["channels"].values())
    if tone_levels_changed and resuming:
        raise ValueError("Tone levels cannot change after saved judgments. Create a new profile for different listening levels")
    playback_settings = replace(settings, noise_level=settings.noise_level if noise_level is None else noise_level,
                                noise_ceiling=settings.noise_ceiling if noise_ceiling is None else noise_ceiling).validate()
    if not 0 <= extra <= 10000 - settings.budget:
        raise ValueError("Extra judgments must keep the total budget between 1 and 10000")
    if extra or tone_levels_changed:
        settings.budget += extra
        playback_settings = replace(playback_settings, budget=settings.budget)
        profile["settings"] = asdict(settings)
        store.save(profile)
    player = Player(playback_settings)
    reference = Reference()
    print(f"Calibrating {profile['name']}: {settings.min_hz:g}–{settings.max_hz:g} Hz")
    try:
        prepared = prepare_volume(player, playback_settings, resuming=resuming, profile_name=profile["name"])
        if prepared is None:
            _session_summary(profile, "Setup cancelled. No new calibration answers were recorded.")
            return
        settings = replace(settings, reference_level=prepared.reference_level, max_level=prepared.max_level).validate()
        if (settings.reference_level, settings.max_level) != (profile["settings"]["reference_level"], profile["settings"]["max_level"]):
            profile["settings"] = asdict(settings)
            store.save(profile)
        playback_settings = prepared
        player.settings = playback_settings
        for channel_index, (channel_name, channel) in enumerate(profile["channels"].items()):
            mode = settings.mode
            if settings.calibration_method == "regional-v2" and "workflow" not in channel:
                if refine(channel, settings, reference):
                    store.save(profile)
                if not choose_local_calibration(store, profile, channel_name, settings):
                    _session_summary(profile, "Local calibration selection cancelled. Saved progress is retained.")
                    return
            while channel["inputs"] < settings.budget:
                if refine(channel, settings, reference):
                    store.save(profile)
                    print("Added nearby frequency points for coverage or contour checks.")
                fit = fit_channel(channel, settings, reference)
                if ready(channel, settings, fit):
                    print(f"\n{channel_name}: consistency thresholds reached.")
                    break
                rng = np.random.default_rng(settings.seed + channel["inputs"] * 1009 + channel_index * 97 + len(channel["blocked"]))
                planning_channel = dict(channel)
                if "workflow" in channel:
                    planning_channel["workflow"] = copy.deepcopy(channel["workflow"])
                planned = next_comparison(planning_channel, settings, rng, reference)
                if planned is None:
                    channel["workflow"] = planning_channel["workflow"]
                    store.save(profile)
                    if planning_channel["workflow"]["stage"] == "review":
                        _session_summary(profile, "Regional shapes need review. Open Settings > Revisit regional shape; saved answers are kept.")
                        return
                    print(f"\n{channel_name}: consistency thresholds and regional checks reached.")
                    break
                channel_label = {"both": "Both channels", "left": "Left ear", "right": "Right ear"}[channel_name]
                steps = 4 if settings.calibration_method == "regional-v2" else 3
                goal = "Compare nearby tones; keep system volume and headphone fit fixed."
                if steps == 4:
                    stage = planning_channel["workflow"]["stage"]
                    goal = ("Check shapes and balance; frozen shapes stay unchanged." if stage == "verify" else
                            "Balance the regions; their shapes are frozen." if stage == "balance" else
                            "Match nearby tones within this region to establish its shape.")
                show_screen(f"Step 3/{steps} | Calibrate | {profile['name']}",
                            (("Goal", (goal,)),),
                            status=(f"Current passage: {progress(planning_channel, settings)}",
                                    f"{channel_label} | Saved answers: {channel['inputs']} / {settings.budget} (maximum)"))
                a, b, order = planned
                order = bool(rng.integers(0, 2)) if order is None else order
                observation, mode = compare(a, b, channel_name, planning_channel, settings, player, reference, rng, mode, order=order)
                if observation == "quit":
                    _session_summary(profile)
                    return
                if observation == "skip":
                    channel["workflow"] = planning_channel["workflow"]
                    channel["blocked"].append([a, b])
                    store.save(profile)
                    continue
                if observation == "skip-region":
                    from .regional import skip_shapes

                    skip_shapes(planning_channel, settings, planning_channel["workflow"]["stage"])
                    channel["workflow"] = planning_channel["workflow"]
                    store.save(profile)
                    continue
                add_observation(channel, settings, a, b, *observation)
                channel["workflow"] = planning_channel["workflow"]
                channel["observations"][-1]["reversed_order"] = order
                store.save(profile)
            else:
                print(f"\n{channel_name}: judgment budget reached; unresolved uncertainty is retained.")
        complete = profile_summary(profile)["status"] == "complete"
        if complete and settings.calibration_method == "regional-v2":
            menu_sound_profile(store, profile, after_calibration=True)
        skipped = any(c.get("workflow", {}).get("skipped") for c in profile["channels"].values())
        message = ("Regional checks complete. Skipped shapes remain uncalibrated; your EQ is ready to preview." if skipped else
                   "Listening checks complete. Your EQ is ready to preview.")
        _session_summary(profile, message if complete else
                         "Answer limit reached. Your calibration remains incomplete.")
    except KeyboardInterrupt:
        store.save(profile)
        _session_summary(profile)


def build_preset(profile, count=12, allow_incomplete=False, bass_db=None, bass_hz=None, filter_mode=None):
    """Calculate a native preset and its report for both preview and export."""
    settings = Settings(**profile["settings"]).validate()
    settings = replace(settings, bass_db=settings.bass_db if bass_db is None else bass_db,
                       bass_hz=settings.bass_hz if bass_hz is None else bass_hz,
                       filter_mode=settings.filter_mode if filter_mode is None else filter_mode).validate()
    if not 1 <= count <= 32:
        raise ValueError("Band count must be 1–32")
    reference = Reference()
    filters, reports = {}, {}
    for channel_name, channel in profile["channels"].items():
        fit = fit_channel(channel, settings, reference)
        if not fit.connected and not (fit.regional or {}).get("exportable", False):
            raise ValueError(f"{channel_name}: comparisons do not connect all sampled frequencies; export would depend on unmeasured priors")
        if not ready(channel, settings, fit) and not allow_incomplete:
            raise ValueError(f"{channel_name}: consistency thresholds or regional checks are not met. Continue calibrating or explicitly use --allow-incomplete")
        correction = reference.correction(fit.frequencies, fit.matches)
        filters[channel_name], metrics = fit_voiced(fit.frequencies, correction, settings, count)
        preference = response(fit.frequencies, [bass_shelf(settings.bass_db, settings.bass_hz)], settings.sample_rate)
        voicing = response(fit.frequencies, voicing_filters(settings.sound_profile), settings.sample_rate)
        reports[channel_name] = {"consistency": quality(channel, settings, reference), "peq_fit": metrics,
                                 "frequency_hz": fit.frequencies.tolist(), "model_correction_db": correction.tolist(),
                                 "skipped_shapes_uncalibrated": (fit.regional or {}).get("skipped_regions", []),
                                 "correction_scope": "Measured regional gains; no local correction in skipped regions" if (fit.regional or {}).get("skipped_regions") else "Local shapes and regional balance",
                                 "bass_preference_curve_db": preference.tolist(),
                                 "sound_preference_curve_db": voicing.tolist(),
                                 "requested_correction_db": (correction + preference + voicing).tolist(),
                                 "peq_shape_tolerance_met": metrics["shape_rmse_db"] <= settings.shape_tolerance}
    preset = easyeffects_preset(filters, settings.sample_rate)
    report = {"profile": profile["name"], "model": profile["model"], "provenance": provenance(),
              "settings": asdict(settings), "channels": reports,
              "input_gain_db": preset["output"]["equalizer#0"]["input-gain"],
              "interpretation": "Approximate model-based correction. Listening consistency and numerical PEQ fitting error do not measure personal diffuse-field accuracy.",
              "hearing_proxy_extrapolated": settings.max_hz > 12500,
              "hearing_proxy_endpoint_held": settings.max_hz > 15000,
              "low_frequency_loudness_is_approximate": settings.min_hz < 2000,
              "bass_preference": {"gain_db": settings.bass_db, "half_gain_hz": settings.bass_hz,
                                  "q": float(1 / np.sqrt(2)), "applied_after_calibration": True,
                                  "bass_region_unmeasured": settings.min_hz >= settings.bass_hz},
              "sound_preference": {"name": settings.sound_profile, "label": SOUND_PROFILES[settings.sound_profile][0],
                                   "applied_after_calibration": True,
                                   "filters": [asdict(band) for band in voicing_filters(settings.sound_profile)],
                                   "outside_calibrated_range_is_preference_only": True},
              "sample_rate_hz": settings.sample_rate,
              "easyeffects_schema_version": "8.2.8"}
    return preset, report


def export_profile(profile, output, count=8, allow_incomplete=False, overwrite=False, bass_db=None, bass_hz=None, filter_mode=None):
    output = Path(output)
    if output.suffix.lower() != ".json":
        raise ValueError("Easy Effects preset output must end with .json")
    report_path = output.with_suffix(".report.json")
    if not overwrite and (output.exists() or report_path.exists()):
        raise ValueError("Export files already exist; choose another path or use --overwrite")
    print("Calculating the EQ preset; fitting can take several seconds...")
    preset, report = build_preset(profile, count, allow_incomplete, bass_db, bass_hz, filter_mode)
    settings = Settings(**report["settings"])
    atomic_json(report_path, report)
    atomic_json(output, preset)
    print(f"Saved Easy Effects preset: {output}")
    print(f"Saved consistency and fit report: {report_path}")
    print(f"  Bass preference: {settings.bass_db:+g} dB at low frequencies, half gain at {settings.bass_hz:g} Hz")
    if settings.bass_db and settings.min_hz >= settings.bass_hz:
        print("  The bass shelf is a preference; this session did not measure its low-frequency region.")
    for channel_name, values in report["channels"].items():
        metrics = values["peq_fit"]
        print(f"  {channel_name}: shape fit RMS {metrics['shape_rmse_db']:.2f} dB; accumulated contour mismatch {metrics['accumulated_contour_error_db']:.2f} dB")
        print(f"    Filter selection: {metrics['selected_topology']} ({metrics['filter_mode']})")
        if not values["peq_shape_tolerance_met"]:
            print("  PEQ fit tolerance was not met; the report retains this limitation.")
    print(f"  Input attenuation for headroom: {report['input_gain_db']:.2f} dB")
    return report


def simulate(store, name, output=None, range_name="treble", bass_db=0.0):
    settings = Settings(**RANGES[range_name], seed=42, bass_db=bass_db)
    profile = store.create(name, settings)
    reference = Reference()
    channel = profile["channels"]["both"]
    start_workflow(channel)
    rng = np.random.default_rng(42)
    for _ in range(settings.budget):
        refine(channel, settings, reference)
        planned = next_comparison(channel, settings, rng, reference)
        if planned is None:
            break
        a, b, order = planned
        frequencies = np.array([a, b])
        bump = 4 * np.exp(-0.5 * (np.log2(frequencies / 6500) / 0.35) ** 2)
        bass_rolloff = -3 * np.maximum(0, np.log2(120 / frequencies))
        matches = -(reference.target(frequencies) + reference.sensitivity(frequencies) + bump + bass_rolloff)
        offset = float(np.clip(matches[1] - matches[0] + rng.normal(0, 0.6), -settings.offset_limit, settings.offset_limit))
        add_observation(channel, settings, a, b, offset, "manual")
        channel["observations"][-1]["reversed_order"] = bool(rng.integers(0, 2)) if order is None else order
    store.save(profile)
    print("Silent simulated session: no audio was played and no human calibration was performed.")
    report_profile(profile)
    if output:
        export_profile(profile, output, count=12, allow_incomplete=True)


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.command is None and not (sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM", "dumb") != "dumb"):
        command.print_help()
        return 0
    store = Store(args.data_dir)
    try:
        if args.command is None:
            from .ui import main_menu

            run_ui(main_menu, store)
        elif args.command == "new":
            preset = args.tolerance
            if preset is None and sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM", "dumb") != "dumb":
                preset = choose_tolerance()
                if preset is None:
                    print("Profile creation cancelled.")
                    return 0
            shape, drift, budget, _ = TOLERANCE_PRESETS[preset or "convinced"]
            names = Settings.__dataclass_fields__
            values = {key: getattr(args, key) for key in names if getattr(args, key, None) is not None}
            if args.bass_style:
                values["bass_hz"] = BASS_STYLES[args.bass_style][0]
            levels = music_reference_levels(args.reference_level, args.max_level)
            settings = Settings(**(RANGES[args.range] | REFERENCE_NOISE_PRESET | dict(shape_tolerance=shape, drift_tolerance=drift, budget=budget, tone_leveling="adaptive-v1", calibration_method="regional-v2") | levels | values)).validate()
            store.create(args.name, settings)
            print(f"Created profile {args.name!r}. Run calibration with: python3 -m phonix --data-dir {shlex.quote(str(args.data_dir))} run {shlex.quote(args.name)}")
            print(f"Shape tolerance {settings.shape_tolerance:g} dB; cross-check disagreement {settings.drift_tolerance:g} dB; cap {settings.budget} judgments per channel, not a required test count.")
        elif args.command == "list":
            list_profiles(store, args.plain)
        elif args.command == "extend":
            profile = store.extend(args.name, args.max_hz)
            print(f"Profile {args.name}: {profile['settings']['min_hz']:g}–{profile['settings']['max_hz']:g} Hz; saved judgments preserved.")
        elif args.command == "tolerance":
            profile = store.set_tolerance(args.name, args.preset)
            settings = Settings(**profile["settings"])
            print(f"{args.name}: {args.preset}, shape tolerance {settings.shape_tolerance:g} dB, cross-check {settings.drift_tolerance:g} dB, cap {settings.budget} judgments per channel; saved judgments preserved.")
        elif args.command == "devices":
            for device in devices():
                print(f"{device.get('object.serial', '?')}: {device.get('node.name', '?')} | {device.get('node.description', '')}")
        elif args.command == "run":
            run_profile(store, store.load(args.name), args.extra_judgments, args.noise_level, args.noise_ceiling,
                        args.reference_level, args.max_level)
        elif args.command == "audio-check":
            check_audio(store.load(args.name), args.device, args.level)
        elif args.command == "report":
            report_profile(store.load(args.name))
        elif args.command == "export":
            bass_hz = BASS_STYLES[args.bass_style][0] if args.bass_style else args.bass_hz
            export_profile(store.load(args.name), args.output, args.bands, args.allow_incomplete, args.overwrite,
                           args.bass_db, bass_hz, args.filter_mode)
        elif args.command == "demo":
            simulate(store, args.name, args.output, args.range, args.bass_db)
    except (ValueError, OSError, subprocess.TimeoutExpired, np.linalg.LinAlgError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    return 0
