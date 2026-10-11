"""Guided setup and launch menu built on the existing terminal picker."""

from dataclasses import asdict, replace
import subprocess

import numpy as np

from .audio import devices
from .cli import (RANGES, _menu_error, _menu_text, browse_profiles, check_audio,
                  choose_bass_shelf, choose_tolerance, simulate)
from .profiles import Settings, TOLERANCE_PRESETS, REFERENCE_NOISE_PRESET, validate_name
from .music import music_reference_levels
from .terminal import select_table


NUMBER_FIELDS = {
    "min_hz": ("Lowest frequency (Hz)", float),
    "max_hz": ("Highest frequency (Hz)", float),
    "budget": ("Maximum answers per channel", int),
    "shape_tolerance": ("Listening shape tolerance (dB)", float),
    "drift_tolerance": ("Repeated-check tolerance (dB)", float),
    "reference_level": ("Starting tone level (dBFS peak)", float),
    "max_level": ("Maximum tone level (dBFS peak)", float),
    "sample_rate": ("Sample rate (Hz)", int),
    "initial_points": ("Starting frequency points", int),
    "max_points": ("Maximum frequency points", int),
    "seed": ("Random seed", int),
    "bass_db": ("Bass level (dB)", float),
}


NUMBER_HINTS = {
    "min_hz": "Lowest tone to calibrate",
    "max_hz": "Highest tone to calibrate",
    "budget": "A limit, not a required number of tests",
    "shape_tolerance": "Lower values ask for more listening consistency",
    "drift_tolerance": "Allowed disagreement between repeated checks",
    "reference_level": "Starting volume for the audible-and-comfortable tone check",
    "max_level": "Highest peak allowed for test tones",
    "sample_rate": "Match your audio system; 48000 is the default",
    "initial_points": "Number of frequencies in the starting grid",
    "max_points": "Limit on frequencies added during calibration",
    "seed": "Controls repeatable ordering; usually leave at 0",
    "bass_db": "0 keeps the DF target; positive adds bass, negative reduces it",
}
CHANNEL_LABELS = {"together": "Both ears together", "separate": "Each ear separately"}
MODE_LABELS = {"guided": "Answer which tone is louder", "manual": "Adjust tones yourself"}


def edit_number(settings, field):
    label, convert = NUMBER_FIELDS[field]
    while True:
        entered = _menu_text(f"{label} [{getattr(settings, field):g}] (q cancels): ")
        if entered is None or not entered or entered.lower() == "q":
            return settings
        try:
            return replace(settings, **{field: convert(entered)}).validate()
        except ValueError as error:
            print(f"Invalid value: {error}")


def edit_numbers(settings, fields, title):
    selected = 0
    while True:
        rows = [(NUMBER_FIELDS[field][0], f"{getattr(settings, field):g}", NUMBER_HINTS[field]) for field in fields]
        rows.append(("Back", "", "Keep these values"))
        selected = select_table(title, ("Setting", "Value", "What it does"), rows, selected, description_column=-1)
        if selected is None or selected == len(fields):
            return settings
        settings = edit_number(settings, fields[selected])


def choose_range(settings):
    names = list(RANGES)
    initial = next((index for index, item in enumerate(RANGES.values())
                    if (item["min_hz"], item["max_hz"]) == (settings.min_hz, settings.max_hz)), 0)
    descriptions = {"bass": "Bass plus a midrange reference", "midrange": "Voices and instruments",
                    "treble": "Detail and brightness", "full": "Bass, midrange and treble (recommended)"}
    rows = [("Full range" if name == "full" else name.capitalize(),
             f"{values['min_hz']:g}-{values['max_hz']:g}", descriptions[name]) for name, values in RANGES.items()]
    selected = select_table("Frequency range", ("Range", "Hz", "What you will calibrate"), rows, initial, description_column=-1)
    if selected is None:
        return settings
    # Range controls sampling; the chosen listening effort controls its cap.
    values = {key: value for key, value in RANGES[names[selected]].items() if key != "budget"}
    return replace(settings, **values).validate()


def choose_device(settings):
    rows = [("System default", "Follow the current system output")]
    targets = [None]
    try:
        for sink in devices():
            target = sink.get("node.name") or sink.get("object.serial")
            if target is None:
                continue
            targets.append(str(target))
            rows.append((sink.get("node.description", str(target)), str(target)))
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        print(f"Could not list outputs: {error}. You can enter a target manually.")
    if settings.device and settings.device not in targets:
        targets.append(settings.device)
        rows.append(("Current saved target", settings.device))
    initial = targets.index(settings.device)
    rows.append(("Manual input", "PipeWire output node name or serial"))
    selected = select_table("Playback output | choose a physical output for calibration", ("Output", "Target"), rows, initial)
    if selected is None:
        return settings
    if selected == len(targets):
        while True:
            entered = _menu_text("PipeWire target (q cancels): ")
            if entered is None or not entered or entered.lower() == "q":
                return settings
            try:
                return replace(settings, device=entered).validate()
            except ValueError as error:
                print(f"Invalid output: {error}")
    return replace(settings, device=targets[selected]).validate()


def choose_option(title, rows, current):
    initial = next(index for index, row in enumerate(rows) if row[0] == current)
    selected = select_table(title, ("Choice", "Details"), [row[1:] for row in rows], initial)
    return current if selected is None else rows[selected][0]


def ask_name(store, purpose="Profile"):
    while True:
        name = _menu_text(f"{purpose} name (q cancels): ")
        if name is None or name.lower() == "q":
            return None
        try:
            validate_name(name)
            if store.path(name).exists():
                raise ValueError("A profile with that name already exists; choose another name")
            return name
        except ValueError as error:
            print(error)


def create_profile(store):
    name = ask_name(store)
    if name is None:
        return None
    shape, drift, budget, _ = TOLERANCE_PRESETS["convinced"]
    settings = Settings(**(RANGES["full"] | REFERENCE_NOISE_PRESET | music_reference_levels() | dict(shape_tolerance=shape, drift_tolerance=drift, budget=budget, tone_leveling="adaptive-v1", calibration_method="regional-v2")))
    selected = 0
    while True:
        rows = [
            ("Output device", settings.device or "System default", "Where test tones will play"),
            ("Frequency range", f"{settings.min_hz:g}-{settings.max_hz:g} Hz", "Which part of the sound to calibrate"),
            ("Listening effort", f"{settings.shape_tolerance:g} dB; up to {settings.budget} answers/channel", "May finish before this limit"),
            ("Bass sound", f"{settings.bass_db:+g} dB; {settings.bass_hz:g} Hz", "Choose bass amount and warmth"),
            ("Ears", CHANNEL_LABELS[settings.channels], "Together uses one shared correction"),
            ("Comparison method", MODE_LABELS[settings.mode], "How you will compare the tones"),
            ("Advanced settings", "Custom numeric controls", "Limits, playback levels and frequency sampling"),
            ("Create profile", "Save and open this profile", "Calibration starts when you choose Start"),
            ("Cancel", "Discard this setup", "Return to Home"),
        ]
        selected = select_table(f"New profile > {name}", ("Setting", "Current choice", "What it does"), rows, selected, description_column=-1)
        if selected is None or selected == 8:
            return None
        if selected == 7:
            profile = store.create(name, settings)
            print(f"Created {name!r}.")
            return profile
        if selected == 0:
            settings = choose_device(settings)
        elif selected == 1:
            settings = choose_range(settings)
        elif selected == 2:
            preset = choose_tolerance(settings)
            if preset is not None:
                shape, drift, budget, _ = TOLERANCE_PRESETS[preset]
                settings = replace(settings, shape_tolerance=shape, drift_tolerance=drift, budget=budget)
        elif selected == 3:
            settings = edit_bass_sound(settings)
        elif selected == 4:
            choice = choose_option("Which ears to calibrate?", [("together", CHANNEL_LABELS["together"], "One correction shared by left and right"),
                                   ("separate", CHANNEL_LABELS["separate"], "Separate correction for left and right")], settings.channels)
            settings = replace(settings, channels=choice)
        elif selected == 5:
            choice = choose_option("How would you like to compare tones?", [("guided", MODE_LABELS["guided"], "The program chooses the next level"),
                                   ("manual", MODE_LABELS["manual"], "Adjust the level until both sound equally loud")], settings.mode)
            settings = replace(settings, mode=choice)
        else:
            fields = tuple(field for field in NUMBER_FIELDS if field != "bass_db")
            settings = edit_numbers(settings, fields, "Advanced profile settings")


def edit_bass_sound(settings):
    selected = 0
    while True:
        rows = [("Bass level", f"{settings.bass_db:+g} dB", NUMBER_HINTS["bass_db"]),
                ("Bass warmth", f"{settings.bass_hz:g} Hz", "Choose where bass changes; has no effect at 0 dB"),
                ("Back", "Keep these choices", "Return to the previous menu")]
        selected = select_table("Bass sound", ("Setting", "Current choice", "What it does"), rows, selected, description_column=-1)
        if selected is None or selected == 2:
            return settings
        settings = edit_number(settings, "bass_db") if selected == 0 else choose_bass_shelf(settings) or settings


def audio_menu(store, profile):
    selected = 0
    rows = [("Check sound", "Play a short tone to check your output"),
            ("Playback settings", "Choose output and edit starting tone levels"),
            ("Back", "Return to Settings")]
    while True:
        selected = select_table(f"{profile['name']} > Settings > Audio and volume",
                                ("Action", "Details"), rows, selected)
        if selected is None or selected == 2:
            return
        try:
            profile = store.load(profile["name"])
            if selected == 0:
                check_audio(profile)
            else:
                edit_playback(store, profile)
        except (ValueError, OSError, subprocess.TimeoutExpired, np.linalg.LinAlgError) as error:
            _menu_error(error, "Audio and volume")


def edit_playback(store, profile):
    """Save playback preferences while protecting the level of prior judgments."""
    settings = Settings(**profile["settings"])
    resuming = any(channel["inputs"] for channel in profile["channels"].values())
    fields = () if resuming else ("reference_level", "max_level", "sample_rate")
    selected = 0
    while True:
        rows = [("Output device", settings.device or "System default", "Where tones and the music preview will play")]
        rows += [(NUMBER_FIELDS[field][0], f"{getattr(settings, field):g}", NUMBER_HINTS[field]) for field in fields]
        if not resuming:
            rows.append(("Default tone levels", "Restore the music-based starting level", "Uses the excerpt's average energy around 1 kHz"))
        rows += [("Save", "Apply these settings", "Return to Audio and volume"),
                 ("Cancel", "Discard these changes", "Keep the previously saved settings")]
        title = "Playback settings | tone levels stay fixed once calibration starts" if resuming else "Playback settings"
        selected = select_table(title, ("Setting", "Current choice", "What it does"), rows, selected, description_column=-1)
        if selected is None or selected == len(rows) - 1:
            return
        if selected == len(rows) - 2:
            profile["settings"] = asdict(settings.validate())
            store.save(profile)
            return
        if not resuming and selected == len(rows) - 3:
            settings = replace(settings, **music_reference_levels()).validate()
        else:
            settings = choose_device(settings) if selected == 0 else edit_number(settings, fields[selected - 1])


def show_devices():
    rows = [(sink.get("node.description", ""), sink.get("node.name", "?"), str(sink.get("object.serial", "?")))
            for sink in devices()]
    if rows:
        select_table("Available outputs | viewing this list does not switch outputs", ("Output", "Node", "Serial"), rows)
    else:
        print("No PipeWire outputs found.")


def tools_menu(store):
    rows = [
        ("Output devices", "View available audio outputs for troubleshooting"),
        ("Silent demo", "Create a simulated profile without playing sound"),
        ("Back", "Return to Home"),
    ]
    selected = 0
    while True:
        selected = select_table("Home > Tools", ("Action", "Details"), rows, selected)
        if selected is None or selected == 2:
            return
        try:
            if selected == 0:
                show_devices()
            else:
                name = ask_name(store, "Demo profile")
                if name is not None:
                    simulate(store, name)
                    _menu_text("Enter to return to Tools: ")
        except (ValueError, OSError, subprocess.TimeoutExpired, np.linalg.LinAlgError) as error:
            _menu_error(error, "Tools")


def help_menu():
    topics = [
        ("Getting started", "Create a profile, set listening volume, calibrate and export"),
        ("Listening controls", "Replay, compare tones, skip and save your place"),
        ("Understanding results", "Profile status, listening effort and EQ limits"),
        ("Back", "Return to Home"),
    ]
    selected = 0
    while True:
        selected = select_table("Home > Help", ("Topic", "Details"), topics, selected)
        if selected is None or selected == 3:
            return
        if selected == 0:
            print("1. Choose New profile and select the output you want to calibrate.")
            print("2. Start calibration sets music volume, checks the reference tone, then opens comparisons.")
            print("   Set your usual system volume with the music excerpt, then confirm the tone is clearly audible and comfortable.")
            print("3. New profiles learn regional shapes, freeze them, then adjust regional balance and verify.")
            print("4. Progress saves after every answer. Continue calibration resumes where you left off.")
            print("5. Choose Neutral/reference, V-shaped, Warm tilt or Bright after calibration.")
            print("   Settings > Sound profile changes that preference later without recalibrating.")
            print("6. Preview EQ shows your filters. Export to Easy Effects saves a preset for import there.")
            print("If tones are silent: Settings > Audio and volume > Check sound. Choose a physical output rather than the Easy Effects virtual sink.")
        elif selected == 1:
            print("Menus: Up/Down selects; Enter opens; q or Escape returns one menu. Ctrl-C exits.")
            print("While music plays: Enter, Esc or s stops the preview early and returns to volume setup.")
            print("Reference-tone check: Enter/c confirms the tone is clearly audible and comfortable; q cancels.")
            print("Reference level: +/- adjusts 1 dB; ++/-- adjusts 3 dB; r replays. Saved judgments lock this level.")
            print("Guided comparisons: 1 first louder, 2 second louder, e equal, r replay, u unsure.")
            print("Manual comparisons: +/- adjusts 0.5 dB; ++/-- adjusts 2 dB; e accepts a match.")
            print("m/g switches method; s skips an inaudible pair; q saves and returns.")
            print("New regional profiles: k skips the active local shape after confirmation. Local calibration offers Skip all local passes.")
            print("h opens help within any calibration step; returning keeps your current comparison.")
            print("Text fields: type and press Enter; Escape cancels. Up/Down or Page Up/Down scrolls output.")
        else:
            print("new: no saved answers. unfinished: calibration is in progress.")
            print("incomplete: the answer limit was reached; Continue offers more answers. complete: listening checks passed.")
            print("Listening effort sets consistency tolerance and a maximum answer count. You may finish before that maximum.")
            print("Tones stay within one octave. New profiles freeze regional shapes before balance and verification.")
            print("Checks stay in one range until its block finishes. Nearby boundary balance runs in separate blocks.")
            print("Conflicting checks: Settings > Revisit regional shape retains answers and repeats balance afterward.")
            print("The target is B&K 5128 diffuse field (DF), interpreted with a fixed hearing model.")
            print("Listening consistency and EQ fitting error do not measure personal acoustic accuracy against DF.")
            print("Skipped shapes stay uncalibrated; only measured regional gains correct them. Settings > Revisit regional shape can calibrate them later.")
            print("Export needs measured local shapes or explicitly skipped shapes with measured regional balance. Unfinished estimates are provisional.")
            print("Bass sound is a preference: 0 dB keeps the DF target; a positive level adds bass.")
        _menu_text("Enter to return to Help: ")


def main_menu(store):
    rows = [("Profiles", "Choose equipment to calibrate or view its EQ"),
            ("New profile", "Set up another pair of headphones or audio device"),
            ("Help", "Getting started, listening controls and results"),
            ("Tools", "Audio output information and a silent demo"),
            ("Quit", "Close the program")]
    selected = 0
    while True:
        selected = select_table("phonix > Home", ("Action", "Details"), rows, selected)
        if selected is None or selected == 4:
            return
        try:
            if selected == 0:
                if store.names():
                    browse_profiles(store)
                else:
                    print("No saved profiles yet. Choose New profile to set up your equipment.")
                    _menu_text("Enter to return to Home: ")
            elif selected == 1:
                profile = create_profile(store)
                if profile is not None:
                    browse_profiles(store, open_name=profile["name"])
            elif selected == 2:
                help_menu()
            else:
                tools_menu(store)
        except (ValueError, OSError, subprocess.TimeoutExpired, np.linalg.LinAlgError) as error:
            _menu_error(error, "Home")
