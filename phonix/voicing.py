"""Optional, fixed post-calibration preferences; no listening-model changes."""

from dataclasses import asdict
import numpy as np

from .peq import Band, _outside_grid, fit_with_bass, response

# Shelf gains are preferences relative to the calibrated reference, not targets
# measured from listeners. Midpoint frequencies and Q stay fixed and explicit.
SOUND_PROFILES = {
    "neutral": ("Neutral/reference", "No added voicing; keep the calibrated DF reference", ()),
    "v-shaped": ("V-shaped", "More bass weight and treble sparkle; mids sit further back",
                 (("Lo-shelf", 180.0, 3.0), ("Hi-shelf", 4000.0, 3.0))),
    "warm-tilt": ("Warm tilt", "More body with a gentler, less bright top end",
                  (("Lo-shelf", 250.0, 2.0), ("Hi-shelf", 2500.0, -2.0))),
    "bright": ("Bright", "A modest lift to treble detail and brightness",
               (("Hi-shelf", 3000.0, 2.0),)),
}


def voicing_filters(name):
    if not isinstance(name, str) or name not in SOUND_PROFILES:
        raise ValueError("Unknown sound profile")
    return [Band(frequency, gain, float(1 / np.sqrt(2)), kind, "Sound preference")
            for kind, frequency, gain in SOUND_PROFILES[name][2]]


def fit_voiced(frequencies, correction, settings, count):
    """Reserve slots for exact preferences, fitting only measured correction."""
    preferences = voicing_filters(settings.sound_profile)
    required = len(preferences) + int(settings.bass_db != 0) + 1
    if count < required:
        raise ValueError(f"This sound profile and bass setting need at least {required} bands, including one correction band")
    bands, metrics = fit_with_bass(frequencies, correction, settings.sample_rate,
                                  count - len(preferences), settings.bass_db,
                                  settings.bass_hz, settings.filter_mode)
    bands.extend(preferences)
    metrics["bands"] = [asdict(band) for band in bands]
    metrics["sound_profile"] = settings.sound_profile
    guards = _outside_grid(frequencies, settings.sample_rate)
    metrics["outside_range_peak_db"] = float(np.max(abs(response(guards, bands, settings.sample_rate))))
    return bands, metrics
