"""Versioned reference data and explicitly approximate perceptual conversion."""

import hashlib
import json
from importlib.resources import files
from io import BytesIO

import numpy as np

MODEL_ID = "df5128-middle-ear-v1"
MIN_HZ = 20.0
MAX_HZ = 16000.0


def _data(name):
    return files("phonix").joinpath("data", name)


def provenance():
    return json.loads(_data("provenance.json").read_text())


def _curve(name, expected_hash):
    data = _data(name).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"Reference data checksum failed: {name}")
    return np.loadtxt(BytesIO(data), delimiter=",", skiprows=1).T


class Reference:
    def __init__(self):
        metadata = provenance()
        self.df = _curve("df5128.csv", metadata["target"]["sha256"])
        self.middle = _curve("middle_ear.csv", metadata["hearing_proxy"]["sha256"])

    @staticmethod
    def _interpolate(curve, frequencies):
        frequencies = np.asarray(frequencies, dtype=float)
        if np.any(~np.isfinite(frequencies)) or np.any(frequencies < MIN_HZ) or np.any(frequencies > MAX_HZ):
            raise ValueError(f"Supported frequency range is {MIN_HZ:g}–{MAX_HZ:g} Hz")
        return np.interp(np.log(frequencies), np.log(curve[0]), curve[1])

    def target(self, frequencies):
        return self._interpolate(self.df, frequencies)

    def sensitivity(self, frequencies):
        # Above the bundled 15 kHz endpoint, explicitly hold its last value.
        # This extends the domain without changing the v1 baseline below it.
        return self._interpolate(self.middle, frequencies)

    def expected_matches(self, frequencies):
        result = -(self.target(frequencies) + self.sensitivity(frequencies))
        return result - result.mean()

    def correction(self, frequencies, matches):
        # Equal matching levels m = -(headphone response h + sensitivity s) + k.
        # Thus the shape-only correction is DF - h = DF + m + s + k'.
        result = self.target(frequencies) + np.asarray(matches) + self.sensitivity(frequencies)
        # Remove overall gain using uniform weighting in log frequency.
        return result - log_weights(frequencies) @ result


def integrate(values, x):
    function = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(function(values, x))


def log_weights(frequencies):
    """Trapezoidal integration weights with unit mass in log frequency."""
    x = np.log(np.asarray(frequencies, dtype=float))
    if x.ndim != 1 or len(x) < 2 or np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("Log-frequency weights require increasing positive frequencies")
    gaps = np.diff(x)
    return np.concatenate([[gaps[0] / 2], (gaps[:-1] + gaps[1:]) / 2, [gaps[-1] / 2]]) / (x[-1] - x[0])
