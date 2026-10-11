"""Bundled music reference, decoded with the standard library and NumPy."""

from functools import lru_cache
import hashlib
from importlib.resources import files
import io
import json
import math
import wave

import numpy as np

from .profiles import REFERENCE_MAX, TONE_CEILING_MAX


MUSIC_SAMPLE_RATE = 48000
MUSIC_DURATION_SECONDS = 25


@lru_cache(maxsize=1)
def preview_info():
    try:
        info = json.loads(files("phonix").joinpath("data/music_preview.json").read_text())
        if not isinstance(info, dict) or not {"sha256", "title", "artist", "license_url", "measured_lufs", "measured_true_peak_dbfs", "tone_reference"} <= info.keys():
            raise ValueError("Incomplete preview information")
        reference = info["tone_reference"]
        if (not isinstance(reference, dict) or reference.get("method") != "music-band-rms-v1"
                or reference.get("band_low_hz") != 935 or reference.get("band_high_hz") != 1065):
            raise ValueError("Invalid music reference method")
        rms, peak = reference.get("band_rms_dbfs"), reference.get("tone_peak_dbfs")
        if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in (rms, peak)) or not -60 <= peak <= REFERENCE_MAX
                or not math.isclose(peak - rms, 10 * math.log10(2), abs_tol=1e-6)):
            raise ValueError("Invalid music reference levels")
        return info
    except (OSError, ValueError) as error:
        raise ValueError("Built-in music information is missing or invalid; reinstall phonix") from error


def music_reference_levels(reference_level=None, max_level=None):
    """New-profile defaults from precomputed band energy; never decode audio."""
    reference = round(preview_info()["tone_reference"]["tone_peak_dbfs"], 1) if reference_level is None else reference_level
    ceiling = min(TONE_CEILING_MAX, reference + 18) if max_level is None else max_level
    return {"reference_level": reference, "max_level": ceiling}


@lru_cache(maxsize=1)
def music_samples():
    """Return the same prepared stereo excerpt for every profile and replay."""
    try:
        content = files("phonix").joinpath("data/music_preview.wav").read_bytes()
    except OSError as error:
        raise ValueError("Built-in music excerpt is missing; reinstall phonix") from error
    if hashlib.sha256(content).hexdigest() != preview_info().get("sha256"):
        raise ValueError("Built-in music excerpt checksum failed; reinstall phonix")
    try:
        with wave.open(io.BytesIO(content), "rb") as stream:
            expected = (2, 2, MUSIC_SAMPLE_RATE, MUSIC_SAMPLE_RATE * MUSIC_DURATION_SECONDS)
            actual = (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes())
            if actual != expected or stream.getcomptype() != "NONE":
                raise ValueError("Unexpected built-in music format; reinstall phonix")
            pcm = stream.readframes(stream.getnframes())
    except (wave.Error, EOFError) as error:
        raise ValueError("Built-in music excerpt is invalid; reinstall phonix") from error
    if len(pcm) != MUSIC_SAMPLE_RATE * MUSIC_DURATION_SECONDS * 4:
        raise ValueError("Built-in music excerpt is truncated; reinstall phonix")
    samples = np.frombuffer(pcm, dtype="<i2").reshape(-1, 2).astype(float) / 32767
    samples.setflags(write=False)
    return samples
