"""Atomic, versioned local profiles with strict configuration validation."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import tempfile
import os

from .model import MAX_HZ, MIN_HZ, MODEL_ID


BASS_STYLES = {
    "sub-bass": (60.0, "Deep rumble, with less added upper-bass warmth"),
    "balanced": (105.0, "Bass weight and punch, with moderate added warmth"),
    "warm": (180.0, "Broader bass lift, with more body and warmth"),
}

# These bound conditional listening uncertainty, not acoustic DF accuracy.
# Cross-check disagreement stays strict at every effort level.
TOLERANCE_PRESETS = {
    "doubt": (4.0, 1.5, 80, "If you are in doubt"),
    "convinced": (3.0, 1.5, 150, "If you are convinced"),
    "obsessed": (2.0, 1.5, 300, "If you are now obsessed, and want to perfect your sound profile"),
}

# Retain these legacy noise fields for saved-profile and command compatibility.
# Current calibration preparation uses music and a tone check, without noise.
REFERENCE_NOISE_PRESET = dict(noise_level=-17.7, noise_ceiling=-4.0)

# Leave at least 6 dB of upward adjustment even at the loudest reference.
TONE_CEILING_MAX = -3.0
REFERENCE_MAX = TONE_CEILING_MAX - 6.0


def validate_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,63}", name):
        raise ValueError("Profile names need 1–64 letters, digits, spaces, underscores or hyphens; start with a letter or digit")
    return name


@dataclass
class Settings:
    min_hz: float = 2000.0
    max_hz: float = 16000.0
    channels: str = "together"
    mode: str = "guided"
    budget: int = 240
    shape_tolerance: float = 1.5
    drift_tolerance: float = 1.5
    reference_level: float = -42.0
    max_level: float = -24.0
    sample_rate: int = 48000
    initial_points: int = 7
    max_points: int = 17
    device: str | None = None
    seed: int = 0
    bass_db: float = 0.0
    bass_hz: float = 105.0
    noise_level: float = -20.0
    noise_ceiling: float = -6.0
    filter_mode: str = "auto"
    # Missing legacy fields retain fixed-A playback. New CLI/UI profiles opt in.
    tone_leveling: str = "fixed"
    calibration_method: str = "global-v1"
    sound_profile: str = "neutral"

    def validate(self):
        for key in ("min_hz", "max_hz", "shape_tolerance", "drift_tolerance", "reference_level", "max_level", "bass_db", "bass_hz", "noise_level", "noise_ceiling"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
        for key in ("budget", "sample_rate", "initial_points", "max_points", "seed"):
            if isinstance(getattr(self, key), bool) or not isinstance(getattr(self, key), int):
                raise ValueError(f"{key} must be an integer")
        if not MIN_HZ <= self.min_hz < self.max_hz <= MAX_HZ:
            raise ValueError(f"Frequency limits must satisfy {MIN_HZ:g} <= min < max <= {MAX_HZ:g} Hz")
        if not -12 <= self.bass_db <= 12 or not 40 <= self.bass_hz <= 300:
            raise ValueError("Bass level must be -12 to +12 dB; shelf frequency must be 40–300 Hz")
        if self.channels not in ("together", "separate") or self.mode not in ("manual", "guided"):
            raise ValueError("Invalid channel configuration or comparison mode")
        if self.filter_mode not in ("auto", "bells"):
            raise ValueError("Filter mode must be auto or bells")
        if self.tone_leveling not in ("fixed", "adaptive-v1"):
            raise ValueError("Unsupported tone-leveling method")
        if self.calibration_method not in ("global-v1", "regional-v2"):
            raise ValueError("Unsupported calibration method")
        from .voicing import SOUND_PROFILES

        if not isinstance(self.sound_profile, str) or self.sound_profile not in SOUND_PROFILES:
            raise ValueError("Unknown sound profile")
        if not 1 <= self.budget <= 10000:
            raise ValueError("Budget must be between 1 and 10000 judgments per channel")
        if not 0 <= self.seed < 2 ** 32:
            raise ValueError("Seed must be an integer between 0 and 2^32 - 1")
        if not 0.1 <= self.shape_tolerance <= 12 or not 0.1 <= self.drift_tolerance <= 12:
            raise ValueError("Tolerances must be between 0.1 and 12 dB")
        if not -60 <= self.reference_level <= REFERENCE_MAX or not self.reference_level + 6 <= self.max_level <= TONE_CEILING_MAX:
            raise ValueError("Reference level must be -60 to -9 dBFS; ceiling must be at least 6 dB higher and no higher than -3 dBFS")
        if not -60 <= self.noise_level <= -12 or not -24 <= self.noise_ceiling <= -3 or self.noise_level > self.noise_ceiling:
            raise ValueError("Noise RMS level must be -60 to -12 dBFS, no higher than its peak ceiling (-24 to -3 dBFS)")
        if self.sample_rate not in (44100, 48000, 96000):
            raise ValueError("Sample rate must be 44100, 48000 or 96000 Hz")
        if not 4 <= self.initial_points <= self.max_points <= 32:
            raise ValueError("Point counts must satisfy 4 <= initial points <= max points <= 32")
        if self.device is not None and (not isinstance(self.device, str) or not self.device or len(self.device) > 256 or any(ord(c) < 32 for c in self.device)):
            raise ValueError("Device must be a PipeWire node name or serial")
        return self

    @property
    def offset_limit(self):
        return min(24.0, self.max_level - self.reference_level, self.reference_level + 78.0)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".saving-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class Store:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, name):
        return self.root / (validate_name(name) + ".json")

    @staticmethod
    def _new_profile(name, settings):
        from .engine import new_channel

        settings.validate()
        channels = ["both"] if settings.channels == "together" else ["left", "right"]
        return {"schema": 1, "name": validate_name(name), "model": MODEL_ID, "settings": asdict(settings),
                "channels": {channel: new_channel(settings) for channel in channels}}

    def create(self, name, settings):
        profile = self._new_profile(name, settings)
        if self.path(name).exists():
            raise ValueError(f"Profile already exists: {name}")
        self.save(profile)
        return profile

    def reset(self, name):
        """Start fresh with saved settings; callers must obtain confirmation."""
        previous = self.load(name)
        profile = self._new_profile(name, Settings(**previous["settings"]))
        self.save(profile)
        return profile

    def save(self, profile):
        atomic_json(self.path(profile["name"]), profile)

    def delete(self, name):
        """Remove only this named profile; callers must obtain confirmation."""
        try:
            self.path(name).unlink()
        except FileNotFoundError as error:
            raise ValueError(f"No profile named {name!r}") from error

    def set_tolerance(self, name, preset):
        """Change stopping thresholds and cap without discarding evidence."""
        from dataclasses import replace

        if preset not in TOLERANCE_PRESETS:
            raise ValueError("Unknown tolerance preset")
        profile = self.load(name)
        shape, drift, budget, _ = TOLERANCE_PRESETS[preset]
        # A smaller cap never makes saved observations invalid or erases them.
        used = max(channel["inputs"] for channel in profile["channels"].values())
        settings = replace(Settings(**profile["settings"]), shape_tolerance=shape,
                           drift_tolerance=drift, budget=max(budget, used)).validate()
        profile["settings"] = asdict(settings)
        self.save(profile)
        return profile

    def extend(self, name, max_hz=MAX_HZ):
        """Expand the upper limit while preserving every saved comparison."""
        from .engine import new_channel

        profile = self.load(name)
        settings = Settings(**profile["settings"])
        if (settings.calibration_method == "regional-v2" and max_hz != settings.max_hz
                and any(c.get("workflow", {}).get("frozen") for c in profile["channels"].values())):
            raise ValueError("Frozen regional shapes cannot change range. Create a new profile for an extended range")
        if max_hz < settings.max_hz:
            raise ValueError("Extend cannot shrink a range; create a new profile for narrower limits")
        if max_hz == settings.max_hz:
            return profile
        settings.max_hz = max_hz
        settings.validate()
        has_history = any(channel["inputs"] or channel["blocked"] for channel in profile["channels"].values())
        if has_history:
            required = max(len(channel["frequencies"]) + 1 for channel in profile["channels"].values())
            settings.max_points = max(settings.max_points, required)
            settings.validate()
            for channel in profile["channels"].values():
                channel["frequencies"].append(max_hz)
        else:
            profile["channels"] = {channel: new_channel(settings) for channel in profile["channels"]}
        profile["settings"] = asdict(settings)
        self.save(profile)
        return profile

    def load(self, name):
        try:
            profile = json.loads(self.path(name).read_text())
        except FileNotFoundError as error:
            raise ValueError(f"No profile named {name!r}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"Profile contains invalid JSON: {name}") from error
        if not isinstance(profile, dict) or profile.get("schema") != 1 or profile.get("model") != MODEL_ID or profile.get("name") != name:
            raise ValueError("Unsupported or mismatched profile schema, name or hearing baseline")
        try:
            settings = Settings(**profile["settings"]).validate()
            channels = profile["channels"]
            expected = {"both"} if settings.channels == "together" else {"left", "right"}
            if not isinstance(channels, dict) or set(channels) != expected:
                raise ValueError("Channel data does not match the configuration")
            from .engine import validate_channel

            for channel in channels.values():
                validate_channel(channel, settings)
        except (KeyError, TypeError) as error:
            raise ValueError("Malformed profile data") from error
        return profile

    def names(self):
        return sorted(path.stem for path in self.root.glob("*.json")) if self.root.exists() else []
