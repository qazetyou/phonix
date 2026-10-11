"""Built-in music, reference noise and ramped tones through PipeWire."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import wave

import numpy as np

from .profiles import TONE_CEILING_MAX
from .music import MUSIC_SAMPLE_RATE, music_samples

# Prevent effects applications from moving calibration streams after linking.
PLAYBACK_PROPERTIES = ("application.name=phonix node.name=phonix "
                       "node.dont-move=true node.dont-fallback=true")

# Feed silence through the same stream before its first sound, allowing output
# startup to settle. A Python sleep before launching pw-play cannot do this.
PLAYBACK_LEAD_IN_SECONDS = 0.5

# Approximately one nominal auditory-filter width at 1 kHz; fixed across
# profiles, not a measurement of the listener's individual auditory filters.
REFERENCE_NOISE_LOW_HZ = 935.0
REFERENCE_NOISE_HIGH_HZ = 1065.0


def tone(frequency, level_dbfs, channel, rate=48000, duration=0.45, ceiling=-24.0):
    if not np.isfinite(frequency) or not 20 <= frequency < rate / 2:
        raise ValueError("Tone frequency must be at least 20 Hz and below Nyquist")
    if not np.isfinite(level_dbfs) or not -78 <= level_dbfs <= ceiling <= TONE_CEILING_MAX:
        raise ValueError("Tone level exceeds configured playback bounds")
    if channel not in ("both", "left", "right") or not 0.05 <= duration <= 2:
        raise ValueError("Invalid tone channel or duration")
    n = int(rate * duration)
    t = np.arange(n) / rate
    samples = 10 ** (level_dbfs / 20) * np.sin(2 * np.pi * frequency * t)
    ramp = min(int(rate * 0.02), n // 2)
    envelope = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp))
    samples[:ramp] *= envelope
    samples[-ramp:] *= envelope[::-1]
    stereo = np.zeros((n, 2))
    if channel in ("both", "left"):
        stereo[:, 0] = samples
    if channel in ("both", "right"):
        stereo[:, 1] = samples
    return stereo


def pink_noise(level_dbfs, rate=48000, duration=5.0, ceiling=-6.0):
    """Repeatable 20 Hz–20 kHz pink noise at the requested RMS, peak limited.

    This supplies the separate reference-matching noise, not music playback.
    It does not measure acoustic SPL. A restrictive peak ceiling
    attenuates the whole signal below its requested RMS; it does not clip it.
    """
    if not np.isfinite(level_dbfs) or not np.isfinite(ceiling) or not -60 <= level_dbfs <= -12 or not -24 <= ceiling <= -3 or level_dbfs > ceiling:
        raise ValueError("Noise level exceeds configured playback bounds")
    if rate not in (44100, 48000, 96000) or not np.isfinite(duration) or not 0.5 <= duration <= 10:
        raise ValueError("Invalid noise sample rate or duration")
    n = int(rate * duration)
    frequencies = np.fft.rfftfreq(n, 1 / rate)
    spectrum = np.fft.rfft(np.random.default_rng(0).normal(size=n))
    weights = np.zeros_like(frequencies)
    in_band = (frequencies >= 20) & (frequencies <= min(20000, rate / 2))
    weights[in_band] = 1 / np.sqrt(frequencies[in_band])
    samples = np.fft.irfft(spectrum * weights, n=n)
    ramp = int(rate * 0.02)
    envelope = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp))
    samples[:ramp] *= envelope
    samples[-ramp:] *= envelope[::-1]
    requested_rms = 10 ** (level_dbfs / 20)
    scale = min(requested_rms / np.sqrt(np.mean(samples ** 2)),
                10 ** (ceiling / 20) / np.max(abs(samples)))
    samples *= scale
    return np.column_stack((samples, samples))


def comparison_levels(offset, settings, base_level=None):
    """Canonical A/B peaks; shift the pair together to retain its difference."""
    if not np.isfinite(offset) or abs(offset) > settings.offset_limit + 1e-6:
        raise ValueError("Comparison offset exceeds configured bounds")
    if base_level is None:
        return settings.reference_level, settings.reference_level + offset
    if isinstance(base_level, bool) or not np.isfinite(base_level):
        raise ValueError("Comparison starting level must be finite")
    # Feasible A levels given B = A + offset. Never clip individual waveforms.
    lower = max(-78.0, -78.0 - offset)
    upper = min(settings.max_level, settings.max_level - offset)
    first = float(np.clip(base_level, lower, upper))
    # Suppress floating-point overshoot at an exact endpoint, not audio clipping.
    return first, float(np.clip(first + offset, -78.0, settings.max_level))


def pair_samples(a, b, offset, reversed_order, channel, settings, base_level=None):
    first_level, second_level = comparison_levels(offset, settings, base_level)
    # Give low bass at least 15 cycles; both tones in a pair have equal duration.
    duration = max(0.45, 15 / min(a, b)) if min(a, b) > 0 else 0.45
    first = tone(a, first_level, channel, settings.sample_rate, duration=duration, ceiling=settings.max_level)
    second = tone(b, second_level, channel, settings.sample_rate, duration=duration, ceiling=settings.max_level)
    if reversed_order:
        first, second = second, first
    silence = np.zeros((int(settings.sample_rate * 0.25), 2))
    return np.vstack([first, silence, second, silence])


def reference_band_noise(settings):
    """Keep 935–1065 Hz from the fixed reference noise, without gain boosts."""
    rate = settings.sample_rate
    source = pink_noise(settings.noise_level, rate, ceiling=settings.noise_ceiling)[:, 0]
    frequencies = np.fft.rfftfreq(len(source), 1 / rate)
    spectrum = np.fft.rfft(source)
    # Unit gain inside the band, zero elsewhere. Do not normalize RMS or peaks
    # after filtering: that would boost the surviving band back up.
    spectrum[(frequencies < REFERENCE_NOISE_LOW_HZ) | (frequencies > REFERENCE_NOISE_HIGH_HZ)] = 0
    noise = np.fft.irfft(spectrum, n=len(source))[:2 * rate].copy()
    ramp = int(rate * 0.02)
    envelope = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp))
    noise[:ramp] *= envelope
    noise[-ramp:] *= envelope[::-1]
    # Filtering can change peak shape; allow only attenuation if needed.
    peak, ceiling = np.max(abs(noise)), 10 ** (settings.noise_ceiling / 20)
    if peak > ceiling:
        noise *= ceiling / peak
    return np.column_stack((noise, noise))


def reference_match_samples(settings, noise_first):
    """Compare the reference noise's unboosted 1 kHz band to a two-second tone."""
    rate = settings.sample_rate
    noise = reference_band_noise(settings)
    reference = tone(1000, settings.reference_level, "both", rate, duration=2.0, ceiling=settings.max_level)
    first, second = (noise, reference) if noise_first else (reference, noise)
    silence = np.zeros((int(rate * 0.25), 2))
    return np.vstack([first, silence, second, silence])


class Player:
    def __init__(self, settings):
        self.settings = settings
        self.executable = shutil.which("pw-play")
        if self.executable is None:
            raise ValueError("pw-play was not found. Install PipeWire's command-line tools first")

    def play(self, a, b, offset, reversed_order, channel, base_level=None):
        samples = pair_samples(a, b, offset, reversed_order, channel, self.settings, base_level)
        self._play_samples(samples)

    def play_tone(self, frequency=1000, channel="both"):
        samples = tone(frequency, self.settings.reference_level, channel, self.settings.sample_rate,
                       duration=1.0, ceiling=self.settings.max_level)
        self._play_samples(samples)

    def play_pink_noise(self):
        samples = pink_noise(self.settings.noise_level, self.settings.sample_rate,
                             ceiling=self.settings.noise_ceiling)
        self._play_samples(samples)

    def play_music(self, cancel=None):
        options = {"sample_rate": MUSIC_SAMPLE_RATE}
        if cancel is not None:
            options["cancel"] = cancel
        return self._play_samples(music_samples(), **options)

    def play_reference_match(self, noise_first):
        self._play_samples(reference_match_samples(self.settings, noise_first))

    def _play_samples(self, samples, sample_rate=None, cancel=None):
        rate = self.settings.sample_rate if sample_rate is None else sample_rate
        timeout = max(15, len(samples) / rate + PLAYBACK_LEAD_IN_SECONDS + 10)
        with tempfile.TemporaryDirectory(prefix="phonix-tones-") as directory:
            path = Path(directory) / "comparison.wav"
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(2)
                stream.setsampwidth(2)
                stream.setframerate(rate)
                lead_in = np.zeros((int(rate * PLAYBACK_LEAD_IN_SECONDS), 2), dtype="<i2")
                stream.writeframes(lead_in.tobytes())
                stream.writeframes(np.rint(samples * 32767).astype("<i2").tobytes())
            command = [self.executable, "--properties", PLAYBACK_PROPERTIES]
            if self.settings.device:
                command += ["--target", self.settings.device]
            command.append(str(path))
            with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as process:
                try:
                    if cancel is None:
                        _, errors = process.communicate(timeout=timeout)
                    else:
                        deadline = time.monotonic() + timeout
                        while True:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise subprocess.TimeoutExpired(command, timeout)
                            try:
                                _, errors = process.communicate(timeout=min(0.05, remaining))
                                break
                            except subprocess.TimeoutExpired:
                                if cancel():
                                    _stop_playback(process)
                                    return False
                except BaseException:
                    _stop_playback(process)
                    raise
                if process.returncode:
                    detail = errors.decode(errors="replace").strip()[-600:]
                    raise ValueError(f"PipeWire playback failed: {detail or 'no error detail'}")
        return True


def _stop_playback(process):
    """Reap a stopped stream before its temporary WAV is removed."""
    try:
        process.terminate()
    except ProcessLookupError:
        pass  # It may have completed between the cancel key and this signal.
    try:
        process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()


def devices():
    import json

    executable = shutil.which("pw-dump")
    if executable is None:
        raise ValueError("pw-dump was not found")
    result = subprocess.run([executable], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise ValueError("Cannot contact PipeWire. Run this command in your desktop session")
    nodes = json.loads(result.stdout)
    return [entry["info"]["props"] for entry in nodes
            if entry.get("info", {}).get("props", {}).get("media.class") == "Audio/Sink"]
