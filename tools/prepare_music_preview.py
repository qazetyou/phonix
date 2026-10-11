"""Rebuild the bundled excerpt offline; requires FFmpeg and the credited source."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import wave

import numpy as np


SOURCE_SHA256 = "73d6443ef90a7c022f164e5aa90e56c2291585930b39b1656d0765abbc1f1779"
RATE = 48000
START = 30.0
DURATION = 25.0
FADE = 0.2
REFERENCE_LOW_HZ = 935.0
REFERENCE_HIGH_HZ = 1065.0


def reference_from_samples(samples, rate):
    """Full-excerpt time-mean band power, averaging stereo powers, not signals."""
    samples = np.asarray(samples)
    if (samples.ndim != 2 or samples.shape[1] != 2 or len(samples) < 2
            or not np.isfinite(samples).all() or rate <= 2 * REFERENCE_HIGH_HZ):
        raise ValueError("Reference analysis needs finite stereo PCM at a supported rate")
    n = len(samples)
    frequencies = np.fft.rfftfreq(n, 1 / rate)
    band = (frequencies >= REFERENCE_LOW_HZ) & (frequencies <= REFERENCE_HIGH_HZ)
    # Parseval: the retained non-DC/non-Nyquist bins each represent two sides.
    # Analyze channels separately so phase cancellation cannot lower the result.
    powers = [2 * np.sum(abs(np.fft.rfft(samples[:, channel])[band]) ** 2) / n ** 2
              for channel in range(2)]
    power = float(np.mean(powers))
    if not np.isfinite(power) or power <= 0:
        raise ValueError("Music excerpt has no usable energy around 1 kHz")
    rms_dbfs = float(10 * np.log10(power))
    return {"method": "music-band-rms-v1", "band_low_hz": REFERENCE_LOW_HZ,
            "band_high_hz": REFERENCE_HIGH_HZ, "band_rms_dbfs": rms_dbfs,
            "tone_peak_dbfs": rms_dbfs + float(10 * np.log10(2)),
            "averaging": "Full-excerpt mean-square energy, then mean of channel powers; no gating",
            "tone_level": "Equivalent steady-state 1 kHz sine RMS; short playback fades reduce time-mean power slightly"}


def reference_from_wav(path):
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) != (2, 2, "NONE"):
            raise ValueError("Reference analysis needs stereo PCM16")
        rate = stream.getframerate()
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2).astype(float) / 32767
    return reference_from_samples(samples, rate)


def measure(path):
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                             "-af", "ebur128=peak=true", "-f", "null", "-"],
                            capture_output=True, text=True, check=True, timeout=30)
    summary = result.stderr[result.stderr.rfind("Summary:"):]
    return (float(re.search(r"I:\s+([-\d.]+) LUFS", summary).group(1)),
            float(re.search(r"Peak:\s+([-\d.]+) dBFS", summary).group(1)))


def write_wav(path, samples):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(np.rint(samples * 32767).astype("<i2").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "phonix/data")
    args = parser.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA256:
        parser.error("Source checksum differs from the documented recording")
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", str(START),
                              "-i", str(args.source), "-t", str(DURATION), "-map", "0:a:0",
                              "-af", (f"afade=t=in:st=0:d={FADE},afade=t=out:st={DURATION - FADE}:d={FADE},"
                                      f"aresample={RATE},atrim=end_sample={int(RATE * DURATION)}"),
                              "-ac", "2", "-ar", str(RATE), "-f", "f32le", "pipe:1"],
                             capture_output=True, check=True, timeout=30)
    samples = np.frombuffer(decoded.stdout, dtype="<f4").reshape(-1, 2).astype(float)
    if samples.shape != (int(RATE * DURATION), 2) or not np.isfinite(samples).all():
        raise ValueError("Source did not produce the expected complete excerpt")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "music_preview.wav"
    normalization = "loudnorm=I=-14:TP=-1:LRA=11"
    with tempfile.TemporaryDirectory(prefix="phonix-preview-build-") as directory:
        draft = Path(directory) / "draft.wav"
        if abs(samples).max() >= 1:
            raise ValueError("Source decoding exceeds full scale")
        write_wav(draft, samples)
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(draft),
                        "-af", normalization, "-ar", str(RATE), "-ac", "2",
                        "-c:a", "pcm_s16le", "-map_metadata", "-1", str(path)],
                       capture_output=True, check=True, timeout=30)
    final_loudness, final_peak = measure(path)
    metadata = {
        "id": "vibe-ace-preview-v1",
        "title": "Vibe Ace", "artist": "Kevin MacLeod",
        "license": "CC-BY-3.0", "license_url": "https://creativecommons.org/licenses/by/3.0/",
        "track_url": "https://freemusicarchive.org/music/Kevin_MacLeod/Jazz_Sampler/Vibe_Ace_1278",
        "source_url": "https://raw.githubusercontent.com/librosa/data/main/audio/Kevin_MacLeod_-_Vibe_Ace.hq.ogg",
        "source_sha256": SOURCE_SHA256,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_rate_hz": RATE, "channels": 2, "sample_width_bytes": 2,
        "source_start_seconds": START, "duration_seconds": DURATION, "fade_seconds": FADE,
        "target_lufs": -14.0, "normalization_filter": normalization,
        "measured_lufs": final_loudness, "measured_true_peak_dbfs": final_peak,
        "changes": "Excerpted, faded, loudness-normalized once with true-peak limiting, and resampled to 48 kHz stereo PCM16. Playback uses the fixed prepared file.",
        "measurement": "Offline FFmpeg ebur128=peak=true; FFmpeg is not a playback dependency.",
        "tone_reference": reference_from_wav(path),
    }
    (args.output_dir / "music_preview.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {DURATION:g}s excerpt: {final_loudness:g} LUFS, {final_peak:g} dBTP")


if __name__ == "__main__":
    main()
