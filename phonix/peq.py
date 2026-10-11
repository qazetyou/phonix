"""Bounded RBJ/APO fitting and an independent bass-preference shelf."""

from dataclasses import asdict, dataclass

import numpy as np

from .model import MAX_HZ, integrate


@dataclass
class Band:
    frequency: float
    gain: float
    q: float
    kind: str = "Bell"
    purpose: str = "Correction"


def bass_shelf(gain, frequency=105.0):
    if not np.isfinite(gain) or not -12 <= gain <= 12 or not np.isfinite(frequency) or not 40 <= frequency <= 300:
        raise ValueError("Bass level must be -12 to +12 dB; shelf frequency must be 40–300 Hz")
    return Band(float(frequency), float(gain), float(1 / np.sqrt(2)), "Lo-shelf", "Bass preference")


def _outside_grid(frequencies, sample_rate):
    # Include actual out-of-range tails even when calibration starts at 20 Hz.
    return np.concatenate([np.geomspace(min(10, frequencies[0] / 2), np.nextafter(frequencies[0], 0), 40),
                           np.geomspace(np.nextafter(frequencies[-1], sample_rate / 2), sample_rate / 2, 40)])


def response(frequencies, bands, sample_rate):
    frequencies = np.asarray(frequencies, dtype=float)
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies < 0) or np.any(frequencies > sample_rate / 2):
        raise ValueError("Filter response frequency outside 0–Nyquist")
    z = np.exp(-2j * np.pi * frequencies / sample_rate)
    result = np.zeros(len(frequencies))
    for band in bands:
        if not all(np.isfinite(value) for value in (band.frequency, band.gain, band.q)) or not 0 < band.frequency < sample_rate / 2 or band.q <= 0:
            raise ValueError("Invalid filter coefficients")
        amplitude = 10 ** (band.gain / 40)
        omega = 2 * np.pi * band.frequency / sample_rate
        alpha = np.sin(omega) / (2 * band.q)
        cosine = np.cos(omega)
        if band.kind == "Bell":
            numerator = (1 + alpha * amplitude) - 2 * cosine * z + (1 - alpha * amplitude) * z ** 2
            denominator = (1 + alpha / amplitude) - 2 * cosine * z + (1 - alpha / amplitude) * z ** 2
        elif band.kind == "Lo-shelf":
            root_alpha = 2 * np.sqrt(amplitude) * alpha
            numerator = amplitude * ((amplitude + 1) - (amplitude - 1) * cosine + root_alpha
                                     + 2 * ((amplitude - 1) - (amplitude + 1) * cosine) * z
                                     + ((amplitude + 1) - (amplitude - 1) * cosine - root_alpha) * z ** 2)
            denominator = ((amplitude + 1) + (amplitude - 1) * cosine + root_alpha
                           - 2 * ((amplitude - 1) + (amplitude + 1) * cosine) * z
                           + ((amplitude + 1) + (amplitude - 1) * cosine - root_alpha) * z ** 2)
        elif band.kind == "Hi-shelf":
            root_alpha = 2 * np.sqrt(amplitude) * alpha
            numerator = amplitude * ((amplitude + 1) + (amplitude - 1) * cosine + root_alpha
                                     - 2 * ((amplitude - 1) + (amplitude + 1) * cosine) * z
                                     + ((amplitude + 1) + (amplitude - 1) * cosine - root_alpha) * z ** 2)
            denominator = ((amplitude + 1) - (amplitude - 1) * cosine + root_alpha
                           + 2 * ((amplitude - 1) - (amplitude + 1) * cosine) * z
                           + ((amplitude + 1) - (amplitude - 1) * cosine - root_alpha) * z ** 2)
        else:
            raise ValueError("Unsupported filter type")
        result += 20 * np.log10(np.maximum(abs(numerator / denominator), 1e-20))
    return result


def fit_with_bass(frequencies, correction, sample_rate=48000, count=8, bass_db=0.0, bass_hz=105.0, filter_mode="auto"):
    """Fit measured neutral correction, then append the exact preference shelf.

    Preference never changes inference or its priors. The fixed shelf consumes
    one export slot; its response cancels when evaluating calibration fit error.
    """
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 1 <= count <= 32:
        raise ValueError("PEQ export needs 1–32 total bands")
    shelf = bass_shelf(bass_db, bass_hz)
    if bass_db and count < 2:
        raise ValueError("A nonzero bass shelf needs at least 2 bands (including one correction band)")
    bands, metrics = fit_peq(frequencies, correction, sample_rate, count - int(bass_db != 0), filter_mode)
    metrics["correction_outside_range_peak_db"] = metrics["outside_range_peak_db"]
    if bass_db:
        bands.append(shelf)
    metrics["bands"] = [asdict(band) for band in bands]
    metrics["bass_preference_db"] = float(bass_db)
    metrics["bass_shelf_hz"] = float(bass_hz)
    # The deliberate preference is also present outside the measured range.
    guards = _outside_grid(frequencies, sample_rate)
    metrics["outside_range_peak_db"] = float(np.max(abs(response(guards, bands, sample_rate))))
    return bands, metrics


def _unpack(parameters, kinds):
    return [Band(float(2 ** center), float(gain), float(np.exp(q)), kind)
            for (center, gain, q), kind in zip(parameters.reshape((-1, 3)), kinds)]


def _shelf_bounds(frequencies):
    """Require in-range observations on both sides and distinct shelf roles."""
    bounds = {"Bell": (frequencies[0], frequencies[-1]),
              "Lo-shelf": (frequencies[1], min(frequencies[-2], 1000.0)),
              "Hi-shelf": (max(frequencies[1], 2000.0), frequencies[-2])}
    return {kind: (float(low), float(high)) for kind, (low, high) in bounds.items() if low < high}


def _fit_error(rendered, desired, weights, x):
    error = rendered - weights @ rendered - desired
    contour = np.diff(error) / np.sqrt(np.diff(x) * (x[-1] - x[0]))
    return error, np.concatenate([np.sqrt(weights) * error, 0.12 * contour])


def _residual_seed(kinds, bounds, grid, desired, weights, x, sample_rate):
    """Place filters where they explain the remaining curve, rather than uniformly."""
    templates = {}
    for kind in set(kinds):
        low, high = bounds[kind]
        templates[kind] = [Band(float(center), 1.0, q, kind)
                           for center in np.geomspace(low, high, 24 if kind == "Bell" else 12)
                           for q in ((0.5, 1.0, 2.0, 4.0) if kind == "Bell" else (float(1 / np.sqrt(2)),))]
    basis = {kind: np.array([_fit_error(response(grid, [band], sample_rate), np.zeros(len(grid)),
                                       weights, x)[1] for band in bands])
             for kind, bands in templates.items()}
    remaining = desired.copy()
    parameters = []
    for kind in kinds:
        target = -_fit_error(np.zeros(len(grid)), remaining, weights, x)[1]
        columns = basis[kind]
        norms = np.sum(columns ** 2, axis=1)
        products = columns @ target
        gains = np.clip(products / np.maximum(norms, 1e-12), -6, 6)
        index = int(np.argmax(2 * gains * products - gains ** 2 * norms))
        template = templates[kind][index]
        band = Band(template.frequency, float(gains[index]), template.q, kind)
        parameters.append([np.log2(band.frequency), band.gain, np.log(band.q)])
        rendered = response(grid, [band], sample_rate)
        remaining -= rendered - weights @ rendered
    return np.asarray(parameters).ravel()


def _optimize(parameters, kinds, bounds, residual):
    lower, upper = [], []
    for kind in kinds:
        low, high = bounds[kind]
        q_low, q_high = (0.4, 8.0) if kind == "Bell" else (1 / np.sqrt(2), 1 / np.sqrt(2))
        lower.extend([np.log2(low), -6.0, np.log(q_low)])
        upper.extend([np.log2(high), 6.0, np.log(q_high)])
    lower, upper = np.asarray(lower), np.asarray(upper)
    parameters = np.clip(parameters, lower, upper)
    active = np.flatnonzero(lower < upper)
    damping = 0.01
    current = residual(parameters)
    score = float(current @ current)
    for _ in range(70):
        columns = []
        for i in active:
            delta = np.zeros(len(parameters))
            delta[i] = 0.001
            columns.append((residual(parameters + delta) - residual(parameters - delta)) / 0.002)
        jacobian = np.array(columns).T
        step = np.linalg.solve(jacobian.T @ jacobian + damping * np.eye(len(active)), -jacobian.T @ current)
        candidate = parameters.copy()
        candidate[active] += step
        candidate = np.clip(candidate, lower, upper)
        candidate_residual = residual(candidate)
        candidate_score = float(candidate_residual @ candidate_residual)
        if candidate_score < score:
            improvement = score - candidate_score
            parameters, current, score = candidate, candidate_residual, candidate_score
            damping = max(1e-6, damping / 2)
            if improvement < 1e-8:
                break
        else:
            damping = min(1e6, damping * 5)
            if damping >= 1e6:
                break
    bands = _unpack(parameters, kinds)
    for band in bands:
        # Undo log/exp roundoff at an exact frequency bound (notably 16 kHz).
        band.frequency = float(np.clip(band.frequency, *bounds[band.kind]))
    return bands, score


def fit_peq(frequencies, correction, sample_rate=48000, count=8, filter_mode="auto"):
    frequencies = np.asarray(frequencies, dtype=float)
    correction = np.asarray(correction, dtype=float)
    if (isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 1 <= count <= 32
            or frequencies.ndim != 1 or len(frequencies) < 4 or correction.shape != frequencies.shape):
        raise ValueError("PEQ fit needs 4+ points and 1–32 bands")
    if filter_mode not in ("auto", "bells"):
        raise ValueError("Filter mode must be auto or bells")
    if (not isinstance(sample_rate, (int, float, np.integer, np.floating)) or isinstance(sample_rate, bool)
            or not np.isfinite(sample_rate) or sample_rate <= 0
            or np.any(~np.isfinite(correction)) or np.any(~np.isfinite(frequencies))
            or np.any(frequencies <= 0) or np.any(np.diff(frequencies) <= 0)
            or frequencies[-1] >= sample_rate / 2):
        raise ValueError("Invalid PEQ target curve")
    grid = np.geomspace(frequencies[0], frequencies[-1], 192)
    x = np.log(grid)
    desired = np.interp(x, np.log(frequencies), correction)
    span = x[-1] - x[0]
    weights = np.ones(len(grid))
    weights[[0, -1]] = 0.5
    weights /= weights.sum()
    desired -= weights @ desired
    guards = _outside_grid(frequencies, sample_rate)
    bounds = _shelf_bounds(frequencies)
    evaluated = []

    def fit(kinds, seed, label):
        def residual(parameters):
            trial = _unpack(parameters, kinds)
            shape = _fit_error(response(grid, trial, sample_rate), desired, weights, x)[1]
            outside = response(guards, trial, sample_rate)
            return np.concatenate([shape, 0.025 * outside / np.sqrt(len(guards)),
                                   0.004 * parameters.reshape((-1, 3))[:, 1]])

        trial, score = _optimize(seed, kinds, bounds, residual)
        shape = _fit_error(response(grid, trial, sample_rate), desired, weights, x)[1]
        shape_score = float(shape @ shape)
        outside_peak = float(np.max(abs(response(guards, trial, sample_rate))))
        evaluated.append({"topology": label, "objective": score,
                          "curve_contour_objective": shape_score, "outside_range_peak_db": outside_peak})
        return trial, score, shape_score, outside_peak

    kinds = ["Bell"] * count
    legacy_seed = np.array([[center, 0.0, np.log(1.5)]
                            for center in np.linspace(np.log2(frequencies[0]), np.log2(frequencies[-1]), count)]).ravel()
    bands, score, shape_score, outside_peak = fit(kinds, legacy_seed, "bells-uniform")
    if score > 1e-8:
        try:
            seed = _residual_seed(kinds, bounds, grid, desired, weights, x, sample_rate)
            trial, trial_score, trial_shape, trial_outside = fit(kinds, seed, "bells-residual")
            if trial_score < score:
                bands, score, shape_score, outside_peak = trial, trial_score, trial_shape, trial_outside
        except np.linalg.LinAlgError:
            evaluated.append({"topology": "bells-residual", "error": "solver failed; retained fallback"})
    bell_score = score
    bell_shape_score = shape_score
    outside_limit = max(6.0, outside_peak)
    selected = "bells"
    if filter_mode == "auto" and score > 1e-8:
        topologies = [[kind] for kind in ("Lo-shelf", "Hi-shelf") if kind in bounds]
        if count >= 2 and all(kind in bounds for kind in ("Lo-shelf", "Hi-shelf")):
            topologies.append(["Lo-shelf", "Hi-shelf"])
        for shelves in topologies:
            kinds = shelves + ["Bell"] * (count - len(shelves))
            label = "+".join(shelves) + "+bells" if count > len(shelves) else "+".join(shelves)
            try:
                seed = _residual_seed(kinds, bounds, grid, desired, weights, x, sample_rate)
                trial, trial_score, trial_shape, trial_outside = fit(kinds, seed, label)
            except np.linalg.LinAlgError:
                evaluated.append({"topology": label, "error": "solver failed; retained fallback"})
                continue
            # Compare actual in-band approximation, rather than assuming an
            # unmeasured shelf plateau ought to return to zero. Tail penalties
            # still guide optimization; explicit tail/headroom bounds gate use.
            if (trial_shape < shape_score - max(1e-6, 0.01 * shape_score)
                    and trial_outside <= outside_limit + 1e-6
                    and headroom({"both": trial}, sample_rate) >= -36):
                bands, score, shape_score, selected = trial, trial_score, trial_shape, label
    bands.sort(key=lambda band: band.frequency)
    rendered = response(grid, bands, sample_rate)
    error = rendered - weights @ rendered - desired
    metrics = {"shape_rmse_db": float(np.sqrt(integrate(error ** 2, x) / span)),
               "max_shape_error_db": float(np.max(abs(error))),
               "accumulated_contour_error_db": float(np.sum(abs(np.diff(error)))),
               "outside_range_peak_db": float(np.max(abs(response(guards, bands, sample_rate)))),
               "bands": [asdict(band) for band in bands], "filter_mode": filter_mode,
               "selected_topology": selected, "fit_objective": score,
               "bell_fit_objective": bell_score, "curve_contour_objective": shape_score,
               "bell_curve_contour_objective": bell_shape_score,
               "correction_outside_limit_db": outside_limit, "candidates": evaluated}
    return bands, metrics


def headroom(bands_by_channel, sample_rate):
    grid = np.geomspace(10, sample_rate * 0.499999, 8192)
    maximum = 0.0
    for bands in bands_by_channel.values():
        combined = np.unique(np.concatenate([grid, [band.frequency for band in bands]]))
        maximum = max(maximum, float(response(combined, bands, sample_rate).max()))
    return -(maximum + 1.0)


def easyeffects_preset(bands_by_channel, sample_rate):
    if set(bands_by_channel) not in ({"both"}, {"left", "right"}):
        raise ValueError("Export needs both channels together or separate left and right data")
    split = "both" not in bands_by_channel
    count = max(len(bands) for bands in bands_by_channel.values())
    if not 1 <= count <= 32:
        raise ValueError("Easy Effects supports 1–32 PEQ bands")
    input_gain = headroom(bands_by_channel, sample_rate)
    if input_gain < -36:
        raise ValueError("Combined filter boost needs more attenuation than this preset supports")
    equalizer = {"bypass": False, "input-gain": input_gain, "output-gain": 0.0,
                 "mode": "IIR", "decramp": "Off", "split-channels": split,
                 "balance": 0.0, "pitch-left": 0.0, "pitch-right": 0.0, "num-bands": count}
    for channel in ("left", "right"):
        bands = bands_by_channel[channel] if split else bands_by_channel["both"]
        values = {}
        for i in range(count):
            band = bands[i] if i < len(bands) else Band(1000.0, 0.0, 1.0)
            gain_limit = 12 if band.kind == "Lo-shelf" else 6
            if band.kind not in ("Bell", "Lo-shelf", "Hi-shelf") or not -gain_limit <= band.gain <= gain_limit or not 0.4 <= band.q <= 8 or not 20 <= band.frequency <= MAX_HZ:
                raise ValueError("Filter is outside export bounds")
            values[f"band{i}"] = {"type": band.kind if i < len(bands) else "Off",
                                  "mode": "APO (DR)", "slope": "x1", "solo": False,
                                  "mute": False, "frequency": band.frequency,
                                  "gain": band.gain, "q": band.q, "width": 4.0}
        equalizer[channel] = values
    return {"output": {"blocklist": [], "plugins_order": ["equalizer#0"], "equalizer#0": equalizer}}
