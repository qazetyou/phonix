"""Adaptive Bayesian pair matching and robust, globally constrained fitting."""

from dataclasses import dataclass
import math

import numpy as np

from .model import Reference, integrate, log_weights

NOISE_DB = 1.5
EQUAL_ZONE_DB = 0.75
LAPSE = 0.04
MANUAL_SIGMA = 1.5
MAX_PAIR_OCTAVES = 1.0


def nearby(a, b):
    """Allow at most one octave between tones, including either endpoint."""
    return abs(math.log2(b / a)) <= MAX_PAIR_OCTAVES + 1e-12


def new_channel(settings):
    frequencies = np.geomspace(settings.min_hz, settings.max_hz, settings.initial_points)
    # Midrange and upper-treble anchors connect the corresponding regions.
    for anchor in (1000.0, 12500.0):
        if settings.min_hz < anchor < settings.max_hz and anchor not in frequencies:
            available = [index for index in range(1, len(frequencies) - 1) if frequencies[index] not in (1000.0, 12500.0)]
            index = min(available, key=lambda index: abs(np.log(frequencies[index] / anchor)))
            frequencies[index] = anchor
            frequencies.sort()
    return {"frequencies": frequencies.tolist(),
            "observations": [], "blocked": [], "inputs": 0}


def validate_channel(channel, settings):
    from .workflow import validate_workflow

    if not isinstance(channel, dict):
        raise ValueError("Invalid channel data")
    frequencies = channel["frequencies"]
    if not isinstance(frequencies, list) or not settings.initial_points <= len(frequencies) <= settings.max_points:
        raise ValueError("Invalid frequency grid")
    if any(isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f) for f in frequencies):
        raise ValueError("Invalid frequency value")
    if sorted(set(frequencies)) != frequencies or frequencies[0] != settings.min_hz or frequencies[-1] != settings.max_hz:
        raise ValueError("Invalid frequency order or limits")
    observations = channel["observations"]
    if not isinstance(observations, list) or len(observations) > 10000:
        raise ValueError("Invalid observations")
    if channel["inputs"] != len(observations):
        raise ValueError("Judgment count does not match observations")
    for observation in observations:
        if observation["a"] not in frequencies or observation["b"] not in frequencies or observation["a"] >= observation["b"]:
            raise ValueError("Observation has invalid frequency pair")
        offset = observation["offset"]
        if not isinstance(offset, (int, float)) or not math.isfinite(offset) or abs(offset) > settings.offset_limit + 1e-6:
            raise ValueError("Observation has invalid level difference")
        if observation["kind"] not in ("manual", "guided") or observation["answer"] not in ("first", "equal", "second"):
            raise ValueError("Observation has invalid response")
        if observation["kind"] == "manual" and observation["answer"] != "equal":
            raise ValueError("Manual matches must be equality observations")
        if "playback" in observation:
            validate_playback(observation["playback"], offset, settings)
    blocked = channel["blocked"]
    if not isinstance(blocked, list) or any(not isinstance(pair, list) or len(pair) != 2 or pair[0] not in frequencies or pair[1] not in frequencies or pair[0] >= pair[1] for pair in blocked):
        raise ValueError("Invalid skipped-pair data")
    validate_workflow(channel, settings)


def probabilities(theta, offset):
    """Response probabilities for tone B at offset dB relative to tone A."""
    difference = (offset - np.asarray(theta)) / NOISE_DB
    zone = EQUAL_ZONE_DB / NOISE_DB
    first = 1 / (1 + np.exp(np.clip(difference + zone, -60, 60)))
    second = 1 / (1 + np.exp(np.clip(-difference + zone, -60, 60)))
    equal = np.maximum(0, 1 - first - second)
    return (1 - LAPSE) * np.array([first, equal, second]) + LAPSE / 3


class Pair:
    def __init__(self, a, b, observations, limit):
        self.a, self.b = a, b
        self.observations = observations
        self.grid = np.linspace(-limit, limit, 241)
        logp = np.zeros(len(self.grid))
        for observation in observations:
            if observation["kind"] == "manual":
                logp -= 0.5 * ((self.grid - observation["offset"]) / MANUAL_SIGMA) ** 2
            else:
                index = ("first", "equal", "second").index(observation["answer"])
                logp += np.log(probabilities(self.grid, observation["offset"])[index])
        logp -= logp.max()
        self.posterior = np.exp(logp)
        self.posterior /= self.posterior.sum()
        self.mean = float(self.posterior @ self.grid)
        self.variance = max(0.04, float(self.posterior @ (self.grid - self.mean) ** 2))
        manual = [o["offset"] for o in observations if o["kind"] == "manual"]
        if len(manual) > 1:
            self.variance = max(self.variance, float(np.var(manual, ddof=1)) / len(manual))
        self.boundary = bool(self.posterior[:4].sum() + self.posterior[-4:].sum() > 0.10)

    def next_offset(self):
        """One-step expected entropy reduction for a guided response."""
        cumulative = self.posterior.cumsum()
        candidates = np.unique(np.interp([0.2, 0.35, 0.5, 0.65, 0.8], cumulative, self.grid))
        scores = []
        for offset in candidates:
            likelihood = probabilities(self.grid, offset)
            joint = likelihood * self.posterior
            outcome = joint.sum(axis=1)
            posterior = joint / outcome[:, None]
            entropy = -np.sum(posterior * np.log(np.maximum(posterior, 1e-30)), axis=1)
            scores.append(float(outcome @ entropy))
        return float(candidates[int(np.argmin(scores))])


def pairs_for(channel, settings):
    grouped = {}
    observations = channel["observations"]
    if settings.calibration_method == "regional-v2":
        from .regional import current_observations

        observations = current_observations(channel)
    for observation in observations:
        grouped.setdefault((observation["a"], observation["b"]), []).append(observation)
    return {key: Pair(*key, observations, settings.offset_limit) for key, observations in grouped.items()}


@dataclass
class Fit:
    frequencies: np.ndarray
    matches: np.ndarray
    covariance: np.ndarray
    connected: bool
    cycles: int
    drift_rms: float | None
    shape95: float
    max95: float
    degrees: np.ndarray
    residuals: dict
    boundary_pairs: int
    regional: dict | None = None

    def ready(self, settings):
        if self.regional is not None and self.regional.get("skipped_regions"):
            return self.regional["balance_checks_met"]
        return ((self.regional is None or self.regional["shapes_frozen"] and not self.regional["review_regions"])
                and self.connected and self.cycles >= 2 and np.all(self.degrees >= 2)
                and self.shape95 <= settings.shape_tolerance
                and self.drift_rms is not None and self.drift_rms <= settings.drift_tolerance)


def reference_base_level(frequency, fit, settings, reference):
    """Place the learned matching shape around the saved 1 kHz reference.

    Outside the sampled range, the fixed baseline supplies the anchor rather
    than extrapolating an unmeasured device response.
    """
    x = np.log(fit.frequencies)
    if fit.frequencies[0] <= 1000 <= fit.frequencies[-1]:
        anchor = float(np.interp(np.log(1000), x, fit.matches))
    else:
        prior = reference.expected_matches(np.append(fit.frequencies, 1000.0))
        anchor = float(prior[-1] + log_weights(fit.frequencies) @ (fit.matches - prior[:-1]))
    matched = float(np.interp(np.log(frequency), x, fit.matches))
    return settings.reference_level + matched - anchor


def validate_playback(playback, offset, settings):
    """Audit actual peaks without changing the canonical B-minus-A likelihood."""
    if not isinstance(playback, dict) or playback.get("method") != "adaptive-v1":
        raise ValueError("Invalid comparison playback metadata")
    values = [playback.get(key) for key in ("a_dbfs", "b_dbfs")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or not -78 <= v <= settings.max_level for v in values):
        raise ValueError("Comparison playback levels exceed configured bounds")
    if abs(values[1] - values[0] - offset) > 1e-6:
        raise ValueError("Comparison playback levels disagree with the saved offset")


def fit_channel(channel, settings, reference=None):
    if settings.calibration_method == "regional-v2":
        from .regional import fit_regional

        return fit_regional(channel, settings, reference or Reference())
    return _fit_global(channel, settings, reference)


def _fit_global(channel, settings, reference=None):
    reference = reference or Reference()
    frequencies = np.asarray(channel["frequencies"])
    n = len(frequencies)
    indices = {float(f): i for i, f in enumerate(frequencies)}
    all_pairs = pairs_for(channel, settings)
    # A posterior pressed against the permitted level range is a censored
    # comparison, not a reliable equality measurement. Retain its observations
    # for audit, but use alternative links to estimate the curve.
    excluded = {key for key, pair in all_pairs.items() if pair.boundary and len(pair.observations) >= 6}
    pairs = {key: pair for key, pair in all_pairs.items() if key not in excluded}
    rows, values, variances = [], [], []
    degrees = np.zeros(n, dtype=int)
    keys = list(pairs)
    for key, pair in pairs.items():
        row = np.zeros(n)
        i, j = indices[key[0]], indices[key[1]]
        row[i], row[j] = -1, 1
        rows.append(row)
        values.append(pair.mean)
        variances.append(pair.variance)
        degrees[i] += 1
        degrees[j] += 1
    design = np.array(rows).reshape((-1, n))
    rank = int(np.linalg.matrix_rank(design)) if len(rows) else 0
    connected = rank == n - 1
    values = np.asarray(values)
    weights = 1 / np.asarray(variances) if variances else np.array([])
    mean = reference.expected_matches(frequencies)
    integration_weights = log_weights(frequencies)
    projection = np.eye(n) - np.ones((n, 1)) * integration_weights
    covariance = projection @ projection.T * 900.0
    if len(rows):
        robust = weights.copy()
        for _ in range(6):
            precision = design.T @ (robust[:, None] * design)
            rhs = design.T @ (robust * values)
            if connected:
                covariance = np.linalg.pinv(precision, hermitian=True)
                mean = covariance @ rhs
            else:
                covariance = np.linalg.inv(precision + np.eye(n) / 900.0)
                mean = covariance @ (rhs + reference.expected_matches(frequencies) / 900.0)
            mean = projection @ mean
            covariance = projection @ covariance @ projection.T
            residual = design @ mean - values
            standardized = abs(residual) / np.sqrt(variances)
            robust = weights * np.minimum(1, 2.5 / np.maximum(standardized, 1e-9))
        cycles = max(0, len(rows) - rank)
        if cycles:
            covariance *= max(1.0, float(np.sum(robust * residual ** 2)) / cycles)
        drift = float(np.sqrt(np.mean(residual ** 2))) if connected and cycles else None
        residuals = {key: float(value) for key, value in zip(keys, residual)}
    else:
        cycles, drift, residuals = 0, None, {}
    variance = np.maximum(0, np.diag(covariance))
    x = np.log(frequencies)
    shape95 = 1.96 * math.sqrt(integrate(variance, x) / (x[-1] - x[0]))
    return Fit(frequencies, mean, covariance, connected, cycles, drift, shape95,
               float(1.96 * np.sqrt(variance.max())), degrees, residuals,
               len(excluded))


def select_pair(channel, settings, rng, reference=None, allowed_pairs=None):
    reference = reference or Reference()
    fit = fit_channel(channel, settings, reference)
    pairs = pairs_for(channel, settings)
    blocked = {tuple(pair) for pair in channel["blocked"]}
    candidates = []
    n = len(fit.frequencies)
    integration_weights = log_weights(fit.frequencies)
    for i in range(n):
        for j in range(i + 1, n):
            key = (float(fit.frequencies[i]), float(fit.frequencies[j]))
            if allowed_pairs is not None and key not in allowed_pairs:
                continue
            if not nearby(*key) or key in blocked:
                continue
            pair = pairs.get(key)
            if pair is not None and pair.boundary and len(pair.observations) >= 6:
                continue
            predicted = fit.matches[j] - fit.matches[i]
            # Finish checking an existing uncertain link before censoring it;
            # otherwise its few initial judgments could leave a biased edge
            # in the fit while its predicted offset prevents further checks.
            if pair is None and abs(predicted) > settings.offset_limit * 0.95 and abs(j - i) > 1:
                continue
            difference = np.zeros(n)
            difference[i], difference[j] = -1, 1
            effect = fit.covariance @ difference
            variance = float(difference @ fit.covariance @ difference)
            # A-optimal one-step approximation: expected reduction in total
            # shape variance from an additional noisy difference measurement.
            score = float(effect @ (integration_weights * effect)) / max(variance + MANUAL_SIGMA ** 2, 1e-9)
            if fit.degrees[i] == 0 or fit.degrees[j] == 0:
                score += 1e5 / (j - i)
            elif not fit.connected:
                score += variance
            if pair is None and (fit.degrees[i] < 2 or fit.degrees[j] < 2):
                score *= 1.5
            if pair is not None:
                score *= 1 + min(4, abs(fit.residuals.get(key, 0)) / settings.drift_tolerance)
                if pair.boundary:
                    score *= 1.5
            # The hard octave limit bounds difficulty. Keep this preference
            # modest so useful local cross-checks still earn listening time.
            score /= 1 + 0.25 * math.log2(key[1] / key[0])
            candidates.append((score, rng.random(), key))
    if not candidates:
        raise ValueError("No usable pairs remain within one octave. Review skipped pairs and level limits, or create a narrower-range profile")
    return max(candidates)[2]


def prepare_local_grid(channel, settings):
    """Add bridges for connected local comparisons and redundant coverage.

    Plan on a copy so a restrictive point cap never leaves a partial grid.
    Existing observations, skipped pairs and frequencies remain intact.
    """
    frequencies = list(channel["frequencies"])
    unavailable = {tuple(pair) for pair in channel["blocked"]}
    unavailable.update(key for key, pair in pairs_for(channel, settings).items()
                       if pair.boundary and len(pair.observations) >= 6)
    while True:
        n = len(frequencies)
        neighbours = [set() for _ in frequencies]
        for i, a in enumerate(frequencies):
            for j in range(i + 1, n):
                if nearby(a, frequencies[j]) and (a, frequencies[j]) not in unavailable:
                    neighbours[i].add(j)
                    neighbours[j].add(i)
        components = [-1] * n
        for i in range(n):
            if components[i] != -1:
                continue
            pending = [i]
            components[i] = i
            while pending:
                node = pending.pop()
                for other in neighbours[node]:
                    if components[other] == -1:
                        components[other] = i
                        pending.append(other)
        # First bridge separate components; then give endpoints/isolated nodes
        # a second local partner so coverage need not rely on a simple chain.
        gaps = [i for i in range(n - 1) if components[i] != components[i + 1]]
        if not gaps:
            gaps = sorted({j for i in range(n) if len(neighbours[i]) < 2
                           for j in (i - 1, i) if 0 <= j < n - 1})
            if not gaps:
                # Supply a third nearby tone to cross-check each possible link.
                # A chain between two endpoint loops cannot check errors in
                # its interior, even though every node has two partners.
                gaps = sorted({k for i in range(n) for j in neighbours[i] if j > i
                               and not neighbours[i].intersection(neighbours[j])
                               for k in range(i, j)})
            cycles = sum(map(len, neighbours)) // 2 - n + 1
            if not gaps and cycles < 2:
                gaps = list(range(n - 1))
        if not gaps:
            break
        if n >= settings.max_points:
            raise ValueError("The frequency grid needs more points for one-octave comparisons and local cross-checks. "
                             f"Maximum frequency points is {settings.max_points}; increase it (up to 32) or create a narrower-range profile. "
                             "Existing frequencies and judgments were kept.")
        index = max(gaps, key=lambda i: math.log2(frequencies[i + 1] / frequencies[i]))
        midpoint = math.sqrt(frequencies[index] * frequencies[index + 1])
        if not frequencies[index] < midpoint < frequencies[index + 1]:
            raise ValueError("Cannot add a distinct local frequency; review skipped pairs or create a narrower-range profile")
        frequencies.insert(index + 1, midpoint)
    if frequencies == channel["frequencies"]:
        return False
    channel["frequencies"] = frequencies
    return True


def refine(channel, settings, reference=None):
    """Add a midpoint for reachable cross-checks or contour detail."""
    if settings.calibration_method == "regional-v2":
        from .regional import prepare_grid

        return prepare_grid(channel, settings)
    if prepare_local_grid(channel, settings):
        return True
    if len(channel["frequencies"]) >= settings.max_points:
        return False
    reference = reference or Reference()
    fit = fit_channel(channel, settings, reference)
    if not fit.connected or fit.cycles < 2 or fit.max95 > 4:
        return False
    x = np.log2(fit.frequencies)
    blocked = {tuple(pair) for pair in channel["blocked"]}
    coverage = []
    for i in range(len(x)):
        if fit.degrees[i] >= 2:
            continue
        reachable = [j for j in range(len(x)) if j != i
                     and nearby(fit.frequencies[i], fit.frequencies[j])
                     and abs(fit.matches[j] - fit.matches[i]) < settings.offset_limit * 0.90
                     and tuple(sorted((float(fit.frequencies[i]), float(fit.frequencies[j])))) not in blocked]
        if len(reachable) >= 2:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(x) and abs(x[j] - x[i]) >= 1 / 6:
                coverage.append((abs(x[j] - x[i]), 2 ** ((x[i] + x[j]) / 2)))
    if coverage:
        midpoint = max(coverage)[1]
        channel["frequencies"] = sorted(channel["frequencies"] + [float(midpoint)])
        return True
    correction = reference.correction(fit.frequencies, fit.matches)
    slopes = np.diff(correction) / np.diff(x)
    candidates = []
    for i in range(len(x) - 1):
        if x[i + 1] - x[i] < 1 / 3:
            continue
        midpoint = 2 ** ((x[i] + x[i + 1]) / 2)
        curve = reference.target([fit.frequencies[i], midpoint, fit.frequencies[i + 1]])
        target_error = abs(curve[1] - (curve[0] + curve[2]) / 2)
        neighbours = slopes[max(0, i - 1):min(len(slopes), i + 2)]
        estimated_error = float(np.ptp(neighbours)) * (x[i + 1] - x[i]) / 8
        candidates.append((max(target_error, estimated_error), midpoint))
    if not candidates:
        return False
    score, midpoint = max(candidates)
    if score <= settings.shape_tolerance / 2:
        return False
    channel["frequencies"] = sorted(channel["frequencies"] + [float(midpoint)])
    return True


def add_observation(channel, settings, a, b, offset, kind, answer="equal", playback=None):
    if a >= b or a not in channel["frequencies"] or b not in channel["frequencies"]:
        raise ValueError("Invalid comparison pair")
    if not math.isfinite(offset) or abs(offset) > settings.offset_limit + 1e-6:
        raise ValueError("Comparison level is outside configured limits")
    if kind not in ("manual", "guided") or answer not in ("first", "equal", "second") or (kind == "manual" and answer != "equal"):
        raise ValueError("Invalid comparison response")
    if channel["inputs"] >= 10000:
        raise ValueError("Maximum stored judgment count reached")
    observation = {"a": float(a), "b": float(b), "offset": float(offset), "kind": kind, "answer": answer}
    if playback is not None:
        validate_playback(playback, offset, settings)
        observation["playback"] = dict(playback)
    channel["observations"].append(observation)
    channel["inputs"] += 1
