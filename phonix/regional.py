"""Independent regional shapes followed by gain-only boundary fitting.

Freeze checkpoints reference retained evidence, rather than mutable curve
snapshots. Boundary evidence after the current balance checkpoint adjusts only
one constant per region. Its covariance carries the frozen shape uncertainty.
"""

import copy
from dataclasses import replace
import math

import numpy as np

from .engine import (Fit, _fit_global, pairs_for, prepare_local_grid, select_pair)
from .model import Reference, log_weights
from .workflow import (_checks, _geometry, _regional_pending, grouped_refinement,
                       mark_range, range_description, regions)

SHAPE_ORDER = ("midrange", "bass", "treble")
STAGES = (*SHAPE_ORDER, "balance", "verify", "review")


def validate_state(channel, settings):
    state = channel["workflow"]
    if (settings is None or settings.calibration_method != "regional-v2"
            or type(state.get("version")) is not int or state["version"] != 2
            or state.get("stage") not in STAGES
            or type(state.get("started_at")) is not int
            or not 0 <= state["started_at"] <= channel["inputs"]):
        raise ValueError("Invalid regional pass state")
    groups = regions(channel)
    frozen = state.get("frozen")
    if (not isinstance(frozen, dict) or not set(frozen) <= set(groups)
            or any(type(at) is not int or not 0 <= at <= channel["inputs"] for at in frozen.values())):
        raise ValueError("Invalid frozen regional checkpoints")
    skipped = state.get("skipped", {})
    if (not isinstance(skipped, dict) or not set(skipped) <= set(frozen)
            or any(type(at) is not int or at != frozen[name] for name, at in skipped.items())):
        raise ValueError("Invalid skipped regional checkpoints")
    for key in ("balance_at", "verify_at", "reopened_at"):
        if key in state and (type(state[key]) is not int or not 0 <= state[key] <= channel["inputs"]):
            raise ValueError("Invalid regional balance checkpoint")
    if state["stage"] in ("balance", "verify", "review"):
        if set(frozen) != set(groups) or "balance_at" not in state or any(at > state["balance_at"] for at in frozen.values()):
            raise ValueError("Regional balance needs all frozen shapes")
    elif state["stage"] in frozen or "balance_at" in state:
        raise ValueError("Shape pass cannot have a frozen active region or balance checkpoint")
    if state["stage"] == "verify" and ("verify_at" not in state or state["verify_at"] < state["balance_at"]):
        raise ValueError("Verification needs a regional balance checkpoint")
    if "review_regions" in state and (not isinstance(state["review_regions"], list)
                                      or not set(state["review_regions"]) <= set(groups)):
        raise ValueError("Invalid regions flagged for review")
    for observation in channel["observations"]:
        if "reversed_order" in observation and type(observation["reversed_order"]) is not bool:
            raise ValueError("Invalid stored playback order")


def current_observations(channel):
    membership = {f: name for name, fs in regions(channel).items() for f in fs}
    at = channel.get("workflow", {}).get("balance_at")
    skipped = channel.get("workflow", {}).get("skipped", {})
    return [o for index, o in enumerate(channel["observations"])
            if (membership[o["a"]] == membership[o["b"]] and membership[o["a"]] not in skipped
                or membership[o["a"]] != membership[o["b"]] and at is not None and index >= at)]


def skip_shapes(channel, settings, name=None):
    """Leave selected internal shapes uncorrected; retain all historical input."""
    if settings.calibration_method != "regional-v2":
        raise ValueError("Skipping local shapes requires a regional profile")
    groups = regions(channel)
    if len(groups) < 2:
        raise ValueError("This range has only one region, so there is no regional balance to perform. Keep local calibration or use a wider range")
    state = copy.deepcopy(channel.get("workflow", {"version": 2,
                          "stage": next(n for n in SHAPE_ORDER if n in groups),
                          "started_at": channel["inputs"], "frozen": {}}))
    if state["stage"] not in SHAPE_ORDER or name is not None and name != state["stage"]:
        raise ValueError("Only an active local shape pass can be skipped")
    selected = [name] if name is not None else [n for n in groups if n not in state["frozen"]]
    skipped = state.setdefault("skipped", {})
    for region in selected:
        state["frozen"][region] = skipped[region] = channel["inputs"]
    remaining = [n for n in SHAPE_ORDER if n in groups and n not in state["frozen"]]
    state["stage"] = remaining[0] if remaining else "balance"
    state["started_at"] = channel["inputs"]
    state.pop("reopened_at", None)
    if not remaining:
        state["balance_at"] = channel["inputs"]
    channel["workflow"] = state


def local_fit(channel, settings, name, reference, cutoff=None):
    fs = regions(channel)[name]
    observations = [o for o in channel["observations"][:cutoff]
                    if o["a"] in fs and o["b"] in fs]
    sub = {"frequencies": fs, "observations": observations, "blocked": [], "inputs": len(observations)}
    return _fit_global(sub, replace(settings, calibration_method="global-v1"), reference)


def local_ready(fit, settings):
    return (fit.connected and fit.cycles >= 2 and np.all(fit.degrees >= 2)
            and fit.shape95 <= settings.shape_tolerance / math.sqrt(2)
            and fit.drift_rms is not None and fit.drift_rms <= settings.drift_tolerance)


def prepare_grid(channel, settings):
    # Once a region is frozen, adding a point could change its shape and even
    # singleton-region membership. Reopening refits on the existing grid.
    if channel.get("workflow", {}).get("frozen"):
        return False
    planned = copy.deepcopy(channel)
    while True:
        before = list(planned["frequencies"])
        prepare_local_grid(planned, settings)
        for name, fs in regions(planned).items():
            capacity = settings.max_points - (len(planned["frequencies"]) - len(fs))
            sub = {"frequencies": fs, "observations": [o for o in planned["observations"] if o["a"] in fs and o["b"] in fs],
                   "blocked": [p for p in planned["blocked"] if p[0] in fs and p[1] in fs]}
            try:
                prepare_local_grid(sub, replace(settings, max_points=capacity))
            except ValueError as error:
                raise ValueError(f"{name.capitalize()} needs more within-region frequency points. Increase Maximum frequency points or narrow the range. Existing evidence was kept") from error
            planned["frequencies"] = sorted(set(planned["frequencies"]) | set(sub["frequencies"]))
        # Insertions can turn a singleton into its own region. Prepare again
        # until memberships and both local/global geometry are stable.
        if planned["frequencies"] == before:
            break
    if planned["frequencies"] == channel["frequencies"]:
        return False
    channel["frequencies"] = planned["frequencies"]
    return True


def fit_regional(channel, settings, reference):
    groups = regions(channel)
    names = list(groups)
    fs = np.asarray(channel["frequencies"])
    n, k = len(fs), len(names)
    indices = {float(f): i for i, f in enumerate(fs)}
    membership = {f: name for name, frequencies in groups.items() for f in frequencies}
    state = channel.get("workflow", {})
    frozen = state.get("frozen", {})
    skipped = state.get("skipped", {})
    shape, shape_cov = np.zeros(n), np.zeros((n, n))
    locals_, details = {}, {}
    for name, frequencies in groups.items():
        fit = local_fit(channel, settings, name, reference, frozen.get(name))
        locals_[name] = fit
        ix = [indices[f] for f in frequencies]
        if name in skipped:
            # This is an explicit no-local-correction constraint, not an
            # estimated headphone response. DF + proxy cancel at export.
            baseline = reference.expected_matches(frequencies)
            shape[ix] = baseline - log_weights(frequencies) @ baseline
        else:
            shape[ix], shape_cov[np.ix_(ix, ix)] = fit.matches, fit.covariance
        details[name] = {"frozen_at": frozen.get(name), "connected": fit.connected,
                         "skipped": name in skipped,
                         "shape_uncertainty_95_db": None if name in skipped else fit.shape95,
                         "cross_check_rms_db": None if name in skipped else fit.drift_rms,
                         "shape_checks_met": name not in skipped and bool(local_ready(fit, settings))}
    anchor = "midrange" if "midrange" in groups else names[0]
    variables = [name for name in names if name != anchor]
    association = np.array([[float(membership[f] == name) for name in variables] for f in fs]).reshape(n, k - 1)
    all_pairs = pairs_for(channel, settings)
    pairs = {key: p for key, p in all_pairs.items() if not (p.boundary and len(p.observations) >= 6)}
    bridges = {key: p for key, p in pairs.items() if membership[key[0]] != membership[key[1]]}
    keys = list(bridges)
    d = np.zeros((len(keys), n))
    for row, (a, b) in enumerate(keys):
        d[row, indices[a]], d[row, indices[b]] = -1, 1
    design = d @ association
    rank = int(np.linalg.matrix_rank(design)) if len(keys) else 0
    balance_connected = rank == k - 1
    prior = reference.expected_matches(fs)
    region_prior = {name: float(log_weights(groups[name]) @ prior[[indices[f] for f in groups[name]]]) for name in names}
    gains = np.array([region_prior[name] - region_prior[anchor] for name in variables])
    covariance = shape_cov.copy()
    if variables and balance_connected:
        values = np.array([bridges[key].mean for key in keys]) - d @ shape
        variances = np.array([bridges[key].variance for key in keys])
        # Shared shape uncertainty makes boundary observations correlated.
        effective = np.diag(variances) + d @ shape_cov @ d.T
        inverse = np.linalg.inv(effective)
        factors = np.ones(len(keys))
        for _ in range(6):
            weights = factors[:, None] * inverse * factors[None, :]
            gain_map = np.linalg.solve(design.T @ weights @ design, design.T @ weights)
            gains = gain_map @ values
            residual = design @ gains - values
            standardized = abs(residual) / np.sqrt(np.diag(effective))
            factors = np.sqrt(np.minimum(1, 2.5 / np.maximum(standardized, 1e-9)))
        # Propagate shapes and probe noise jointly. Within-region differences
        # cancel the constant gains, retaining exactly their frozen covariance.
        mapping = association @ gain_map
        shape_mapping = np.eye(n) - mapping @ d
        covariance = shape_mapping @ shape_cov @ shape_mapping.T + mapping @ np.diag(variances) @ mapping.T
    elif variables:
        covariance += association @ association.T * 900
    matches = shape + association @ gains
    projection = np.eye(n) - np.ones((n, 1)) * log_weights(fs)
    matches = projection @ matches
    covariance = projection @ covariance @ projection.T
    residuals = {key: float(matches[indices[key[1]]] - matches[indices[key[0]]] - pair.mean)
                 for key, pair in pairs.items()}
    drift = float(np.sqrt(np.mean(np.square(list(residuals.values()))))) if residuals else None
    # Do not dilute a bad boundary with many good local edges.
    bridge_drift = float(np.sqrt(np.mean([residuals[key] ** 2 for key in keys]))) if keys else 0.0
    local_drifts = [math.sqrt(np.mean([value ** 2 for key, value in residuals.items()
                                     if membership[key[0]] == membership[key[1]] == name]))
                    for name in names if any(membership[a] == membership[b] == name for a, b in residuals)]
    drift = max([drift or 0.0, bridge_drift, *local_drifts]) if residuals else None
    review = {name for name, fit in locals_.items() if name in frozen and name not in skipped and not local_ready(fit, settings)}
    for key, pair in pairs.items():
        a, b = [indices[f] for f in key]
        uncertainty = math.sqrt(max(0, shape_cov[a, a] + shape_cov[b, b] - 2 * shape_cov[a, b]) + pair.variance)
        enough = sum(o["kind"] == "manual" for o in pair.observations) >= 2 or len(pair.observations) >= 6
        if enough and abs(residuals[key]) > settings.drift_tolerance + 1.96 * uncertainty:
            review.update((membership[key[0]], membership[key[1]]))
    degree = np.zeros(n, dtype=int)
    for a, b in pairs:
        degree[indices[a]] += 1
        degree[indices[b]] += 1
    connected = not skipped and balance_connected and all(f.connected for f in locals_.values())
    cycles = sum(f.cycles for name, f in locals_.items() if name not in skipped) + max(0, len(keys) - k + 1)
    conditional95 = float(1.96 * np.sqrt(log_weights(fs) @ np.maximum(0, np.diag(covariance))))
    # Unknown skipped shapes must not acquire falsely narrow confidence
    # intervals just because we constrained their exported correction to flat.
    for name in skipped:
        ix = [indices[f] for f in groups[name]]
        local_projection = np.eye(len(ix)) - np.ones((len(ix), 1)) * log_weights(groups[name])
        covariance[np.ix_(ix, ix)] += local_projection @ local_projection.T * 900
    covariance = projection @ covariance @ projection.T
    variance = np.maximum(0, np.diag(covariance))
    shape95 = float(1.96 * np.sqrt(log_weights(fs) @ variance))
    offsets = {anchor: 0.0} | {name: float(gain) for name, gain in zip(variables, gains)}
    settled = set(frozen) == set(names) and all(name in skipped or local_ready(f, settings) for name, f in locals_.items())
    exportable = balance_connected and set(frozen) == set(names) and (not skipped or k > 1) and all(name in skipped or f.connected for name, f in locals_.items())
    diagnostics = {"shapes_frozen": settled,
                   "regions": details, "anchor": anchor, "matching_offsets_db": offsets,
                   "balance_connected": balance_connected, "boundary_cross_check_rms_db": bridge_drift,
                   "skipped_regions": sorted(skipped), "exportable": exportable,
                   "balance_uncertainty_95_db": conditional95,
                   "balance_checks_met": bool(settled and exportable and not review and conditional95 <= settings.shape_tolerance
                                              and drift is not None and drift <= settings.drift_tolerance),
                   "review_regions": sorted(review)}
    return Fit(fs, matches, covariance, connected, cycles, drift, shape95,
               float(1.96 * np.sqrt(variance.max())), degree, residuals, len(all_pairs) - len(pairs), diagnostics)


def verified_regional(channel, settings):
    state = channel.get("workflow", {})
    if state.get("version") != 2 or state.get("stage") != "verify":
        return False
    _, local, bridges, _, _ = _geometry(channel, settings)
    local = {name: pairs for name, pairs in local.items() if name not in state.get("skipped", {})}
    try:
        return not _checks(channel, local | bridges, 1, True)
    except ValueError:
        return False


def regional_progress(channel):
    state = channel.get("workflow", {})
    stage = state.get("stage")
    if stage in SHAPE_ORDER:
        fs = regions(channel)[stage]
        return f"{stage.capitalize()} shape ({fs[0]:.0f}–{fs[-1]:.0f} Hz)"
    if stage == "review":
        return "Shape review needed: " + ", ".join(state.get("review_regions", []))
    label = {"balance": "Regional balance | shapes frozen", "verify": "Verify shape and balance | shapes frozen"}.get(stage, "Not started")
    return label + " | " + range_description(channel) if range_description(channel) else label


def reopen_region(channel, settings, name):
    if settings.calibration_method != "regional-v2" or name not in regions(channel):
        raise ValueError("This profile has no such regional shape pass")
    state = channel.get("workflow")
    if state is None or name not in state.get("frozen", {}):
        raise ValueError("This region has not been frozen yet")
    frozen = dict(state["frozen"])
    del frozen[name]
    skipped = {n: at for n, at in state.get("skipped", {}).items() if n != name}
    channel["workflow"] = {"version": 2, "stage": name, "started_at": channel["inputs"],
                           "reopened_at": channel["inputs"], "frozen": frozen, "skipped": skipped}


def next_regional(channel, settings, rng, reference=None):
    reference = reference or Reference()
    groups, local, bridges, observed, _ = _geometry(channel, settings)
    if "workflow" not in channel:
        first = next(name for name in SHAPE_ORDER if name in groups)
        channel["workflow"] = {"version": 2, "stage": first, "started_at": channel["inputs"], "frozen": {}}
    state = channel["workflow"]
    while True:
        stage = state["stage"]
        if stage == "review":
            return None
        if stage in groups:
            fit = local_fit(channel, settings, stage, reference)
            if "reopened_at" in state:
                fresh = _checks(channel, {stage: local[stage]}, 2, True)
                if fresh:
                    _, _, candidates, orders = fresh[0]
                    a, b = select_pair(channel, settings, rng, reference, allowed_pairs=candidates)
                    order = next((value for value in (False, True) if value not in orders[(a, b)]), False)
                    return mark_range(channel, stage, (a, b, order))
            if local_ready(fit, settings):
                state["frozen"][stage] = channel["inputs"]
                remaining = [name for name in SHAPE_ORDER if name in groups and name not in state["frozen"]]
                state["stage"] = remaining[0] if remaining else "balance"
                state["started_at"] = channel["inputs"]
                state.pop("reopened_at", None)
                if not remaining:
                    state["balance_at"] = channel["inputs"]
                for key in ("comparison_range", "refinement_range", "refinement_at"):
                    state.pop(key, None)
                continue
            pending = _regional_pending(groups[stage], local[stage], observed)
            a, b = select_pair(channel, settings, rng, reference, allowed_pairs=pending or local[stage])
            return mark_range(channel, stage, (a, b, None))
        measured_local = {name: pairs for name, pairs in local.items() if name not in state.get("skipped", {})}
        categories = bridges if stage == "balance" else measured_local | bridges
        if any(len(possible) < 2 for possible in bridges.values()):
            raise ValueError("Regional balance needs two usable nearby links per boundary. Review skipped pairs and level limits")
        pending = _checks(channel, categories, 2 if stage == "balance" else 1, True)
        if pending:
            _, name, candidates, orders = pending[0]
            a, b = select_pair(channel, settings, rng, reference, allowed_pairs=candidates)
            order = next((value for value in (False, True) if value not in orders[(a, b)]), False)
            return mark_range(channel, name, (a, b, order))
        fit = fit_regional(channel, settings, reference)
        if fit.regional["review_regions"]:
            state["stage"] = "review"
            state["review_regions"] = fit.regional["review_regions"]
            return None
        if fit.ready(settings):
            if stage == "verify":
                return None
            state["stage"] = "verify"
            state["started_at"] = state["verify_at"] = channel["inputs"]
            for key in ("comparison_range", "refinement_range", "refinement_at"):
                state.pop(key, None)
            continue
        # Never silently reshape a frozen region to satisfy balance evidence.
        if not bridges:
            raise ValueError("Frozen shape uncertainty or disagreement needs a regional review. Use Settings > Revisit regional shape")
        return grouped_refinement(channel, settings, rng, reference, bridges)
