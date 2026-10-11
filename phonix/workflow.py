"""Regional listening passes over a single shared comparison graph."""

import math

from .engine import fit_channel, nearby, pairs_for, select_pair

STAGES = ("early", "midrange", "bass", "treble", "balance", "mixed")
LABELS = {"early": "Early boundary checks", "midrange": "Midrange shape",
          "bass": "Bass shape", "treble": "Treble shape",
          "balance": "Regional balance", "mixed": "Grouped verification / refinement"}
CHECK_ORDER = ("midrange", "bass", "treble", "bass/midrange", "midrange/treble", "bass/treble")


def validate_group_state(state, inputs):
    for key in ("comparison_range", "refinement_range"):
        if key in state and (not isinstance(state[key], str) or state[key] not in CHECK_ORDER):
            raise ValueError("Invalid grouped comparison range")
    if ("refinement_range" in state) != ("refinement_at" in state):
        raise ValueError("Invalid grouped refinement checkpoint")
    if "refinement_at" in state and (type(state["refinement_at"]) is not int
                                    or not state["started_at"] <= state["refinement_at"] <= inputs):
        raise ValueError("Invalid grouped refinement checkpoint")


def validate_workflow(channel, settings=None):
    if not isinstance(channel, dict):
        raise ValueError("Invalid channel data")
    if "workflow" in channel:
        state = channel["workflow"]
        if (settings is not None and settings.calibration_method == "regional-v2"
                and (not isinstance(state, dict) or state.get("version") != 2)):
            raise ValueError("Invalid regional pass state")
        if isinstance(state, dict) and state.get("version") == 2:
            from .regional import validate_state

            validate_state(channel, settings)
            validate_group_state(state, channel["inputs"])
            return
        if (not isinstance(state, dict) or type(state.get("version")) is not int
                or state["version"] != 1 or state.get("stage") not in STAGES
                or type(state.get("started_at")) is not int
                or not 0 <= state["started_at"] <= channel["inputs"]):
            raise ValueError("Invalid regional pass state")
        validate_group_state(state, channel["inputs"])
    for observation in channel.get("observations", []):
        if "reversed_order" in observation and type(observation["reversed_order"]) is not bool:
            raise ValueError("Invalid stored playback order")


def start_workflow(channel):
    """Upgrade unfinished sessions without changing any historical judgment."""
    if "workflow" not in channel:
        channel["workflow"] = {"version": 1, "stage": "early", "started_at": channel["inputs"]}


def regions(channel):
    groups = {"bass": [], "midrange": [], "treble": []}
    for frequency in channel["frequencies"]:
        region = "bass" if frequency < 250 else "midrange" if frequency < 2000 else "treble"
        groups[region].append(frequency)
    # A lone endpoint belongs with its neighbors rather than requiring a
    # separate pass that cannot contain even one within-region comparison.
    for name, frequencies in groups.items():
        if len(frequencies) == 1:
            other = min((key for key in groups if key != name and groups[key]),
                        key=lambda key: min(abs(math.log2(f / frequencies[0])) for f in groups[key]))
            groups[other].extend(frequencies)
            groups[name] = []
    return {name: sorted(frequencies) for name, frequencies in groups.items() if frequencies}


def _geometry(channel, settings):
    pairs = pairs_for(channel, settings)
    unavailable = {tuple(pair) for pair in channel["blocked"]}
    censored = {key for key, pair in pairs.items() if pair.boundary and len(pair.observations) >= 6}
    frequencies = channel["frequencies"]
    possible = {(a, b) for i, a in enumerate(frequencies) for b in frequencies[i + 1:]
                if nearby(a, b) and (a, b) not in unavailable | censored}
    observed = set(pairs) - censored
    groups = regions(channel)
    local = {name: {key for key in possible if key[0] in fs and key[1] in fs}
             for name, fs in groups.items()}
    ordered = sorted(groups, key=lambda name: groups[name][0])
    bridges = {}
    for lower, upper in zip(ordered, ordered[1:]):
        bridges[f"{lower}/{upper}"] = {key for key in possible
                                      if key[0] in groups[lower] and key[1] in groups[upper]}
    return groups, local, bridges, observed, possible


def _coverage(frequencies, edges):
    neighbors = {f: set() for f in frequencies}
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    remaining = set(frequencies)
    components = 0
    while remaining:
        pending = [remaining.pop()]
        components += 1
        while pending:
            for neighbor in neighbors[pending.pop()] & remaining:
                remaining.remove(neighbor)
                pending.append(neighbor)
    return {f: len(ns) for f, ns in neighbors.items()}, components, len(edges) - len(frequencies) + components


def _regional_pending(frequencies, possible, observed):
    """Build redundant preliminary coverage; final precision stays global."""
    measured = observed & possible
    degree, components, cycles = _coverage(frequencies, measured)
    available_degree, available_components, available_cycles = _coverage(frequencies, possible)
    missing = {f for f in frequencies if degree[f] < min(2, available_degree[f])}
    if components <= available_components and cycles >= min(1, available_cycles) and not missing:
        return set()
    # Prioritize new links during coverage passes rather than spending the
    # entire regional pass refining an already observed posterior.
    unmeasured = possible - observed
    touching = {key for key in unmeasured if missing & set(key)}
    return touching or unmeasured


def _checks(channel, categories, distinct, both_orders):
    state = channel["workflow"]
    recent = channel["observations"][state["started_at"]:]
    pending = []
    for name, possible in categories.items():
        if not possible:
            raise ValueError(f"No usable local checks remain for {name}. The regional pass is incomplete; review skipped pairs and level limits")
        orders = {key: set() for key in possible}
        for observation in recent:
            key = (observation["a"], observation["b"])
            if key in orders and "reversed_order" in observation:
                orders[key].add(observation["reversed_order"])
        complete = {key for key, values in orders.items()
                    if len(values) >= (2 if both_orders else 1)}
        required = min(distinct, len(possible))
        if len(complete) >= required:
            continue
        partial = {key for key, values in orders.items() if values and key not in complete}
        candidates = partial or (possible - complete)
        # Complete one category, including reverse presentations, before
        # moving on. Counts never rotate the listener into another range.
        pending.append((sum(len(values) for values in orders.values()), name, candidates, orders))
    return sorted(pending, key=lambda item: CHECK_ORDER.index(item[1]))


def mark_range(channel, name, planned):
    channel["workflow"]["comparison_range"] = name
    return planned


def range_description(channel):
    name = channel.get("workflow", {}).get("comparison_range")
    groups = regions(channel)
    if name in groups:
        fs = groups[name]
        return f"{name.capitalize()} ({fs[0]:.0f}–{fs[-1]:.0f} Hz)"
    return " / ".join(part.capitalize() for part in name.split("/")) + " boundary" if name else ""


def grouped_refinement(channel, settings, rng, reference, categories):
    """Keep further refinement in one range through independent repeats.

    Each range finishes two distinct links in both orders before another
    range is considered. Readiness may stop a block early; unchanged global
    quality checks decide completion. No additional answers are fabricated.
    """
    names = [name for name in CHECK_ORDER if name in categories and categories[name]]
    if not names:
        raise ValueError("No usable grouped comparisons remain; review skipped pairs and level limits")
    state = channel["workflow"]
    name = state.get("refinement_range")
    pending = []
    if name in names:
        checking = dict(channel, workflow=dict(state, started_at=state["refinement_at"]))
        pending = _checks(checking, {name: categories[name]}, 2, True)
    if not pending:
        # Pick the next useful range only when the current block is finished.
        # This keeps global error reduction adaptive without tone-by-tone jumps.
        best = select_pair(channel, settings, rng, reference,
                           allowed_pairs=set().union(*(categories[n] for n in names)))
        name = next(n for n in names if best in categories[n])
        state["refinement_range"], state["refinement_at"] = name, channel["inputs"]
        checking = dict(channel, workflow=dict(state, started_at=state["refinement_at"]))
        pending = _checks(checking, {name: categories[name]}, 2, True)
    _, _, candidates, orders = pending[0]
    a, b = _choose(channel, settings, rng, reference, candidates, candidates)
    order = next((value for value in (False, True) if value not in orders[(a, b)]), False)
    return mark_range(channel, name, (a, b, order))


def _choose(channel, settings, rng, reference, preferred, available):
    try:
        return select_pair(channel, settings, rng, reference, allowed_pairs=preferred)
    except ValueError as error:
        # The predicted-level guard can reject a preferred, unmeasured link.
        # Try alternatives in the same region/boundary and distance limit.
        if preferred == available or not str(error).startswith("No usable pairs remain"):
            raise
        return select_pair(channel, settings, rng, reference, allowed_pairs=available)


def verified(channel, settings):
    if settings.calibration_method == "regional-v2":
        from .regional import verified_regional

        return verified_regional(channel, settings)
    if "workflow" not in channel:
        return True  # Existing completed profiles keep their original status.
    if channel["workflow"]["stage"] != "mixed":
        return False
    _, local, bridges, _, _ = _geometry(channel, settings)
    try:
        return not _checks(channel, local | bridges, 1, True)
    except ValueError:
        return False


def ready(channel, settings, fit=None):
    fit = fit if fit is not None else fit_channel(channel, settings)
    return fit.ready(settings) and verified(channel, settings)


def progress(channel, settings):
    if settings.calibration_method == "regional-v2":
        from .regional import regional_progress

        return regional_progress(channel)
    if "workflow" not in channel:
        return "Not started" if not channel["inputs"] else "Legacy session"
    state = channel["workflow"]
    groups = regions(channel)
    name = LABELS[state["stage"]]
    if state["stage"] in groups:
        frequencies = groups[state["stage"]]
        name += f" ({frequencies[0]:.0f}–{frequencies[-1]:.0f} Hz)"
    elif range_description(channel):
        name += " | " + range_description(channel)
    return name


def next_comparison(channel, settings, rng, reference=None):
    """Advance completed passes and return (a, b, forced playback order).

    None order uses the usual randomized presentation. Closing checks pair
    each presentation with its reversed order, without adding judgments for
    replay, unsure, skip, or cancellation.
    """
    if settings.calibration_method == "regional-v2":
        from .regional import next_regional

        return next_regional(channel, settings, rng, reference)
    start_workflow(channel)
    groups, local, bridges, observed, _ = _geometry(channel, settings)
    while True:
        state = channel["workflow"]
        stage = state["stage"]
        if stage in ("early", "balance", "mixed"):
            categories = (local | bridges) if stage == "mixed" else bridges
            pending = _checks(channel, categories, 1 if stage == "mixed" else 2, stage != "early")
            if pending:
                _, name, candidates, orders = pending[0]
                available = candidates
                # Nearby boundary links first; choose an informative local
                # shape pair for the mixed checks instead of a distant anchor.
                if name in bridges:
                    gap = min(math.log2(b / a) for a, b in candidates)
                    candidates = {key for key in candidates if math.log2(key[1] / key[0]) <= gap + 1e-12}
                a, b = _choose(channel, settings, rng, reference, candidates, available)
                if stage == "early":
                    return mark_range(channel, name, (a, b, None))
                order = next((value for value in (False, True) if value not in orders[(a, b)]), False)
                return mark_range(channel, name, (a, b, order))
            if stage == "mixed":
                if ready(channel, settings, fit_channel(channel, settings, reference)):
                    return None
                return grouped_refinement(channel, settings, rng, reference, local | bridges)
        elif stage in groups:
            fit = fit_channel(channel, settings, reference)
            indices = {float(f): index for index, f in enumerate(fit.frequencies)}
            # Match the selector's predicted-level guard. Preliminary passes
            # must not demand unplayable new links; global refinement can
            # later add reachable partners without relaxing stopping checks.
            reachable = {key for key in local[stage]
                         if key in observed or indices[key[1]] - indices[key[0]] == 1
                         or abs(fit.matches[indices[key[1]]] - fit.matches[indices[key[0]]]) <= settings.offset_limit * 0.95}
            pending = _regional_pending(groups[stage], reachable, observed)
            if pending:
                a, b = _choose(channel, settings, rng, reference, pending, reachable - observed)
                return mark_range(channel, stage, (a, b, None))
        state["stage"] = STAGES[STAGES.index(stage) + 1]
        state["started_at"] = channel["inputs"]
        for key in ("comparison_range", "refinement_range", "refinement_at"):
            state.pop(key, None)
