"""Benchmark condition tables.

A condition is a plain dict (see DEFAULT_CONDITION in the pysim engine) plus
three bookkeeping keys: name, group (one X axis of the figures), and axis
(the numeric value along that axis).
"""
import numpy as np

from . import engine
from .engine import make_condition, route_cmd_mode

NOMINAL_DR = dict(mass=0.391, radius=0.10, foot=0.8, ball=0.5)   # deploy nominal
# every condition pins latency to the deployment nominal — the REAL robot
# pipeline's ball-obs lag (2 policy steps) and action lag (10 ms), a property of
# the deployment stack, not of the checkpoint; the latency axes override their own
DEPLOY_LATENCY = dict(ball_obs_delay_steps=2, action_delay_ms=10)


def condition_row(name, group, axis, **overrides):
    for key, value in DEPLOY_LATENCY.items():
        overrides.setdefault(key, value)
    return {**make_condition(**overrides), "name": name, "group": group, "axis": float(axis)}


def robustness_conditions(train=None):
    """Robustness test: perturb the environment, keep the nominal command (human
    routes at the trained speed law, fixed route bank). No fail-fast: episodes run
    the full episode budget; metrics are survival / possession / speed / tracking.

    Axis values are ANCHORED on the checkpoint's own training DR (train =
    train_dr.read_train_dr record): each perturbation axis walks multiples
    {0.5, 1, 1.5, 2}x of the trained magnitude (in-distribution points, the 1.5x
    probe just past the trained envelope, a 2x stress point), then a DENSE
    uniform grid up to the legacy stress max — archived runs showed the failure
    cliffs sit far outside small trained magnitudes (ball push signal at 1.5-2
    m/s, obs-lag cliff between 5 and 8 steps), so a purely anchored axis would
    read flat, and coarse stress points leave the cliff unlocalized. Latency
    axes test every engine-representable grid point (1 policy step / 5 ms).
    For the v2 checkpoints the result is a superset of the legacy axes ->
    longitudinal comparability with archived CSVs is preserved.
    Physics DR is tested one parameter at a time (see the sweep block below);
    the old joint dr_scale axis is gone — a joint sample cannot attribute a
    failure to any single parameter. A channel the checkpoint never trained with
    falls back to the fixed legacy axis (a pure stress probe — flagged in print).
    Push cadence stays a PROTOCOL constant (every 5 s): deriving it per
    checkpoint would change test harshness under identical condition names.
    """
    train = train or {}
    MAX_LEVELS = 13   # thin an axis rather than explode the episode budget

    def thin(values):
        values = tuple(sorted(set(values)))
        while len(values) > MAX_LEVELS:   # keep endpoints, drop every other interior
            values = values[::2] if values[-1] in values[::2] else values[::2] + values[-1:]
        return values

    def anchored(channel, fallback_values, tail_step=0.25):
        """In-distribution multiples {0.5,1,1.5,2}x the trained magnitude, then a
        uniform tail_step grid up to the legacy stress max (archived cliffs sit
        far outside small trained magnitudes, and the cliff needs localizing)."""
        p = train.get(channel)
        if not p:
            print(f"[conditions] {channel}: not in training DR — legacy stress axis "
                  f"{fallback_values}")
            return fallback_values
        fine = [round(m * p["dv"], 3) for m in (0.5, 1.0, 1.5, 2.0)]
        top = max(max(fallback_values), fine[-1])
        tail = [round(v, 3) for v in np.arange(0.0, top + 1e-9, tail_step)
                if v > fine[-1] + 1e-9]
        values = thin(fine + tail)
        print(f"[conditions] {channel}: axis {values} m/s (trained dv<={p['dv']:g})")
        return values

    def latency_axis(pair, fallback_values, unit=1):
        """Every representable grid point (unit = the engine's delay granularity)
        from 0 up to max(2x the trained upper bound, the legacy stress max)."""
        if pair is None or pair[1] <= 0:
            print(f"[conditions] latency: not in training DR — legacy stress axis "
                  f"{fallback_values}")
            return fallback_values
        lo, hi = pair
        top = max(int(np.ceil(2.0 * hi / unit)) * unit, max(fallback_values))
        values = thin(range(0, top + 1, unit))
        print(f"[conditions] latency axis {values} (trained [{lo:g}, {hi:g}])")
        return values

    push_dvs = anchored("push_robot", (0.25, 0.5, 0.75, 1.0, 1.5))
    ballpush_dvs = anchored("push_ball", (0.5, 1.0, 1.5, 2.0))
    obslag_steps = latency_axis(train.get("ball_obs_delay_steps"), (0, 1, 2, 3, 5, 8))
    actlag_ms = latency_axis(train.get("action_delay_ms"), (0, 10, 20, 30, 40), unit=5)

    table = [condition_row("nominal", "baseline", 0.0, dr=NOMINAL_DR)]
    # per-parameter physics sweeps — the field-standard one-factor-at-a-time
    # protocol (e.g. sim2sim stress tests sweeping friction / mass scale alone):
    # ONE parameter walks a dense grid over the checkpoint's sweep band (training
    # range expanded 1.5x, centered), everything else pinned at deploy nominal,
    # so a drop is attributable to that parameter and no other. The deploy
    # nominal is always inserted as a grid point (visible anchor even when it
    # sits off the trained band's center, e.g. foot friction 0.8 vs band center
    # 0.75). Parameters the checkpoint never randomized have degenerate bands
    # and are skipped.
    for group, dr_key in (("ball_mass", "mass"), ("ball_radius", "radius"),
                          ("foot_friction", "foot"), ("ball_friction", "ball")):
        lo, hi = engine.SWEEP_RANGES[group]
        if hi - lo < 1e-9:
            print(f"[conditions] {group}: not randomized in training — skipped "
                  f"(fixed {lo:g})")
            continue
        values = sorted(set(round(float(v), 4) for v in np.linspace(lo, hi, 9))
                        | {NOMINAL_DR[dr_key]})
        print(f"[conditions] {group}: axis {values}")
        for v in values:
            table.append(condition_row(f"{group}_{v:g}", group, v,
                                       dr={**NOMINAL_DR, dr_key: v}))
    for dv in push_dvs:
        table.append(condition_row(f"push_{dv:g}", "base_push", dv,
                                   dr=NOMINAL_DR, push_dv=dv))
    for dv in ballpush_dvs:
        table.append(condition_row(f"ballpush_{dv:g}", "ball_push", dv,
                                   dr=NOMINAL_DR, ball_push_dv=dv))
    for steps in obslag_steps:
        table.append(condition_row(f"obslag_{steps}", "obs_latency", steps,
                                   dr=NOMINAL_DR, ball_obs_delay_steps=steps))
    for ms in actlag_ms:
        table.append(condition_row(f"actlag_{ms}", "act_latency", ms,
                                   dr=NOMINAL_DR, action_delay_ms=ms))
    return table


# Training-code class defaults (SoftTouch-multiagent dribble_env.py) for route
# fields the env.yaml only dumps when explicitly overridden (older dumps lack
# the keys entirely; newer dumps carry null for "class default").
TRAIN_CODE_DEFAULTS = dict(route_cruise_range=(1.1, 2.0), route_uturn_kappa=(2.0, 4.0),
                           route_uturn_angle_deg=(160.0, 200.0),
                           route_uturn_cruise_m=(1.5, 4.0), route_human_kappa_cap=0.5)


def capability_conditions(train=None):
    """Capability (performance) test: clean nominal env + small reset jitter,
    extreme commands, fail-fast control criteria (the episode FAILS the moment
    the ball is >0.8 m off the route or >1.2 m from the robot).
    Metric = SUCCESS RATE.

    Axes are ANCHORED on the checkpoint's trained command distribution (env.yaml
    via train_dr; fields the dump lacks fall back to TRAIN_CODE_DEFAULTS with a
    printed note) and swept on a dense grid past it:

    straight_speed — commanded speed from 0.5x the trained cruise max up to the
    3.5 m/s stress point, step 0.25; success = kept control the whole episode.
    corner_turn — random straight lead-in U(1.5, 4) m, ONE arc of U(150, 180)
    deg at constant kappa, straight exit; speed follows the trained law
    min(vmax, sqrt(kv/|kappa|)); success additionally requires finishing the
    turn. kappa sweeps 0.4x-2x the trained route_human_kappa_cap, step 0.2x,
    both directions. Episode budget computed from the slowest geometry (max
    lead + max arc at the kv-law pace + 4 s margin) so gentle-kappa points are
    not clipped into fake failures.
    human_dribble — the nominal task itself: route_human_kappa_cap swept
    0.2x-2.2x the trained cap, step 0.2x (weave magnitude and big-turn
    curvature both scale with the cap); 20 s episodes.
    u_turn — the about-face drill matching the training u_turn mode: run-in
    U(trained ROUTE_UTURN_CRUISE_M), ONE constant-kappa turn of U(trained
    ROUTE_UTURN_ANGLE_DEG), straight exit; kappa dense (step 0.25) from 0.5x
    the trained lower bound through the low end (where archived runs show the
    only success signal), coarse (step 0.5) up to the trained max; both
    directions; its own figure (uturn_compare.png).
    speed_tracking — speed CONTROLLABILITY on nominal human-dribble routes with
    the TRAINED cruise distribution U(route_cruise_range); per-step (commanded,
    actual) pairs, actual smoothed over 0.5 s, per-episode Pearson r; no
    fail-fast, 20 s episodes.
    """
    train = train or {}

    def resolved(key):
        value = train.get(key)
        if value is None:
            value = TRAIN_CODE_DEFAULTS[key]
            print(f"[conditions] {key}: not dumped in env.yaml — training-code "
                  f"default {value}")
        return value

    cruise = resolved("route_cruise_range")
    cap = float(resolved("route_human_kappa_cap"))
    ut_kappa = resolved("route_uturn_kappa")
    ut_angle = [float(a) for a in resolved("route_uturn_angle_deg")]
    ut_lead = [float(m) for m in resolved("route_uturn_cruise_m")]

    def grid(start, stop, step):
        return tuple(round(v, 3) for v in np.arange(start, stop + 1e-9, step))

    def turn_budget(kappa, angle_hi_deg, floor_s):
        # slowest geometry must finish inside the budget: max lead at vmax + the
        # full arc at the trained kv-law pace + margin. Old grid points keep
        # their historical budgets via the floor.
        vmax = engine.ROUTE_CFG["routeVmax"]; kv = engine.ROUTE_CFG["routeKvScale"]
        v_arc = min(vmax, np.sqrt(kv / kappa))
        return float(max(floor_s, np.ceil(4.0 / vmax + np.deg2rad(angle_hi_deg) / kappa / v_arc + 4.0)))

    straight_v = grid(round(0.5 * cruise[1] * 4) / 4, max(3.5, 1.75 * cruise[1]), 0.25)
    corner_k = grid(0.4 * cap, 2.0 * cap, 0.2 * cap)
    human_caps = grid(0.2 * cap, 2.2 * cap, 0.2 * cap)
    uturn_k = tuple(sorted(set(grid(0.5 * ut_kappa[0], min(ut_kappa[0] + 0.5, ut_kappa[1]), 0.25))
                           | set(grid(ut_kappa[0] + 1.0, ut_kappa[1], 0.5))))
    print(f"[conditions] straight_speed: {straight_v} m/s (trained cruise {cruise})")
    print(f"[conditions] corner_turn: |kappa| {corner_k} (trained cap {cap:g})")
    print(f"[conditions] human_dribble: cap {human_caps} (trained cap {cap:g})")
    print(f"[conditions] u_turn: |kappa| {uturn_k} (trained {ut_kappa})")

    BUDGET_S = 10.0
    FAIL_FAST = dict(offroute_fail_m=0.8, ball_far_fail_m=1.2, episode_s=BUDGET_S,
                     reset_jitter=True, dr=NOMINAL_DR)
    table = []
    for v in straight_v:
        table.append(condition_row(f"straight_{v:g}", "straight_speed", v,
                                   route_mode="straight", route_vmax=v,
                                   route_len_m=v * BUDGET_S * 1.2 + 5.0, **FAIL_FAST))
    for kappa in corner_k:
        route_len = 4.0 + np.pi / kappa + 3.0   # max lead + 180 deg arc + exit & margin
        for sign, tag in ((1.0, "L"), (-1.0, "R")):
            table.append(condition_row(f"corner_{tag}_{kappa:g}", "corner_turn",
                                       sign * kappa, route_mode="arc",
                                       arc_kappa=sign * kappa, lead_in_m=[1.5, 4.0],
                                       arc_angle_deg=[150.0, 180.0],
                                       route_len_m=route_len,
                                       **{**FAIL_FAST,
                                          "episode_s": turn_budget(kappa, 180.0, 12.0)}))
    for kappa_cap in human_caps:
        table.append(condition_row(f"human_{kappa_cap:g}", "human_dribble", kappa_cap,
                                   route_mode="human", human_kappa_cap=kappa_cap,
                                   route_len_m=50.0,
                                   **{**FAIL_FAST, "episode_s": 20.0}))
    for kappa in uturn_k:
        route_len = 4.0 + np.deg2rad(ut_angle[1]) / kappa + 3.0   # max run-in + turn + exit
        for sign, tag in ((1.0, "L"), (-1.0, "R")):
            table.append(condition_row(f"uturn_{tag}_{kappa:g}", "u_turn",
                                       sign * kappa, route_mode="arc",
                                       arc_kappa=sign * kappa, lead_in_m=ut_lead,
                                       arc_angle_deg=ut_angle,
                                       route_len_m=route_len,
                                       **{**FAIL_FAST,
                                          "episode_s": turn_budget(kappa, ut_angle[1], BUDGET_S)}))
    table.append(condition_row("tracking", "speed_tracking",
                               round(0.5 * (cruise[0] + cruise[1]), 2),
                               route_mode="human", route_vmax=list(cruise),
                               route_len_m=50.0, episode_s=20.0,
                               dr=NOMINAL_DR, record_speed_pairs=True))
    return table


def load_conditions_json(path):
    """Custom table: a JSON list of {name, group, axis, <condition keys>} dicts.
    Names/groups default from the index; unknown keys and invalid values raise
    EAGERLY (a bad condition must not crash a multi-hour run halfway through)."""
    import json
    table = []
    for i, item in enumerate(json.load(open(path))):
        name = item.pop("name", f"cond_{i}"); group = item.pop("group", "custom")
        axis = float(item.pop("axis", i))
        try:
            condition = make_condition(**item)
            mode = route_cmd_mode(condition)   # raises on unknown string modes
            if mode in (1, 2, 3) and condition["arc_kappa"] is None:
                raise ValueError(f"route_mode={condition['route_mode']!r} needs an explicit "
                                 "'arc_kappa' (arc geometry comes only from the constant kappa)")
            if condition["dr"] is not None and set(condition["dr"]) != {"mass", "radius", "foot", "ball"}:
                raise ValueError(f"dr must have exactly the keys mass/radius/foot/ball, "
                                 f"got {sorted(condition['dr'])}")
            if condition["arc_angle_deg"] is not None and (
                    len(condition["arc_angle_deg"]) != 2 or condition["arc_kappa"] is None):
                raise ValueError("arc_angle_deg must be [min_deg, max_deg] and requires 'arc_kappa'")
            if isinstance(condition["route_vmax"], (list, tuple)) and len(condition["route_vmax"]) != 2:
                raise ValueError("route_vmax must be a scalar or a [min, max] per-episode range")
            for key in ("episode_s", "offroute_fail_m", "ball_far_fail_m"):
                if condition[key] is not None and float(condition[key]) <= 0:
                    raise ValueError(f"{key} must be > 0")
        except (ValueError, KeyError) as e:
            raise ValueError(f"conditions[{i}] ({name}): {e}") from e
        table.append({**condition, "name": name, "group": group, "axis": axis})
    print(f"[conditions] loaded {len(table)} conditions from {path}")
    return table
