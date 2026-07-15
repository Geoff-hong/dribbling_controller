"""Benchmark condition tables.

A condition is a plain dict (see DEFAULT_CONDITION in the pysim engine) plus
three bookkeeping keys: name, group (one X axis of the figures), and axis
(the numeric value along that axis).
"""
import numpy as np

from dribble_pysim_multi import make_condition, route_cmd_mode

NOMINAL_DR = dict(mass=0.391, radius=0.10, foot=0.8, ball=0.5)   # deploy nominal
# every condition pins latency to the deployment nominal (= training DR midpoints:
# ball-obs lag 2 policy steps, action lag 10 ms); the latency axes override their own
DEPLOY_LATENCY = dict(ball_obs_delay_steps=2, action_delay_ms=10)


def condition_row(name, group, axis, **overrides):
    for key, value in DEPLOY_LATENCY.items():
        overrides.setdefault(key, value)
    return {**make_condition(**overrides), "name": name, "group": group, "axis": float(axis)}


def robustness_conditions():
    """Robustness test: perturb the environment, keep the nominal command (human
    routes at the trained speed law, fixed route bank). No fail-fast: episodes run
    the full episode budget; metrics are survival / possession / speed / tracking.
    """
    table = [condition_row("nominal", "baseline", 0.0, dr=NOMINAL_DR)]
    for alpha in (0.5, 1.0, 1.5, 2.0):
        table.append(condition_row(f"dr_x{alpha:g}", "dr_scale", alpha, dr_scale=alpha))
    for dv in (0.25, 0.5, 0.75, 1.0, 1.5):
        table.append(condition_row(f"push_{dv:g}", "base_push", dv,
                                   dr=NOMINAL_DR, push_dv=dv))
    for dv in (0.5, 1.0, 1.5, 2.0):
        table.append(condition_row(f"ballpush_{dv:g}", "ball_push", dv,
                                   dr=NOMINAL_DR, ball_push_dv=dv))
    for steps in (0, 1, 2, 3, 5, 8):
        table.append(condition_row(f"obslag_{steps}", "obs_latency", steps,
                                   dr=NOMINAL_DR, ball_obs_delay_steps=steps))
    for ms in (0, 10, 20, 30, 40):
        table.append(condition_row(f"actlag_{ms}", "act_latency", ms,
                                   dr=NOMINAL_DR, action_delay_ms=ms))
    return table


def capability_conditions():
    """Capability (performance) test: clean nominal env + small reset jitter,
    extreme commands, fail-fast control criteria (the episode FAILS the moment
    the ball is >0.8 m off the route or >1.2 m from the robot), 10 s budget.
    Metric = SUCCESS RATE.

    straight_speed — straight route, sweep the commanded speed; success = kept
    control for the whole 10 s.
    corner_turn — random straight lead-in U(0.5, 2) m, ONE arc of U(150, 180) deg
    at constant kappa, straight exit; speed follows the trained law
    min(2, sqrt(0.75/|kappa|)); success additionally requires finishing the turn.
    kappa < 0.4 is not swept: a 150-180 deg arc at the trained speed law cannot
    finish within 10 s (arc time = (angle/kappa)/speed, e.g. 8.1 s at kappa=0.2).
    """
    BUDGET_S = 10.0
    FAIL_FAST = dict(offroute_fail_m=0.8, ball_far_fail_m=1.2, episode_s=BUDGET_S,
                     reset_jitter=True, dr=NOMINAL_DR)
    table = []
    for v in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        table.append(condition_row(f"straight_{v:g}", "straight_speed", v,
                                   route_mode="straight", route_vmax=v,
                                   route_len_m=v * BUDGET_S * 1.2 + 5.0, **FAIL_FAST))
    for kappa in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        route_len = 2.0 + np.pi / kappa + 3.0   # max lead + 180 deg arc + exit & margin
        for sign, tag in ((1.0, "L"), (-1.0, "R")):
            table.append(condition_row(f"corner_{tag}_{kappa:g}", "corner_turn",
                                       sign * kappa, route_mode="arc",
                                       arc_kappa=sign * kappa, lead_in_m=[0.5, 2.0],
                                       arc_angle_deg=[150.0, 180.0],
                                       route_len_m=route_len, **FAIL_FAST))
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
            for key in ("episode_s", "offroute_fail_m", "ball_far_fail_m"):
                if condition[key] is not None and float(condition[key]) <= 0:
                    raise ValueError(f"{key} must be > 0")
        except (ValueError, KeyError) as e:
            raise ValueError(f"conditions[{i}] ({name}): {e}") from e
        table.append({**condition, "name": name, "group": group, "axis": axis})
    print(f"[conditions] loaded {len(table)} conditions from {path}")
    return table
