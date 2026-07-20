"""Read a checkpoint's ACTUAL training DR from the Isaac Lab env.yaml dumped
next to it (checkpoints/<run>/env.yaml, written by the training entry point).

The benchmark used to hardcode the training DR ranges; they silently went stale
as checkpoints evolved (e.g. ball radius DR and ball pushes exist only in newer
runs, and the oldest checkpoint used a 0.34 kg / 0.09 m ball). Every value the
robustness table is anchored on now comes from here; the hardcoded engine
constants remain only as an explicit fallback for policies without an env.yaml.

The parsed record is a plain dict; every range key is a (lo, hi) tuple or None
when that channel was NOT randomized in training.
"""
import os
import re

import yaml


def _tolerant_loader():
    # Isaac Lab dumps carry python-object tags (!!python/tuple, builtins.slice ...).
    # Tuples matter; every other python object is metadata we can drop.
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor("tag:yaml.org,2002:python/tuple",
                           lambda ld, node: tuple(ld.construct_sequence(node)))
    for tag in ("tag:yaml.org,2002:python/object/apply:",
                "tag:yaml.org,2002:python/object/new:",
                "tag:yaml.org,2002:python/object:",
                "tag:yaml.org,2002:python/name:"):
        Loader.add_multi_constructor(tag, lambda ld, suffix, node: None)
    return Loader


def find_env_yaml(path):
    """Locate the env.yaml for a checkpoint given its dir / onnx / model path
    (or an env.yaml path directly). Probes the curated checkpoints/ layout
    (env.yaml next to the model) and the raw RSL-RL log layout (params/env.yaml).
    Returns None when there is none."""
    if not path:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    if path.endswith((".yaml", ".yml")):
        return path if os.path.isfile(path) else None
    base = path if os.path.isdir(path) else os.path.dirname(path)
    for candidate in (os.path.join(base, "env.yaml"),
                      os.path.join(base, "params", "env.yaml")):
        if os.path.isfile(candidate):
            return candidate
    return None


def _pair(value):
    if value is None:
        return None
    lo, hi = (float(value[0]), float(value[1]))
    return (lo, hi) if lo <= hi else (hi, lo)


def _ball_spawn_props(cfg):
    """Nominal ball mass / radius / friction from scene.ball.spawn. Newer runs
    spawn a multi-asset ball (radius-DR bins) -> read the shared props off the
    first variant and the radius span off the bin list."""
    spawn = (cfg.get("scene", {}).get("ball") or {}).get("spawn") or {}
    assets = spawn.get("assets_cfg") or []
    first = assets[0] if assets else spawn
    mass = ((first.get("mass_props") or {}).get("mass"))
    friction = ((first.get("physics_material") or {}).get("dynamic_friction"))
    radii = [a.get("radius") for a in assets if a.get("radius") is not None]
    if radii:
        radius = 0.5 * (min(radii) + max(radii))
        radius_span = (min(radii), max(radii)) if max(radii) > min(radii) else None
    else:
        radius = spawn.get("radius")
        radius_span = None
    return mass, radius, friction, radius_span


def _find_obs_term(cfg, func_substr):
    for group in (cfg.get("observations") or {}).values():
        for term in (group or {}).values():
            if isinstance(term, dict) and func_substr in str(term.get("func", "")):
                return term
    return None


def _checkpoint_iteration(env_yaml_dir):
    """Highest model_<iter>.pt near the env.yaml; None when undeterminable.
    Curated layout keeps models next to env.yaml, the raw RSL-RL layout keeps
    them one level above params/."""
    import glob
    iters = []
    for d in (env_yaml_dir, os.path.dirname(env_yaml_dir)):
        for p in glob.glob(os.path.join(d, "model_*.pt")):
            m = re.match(r"model_(\d+)\.pt$", os.path.basename(p))
            if m:
                iters.append(int(m.group(1)))
    return max(iters) if iters else None


def _find_action_delay(cfg):
    for term in (cfg.get("actions") or {}).values():
        if isinstance(term, dict) and term.get("action_delay_ms_range") is not None:
            return _pair(term["action_delay_ms_range"]), term.get("action_delay_zero_prob")
    return None, None


def read_train_dr(path):
    """Parse the checkpoint's env.yaml into a normalized training-DR record.
    Returns None when no env.yaml exists next to `path`."""
    env_yaml = find_env_yaml(path)
    if env_yaml is None:
        return None
    cfg = yaml.load(open(env_yaml), Loader=_tolerant_loader())
    events = cfg.get("events") or {}
    curriculum = cfg.get("curriculum") or {}

    def event_params(name):
        return (events.get(name) or {}).get("params") or {}

    mass_nom, radius_nom, fric_nom, radius_span = _ball_spawn_props(cfg)

    mass_range = None
    mp = _pair(event_params("ball_mass").get("mass_distribution_params"))
    if mp is not None and mass_nom is not None:
        op = event_params("ball_mass").get("operation", "scale")
        mass_range = {"scale": (mass_nom * mp[0], mass_nom * mp[1]),
                      "add": (mass_nom + mp[0], mass_nom + mp[1]),
                      "abs": mp}[op]

    radius_range = _pair(cfg.get("ball_radius_dr_range")) or radius_span

    # push_ball ships with max_speed 0 and is switched on by the ball_push_onset
    # curriculum -> the curriculum's end-state speed is the trained magnitude,
    # but only if the checkpoint actually trained PAST the onset iteration
    push_ball = None
    if "push_ball" in events:
        onset = (curriculum.get("ball_push_onset") or {}).get("params") or {}
        dv = float(onset.get("max_speed", event_params("push_ball").get("max_speed", 0.0)) or 0.0)
        ck_iter = _checkpoint_iteration(os.path.dirname(env_yaml))
        if onset.get("onset_iter") is not None and ck_iter is not None \
                and ck_iter <= int(onset["onset_iter"]):
            print(f"[train_dr] push_ball onset at iter {onset['onset_iter']} but checkpoint "
                  f"is iter {ck_iter} — the policy never saw ball pushes")
            dv = 0.0
        if dv > 0.0:
            push_ball = dict(dv=dv, interval_s=_pair(events["push_ball"].get("interval_range_s")),
                             onset_iter=onset.get("onset_iter"))

    push_robot = None
    vr = event_params("push_robot").get("velocity_range")
    if vr:
        dv = max(abs(float(b)) for axis in ("x", "y") for b in (vr.get(axis) or (0, 0)))
        if dv > 0.0:
            push_robot = dict(dv=dv, interval_s=_pair(events["push_robot"].get("interval_range_s")))

    # ":ball_pos_b" matches both the delayed (new) and plain (old) obs funcs —
    # the old runs carry the same +/-0.02 uniform noise but no delay params
    ball_obs = _find_obs_term(cfg, ":ball_pos_b") or {}
    delay_range = _pair((ball_obs.get("params") or {}).get("delay_steps_range"))
    obs_noise = ball_obs.get("noise") or {}
    action_delay_ms, action_delay_zero_prob = _find_action_delay(cfg)

    return dict(
        source=env_yaml,
        ball_mass_nominal=mass_nom, ball_mass_range=mass_range,
        ball_radius_nominal=radius_nom, ball_radius_range=radius_range,
        ball_friction_nominal=fric_nom,
        ball_friction_range=_pair(event_params("ball_physics_material").get("dynamic_friction_range")),
        foot_friction_range=_pair(event_params("physics_material").get("dynamic_friction_range")),
        push_robot=push_robot, push_ball=push_ball,
        ball_obs_delay_steps=(None if delay_range is None
                              else (int(delay_range[0]), int(delay_range[1]))),
        ball_pos_noise=_pair((obs_noise.get("n_min"), obs_noise.get("n_max"))
                             if obs_noise.get("n_max") is not None else None),
        action_delay_ms=action_delay_ms, action_delay_zero_prob=action_delay_zero_prob,
        # route/command training params (capability-axis anchors). Fields that are
        # None/absent were class defaults at training time (not dumped) — the
        # consumer falls back to the documented training-code defaults.
        route_human_kappa_cap=cfg.get("route_human_kappa_cap"),
        cmd_speed_range=_pair(cfg.get("cmd_speed_range")),
        route_cruise_range=_pair(cfg.get("route_cruise_range")),
        route_uturn_kappa=_pair(cfg.get("route_uturn_kappa")),
        route_uturn_angle_deg=_pair(cfg.get("route_uturn_angle_deg")),
        route_uturn_cruise_m=_pair(cfg.get("route_uturn_cruise_m")),
        route_vmax=cfg.get("route_vmax"),
    )


def describe(train):
    """One line per channel: the trained range, or the explicit 'not randomized'."""
    if train is None:
        return ["no env.yaml found — falling back to the hardcoded legacy training DR"]
    def rng(pair, fmt="{:.3f}"):
        return "not randomized" if pair is None else \
            f"[{fmt.format(pair[0])}, {fmt.format(pair[1])}]"
    lines = [f"training DR from {train['source']}"]
    lines.append(f"  ball mass    nominal {train['ball_mass_nominal']} kg, DR {rng(train['ball_mass_range'])}")
    lines.append(f"  ball radius  nominal {train['ball_radius_nominal']} m, DR {rng(train['ball_radius_range'])}")
    lines.append(f"  ball fric    nominal {train['ball_friction_nominal']}, DR {rng(train['ball_friction_range'])}")
    lines.append(f"  foot fric    DR {rng(train['foot_friction_range'])}")
    for key, unit in (("push_robot", "m/s"), ("push_ball", "m/s")):
        p = train[key]
        lines.append(f"  {key:12} " + ("not trained" if p is None else
                     f"dv <= {p['dv']:g} {unit}, every {p['interval_s']} s"))
    lines.append(f"  ball obs lag {rng(train['ball_obs_delay_steps'], '{:.0f}')} policy steps")
    lines.append(f"  action lag   {rng(train['action_delay_ms'], '{:.0f}')} ms"
                 + ("" if train["action_delay_zero_prob"] is None
                    else f", zero_prob {train['action_delay_zero_prob']:g}"))
    return lines
