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
    """Nominal ball mass / radius / friction / angular damping from
    scene.ball.spawn. Newer runs spawn a multi-asset ball (radius-DR x
    damping-DR bins) -> read the shared props off the first variant and the
    radius / damping spans off the bin list.

    Angular damping is the roll brake (PhysX has no rolling-friction material):
    it is NEVER dumped as a top-level cfg key, only baked into the spawn props,
    so the spawn is the only ground truth for it."""
    spawn = (cfg.get("scene", {}).get("ball") or {}).get("spawn") or {}
    assets = spawn.get("assets_cfg") or []
    first = assets[0] if assets else spawn
    mass = ((first.get("mass_props") or {}).get("mass"))
    friction = ((first.get("physics_material") or {}).get("dynamic_friction"))

    def _span(values):
        if not values:
            return None, None
        lo, hi = min(values), max(values)
        return 0.5 * (lo + hi), ((lo, hi) if hi > lo else None)

    radii = [a.get("radius") for a in assets if a.get("radius") is not None]
    radius, radius_span = _span(radii)
    if radius is None:
        radius = spawn.get("radius")
    dampings = [(a.get("rigid_props") or {}).get("angular_damping")
                for a in (assets or [spawn])]
    damping, damping_span = _span([d for d in dampings if d is not None])
    return mass, radius, friction, radius_span, damping, damping_span


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


def _obs_noise(cfg):
    """{obs term name: uniform noise half-width} for every term that carries one.

    Isaac Lab hangs `noise` on the observation TERM, and those term names
    (base_ang_vel / projected_gravity / joint_pos / joint_vel / ball_pos_b /
    ball_lin_vel_b ...) are exactly the keys engine._obs builds its `term` dict
    with, so the record can be applied by name with no translation table.

    Only symmetric uniform noise is represented; anything else is skipped with a
    note rather than silently approximated."""
    out, skipped = {}, []
    for group in (cfg.get("observations") or {}).values():
        for name, term in (group or {}).items():
            if not isinstance(term, dict):
                continue
            noise = term.get("noise")
            if not isinstance(noise, dict):
                continue
            lo, hi = noise.get("n_min"), noise.get("n_max")
            if lo is None or hi is None:
                skipped.append(name)
                continue
            half = 0.5 * (float(hi) - float(lo))
            if abs(float(hi) + float(lo)) > 1e-9:
                skipped.append(f"{name} (asymmetric)")
                continue
            if half > 0:
                out[name] = half
    if skipped:
        print(f"[train_dr] obs noise: skipped non-uniform/asymmetric terms {skipped}")
    return out or None


def _find_action_delay(cfg):
    for term in (cfg.get("actions") or {}).values():
        if isinstance(term, dict) and term.get("action_delay_ms_range") is not None:
            return _pair(term["action_delay_ms_range"]), term.get("action_delay_zero_prob")
    return None, None


def _cmd_mode_uturn_shares(curriculum):
    """(u_turn_share, human_uturn_share) the checkpoint's cmd-mode curriculum
    reaches at ANY point of its schedule — the peak share of route modes 5/6.

    Dispatched on the curriculum term's `func`, NOT its key: a resume run may
    dump `hard_modes_piecewise` under the legacy key `human_dribble_ramp`
    (observed in g1_dribble_s3_uturn_vfix_iter60000). We take the schedule
    MAXIMUM rather than the value at the checkpoint's iteration on purpose:
    `env.common_step_counter` resets to 0 on every fresh/resume process, so the
    per-iteration value is not recoverable from the dump, but "did training ever
    give u_turn any mass" is — and that is exactly the gate the benchmark needs
    (u_turn/human_uturn modes only exist under the v2 route generator, and are
    0 in the base cmd-mode mix). Interpolation between knots is a convex blend,
    so the schedule max equals the max over the knot arrays.
    """
    for term in curriculum.values():
        if not isinstance(term, dict):
            continue
        func = str(term.get("func", ""))
        p = term.get("params") or {}
        if func.endswith("hard_modes_piecewise"):
            ut = max([float(v) for v in (p.get("knot_u_turn") or [0.0])])
            hu = max([float(v) for v in (p.get("knot_human_uturn") or [0.0])])
            return ut, hu
        if func.endswith(("human_dribble_ramp", "human_dribble_piecewise")):
            # modes 5/6 are fixed carve-outs here (default 0 when absent)
            return float(p.get("u_turn_p") or 0.0), float(p.get("human_uturn_p") or 0.0)
    return 0.0, 0.0


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

    (mass_nom, radius_nom, fric_nom, radius_span,
     damping_nom, damping_span) = _ball_spawn_props(cfg)

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
    _uturn_share, _human_uturn_share = _cmd_mode_uturn_shares(curriculum)

    return dict(
        source=env_yaml,
        ball_mass_nominal=mass_nom, ball_mass_range=mass_range,
        ball_radius_nominal=radius_nom, ball_radius_range=radius_range,
        ball_friction_nominal=fric_nom,
        ball_friction_range=_pair(event_params("ball_physics_material").get("dynamic_friction_range")),
        ball_damping=damping_nom, ball_damping_range=_pair(damping_span),
        foot_friction_range=_pair(event_params("physics_material").get("dynamic_friction_range")),
        push_robot=push_robot, push_ball=push_ball,
        ball_obs_delay_steps=(None if delay_range is None
                              else (int(delay_range[0]), int(delay_range[1]))),
        ball_pos_noise=_pair((obs_noise.get("n_min"), obs_noise.get("n_max"))
                             if obs_noise.get("n_max") is not None else None),
        action_delay_ms=action_delay_ms, action_delay_zero_prob=action_delay_zero_prob,
        # ---- robot-side DR: everything above perturbs the BALL; these are the
        # channels most likely to break a PhysX->MuJoCo->real transfer, and the
        # benchmark had none of them.
        obs_noise=_obs_noise(cfg),
        obs_corruption=(cfg.get("observations") or {}).get("policy", {}).get(
            "enable_corruption") if isinstance(
                (cfg.get("observations") or {}).get("policy"), dict) else None,
        actuator_gain_range=_pair(
            event_params("actuator_gains").get("stiffness_distribution_params")),
        payload_kg_range=_pair(
            event_params("add_torso_payload").get("mass_distribution_params")),
        joint_offset_range=_pair(
            event_params("add_joint_default_pos").get("pos_distribution_params")),
        base_com_range={k: _pair(v) for k, v in
                        (event_params("base_com").get("com_range") or {}).items()} or None,
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
        # cmd-mode curriculum: the peak schedule share of route modes 5/6, used
        # to gate the u_turn capability group (0 in the base cmd-mode mix).
        u_turn_share=_uturn_share, human_uturn_share=_human_uturn_share,
        route_v2_geom=cfg.get("route_v2_geom"),
        # task-start ball placement: only present when the run OVERRODE the class
        # default (RESET_BALL_FORWARD_RANGE / RESET_BALL_BEARING_DEG); a null here
        # means the always-active class default applies (engine keeps its own).
        reset_ball_forward_range=_pair(cfg.get("reset_ball_forward_range")),
        reset_ball_bearing_deg=_pair(cfg.get("reset_ball_bearing_deg")),
        # training standby-PD takeover window (settle_time_range_s); None/(0,0) =
        # the policy took over on step 0 (all current checkpoints). Drives the
        # whole-run --settle-s default so the benchmark start matches training.
        settle_time_range_s=_pair(cfg.get("settle_time_range_s")),
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
    c = train["ball_damping"]
    lines.append(f"  ball damping c = {'unknown' if c is None else f'{c:g}'}"
                 + ("" if c is None else f" (roll decay k = {2 / 7 * c:.3g}/s, "
                    f"1 m/s ball rolls {3.5 / c:.2f} m)")
                 + f", DR {rng(train['ball_damping_range'])}")
    for key, unit in (("push_robot", "m/s"), ("push_ball", "m/s")):
        p = train[key]
        lines.append(f"  {key:12} " + ("not trained" if p is None else
                     f"dv <= {p['dv']:g} {unit}, every {p['interval_s']} s"))
    lines.append(f"  ball obs lag {rng(train['ball_obs_delay_steps'], '{:.0f}')} policy steps")
    noise = train.get("obs_noise")
    lines.append("  obs noise    " + ("not randomized" if not noise else
                 ", ".join(f"{k} +/-{v:g}" for k, v in sorted(noise.items()))))
    for key, label, fmt in (("actuator_gain_range", "actuator gain", "{:.3g}"),
                            ("payload_kg_range", "torso payload", "{:.3g}"),
                            ("joint_offset_range", "joint offset", "{:.3g}")):
        lines.append(f"  {label:12} {rng(train.get(key), fmt)}")
    com = train.get("base_com_range")
    lines.append("  torso CoM    " + ("not randomized" if not com else
                 ", ".join(f"{k} {rng(v, '{:.3g}')}" for k, v in sorted(com.items()))))
    lines.append(f"  action lag   {rng(train['action_delay_ms'], '{:.0f}')} ms"
                 + ("" if train["action_delay_zero_prob"] is None
                    else f", zero_prob {train['action_delay_zero_prob']:g}"))
    ut, hu = train.get("u_turn_share") or 0.0, train.get("human_uturn_share") or 0.0
    lines.append(f"  u_turn       " + ("not trained (share 0 in cmd-mode mix)"
                 if max(ut, hu) <= 0.0 else
                 f"trained (peak share u_turn {ut:g}, human_uturn {hu:g})"))
    fr = train.get("reset_ball_forward_range")
    br = train.get("reset_ball_bearing_deg")
    lines.append(f"  ball start   " + (
        "class-default dist / bearing (env.yaml did not override)"
        if fr is None and br is None else
        f"dist {rng(fr)} m, bearing {rng(br, '{:.0f}')} deg (env.yaml override)"))
    st = train.get("settle_time_range_s")
    lines.append("  settle       " + (
        "none (policy took over on step 0)" if st is None or st[1] <= 0
        else f"{rng(st, '{:.2f}')} s standby-PD takeover window"))
    return lines
