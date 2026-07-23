"""Benchmark condition tables.

A condition is a plain dict (see DEFAULT_CONDITION in the pysim engine) plus
three bookkeeping keys: name, group (one X axis of the figures), and axis
(the numeric value along that axis).
"""
import numpy as np

from . import engine
from .engine import make_condition, route_cmd_mode
from .real_world import REAL_WORLD

NOMINAL_DR = dict(mass=0.391, radius=0.10, foot=0.8, ball=0.5)   # deploy nominal
# Match the C++ ros2_control sim2sim path exactly. Its structural timing (read
# from mujoco_sim_ros2 PhysicsLoop + MujocoRos2Control::update):
#   - action: CM write lands in the SAME sim step's mj_step2 -> 0 delay;
#   - ball AND base state: published by the bridge at 100 Hz from the physics
#     thread, consumed via subscriptions in the CM executor thread, so the
#     policy tick always reads the PREVIOUS publish -> one publish period
#     (10 ms) stale. engine models this as bridge_delay_ms (DEFAULT_CONDITION
#     default 10), so every condition carries it implicitly.
# The two SYNTHETIC channels stay zero at nominal: real vision/actuation
# latency is unmeasured (real_world.py) and must not be guessed in. Latency
# axes stack exactly one synthetic channel on top of the structural hop.
CPP_SIM2SIM_LATENCY = dict(ball_obs_delay_steps=0, action_delay_ms=0)

# Roll brake: the real value sits far outside the trained one (see real_world),
# so this axis is REAL-anchored, not training-anchored.
REAL_BALL_DAMPING = REAL_WORLD["ball_damping"]["nominal"]
REAL_BALL_DAMPING_BAND = REAL_WORLD["ball_damping"]["band"]


def condition_row(name, group, axis, **overrides):
    for key, value in CPP_SIM2SIM_LATENCY.items():
        overrides.setdefault(key, value)
    return {**make_condition(**overrides), "name": name, "group": group, "axis": float(axis)}


def ball_damping_axis(train_c, real=REAL_BALL_DAMPING, band=REAL_BALL_DAMPING_BAND):
    """Roll-brake axis spanning the REAL band and the TRAINED value in one sweep.

    Step size widens with c because the roll decay k = 2/7 c is exponential in
    time: the same delta-c changes the roll distance 3.5/c far more at the
    slippery (hardware-real) end, which is also where the behavior differentiates
    — the same reason the training side spaces its damping DR bins logarithmically.
    Both anchors (`real` and the checkpoint's trained c) are always grid points.
    """
    lo = min(band[0], train_c or band[0])
    hi = max(band[1], train_c or band[1])
    values = set()
    for start, stop, step in ((lo, min(1.2, hi), 0.1), (1.2, min(2.0, hi), 0.2),
                              (2.0, hi, 0.4)):
        if stop >= start:
            values |= {round(float(v), 3) for v in np.arange(start, stop + 1e-9, step)}
    values |= {round(float(real), 3)} | ({round(float(train_c), 3)} if train_c else set())
    values = tuple(sorted(v for v in values if lo - 1e-9 <= v <= hi + 1e-9))
    print(f"[conditions] ball_damping: axis {values} "
          f"(real {real:g}, trained {train_c if train_c else 'unknown'}; "
          f"roll distance {3.5 / values[-1]:.2f}-{3.5 / values[0]:.2f} m at 1 m/s)")
    return values


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

    # Two baselines. `nominal` is the DEPLOY nominal robot: every robot-DR
    # channel explicitly off, so it is the clean reference the physics axes are
    # read against (it used to sample gain/payload/CoM/encoder silently, which
    # made it neither clean nor reproducible -- see engine.TRAIN_DR).
    # `train_dr` turns all four back on at the checkpoint's own trained ranges,
    # so "how much of the gap is just training-level robot DR" stays measurable.
    table = [condition_row("nominal", "baseline", 0.0, dr=NOMINAL_DR),
             condition_row("train_dr", "baseline", 1.0, dr=NOMINAL_DR,
                           actuator_gain_scale=engine.TRAIN_DR,
                           payload_kg=engine.TRAIN_DR,
                           base_com_scale=engine.TRAIN_DR,
                           joint_offset_rad=engine.TRAIN_DR)]
    if (engine.JOINT_FRICTION_RANGE is not None
            and engine.JOINT_FRICTION_RANGE[1] > engine.JOINT_FRICTION_RANGE[0]):
        print(f"[conditions] WARNING: this checkpoint trained joint friction over "
              f"{engine.JOINT_FRICTION_RANGE} N*m, which every unpinned condition "
              f"(including `nominal`) samples per episode — the baseline is NOT "
              f"friction-clean. The joint_friction axis pins it.")
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
    for c in ball_damping_axis(train.get("ball_damping")):
        table.append(condition_row(f"balldamp_{c:g}", "ball_damping", c,
                                   dr=NOMINAL_DR, ball_damping=c))
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

    # ---- ROBOT-SIDE axes -------------------------------------------------
    # Everything above perturbs the BALL. Training also randomizes the robot
    # (sensor noise, actuator gains, torso payload/CoM, encoder calibration) and
    # the benchmark tested none of it, so every previous run evaluated a
    # nominal, perfectly-calibrated robot with noise-free sensors -- easier than
    # the policy's own training distribution, and blind to the channels most
    # likely to break a real transfer. Each axis is a MULTIPLE of the
    # checkpoint's own trained magnitude, so 1.0 reproduces training.
    def scaled_axis(channel, present, values):
        if not present:
            print(f"[conditions] {channel}: not randomized in training — skipped")
            return ()
        print(f"[conditions] {channel}: scale axis {values} (x trained magnitude)")
        return values

    # obs noise is OFF at the deploy nominal (see engine.DEFAULT_CONDITION for
    # why), so this axis is the only place the noise sensitivity is measured --
    # 1.0 is the level the checkpoint actually trained with.
    SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    for scale in scaled_axis("obs_noise", train.get("obs_noise"), SCALES):
        table.append(condition_row(f"obsnoise_{scale:g}", "obs_noise", scale,
                                   dr=NOMINAL_DR, obs_noise_scale=scale))
    for scale in scaled_axis("base_com", train.get("base_com_range"), SCALES[:7]):
        table.append(condition_row(f"basecom_{scale:g}", "base_com", scale,
                                   dr=NOMINAL_DR, base_com_scale=scale))

    # actuator gain is a two-sided multiplier, so its axis walks the multiplier
    # itself out from 1.0 rather than a magnitude scale
    gain_range = train.get("actuator_gain_range")
    if gain_range:
        half = max(abs(gain_range[1] - 1.0), abs(1.0 - gain_range[0]))
        gains = thin(sorted({round(1.0 + m * half, 3)
                             for m in (-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3)}))
        print(f"[conditions] actuator_gain: axis {gains} (trained {gain_range})")
        for g in gains:
            table.append(condition_row(f"gain_{g:g}", "actuator_gain", g,
                                       dr=NOMINAL_DR, actuator_gain_scale=g))
    else:
        print("[conditions] actuator_gain: not randomized in training — skipped")

    # payload / encoder offset are absolute (their trained ranges are one-sided),
    # anchored on the trained maximum
    for channel, group, key, rng_key in (
            ("payload", "payload", "payload_kg", "payload_kg_range"),
            ("encoder offset", "encoder_offset", "joint_offset_rad",
             "joint_offset_range")):
        trained = train.get(rng_key)
        if not trained:
            print(f"[conditions] {channel}: not randomized in training — skipped")
            continue
        top = max(abs(trained[0]), abs(trained[1]))
        values = thin([round(m * top, 4) for m in (0, 0.5, 1.0, 1.5, 2.0, 3.0)])
        print(f"[conditions] {channel}: axis {values} (trained max {top:g})")
        for v in values:
            table.append(condition_row(f"{group}_{v:g}", group, v,
                                       dr=NOMINAL_DR, **{key: v}))

    # joint friction (leg+waist dof_frictionloss). The iter-80000 lineage trained
    # FRICTIONLESS, so nominal is 0 (set by configure_train_dr) -- but the real
    # robot has 1-10 N*m of it (the training joint_friction event's sim2real DR
    # estimate is U(0, 1.5) abs), and the benchmark's MJCF silently compiled 0.1.
    # So this axis is a pure sim2real probe: how much joint friction the policy
    # tolerates, anchored on that estimate and swept past it. Always built (real
    # hardware has friction regardless of what training used), like ball_damping.
    for jf in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        table.append(condition_row(f"jointfric_{jf:g}", "joint_friction", jf,
                                   dr=NOMINAL_DR, joint_friction=jf))
    print(f"[conditions] joint_friction: axis {(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)} "
          f"N*m (sim2real probe; trained frictionless, real ~1-10)")

    # believed-vs-true ball radius. Training feeds the policy the TRUE radius;
    # the deployment C++ feeds a configured constant, so the real failure mode is
    # the configured value disagreeing with the ball on the floor. Nothing in
    # training covers this, so it is a pure probe around the deploy nominal.
    for delta in (-0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02):
        table.append(condition_row(f"radiusobs_{delta:g}", "ball_radius_obs", delta,
                                   dr=NOMINAL_DR,
                                   ball_radius_obs_m=NOMINAL_DR["radius"] + delta))

    # ---- TASK-START ball placement -------------------------------------------
    # Training places the ball per episode at dist * (forward rotated by bearing)
    # from the robot -- an ACTIVE DR (class default, see the env-yaml-null gotcha)
    # the benchmark previously ignored, spawning every ball at a fixed 0.65 m / 0
    # deg. Two axes sweep it: distance (how far ahead the ball starts) and bearing
    # (how far off-centre). Each anchors on the trained range and stress-probes
    # past it; along each axis the OTHER coordinate is held at the narrow
    # physics-axis start (below) so only the swept coordinate varies.
    #
    # PHYSICS_START is the tight start band every NON-baseline condition uses: a
    # DR/latency/etc. axis should isolate its own parameter, so it gets only a
    # small start spread (enough to de-determinize survival), NOT the full trained
    # 0.5-0.85 m x +/-20 deg. The baseline alone samples the full trained start,
    # so the nominal number reflects the real task-start distribution.
    # A checkpoint that predates task-start DR has a DEGENERATE trained range
    # (engine pins it, see configure_train_dr); it must not be handed a band it
    # never saw, so PHYSICS_START collapses onto its trained point.
    d_lo, d_hi = engine.RESET_BALL_DIST_RANGE
    PHYSICS_START = (0.6, 0.7) if d_hi > d_lo else (d_lo, d_hi)
    dist_vals = sorted(set(round(float(v), 3) for v in np.linspace(0.5 * d_lo, 1.5 * d_hi, 9))
                       | {round(engine.RESET_BALL_DIST_DEFAULT, 3), round(d_lo, 3), round(d_hi, 3)})
    print(f"[conditions] reset_ball_dist: axis {tuple(dist_vals)} m "
          f"(trained {engine.RESET_BALL_DIST_RANGE})")
    for v in dist_vals:
        table.append(condition_row(f"resetdist_{v:g}", "reset_ball_dist", v,
                                   dr=NOMINAL_DR, reset_ball_dist=v, reset_ball_bearing=0.0))
    b_hi = max(abs(engine.RESET_BALL_BEARING_DEG[0]), abs(engine.RESET_BALL_BEARING_DEG[1]))
    if b_hi <= 0.0:
        # trained dead-ahead: anchor the probe on the ball-radius scale instead of
        # on a trained band, and say so rather than emitting nine copies of 0 deg
        bearing_vals = (-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0)
        print(f"[conditions] reset_ball_bearing: axis {bearing_vals} deg "
              f"(NOT trained — ball started dead ahead; pure sim2real probe)")
    else:
        bearing_vals = tuple(sorted({round(float(v), 1) for v in
                                     (-3 * b_hi, -2 * b_hi, -b_hi, -0.5 * b_hi, 0.0,
                                      0.5 * b_hi, b_hi, 2 * b_hi, 3 * b_hi)}))
        print(f"[conditions] reset_ball_bearing: axis {bearing_vals} deg "
              f"(trained {engine.RESET_BALL_BEARING_DEG})")
    for v in bearing_vals:
        table.append(condition_row(f"resetbearing_{v:g}", "reset_ball_bearing", v,
                                   dr=NOMINAL_DR, reset_ball_dist=list(PHYSICS_START),
                                   reset_ball_bearing=v))

    # ---- DEPLOY HAND-OFF probe -----------------------------------------------
    # Real deployment does NOT hand the policy a clean kinematic reset: the
    # StandbyController holds the standby pose at stiff gains, gravity settles the
    # robot, then control HARD-swaps to the policy (soft gains, memory reset).
    # Training never saw this -- it takes over on step 0 from the normal_pose clip
    # -- so it is a pure sim2real probe, isolated in its own axis and flagged
    # deploy-only rather than folded into the main survival. The axis sweeps how
    # long the robot settles at standby before the swap. (settle_time_range_s, the
    # training-side takeover window, is separate: the whole-run --settle-s.)
    for hold in (0.5, 1.0, 2.0):
        table.append(condition_row(f"handover_{hold:g}", "handover", hold,
                                   dr=NOMINAL_DR, standby_hold_s=hold))
    print(f"[conditions] handover: standby hold {(0.5, 1.0, 2.0)} s "
          f"(deploy hard hand-off probe)")

    # Randomized start on every robustness episode. The baseline (dist/bearing
    # None) samples the FULL trained start so its number is training-faithful;
    # every other axis gets the narrow PHYSICS_START band + bearing 0, so a drop
    # is attributable to that axis and not to start variance. The two reset_ball_*
    # axes above already pin their swept coordinate.
    for c in table:
        c["reset_ball_random"] = True
        if c["group"] not in ("baseline", "reset_ball_dist", "reset_ball_bearing"):
            c["reset_ball_dist"] = list(PHYSICS_START)
            c["reset_ball_bearing"] = 0.0
    return table


# Training-code class defaults (SoftTouch-multiagent dribble_env.py) for route
# fields the env.yaml only dumps when explicitly overridden (older dumps lack
# the keys entirely; newer dumps carry null for "class default").
TRAIN_CODE_DEFAULTS = dict(route_cruise_range=(1.1, 2.0), route_uturn_kappa=(2.0, 4.0),
                           route_uturn_angle_deg=(160.0, 200.0),
                           route_uturn_cruise_m=(1.5, 4.0), route_human_kappa_cap=0.5)


def capability_conditions(train=None):
    """Capability (performance) test: DEPLOY-nominal env + small reset jitter,
    extreme commands, fail-fast control criteria (the episode fails once the ball
    has been >0.8 m off the route for OFFROUTE_GRACE_S, or is >1.2 m from the
    robot). Metric = SUCCESS RATE, reported at three nesting strictnesses
    (possession / route-adherence / strict — see engine.episode_metrics).

    "Deploy-nominal" here means parity with the C++ sim2sim implementation:
    ball/base observations one bridge publish (10 ms, engine bridge_delay_ms)
    stale, fresh joint state, position target applied the same sim step, and no
    synthetic queue on top. Hardware latency has not been measured yet
    (real_world.py), so the nominal benchmark does not guess one. The dedicated
    latency axes report sensitivity with the other synthetic channel held at
    zero. engine.py still clips the policy target to 90% of the joint soft
    limit, as the C++ controller does.

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

    # u_turn is a v2-route-generator mode with 0 share in the base cmd-mode mix,
    # so it is only a fair capability probe for a checkpoint whose curriculum
    # actually gave modes 5/6 mass (see train_dr._cmd_mode_uturn_shares). Gate
    # the whole group on that, matching how physics axes skip channels the
    # checkpoint never randomized -- otherwise every human-only policy is scored
    # on a drill it never trained.
    u_turn_share = max(train.get("u_turn_share") or 0.0,
                       train.get("human_uturn_share") or 0.0)
    u_turn_enabled = u_turn_share > 0.0 and train.get("route_v2_geom") is not False

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
    print(f"[conditions] straight_speed: {straight_v} m/s (trained cruise {cruise})")
    print(f"[conditions] corner_turn: |kappa| {corner_k} (trained cap {cap:g})")
    print(f"[conditions] human_dribble: cap {human_caps} (trained cap {cap:g})")
    if u_turn_enabled:
        # resolve the u_turn geometry ONLY when the group is built, so a
        # human-only checkpoint does not print misleading fallback notes for a
        # drill it will never be scored on
        ut_kappa = resolved("route_uturn_kappa")
        ut_angle = [float(a) for a in resolved("route_uturn_angle_deg")]
        ut_lead = [float(m) for m in resolved("route_uturn_cruise_m")]
        uturn_k = tuple(sorted(set(grid(0.5 * ut_kappa[0], min(ut_kappa[0] + 0.5, ut_kappa[1]), 0.25))
                               | set(grid(ut_kappa[0] + 1.0, ut_kappa[1], 0.5))))
        print(f"[conditions] u_turn: |kappa| {uturn_k} (trained {ut_kappa}, "
              f"peak share {u_turn_share:g})")
    else:
        print(f"[conditions] u_turn: not trained (mode 5/6 share 0) — skipped")

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
    if u_turn_enabled:
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
            if condition["bridge_delay_ms"] is not None and float(condition["bridge_delay_ms"]) < 0:
                raise ValueError("bridge_delay_ms must be >= 0 (10 = C++ sim2sim parity)")
            if condition["ball_damping"] is not None and float(condition["ball_damping"]) < 0:
                raise ValueError("ball_damping must be >= 0 (0 = free-rolling ball)")
        except (ValueError, KeyError) as e:
            raise ValueError(f"conditions[{i}] ({name}): {e}") from e
        table.append({**condition, "name": name, "group": group, "axis": axis})
    print(f"[conditions] loaded {len(table)} conditions from {path}")
    return table
