"""Multi-robot MuJoCo dribble simulation engine.

Composes N prefixed G1 robots (each with its own ball + route dots) into ONE
MuJoCo model via MjSpec and runs the exported deployment ONNX on each. Every
robot is driven per episode by a benchmark "condition" (route shape, DR, pushes,
latency, fail-fast criteria — see DEFAULT_CONDITION) through
Robot.reset(condition=...).

The policy is yaw/position-invariant (no world-frame obs terms), so each robot's
obs depends only on its own relative quantities -> the grid offset cancels and
the robots are fully independent.

Pure library: the benchmark runner (`python -m sim2sim_benchmark`) and the
interactive CLI (`python -m sim2sim_benchmark.pysim`) both build on it.
"""
import functools
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import onnxruntime as ort

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dribbling_controller/
SINGLE_MJCF = os.path.join(REPO_DIR, "mjcf", "g1_softtouch_dribble.xml")
# default policy/reset pair: the DR-trained v2 checkpoint committed under
# checkpoints/, which was trained with standby-pose reset mixing -> standby reset
DEFAULT_ONNX = os.path.join(REPO_DIR, "checkpoints", "g1_dribble_s3_human_dr_iter80000",
                            "softtouch_dribble_deploy.onnx")
DEFAULT_RESET = os.path.join(REPO_DIR, "config", "g1", "softtouch_mujoco_reset_standby.txt")

ROUTE_CFG = dict(
    routeLength=50.0, routeSegmentLength=0.25, routeLookahead=0.8, routePreviewArc=1.0,
    routeCurvatureMin=0.0, routeCurvatureMax=0.0, routeSFlipArc=2.5,
    routeHumanKappaCap=0.5, routeHumanPersist=0.6, routeHumanWeaveMin=0.4, routeHumanWeaveMax=1.0,
    routeHumanBigProbability=0.09, routeHumanBigAngleMinDeg=40.0, routeHumanBigAngleMaxDeg=180.0,
    routeHumanCumCapDeg=190.0,
    # human-route self-clearance: sections farther apart than the window along
    # the route must stay at least this far apart in space, otherwise the ball's
    # nearest-segment projection (global, as in training) can jump branches.
    # 1.8 m = 2x the off-route fail distance (0.8 m) + margin, and still admits a
    # single 180-deg turn at kappa_eff ~1.05 (legs 1.9 m apart).
    routeMinClearanceM=1.8, routeClearanceWindowM=5.0,
    routeKvScale=0.75, routeVmax=2.0, routeLazyExtend=True, routeInitSegments=9,
    routeExtendChunk=1, routeExtendAheadMarginSegments=10,
    # optional training-style velocity onset (vfix planner): accel-limited speed
    # profile from routeStartSpeed along arc length. None -> legacy step commands.
    routeStartSpeed=None, routeAccelLimit=None,
)
CMD_MODE = 4
JOINT_LIMIT_FACTOR = 0.9
DECIMATION = 4
# Fall criterion, matched term-for-term to the TRAINING termination
# (multiagent_sim/tasks/kick/mdp/terminations.py:fall, which the checkpoints'
# env.yaml selects): pelvis below height_min OR tilted past ~45 deg, where the
# tilt test is on the body-frame projected gravity z. The benchmark used to test
# height alone at 0.4 m -- a very late trigger against a 0.79 m standing pelvis,
# so kneeling / heavily-pitched states well past recovery scored as "survived"
# and survival rates were not comparable to training's.
FALL_Z = 0.5
FALL_TILT_GVEC_Z = -0.7

# Lost-ball ("possession") criterion, also matched to training
# (DribbleRLEnv: BALL_LOST_DIST_THRESHOLD_M / BALL_LOST_GRACE_S): the NEAREST
# FOOT's distance to the ball SURFACE exceeds the threshold continuously for the
# grace period -> sticky ball_lost.
#
# The old criterion measured pelvis-to-ball-CENTRE at 1.5 m for 2.0 s. That is
# ~3x looser than the state training TERMINATES on, and 2 s of continuous loss
# cannot even accumulate in an episode that ends early, so the flag fired on 1
# episode in 3504 and the derived "possession" metric was a flat 100% line.
#
# Training's `_first_touch_done` gate is reproduced as "the foot has been within
# the TIGHTEST threshold at least once this episode" -- see policy_step. It is
# not optional: the reset spawns the ball ~0.65 m ahead, i.e. already outside the
# threshold, so without the gate the sticky flag fires on every episode.
#
# EVAL is deliberately NOT bound to the training threshold. Training TERMINATES
# on 0.5 m foot-surface, which is tight: the ankle_roll body origin sits
# ~0.10-0.15 m BEHIND the toe, so a 0.5 m foot-origin-to-surface trip is only
# ~0.4 m toe-to-surface -- a brief kick past the dribble pocket, not a lost ball.
# So we record the sticky lost flag at a fixed GRID of thresholds: possession is
# read at an eval-appropriate distance (LOST_BALL_MAIN), while the
# training-faithful 0.5 m stays available for train_survival. The grid is fixed
# (not CLI-derived) so the CSV columns are stable across runs.
LOST_BALL_DISTS = (0.5, 0.8, 1.0)
LOST_BALL_MAIN = 0.8              # feeds the `ball_lost` column + possession metric
LOST_BALL_MAIN_IDX = LOST_BALL_DISTS.index(LOST_BALL_MAIN)
LOST_BALL_T = 0.1

# Off-route fail-fast dwell. The threshold used to fire on a SINGLE frame over
# the line, with no dwell, no hysteresis and no first-touch gate -- the strictest
# criterion in the benchmark, applied to the one quantity training never
# terminates on (training's done-set is time_out / fall / ball_lost only). The
# result dominated the failure histogram: 934 of the capability failures were
# off_route, and only 28.6% of those had even reached training's own 0.5 m
# ball-lost state, i.e. most "failures" were still dribbling, just briefly wide
# of the planned line. A dwell makes it mean "left the route", not "clipped the
# boundary for one 20 ms tick"; the first-touch gate mirrors the ball-lost one
# (the reset spawns the ball ahead of the robot, so the route error before the
# first touch is the task setup, not a control failure).
OFFROUTE_GRACE_S = 0.2

# reset jitter (condition key reset_jitter): deterministic clean-env conditions
# need SOME spread to make per-condition survival a probability, not a repeated
# 0/1 outcome
JITTER_YAW_DEG = 5.0
JITTER_XY = 0.02

# Task-start ball placement (condition key reset_ball_random). Training places
# the ball per episode at dist * (body-forward rotated by bearing):
# dribble_env.py RESET_BALL_FORWARD_RANGE / RESET_BALL_BEARING_DEG (active by
# CLASS DEFAULT even when the env.yaml dumps them null -- see env-yaml-null gotcha).
# The range below is ball-to-PELVIS distance: the chest variant (0.43, 0.78) was
# frame-compensated to yield the SAME pelvis distance, so one pelvis-anchored
# range is faithful to both obs frames. Overridable by configure_train_dr.
RESET_BALL_DIST_RANGE = (0.5, 0.85)      # trained ball-to-pelvis distance (m)
RESET_BALL_BEARING_DEG = (-20.0, 20.0)   # trained bearing off body-forward (deg)
RESET_BALL_DIST_DEFAULT = 0.65           # the fixed placement when not randomized

# Training latency DR, sampled when Robot.latency is on (the pysim --latency
# flag) and no condition pins the channels. FALLBACK values (the 2026-06-21 v2
# policy); configure_train_dr() overwrites them from the checkpoint's env.yaml.
# Ball obs lag: per-episode constant d policy steps. Action lag: per-episode
# constant d sub-steps (sim_dt=0.005 -> 5 ms each), zero_prob episodes forced
# to zero, applied at sub-step granularity.
BALL_DELAY_RANGE = (1, 3)
ACT_DELAY_SUBSTEPS = (0, 4)
ACT_DELAY_ZERO_PROB = 0.3
EFFORT_LIMIT = np.array([88., 88., 88., 139., 139., 50., 88., 88., 50., 139., 139.,
                         25., 25., 50., 50., 25., 25., 50., 50., 25., 25., 25., 25.,
                         25., 25., 5., 5., 5., 5.])
JLO = np.array([-2.5307, -2.5307, -2.618, -0.5236, -2.9671, -0.52, -2.7576, -2.7576, -0.52,
                -0.087267, -0.087267, -3.0892, -3.0892, -0.87267, -0.87267, -1.5882, -2.2515,
                -0.2618, -0.2618, -2.618, -2.618, -1.0472, -1.0472, -1.97222, -1.97222,
                -1.61443, -1.61443, -1.61443, -1.61443])
JHI = np.array([2.8798, 2.8798, 2.618, 2.9671, 0.5236, 0.52, 2.7576, 2.7576, 0.52, 2.8798, 2.8798,
                2.6704, 2.6704, 0.5236, 0.5236, 2.2515, 1.5882, 0.2618, 0.2618, 2.618, 2.618,
                2.0944, 2.0944, 1.97222, 1.97222, 1.61443, 1.61443, 1.61443, 1.61443])
JC = 0.5 * (JLO + JHI); JHW = 0.5 * (JHI - JLO) * JOINT_LIMIT_FACTOR

# StandbyController PD gains (config/g1/softtouch_dribble_controllers.yaml). The policy's
# own gains are far softer (it balances actively, ~kp 40/99/28 for hip/knee/ankle); a
# STATIC pose hold needs these stiff gains, so the --standby-hold-s phase uses them.
STANDBY_GAINS = {}
for _s in ("left", "right"):
    STANDBY_GAINS[f"{_s}_hip_pitch_joint"] = (350., 5.)
    STANDBY_GAINS[f"{_s}_hip_roll_joint"] = (200., 5.)
    STANDBY_GAINS[f"{_s}_hip_yaw_joint"] = (200., 5.)
    STANDBY_GAINS[f"{_s}_knee_joint"] = (300., 10.)
    STANDBY_GAINS[f"{_s}_ankle_pitch_joint"] = (300., 5.)
    STANDBY_GAINS[f"{_s}_ankle_roll_joint"] = (150., 5.)
    for _a in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
               "wrist_roll", "wrist_pitch", "wrist_yaw"):
        STANDBY_GAINS[f"{_s}_{_a}_joint"] = (40., 3.)
for _w in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
    STANDBY_GAINS[_w] = (200., 5.)

# --eval ranges = the training DR. FALLBACK values (the v2 human_dr policies);
# configure_train_dr() overwrites them from the checkpoint's env.yaml so a run
# always evaluates against the DR its policy was actually trained with.
DR = dict(ball_mass=(0.352, 0.430), ball_radius=(0.09, 0.11),
          foot_friction=(0.50, 1.00), ball_friction=(0.475, 0.525))

# --sweep ranges = the training range expanded 1.5x (centered): probe a bit past
# the trained envelope but NOT into meaningless OOD territory. Recomputed from
# the checkpoint's env.yaml by configure_train_dr().
SWEEP_RANGES = dict(ball_mass=(0.3325, 0.4495), ball_radius=(0.09, 0.11),
                    foot_friction=(0.375, 1.125), ball_friction=(0.4625, 0.5375))

# Ball roll brake. PhysX (training side) exposes no rolling-friction material, so
# the roll is braked by rigid-body ANGULAR DAMPING c: w_dot = -c*w, which for a
# rolling sphere gives speed decay v_dot = -(2/7)*c*v -> a 1 m/s ball rolls 3.5/c
# metres. MuJoCo's dof_damping is a torque coefficient (tau = -d*w), so d = c*I
# reproduces PhysX's c exactly. FALLBACK = the m16000-era training value;
# configure_train_dr() overwrites it from the checkpoint's env.yaml, and a
# condition's ball_damping key overrides it per condition.
# Reference points: 4.0 = grass calibration (the trained value), 0.9 = the
# 2026-07-17 hardware measurement on the indoor test floor (free-roll k ~0.26/s).
BALL_DAMPING = 4.0

# ---- robot-side DR --------------------------------------------------------
# Everything above perturbs the BALL. These four channels perturb the ROBOT, and
# the benchmark had none of them even though training randomized all four -- so
# every previous run evaluated a nominal-mass, nominal-gain, perfectly-calibrated
# robot with noise-free sensors, i.e. easier than the policy's own training
# distribution. configure_train_dr() fills them from the checkpoint's env.yaml;
# None means "that checkpoint did not randomize it" and the axis falls back to a
# pure stress probe.
OBS_NOISE = {}            # {obs term name: uniform half-width}, applied post-delay
ACTUATOR_GAIN_RANGE = None    # kp multiplier (training: stiffness_distribution_params)
ACTUATOR_DAMPING_RANGE = None # kd multiplier; None -> share ACTUATOR_GAIN_RANGE
PAYLOAD_KG_RANGE = None       # added to torso_link mass
JOINT_OFFSET_RANGE = None     # encoder calibration error on default_joint_pos
BASE_COM_RANGE = None         # {axis: (lo, hi)} offset of torso_link CoM
# leg+waist joint frictionloss (N*m). (lo, hi) -> sampled per episode; a
# degenerate (0, 0) means the checkpoint trained FRICTIONLESS (event removed /
# absent) and the benchmark sets 0 rather than the compiled MJCF 0.1; None ->
# leave the compiled default (only when no env.yaml resolved).
JOINT_FRICTION_RANGE = None

# Robot-DR condition value meaning "draw this channel from the checkpoint's own
# trained range, once per episode". Anything else PINS the channel, and the
# DEFAULT (None) is the deploy nominal: gain 1, no payload, no CoM offset, no
# encoder error.
#
# It used to be the other way round -- an unset channel meant "sample the
# training DR" -- so the `nominal` baseline and EVERY single-factor physics axis
# silently carried four hidden random variables (gain 0.9-1.1, payload 0-3 kg,
# CoM +/-0.05 m, encoder +/-0.03 rad), none of which reached the CSV. That
# contradicted the one-factor-at-a-time protocol the physics axes document
# (conditions.py), made "clean nominal" a misnomer, and left the extra variance
# impossible to condition out after the fact. The trained distribution is still
# reachable -- explicitly, via the `train_dr` baseline condition.
TRAIN_DR = "train"


def _expand(pair, scale, lo_clamp=0.02):
    center, half = 0.5 * (pair[0] + pair[1]), 0.5 * (pair[1] - pair[0]) * scale
    return (max(center - half, lo_clamp), center + half)


def configure_train_dr(train, sweep_scale=1.5):
    """Overwrite DR / SWEEP_RANGES / latency-DR constants in place from a parsed
    checkpoint env.yaml (train_dr.read_train_dr record). A ball channel the
    checkpoint did NOT randomize becomes a degenerate (nominal, nominal) range —
    evaluating a policy against DR it never trained with would misattribute the
    failures — EXCEPT ball radius, which keeps a synthetic +/-10% band around the
    trained nominal (the deploy ball's radius is uncertain regardless of training;
    the band is flagged and NOT sweep-expanded). foot_friction has no nominal in
    the record, so when untrained it KEEPS the hardcoded fallback band (flagged).
    train=None -> keep every hardcoded fallback.
    """
    global BALL_DELAY_RANGE, ACT_DELAY_SUBSTEPS, ACT_DELAY_ZERO_PROB, BALL_DAMPING
    global OBS_NOISE, ACTUATOR_GAIN_RANGE, PAYLOAD_KG_RANGE, JOINT_OFFSET_RANGE
    global BASE_COM_RANGE, RESET_BALL_DIST_RANGE, RESET_BALL_BEARING_DEG
    global JOINT_FRICTION_RANGE, ACTUATOR_DAMPING_RANGE
    if train is None:
        print("[train_dr] WARNING: no env.yaml — using hardcoded legacy DR ranges")
        return
    # Ball task-start placement. An explicit value overrides; an explicit null
    # keeps the (always-active) class-default range. But a dump MISSING the keys
    # entirely predates the feature, so that checkpoint trained with the ball
    # FIXED straight ahead -- handing it today's randomized start would evaluate
    # it out of its own distribution and charge the resulting failures to
    # whatever axis happened to be running.
    if train.get("reset_ball_dr_absent"):
        RESET_BALL_DIST_RANGE = (RESET_BALL_DIST_DEFAULT, RESET_BALL_DIST_DEFAULT)
        RESET_BALL_BEARING_DEG = (0.0, 0.0)
        print(f"[train_dr] ball start: env.yaml has no reset_ball_* keys — this "
              f"checkpoint predates task-start DR, pinning {RESET_BALL_DIST_DEFAULT} m "
              f"straight ahead (was inheriting today's randomized start)")
    else:
        if train.get("reset_ball_forward_range") is not None:
            RESET_BALL_DIST_RANGE = tuple(train["reset_ball_forward_range"])
        if train.get("reset_ball_bearing_deg") is not None:
            RESET_BALL_BEARING_DEG = tuple(train["reset_ball_bearing_deg"])
    synthetic = []
    nominal = dict(ball_mass=train["ball_mass_nominal"], ball_radius=train["ball_radius_nominal"],
                   ball_friction=train["ball_friction_nominal"])
    for key in ("ball_mass", "ball_radius", "ball_friction", "foot_friction"):
        rng = train[f"{key}_range"]
        if rng is not None:
            DR[key] = (float(rng[0]), float(rng[1]))
            SWEEP_RANGES[key] = _expand(DR[key], sweep_scale)
        elif key == "ball_radius" and nominal["ball_radius"] is not None:
            DR[key] = SWEEP_RANGES[key] = (0.9 * nominal["ball_radius"], 1.1 * nominal["ball_radius"])
            synthetic.append(key)
        elif nominal.get(key) is not None:
            DR[key] = SWEEP_RANGES[key] = (float(nominal[key]), float(nominal[key]))
        else:
            print(f"[train_dr] WARNING: {key} missing from env.yaml — keeping the "
                  f"hardcoded fallback range {DR[key]}")
    if train["ball_obs_delay_steps"] is not None:
        BALL_DELAY_RANGE = train["ball_obs_delay_steps"]
    else:
        BALL_DELAY_RANGE = (0, 0)
    if train["action_delay_ms"] is not None:
        ACT_DELAY_SUBSTEPS = (int(round(train["action_delay_ms"][0] / 5.0)),
                              int(round(train["action_delay_ms"][1] / 5.0)))
        if train["action_delay_zero_prob"] is not None:
            ACT_DELAY_ZERO_PROB = float(train["action_delay_zero_prob"])
    else:
        ACT_DELAY_SUBSTEPS = (0, 0)
    for key in DR:
        tag = " (synthetic +/-10% band)" if key in synthetic else ""
        print(f"[train_dr] {key}: DR [{DR[key][0]:.4g}, {DR[key][1]:.4g}]"
              f"  sweep(x{sweep_scale:g}) [{SWEEP_RANGES[key][0]:.4g}, {SWEEP_RANGES[key][1]:.4g}]{tag}")
    if train.get("ball_damping") is not None:
        BALL_DAMPING = float(train["ball_damping"])
    print(f"[train_dr] latency: ball obs {BALL_DELAY_RANGE} steps, action "
          f"{tuple(5 * s for s in ACT_DELAY_SUBSTEPS)} ms (zero_prob {ACT_DELAY_ZERO_PROB:g})")
    OBS_NOISE = dict(train.get("obs_noise") or {})
    ACTUATOR_GAIN_RANGE = train.get("actuator_gain_range")
    # training draws stiffness and damping SEPARATELY (both "scale" operations on
    # the same event); an env.yaml carrying only the stiffness pair shares it
    ACTUATOR_DAMPING_RANGE = train.get("actuator_damping_range") or ACTUATOR_GAIN_RANGE
    PAYLOAD_KG_RANGE = train.get("payload_kg_range")
    JOINT_OFFSET_RANGE = train.get("joint_offset_range")
    BASE_COM_RANGE = train.get("base_com_range")
    # joint friction: a real trained range if the event was active, else a
    # degenerate (0, 0) meaning the checkpoint trained frictionless (event
    # removed / predates it) -> the benchmark sets 0, NOT the compiled MJCF 0.1
    jf = train.get("joint_friction_range")
    JOINT_FRICTION_RANGE = tuple(jf) if jf is not None else (0.0, 0.0)
    print(f"[train_dr] joint friction: "
          + ("trained frictionless — nominal 0 (overrides the MJCF 0.1 default)"
             if JOINT_FRICTION_RANGE == (0.0, 0.0)
             else f"trained range {JOINT_FRICTION_RANGE} N*m (sampled per episode)"))
    if not OBS_NOISE:
        print("[train_dr] WARNING: no obs noise in env.yaml — the benchmark will "
              "run NOISE-FREE, which is easier than the training distribution")
    print(f"[train_dr] ball_damping c = {BALL_DAMPING:g} "
          f"(roll decay k = {2 / 7 * BALL_DAMPING:.3g}/s, 1 m/s ball rolls "
          f"{3.5 / BALL_DAMPING:.2f} m)")

# distinct per-robot colors (ball + its route dots) so each trajectory is identifiable
COLORS = [(0.90, 0.25, 0.25), (0.25, 0.80, 0.35), (0.30, 0.55, 1.00), (0.95, 0.85, 0.15),
          (0.85, 0.40, 0.95), (0.20, 0.85, 0.85), (0.95, 0.55, 0.15), (0.6, 0.6, 0.6)]


def rot_vec(quat_wxyz, vec):
    res = np.zeros(3)
    mujoco.mju_rotVecQuat(res, np.ascontiguousarray(vec, dtype=np.float64),
                          np.ascontiguousarray(quat_wxyz, dtype=np.float64))
    return res


def world_to_body(quat_wxyz, vec):
    negq = np.zeros(4); res = np.zeros(3)
    mujoco.mju_negQuat(negq, np.ascontiguousarray(quat_wxyz, dtype=np.float64))
    mujoco.mju_rotVecQuat(res, np.ascontiguousarray(vec, dtype=np.float64), negq)
    return res


def csv_floats(s):
    return np.array([float(x) for x in re.split(r"[,\s]+", s.strip()) if x != ""])


# One eval "condition" bundles everything a test condition can vary per episode.
#
# Route     route_mode: "human" (trained generator) / "straight" / "arc" (or a raw
#           int cmd_mode). Arc conditions take arc_kappa (signed 1/m), a straight
#           lead_in_m (scalar, or [min,max] drawn per episode from the route rng),
#           and optionally arc_angle_deg [min,max] for ONE finite turn (None = an
#           endless arc). route_vmax / route_len_m override the global route config.
# Failure   offroute_fail_m: ball this far from the route -> episode fails now.
#           ball_far_fail_m: ball this far from the robot -> episode fails now.
#           (None = no fail-fast; robustness conditions run the full episode.)
# Timing    episode_s overrides the global --episode-s for this condition.
# Physics   dr pins the DR params exactly; dr_scale samples the CENTERED training
#           ranges scaled by alpha (0 -> centers, 1 -> training DR); neither ->
#           sample the full training DR. push_dv / ball_push_dv kick the base/ball
#           every push_interval_s (random phase + direction).
# Latency   ball_obs_delay_steps / action_delay_ms pin a channel exactly;
#           None -> the --latency flag decides (random training latency DR).
# Tracking  record_speed_pairs: keep per-step (commanded speed, ball speed along
#           the commanded direction) pairs for the speed-controllability test.
DEFAULT_CONDITION = dict(
    route_mode="human", arc_kappa=None, route_vmax=None, route_len_m=None,
    human_kappa_cap=None, lead_in_m=1.0, arc_angle_deg=None,
    offroute_fail_m=None, ball_far_fail_m=None, episode_s=None,
    push_dv=0.0, ball_push_dv=0.0, push_interval_s=5.0,
    ball_obs_delay_steps=None, action_delay_ms=None, reset_jitter=False,
    dr=None, dr_scale=None, ball_damping=None, record_speed_pairs=False,
    route_start_speed=None, route_accel_limit=None,
    # Task-start ball placement. reset_ball_random=False -> the fixed
    # RESET_BALL_DIST_DEFAULT straight ahead (bit-identical to the legacy reset,
    # no rng consumed). True -> place at dist * (forward rotated by bearing),
    # training-faithful; reset_ball_dist / reset_ball_bearing pin a value
    # (the robustness axes), None -> sample the trained range.
    reset_ball_random=False, reset_ball_dist=None, reset_ball_bearing=None,
    # Deploy hand-off probe (the `handover` robustness axis): stiff-gain standby
    # hold for this many seconds before the policy takes over. None -> the
    # whole-run --standby-hold-s default.
    standby_hold_s=None,
    # Robot-side DR. None = the DEPLOY NOMINAL (gain 1, no payload, no CoM
    # offset, no encoder error); engine.TRAIN_DR = draw from the checkpoint's own
    # trained range; a number pins it. obs_noise_scale / base_com_scale are
    # MULTIPLES of the trained magnitude, the rest are absolute.
    # ball_radius_obs_m is an absolute BELIEVED radius, not a scale.
    #
    # obs_noise_scale defaults to 0, NOT 1, and that is a deliberate protocol
    # choice rather than an oversight. Measured 2026-07-20 on
    # g1_dribble_s3_human_dr_iter80000, nominal condition, n=24: fall rate 0.50
    # noise-free vs 0.88 at 1x the trained noise. A baseline that already fails
    # 88% of the time floors every perturbation axis and destroys the
    # benchmark's ability to separate checkpoints.
    # The deeper reason: training noise is a REGULARIZER, not a measurement of
    # the deployment sensors -- nobody has measured those. This mirrors how
    # conditions.DEPLOY_LATENCY pins latency to the deployment stack's value
    # instead of sampling the training range. The obs_noise sweep group carries
    # the sensitivity information (0 -> 4x), which is where it belongs.
    obs_noise_scale=0.0, actuator_gain_scale=None, payload_kg=None,
    base_com_scale=None, joint_offset_rad=None, ball_radius_obs_m=None,
    # leg+waist joint frictionloss (absolute N*m). None -> the trained nominal
    # (0 for the frictionless lineage); a scalar pins it (the joint_friction axis).
    joint_friction=None)


def make_condition(**overrides):
    unknown = set(overrides) - set(DEFAULT_CONDITION) - {"name", "group", "axis"}
    if unknown:
        raise ValueError(f"unknown condition keys: {sorted(unknown)}")
    return {**DEFAULT_CONDITION, **overrides}


def route_cmd_mode(condition):
    """Canonical int cmd_mode ('human'/4 -> 4, 'straight'/0 -> 0, 'arc' -> 1)."""
    mode = condition["route_mode"]
    return int(mode) if not isinstance(mode, str) else {"human": 4, "straight": 0, "arc": 1}[mode]


def pearson_r(x, y, min_samples=100, min_x_std=1e-2):
    """Pearson correlation, NaN when there is not enough data or the command
    barely varies (a constant command carries no controllability signal)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < min_samples or x.std() < min_x_std or y.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


class Route:
    def __init__(self, cfg, seed):
        self.cfg = dict(cfg)   # own copy: --matrix conditions override vmax/length per robot
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.const_kappa = None   # not None -> constant-curvature arc (signed 1/m)
        self.lead_segments = 0    # straight lead-in before the arc starts
        self.lead_range = None    # (min,max) m -> lead length drawn per episode from route rng
        self.vmax_range = None    # (min,max) m/s -> per-episode cruise pace, like the
                                  # training ROUTE_CRUISE_RANGE sampling
        self.arc_deg = None       # (min,max) deg -> FINITE arc, angle drawn per episode;
        self.arc_segments = None  #                  None -> infinite arc (full circles)
        self.arc_kappa_eff = None  # back-solved effective curvature of a finite arc
        self.arc_end_s = None     # arc-exit arc length (lead + arc), for completion checks
        self._ramp_v = None       # accel-limited speed carried across lazy extensions
        self._alloc()
        self.filled = 0; self.end_heading = 0.0; self.last_seg = -1
        self.h_sign = 1.0; self.big_remain = 0.0; self.big_sign = 1.0
        self.heading_cum = 0.0
        self.last_s = 0.0; self.max_s = 0.0   # ball arc-length progress along the route

    def _alloc(self):
        n = max(1, int(round(self.cfg["routeLength"] / max(self.cfg["routeSegmentLength"], 1e-9))))
        if not hasattr(self, "speed") or len(self.speed) != n:
            self.points = np.zeros((n + 1, 2)); self.speed = np.zeros(n)

    def _u(self, lo, hi): return self.rng.uniform(lo, hi)

    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v); return np.array([1.0, 0.0]) if n < 1e-9 else v / n

    def reset(self, origin, forward, cmd_mode):
        self.cmd_mode = cmd_mode; self.cmd_sign = -1.0 if cmd_mode == 2 else 1.0
        self.last_seg = -1; self.filled = 0
        self.last_s = 0.0; self.max_s = 0.0
        self._alloc()   # condition overrides may have changed routeLength
        # per-episode cruise pace (training samples ROUTE_CRUISE_RANGE the same way);
        # drawn from the route rng -> route_seed reproduces the pace too
        if self.vmax_range is not None:
            self.cfg["routeVmax"] = float(self._u(*self.vmax_range))
        # turn-into-corner test: draw per-episode lead length / arc angle from the
        # route rng (route_seed-controlled -> the same seed reproduces the same
        # lead+angle across conditions and experiments, pairing the comparisons)
        if self.const_kappa is not None:
            ds = self.cfg["routeSegmentLength"]
            if self.lead_range is not None:
                self.lead_segments = max(0, int(round(self._u(*self.lead_range) / ds)))
            if self.arc_deg is not None:
                # sharp turns span only a few coarse segments (kappa*ds up to
                # 1 rad/segment), so keeping the sampled kappa would skew the
                # swept angle by up to ~kappa*ds/2. Same fix as the training
                # route builder: ceil the segment count, then back-solve the
                # EFFECTIVE kappa = angle/(nseg*ds) — the swept angle is exact
                # and |kappa_eff| <= sampled (never tighter).
                th = np.deg2rad(self._u(*self.arc_deg))
                self.arc_segments = max(3, int(np.ceil(th / (abs(self.const_kappa) * ds))))
                self.arc_kappa_eff = np.sign(self.const_kappa) * th / (self.arc_segments * ds)
                self.arc_end_s = (self.lead_segments + self.arc_segments) * ds
            else:
                self.arc_segments = None; self.arc_kappa_eff = None; self.arc_end_s = None
        max_seg = len(self.speed)
        origin = np.asarray(origin, float); forward = self._unit(np.asarray(forward, float))
        if cmd_mode == 4:
            # human routes are built EAGERLY and re-drawn (same rng stream, so a
            # given route_seed stays deterministic) until no two far-apart route
            # sections come closer than the clearance -> the ball can never sit
            # nearer to another branch of its own route
            best_clearance, best_state = -np.inf, None
            for _ in range(30):
                self._build(max_seg, True, origin, forward)
                clearance = self._self_clearance()
                if clearance >= self.cfg["routeMinClearanceM"]:
                    best_state = None
                    break
                if clearance > best_clearance:
                    best_clearance = clearance
                    best_state = (self.points.copy(), self.speed.copy(),
                                  self.end_heading, self.filled)
            if best_state is not None:
                self.points, self.speed, self.end_heading, self.filled = best_state
        else:
            init = (int(np.clip(self.cfg["routeInitSegments"], 1, max_seg))
                    if self.cfg["routeLazyExtend"] else max_seg)
            self._build(init, True, origin, forward)
        return self.update(origin)

    def _self_clearance(self):
        """Smallest spatial distance between route points farther apart than the
        clearance window along the route (inf when the route is short)."""
        ds = self.cfg["routeSegmentLength"]
        window = int(self.cfg["routeClearanceWindowM"] / ds)
        pts = self.points[: self.filled + 1]
        if len(pts) <= window + 1:
            return np.inf
        dist = np.hypot(*(pts[None, :, :] - pts[:, None, :]).transpose(2, 0, 1))
        index = np.arange(len(pts))
        far = np.abs(index[None, :] - index[:, None]) > window
        return float(dist[far].min())

    def update(self, ball_xy):
        """Serve the route command for the current ball position.

        This reproduces training's LEGACY serve (dribble_env.py: nearest-segment
        projection + per-segment speed lookup, no rate limiter, no served-speed
        accel cap), which is what every checkpoint under checkpoints/ trained on
        -- their env.yaml either lacks `route_v2_vel` or carries null. A
        route_v2_vel=True checkpoint would need the v2 serve instead: an s*
        projection rate limiter (ROUTE_PROJ_RATE_M = 0.06 m/step) and a 3.0 m/s^2
        clamp on the served speed. Benchmarking one against this serve would
        measure the serve, not the policy.

        One known and accepted deviation from training's timing: the command is
        recomputed from the CURRENT ball state just before the observation is
        built, whereas Isaac Lab's manager order leaves training's command one
        policy tick stale. At 50 Hz that is 20 ms / ~0.04 m at 2 m/s, far below
        the 0.8 m off-route threshold.
        """
        self._extend(); ball_xy = np.asarray(ball_xy, float)
        filled = max(1, self.filled)
        # a tight constant-curvature arc revisits the same xy after one lap, so a
        # GLOBAL nearest-segment projection can jump back a whole lap; restrict the
        # search to a window around the last matched segment (the ball moves less
        # than one segment per control period).
        lo, hi = 0, filled
        if self.const_kappa is not None and self.last_seg >= 0:
            lo = max(0, self.last_seg - 4); hi = min(filled, self.last_seg + 12)
        best_d2 = np.inf; best_t = 0.0; best_seg = lo; best_proj = self.points[lo]
        for i in range(lo, hi):
            a = self.points[i]; b = self.points[i + 1]; ab = b - a
            ab2 = max(ab @ ab, 1e-9)
            t = np.clip((ball_xy - a) @ ab / ab2, 0.0, 1.0); proj = a + t * ab
            d2 = (ball_xy - proj) @ (ball_xy - proj)
            if d2 < best_d2:
                best_d2, best_t, best_seg, best_proj = d2, t, i, proj
        self.last_seg = best_seg
        s = (best_seg + best_t) * self.cfg["routeSegmentLength"]
        self.last_s = s; self.max_s = max(self.max_s, s)
        nsi = int(np.clip(np.floor((s + self.cfg["routeLookahead"]) / self.cfg["routeSegmentLength"]),
                          0, len(self.speed) - 1))
        return dict(target_speed=self.speed[best_seg], next_target_speed=self.speed[nsi],
                    target_dir=self._unit(self._point_at(s + self.cfg["routeLookahead"]) - ball_xy),
                    next_target_dir=self._unit(self._point_at(s + self.cfg["routeLookahead"] + self.cfg["routePreviewArc"]) - ball_xy),
                    crosstrack=float(np.sqrt(best_d2)))

    def _point_at(self, arc):
        max_f = max(0.0, self.filled - 1e-4)
        f = min(max(arc / self.cfg["routeSegmentLength"], 0.0), max_f)
        i = int(np.clip(int(f), 0, len(self.points) - 2)); frac = f - i
        return self.points[i] + frac * (self.points[i + 1] - self.points[i])

    def _build(self, num, init, origin=None, forward=None):
        ds = self.cfg["routeSegmentLength"]; seg_off = 0 if init else self.filled
        org = origin if init else self.points[seg_off]; theta = self.end_heading
        if init:
            theta = np.arctan2(forward[1], forward[0])
            self.h_sign = 1.0 if self._u(0, 1) < 0.5 else -1.0
            self.big_remain = 0.0; self.big_sign = 1.0; self.points[0] = org
            self.heading_cum = 0.0   # net signed heading for the loop governor
        if self.const_kappa is not None:
            # constant-curvature arc (capability axis): straight lead-in so the
            # reset transient settles before the turn starts, then constant kappa
            # (finite arc_segments -> straight exit tail after the turn).
            # Speed still follows the trained law min(vmax, sqrt(kv/|kappa|)).
            arc_kappa = self.const_kappa if self.arc_segments is None else self.arc_kappa_eff
            kappa = np.zeros(num)
            for i in range(num):
                gi = seg_off + i
                if gi >= self.lead_segments and (self.arc_segments is None
                                                 or gi < self.lead_segments + self.arc_segments):
                    kappa[i] = arc_kappa
        elif self.cmd_mode == 4:
            kappa = self._human_kappa(num)
        else:
            kappa = np.zeros(num)   # mode 0: straight line, speed = constant vmax
        heading = theta; point = np.asarray(org, float).copy()
        if init:
            # ramp state restarts on every fresh build (incl. clearance re-draws);
            # lazy extension continues from the stored end-of-route speed
            self._ramp_v = self.cfg.get("routeStartSpeed")
        for i in range(num):
            point = point + np.array([np.cos(heading), np.sin(heading)]) * ds
            self.points[seg_off + 1 + i] = point
            kabs = max(abs(kappa[i]), 1e-3)
            v = min(self.cfg["routeVmax"], np.sqrt(self.cfg["routeKvScale"] / kabs))
            if self.cfg.get("routeAccelLimit"):
                v0 = v if self._ramp_v is None else self._ramp_v
                v = min(v, np.sqrt(v0 * v0 + 2.0 * self.cfg["routeAccelLimit"] * ds))
                self._ramp_v = v
            self.speed[seg_off + i] = v
            heading += kappa[i] * ds
        self.end_heading = heading; self.filled = seg_off + num

    def _human_kappa(self, num):
        cap = self.cfg["routeHumanKappaCap"]; ds = self.cfg["routeSegmentLength"]
        amin = np.deg2rad(self.cfg["routeHumanBigAngleMinDeg"]); amax = np.deg2rad(self.cfg["routeHumanBigAngleMaxDeg"])
        cum_cap = np.deg2rad(self.cfg["routeHumanCumCapDeg"])
        out = np.zeros(num)
        for i in range(num):
            in_big = self.big_remain > 0.0
            if not in_big and self._u(0, 1) < self.cfg["routeHumanBigProbability"]:
                angle = self._u(amin, amax)
                self.big_remain = max(2.0, np.ceil(angle / (cap * ds)))
                sign = 1.0 if self._u(0, 1) < 0.5 else -1.0
                # loop GOVERNOR (as in the training route builder): flip the turn
                # direction when a same-direction turn would push the net signed
                # heading past +/-cum_cap -> the route can never close a loop.
                cum_sign = np.sign(self.heading_cum)
                if (abs(self.heading_cum) + angle) > cum_cap and sign == cum_sign != 0.0:
                    sign = -cum_sign
                self.big_sign = sign; in_big = True
            if self._u(0, 1) > self.cfg["routeHumanPersist"]:
                self.h_sign = -self.h_sign
            # weave-side governor: this generator's weave is strong (0.4-1.0 x cap,
            # unlike the near-straight cruise of the new training one), so weave
            # drift alone can wind past the cap — beyond it, weave only back to 0
            if abs(self.heading_cum) > cum_cap and self.h_sign == np.sign(self.heading_cum):
                self.h_sign = -np.sign(self.heading_cum)
            mag = self._u(self.cfg["routeHumanWeaveMin"], self.cfg["routeHumanWeaveMax"]) * cap
            out[i] = self.big_sign * cap if in_big else self.h_sign * mag
            if in_big: self.big_remain -= 1.0
            self.heading_cum += out[i] * ds
        return out

    def _extend(self):
        if not self.cfg["routeLazyExtend"] or self.last_seg < 0: return
        max_seg = len(self.speed)
        if self.filled >= max_seg or (self.filled - self.last_seg) >= self.cfg["routeExtendAheadMarginSegments"]: return
        num = min(self.cfg["routeExtendChunk"], max_seg - self.filled)
        if num > 0: self._build(num, False)


def build_world(n_robots, spacing, onnx_path, reset_path, seed, visual=True):
    """Compose the multi-robot model and its Robot wrappers (shared by main() and
    the sim2sim_benchmark package).

    visual=False strips the render-only meshes (see _single_robot_xml) — pass it
    for any world that is never rendered; physics is unaffected."""
    model, grid = compose(n_robots, spacing, visual=visual)
    data = mujoco.MjData(model)
    print(f"[multi] policy: {onnx_path}")
    # tiny batch-1 MLPs: ORT's default core-count threadpool only spin-waits
    # (inflating CPU% ~10x for zero speedup) -> pin it small
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, sess_options=so,
                                   providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    # log the obs contract, not just the path: feeding a chest-frame policy
    # pelvis-frame observations does not error, it just produces quietly awful
    # numbers (measured: straight-line success 92% -> 0%), and the only record
    # of which frame a finished run used was buried in the ONNX metadata
    print(f"[multi] obs frame: {meta.get('obs_frame', 'pelvis (no obs_frame key)')}"
          f" | history {meta.get('actor_history_length', '1')}"
          f" | actor obs {meta.get('actor_observation_dim', '?')}"
          f" | latent {meta.get('latent_dim', '?')}")
    reset_state = parse_reset(reset_path)
    robots = [Robot(model, k, grid[k], session, meta, reset_state, seed)
              for k in range(n_robots)]
    return model, data, robots


def step_control_period(model, data, robots, standby_hold_s=0.0):
    """Run one 50 Hz control period (policy once, PD torque every physics sub-step)
    and return the indices of robots whose episode just ended (fall / fail-fast /
    episode-length timeout). The start-phase override is PER-ROBOT (resolved at
    reset from the condition + whole-run defaults): a stiff standby HOLD (deploy
    hand-off probe) or a soft policy SETTLE (training settle_time_range_s). The
    standby_hold_s arg is retained for signature stability but no longer read --
    rb.hold_s / rb.settle_s carry it."""
    for rb in robots:
        warm = data.time - rb.ep_start
        rb.policy_step(data, hold=warm < rb.hold_s,
                       settle=(rb.hold_s <= 0.0 and warm < rb.settle_s))
        rb.maybe_push(data)
    for _ in range(DECIMATION):
        for rb in robots:
            rb.apply(data)
        mujoco.mj_step(model, data)
    t = data.time
    return [j for j, rb in enumerate(robots)
            if rb.fell(data) or rb.fail_reason or (t - rb.ep_start) >= rb.episode_len]


@functools.lru_cache(maxsize=2)
def _single_robot_xml(visual):
    """The single-robot MJCF, optionally with every mesh asset and mesh geom
    stripped out.

    The 35 STL meshes dominate a compiled model's memory, and `compose()` pays
    for them once PER ROBOT (MjSpec.attach copies assets, so nmesh scales with
    n). A 32-robot world costs ~3.4 GB with meshes and a fraction of that
    without — and a statistics run never renders, so it is paying purely for
    geometry nothing reads. Fanning such workers out is what OOM-killed the
    2026-07-20 benchmark runs.

    Stripping is physically exact, on three checks against this MJCF:
      - all 38 mesh geoms are contype=0/conaffinity=0, so they generate no
        contacts (verified: the set of (contype, conaffinity) over mesh geoms
        is {(0, 0)});
      - every robot link carries an explicit <inertial>, so no geom feeds
        inertia (the bodies without one are the ball and the route dots, which
        are primitives, not meshes);
      - floors are planes and route dots are spheres, so both survive.
    Both textures are `builtin` and both materials are inline, so the stripped
    XML has no external asset dependencies and loads via from_string.

    Prefer this over the compiler's `discardvisual`, which keys off
    contype==0 && conaffinity==0 and would therefore also drop the route dots
    that Robot.__init__ recolors through model.body_geomadr.
    """
    if visual:
        return open(SINGLE_MJCF).read()
    root = ET.parse(SINGLE_MJCF).getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "mesh" or (child.tag == "geom" and "mesh" in child.attrib):
                parent.remove(child)
    return ET.tostring(root, encoding="unicode")


def compose(n, spacing, visual=True):
    sp = mujoco.MjSpec()
    sp.option.timestep = 0.005
    sp.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    sp.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    sp.option.impratio = 10.0
    sp.visual.global_.offwidth = 1280   # larger offscreen framebuffer for --record
    sp.visual.global_.offheight = 720
    cols = int(np.ceil(np.sqrt(n)))
    grid = []
    for k in range(n):
        gx = (k % cols) * spacing; gy = (k // cols) * spacing
        child = mujoco.MjSpec.from_string(_single_robot_xml(visual))
        fr = sp.worldbody.add_frame(); fr.pos = [gx, gy, 0.0]
        sp.attach(child, prefix=f"r{k}_", frame=fr)
        grid.append((gx, gy))
    # the 4 stacked copies each bring their own light + floor -> dedupe so we don't
    # get 4x lighting (overexposure) or 4 coincident floor planes (z-fighting).
    for li, light in enumerate(sp.lights):
        if li > 0:
            light.active = 0
    sp.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    sp.visual.headlight.ambient = [0.3, 0.3, 0.3]
    sp.visual.headlight.specular = [0.0, 0.0, 0.0]
    # CRITICAL: the floor is conaffinity=1, so N coincident infinite floor planes would
    # auto-collide with every ball -> N redundant ball-floor contacts -> solver blows the
    # robots over. This used to be solved by zeroing contype/conaffinity on ALL floors,
    # leaving only the explicit foot-floor / ball-floor <pair>s -- which also deleted
    # every NON-FOOT body's ground contact: knees, shins, pelvis, torso, head, arms and
    # hands passed straight through the floor, so a robot could never kneel, scuff or
    # catch itself, and near-fall states went straight to the height/tilt fall test.
    # Training has no such hole (TerrainImporterCfg(collision_group=-1): every collision
    # link hits the ground). Instead give floor k the SAME contype/conaffinity bit as
    # robot k below, so each robot sees exactly one floor and no other robot's -- the
    # redundant-plane problem is solved by the bit, not by deleting the collision.
    rx = re.compile(r"^r(\d+)_")
    floors = [g for g in sp.geoms if (g.name or "").endswith("floor")]
    for gi, g in enumerate(floors):
        mt = rx.match(g.name or "")
        bit = 1 << (int(mt.group(1)) % 31) if mt else 1
        g.contype = bit; g.conaffinity = bit
        if gi > 0:
            rgba = list(g.rgba); rgba[3] = 0.0; g.rgba = rgba  # hide the coincident copies
    # ISOLATE robots: give every robot's auto-colliding geoms (body capsules + feet + ball)
    # a UNIQUE contype/conaffinity bit. Within robot k all share bit k -> self-collisions are
    # identical to single-robot; across robots bit_a & bit_b = 0 -> NO collision. The foot-floor
    # / foot-ball / ball-floor contacts are explicit <pair>s (ignore contype) so they're
    # untouched. This removes the cross-robot collision volume that knocked neighbours over at
    # tight (video) spacing, without changing any single robot's physics. (>31 robots reuse a
    # bit, but eval/sweep run at 120m spacing where contact is geometrically impossible anyway.)
    # (Floors were already given their robot's bit above; re-stamping them here is a no-op.)
    for g in sp.geoms:
        if not (g.contype or g.conaffinity):   # skip visuals / route dots
            continue
        mt = rx.match(g.name or "")
        if mt:
            bit = 1 << (int(mt.group(1)) % 31)
            g.contype = bit; g.conaffinity = bit
    return sp.compile(), grid


def refresh_model_constants(model, data):
    """Recompute the mjModel constants derived from mass / inertia / geom size.

    Robot.apply_dr and Robot.apply_robot_dr hot-edit body_mass, body_inertia,
    body_ipos, geom_size and geom_rbound on the ALREADY-COMPILED model. MuJoCo
    caches quantities derived from those (body_subtreemass, dof_M0,
    dof_invweight0 / body_invweight0, actuator_acc0 ...) at compile time and does
    NOT refresh them on assignment, so without this the solver keeps running on
    the pre-DR robot's constants. Measured on this MJCF: a 3 kg torso payload
    leaves body_subtreemass at 38.256 instead of 41.256, and the constraint
    invweights feeding contact impedance are off by up to 235 -- a 600-step
    rollout diverges from the first contact step on (2.4e-4 m/s in qvel at step
    1, growing chaotically from there). It is a real dynamics error, not a
    cosmetic one.

    mj_setConst uses `data` as scratch and OVERWRITES data.qpos with qpos0 (it
    computes the constants at the default configuration), so the live state is
    saved and restored around it -- verified bit-identical. qvel and time are
    untouched by mj_setConst, and every position-dependent field it scribbles on
    is recomputed by the next mj_step / mj_kinematics.
    """
    qpos = data.qpos.copy()
    mujoco.mj_setConst(model, data)
    data.qpos[:] = qpos


def parse_reset(path):
    rs = {}
    for line in open(path):
        t = line.split()
        if t and not t[0].startswith("#"):
            rs[t[0]] = t[1:]
    return rs


class Robot:
    def __init__(self, model, k, grid, sess, meta, rs, seed):
        self.k = k; self.pfx = f"r{k}_"; self.gx, self.gy = grid; self.sess = sess
        # kp0/kd0/dq0 are the PRISTINE values; the per-episode robot-side DR
        # writes self.kp/self.kd/self.dq from them, so a scale never compounds
        # across episodes
        self.kp0 = csv_floats(meta["joint_stiffness"]); self.kd0 = csv_floats(meta["joint_damping"])
        self.kp = self.kp0.copy(); self.kd = self.kd0.copy()
        self.skp = np.array([STANDBY_GAINS[n][0] for n in meta["joint_names"].split(",")])
        self.skd = np.array([STANDBY_GAINS[n][1] for n in meta["joint_names"].split(",")])
        self._holding = False
        self.ascale = csv_floats(meta["action_scale"])
        self.dq0 = csv_floats(meta["default_joint_pos"]); self.dq = self.dq0.copy()
        self.jnames = meta["joint_names"].split(",")
        # actor obs is built term-by-term in this order so one code path serves the
        # 82-dim (invariant), 90-dim (world-frame) and 83x10 (history) policies.
        self.actor_names = meta["actor_obs_names"].split(",")
        # single-frame dims for the actor terms (first len(actor_names) of observation_dims)
        all_dims = [int(x) for x in meta["observation_dims"].split(",")]
        self.actor_dims = all_dims[: len(self.actor_names)]
        self.sf_dim = sum(self.actor_dims)                         # single-frame actor width
        self.hist_len = int(meta.get("actor_history_length", "1"))  # 10 for the v2 policy
        # obs frame for ball_pos_b / ball_lin_vel_b / target_dir_b: "pelvis" (root,
        # the default) or "chest" (torso_link + local offset) for policies trained
        # with --v2_body_frame. base_ang_vel / projected_gravity stay root-frame in
        # BOTH cases — training reads them through the stock isaaclab mdp functions.
        self.obs_frame = meta.get("obs_frame", "pelvis")
        self.chest_body = None
        if self.obs_frame == "chest":
            self.chest_body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY,
                self.pfx + meta.get("obs_frame_body", "torso_link"))
            if self.chest_body < 0:
                raise RuntimeError(f"obs_frame=chest: body "
                                   f"{meta.get('obs_frame_body', 'torso_link')} not in the model")
            self.chest_offset = csv_floats(meta.get("obs_frame_offset", "0.077,0,0.148"))
        # per-term [start, stop) column ranges inside one single frame
        offs = np.concatenate([[0], np.cumsum(self.actor_dims)])
        self.term_cols = [(int(offs[i]), int(offs[i + 1])) for i in range(len(self.actor_names))]
        self.obs_hist = None    # (hist_len, sf_dim) ring, row 0 = oldest; filled on first _obs
        self.latency = False    # set per-robot in main() from --latency
        self.rng = np.random.Generator(np.random.PCG64(1000 + seed + 17 * k))
        nid = lambda t, nm: mujoco.mj_name2id(model, t, self.pfx + nm)
        self.qadr = np.array([model.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in self.jnames])
        self.vadr = np.array([model.jnt_dofadr[nid(mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in self.jnames])
        self.aadr = np.array([nid(mujoco.mjtObj.mjOBJ_ACTUATOR, nm) for nm in self.jnames])
        bj = nid(mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        self.bq = model.jnt_qposadr[bj]; self.bv = model.jnt_dofadr[bj]
        balj = nid(mujoco.mjtObj.mjOBJ_JOINT, "softtouch_ball_freejoint")
        self.ballq = model.jnt_qposadr[balj]; self.ballv = model.jnt_dofadr[balj]
        self.ball_body = nid(mujoco.mjtObj.mjOBJ_BODY, "softtouch_ball")
        self.ball_geom = nid(mujoco.mjtObj.mjOBJ_GEOM, "softtouch_ball_geom")
        # feet, for the training-matched lost-ball criterion (nearest foot to the
        # ball SURFACE, not pelvis to ball centre)
        self.foot_bodies = [nid(mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
                            for side in ("left", "right")]
        # payload / CoM DR target torso_link (env.yaml asset_cfg.body_names)
        self.torso_body = nid(mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.torso_mass0 = float(model.body_mass[self.torso_body])
        self.torso_ipos0 = model.body_ipos[self.torso_body].copy()
        self.torso_inertia0 = model.body_inertia[self.torso_body].copy()
        # dof addresses of the joints the training joint_friction event targets
        # (legs + waist; arms are second-order for dribbling and were excluded).
        # frictionloss0 is the compiled MJCF default, restored when a condition
        # does not touch the channel.
        self.jf_dofs = np.array([self.vadr[i] for i, nm in enumerate(self.jnames)
                                 if re.search(r"(hip|knee|ankle|waist)", nm)])
        self.jf_frictionloss0 = (model.dof_frictionloss[self.jf_dofs].copy()
                                 if len(self.jf_dofs) else np.array([]))
        # distinct color per robot for ball + its route dots
        self.color = COLORS[k % len(COLORS)]
        model.geom_rgba[self.ball_geom] = [*self.color, 1.0]
        self.dots = []
        for j in range(40):
            bid = nid(mujoco.mjtObj.mjOBJ_BODY, f"route_dot_{j}")
            self.dots.append(model.body_mocapid[bid])
            model.geom_rgba[model.body_geomadr[bid]] = [*self.color, 0.85]
        # classify this robot's contact pairs. THREE kinds, not two, because PhysX
        # resolves each one differently -- see apply_dr.
        self.foot_pairs = []; self.foot_ball_pairs = []; self.floor_ball_pairs = []
        for pid in range(model.npair):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_PAIR, pid)
            if nm is None or not nm.startswith(self.pfx):
                continue
            if "ball" not in nm:
                self.foot_pairs.append(pid)
            elif "floor" in nm:
                self.floor_ball_pairs.append(pid)
            else:
                self.foot_ball_pairs.append(pid)
        self.rs = rs
        self.route = Route(ROUTE_CFG, seed * 100 + k)
        self.prev_latent = np.zeros(int(meta.get("latent_dim", "8")), np.float32)
        self.prev_decoded = np.zeros(29, np.float32)
        self.target = np.zeros(29); self.ep_start = 0.0; self.dr = {}
        self.robot_dr = {}                     # per-episode robot-DR draws, for the CSV
        self.ct_sum = 0.0; self.ct_count = 0   # cross-track (ball deviation from route) accumulators
        self.default_condition = make_condition()  # main() fills from the single-run CLI knobs
        # start-phase override: hold_s = stiff standby freeze (deploy hand-off,
        # the `handover` robustness axis); settle_s = soft policy takeover window
        # (training settle_time_range_s). hold_s_default / settle_range are the
        # whole-run fallbacks the runner sets; reset() resolves the per-episode
        # values (and warm_s = max, the timing offset for move_start / pushes).
        self.hold_s = 0.0
        self.hold_s_default = 0.0              # runner sets = args.standby_hold_s
        self.settle_range = None               # runner sets = args.settle_s (lo, hi) or None
        self.settle_s = 0.0
        self.warm_s = 0.0
        self.episode_len_default = 20.0        # main() sets = args.episode_s
        self.episode_len = 20.0
        self.offroute_fail_m = None            # per-condition fail-fast thresholds
        self.ball_far_fail_m = None
        self.fail_reason = ""                  # "" | "off_route" | "ball_far"
        self.offroute_since = None             # OFFROUTE_GRACE_S dwell timer
        self.max_crosstrack = 0.0              # worst route deviation this episode
        self.off_route_t = float("nan")        # when off_route was declared (s after move_start)
        self.completed_t = float("nan")        # when the arc route was finished
        self.lat_active = False
        self.cmd_speed_sum = 0.0               # commanded target_speed accumulator
        self.speed_pairs = None                # per-step (cmd, actual) speed pairs, opt-in
        self.ball_dist_sum = 0.0; self.ball_dist_count = 0
        # per-threshold sticky lost flag / grace timer / first-loss time
        n_thr = len(LOST_BALL_DISTS)
        self.ball_lost = [False] * n_thr
        self.lost_since = [None] * n_thr
        self.ball_lost_t = [float("nan")] * n_thr
        self.foot_dist_sum = 0.0; self.foot_dist_count = 0
        self.first_touch = False
        self.min_pelvis_z = float("inf"); self.max_tilt_gvec_z = -float("inf")
        self.move_start = 0.0                  # episode start + standby hold
        self.push_dv = 0.0; self.ball_push_dv = 0.0; self.push_interval_s = 5.0
        self.next_push_t = None; self.next_ball_push_t = None
        self.ball_damping = None               # per-condition override; None -> BALL_DAMPING
        self.ball_radius_obs_m = None          # None -> feed the true (DR'd) radius
        self.obs_noise = {}
        self.obs_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(0)))

    def apply_dr(self, model, dr):
        self.dr = dict(dr)
        m, r, ff, bf = dr["mass"], dr["radius"], dr["foot"], dr["ball"]
        c = BALL_DAMPING if self.ball_damping is None else float(self.ball_damping)
        I = 0.4 * m * r * r
        model.body_mass[self.ball_body] = m
        model.body_inertia[self.ball_body] = [I, I, I]
        model.geom_size[self.ball_geom][0] = r
        # geom_rbound is compiled from the ORIGINAL radius; broadphase uses it, so
        # an enlarged ball would have shin/torso contacts pruned at marginal
        # separations (foot contacts are explicit <pair>s and immune). Without
        # this the ball_radius axis partly measures a collision-detection artifact.
        model.geom_rbound[self.ball_geom] = r
        model.dof_damping[self.ballv:self.ballv + 3] = 0.0
        model.dof_damping[self.ballv + 3:self.ballv + 6] = c * I
        # Contact friction, resolved the way PhysX resolves it in training.
        # Each shape carries its own coefficient AND its own combine mode, and
        # PhysX takes the HIGHER-priority mode of the pair
        # (average < min < multiply < max). In this scene:
        #   robot bodies  mu = dr["foot"] (G1_BODY_MATERIAL_DR 0.5-1.0), "average"
        #                 (IsaacLab RigidBodyMaterialCfg default, never overridden)
        #   ball          mu = dr["ball"] (0.475-0.525),                 "average"
        #   terrain       mu = 1.0,                                      "multiply"
        # ->  foot-floor : multiply wins -> mu_foot * 1.0 = dr["foot"]
        #     floor-ball : multiply wins -> mu_ball * 1.0 = dr["ball"]
        #     foot-ball  : both "average" -> (mu_foot + mu_ball) / 2
        # The foot-ball pair used to get dr["ball"] alone, which is only right at
        # the very bottom of the foot-friction range. It also meant the
        # foot_friction sweep axis moved the foot-GROUND friction while leaving
        # the foot-BALL friction pinned -- i.e. it measured half of what it
        # claimed to. At the deploy nominal the correct value is (0.8+0.5)/2 =
        # 0.65 against the 0.5 used before.
        foot_ball = 0.5 * (ff + bf)
        for pid in self.foot_pairs:
            model.pair_friction[pid][0] = ff; model.pair_friction[pid][1] = ff
        for pid in self.foot_ball_pairs:
            model.pair_friction[pid][0] = foot_ball; model.pair_friction[pid][1] = foot_ball
        for pid in self.floor_ball_pairs:
            model.pair_friction[pid][0] = bf; model.pair_friction[pid][1] = bf
        return r

    def apply_robot_dr(self, model, condition):
        """Per-episode robot-side DR: sensor noise, actuator gains, torso payload
        and CoM, encoder calibration.

        Every channel DEFAULTS TO THE DEPLOY NOMINAL. engine.TRAIN_DR asks for a
        draw from the checkpoint's own trained range; any other value pins it. A
        channel the checkpoint never randomized has range None, so even TRAIN_DR
        leaves it nominal -- evaluating a policy against DR it never saw would
        misattribute the failures.

        obs_noise_scale and base_com_scale are MULTIPLES of the trained magnitude
        (1.0 reproduces training); actuator_gain_scale / payload_kg /
        joint_offset_rad are absolute, because their trained ranges are centred
        on 1 or one-sided.

        Draws land in self.robot_dr so the CSV records what each episode actually
        ran with: a hidden draw that is never written down cannot be conditioned
        out afterwards, which is exactly how four robot-DR channels stayed
        invisible in every run before this.

        NOTE the draw order here is fixed and happens BEFORE the ball DR draw, so
        adding a channel changes every downstream random number. That is fine on
        a --fresh run and is why the condition-table fingerprint covers it.
        """
        u = self.rng.uniform
        n_j = len(self.dq0)

        # 1. observation noise (applied in _obs, post-delay, as Isaac Lab does)
        scale = condition["obs_noise_scale"]
        self.obs_noise = ({k: v * float(scale) for k, v in OBS_NOISE.items()}
                          if OBS_NOISE and scale else {})
        # separate stream: obs noise is drawn every policy step, so sharing
        # self.rng would make the push/DR draws depend on episode length
        self.obs_rng = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence(int(self.rng.integers(0, 2**31 - 1)))))

        # 2. actuator gains. Training (isaaclab randomize_actuator_gains, operation
        # "scale") draws stiffness and damping INDEPENDENTLY and PER JOINT, so a
        # trained robot is never uniformly soft or uniformly stiff. One scalar
        # across all 29 joints -- what this used to do -- manufactures exactly
        # that correlated extreme: "every joint stiffer than nominal" has
        # probability 2^-29 in training but 1/2 with a shared draw. A PINNED
        # scalar keeps that whole-robot semantics deliberately: it is what the
        # actuator_gain AXIS is probing.
        gain = condition["actuator_gain_scale"]
        if gain == TRAIN_DR:
            kp_mul = u(*ACTUATOR_GAIN_RANGE, size=n_j) if ACTUATOR_GAIN_RANGE else np.ones(n_j)
            kd_mul = (u(*ACTUATOR_DAMPING_RANGE, size=n_j) if ACTUATOR_DAMPING_RANGE
                      else np.ones(n_j))
        else:
            kp_mul = kd_mul = np.full(n_j, 1.0 if gain is None else float(gain))
        self.kp = self.kp0 * kp_mul
        self.kd = self.kd0 * kd_mul

        # 3. torso payload. Training's randomize_rigid_body_mass recomputes the
        # inertia along with the mass (recompute_inertia defaults to True), i.e.
        # the added mass is distributed like the body it lands on. Scaling
        # body_inertia by the same ratio reproduces that; writing body_mass alone
        # gives a 3 kg payload the rotational inertia of a point mass at the CoM,
        # which is a materially easier robot to swing around.
        payload = condition["payload_kg"]
        if payload == TRAIN_DR:
            payload = u(*PAYLOAD_KG_RANGE) if PAYLOAD_KG_RANGE else 0.0
        payload = float(payload or 0.0)
        mass = self.torso_mass0 + payload
        model.body_mass[self.torso_body] = mass
        model.body_inertia[self.torso_body] = self.torso_inertia0 * (mass / self.torso_mass0)

        # 4. torso CoM offset (scale on the trained per-axis range)
        com_scale = condition["base_com_scale"]
        ipos = self.torso_ipos0.copy()
        if BASE_COM_RANGE and com_scale is not None:
            k = 1.0 if com_scale == TRAIN_DR else float(com_scale)
            for i, axis in enumerate(("x", "y", "z")):
                rng_axis = BASE_COM_RANGE.get(axis)
                if rng_axis and k:
                    ipos[i] += k * u(*rng_axis)
        model.body_ipos[self.torso_body] = ipos

        # 5. encoder calibration error. dq feeds BOTH the joint_pos observation
        # (q - dq) and the action offset (dq + scale*a), which is exactly what
        # the training event randomize_joint_default_pos perturbs.
        off = condition["joint_offset_rad"]
        if off == TRAIN_DR:
            self.dq = (self.dq0 + u(*JOINT_OFFSET_RANGE, size=n_j) if JOINT_OFFSET_RANGE
                       else self.dq0.copy())
        elif off:
            self.dq = self.dq0 + u(-float(off), float(off), n_j)
        else:
            self.dq = self.dq0.copy()

        # 6. joint friction (leg+waist dof_frictionloss). The iter-80000 lineage
        # trained FRICTIONLESS (the joint_friction event was removed / predates
        # it), but the MJCF compiles a 0.1 default and the deploy robot has real
        # friction, so this is BOTH a training-faithful nominal (0 here) and a
        # sim2real probe (the joint_friction axis sweeps it). condition pins an
        # absolute N*m; None -> JOINT_FRICTION_RANGE (a real range is sampled, a
        # degenerate (v,v) is set without a draw so the frictionless rng stream
        # is unchanged); range None -> keep the compiled default.
        if len(self.jf_dofs):
            jf = condition["joint_friction"]
            if jf is not None:
                model.dof_frictionloss[self.jf_dofs] = float(jf)
            elif JOINT_FRICTION_RANGE is not None:
                lo, hi = JOINT_FRICTION_RANGE
                model.dof_frictionloss[self.jf_dofs] = (
                    lo if hi <= lo else float(u(lo, hi)))
            else:
                model.dof_frictionloss[self.jf_dofs] = self.jf_frictionloss0

        # What this episode ACTUALLY ran with -> the CSV. Per-joint channels are
        # reduced to one scalar each (mean multiplier, RMS offset): enough to
        # stratify or regress a rate on afterwards, which is the whole point.
        self.robot_dr = dict(
            gain_kp=float(np.mean(kp_mul)), gain_kd=float(np.mean(kd_mul)),
            payload=payload,
            com_dx=float(ipos[0] - self.torso_ipos0[0]),
            com_dy=float(ipos[1] - self.torso_ipos0[1]),
            com_dz=float(ipos[2] - self.torso_ipos0[2]),
            joint_offset_rms=float(np.sqrt(np.mean((self.dq - self.dq0) ** 2))),
            joint_friction=(float(model.dof_frictionloss[self.jf_dofs[0]])
                            if len(self.jf_dofs) else float("nan")),
            obs_noise=float(scale or 0.0))

    def sample_dr(self, model):
        u = self.rng.uniform
        return self.apply_dr(model, dict(mass=u(*DR["ball_mass"]), radius=u(*DR["ball_radius"]),
                                         foot=u(*DR["foot_friction"]), ball=u(*DR["ball_friction"])))

    def sample_dr_scaled(self, model, alpha):
        # DR-magnitude axis: all params jointly sampled from the CENTERED training
        # ranges scaled by alpha (0 -> range centers, 1 -> training DR, >1 -> beyond).
        d = {}
        for key, name in (("mass", "ball_mass"), ("radius", "ball_radius"),
                          ("foot", "foot_friction"), ("ball", "ball_friction")):
            lo, hi = DR[name]; c = 0.5 * (lo + hi); h = 0.5 * (hi - lo) * alpha
            d[key] = float(max(self.rng.uniform(c - h, c + h), 0.02))
        return self.apply_dr(model, d)

    def reset(self, model, data, t, dr=None, route_seed=None, condition=None):
        # route_seed (sweep / condition-table route control): re-seed this robot's route
        # RNG so the SAME route (and, for corner conditions, the same lead/angle draws)
        # is reproduced -> conditions and experiments are compared on paired routes,
        # cancelling route-difficulty variance (which otherwise dwarfs the effects).
        if route_seed is not None:
            self.route.rng = np.random.Generator(np.random.PCG64(int(route_seed)))
        condition = condition if condition is not None else self.default_condition
        # route-shape overrides (capability conditions); None -> global defaults.
        # route_vmax: scalar pins the pace; [min,max] samples it per episode from
        # the route rng (the training-side ROUTE_CRUISE_RANGE behavior).
        route_cfg = self.route.cfg
        vmax = condition["route_vmax"]
        if isinstance(vmax, (list, tuple)):
            self.route.vmax_range = tuple(float(x) for x in vmax)
        else:
            self.route.vmax_range = None
            route_cfg["routeVmax"] = ROUTE_CFG["routeVmax"] if vmax is None else float(vmax)
        route_cfg["routeHumanKappaCap"] = (ROUTE_CFG["routeHumanKappaCap"]
                                           if condition["human_kappa_cap"] is None
                                           else float(condition["human_kappa_cap"]))
        for cond_key, cfg_key in (("route_start_speed", "routeStartSpeed"),
                                  ("route_accel_limit", "routeAccelLimit")):
            route_cfg[cfg_key] = (None if condition[cond_key] is None
                                  else float(condition[cond_key]))
        route_cfg["routeLength"] = (ROUTE_CFG["routeLength"] if condition["route_len_m"] is None
                                    else float(condition["route_len_m"]))
        self.route.const_kappa = condition["arc_kappa"]
        if condition["arc_kappa"] is not None:
            lead = condition["lead_in_m"]
            self.route.lead_range = (tuple(float(x) for x in lead)
                                     if isinstance(lead, (list, tuple)) else (float(lead), float(lead)))
            self.route.arc_deg = (tuple(float(x) for x in condition["arc_angle_deg"])
                                  if condition["arc_angle_deg"] is not None else None)
        else:
            self.route.lead_range = None; self.route.arc_deg = None
            self.route.lead_segments = 0
            self.route.arc_segments = None; self.route.arc_end_s = None
        self.episode_len = (float(condition["episode_s"]) if condition["episode_s"]
                            else self.episode_len_default)
        self.offroute_fail_m = condition["offroute_fail_m"]
        self.ball_far_fail_m = condition["ball_far_fail_m"]
        cmd_mode = route_cmd_mode(condition)
        # roll brake: read BEFORE the DR branch, apply_dr scales it by the ball inertia
        self.ball_damping = condition["ball_damping"]
        self.ball_radius_obs_m = condition["ball_radius_obs_m"]
        self.apply_robot_dr(model, condition)
        # DR: explicit dict > dr_scale (centered training ranges x alpha) > training DR
        if dr is None:
            dr = condition["dr"]
        if dr is not None:
            r = self.apply_dr(model, dr)
        elif condition["dr_scale"] is not None:
            r = self.sample_dr_scaled(model, float(condition["dr_scale"]))
        else:
            r = self.sample_dr(model)
        rs = self.rs; off = np.array([self.gx, self.gy, 0.0])
        data.qpos[self.bq:self.bq + 3] = np.array([float(x) for x in rs["root_pos"]]) + off
        data.qpos[self.bq + 3:self.bq + 7] = [float(x) for x in rs["root_quat"]]
        data.qvel[self.bv:self.bv + 3] = [float(x) for x in rs["root_lin_vel"]]
        data.qvel[self.bv + 3:self.bv + 6] = [float(x) for x in rs["root_ang_vel_body"]]
        n2i = {nm: j for j, nm in enumerate(rs["joint_names"])}
        rp = [float(x) for x in rs["joint_pos"]]; rv = [float(x) for x in rs["joint_vel"]]
        for i, nm in enumerate(self.jnames):
            data.qpos[self.qadr[i]] = rp[n2i[nm]]; data.qvel[self.vadr[i]] = rv[n2i[nm]]
        bp = np.array([float(x) for x in rs["ball_pos"]]) + off; bp[2] = r
        # reset jitter (gated so rng draw sequences are unchanged when off): small yaw +
        # base/ball xy noise de-determinizes clean-env capability conditions.
        if condition["reset_jitter"]:
            dyaw = self.rng.uniform(-np.deg2rad(JITTER_YAW_DEG), np.deg2rad(JITTER_YAW_DEG))
            qz = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)])
            qn = np.zeros(4)
            mujoco.mju_mulQuat(qn, qz, data.qpos[self.bq + 3:self.bq + 7].copy())
            data.qpos[self.bq + 3:self.bq + 7] = qn
            data.qpos[self.bq:self.bq + 2] += self.rng.uniform(-JITTER_XY, JITTER_XY, 2)
            bp[:2] += self.rng.uniform(-JITTER_XY, JITTER_XY, 2)
        # training-faithful task-start placement (gated so rng is unchanged when
        # off): ball at dist * (base-forward rotated by bearing) from the pelvis,
        # replacing the fixed straight-ahead ball_pos. dist/bearing pinned by the
        # condition (the reset_ball_* axes) or sampled from the trained range.
        # Base-forward (not chest) is used deliberately: the trained pelvis
        # distance is frame-agnostic (see RESET_BALL_DIST_RANGE) and data.xquat
        # for the chest body is stale until the runner's mj_kinematics.
        if condition["reset_ball_random"]:
            fwd0 = np.zeros(3)
            mujoco.mju_rotVecQuat(fwd0, np.array([1.0, 0, 0]),
                                  data.qpos[self.bq + 3:self.bq + 7].copy())
            fxy = fwd0[:2] / (np.hypot(fwd0[0], fwd0[1]) or 1.0)
            # each coord: None -> sample the full trained range; [lo,hi] -> sample
            # that band (the narrow physics-axis start); scalar -> pin exactly
            def _draw(val, full):
                if val is None:
                    return float(self.rng.uniform(*full))
                if isinstance(val, (list, tuple)):
                    return float(self.rng.uniform(float(val[0]), float(val[1])))
                return float(val)
            dist = _draw(condition["reset_ball_dist"], RESET_BALL_DIST_RANGE)
            bearing = np.deg2rad(_draw(condition["reset_ball_bearing"], RESET_BALL_BEARING_DEG))
            cb, sb = np.cos(bearing), np.sin(bearing)
            direction = np.array([fxy[0] * cb - fxy[1] * sb, fxy[0] * sb + fxy[1] * cb])
            bp[:2] = data.qpos[self.bq:self.bq + 2] + dist * direction
        data.qpos[self.ballq:self.ballq + 3] = bp
        data.qpos[self.ballq + 3:self.ballq + 7] = [1, 0, 0, 0]
        data.qvel[self.ballv:self.ballv + 6] = 0.0
        self.prev_latent[:] = 0; self.prev_decoded[:] = 0
        self.obs_hist = None    # history buffer refills from the first post-reset frame
        # per-episode latencies. The condition pins a channel exactly (latency axes /
        # deployment-nominal conditions); an unpinned channel falls through to the
        # --latency sampled training DR. Gated so the rng draw sequence (and thus DR
        # sampling) is unchanged when everything is off.
        self.ball_pos_hist = None; self.ball_vel_hist = None
        self.ball_delay = 0; self.act_delay = 0
        pin_ball = condition["ball_obs_delay_steps"] is not None
        pin_act = condition["action_delay_ms"] is not None
        self.lat_active = self.latency or pin_ball or pin_act
        if pin_ball:
            self.ball_delay = int(condition["ball_obs_delay_steps"])
        elif self.latency:
            self.ball_delay = int(self.rng.integers(BALL_DELAY_RANGE[0], BALL_DELAY_RANGE[1] + 1))
        if pin_act:
            self.act_delay = int(round(float(condition["action_delay_ms"]) / (model.opt.timestep * 1000.0)))
        elif self.latency:
            if self.rng.random() >= ACT_DELAY_ZERO_PROB:
                self.act_delay = int(self.rng.integers(ACT_DELAY_SUBSTEPS[0], ACT_DELAY_SUBSTEPS[1] + 1))
        # ring depths sized to the actual delays (pinned delays can exceed the training range)
        # Tiled with the RESET POSE, not the default joint pos: with the
        # deploy-nominal 10 ms action lag the first control period's early
        # sub-steps PD toward whatever is in this ring, so seeding it with dq gave
        # every episode a small startup kick away from the pose it just reset to.
        self.tgt_hist = np.tile(data.qpos[self.qadr].copy(),
                                (max(2, -(-self.act_delay // DECIMATION) + 1), 1))
        self.substep = 0
        # start-phase override durations (per episode). hold = stiff standby
        # freeze (deploy hand-off probe, the `handover` axis); settle = soft
        # policy takeover window (training). Mutually exclusive: a pinned hold
        # wins, so settle only samples when hold is 0. warm_s = the total start
        # phase, i.e. the offset for move_start / push onset. Settle sampling is
        # gated so rng is unchanged when --settle-s is off.
        self.hold_s = (float(condition["standby_hold_s"])
                       if condition["standby_hold_s"] is not None else self.hold_s_default)
        self.settle_s = (float(self.rng.uniform(*self.settle_range))
                         if (self.settle_range is not None and self.hold_s <= 0.0) else 0.0)
        self.warm_s = max(self.hold_s, self.settle_s)
        # push schedule: velocity kicks every push_interval_s, random phase/direction
        self.push_dv = float(condition["push_dv"])
        self.ball_push_dv = float(condition["ball_push_dv"])
        self.push_interval_s = float(condition["push_interval_s"])
        self.next_push_t = None; self.next_ball_push_t = None
        if self.push_dv > 0.0:
            self.next_push_t = t + self.warm_s + self.rng.uniform(1.0, self.push_interval_s)
        if self.ball_push_dv > 0.0:
            self.next_ball_push_t = t + self.warm_s + self.rng.uniform(1.0, self.push_interval_s)
        bq = data.qpos[self.bq + 3:self.bq + 7].copy()
        fwd = np.zeros(3); mujoco.mju_rotVecQuat(fwd, np.array([1.0, 0, 0]), bq)
        self.cmd = self.route.reset(bp[:2], fwd[:2], cmd_mode)
        self.target = data.qpos[self.qadr].copy(); self.ep_start = t
        self.move_start = t + self.warm_s
        self.hold_target = self.target.copy()   # standby pose to PD-hold during the hold phase
        self.ct_sum = 0.0; self.ct_count = 0   # per-episode cross-track
        self.cmd_speed_sum = 0.0
        self.speed_pairs = [] if condition["record_speed_pairs"] else None
        self.ball_dist_sum = 0.0; self.ball_dist_count = 0
        # per-threshold sticky lost flag / grace timer / first-loss time
        n_thr = len(LOST_BALL_DISTS)
        self.ball_lost = [False] * n_thr
        self.lost_since = [None] * n_thr
        self.ball_lost_t = [float("nan")] * n_thr
        self.foot_dist_sum = 0.0; self.foot_dist_count = 0
        self.first_touch = False
        self.min_pelvis_z = float("inf"); self.max_tilt_gvec_z = -float("inf")
        self.fail_reason = ""
        self.offroute_since = None; self.max_crosstrack = 0.0
        self.off_route_t = float("nan"); self.completed_t = float("nan")

    def _obs(self, data):
        bq = data.qpos[self.bq + 3:self.bq + 7]; pelvis = data.qpos[self.bq:self.bq + 3]
        bav = data.qvel[self.bv + 3:self.bv + 6]
        q = data.qpos[self.qadr] - self.dq; qd = data.qvel[self.vadr]
        cmd = self.cmd
        w, x, y, z = bq
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        # anchor of the ball/cmd obs terms: pelvis root, or the chest frame
        # (torso_link + local offset, rotating with the waist DOFs) for
        # --v2_body_frame policies
        if self.chest_body is None:
            fq, fpos = bq, pelvis
        else:
            fq = data.xquat[self.chest_body]
            fpos = data.xpos[self.chest_body] + rot_vec(fq, self.chest_offset)
        # ball obs in the anchor frame, optionally lagged by the per-episode camera latency
        ball_b_cur = world_to_body(fq, data.qpos[self.ballq:self.ballq + 3] - fpos)
        ball_vb_cur = world_to_body(fq, data.qvel[self.ballv:self.ballv + 3])
        if self.lat_active:
            if self.ball_pos_hist is None:
                K = self.ball_delay + 1
                self.ball_pos_hist = np.tile(ball_b_cur, (K, 1))
                self.ball_vel_hist = np.tile(ball_vb_cur, (K, 1))
            else:
                self.ball_pos_hist = np.roll(self.ball_pos_hist, 1, axis=0); self.ball_pos_hist[0] = ball_b_cur
                self.ball_vel_hist = np.roll(self.ball_vel_hist, 1, axis=0); self.ball_vel_hist[0] = ball_vb_cur
            ball_b = self.ball_pos_hist[self.ball_delay]      # value from ball_delay steps ago
            ball_vb = self.ball_vel_hist[self.ball_delay]     # same per-env lag for pos & vel
        else:
            ball_b, ball_vb = ball_b_cur, ball_vb_cur
        # match the C++ deployment obs terms exactly (SoftTouchDribbleObservation.cpp)
        term = {
            "base_ang_vel": bav,
            "projected_gravity": world_to_body(bq, [0, 0, -1.0]),
            "joint_pos": q,
            "joint_vel": qd,
            "last_latent_action": self.prev_latent,
            "ball_pos_b": ball_b,
            "ball_lin_vel_b": ball_vb,
            "target_dir_b": world_to_body(fq, [cmd["target_dir"][0], cmd["target_dir"][1], 0.0])[:2],
            "target_speed": [cmd["target_speed"]],
            "cmd_dir_w": [cmd["target_dir"][0], cmd["target_dir"][1]],
            "next_cmd_dir_w": [cmd["next_target_dir"][0], cmd["next_target_dir"][1]],
            "next_target_speed": [cmd["next_target_speed"]],
            "pelvis_pos_xy_w": [pelvis[0], pelvis[1]],
            "pelvis_yaw_cossin_w": [np.cos(yaw), np.sin(yaw)],
        }
        # ball_radius obs. Training feeds the TRUE (DR'd) radius; the deployment
        # C++ builds it from a configured constant (cfg_.resetBallZ). Default
        # matches training; ball_radius_obs_m pins the BELIEVED radius instead, so
        # a sweep can test the real deployment failure mode -- the configured
        # value disagreeing with the actual ball.
        believed = (self.dr["radius"] if self.ball_radius_obs_m is None
                    else float(self.ball_radius_obs_m))
        term["ball_radius"] = [believed - 0.10]   # v2: r - nominal 0.10 m
        # Observation noise, applied HERE: Isaac Lab's ObservationManager adds it
        # to the term output, i.e. AFTER the delay functions, so it must land on
        # the already-lagged ball terms and before history/concat. Uses a
        # dedicated rng stream so turning noise off leaves every other per-episode
        # draw (DR, pushes, jitter) bit-identical.
        for name, half in self.obs_noise.items():
            value = term.get(name)
            if value is None:
                continue
            value = np.asarray(value, dtype=float)
            term[name] = value + self.obs_rng.uniform(-half, half, value.shape)
        single = np.concatenate([np.ravel(term[n]) for n in self.actor_names])  # one frame (sf_dim,)
        if self.obs_hist is None:
            # first frame after reset: isaaclab CircularBuffer fills all slots with it
            self.obs_hist = np.tile(single, (self.hist_len, 1))
        else:
            self.obs_hist = np.roll(self.obs_hist, -1, axis=0)
            self.obs_hist[-1] = single                       # row 0 = oldest, row -1 = newest
        # isaaclab flattens history PER TERM (oldest->newest), then concatenates terms
        actor = np.concatenate([self.obs_hist[:, c0:c1].reshape(-1) for (c0, c1) in self.term_cols])
        decoder = np.concatenate([bav, q, qd, self.prev_decoded])
        return np.concatenate([actor, decoder]).astype(np.float32)[None, :]

    def policy_step(self, data, hold=False, settle=False):
        self._holding = hold
        self.cmd = self.route.update(data.qpos[self.ballq:self.ballq + 2].copy())
        self._update_dots(data)
        if hold:
            # deploy standby hold: stiff PD toward the standby pose (no policy),
            # letting gravity settle the robot before the hard hand-off. memory
            # stays cleared. Used by the `handover` robustness axis. (No 2 s ramp:
            # the reset pose already IS the standby pose, so the real controller's
            # ramp-from-current-pose is a no-op here.)
            self.target = self.hold_target
            return
        if settle:
            # training settle window (settle_time_range_s): the policy STEPS -- we
            # advance the obs history via _obs so it is populated when the policy
            # takes over -- but its action is REPLACED by the default (standby)
            # pose and the last-action memory zeroed, on the policy's own soft
            # gains (not _holding). Reproduces the trained hand-off (a zeroed
            # last-action obs from a settled state). No metrics accumulate yet.
            self._obs(data)
            self.prev_latent[:] = 0; self.prev_decoded[:] = 0
            self.target = self.dq.copy()
            return
        self.ct_sum += self.cmd["crosstrack"]; self.ct_count += 1
        self.cmd_speed_sum += self.cmd["target_speed"]
        if self.speed_pairs is not None:
            # actual speed both ways: projected on the commanded direction (the
            # tracking signal) and the raw planar magnitude (for trace plots)
            ball_vel = data.qvel[self.ballv:self.ballv + 2]
            self.speed_pairs.append((self.cmd["target_speed"],
                                     float(ball_vel @ self.cmd["target_dir"]),
                                     float(np.hypot(*ball_vel))))
        self.track_fall_margin(data)
        ball_dist = float(np.hypot(*(data.qpos[self.ballq:self.ballq + 2]
                                     - data.qpos[self.bq:self.bq + 2])))
        self.ball_dist_sum += ball_dist; self.ball_dist_count += 1
        crosstrack = float(self.cmd["crosstrack"])
        self.max_crosstrack = max(self.max_crosstrack, crosstrack)
        # possession: sticky lost-ball flag at each LOST_BALL_DISTS threshold
        # (nearest foot to ball surface, held > LOST_BALL_T). Unlike training
        # this does NOT end the episode -- keeping it a metric is what lets
        # survival and possession be read as separate failure modes.
        # Computed BEFORE the fail-fast block because the off-route gate needs
        # `first_touch`, which this is what arms.
        foot_dist = self.foot_ball_dist(data)
        self.foot_dist_sum += foot_dist; self.foot_dist_count += 1
        # first-acquisition gate (shared across thresholds), standing in for
        # training's `_first_touch_done`, armed by the TIGHTEST threshold =
        # "the ball was in the pocket at least once". WITHOUT it the metric is
        # meaningless: the reset places the ball ~0.65 m ahead, so foot-to-
        # surface starts ABOVE 0.5 m, the 0.1 s timer fills before the robot has
        # touched anything, and the sticky flag fires on 100% of episodes
        # (measured 182/182 on the 2026-07-20 smoke run). "Lost" must mean
        # losing a ball you had.
        if foot_dist <= LOST_BALL_DISTS[0]:
            self.first_touch = True
        if self.first_touch:
            for i, dist_thr in enumerate(LOST_BALL_DISTS):
                if foot_dist > dist_thr:
                    if self.lost_since[i] is None:
                        self.lost_since[i] = data.time
                    elif (data.time - self.lost_since[i]) >= LOST_BALL_T:
                        if not self.ball_lost[i]:
                            self.ball_lost_t[i] = data.time - self.move_start
                        self.ball_lost[i] = True
                else:
                    self.lost_since[i] = None
        # route completion instant (arc routes only), recorded even when the
        # episode later fails: "finished the turn" and "was still alive at the
        # end of the budget" are different questions and used to be one number.
        if (self.route.arc_end_s is not None and not np.isfinite(self.completed_t)
                and self.route.max_s >= self.route.arc_end_s + 0.5):
            self.completed_t = data.time - self.move_start
        # capability fail-fast: ball off the route for longer than the dwell, or
        # too far from the robot. off_route is gated on first_touch and needs
        # OFFROUTE_GRACE_S of CONTINUOUS violation (same shape as the ball-lost
        # timer above); a dip back inside resets the clock.
        if not self.fail_reason:
            if (self.offroute_fail_m is not None and self.first_touch
                    and crosstrack > self.offroute_fail_m):
                if self.offroute_since is None:
                    self.offroute_since = data.time
                elif (data.time - self.offroute_since) >= OFFROUTE_GRACE_S:
                    self.fail_reason = "off_route"
                    self.off_route_t = data.time - self.move_start
            else:
                self.offroute_since = None
            if not self.fail_reason and self.ball_far_fail_m is not None \
                    and ball_dist > self.ball_far_fail_m:
                self.fail_reason = "ball_far"
        actions, latent, *_ = self.sess.run(None, {"obs": self._obs(data)})
        self.prev_decoded = actions[0].copy(); self.prev_latent = latent[0].copy()
        self.target = np.clip(self.dq + self.ascale * actions[0], JC - JHW, JC + JHW)
        if self.lat_active:
            # action-delay ring: 0 = this step's target; reset the within-step counter
            self.tgt_hist = np.roll(self.tgt_hist, 1, axis=0); self.tgt_hist[0] = self.target
            self.substep = 0

    def torque(self, data, target):
        q = data.qpos[self.qadr]; qd = data.qvel[self.vadr]
        kp, kd = (self.skp, self.skd) if self._holding else (self.kp, self.kd)
        return np.clip(kp * (target - q) - kd * qd, -EFFORT_LIMIT, EFFORT_LIMIT)

    def apply(self, data):
        if self.lat_active and not self._holding:
            # at sub-step s with delay d, apply the target from ceil(max(d-s,0)/dec) steps ago
            deficit = max(self.act_delay - self.substep, 0)
            back = min(len(self.tgt_hist) - 1, -(-deficit // DECIMATION))
            target = self.tgt_hist[back]
            self.substep += 1
        else:
            target = self.target
        data.ctrl[self.aadr] = self.torque(data, target)

    def maybe_push(self, data):
        # velocity kicks (perturbation axes); skipped during the start phase
        # (stiff hold or soft settle)
        t = data.time
        if t - self.ep_start < self.warm_s:
            return
        if self.next_push_t is not None and t >= self.next_push_t:
            # Shaped like the training event (isaaclab events:push_by_setting_velocity
            # with the checkpoint's velocity_range): independent uniform draws per
            # axis, WITH a vertical component and — the part that actually matters
            # for a dribbler — an ANGULAR kick. The old version applied a
            # fixed-magnitude planar shove only, so a `push_dv` axis point labelled
            # "1x the trained magnitude" omitted half of what training pushed with.
            # Ratios are the checkpoint's own (z 0.4x, roll/pitch 1.04x, yaw 1.56x
            # of the planar dv), so scaling push_dv scales the whole 6-D kick.
            dv = self.push_dv
            data.qvel[self.bv:self.bv + 3] += dv * np.array([
                self.rng.uniform(-1.0, 1.0), self.rng.uniform(-1.0, 1.0),
                0.4 * self.rng.uniform(-1.0, 1.0)])
            data.qvel[self.bv + 3:self.bv + 6] += dv * np.array([
                1.04 * self.rng.uniform(-1.0, 1.0),
                1.04 * self.rng.uniform(-1.0, 1.0),
                1.56 * self.rng.uniform(-1.0, 1.0)])
            self.next_push_t = t + self.push_interval_s
        if self.next_ball_push_t is not None and t >= self.next_ball_push_t:
            # Training (dribble/mdp/events.py:push_ball_random_horizontal) draws a
            # uniform heading AND a magnitude U(0, max_speed). Using the max every
            # time -- what this did before -- doubles the mean kick, so a
            # ball_push axis point labelled "1x the trained magnitude" was really
            # 2x it in expectation.
            heading = self.rng.uniform(0.0, 2.0 * np.pi)
            dv = self.rng.uniform(0.0, self.ball_push_dv)
            data.qvel[self.ballv:self.ballv + 2] += dv * np.array([np.cos(heading), np.sin(heading)])
            self.next_ball_push_t = t + self.push_interval_s

    def speed_pair_arrays(self):
        """(commanded, actual) speed arrays for the controllability test; the actual
        speed is smoothed over ~0.5 s because dribbling is impulsive (kick -> roll ->
        catch makes the instantaneous ball speed oscillate around its mean).
        None when recording is off or the episode was too short."""
        if not self.speed_pairs or len(self.speed_pairs) < 25:
            return None
        pairs = np.asarray(self.speed_pairs)
        window = min(25, len(pairs))   # 25 policy steps = 0.5 s at 50 Hz
        kernel = np.ones(window)
        # normalise by the kernel's own overlap instead of dividing by `window`:
        # np.convolve(mode="same") ZERO-PADS, so a plain /window attenuated the
        # first and last ~window/2 samples toward zero, which then correlated
        # spuriously with the command ramp at episode start/end.
        smoothed = (np.convolve(pairs[:, 1], kernel, mode="same")
                    / np.convolve(np.ones(len(pairs)), kernel, mode="same"))
        return pairs[:, 0], smoothed

    def speed_trace(self):
        """Raw per-step (cmd, v-along-cmd, |v|) arrays at the policy rate (50 Hz)
        for the controllability trace plots; None when recording is off."""
        if not self.speed_pairs:
            return None
        pairs = np.asarray(self.speed_pairs)
        return pairs[:, 0], pairs[:, 1], pairs[:, 2]

    def episode_metrics(self, data, t):
        # everything one episode contributes to a CSV row (fall, tracking, progress,
        # achieved speed along the route, commanded speed, ball possession).
        # An episode that never left the standby hold (ct_count == 0) has no motion
        # data -> NaN, not 0/1e-6 garbage that would poison downstream means.
        move_s = t - self.move_start
        moved = self.ct_count > 0 and move_s > 1e-6
        nan = float("nan")
        # include the TERMINATING frame: policy_step samples one control period
        # before the physics that actually ends the episode, so without this the
        # recorded minimum would miss the value the verdict was made on
        self.track_fall_margin(data)
        fell = self.fell(data)
        fail_reason = "fell" if fell else self.fail_reason
        # capability (fail-fast) episodes get a success verdict; corner-turn episodes
        # additionally require finishing the turn (+0.5 m into the exit straight).
        #
        # THREE NESTED VERDICTS, because "failed" was carrying three different
        # questions in one column and the strictest one dominated the answer:
        #   possession -- stayed up AND kept the ball (training's own done-set)
        #   route      -- + never left the route / lost the ball to distance
        #   strict     -- + finished the route geometry, and was still alive at
        #                  the end of the full budget
        # Each adds one constraint, so possession >= route >= strict always.
        # NOTE on censoring: an episode that trips the off-route fail-fast STOPS
        # there, so its possession verdict is measured over a shorter window than
        # a survivor's. `duration_s` is in the CSV to make that auditable.
        completed = success = nan
        success_possession = success_route = nan
        if self.route.arc_end_s is not None:
            completed = 1.0 if self.route.max_s >= self.route.arc_end_s + 0.5 else 0.0
        if self.offroute_fail_m is not None or self.ball_far_fail_m is not None:
            kept_ball = not self.ball_lost[LOST_BALL_MAIN_IDX]
            success_possession = 1.0 if (not fell and kept_ball) else 0.0
            success_route = 1.0 if (success_possession and fail_reason == "") else 0.0
            success = 1.0 if (fail_reason == "" and completed != 0.0) else 0.0
        # Controllability. r alone does NOT measure it: correlation is invariant
        # to scale and offset, so a policy that runs at a constant 0.5x the
        # commanded speed scores r = 1.0. The regression of actual on commanded
        # (slope -> 1, bias -> 0, residual -> 0 is perfect tracking) is what
        # actually answers "does it go the speed it was told".
        speed_corr_r = speed_slope = speed_bias = speed_resid = nan
        pair_arrays = self.speed_pair_arrays()
        if pair_arrays is not None:
            cmd, act = pair_arrays
            speed_corr_r = pearson_r(cmd, act)
            if len(cmd) >= 100 and float(np.std(cmd)) > 1e-2:
                slope, bias = np.polyfit(cmd, act, 1)
                speed_slope, speed_bias = float(slope), float(bias)
                speed_resid = float(np.sqrt(np.mean((act - (slope * cmd + bias)) ** 2)))
        return dict(fell=1.0 if fell else 0.0,
                    fail_reason=fail_reason,
                    duration=t - self.ep_start,
                    cross_track=self.ct_sum / self.ct_count if self.ct_count else nan,
                    progress=float(self.route.max_s),
                    ach_speed=float(self.route.max_s) / move_s if moved else nan,
                    cmd_speed=self.cmd_speed_sum / self.ct_count if self.ct_count else nan,
                    # `ball_lost` mirrors the MAIN threshold (possession metric);
                    # the full grid is emitted as ball_lost_<thr> for post-hoc
                    # threshold choice (0.5 feeds train_survival, unchanged).
                    ball_lost=1.0 if self.ball_lost[LOST_BALL_MAIN_IDX] else 0.0,
                    ball_lost_t=self.ball_lost_t[LOST_BALL_MAIN_IDX],
                    ball_lost_grid=[1.0 if f else 0.0 for f in self.ball_lost],
                    foot_ball_dist=(self.foot_dist_sum / self.foot_dist_count
                                    if self.foot_dist_count else nan),
                    ball_dist=self.ball_dist_sum / self.ball_dist_count if self.ball_dist_count else nan,
                    completed=completed, success=success,
                    success_possession=success_possession, success_route=success_route,
                    # failure-criterion raw quantities, so any dwell / threshold
                    # can be re-derived offline instead of re-run
                    max_cross_track=self.max_crosstrack,
                    terminal_cross_track=float(self.cmd["crosstrack"]) if self.ct_count else nan,
                    off_route_t=self.off_route_t, completed_t=self.completed_t,
                    terminal_ball_dist=float(np.hypot(
                        *(data.qpos[self.ballq:self.ballq + 2]
                          - data.qpos[self.bq:self.bq + 2]))),
                    terminal_foot_ball_dist=self.foot_ball_dist(data),
                    speed_corr_r=speed_corr_r,
                    speed_slope=speed_slope, speed_bias=speed_bias,
                    speed_resid=speed_resid,
                    min_pelvis_z=(self.min_pelvis_z if np.isfinite(self.min_pelvis_z) else nan),
                    max_tilt_gvec_z=(self.max_tilt_gvec_z
                                     if np.isfinite(self.max_tilt_gvec_z) else nan))

    def _update_dots(self, data):
        npts = self.route.filled + 1; nd = len(self.dots)
        for i, m in enumerate(self.dots):
            if m < 0: continue
            if npts < 2:
                data.mocap_pos[m] = [0, 0, -1.0]; continue
            idx = (i * (npts - 1) + (nd - 1) // 2) // (nd - 1)
            p = self.route.points[idx]; data.mocap_pos[m] = [p[0], p[1], 0.10]

    def base_z(self, data):
        return data.qpos[self.bq + 2]

    def track_fall_margin(self, data):
        """Record how close this episode came to the fall criteria.

        Storing the raw quantities (lowest pelvis, largest tilt) rather than just
        the boolean lets ANY threshold be re-derived after the fact -- which is
        what makes a criterion change auditable instead of a silent re-ruling.
        They also read as a continuous "margin to failure" in their own right."""
        self.min_pelvis_z = min(self.min_pelvis_z, float(self.base_z(data)))
        gvec_z = world_to_body(data.qpos[self.bq + 3:self.bq + 7], [0, 0, -1.0])[2]
        self.max_tilt_gvec_z = max(self.max_tilt_gvec_z, float(gvec_z))

    def fell(self, data):
        """Training-matched fall test: pelvis too low OR tilted past ~45 deg.

        Mirrors multiagent_sim/tasks/kick/mdp/terminations.py:fall, which the
        checkpoints' env.yaml selects. The tilt term is what the height-only test
        missed: a robot folded over its own knees keeps the pelvis above 0.4 m
        for a long time after it is unrecoverable."""
        if self.base_z(data) < FALL_Z:
            return True
        gvec_z = world_to_body(data.qpos[self.bq + 3:self.bq + 7], [0, 0, -1.0])[2]
        return bool(gvec_z > FALL_TILT_GVEC_Z)

    def foot_ball_dist(self, data):
        """Nearest-foot-to-ball-SURFACE horizontal distance, as training measures
        it (DribbleRLEnv: min over feet of |foot_xy - ball_xy| - ball_radius,
        clamped at 0)."""
        ball_xy = data.qpos[self.ballq:self.ballq + 2]
        best = min(float(np.hypot(*(data.xpos[b][:2] - ball_xy)))
                   for b in self.foot_bodies)
        return max(best - float(self.dr["radius"]), 0.0)
