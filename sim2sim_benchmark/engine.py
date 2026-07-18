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
import os
import re

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
    routeLength=20.0, routeSegmentLength=0.25, routeLookahead=0.8, routePreviewArc=1.0,
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
FALL_Z = 0.4

# lost-ball ("possession") criterion: horizontal ball-pelvis distance stays above
# LOST_BALL_DIST for LOST_BALL_T continuous seconds -> the episode is flagged
# ball_lost (sticky). A robot that stays up but kicks the ball away is a task
# failure that fall-rate alone cannot see.
LOST_BALL_DIST = 1.5
LOST_BALL_T = 2.0

# reset jitter (condition key reset_jitter): deterministic clean-env conditions
# need SOME spread to make per-condition survival a probability, not a repeated
# 0/1 outcome
JITTER_YAW_DEG = 5.0
JITTER_XY = 0.02

# training latency DR (2026-06-21 v2 policy), sampled when Robot.latency is on
# (the pysim --latency flag) and no condition pins the channels.
# Ball obs lag: per-episode constant d in {1,2,3} policy steps (delay_steps_range (1,3)).
# Action lag: per-episode constant d in [0,4] sub-steps (action_delay_ms_range (0,20ms) at
# sim_dt=0.005), 30% of episodes forced to zero (zero_prob), applied at sub-step granularity.
BALL_DELAY_RANGE = (1, 3)
ACT_DELAY_SUBSTEPS = (0, 4)
ACT_DELAY_ZERO_PROB = 0.3
ACT_DELAY_K = (ACT_DELAY_SUBSTEPS[1] + DECIMATION - 1) // DECIMATION + 1  # ring depth (policy steps)
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

# --eval ranges = the ACTUAL training DR (verified against dribble_env_cfg.py):
#   ball_mass x[0.9,1.1] of 0.391 -> [0.352,0.430], ball friction [0.475,0.525],
#   body/foot dynamic friction [0.5,1.0]. ball_radius was NOT randomized in training
#   (fixed 0.10), so per user request we give it a +/-10% band [0.09,0.11].
DR = dict(ball_mass=(0.352, 0.430), ball_radius=(0.09, 0.11),
          foot_friction=(0.50, 1.00), ball_friction=(0.475, 0.525))

# --sweep ranges = 1.5x the training range (centered), so we probe a bit past the
# trained envelope but NOT into meaningless OOD territory. radius has no trained
# range, so use the same +/-10% band [0.09,0.11] around 0.10.
SWEEP_RANGES = dict(ball_mass=(0.3325, 0.4495), ball_radius=(0.09, 0.11),
                    foot_friction=(0.375, 1.125), ball_friction=(0.4625, 0.5375))

# distinct per-robot colors (ball + its route dots) so each trajectory is identifiable
COLORS = [(0.90, 0.25, 0.25), (0.25, 0.80, 0.35), (0.30, 0.55, 1.00), (0.95, 0.85, 0.15),
          (0.85, 0.40, 0.95), (0.20, 0.85, 0.85), (0.95, 0.55, 0.15), (0.6, 0.6, 0.6)]


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
    dr=None, dr_scale=None, record_speed_pairs=False,
    route_start_speed=None, route_accel_limit=None)


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


def build_world(n_robots, spacing, onnx_path, reset_path, seed):
    """Compose the multi-robot model and its Robot wrappers (shared by main() and
    the sim2sim_benchmark package)."""
    model, grid = compose(n_robots, spacing)
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
    reset_state = parse_reset(reset_path)
    robots = [Robot(model, k, grid[k], session, meta, reset_state, seed)
              for k in range(n_robots)]
    return model, data, robots


def step_control_period(model, data, robots, standby_hold_s=0.0):
    """Run one 50 Hz control period (policy once, PD torque every physics sub-step)
    and return the indices of robots whose episode just ended (fall / fail-fast /
    episode-length timeout)."""
    for rb in robots:
        rb.policy_step(data, hold=(data.time - rb.ep_start) < standby_hold_s)
        rb.maybe_push(data)
    for _ in range(DECIMATION):
        for rb in robots:
            rb.apply(data)
        mujoco.mj_step(model, data)
    t = data.time
    return [j for j, rb in enumerate(robots)
            if rb.base_z(data) < FALL_Z or rb.fail_reason or (t - rb.ep_start) >= rb.episode_len]


def compose(n, spacing):
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
        child = mujoco.MjSpec.from_file(SINGLE_MJCF)
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
    # robots over. Disable auto-collision on all floors; the explicit foot-floor / ball-floor
    # <pair>s still fire (pairs ignore contype/conaffinity), each robot hitting only its floor.
    floors = [g for g in sp.geoms if (g.name or "").endswith("floor")]
    for gi, g in enumerate(floors):
        g.contype = 0; g.conaffinity = 0
        if gi > 0:
            rgba = list(g.rgba); rgba[3] = 0.0; g.rgba = rgba  # keep (now explicit-only) collision, hide visually
    # ISOLATE robots: give every robot's auto-colliding geoms (body capsules + feet + ball)
    # a UNIQUE contype/conaffinity bit. Within robot k all share bit k -> self-collisions are
    # identical to single-robot; across robots bit_a & bit_b = 0 -> NO collision. The foot-floor
    # / foot-ball / ball-floor contacts are explicit <pair>s (ignore contype) so they're
    # untouched. This removes the cross-robot collision volume that knocked neighbours over at
    # tight (video) spacing, without changing any single robot's physics. (>31 robots reuse a
    # bit, but eval/sweep run at 120m spacing where contact is geometrically impossible anyway.)
    rx = re.compile(r"^r(\d+)_")
    for g in sp.geoms:
        if not (g.contype or g.conaffinity):   # skip visuals / route dots / already-zeroed floors
            continue
        mt = rx.match(g.name or "")
        if mt:
            bit = 1 << (int(mt.group(1)) % 31)
            g.contype = bit; g.conaffinity = bit
    return sp.compile(), grid


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
        self.kp = csv_floats(meta["joint_stiffness"]); self.kd = csv_floats(meta["joint_damping"])
        self.skp = np.array([STANDBY_GAINS[n][0] for n in meta["joint_names"].split(",")])
        self.skd = np.array([STANDBY_GAINS[n][1] for n in meta["joint_names"].split(",")])
        self._holding = False
        self.ascale = csv_floats(meta["action_scale"]); self.dq = csv_floats(meta["default_joint_pos"])
        self.jnames = meta["joint_names"].split(",")
        # actor obs is built term-by-term in this order so one code path serves the
        # 82-dim (invariant), 90-dim (world-frame) and 83x10 (history) policies.
        self.actor_names = meta["actor_obs_names"].split(",")
        # single-frame dims for the actor terms (first len(actor_names) of observation_dims)
        all_dims = [int(x) for x in meta["observation_dims"].split(",")]
        self.actor_dims = all_dims[: len(self.actor_names)]
        self.sf_dim = sum(self.actor_dims)                         # single-frame actor width
        self.hist_len = int(meta.get("actor_history_length", "1"))  # 10 for the v2 policy
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
        # distinct color per robot for ball + its route dots
        self.color = COLORS[k % len(COLORS)]
        model.geom_rgba[self.ball_geom] = [*self.color, 1.0]
        self.dots = []
        for j in range(40):
            bid = nid(mujoco.mjtObj.mjOBJ_BODY, f"route_dot_{j}")
            self.dots.append(model.body_mocapid[bid])
            model.geom_rgba[model.body_geomadr[bid]] = [*self.color, 0.85]
        # classify this robot's contact pairs
        self.foot_pairs = []; self.ball_pairs = []
        for pid in range(model.npair):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_PAIR, pid)
            if nm is None or not nm.startswith(self.pfx):
                continue
            (self.ball_pairs if "ball" in nm else self.foot_pairs).append(pid)
        self.rs = rs
        self.route = Route(ROUTE_CFG, seed * 100 + k)
        self.prev_latent = np.zeros(int(meta.get("latent_dim", "8")), np.float32)
        self.prev_decoded = np.zeros(29, np.float32)
        self.target = np.zeros(29); self.ep_start = 0.0; self.dr = {}
        self.ct_sum = 0.0; self.ct_count = 0   # cross-track (ball deviation from route) accumulators
        self.default_condition = make_condition()  # main() fills from the single-run CLI knobs
        self.hold_s = 0.0                      # main() sets = args.standby_hold_s
        self.episode_len_default = 20.0        # main() sets = args.episode_s
        self.episode_len = 20.0
        self.offroute_fail_m = None            # per-condition fail-fast thresholds
        self.ball_far_fail_m = None
        self.fail_reason = ""                  # "" | "off_route" | "ball_far"
        self.lat_active = False
        self.cmd_speed_sum = 0.0               # commanded target_speed accumulator
        self.speed_pairs = None                # per-step (cmd, actual) speed pairs, opt-in
        self.ball_dist_sum = 0.0; self.ball_dist_count = 0
        self.ball_lost = False; self.lost_since = None
        self.move_start = 0.0                  # episode start + standby hold
        self.push_dv = 0.0; self.ball_push_dv = 0.0; self.push_interval_s = 5.0
        self.next_push_t = None; self.next_ball_push_t = None

    def apply_dr(self, model, dr):
        self.dr = dict(dr)
        m, r, ff, bf = dr["mass"], dr["radius"], dr["foot"], dr["ball"]
        I = 0.4 * m * r * r
        model.body_mass[self.ball_body] = m
        model.body_inertia[self.ball_body] = [I, I, I]
        model.geom_size[self.ball_geom][0] = r
        model.dof_damping[self.ballv:self.ballv + 3] = 0.0
        model.dof_damping[self.ballv + 3:self.ballv + 6] = 4.0 * I
        for pid in self.foot_pairs:
            model.pair_friction[pid][0] = ff; model.pair_friction[pid][1] = ff
        for pid in self.ball_pairs:
            model.pair_friction[pid][0] = bf; model.pair_friction[pid][1] = bf
        return r

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
        self.tgt_hist = np.tile(self.dq, (max(2, -(-self.act_delay // DECIMATION) + 1), 1))
        self.substep = 0
        # push schedule: velocity kicks every push_interval_s, random phase/direction
        self.push_dv = float(condition["push_dv"])
        self.ball_push_dv = float(condition["ball_push_dv"])
        self.push_interval_s = float(condition["push_interval_s"])
        self.next_push_t = None; self.next_ball_push_t = None
        if self.push_dv > 0.0:
            self.next_push_t = t + self.hold_s + self.rng.uniform(1.0, self.push_interval_s)
        if self.ball_push_dv > 0.0:
            self.next_ball_push_t = t + self.hold_s + self.rng.uniform(1.0, self.push_interval_s)
        bq = data.qpos[self.bq + 3:self.bq + 7].copy()
        fwd = np.zeros(3); mujoco.mju_rotVecQuat(fwd, np.array([1.0, 0, 0]), bq)
        self.cmd = self.route.reset(bp[:2], fwd[:2], cmd_mode)
        self.target = data.qpos[self.qadr].copy(); self.ep_start = t
        self.move_start = t + self.hold_s
        self.hold_target = self.target.copy()   # standby pose to PD-hold during --standby-hold-s
        self.ct_sum = 0.0; self.ct_count = 0   # per-episode cross-track
        self.cmd_speed_sum = 0.0
        self.speed_pairs = [] if condition["record_speed_pairs"] else None
        self.ball_dist_sum = 0.0; self.ball_dist_count = 0
        self.ball_lost = False; self.lost_since = None
        self.fail_reason = ""

    def _obs(self, data):
        bq = data.qpos[self.bq + 3:self.bq + 7]; pelvis = data.qpos[self.bq:self.bq + 3]
        bav = data.qvel[self.bv + 3:self.bv + 6]
        q = data.qpos[self.qadr] - self.dq; qd = data.qvel[self.vadr]
        cmd = self.cmd
        w, x, y, z = bq
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        # ball obs in pelvis frame, optionally lagged by the per-episode camera latency
        ball_b_cur = world_to_body(bq, data.qpos[self.ballq:self.ballq + 3] - pelvis)
        ball_vb_cur = world_to_body(bq, data.qvel[self.ballv:self.ballv + 3])
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
            "target_dir_b": world_to_body(bq, [cmd["target_dir"][0], cmd["target_dir"][1], 0.0])[:2],
            "target_speed": [cmd["target_speed"]],
            "cmd_dir_w": [cmd["target_dir"][0], cmd["target_dir"][1]],
            "next_cmd_dir_w": [cmd["next_target_dir"][0], cmd["next_target_dir"][1]],
            "next_target_speed": [cmd["next_target_speed"]],
            "pelvis_pos_xy_w": [pelvis[0], pelvis[1]],
            "pelvis_yaw_cossin_w": [np.cos(yaw), np.sin(yaw)],
        }
        term["ball_radius"] = [self.dr["radius"] - 0.10]   # v2: r - nominal 0.10 m
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

    def policy_step(self, data, hold=False):
        self._holding = hold
        self.cmd = self.route.update(data.qpos[self.ballq:self.ballq + 2].copy())
        self._update_dots(data)
        if hold:
            # standby phase: PD-hold the reset pose, no policy, memory stays cleared
            self.target = self.hold_target
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
        ball_dist = float(np.hypot(*(data.qpos[self.ballq:self.ballq + 2]
                                     - data.qpos[self.bq:self.bq + 2])))
        self.ball_dist_sum += ball_dist; self.ball_dist_count += 1
        # capability fail-fast: ball too far off the route, or too far from the robot
        if not self.fail_reason:
            if self.offroute_fail_m is not None and self.cmd["crosstrack"] > self.offroute_fail_m:
                self.fail_reason = "off_route"
            elif self.ball_far_fail_m is not None and ball_dist > self.ball_far_fail_m:
                self.fail_reason = "ball_far"
        # robustness possession metric: sticky lost-ball flag (does NOT end the episode)
        if ball_dist > LOST_BALL_DIST:
            if self.lost_since is None:
                self.lost_since = data.time
            elif (data.time - self.lost_since) >= LOST_BALL_T:
                self.ball_lost = True
        else:
            self.lost_since = None
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
        # velocity kicks (perturbation axes); skipped during the standby hold
        t = data.time
        if t - self.ep_start < self.hold_s:
            return
        if self.next_push_t is not None and t >= self.next_push_t:
            heading = self.rng.uniform(0.0, 2.0 * np.pi)
            data.qvel[self.bv:self.bv + 2] += self.push_dv * np.array([np.cos(heading), np.sin(heading)])
            self.next_push_t = t + self.push_interval_s
        if self.next_ball_push_t is not None and t >= self.next_ball_push_t:
            heading = self.rng.uniform(0.0, 2.0 * np.pi)
            data.qvel[self.ballv:self.ballv + 2] += self.ball_push_dv * np.array([np.cos(heading), np.sin(heading)])
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
        smoothed = np.convolve(pairs[:, 1], np.ones(window) / window, mode="same")
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
        fell = self.base_z(data) < FALL_Z
        fail_reason = "fell" if fell else self.fail_reason
        # capability (fail-fast) episodes get a success verdict; corner-turn episodes
        # additionally require finishing the turn (+0.5 m into the exit straight).
        completed = success = nan
        if self.route.arc_end_s is not None:
            completed = 1.0 if self.route.max_s >= self.route.arc_end_s + 0.5 else 0.0
        if self.offroute_fail_m is not None or self.ball_far_fail_m is not None:
            success = 1.0 if (fail_reason == "" and completed != 0.0) else 0.0
        speed_corr_r = nan
        pair_arrays = self.speed_pair_arrays()
        if pair_arrays is not None:
            speed_corr_r = pearson_r(*pair_arrays)
        return dict(fell=1.0 if fell else 0.0,
                    fail_reason=fail_reason,
                    duration=t - self.ep_start,
                    cross_track=self.ct_sum / self.ct_count if self.ct_count else nan,
                    progress=float(self.route.max_s),
                    ach_speed=float(self.route.max_s) / move_s if moved else nan,
                    cmd_speed=self.cmd_speed_sum / self.ct_count if self.ct_count else nan,
                    ball_lost=1.0 if self.ball_lost else 0.0,
                    ball_dist=self.ball_dist_sum / self.ball_dist_count if self.ball_dist_count else nan,
                    completed=completed, success=success, speed_corr_r=speed_corr_r)

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
