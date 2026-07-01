#!/usr/bin/env python3
"""Multi-robot standalone MuJoCo dribble sim (Phase 2).

Composes N prefixed G1 robots (each with its own ball + route dots) into ONE
MuJoCo model via MjSpec, runs the exported ONNX policy on each, and gives every
robot independent per-EPISODE domain randomization (ball mass/radius/friction,
constant within an episode). An episode ends on a fall or after --episode-s
seconds -> that robot resets and re-samples its DR. One viewer window shows all.

The policy is yaw/position-invariant (no world-frame obs terms), so each robot's
obs depends only on its own relative quantities -> the grid offset cancels and
the robots are fully independent.

  # watch (your terminal, has display):
  ~/miniconda3/envs/multiagentsim/bin/python tools/dribble_pysim_multi.py
  # headless smoke test:
  ... tools/dribble_pysim_multi.py --headless --seconds 35
"""
import argparse
import os
import re
import sys
import time
import numpy as np
if "--record" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen render needs a headless GL backend
import mujoco
import mujoco.viewer
import onnxruntime as ort

SINGLE_MJCF = "/home/aldebaran/Desktop/dribbling_controller/mjcf/g1_softtouch_dribble.xml"
ONNX = "/home/aldebaran/Desktop/SoftTouch-multiagent/logs/rsl_rl/g1_dribble/2026-06-17_16-44-29/softtouch_dribble_deploy_iter50000.onnx"
RESET_FILE = "/home/aldebaran/Desktop/dribbling_controller/config/g1/softtouch_mujoco_reset_walkf_rf_frame0.txt"
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dribbling_controller/


# all batch outputs (video + plots + csvs) land under this base dir; --out-dir overrides
_OUT_BASE = REPO_DIR


def resolve_out(path):
    """Bare filenames / relative paths land in the output base dir; absolute paths respected."""
    if not path:
        return ""
    path = os.path.expanduser(path)
    return path if os.path.isabs(path) else os.path.join(_OUT_BASE, path)

ROUTE_CFG = dict(
    routeLength=20.0, routeSegmentLength=0.25, routeLookahead=0.8, routePreviewArc=1.0,
    routeCurvatureMin=0.0, routeCurvatureMax=0.0, routeSFlipArc=2.5,
    routeHumanKappaCap=0.5, routeHumanPersist=0.6, routeHumanWeaveMin=0.4, routeHumanWeaveMax=1.0,
    routeHumanBigProbability=0.09, routeHumanBigAngleMinDeg=40.0, routeHumanBigAngleMaxDeg=180.0,
    routeKvScale=0.75, routeVmax=2.0, routeLazyExtend=True, routeInitSegments=9,
    routeExtendChunk=1, routeExtendAheadMarginSegments=10,
)
CMD_MODE = 4
JOINT_LIMIT_FACTOR = 0.9
DECIMATION = 4
FALL_Z = 0.4

# --latency DR (2026-06-21 v2 policy training-time perturbations), opt-in via --latency.
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


def _font(sz):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_legend(frame, robots):
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame); d = ImageDraw.Draw(img); font = _font(18)
    y = 6
    for k, rb in enumerate(robots):
        col = tuple(int(255 * c) for c in rb.color)
        d.rectangle([8, y + 3, 26, y + 21], fill=col, outline=(0, 0, 0))
        ct = rb.ct_sum / max(1, rb.ct_count)
        dr = rb.dr
        txt = (f"r{k}  m={dr['mass']:.2f}kg  r={dr['radius']:.3f}m  "
               f"uf={dr['foot']:.2f} ub={dr['ball']:.2f}  | cross-track={ct:.3f}m")
        d.text((32, y), txt, fill=(255, 255, 255), font=font)
        y += 26
    return np.asarray(img)


def aggregate_eval(records, args):
    if not records:
        print("[eval] no episodes completed — run longer (--seconds) or more --robots"); return
    arr = np.array(records)  # cols: mass, radius, foot, ball, cross_track, fell, duration
    ct, fell, dur = arr[:, 4], arr[:, 5], arr[:, 6]
    print(f"\n=== EVAL: {len(arr)} episodes | {args.robots} robots | episode {args.episode_s:.0f}s | route {args.route_len:.0f}m ===")
    print(f"OVERALL  cross-track={ct.mean():.3f}m (median {np.median(ct):.3f})  "
          f"fall-rate={100*fell.mean():.0f}%  mean-survival={dur.mean():.1f}s")
    labels = ["ball_mass(kg)", "ball_radius(m)", "foot_friction", "ball_friction"]
    for ci, lab in enumerate(labels):
        v = arr[:, ci]; edges = np.quantile(v, [0, 1/3, 2/3, 1.0])
        print(f"  by {lab}:")
        for b in range(3):
            lo, hi = edges[b], edges[b + 1]
            mask = (v >= lo) & ((v <= hi) if b == 2 else (v < hi))
            if mask.sum() == 0:
                continue
            print(f"     [{lo:.3f},{hi:.3f}]  n={int(mask.sum()):3d}  "
                  f"cross-track={ct[mask].mean():.3f}m  fall={100*fell[mask].mean():.0f}%")
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ball_mass", "ball_radius", "foot_fric", "ball_fric", "cross_track_m", "fell", "duration_s"])
            for r in records:
                w.writerow([f"{x:.5f}" for x in r])
        print(f"  saved {args.csv} ({len(records)} rows)")


def _panel(ax_ct, ax_fr, v, ct, fell, lab, W, nominal=None):
    """Draw one DR-parameter column: cross-track (top) + fall-rate (bottom), both as
    rolling-window curves over param-sorted episodes. No bar charts; fall-rate of a 0/1
    outcome vs a continuous x is a smoothed PROPORTION, which is what a rolling mean gives."""
    order = np.argsort(v); vs = v[order]; cts = ct[order]; fls = fell[order]
    h = W // 2
    roll_ct = np.array([np.nanmean(cts[max(0, i - h):i + h + 1]) for i in range(len(vs))])
    roll_fr = np.array([fls[max(0, i - h):i + h + 1].mean() for i in range(len(vs))])
    surv = fls < 0.5
    # cross-track: faint scatter for raw spread, bold rolling mean for the trend.
    ax_ct.scatter(vs[surv], cts[surv], c="#3aa655", s=7, alpha=0.22, edgecolors="none", label="survived")
    ax_ct.scatter(vs[~surv], cts[~surv], c="#d1495b", marker="x", s=13, alpha=0.40, lw=0.7, label="fell")
    ax_ct.plot(vs, roll_ct, color="#1f4e79", lw=2.6, label=f"rolling mean (w={W})")
    ax_ct.set_xlabel(lab); ax_ct.set_ylabel("cross-track (m)")
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator, LogLocator, ScalarFormatter, NullFormatter
    # LOG (non-uniform) y so the rolling mean (~0.2-0.4 m) isn't squashed at the bottom
    # by a few multi-metre outliers -> the actual mean value is now readable off the grid.
    pos = cts[cts > 1e-3]
    lo = float(np.nanpercentile(pos, 2)) if len(pos) else 0.05
    hi = float(np.nanpercentile(cts, 96)) if len(cts) else 1.0
    ax_ct.set_yscale("log")
    ax_ct.set_ylim(max(lo, 0.03), max(hi, lo * 3))
    ax_ct.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0), numticks=20))
    ax_ct.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(1.0, 10.0) * 0.1), numticks=100))
    ax_ct.yaxis.set_major_formatter(ScalarFormatter()); ax_ct.yaxis.set_minor_formatter(NullFormatter())
    ax_ct.grid(which="major", axis="y", alpha=0.30); ax_ct.grid(which="minor", axis="y", alpha=0.12)
    ax_ct.grid(which="major", axis="x", alpha=0.22)
    # fall-rate: clean rolling-proportion LINE (no fill / no bars)
    ax_fr.plot(vs, roll_fr * 100, color="#d1495b", lw=2.6)
    ax_fr.set_xlabel(lab); ax_fr.set_ylabel("fall rate (%)"); ax_fr.set_ylim(0, 100)
    # finer x granularity: ~8 labelled major ticks + 5 unlabelled minor ticks between
    # them (the fine sub-divisions), with a vertical grid so you can read where the
    # fall-rate curve changes. Labels rotated + small so they don't collide.
    ax_fr.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax_fr.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_fr.yaxis.set_major_locator(MaxNLocator(nbins=10))    # fall-rate every ~10%
    ax_fr.tick_params(axis="x", which="major", labelsize=7, rotation=45)
    ax_fr.tick_params(axis="x", which="minor", length=3)
    ax_fr.grid(which="major", axis="both", alpha=0.30)
    ax_fr.grid(which="minor", axis="x", alpha=0.13)
    if nominal is not None:
        ax_ct.axvline(nominal, color="gray", ls="--", lw=1.1)
        ax_fr.axvline(nominal, color="gray", ls="--", lw=1.1, label="nominal")
    return roll_fr, vs


def plot_eval(records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr = np.array(records)
    ct, fell = arr[:, 4], arr[:, 5]
    cand = [("ball mass (kg)", 0), ("ball radius (m)", 1), ("foot friction", 2), ("ball friction", 3)]
    # only plot params that were actually randomized (radius is fixed -> skip its dead column)
    params = [(lab, ci) for lab, ci in cand if float(np.ptp(arr[:, ci])) > 1e-6]
    ncol = len(params)
    W = max(20, len(arr) // 8)
    fig, ax = plt.subplots(2, ncol, figsize=(4.7 * ncol, 7.4), squeeze=False)
    for c, (lab, ci) in enumerate(params):
        _panel(ax[0, c], ax[1, c], arr[:, ci], ct, fell, lab, W)
    ax[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Random-DR eval — {len(arr)} episodes  |  fall-rate {100*fell.mean():.0f}%  |  "
                 f"median cross-track {np.median(ct):.2f} m   "
                 f"(all {ncol} DR params varied simultaneously; marginal trends shown)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)
    print(f"  saved plot {path}")


def sweep_report(srec, axes, args, nominal):
    if not srec:
        print("[sweep] no episodes completed"); return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr = np.array(srec)  # cols: param_idx, level_value, cross_track, fell
    print(f"\n=== SWEEP (route-controlled): {len(arr)} episodes ===")
    fig, ax = plt.subplots(2, len(axes), figsize=(4.7 * len(axes), 7.4), squeeze=False)
    for pi, (lab, key, rng) in enumerate(axes):
        sub = arr[arr[:, 0] == pi]
        levels = np.unique(sub[:, 1]); nv = nominal[key]
        mct = []; sect = []; fr = []; frse = []
        for lv in levels:
            m = sub[:, 1] == lv; ct = sub[m, 2]; fl = sub[m, 3]; n = max(1, m.sum())
            mct.append(ct.mean()); sect.append(ct.std() / np.sqrt(n))     # SEM of the mean
            p = fl.mean(); fr.append(p); frse.append(np.sqrt(p * (1 - p) / n))  # binomial SE
        mct = np.array(mct); sect = np.array(sect); fr = np.array(fr) * 100; frse = np.array(frse) * 100
        # cross-track: per-level mean over the shared routes, +/-1 SEM band
        a = ax[0, pi]
        a.plot(levels, mct, "o-", color="#1f4e79", lw=2.2, ms=5)
        a.fill_between(levels, np.maximum(mct - sect, 1e-3), mct + sect, color="#1f4e79", alpha=0.18)
        a.axvline(nv, color="gray", ls="--", lw=1.1, label="nominal")
        a.set_xlabel(lab); a.set_ylabel("cross-track (m)  [mean +/- SEM]"); a.grid(alpha=0.25)
        # fall-rate: per-level proportion over the shared routes, +/-1 binomial SE band
        b = ax[1, pi]
        b.plot(levels, fr, "o-", color="#d1495b", lw=2.2, ms=5)
        b.fill_between(levels, np.maximum(fr - frse, 0), np.minimum(fr + frse, 100), color="#d1495b", alpha=0.18)
        b.axvline(nv, color="gray", ls="--", lw=1.1)
        b.set_xlabel(lab); b.set_ylabel("fall rate (%)  [+/- SE]"); b.set_ylim(0, 100); b.grid(alpha=0.25)
        i_nom = int(np.argmin(np.abs(levels - nv)))
        print(f"  {lab}: nominal ct~{mct[i_nom]:.3f}m fall~{fr[i_nom]:.0f}%  | "
              f"ct range [{mct.min():.3f},{mct.max():.3f}]m  fall range [{fr.min():.0f},{fr.max():.0f}]%")
    ax[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"1-param sweep, ROUTE-CONTROLLED (each level on the same routes; dashed = nominal) — "
                 f"{len(arr)} episodes  (top: tracking, bottom: stability)")
    fig.tight_layout()
    out = args.plot or resolve_out("sweep_plot.png")
    fig.savefig(out, dpi=120)
    print(f"  saved plot {out}")
    if args.csv:
        import csv
        names = [a[0] for a in axes]
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["param_idx", "param_name", "level_value", "cross_track_m", "fell"])
            for pi, val, ct, fl in srec:
                w.writerow([int(pi), names[int(pi)], f"{val:.5f}", f"{ct:.5f}", int(fl)])
        print(f"  saved {args.csv} ({len(srec)} rows)")


def print_eval(robots, resets):
    print("==== eval (per robot) ====  (cross-track = mean ball deviation from route line)")
    for k, rb in enumerate(robots):
        ct = rb.ct_sum / max(1, rb.ct_count)
        dr = rb.dr
        print(f"  r{k}: cross-track={ct:.3f} m | resets={resets[k]} | last DR "
              f"m={dr['mass']:.2f}kg r={dr['radius']:.3f}m uf={dr['foot']:.2f} ub={dr['ball']:.2f}")


def world_to_body(quat_wxyz, vec):
    negq = np.zeros(4); res = np.zeros(3)
    mujoco.mju_negQuat(negq, np.ascontiguousarray(quat_wxyz, dtype=np.float64))
    mujoco.mju_rotVecQuat(res, np.ascontiguousarray(vec, dtype=np.float64), negq)
    return res


def csv_floats(s):
    return np.array([float(x) for x in re.split(r"[,\s]+", s.strip()) if x != ""])


class Route:
    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.rng = np.random.Generator(np.random.PCG64(seed))
        n = max(1, int(round(cfg["routeLength"] / max(cfg["routeSegmentLength"], 1e-9))))
        self.points = np.zeros((n + 1, 2)); self.speed = np.zeros(n)
        self.filled = 0; self.end_heading = 0.0; self.last_seg = -1
        self.h_sign = 1.0; self.big_remain = 0.0; self.big_sign = 1.0

    def _u(self, lo, hi): return self.rng.uniform(lo, hi)

    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v); return np.array([1.0, 0.0]) if n < 1e-9 else v / n

    def reset(self, origin, forward, cmd_mode):
        self.cmd_mode = cmd_mode; self.cmd_sign = -1.0 if cmd_mode == 2 else 1.0
        self.last_seg = -1; self.filled = 0
        max_seg = len(self.speed)
        init = int(np.clip(self.cfg["routeInitSegments"], 1, max_seg)) if self.cfg["routeLazyExtend"] else max_seg
        self._build(init, True, np.asarray(origin, float), self._unit(np.asarray(forward, float)))
        return self.update(origin)

    def update(self, ball_xy):
        self._extend(); ball_xy = np.asarray(ball_xy, float)
        filled = max(1, self.filled)
        best_d2 = np.inf; best_t = 0.0; best_seg = 0; best_proj = self.points[0]
        for i in range(filled):
            a = self.points[i]; b = self.points[i + 1]; ab = b - a
            ab2 = max(ab @ ab, 1e-9)
            t = np.clip((ball_xy - a) @ ab / ab2, 0.0, 1.0); proj = a + t * ab
            d2 = (ball_xy - proj) @ (ball_xy - proj)
            if d2 < best_d2:
                best_d2, best_t, best_seg, best_proj = d2, t, i, proj
        self.last_seg = best_seg
        s = (best_seg + best_t) * self.cfg["routeSegmentLength"]
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
        kappa = self._human_kappa(num) if self.cmd_mode == 4 else np.zeros(num)
        heading = theta; point = np.asarray(org, float).copy()
        for i in range(num):
            point = point + np.array([np.cos(heading), np.sin(heading)]) * ds
            self.points[seg_off + 1 + i] = point
            kabs = max(abs(kappa[i]), 1e-3)
            self.speed[seg_off + i] = min(self.cfg["routeVmax"], np.sqrt(self.cfg["routeKvScale"] / kabs))
            heading += kappa[i] * ds
        self.end_heading = heading; self.filled = seg_off + num

    def _human_kappa(self, num):
        cap = self.cfg["routeHumanKappaCap"]; ds = self.cfg["routeSegmentLength"]
        amin = np.deg2rad(self.cfg["routeHumanBigAngleMinDeg"]); amax = np.deg2rad(self.cfg["routeHumanBigAngleMaxDeg"])
        out = np.zeros(num)
        for i in range(num):
            in_big = self.big_remain > 0.0
            if not in_big and self._u(0, 1) < self.cfg["routeHumanBigProbability"]:
                self.big_remain = max(2.0, np.ceil(self._u(amin, amax) / (cap * ds)))
                self.big_sign = 1.0 if self._u(0, 1) < 0.5 else -1.0; in_big = True
            if self._u(0, 1) > self.cfg["routeHumanPersist"]:
                self.h_sign = -self.h_sign
            mag = self._u(self.cfg["routeHumanWeaveMin"], self.cfg["routeHumanWeaveMax"]) * cap
            out[i] = self.big_sign * cap if in_big else self.h_sign * mag
            if in_big: self.big_remain -= 1.0
        return out

    def _extend(self):
        if not self.cfg["routeLazyExtend"] or self.last_seg < 0: return
        max_seg = len(self.speed)
        if self.filled >= max_seg or (self.filled - self.last_seg) >= self.cfg["routeExtendAheadMarginSegments"]: return
        num = min(self.cfg["routeExtendChunk"], max_seg - self.filled)
        if num > 0: self._build(num, False)


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
        self.prev_latent = np.zeros(8, np.float32); self.prev_decoded = np.zeros(29, np.float32)
        self.target = np.zeros(29); self.ep_start = 0.0; self.dr = {}
        self.ct_sum = 0.0; self.ct_count = 0   # cross-track (ball deviation from route) accumulators

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

    def reset(self, model, data, t, dr=None, route_seed=None):
        # route_seed (sweep route-control): re-seed this robot's route RNG so the SAME
        # route is reproduced -> different DR levels can be compared on identical routes,
        # cancelling route-difficulty variance (which otherwise dwarfs the DR effect).
        if route_seed is not None:
            self.route.rng = np.random.Generator(np.random.PCG64(int(route_seed)))
        r = self.apply_dr(model, dr) if dr is not None else self.sample_dr(model)
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
        data.qpos[self.ballq:self.ballq + 3] = bp
        data.qpos[self.ballq + 3:self.ballq + 7] = [1, 0, 0, 0]
        data.qvel[self.ballv:self.ballv + 6] = 0.0
        self.prev_latent[:] = 0; self.prev_decoded[:] = 0
        self.obs_hist = None    # history buffer refills from the first post-reset frame
        # per-episode constant latencies (resampled each reset, like training). Gated on
        # self.latency so the rng draw sequence (and thus DR sampling) is unchanged when off.
        self.ball_pos_hist = None; self.ball_vel_hist = None
        self.tgt_hist = np.tile(self.dq, (ACT_DELAY_K, 1)); self.substep = 0
        self.ball_delay = 0; self.act_delay = 0
        if self.latency:
            self.ball_delay = int(self.rng.integers(BALL_DELAY_RANGE[0], BALL_DELAY_RANGE[1] + 1))
            if self.rng.random() >= ACT_DELAY_ZERO_PROB:
                self.act_delay = int(self.rng.integers(ACT_DELAY_SUBSTEPS[0], ACT_DELAY_SUBSTEPS[1] + 1))
        bq = data.qpos[self.bq + 3:self.bq + 7].copy()
        fwd = np.zeros(3); mujoco.mju_rotVecQuat(fwd, np.array([1.0, 0, 0]), bq)
        self.cmd = self.route.reset(bp[:2], fwd[:2], CMD_MODE)
        self.target = data.qpos[self.qadr].copy(); self.ep_start = t
        self.hold_target = self.target.copy()   # standby pose to PD-hold during --standby-hold-s
        self.ct_sum = 0.0; self.ct_count = 0   # per-episode cross-track

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
        if self.latency:
            if self.ball_pos_hist is None:
                K = BALL_DELAY_RANGE[1] + 1
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
        actions, latent, *_ = self.sess.run(None, {"obs": self._obs(data)})
        self.prev_decoded = actions[0].copy(); self.prev_latent = latent[0].copy()
        self.target = np.clip(self.dq + self.ascale * actions[0], JC - JHW, JC + JHW)
        if self.latency:
            # action-delay ring: 0 = this step's target; reset the within-step counter
            self.tgt_hist = np.roll(self.tgt_hist, 1, axis=0); self.tgt_hist[0] = self.target
            self.substep = 0

    def torque(self, data, target):
        q = data.qpos[self.qadr]; qd = data.qvel[self.vadr]
        kp, kd = (self.skp, self.skd) if self._holding else (self.kp, self.kd)
        return np.clip(kp * (target - q) - kd * qd, -EFFORT_LIMIT, EFFORT_LIMIT)

    def apply(self, data):
        if self.latency and not self._holding:
            # at sub-step s with delay d, apply the target from ceil(max(d-s,0)/dec) steps ago
            deficit = max(self.act_delay - self.substep, 0)
            back = min(ACT_DELAY_K - 1, -(-deficit // DECIMATION))
            target = self.tgt_hist[back]
            self.substep += 1
        else:
            target = self.target
        data.ctrl[self.aadr] = self.torque(data, target)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=None,
                    help="grid spacing (m). Default: tight 6m for viewer/record (visual), "
                         "but eval/sweep auto-widen to 2*route_len+20 so robots can NEVER reach "
                         "a neighbour mid-episode (inter-robot collisions otherwise inflate fall-rate).")
    ap.add_argument("--episode-s", type=float, default=20.0)
    ap.add_argument("--route-len", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--eval", action="store_true", help="batch random-DR eval (no window/video), prints + CSV")
    ap.add_argument("--sweep", action="store_true", help="systematic 1-param-at-a-time sweep (others at nominal)")
    ap.add_argument("--sweep-levels", type=int, default=9, help="route-controlled sweep: discrete values per param")
    ap.add_argument("--route-bank", type=int, default=12, help="route-controlled sweep: # of fixed routes EVERY level is tested on")
    ap.add_argument("--sweep-reps", type=int, default=150, help="(deprecated — sweep now uses --sweep-levels x --route-bank)")
    ap.add_argument("--csv", default="", help="write per-episode records to this CSV")
    ap.add_argument("--plot", default="", help="write a DR-sweep figure (png) of cross-track & fall-rate vs each DR param")
    ap.add_argument("--record", default="", help="output mp4 path (offscreen render, no window)")
    ap.add_argument("--out-dir", default="", help="folder for ALL outputs (video+plots+csvs); created if missing")
    ap.add_argument("--seconds", type=float, default=35.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--onnx", default=ONNX, help="policy ONNX to evaluate (default: iter50000)")
    ap.add_argument("--reset", default=RESET_FILE, help="reset-state file (default: WalkF_RF motion frame 0)")
    ap.add_argument("--latency", action="store_true",
                    help="replicate the v2 training latency DR: per-episode ball-obs lag (1-3 "
                         "steps) + action lag (0-20ms, 30%% zero). Off = feed current values.")
    ap.add_argument("--standby-hold-s", type=float, default=0.0,
                    help="hold the standby reset pose (PD, no policy) for this many seconds at the "
                         "start of every episode, then hand off to the policy with fresh memory")
    args = ap.parse_args()
    ROUTE_CFG["routeLength"] = args.route_len   # route must outlast the episode (no run-out)
    if args.spacing is None:
        # eval/sweep are METRIC runs: widen so a robot's whole route (<= route_len from its
        # origin) can't overlap a neighbour's -> zero inter-robot collisions skewing fall-rate.
        # viewer/record are VISUAL: keep tight so all robots fit in one camera frame.
        args.spacing = (2.0 * args.route_len + 20.0) if (args.eval or args.sweep) else 6.0
        print(f"[multi] spacing auto -> {args.spacing:.0f}m "
              f"({'metric: isolated' if (args.eval or args.sweep) else 'visual: tight'})")
    global _OUT_BASE
    if args.out_dir:
        _OUT_BASE = os.path.expanduser(args.out_dir)
        if not os.path.isabs(_OUT_BASE):
            _OUT_BASE = os.path.join(REPO_DIR, _OUT_BASE)
        os.makedirs(_OUT_BASE, exist_ok=True)
        print(f"[multi] outputs -> {_OUT_BASE}")
    args.plot = resolve_out(args.plot); args.csv = resolve_out(args.csv)
    args.record = resolve_out(args.record)
    if args.robots > 64:
        print(f"[multi] capping --robots {args.robots} -> 64 (one MuJoCo model can't hold hundreds; "
              f"eval episodes accumulate over time, so raise --seconds for more data instead).")
        args.robots = 64

    model, grid = compose(args.robots, args.spacing)
    data = mujoco.MjData(model)
    print(f"[multi] policy: {args.onnx}")
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    rs = parse_reset(args.reset)

    robots = [Robot(model, k, grid[k], sess, meta, rs, args.seed) for k in range(args.robots)]
    for rb in robots:
        rb.latency = args.latency
    mujoco.mj_resetData(model, data)
    for rb in robots:
        rb.reset(model, data, 0.0)
    mujoco.mj_forward(model, data)

    ts = model.opt.timestep
    period_dt = ts * DECIMATION          # one control period (50 Hz)
    resets = [0] * args.robots
    records = []   # one row per COMPLETED episode: (mass,radius,foot,ball,cross_track,fell,duration)

    def advance():
        # one control period: policy once, PD every physics step; return ended robots
        for rb in robots:
            rb.policy_step(data, hold=(data.time - rb.ep_start) < args.standby_hold_s)
        for _ in range(DECIMATION):
            for rb in robots:
                rb.apply(data)
            mujoco.mj_step(model, data)
        t = data.time
        return [j for j, rb in enumerate(robots)
                if rb.base_z(data) < FALL_Z or (t - rb.ep_start) >= args.episode_s]

    def control_period():
        ended = advance()
        t = data.time
        for j in ended:
            rb = robots[j]; fell = rb.base_z(data) < FALL_Z
            records.append((rb.dr["mass"], rb.dr["radius"], rb.dr["foot"], rb.dr["ball"],
                            rb.ct_sum / max(1, rb.ct_count), 1.0 if fell else 0.0, t - rb.ep_start))
            rb.reset(model, data, data.time); resets[j] += 1

    n_periods = int(args.seconds / period_dt)
    gcx = float(np.mean([g[0] for g in grid])); gcy = float(np.mean([g[1] for g in grid]))
    cam_dist = max(args.spacing * 2.4 + 6.0, 14.0)

    if args.eval:
        for p in range(n_periods):
            control_period()
            if not np.all(np.isfinite(data.qpos)):
                print("[eval] DIVERGED — stopping"); break
            if p % 250 == 0:
                print(f"\r[eval] {p*period_dt:5.0f}/{args.seconds:.0f}s  episodes={len(records)}", end="", flush=True)
        aggregate_eval(records, args)
        if records:
            plot_eval(records, args.plot or resolve_out("eval_plot.png"))
    elif args.sweep:
        NOMINAL = dict(mass=0.391, radius=0.10, foot=0.8, ball=0.5)  # deploy nominal
        axes = [("ball_mass (kg)", "mass", SWEEP_RANGES["ball_mass"]),
                ("ball_radius (m)", "radius", SWEEP_RANGES["ball_radius"]),
                ("foot_friction", "foot", SWEEP_RANGES["foot_friction"]),
                ("ball_friction", "ball", SWEEP_RANGES["ball_friction"])]
        # ROUTE-CONTROLLED sweep: discretise each param into L levels and run EVERY level
        # on the SAME bank of R fixed routes. Route difficulty then cancels between levels,
        # so a level-to-level change in fall/cross-track is the PARAM's effect, not route luck.
        L = max(2, args.sweep_levels); R = max(1, args.route_bank)
        route_seeds = list(range(R))
        conds = []
        for pi, (lab, key, rng) in enumerate(axes):
            for lv in np.linspace(rng[0], rng[1], L):
                dr = dict(NOMINAL); dr[key] = float(lv)
                for rseed in route_seeds:
                    conds.append((pi, float(lv), dr, rseed))
        srng = np.random.default_rng(args.seed)
        srng.shuffle(conds)                              # interleave over time/robots
        total = len(conds); qi = 0; srec = []
        print(f"[sweep] {len(axes)} params x {L} levels x {R} routes = {total} episodes "
              f"(each level on the SAME {R} routes)")
        for rb in robots:
            if qi < total:
                pi, lv, dr, rseed = conds[qi]; qi += 1
                rb.cond = (pi, lv); rb.reset(model, data, 0.0, dr=dr, route_seed=rseed)
            else:
                rb.cond = None
        mujoco.mj_forward(model, data)
        done = 0
        while done < total:
            ended = advance()
            for j in ended:
                rb = robots[j]
                if rb.cond is not None:
                    fell = rb.base_z(data) < FALL_Z
                    srec.append((rb.cond[0], rb.cond[1], rb.ct_sum / max(1, rb.ct_count), 1.0 if fell else 0.0))
                    done += 1
                if qi < total:
                    pi, lv, dr, rseed = conds[qi]; qi += 1
                    rb.cond = (pi, lv); rb.reset(model, data, data.time, dr=dr, route_seed=rseed)
                else:
                    rb.cond = None; rb.reset(model, data, data.time)
            if not np.all(np.isfinite(data.qpos)):
                print("[sweep] DIVERGED"); break
            print(f"\r[sweep] {done}/{total} episodes", end="", flush=True)
        print()
        sweep_report(srec, axes, args, NOMINAL)
    elif args.headless:
        for _ in range(n_periods):
            control_period()
            if not np.all(np.isfinite(data.qpos)):
                print("DIVERGED"); return
        print(f"[multi] {args.robots} robots, {args.seconds:.0f}s | resets/robot={resets}")
        print_eval(robots, resets)
    elif args.record:
        import imageio
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat = [gcx, gcy, 0.4]; cam.distance = cam_dist
        cam.elevation = -28.0; cam.azimuth = 90.0
        rnd = mujoco.Renderer(model, height=720, width=1280)
        render_every = max(1, round((1.0 / args.fps) / period_dt))
        eff_fps = 1.0 / (period_dt * render_every)
        writer = imageio.get_writer(args.record, fps=eff_fps, codec="libx264",
                                    quality=8, macro_block_size=None, ffmpeg_log_level="error")
        nframes = 0
        for p in range(n_periods):
            control_period()
            if p % render_every == 0:
                rnd.update_scene(data, camera=cam)
                writer.append_data(draw_legend(rnd.render(), robots)); nframes += 1
        writer.close()
        print(f"[multi] saved {args.record} ({nframes} frames @ {eff_fps:.0f}fps, {args.seconds:.0f}s, "
              f"{args.robots} robots) | resets/robot={resets}")
        print_eval(robots, resets)
    else:
        with mujoco.viewer.launch_passive(model, data) as v:
            v.cam.lookat = [gcx, gcy, 0.4]; v.cam.distance = cam_dist
            v.cam.elevation = -28.0; v.cam.azimuth = 90.0
            while v.is_running():
                t0 = time.time()
                control_period()
                v.sync()
                dt = period_dt - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)


if __name__ == "__main__":
    main()
