"""Interactive / legacy pysim CLI on top of the benchmark engine.

Modes (mutually exclusive):
  (default)    live MuJoCo viewer, N robots dribbling side by side
  --record F   offscreen mp4 with a per-robot legend overlay
  --headless   step without a viewer (smoke test)
  --eval       batch random-DR eval (Monte Carlo over the training DR ranges)
  --sweep      1-param-at-a-time DR sweep on a fixed route bank (diagnostics:
               WHICH parameter hurts; the robustness benchmark answers how much
               the combined DR does)

  # watch (a terminal with a display):
  python -m sim2sim_benchmark.pysim
  # headless smoke test:
  python -m sim2sim_benchmark.pysim --headless --seconds 35

The standard benchmark lives in `python -m sim2sim_benchmark` (robustness /
capability condition tables); this module is for eyeballing and quick DR
diagnostics. Single-run knobs (--route-kappa, --push-dv, --ball-delay-steps,
...) apply one benchmark condition to every robot for viewer inspection.
"""
import argparse
import os
import sys
import time

import numpy as np

if "--record" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen render needs a headless GL backend
import mujoco
import mujoco.viewer

from . import engine

# all batch outputs (video + plots + csvs) land under this base dir; --out-dir overrides
_OUT_BASE = engine.REPO_DIR


def resolve_out(path):
    """Bare filenames / relative paths land in the output base dir; absolute paths respected."""
    if not path:
        return ""
    path = os.path.expanduser(path)
    return path if os.path.isabs(path) else os.path.join(_OUT_BASE, path)


def _font(size):
    from PIL import ImageFont
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_legend(frame, robots):
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame); draw = ImageDraw.Draw(img); font = _font(18)
    y = 6
    for k, rb in enumerate(robots):
        color = tuple(int(255 * c) for c in rb.color)
        draw.rectangle([8, y + 3, 26, y + 21], fill=color, outline=(0, 0, 0))
        cross_track = rb.ct_sum / max(1, rb.ct_count)
        dr = rb.dr
        text = (f"r{k}  m={dr['mass']:.2f}kg  r={dr['radius']:.3f}m  "
                f"uf={dr['foot']:.2f} ub={dr['ball']:.2f}  | cross-track={cross_track:.3f}m")
        draw.text((32, y), text, fill=(255, 255, 255), font=font)
        y += 26
    return np.asarray(img)


def aggregate_eval(records, args):
    if not records:
        print("[eval] no episodes completed — run longer (--seconds) or more --robots"); return
    # cols: mass, radius, foot, ball, cross_track, fell, duration,
    #       progress, ach_speed, cmd_speed, ball_lost, ball_dist
    arr = np.array(records)
    ct, fell, dur = arr[:, 4], arr[:, 5], arr[:, 6]
    ach, cmd, lost = arr[:, 8], arr[:, 9], arr[:, 10]
    print(f"\n=== EVAL: {len(arr)} episodes | {args.robots} robots | episode {args.episode_s:.0f}s | route {args.route_len:.0f}m ===")
    print(f"OVERALL  cross-track={np.nanmean(ct):.3f}m (median {np.nanmedian(ct):.3f})  "
          f"fall-rate={100*fell.mean():.0f}%  mean-survival={dur.mean():.1f}s")
    ok = np.isfinite(ach) & np.isfinite(cmd) & (cmd > 0.05)
    ratio = np.where(ok, ach / np.maximum(cmd, 1e-6), np.nan)
    print(f"         possession={100*(1-lost.mean()):.0f}%  ach-speed={np.nanmean(ach):.2f}m/s  "
          f"speed-ratio={np.nanmean(ratio):.2f}  progress={arr[:, 7].mean():.1f}m")
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
                  f"cross-track={np.nanmean(ct[mask]):.3f}m  fall={100*fell[mask].mean():.0f}%")
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ball_mass", "ball_radius", "foot_fric", "ball_fric", "cross_track_m", "fell", "duration_s",
                        "progress_m", "ach_speed_mps", "cmd_speed_mps", "ball_lost", "ball_dist_m"])
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
    arr = np.array(srec)  # cols: param_idx, level_value, cross_track, fell, duration
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
            w.writerow(["param_idx", "param_name", "level_value", "cross_track_m", "fell", "duration_s"])
            for pi, val, ct, fl, dur in srec:
                w.writerow([int(pi), names[int(pi)], f"{val:.5f}", f"{ct:.5f}", int(fl), f"{dur:.3f}"])
        print(f"  saved {args.csv} ({len(srec)} rows)")


def print_eval(robots, resets):
    print("==== eval (per robot) ====  (cross-track = mean ball deviation from route line)")
    for k, rb in enumerate(robots):
        ct = rb.ct_sum / max(1, rb.ct_count)
        dr = rb.dr
        print(f"  r{k}: cross-track={ct:.3f} m | resets={resets[k]} | last DR "
              f"m={dr['mass']:.2f}kg r={dr['radius']:.3f}m uf={dr['foot']:.2f} ub={dr['ball']:.2f}")


def main():
    ap = argparse.ArgumentParser(prog="sim2sim_benchmark.pysim", description=__doc__)
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
    ap.add_argument("--csv", default="", help="write per-episode records to this CSV")
    ap.add_argument("--plot", default="", help="write a DR-sweep figure (png) of cross-track & fall-rate vs each DR param")
    ap.add_argument("--record", default="", help="output mp4 path (offscreen render, no window)")
    ap.add_argument("--out-dir", default="", help="folder for ALL outputs (video+plots+csvs); created if missing")
    ap.add_argument("--seconds", type=float, default=35.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--onnx", default=engine.DEFAULT_ONNX,
                    help="policy ONNX (default: checkpoints/g1_dribble_s3_human_dr_iter80000)")
    ap.add_argument("--reset", default=engine.DEFAULT_RESET,
                    help="reset-state file (default: the standby reset, matching the default policy)")
    ap.add_argument("--latency", action="store_true",
                    help="replicate the checkpoint's training latency DR (per-episode ball-obs "
                         "lag + action lag, read from its env.yaml). Off = feed current values.")
    ap.add_argument("--standby-hold-s", type=float, nargs="+", default=[0.0],
                    metavar="S",
                    help="hold the standby reset pose (PD, no policy) at the start of every "
                         "episode, then hand off to the policy with fresh memory. One value "
                         "= fixed hold; two values = per-episode U(LO,HI) draw")
    # ---- single-run knobs: apply ONE benchmark condition to every robot ----
    ap.add_argument("--meshfoot", action="store_true",
                    help="mesh foot collision (ankle_roll STL) instead of the 7 capsules per foot")
    ap.add_argument("--hybridfoot", action="store_true",
                    help="hybrid foot collision: 7 capsules keep the floor contact, the "
                         "ankle_roll STL mesh carries the ball contact only")
    ap.add_argument("--cmd-mode", type=int, default=engine.CMD_MODE, help="route mode (4=human, 0=straight)")
    ap.add_argument("--route-vmax", type=float, default=None, help="commanded speed cap (m/s, default 2.0)")
    ap.add_argument("--route-v2-vel", action="store_true",
                    help="serve the TRAINING v2 speed chain (cruise mixture + pace modulation "
                         "+ brake/accel planning + windowed rate-limited C1-spline serve) for "
                         "route_v2_vel=true checkpoints. The C++ stack still serves the legacy "
                         "law, so this probes the training command model, not deploy parity")
    ap.add_argument("--route-cruise", type=float, nargs="+", default=None, metavar="V",
                    help="v2 cruise pace (with --route-v2-vel): one value pins it and turns "
                         "pace modulation off; two values sample U(min,max) per episode; "
                         "default = the training mixture U(1.1,2)/25%% slow U(0.5,1.1)")
    ap.add_argument("--route-kappa", type=float, default=None,
                    help="constant-curvature arc (signed 1/m); speed follows the trained kv law")
    ap.add_argument("--route-lead-m", type=float, default=1.0, help="straight lead-in before the arc (m)")
    ap.add_argument("--arc-angle-deg", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="finite arc angle range (deg); omit = endless arc (full circles)")
    ap.add_argument("--offroute-fail-m", type=float, default=None,
                    help="fail the episode when the ball is this far off the route (m)")
    ap.add_argument("--ball-far-fail-m", type=float, default=None,
                    help="fail the episode when the ball is this far from the robot (m)")
    ap.add_argument("--push-dv", type=float, default=0.0, help="base velocity kick (m/s) every --push-interval-s")
    ap.add_argument("--ball-push-dv", type=float, default=0.0, help="ball velocity kick (m/s)")
    ap.add_argument("--push-interval-s", type=float, default=5.0, help="seconds between pushes")
    ap.add_argument("--ball-delay-steps", type=int, nargs="+", default=None, metavar="N",
                    help="ball-obs lag (policy steps): one value pins it; two values = a fresh "
                         "per-STEP uniform draw U{LO..HI} (jitter, not a per-episode constant)")
    ap.add_argument("--act-delay-ms", type=float, default=None, help="pin action lag (ms)")
    ap.add_argument("--bridge-delay-ms", type=float, default=None,
                    help="C++ bridge staleness on ball+base obs (default 10 = deploy parity; "
                         "0 = legacy fresh-state)")
    ap.add_argument("--jitter", action="store_true", help="small reset yaw/xy jitter (de-determinize clean env)")
    # ---- measured real-world knobs (2026-07 sim2real debugging) ----
    ap.add_argument("--ball-damping", type=float, default=None,
                    help="pin the ball roll brake c (None = trained value; measured: "
                         "indoor floor 0.9, short-pile turf median 2.5, IQR 1.7-3.4)")
    ap.add_argument("--ball-roll-fric", type=float, default=None,
                    help="ball-floor Coulomb rolling friction coeff (None = MJCF 0.0001; "
                         "turf measured ~0.0017, pairs with --ball-damping 1.63)")
    ap.add_argument("--ball-bounce", type=float, default=None,
                    help="ball contact solref dampratio (None = 1.0, dead ball e~0.2; "
                         "0.6 -> kick restitution e~0.57, real ball vs shoe; avoid <=0.3)")
    ap.add_argument("--ground-tc", type=float, default=None,
                    help="foot-floor solref timeconst (s): elastic ground; None = rigid 0.01. "
                         "Short-pile turf on hard base ~0.02 (~7 mm sink)")
    ap.add_argument("--foot-fric", type=float, default=None,
                    help="pin foot-floor friction (ball mass/radius/friction keep sampling "
                         "the training DR); rubber sole on turf ~1.0")
    ap.add_argument("--joint-friction", type=float, default=None,
                    help="leg+waist frictionloss (N*m); None = trained nominal (0 for the "
                         "frictionless lineage). Real G1 joints ~1.0")
    ap.add_argument("--obs-noise-scale", type=float, default=0.0,
                    help="sensor-noise multiple of the training noise model (1 = realistic "
                         "IMU/encoder noise; protocol default 0)")
    ap.add_argument("--pile-drag", type=float, default=0.0,
                    help="turf pile drag coeff (N*s/m) on any foot below --pile-height; "
                         "conservative short-pile estimate 10")
    ap.add_argument("--pile-height", type=float, default=0.03,
                    help="pile layer height (m) for --pile-drag")
    ap.add_argument("--motor-curve", action="store_true",
                    help="torque-speed ceiling: tau_max(w) = peak*(1-|w|/vel_limit) "
                         "instead of the flat URDF-peak clip")
    ap.add_argument("--motor-vel-scale", type=float, default=None,
                    help="motor curve steepness: torque reaches zero at scale*vel_limit "
                         "(1.0 = URDF no-load speed, 0.5 = twice as steep/extreme)")
    ap.add_argument("--motor-peak-scale", type=float, default=None,
                    help="scale all effort limits for the firmware variant "
                         "(G1 EDU knee 120/139 ~ 0.86, basic 90/139 ~ 0.65)")
    ap.add_argument("--tether-back-n", type=float, nargs="+", default=[0.0], metavar="N",
                    help="tether pull opposite the robot's heading (N); one value = constant, "
                         "two values = piecewise-constant random U(LO,HI) redrawn every "
                         "U(0.3,1.0) s (intermittent cable tugs; negative = slack push)")
    ap.add_argument("--tether-down-n", type=float, nargs="+", default=[0.0], metavar="N",
                    help="downward tether load (N); same 1-or-2-value semantics as --tether-back-n")
    ap.add_argument("--tether-point", type=float, nargs=3, default=(-0.126, 0.0, 0.316),
                    metavar=("X", "Y", "Z"),
                    help="tether attachment in torso_link coords; default = the G1 back-plate "
                         "handle center (world ~(-0.13, 0, 1.12) at stand)")
    ap.add_argument("--ball-obs-bias", type=float, nargs=3, default=None,
                    metavar=("BX", "BY", "BZ"),
                    help="constant world-frame ball-center estimate bias (m), the "
                         "visible-surface centroid effect (e.g. 0 0.03 0.03)")
    ap.add_argument("--chest-yaw-bias", type=float, default=None,
                    help="constant obs-frame (chest rigid body) yaw miscalibration (deg)")
    ap.add_argument("--chest-rpy-bias", type=float, nargs=3, default=None,
                    metavar=("R", "P", "Y"),
                    help="three-axis obs-frame miscalibration (roll pitch yaw, deg), "
                         "right-multiplied; pitch/roll shift ball_pos_b first-order via "
                         "the ~0.84 m height lever (7.3 cm at 5 deg). Overrides --chest-yaw-bias")
    ap.add_argument("--bias-gravity", action="store_true",
                    help="apply the --chest-rpy-bias frame error to the BASE quat too "
                         "(tilts projected_gravity — hallucinated lean). Only deploy-"
                         "faithful if the C++ gravity source is the same mocap frame")
    ap.add_argument("--ball-mass", type=float, nargs="+", default=None, metavar="KG",
                    help="ball mass (kg): one value pins it, two = per-episode U(LO,HI); "
                         "default = the training DR range when --foot-fric is set")
    ap.add_argument("--ball-radius", type=float, nargs="+", default=None, metavar="M",
                    help="ball radius (m): same semantics as --ball-mass")
    ap.add_argument("--ball-start-dist", type=float, default=None,
                    help="task-start ball distance from pelvis (m). None = legacy fixed "
                         "0.65 straight ahead. Real 07-24 placements: median 0.57, range 0.43-0.61")
    ap.add_argument("--ball-start-bearing", type=float, default=0.0,
                    help="task-start ball bearing off body-forward (deg), used with --ball-start-dist")
    ap.add_argument("--sweep-scale", type=float, default=1.5,
                    help="--sweep envelope as a multiple of the checkpoint's training DR range")
    args = ap.parse_args()
    engine.ROUTE_CFG["routeLength"] = args.route_len   # route must outlast the episode (no run-out)
    if args.spacing is None:
        # eval/sweep are METRIC runs: widen so a robot's whole route (<= route_len from its
        # origin) can't overlap a neighbour's -> zero inter-robot collisions skewing fall-rate.
        # viewer/record are VISUAL: keep tight so all robots fit in one camera frame.
        metric = args.eval or args.sweep
        args.spacing = (2.0 * args.route_len + 20.0) if metric else 6.0
        print(f"[multi] spacing auto -> {args.spacing:.0f}m "
              f"({'metric: isolated' if metric else 'visual: tight'})")
    global _OUT_BASE
    if args.out_dir:
        _OUT_BASE = os.path.expanduser(args.out_dir)
        if not os.path.isabs(_OUT_BASE):
            _OUT_BASE = os.path.join(engine.REPO_DIR, _OUT_BASE)
        os.makedirs(_OUT_BASE, exist_ok=True)
        print(f"[multi] outputs -> {_OUT_BASE}")
    args.plot = resolve_out(args.plot); args.csv = resolve_out(args.csv)
    args.record = resolve_out(args.record)
    if args.robots > 64:
        print(f"[multi] capping --robots {args.robots} -> 64 (one MuJoCo model can't hold hundreds; "
              f"eval episodes accumulate over time, so raise --seconds for more data instead).")
        args.robots = 64

    # DR/sweep ranges + latency DR anchor on the checkpoint's own training config
    from .train_dr import read_train_dr
    train_cfg = read_train_dr(args.onnx)
    engine.configure_train_dr(train_cfg, sweep_scale=args.sweep_scale)
    if args.route_cruise is not None and len(args.route_cruise) > 2:
        ap.error("--route-cruise takes 1 (pin) or 2 (min max) values")
    if len(args.standby_hold_s) > 2:
        ap.error("--standby-hold-s takes 1 (fixed) or 2 (min max) values")
    ckpt_v2_vel = bool(train_cfg.get("route_v2_vel"))
    if ckpt_v2_vel != args.route_v2_vel:
        print(f"[multi] NOTE: checkpoint trained route_v2_vel={ckpt_v2_vel} but the "
              f"benchmark serves the {'v2' if args.route_v2_vel else 'legacy/C++'} speed "
              f"law ({'--route-v2-vel set' if args.route_v2_vel else 'add --route-v2-vel for the training-side model'})")

    if args.meshfoot and args.hybridfoot:
        ap.error("--meshfoot and --hybridfoot are mutually exclusive")
    if args.meshfoot:
        engine.set_meshfoot(True)
        print("[multi] foot collision: mesh (ankle_roll STL)")
    if args.hybridfoot:
        engine.set_hybridfoot(True)
        print("[multi] foot collision: hybrid (capsules->floor, ankle_roll STL->ball)")
    model, data, robots = engine.build_world(args.robots, args.spacing, args.onnx, args.reset, args.seed)
    # single-run CLI knobs become every robot's default condition
    cli_condition = engine.make_condition(
        route_mode=("arc" if args.route_kappa is not None else args.cmd_mode),
        arc_kappa=args.route_kappa, route_vmax=args.route_vmax, lead_in_m=args.route_lead_m,
        arc_angle_deg=args.arc_angle_deg,
        route_v2_vel=args.route_v2_vel,
        route_cruise=(None if args.route_cruise is None
                      else (args.route_cruise[0] if len(args.route_cruise) == 1
                            else list(args.route_cruise))),
        offroute_fail_m=args.offroute_fail_m, ball_far_fail_m=args.ball_far_fail_m,
        push_dv=args.push_dv, ball_push_dv=args.ball_push_dv,
        push_interval_s=args.push_interval_s,
        ball_obs_delay_steps=(None if args.ball_delay_steps is None
                              else (args.ball_delay_steps[0] if len(args.ball_delay_steps) == 1
                                    else [min(args.ball_delay_steps), max(args.ball_delay_steps)])),
        action_delay_ms=args.act_delay_ms,
        reset_jitter=args.jitter,
        ball_damping=args.ball_damping, ground_solref_tc=args.ground_tc,
        ball_roll_fric=args.ball_roll_fric, ball_bounce_dampratio=args.ball_bounce,
        joint_friction=args.joint_friction, obs_noise_scale=args.obs_noise_scale,
        ball_obs_bias=args.ball_obs_bias, chest_yaw_bias_deg=args.chest_yaw_bias,
        obs_frame_rpy_bias_deg=args.chest_rpy_bias,
        obs_frame_bias_gravity=args.bias_gravity,
        motor_curve=args.motor_curve, motor_peak_scale=args.motor_peak_scale,
        motor_vel_scale=args.motor_vel_scale,
        **({} if args.ball_start_dist is None
           else {"reset_ball_random": True, "reset_ball_dist": args.ball_start_dist,
                 "reset_ball_bearing": args.ball_start_bearing}),
        **({} if (args.foot_fric is None and not args.ball_mass and not args.ball_radius)
           else {"dr": dict(
               mass=(tuple(engine.DR["ball_mass"]) if not args.ball_mass
                     else (args.ball_mass[0] if len(args.ball_mass) == 1
                           else (min(args.ball_mass), max(args.ball_mass)))),
               radius=(tuple(engine.DR["ball_radius"]) if not args.ball_radius
                       else (args.ball_radius[0] if len(args.ball_radius) == 1
                             else (min(args.ball_radius), max(args.ball_radius)))),
               foot=(tuple(engine.DR["foot_friction"]) if args.foot_fric is None
                     else args.foot_fric),
               ball=tuple(engine.DR["ball_friction"]))}),
        standby_hold_s=(None if len(args.standby_hold_s) == 1 and args.standby_hold_s[0] <= 0
                        else (args.standby_hold_s[0] if len(args.standby_hold_s) == 1
                              else [min(args.standby_hold_s), max(args.standby_hold_s)])),
        **({} if args.bridge_delay_ms is None else {"bridge_delay_ms": args.bridge_delay_ms}))
    for rb in robots:
        rb.latency = args.latency
        rb.hold_s_default = 0.0   # the hold comes from the condition (range-capable)
        rb.episode_len_default = args.episode_s
        rb.default_condition = cli_condition
    mujoco.mj_resetData(model, data)
    for rb in robots:
        rb.reset(model, data, 0.0)
    engine.refresh_model_constants(model, data)
    mujoco.mj_forward(model, data)

    period_dt = model.opt.timestep * engine.DECIMATION   # one control period (50 Hz)
    resets = [0] * args.robots
    records = []   # one row per COMPLETED episode (see aggregate_eval column comment)

    _pile_prev = {}

    def _apply_pile_drag():
        # turf pile: horizontal -c*v on any foot whose sole (ankle z - 0.035)
        # is inside the pile layer. Foot velocity by finite difference; a reset
        # teleport shows up as an absurd velocity -> skip that tick.
        for rb in robots:
            for bid in rb.foot_bodies:
                pos = data.xpos[bid].copy()
                vel = (pos - _pile_prev.get(bid, pos)) / period_dt
                _pile_prev[bid] = pos
                sole = pos[2] - 0.035
                if sole < args.pile_height and np.linalg.norm(vel) < 5.0:
                    F = -args.pile_drag * vel
                    F[2] = 0.0
                    n = np.linalg.norm(F)
                    if n > 60.0:
                        F *= 60.0 / n
                    data.xfrc_applied[bid, :3] = F
                else:
                    data.xfrc_applied[bid, :3] = 0.0

    _tether_local = np.array(args.tether_point)
    _tether_on = any(abs(v) > 0 for v in args.tether_back_n + args.tether_down_n)
    _teth_rng = np.random.default_rng(args.seed + 0x7e7)
    _teth_state = {}   # robot idx -> {back, down, next_t}

    def _tether_forces(k, t):
        # 1 value = constant; 2 values = piecewise-constant random level in
        # U(LO,HI), redrawn every U(0.3,1.0) s (intermittent cable tugs)
        vb, vd = args.tether_back_n, args.tether_down_n
        if len(vb) == 1 and len(vd) == 1:
            return vb[0], vd[0]
        st = _teth_state.get(k)
        if st is None or t >= st["next_t"]:
            st = {"back": vb[0] if len(vb) == 1 else float(_teth_rng.uniform(min(vb), max(vb))),
                  "down": vd[0] if len(vd) == 1 else float(_teth_rng.uniform(min(vd), max(vd))),
                  "next_t": t + float(_teth_rng.uniform(0.3, 1.0))}
            _teth_state[k] = st
        return st["back"], st["down"]

    def _apply_tether():
        # trailing cable at the back-plate handle: pull opposite the robot's
        # heading + downward load, applied at the actual attachment point via
        # mj_applyFT so the induced pitch torque is included.
        data.qfrc_applied[:] = 0.0
        zero_t = np.zeros(3)
        for k, rb in enumerate(robots):
            back, down = _tether_forces(k, data.time)
            w, x, y, z = data.qpos[rb.bq + 3:rb.bq + 7]
            yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            force = np.array([-back * np.cos(yaw), -back * np.sin(yaw), -down])
            R = data.xmat[rb.torso_body].reshape(3, 3)
            point = data.xpos[rb.torso_body] + R @ _tether_local
            mujoco.mj_applyFT(model, data, force, zero_t, point,
                              rb.torso_body, data.qfrc_applied)

    def advance():
        if args.pile_drag > 0.0:
            _apply_pile_drag()
        if _tether_on:
            _apply_tether()
        return engine.step_control_period(model, data, robots)

    def control_period():
        ended = advance()
        t = data.time
        for j in ended:
            rb = robots[j]; m = rb.episode_metrics(data, t)
            records.append((rb.dr["mass"], rb.dr["radius"], rb.dr["foot"], rb.dr["ball"],
                            m["cross_track"], m["fell"], m["duration"], m["progress"],
                            m["ach_speed"], m["cmd_speed"], m["ball_lost"], m["ball_dist"]))
            rb.reset(model, data, data.time); resets[j] += 1
        if ended:
            engine.refresh_model_constants(model, data)

    n_periods = int(args.seconds / period_dt)
    grid_cx = float(np.mean([rb.gx for rb in robots]))
    grid_cy = float(np.mean([rb.gy for rb in robots]))
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
        axes = [("ball_mass (kg)", "mass", engine.SWEEP_RANGES["ball_mass"]),
                ("ball_radius (m)", "radius", engine.SWEEP_RANGES["ball_radius"]),
                ("foot_friction", "foot", engine.SWEEP_RANGES["foot_friction"]),
                ("ball_friction", "ball", engine.SWEEP_RANGES["ball_friction"])]
        # a channel the checkpoint never randomized has a degenerate range — L
        # identical levels there would burn L x route_bank episodes for nothing
        for lab, key, rng in [a for a in axes if a[2][1] - a[2][0] < 1e-9]:
            print(f"[sweep] skipping {lab}: not randomized in training (fixed {rng[0]:g})")
        axes = [a for a in axes if a[2][1] - a[2][0] >= 1e-9]
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
        engine.refresh_model_constants(model, data)
        mujoco.mj_forward(model, data)
        done = 0
        while done < total:
            ended = advance()
            for j in ended:
                rb = robots[j]
                if rb.cond is not None:
                    fell = rb.base_z(data) < engine.FALL_Z
                    srec.append((rb.cond[0], rb.cond[1], rb.ct_sum / max(1, rb.ct_count),
                                 1.0 if fell else 0.0, data.time - rb.ep_start))
                    done += 1
                if qi < total:
                    pi, lv, dr, rseed = conds[qi]; qi += 1
                    rb.cond = (pi, lv); rb.reset(model, data, data.time, dr=dr, route_seed=rseed)
                else:
                    rb.cond = None; rb.reset(model, data, data.time)
            if ended:
                engine.refresh_model_constants(model, data)
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
        cam.lookat = [grid_cx, grid_cy, 0.4]; cam.distance = cam_dist
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
        # passive viewer: WE own the physics loop, so space/pause must be
        # implemented here (the managed viewer's built-in keys don't apply)
        ui = {"paused": False, "step": False}

        def _key(keycode):
            if keycode == 32:        # space: pause / resume
                ui["paused"] = not ui["paused"]
                print("[multi] paused" if ui["paused"] else "[multi] resumed")
            elif keycode == 262:     # right arrow: single control period while paused
                ui["step"] = True

        with mujoco.viewer.launch_passive(model, data, key_callback=_key) as v:
            v.cam.lookat = [grid_cx, grid_cy, 0.4]; v.cam.distance = cam_dist
            v.cam.elevation = -28.0; v.cam.azimuth = 90.0
            while v.is_running():
                t0 = time.time()
                if not ui["paused"] or ui["step"]:
                    control_period()
                    ui["step"] = False
                v.sync()
                dt = period_dt - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)


if __name__ == "__main__":
    main()
