"""Self-contained interactive HTML comparison report (tensorboard-style).

  python -m sim2sim_benchmark.html_report \
      --run-dirs sim2sim_eval_results/runs/m80000 sim2sim_eval_results/runs/m90000 \
      --labels iter80000 iter90000 \
      --out sim2sim_eval_results/compare/report.html

  python -m sim2sim_benchmark.html_report --serve      # live mode: refresh (F5)
      # re-discovers runs and re-reads the CSVs on every page load

One HTML file, no external assets: experiment checkboxes in the sidebar select
which runs are drawn. A topline summary table leads, then a DIFFERENCE MAP that
answers "where do these runs actually differ, and is it real" -- every condition
carries a 95 % bootstrap CI on the difference (paired on route via
(condition, rep)), and nothing is coloured unless that interval clears zero. At
the default n=48 the rate noise floor alone is ~7 points, so an uncoloured cell
means "cannot tell apart", NOT "equal".

Remaining sections mirror the PNG figures (robustness axes, corner turn, human
dribble, u-turn, speed, control traces) with hover tooltips, +/-1 SE bands on
every metric, failure-mode breakdowns, and a per-condition video index with a
side-by-side lightbox
(links resolve relative to the report location, so keep the report under
sim2sim_eval_results/compare/ with runs under sim2sim_eval_results/runs/).
All aggregation happens here in Python; the embedded JS only toggles and
draws. Light/dark theme; colorblind-validated palette (grays are reserved for
reference lines).
"""
import argparse
import csv
import datetime
import html as html_lib
import json
import os
import re
import shlex
import sys

import numpy as np

from . import stats
from .real_world import REAL_WORLD

# CVD-validated 8-slot categorical order (adjacent-pair simulated dE >= 8 in
# both modes); slot index i is stored per run, hex lives in the CSS as --sI.
SERIES_LIGHT = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
                "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#2fb84a", "#d55181", "#c98500",
               "#199e70", "#d95926", "#9085e9", "#e66767"]

ROB_METRICS = [("survival", "survival rate (%)", "up"),
               ("train_survival", "training-faithful survival (%)", "up"),
               ("possession", "ball possession (%)", "up"),
               ("foot_ball_dist_p90", "foot-ball surface distance (m, p90)", "down"),
               ("ball_dist_p90", "robot-ball distance (m, p90)", "down"),
               ("speed_ratio", "speed ratio (achieved/cmd, survivors)", "one"),
               ("cross_track", "cross-track (m, survivors)", "down"),
               ("min_pelvis_z_p5", "lowest pelvis height (m, p5)", "up"),
               ("mean_duration", "mean episode duration (s)", "up")]
# Robustness groups are DISCOVERED from the CSVs (see robustness_groups); this
# table only supplies nicer axis labels for the ones we know about, so a new
# sweep group shows up automatically instead of being silently invisible.
ROB_GROUP_LABEL = {
    "ball_mass": "ball mass (kg)", "ball_radius": "ball radius (m)",
    "foot_friction": "foot friction", "ball_friction": "ball friction",
    "ball_damping": "ball roll brake c (1 m/s rolls 3.5/c m)",
    "base_push": "base push dv (m/s)", "ball_push": "ball push dv (m/s)",
    "obs_latency": "ball-obs latency (steps @ 50 Hz)",
    "act_latency": "action latency (ms)",
    "obs_noise": "obs noise scale (x trained)",
    "actuator_gain": "actuator kp/kd scale",
    "payload": "torso payload (kg)", "base_com": "torso CoM offset (m)",
    "encoder_offset": "joint encoder offset (rad)",
    "ball_radius_obs": "believed - true ball radius (m)",
    "reset_ball_dist": "task-start ball distance (m)",
    "reset_ball_bearing": "task-start ball bearing (deg)",
    "handover": "deploy standby hold before hand-off (s)",
    "dr_scale": "DR scale alpha",   # legacy CSVs only
}
# groups that belong to the capability sections, never the robustness grid
CAP_GROUPS = {"baseline", "straight_speed", "corner_turn", "u_turn",
              "human_dribble", "speed_tracking"}
CAP_METRICS = [("success", "success rate (%)", "up"),
               ("survival", "survival rate (%)", "up"),
               ("progress", "progress before termination (m)", "up"),
               ("ball_dist_p90", "robot-ball distance (m, p90)", "down"),
               ("cross_track", "cross-track (m, survivors)", "down")]


# fail_reason -> (legend label, CSS colour var). "timeout"/"completed" are the
# two readings of an empty fail_reason (see condition_stats).
REASON_STYLE = {
    "completed": ("completed", "var(--rz-done)"),
    "timeout": ("ran full clock", "var(--rz-done)"),
    "ball_far": ("ball lost", "var(--rz-far)"),
    "off_route": ("off route", "var(--rz-off)"),
    "fell": ("fell", "var(--rz-fell)"),
}
REASON_ORDER = ["completed", "timeout", "ball_far", "off_route", "fell"]
# extra slots for reason strings this file has never seen; cycles if exhausted
REASON_EXTRA_COLORS = ["var(--rz-x0)", "var(--rz-x1)", "var(--rz-x2)"]


def _ball_lost_train(r):
    """Sticky lost flag at the TRAINING-faithful 0.5 m threshold. New runs carry
    it in `ball_lost_05`; on pre-multi-threshold runs `ball_lost` itself was the
    0.5 m flag, so fall back to it. None (unknown) only on truly old runs with no
    possession column at all."""
    bl = r.get("ball_lost_05")
    return bl if bl is not None else r.get("ball_lost")


# Metrics the significance machinery compares. (key, label, row-extractor,
# statistic, is_rate). "down" metrics are handled in the JS by flipping the
# colour, not by negating the delta.
DIFF_METRICS = [
    ("survival", "survival rate (%)", lambda r: 1.0 - r["fell"], stats.rate_stat, True, "up"),
    # training's actual done-set: fall OR ball_lost. None on old-criterion runs
    # (no foot column) so they drop out of the pair instead of comparing a
    # never-firing flag against a real one -- see condition_stats.
    ("train_survival", "training-faithful survival (%)",
     lambda r: None if r["foot_ball_dist"] is None
     else (1.0 if (r["fell"] < 0.5 and _ball_lost_train(r) < 0.5) else 0.0),
     stats.rate_stat, True, "up"),
    ("success", "success rate (%)", lambda r: r["success"], stats.rate_stat, True, "up"),
    ("cross_track", "cross-track (m)", lambda r: r["ct"] if r["fell"] < 0.5 else None,
     stats.mean_stat, False, "down"),
    ("foot_ball_dist", "foot-ball distance (m)", lambda r: r["foot_ball_dist"],
     stats.mean_stat, False, "down"),
    ("ball_dist", "robot-ball distance (m)", lambda r: r["ball_dist"],
     stats.mean_stat, False, "down"),
    ("progress", "progress (m)", lambda r: r["progress"], stats.mean_stat, False, "up"),
]
# Up to this many runs we compute EVERY pair, so any run can serve as the
# subject; beyond it only pairs against the first run are computed and the page
# says so instead of silently offering a dead control.
#
# 8 = what the sidebar's "1-8 toggle" already lets you select, so the picker no
# longer dies before the run list does. Measured 2026-07-21 on 199 conditions x
# 6 metrics: 1.4 s per pair, i.e. 14 s at 5 runs and 39 s at 8 (28 pairs) --
# once, at generation time. The old cap of 4 was set against an estimate of
# ~1.5 s/pair over ~130 conditions, so it was never the cost that justified it.
MAX_DIFF_RUNS = 8


def _diff_rows(rows, extract):
    """Episode rows reduced to {pair, value} for one diff metric."""
    out = []
    for r in rows:
        value = extract(r)
        out.append(dict(pair=stats.pair_key(r),
                        v=float("nan") if value is None else float(value)))
    return out


def condition_diffs(parsed, labels):
    """Bootstrap CIs on every condition, for every comparable pair of runs.

    Returns {"i>j": {metric: [{cond, group, x, delta, lo, hi, sig, paired, n}]}}
    where i is the baseline and j the comparison. Only i<j is stored; the JS
    negates for the reverse direction.

    This is THE view the report was missing: the summary table used to colour
    any non-zero delta green/red, while the measured noise floor at n=48 is
    ~7 points. Here a delta is only marked significant when the 95% bootstrap
    CI on the difference excludes zero.
    """
    n = len(parsed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if n <= MAX_DIFF_RUNS or i == 0]
    if n > MAX_DIFF_RUNS:
        print(f"[html_report] {n} runs: computing significance only against "
              f"{labels[0]} (pairwise would be {n * (n - 1) // 2} comparisons)")
    out = {}
    for i, j in pairs:
        per_metric = {}
        for key, _label, extract, stat_fn, is_rate, _dir in DIFF_METRICS:
            entries = []
            for table in (0, 1):          # robustness, capability
                rows_a, rows_b = parsed[i][table], parsed[j][table]
                by_cond_a, by_cond_b = {}, {}
                for r in rows_a:
                    by_cond_a.setdefault(r["condition"], []).append(r)
                for r in rows_b:
                    by_cond_b.setdefault(r["condition"], []).append(r)
                for cond in sorted(set(by_cond_a) & set(by_cond_b)):
                    a = _diff_rows(by_cond_a[cond], extract)
                    b = _diff_rows(by_cond_b[cond], extract)
                    if not any(np.isfinite(r["v"]) for r in a):
                        continue          # metric undefined for this condition
                    ci = stats.bootstrap_diff_ci(a, b, "v", stat_fn,
                                                 pair_key="pair", rate=is_rate)
                    if not np.isfinite(ci["delta"]):
                        continue
                    entries.append(dict(
                        cond=cond, group=by_cond_a[cond][0]["group"],
                        x=by_cond_a[cond][0]["axis"],
                        delta=round(ci["delta"], 4),
                        lo=round(ci["lo"], 4) if np.isfinite(ci["lo"]) else None,
                        hi=round(ci["hi"], 4) if np.isfinite(ci["hi"]) else None,
                        sig=ci["significant"], paired=ci["paired"],
                        n=ci["n_pairs"] or min(ci["n_a"], ci["n_b"])))
            if entries:
                per_metric[key] = entries
        if per_metric:
            out[f"{i}>{j}"] = per_metric
    return out


def reason_legend(runs):
    """[(key, label, cssvar), ...] over the fail_reasons actually present.

    The old fixed 4-element list dropped anything else on the floor: the bars
    would quietly stop summing to 100% with no visual cue. Unknown reasons now
    get their own slot and legend entry."""
    present = set()
    for run in runs:
        for series in list(run["robustness"].values()) + [
                run["straight"], run["human"], run["tracking"],
                run["corner"]["L"], run["corner"]["R"],
                run["uturn"]["L"], run["uturn"]["R"]]:
            for point in series:
                present.update(point.get("reasons", {}))
    known = [k for k in REASON_ORDER if k in present]
    unknown = sorted(present - set(known))
    out = [(k, *REASON_STYLE[k]) for k in known]
    for i, k in enumerate(unknown):
        out.append((k, k, REASON_EXTRA_COLORS[i % len(REASON_EXTRA_COLORS)]))
    # stacked bars draw bottom-up; keep the "good" outcomes at the bottom
    return out


def robustness_groups(*row_lists):
    """Ordered (group, label) pairs actually present in the data.

    Known groups keep their curated order and label; anything else is appended
    alphabetically under its raw name. Groups with no episodes are dropped, so
    the report no longer renders empty panels for channels this checkpoint never
    swept (ball_damping / dr_scale on the current runs)."""
    present = {r["group"] for rows in row_lists for r in rows} - CAP_GROUPS - {""}
    known = [g for g in ROB_GROUP_LABEL if g in present]
    unknown = sorted(present - set(known))
    return [(g, ROB_GROUP_LABEL.get(g, g)) for g in known + unknown]


def _f(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def read_rows(path):
    if not os.path.exists(path):
        return []
    out, bad = [], 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        tail_col = reader.fieldnames[-1] if reader.fieldnames else None
        for r in reader:
            if tail_col is not None and r.get(tail_col) is None:
                bad += 1                     # truncated line from a hard-killed run
                continue
            fell, lost, axis = _f(r.get("fell")), _f(r.get("ball_lost")), _f(r.get("axis_value"))
            if None in (fell, lost, axis) or not r.get("condition"):
                bad += 1
                continue
            out.append(dict(
                condition=r["condition"], group=r.get("group", ""), axis=axis,
                # rep keys the route: the runner assigns route_seed = rep % bank,
                # so (condition, rep) names the SAME route in every run of the
                # same table -> stats.bootstrap_diff_ci can pair on it
                rep=r.get("rep", ""), route_seed=r.get("route_seed", ""),
                fell=fell, ball_lost=lost,
                # `ball_lost` is the MAIN threshold (0.8 m, possession); the
                # sticky flag at the training-faithful 0.5 m rides in ball_lost_05
                # (absent on pre-multi-threshold runs, where `ball_lost` WAS 0.5)
                ball_lost_05=_f(r.get("ball_lost_05")),
                ball_lost_10=_f(r.get("ball_lost_10")),
                success=_f(r.get("success")),
                ach=_f(r.get("ach_speed_mps")), cmd=_f(r.get("cmd_speed_mps")),
                ct=_f(r.get("cross_track_m")), r=_f(r.get("speed_corr_r")),
                duration=_f(r.get("duration_s")),
                progress=_f(r.get("progress_m")),
                ball_dist=_f(r.get("ball_dist_m")),
                # training's own possession measure (nearest foot to ball
                # surface); absent from pre-2026-07-20 CSVs, hence .get
                foot_ball_dist=_f(r.get("foot_ball_dist_m")),
                ball_lost_t=_f(r.get("ball_lost_t_s")),
                min_z=_f(r.get("min_pelvis_z")), max_tilt=_f(r.get("max_tilt_gvec_z")),
                slope=_f(r.get("speed_slope")), bias=_f(r.get("speed_bias")),
                resid=_f(r.get("speed_resid_mps")),
                reason=(r.get("fail_reason") or "").strip()))
    if bad:
        print(f"[html_report] {path}: skipped {bad} malformed rows")
    return out


def finite(values):
    return [v for v in values if v is not None and np.isfinite(v)]


def condition_stats(rows, fail_fast=None):
    """rows of one condition -> point stats used by every panel.

    `fail_fast` (None = infer) decides how an empty fail_reason is labelled.
    engine.episode_metrics writes `success` ONLY when a fail-fast criterion is
    armed, so a finite success column is an exact marker for it -- no group-name
    list to keep in sync.

    Continuous metrics are SURVIVORS-ONLY (`alive`). An episode truncated by a
    fall covers its route distance in less wall time, so an unfiltered
    ach_speed / speed_ratio mean RISES as the condition gets harder -- it moves
    opposite to what it claims to measure. cross-track was already filtered this
    way; ach/ratio now match it, and every panel reports the sample it used.

    foot_ball_dist / ball_dist percentiles are the continuous form of
    possession. Pre-2026-07-20 runs used a 1.5 m / 2.0 s pelvis-to-ball criterion
    that fired on ~1 episode in 3500, making the possession panel a flat 100%
    line; those CSVs lack the foot column and fall back to ball_dist.
    """
    n = len(rows)
    alive = [r for r in rows if r["fell"] < 0.5]
    if fail_fast is None:
        fail_fast = any(r["success"] is not None for r in rows)
    surv_p = 1.0 - float(np.mean([r["fell"] for r in rows]))
    # possession at the MAIN (eval) threshold; train_survival below uses the
    # training-faithful 0.5 m flag instead (see _ball_lost_train)
    poss_p = 1.0 - float(np.mean([r["ball_lost"] for r in rows]))
    # TRAINING-faithful survival. Training terminates on fall OR ball_lost OR
    # time_out (env.yaml `terminations`); the benchmark deliberately keeps
    # ball_lost as a metric so survival and possession read as separate failure
    # modes, which makes plain `survival` LOOSER than the thing training
    # optimised. Recombining them costs nothing -- both flags are already on
    # every row -- and it changes rankings: a policy that stays upright while
    # the ball rolls away scores well on `survival` and badly here.
    # None on runs whose ball_lost came from the old 1.5 m / 2.0 s pelvis
    # criterion (detected by the missing foot column): that flag fired on ~1
    # episode in 3500, so recombining it would just restate `survival` under a
    # name that promises training parity.
    train_surv_p = (None if all(r["foot_ball_dist"] is None for r in rows) else
                    float(np.mean([1.0 if (r["fell"] < 0.5 and _ball_lost_train(r) < 0.5)
                                   else 0.0 for r in rows])))

    def rate_se(p, m):
        return round(100.0 * (max(p * (1.0 - p), 0.0) / m) ** 0.5, 2) if m else None

    def mean_se(values, digits=4):
        """SEM of a continuous metric. Reported for the same reason the rates
        carry a binomial SE: episodes are independent draws (the trajectory
        follows the robot slot the episode landed on), so a bare mean invites
        reading noise as trend. Note the divisor is len(values), NOT n -- e.g.
        cross-track is survivors-only, so its sample is smaller than the
        condition's episode count and its SE correspondingly wider."""
        if len(values) < 2:
            return None
        return round(float(np.std(values, ddof=1) / len(values) ** 0.5), digits)

    def pct(values, q, digits=3):
        values = finite(values)
        return round(float(np.percentile(values, q)), digits) if values else None

    succ_vals = finite([r["success"] for r in rows])
    succ_p = float(np.mean(succ_vals)) if succ_vals else None
    ratios = [r["ach"] / r["cmd"] for r in alive
              if r["ach"] is not None and r["cmd"] is not None and r["cmd"] > 0.05]
    ct_vals = finite([r["ct"] for r in alive])
    ach_vals = finite([r["ach"] for r in alive])
    dur_vals = finite([r["duration"] for r in rows])
    prog_vals = finite([r["progress"] for r in rows])
    bd_vals = finite([r["ball_dist"] for r in rows])
    fb_vals = finite([r["foot_ball_dist"] for r in rows])
    mz_vals = finite([r["min_z"] for r in rows])
    reasons = {}
    for r in rows:
        # an empty fail_reason means "the episode was never cut short". In a
        # fail-fast (capability) condition that IS a clean completion; in a
        # robustness condition there is no fail-fast at all, so it only means the
        # clock ran out -- calling that "completed" would paint an episode that
        # drifted 5 m off route as a success.
        key = r["reason"] or ("completed" if fail_fast else "timeout")
        reasons[key] = reasons.get(key, 0) + 1
    return dict(
        n=n, n_alive=len(alive),
        survival=round(100.0 * surv_p, 2), survival_se=rate_se(surv_p, n),
        train_survival=(None if train_surv_p is None
                        else round(100.0 * train_surv_p, 2)),
        train_survival_se=(None if train_surv_p is None
                           else rate_se(train_surv_p, n)),
        possession=round(100.0 * poss_p, 2), possession_se=rate_se(poss_p, n),
        success=None if succ_p is None else round(100.0 * succ_p, 2),
        success_se=None if succ_p is None else rate_se(succ_p, len(succ_vals)),
        speed_ratio=round(float(np.mean(ratios)), 4) if ratios else None,
        speed_ratio_se=mean_se(ratios),
        speed_ratio_n=len(ratios),
        cross_track=round(float(np.mean(ct_vals)), 4) if ct_vals else None,
        cross_track_se=mean_se(ct_vals),
        cross_track_n=len(ct_vals),      # survivors only -- smaller than n
        ach_speed=round(float(np.mean(ach_vals)), 4) if ach_vals else None,
        ach_speed_se=mean_se(ach_vals),
        ach_speed_n=len(ach_vals),
        ball_dist_p50=pct(bd_vals, 50), ball_dist_p90=pct(bd_vals, 90),
        ball_dist_n=len(bd_vals),
        foot_ball_dist=round(float(np.mean(fb_vals)), 4) if fb_vals else None,
        foot_ball_dist_se=mean_se(fb_vals),
        foot_ball_dist_p90=pct(fb_vals, 90), foot_ball_dist_n=len(fb_vals),
        # how close the episode came to the fall threshold. Recorded raw so a
        # criterion change stays auditable instead of silently re-ruling old runs
        min_pelvis_z_p5=pct(mz_vals, 5), min_pelvis_z=round(
            float(np.median(mz_vals)), 4) if mz_vals else None,
        min_pelvis_z_n=len(mz_vals),
        progress=round(float(np.mean(prog_vals)), 3) if prog_vals else None,
        progress_se=mean_se(prog_vals, 3),
        progress_n=len(prog_vals),
        mean_duration=round(float(np.mean(dur_vals)), 2) if dur_vals else None,
        mean_duration_se=mean_se(dur_vals, 2),
        reasons=reasons)


def group_series(rows, group, split_sign=False):
    """[(x, stats), ...] sorted by x; split_sign -> {'L': [...], 'R': [...]}
    keyed by the sign of the axis (left/right turns), x = |axis|."""
    by_axis = {}
    for r in rows:
        if r["group"] == group:
            by_axis.setdefault(r["axis"], []).append(r)
    if not split_sign:
        return [dict(x=x, **condition_stats(g)) for x, g in sorted(by_axis.items())]
    out = {"L": [], "R": []}
    for x, g in sorted(by_axis.items(), key=lambda kv: abs(kv[0])):
        out["L" if x >= 0 else "R"].append(dict(x=abs(x), **condition_stats(g)))
    return out


def binned_pairs(path, nbins=16):
    """capability_speed_pairs.csv -> binned cmd-vs-actual curve (mean +/- sd
    per bin) plus pooled r, least-squares slope, and pair count."""
    if not os.path.exists(path):
        return None
    cmd, act = [], []
    for r in csv.DictReader(open(path)):
        c, a = _f(r.get("cmd_speed_mps")), _f(r.get("ball_speed_mps"))
        if c is not None and a is not None:
            cmd.append(c); act.append(a)
    cmd = np.array(cmd); act = np.array(act)
    if len(cmd) < 100 or cmd.std() < 1e-3:
        return None
    r = float(np.corrcoef(cmd, act)[0, 1]) if act.std() > 1e-9 else float("nan")
    slope = float(np.polyfit(cmd, act, 1)[0])
    edges = np.linspace(cmd.min(), cmd.max(), nbins + 1)
    pts = []
    for i in range(nbins):
        m = (cmd >= edges[i]) & (cmd < edges[i + 1] if i < nbins - 1 else cmd <= edges[i + 1])
        if m.sum() >= 20:
            pts.append(dict(x=round(float(0.5 * (edges[i] + edges[i + 1])), 4),
                            y=round(float(act[m].mean()), 4),
                            sd=round(float(act[m].std()), 4)))
    return dict(r=round(r, 3) if np.isfinite(r) else None,
                slope=round(slope, 4), n=int(len(cmd)), points=pts)


def traces(path, smooth_steps=25, keep_every=5):
    """capability_speed_traces.csv -> per-episode downsampled cmd + smoothed
    along-command speed (50 Hz -> 10 Hz after a 0.5 s moving average). Only
    the first axis_value present is kept (one traced condition per run)."""
    if not os.path.exists(path):
        return None
    rows = []
    for r in csv.DictReader(open(path)):
        try:
            rows.append((r["axis_value"], int(r["episode"]), int(r["step"]),
                         float(r["cmd_speed_mps"]), float(r["ball_speed_along_cmd_mps"])))
        except (KeyError, ValueError, TypeError):
            continue
    if not rows:
        return None
    axes = sorted({r[0] for r in rows})
    if len(axes) > 1:
        print(f"[html_report] {path}: {len(axes)} axis values, keeping {rows[0][0]}")
    first_axis = rows[0][0]
    eps = {}
    for axis, ep, step, cmd, along in rows:
        if axis == first_axis:
            eps.setdefault(ep, []).append((step, cmd, along))

    def clean(values):
        return [round(float(v), 3) if np.isfinite(v) else None for v in values]

    out = {}
    for ep, items in sorted(eps.items()):
        items.sort()
        cmd = np.array([i[1] for i in items]); along = np.array([i[2] for i in items])
        w = max(1, min(smooth_steps, len(along)))
        k = np.ones(w)
        along_s = np.convolve(along, k, mode="same") / np.convolve(np.ones(len(along)), k, mode="same")
        out[str(ep)] = dict(
            dt=0.02 * keep_every,
            mean_cmd=round(float(np.mean(finite(cmd))), 3) if finite(cmd) else None,
            mean_act=round(float(np.mean(finite(along_s))), 3) if finite(along_s) else None,
            cmd=clean(cmd[::keep_every]), act=clean(along_s[::keep_every]))
    return out


def video_index(run_dir, report_dir):
    out = {}
    root = os.path.join(run_dir, "videos")
    if not os.path.isdir(root):
        return out
    for test in sorted(os.listdir(root)):
        vdir = os.path.join(root, test)
        if not os.path.isdir(vdir):
            continue
        vids = {os.path.splitext(f)[0]: os.path.relpath(os.path.join(vdir, f), report_dir)
                for f in sorted(os.listdir(vdir)) if f.endswith(".mp4")}
        if vids:
            out[test] = vids
    return out


def toplines(straight, corner, human, uturn, cap_rows, pairs):
    """Scalar headline numbers for the summary table."""
    def passing(series, threshold=50.0):
        best = None
        for p in series:                     # sorted by x; stop at the first failure
            if p.get("success") is None:
                continue
            if p["success"] < threshold:
                break
            best = p["x"]
        return best

    track = [r for r in cap_rows if r["group"] == "speed_tracking"]
    tr = finite([r["r"] for r in track])

    def avg(key, digits=3):
        vals = finite([r.get(key) for r in track])
        return round(float(np.mean(vals)), digits) if vals else None

    return dict(
        tracking_slope=avg("slope"), tracking_bias=avg("bias"),
        tracking_resid=avg("resid"),
        max_speed=passing(straight),
        corner_L=passing(corner["L"]), corner_R=passing(corner["R"]),
        uturn_L=passing(uturn["L"]), uturn_R=passing(uturn["R"]),
        human_cap=passing(human),
        tracking_r=round(float(np.mean(tr)), 3) if tr else None)


def run_provenance(run_dir):
    """The two sidecars every run already writes, which no plotting module read.

    `<test>.fingerprint.json` is the semantic identity of the condition table
    (__main__.table_fingerprint). If two runs disagree, their curves share an x
    axis by coincidence only -- plot.py warned about this on the console, but the
    interactive report, which is where runs actually get compared, had no idea.

    `train_dr.json` records the DR the policy was TRAINED with, which is exactly
    the "match ITS training params" check CLAUDE.md mandates before trusting a
    comparison."""
    out = {"fingerprints": {}, "train": None}
    for test in ("robustness", "capability"):
        path = os.path.join(run_dir, f"{test}.fingerprint.json")
        if os.path.exists(path):
            try:
                out["fingerprints"][test] = json.load(open(path))["fingerprint"][:12]
            except (ValueError, KeyError, OSError):
                out["fingerprints"][test] = "unreadable"
    path = os.path.join(run_dir, "train_dr.json")
    if os.path.exists(path):
        try:
            blob = json.load(open(path))
            train = blob.get("train") or {}
            out["train"] = dict(
                onnx=os.path.basename(str(blob.get("onnx") or "")),
                source=train.get("source"),
                ball_mass=train.get("ball_mass_range"),
                ball_radius=train.get("ball_radius_range"),
                ball_friction=train.get("ball_friction_range"),
                foot_friction=train.get("foot_friction_range"),
                ball_damping=train.get("ball_damping"),
                obs_delay=train.get("ball_obs_delay_steps"),
                act_delay=train.get("action_delay_ms"),
                push_robot=(train.get("push_robot") or {}).get("dv"),
                push_ball=(train.get("push_ball") or {}).get("dv"))
        except (ValueError, KeyError, OSError):
            pass
    return out


def collect_run(run_dir, label, index, report_dir, rob_groups=None, rows=None):
    """One run -> the JSON blob the page draws. `rob_groups` is the union of
    robustness groups across ALL selected runs (so a group only one run has still
    gets a panel); `rows` reuses an already-parsed (rob, cap) pair."""
    rob, cap = rows if rows is not None else (
        read_rows(os.path.join(run_dir, "robustness.csv")),
        read_rows(os.path.join(run_dir, "capability.csv")))
    if rob_groups is None:
        rob_groups = robustness_groups(rob, cap)
    nominal = [r for r in rob if r["group"] == "baseline"]
    corner = group_series(cap, "corner_turn", split_sign=True)
    human = group_series(cap, "human_dribble")
    uturn = group_series(cap, "u_turn", split_sign=True)
    straight = group_series(cap, "straight_speed")
    csv_paths = [os.path.join(run_dir, f) for f in
                 ("robustness.csv", "capability.csv",
                  "capability_speed_pairs.csv", "capability_speed_traces.csv")]
    mtimes = [os.path.getmtime(p) for p in csv_paths if os.path.exists(p)]
    pairs = binned_pairs(os.path.join(run_dir, "capability_speed_pairs.csv"))
    return dict(
        label=label, color=index % 8,
        prov=run_provenance(run_dir),
        info=dict(dir=os.path.relpath(run_dir), n_rob=len(rob), n_cap=len(cap),
                  data_time=datetime.datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d %H:%M")
                  if mtimes else None),
        nominal=condition_stats(nominal) if nominal else None,
        robustness={g: group_series(rob, g) for g, _ in rob_groups},
        straight=straight, corner=corner, human=human, uturn=uturn,
        tracking=group_series(cap, "speed_tracking"),
        pairs=pairs,
        traces=traces(os.path.join(run_dir, "capability_speed_traces.csv")),
        videos=video_index(run_dir, report_dir),
        top=toplines(straight, corner, human, uturn, cap, pairs))


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="icon" href='data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect width="16" height="16" rx="4" fill="%232a78d6"/><circle cx="8" cy="8" r="3.2" fill="white"/></svg>'>
<style>
  :root {
    color-scheme: light;
    --page:#f6f6f2; --panel:#fdfdfc;
    --border:rgba(20,18,12,.09); --border2:rgba(20,18,12,.18);
    --rowhover:rgba(20,18,12,.03);
    --text:#151412; --text2:#4e4c47; --muted:#6f6d66;
    --grid:#e8e7e1; --axis:#c6c5bd;
    --accent:#2159a8; --wash:rgba(42,120,214,.10);
    --dgood:#0a6b0a; --dbad:#c22f2f;
    --shadow:0 1px 2px rgba(24,22,16,.05), 0 3px 12px rgba(24,22,16,.05);
    --shadow2:0 2px 6px rgba(24,22,16,.08), 0 10px 26px rgba(24,22,16,.09);
    --s0:#2a78d6; --s1:#008300; --s2:#e87ba4; --s3:#eda100;
    --s4:#1baf7a; --s5:#eb6834; --s6:#4a3aa7; --s7:#e34948;
    --rz-fell:#d03b3b; --rz-off:#6c5fc7; --rz-far:#fab219; --rz-done:#d9d8d1;
    --rz-x0:#00868b; --rz-x1:#a1568c; --rz-x2:#7a6a3a;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --page:#111110; --panel:#1c1c1a;
      --border:rgba(255,255,255,.09); --border2:rgba(255,255,255,.20);
      --rowhover:rgba(255,255,255,.045);
      --text:#f5f4f0; --text2:#c6c5bc; --muted:#98968d;
      --grid:#2a2a28; --axis:#3d3d39;
      --accent:#7fb0e8; --wash:rgba(107,156,224,.14);
      --dgood:#0ca30c; --dbad:#e05d5d;
      --shadow:0 1px 2px rgba(0,0,0,.4);
      --shadow2:0 4px 16px rgba(0,0,0,.5);
      --s0:#3987e5; --s1:#2fb84a; --s2:#d55181; --s3:#c98500;
      --s4:#199e70; --s5:#d95926; --s6:#9085e9; --s7:#e66767;
      --rz-fell:#e05a5a; --rz-off:#8b7fe0; --rz-far:#e0a930; --rz-done:#3a3a37;
      --rz-x0:#31a5a9; --rz-x1:#c07aa9; --rz-x2:#a3915a;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page:#111110; --panel:#1c1c1a;
    --border:rgba(255,255,255,.09); --border2:rgba(255,255,255,.20);
    --rowhover:rgba(255,255,255,.045);
    --text:#f5f4f0; --text2:#c6c5bc; --muted:#98968d;
    --grid:#2a2a28; --axis:#3d3d39;
    --accent:#7fb0e8; --wash:rgba(107,156,224,.14);
    --dgood:#0ca30c; --dbad:#e05d5d;
    --shadow:0 1px 2px rgba(0,0,0,.4);
    --shadow2:0 4px 16px rgba(0,0,0,.5);
    --s0:#3987e5; --s1:#2fb84a; --s2:#d55181; --s3:#c98500;
    --s4:#199e70; --s5:#d95926; --s6:#9085e9; --s7:#e66767;
    --rz-fell:#e05a5a; --rz-off:#8b7fe0; --rz-far:#e0a930; --rz-done:#3a3a37;
    --rz-x0:#31a5a9; --rz-x1:#c07aa9; --rz-x2:#a3915a;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; scrollbar-width:thin; scrollbar-color:var(--axis) transparent; }
  body { margin:0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         font-size:14px; background:var(--page); color:var(--text);
         -webkit-font-smoothing:antialiased; }
  :focus-visible { outline:2px solid var(--accent); outline-offset:2px; border-radius:4px; }
  #layout { display:flex; min-height:100vh; }

  #sidebar { width:268px; padding:16px 14px; border-right:1px solid var(--border);
             background:var(--panel); position:sticky; top:0; height:100vh; overflow-y:auto;
             flex-shrink:0; scrollbar-width:thin; }
  #dragbar { width:9px; margin:0 -4px; flex-shrink:0; position:sticky; top:0; height:100vh;
             cursor:col-resize; z-index:5; }
  #dragbar:hover, #dragbar.dragging {
    background:linear-gradient(to right, transparent 3px, var(--accent) 3px,
               var(--accent) 6px, transparent 6px); }
  body.resizing { cursor:col-resize; user-select:none; }
  #sidebar h1 { font-size:14.5px; margin:0 0 12px; display:flex; align-items:center; gap:8px;
                letter-spacing:-.01em; }
  .brandmark { width:13px; height:13px; border-radius:4px; flex-shrink:0;
               background:linear-gradient(135deg, var(--s0), var(--s4)); }
  #sidebar h2 { font-size:10.5px; text-transform:uppercase; letter-spacing:.09em;
                color:var(--muted); margin:20px 0 7px; display:flex; align-items:center;
                justify-content:space-between; font-weight:600; }
  #sidebar button { font:inherit; font-size:11.5px; color:var(--accent); background:none;
                    border:1px solid var(--border); border-radius:6px; padding:2px 9px;
                    cursor:pointer; transition:background .12s, border-color .12s; }
  #sidebar button:hover { background:var(--wash); border-color:var(--accent); }
  .runrow { display:flex; align-items:center; gap:7px; padding:4px 6px; font-size:13px;
            border-radius:7px; transition:background .12s; }
  .runrow:hover { background:var(--rowhover); }
  .runrow input { cursor:pointer; accent-color:var(--accent); margin:0; }
  #sidebar button.runname { all:unset; cursor:pointer; flex:1; overflow:hidden;
                            text-overflow:ellipsis; white-space:nowrap; font-size:13px;
                            color:var(--text); }
  #sidebar button.runname:hover { color:var(--accent); text-decoration:underline dotted; }
  #sidebar button.runname:focus-visible { outline:2px solid var(--accent);
                                          outline-offset:1px; border-radius:3px; }
  .runn { font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; }
  .swatch { width:12px; height:12px; border-radius:4px; flex-shrink:0; }
  .navlink { display:block; font-size:13px; color:var(--text2); text-decoration:none;
             padding:3px 10px; border-radius:7px; transition:background .12s, color .12s; }
  .navlink:hover { color:var(--accent); background:var(--rowhover); }
  .navlink.active { color:var(--accent); background:var(--wash); font-weight:600; }

  #main { flex:1; padding:22px 28px 70px; min-width:0; }
  #pagehead { display:flex; justify-content:space-between; align-items:flex-end;
              gap:16px 24px; flex-wrap:wrap; margin:2px 0 28px; }
  #pagehead h1 { margin:0; font-size:24px; letter-spacing:-.015em; }
  .eyebrow { font-size:10.5px; font-weight:600; text-transform:uppercase;
             letter-spacing:.1em; color:var(--accent); }
  #pagehead .eyebrow { margin-bottom:3px; }
  .headmeta { display:flex; gap:8px; flex-wrap:wrap; font-size:12.5px; color:var(--muted); }
  .mchip { border:1px solid var(--border); background:var(--panel); border-radius:999px;
           padding:3px 12px; font-variant-numeric:tabular-nums; }
  .mchip.live { border-color:var(--accent); color:var(--accent); }

  section { margin-bottom:44px; scroll-margin-top:10px; }
  section > h2 { font-size:19px; letter-spacing:-.012em; border-bottom:1px solid var(--border);
                 padding-bottom:9px; margin:0 0 10px; }
  section > h2 .eyebrow { display:block; margin-bottom:3px; }
  .note { font-size:12.5px; color:var(--muted); margin:4px 0 12px; max-width:960px;
          line-height:1.45; }

  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(310px, 1fr)); gap:16px; }
  .robrow { display:grid; grid-template-columns:repeat(auto-fill, minmax(250px, 1fr));
            gap:14px; margin:10px 0 6px; }
  details.robgroup { margin-bottom:14px; }
  details.robgroup > summary { cursor:pointer; font-size:13.5px; font-weight:600;
                               padding:5px 8px; color:var(--text2); list-style:none;
                               display:flex; align-items:center; gap:8px; border-radius:7px;
                               transition:background .12s; width:fit-content; }
  details.robgroup > summary::-webkit-details-marker { display:none; }
  details.robgroup > summary::before { content:"\25B8"; font-size:11px; color:var(--muted);
                                       transition:transform .15s; }
  details.robgroup[open] > summary::before { transform:rotate(90deg); }
  details.robgroup > summary:hover { background:var(--rowhover); }

  .panel { background:var(--panel); border:1px solid var(--border); border-radius:11px;
           padding:11px 13px 6px; box-shadow:var(--shadow); transition:box-shadow .18s; }
  .panel:hover { box-shadow:var(--shadow2); }
  .panel h3 { font-size:12.5px; margin:1px 0 5px; font-weight:600; color:var(--text2);
              display:flex; justify-content:space-between; align-items:baseline; gap:6px; }
  .panel h3 .dir { font-size:10.5px; font-weight:400; color:var(--muted); white-space:nowrap; }
  .panelsub { font-size:11px; color:var(--muted); margin:-3px 0 3px; }

  svg.chart { display:block; }
  svg text { font-size:10px; fill:var(--muted); font-variant-numeric:tabular-nums; }
  svg .axisline { stroke:var(--axis); stroke-width:1; }
  svg .gridline { stroke:var(--grid); stroke-width:1; }
  svg .crossline { stroke:var(--muted); stroke-width:1; }
  #main [data-run] { transition:opacity .15s; }

  .legend { display:flex; flex-wrap:wrap; gap:7px 8px; align-items:center;
            font-size:12px; color:var(--text2); margin:2px 0 12px; }
  .chip { display:inline-flex; align-items:center; gap:6px; border:1px solid var(--border);
          background:var(--panel); border-radius:999px; padding:3px 11px;
          transition:border-color .12s, background .12s; }
  .chip input { margin:0; accent-color:var(--accent); cursor:pointer; }
  .chip:has(input) { cursor:pointer; }
  .chip:has(input:checked) { border-color:var(--accent); background:var(--wash); }
  .realtag { margin-left:9px; font-size:11px; font-weight:500; color:var(--text2);
             border:1px dashed var(--border); border-radius:999px; padding:1px 8px;
             cursor:help; }
  .filterrow { display:flex; flex-wrap:wrap; gap:6px 8px; font-size:12.5px;
               color:var(--text2); margin:0 0 14px; }
  .filterrow label { display:inline-flex; gap:6px; align-items:center; cursor:pointer;
                     border:1px solid var(--border); background:var(--panel);
                     border-radius:999px; padding:3px 11px;
                     transition:border-color .12s, background .12s; }
  .filterrow label:has(input:checked) { border-color:var(--accent); background:var(--wash); }
  .filterrow input { margin:0; accent-color:var(--accent); }

  table.summary { border-collapse:separate; border-spacing:0; font-size:13px;
                  background:var(--panel); border:1px solid var(--border);
                  border-radius:11px; overflow:hidden; box-shadow:var(--shadow); }
  table.summary th, table.summary td { padding:6px 14px; text-align:right;
                                       font-variant-numeric:tabular-nums;
                                       border-bottom:1px solid var(--border); }
  table.summary thead th { font-size:10.5px; font-weight:600; text-transform:uppercase;
                           letter-spacing:.06em; color:var(--muted);
                           background:var(--rowhover); }
  table.summary tbody tr:last-child td { border-bottom:none; }
  table.summary tbody tr:hover td { background:var(--rowhover); }
  table.summary tr.sgroup td { font-size:10px; font-weight:600; text-transform:uppercase;
                               letter-spacing:.08em; color:var(--accent);
                               background:var(--rowhover); text-align:left;
                               padding:5px 14px 4px; }
  table.summary td.mname, table.summary th.mname { text-align:left;
                                                   font-variant-numeric:normal; }
  table.summary td.mname .dir { color:var(--muted); font-size:11px; margin-left:5px; }
  table.summary .best { font-weight:650; }
  table.summary .best::before { content:"\25CF"; color:var(--accent); font-size:7px;
                                vertical-align:2px; margin-right:5px; }
  table.summary .delta { font-size:11px; margin-left:5px; color:var(--muted); }
  .dgood { color:var(--dgood); } .dbad { color:var(--dbad); }
  .dnull { color:var(--muted); }
  #cmpbanner { display:none; margin:0 0 14px; padding:10px 13px; border-radius:8px;
               font-size:12.5px; line-height:1.5;
               background:color-mix(in srgb, var(--dbad) 11%, var(--card));
               border:1px solid color-mix(in srgb, var(--dbad) 45%, transparent); }
  #cmpbanner b { color:var(--dbad); }
  #trainbox { margin:0 0 16px; font-size:11.5px; }
  #trainbox table { border-collapse:collapse; }
  #trainbox td, #trainbox th { padding:2px 10px 2px 0; text-align:left;
                               white-space:nowrap; color:var(--muted); }
  #trainbox th { color:var(--fg); font-weight:600; }
  #trainbox td.mismatch { color:var(--dbad); font-weight:600; }
  .ci { color:var(--muted); font-size:11px; margin-left:4px; white-space:nowrap; }
  .ctlrow { display:flex; flex-wrap:wrap; gap:6px; align-items:center;
            width:100%; margin:1px 0; }
  .ctllabel { color:var(--muted); font-size:11px; text-transform:uppercase;
              letter-spacing:.05em; margin-right:2px; min-width:112px; }
  .ctlnote { color:var(--muted); font-size:11.5px; }
  .foldcard { margin:6px 0; }
  .foldcard > summary { cursor:pointer; padding:5px 2px; font-size:12.5px;
                        color:var(--muted); }
  .foldcard > summary:hover { color:var(--fg); }
  /* summary doubles as the card title once every comparison is foldable */
  .foldcard[open] > summary { color:var(--fg); font-size:13.5px; font-weight:600; }
  .colorkey { display:flex; flex-wrap:wrap; gap:14px; margin:0 0 10px;
              font-size:11.5px; color:var(--muted); }
  .keyitem { display:inline-flex; gap:5px; align-items:center; }
  .keyswatch { width:22px; height:11px; border-radius:2px; display:inline-block;
               background:color-mix(in srgb, var(--muted) 17%, transparent); }
  .keyswatch.cgood { background:var(--dgood); }
  .keyswatch.cbad { background:var(--dbad); }
  .verdict { margin:2px 0 4px; font-size:13px; }
  .verdict .vgood { color:var(--dgood); font-weight:650; }
  .verdict .vbad { color:var(--dbad); font-weight:650; }
  .verdict .vnull { color:var(--muted); font-weight:650; }
  .verdict .vsep { color:var(--muted); margin:0 7px; }
  .verdict .vnote { color:var(--muted); }
  .maprow { display:grid; grid-template-columns:210px 1fr 46px; gap:10px;
            align-items:center; padding:3px 4px; border-radius:5px; font-size:12px; }
  .maprow:hover { background:var(--wash); }
  .maprow.quiet .mapname { color:var(--muted); }
  .mapname { display:flex; gap:6px; align-items:center; white-space:nowrap;
             overflow:hidden; text-overflow:ellipsis; user-select:none; }
  .mapcaret { color:var(--muted); font-size:10px; width:9px; }
  .mapstrip { display:flex; gap:2px; }
  .mapcell { flex:1 1 0; height:15px; border-radius:2px;
             background:color-mix(in srgb, var(--muted) 17%, transparent); }
  .mapcell.cgood { background:var(--dgood); }
  .mapcell.cbad { background:var(--dbad); }
  .maptally { text-align:right; color:var(--muted); font-size:11px;
              font-variant-numeric:tabular-nums; }
  .maptally.on { color:var(--fg); font-weight:600; }
  .forestwrap { margin:2px 0 10px 24px; padding:6px 0 6px 10px;
                border-left:2px solid var(--grid); }
  .forestrow { display:grid; grid-template-columns:170px 1fr 148px; gap:8px;
               align-items:center; padding:1px 0; font-size:11px; }
  .forestrow .fname { color:var(--fg); overflow:hidden; text-overflow:ellipsis;
                      white-space:nowrap; }
  .forestrow .fnum { color:var(--muted); text-align:right; font-variant-numeric:tabular-nums; }
  .runth { display:inline-flex; align-items:center; gap:6px; }

  #videos-host h3 { font-size:14px; margin:18px 0 10px; text-transform:uppercase;
                    letter-spacing:.05em; color:var(--text2); }
  .vcat { margin:0 0 16px; }
  .vcat h4 { font-size:12.5px; font-weight:600; color:var(--text2); margin:0 0 6px;
             display:flex; align-items:center; gap:7px; }
  .vstrip { display:flex; gap:10px; overflow-x:auto; padding:2px 2px 8px;
            scrollbar-width:thin; }
  .vtile { flex:0 0 auto; width:216px; }
  .vtile video { width:100%; display:block; border-radius:9px; background:#000;
                 cursor:pointer; border:1px solid var(--border);
                 transition:border-color .12s, box-shadow .12s; }
  .vtile video:hover { border-color:var(--accent); box-shadow:var(--shadow); }
  .vtile .vcap { font-size:11.5px; color:var(--muted); margin-top:3px;
                 display:flex; justify-content:space-between; align-items:center;
                 font-variant-numeric:tabular-nums; }
  .vtile .vcap button { all:unset; cursor:pointer; color:var(--accent); font-size:12px;
                        padding:0 4px; border-radius:4px; }
  .vtile .vcap button:hover { background:var(--wash); }

  #tip { position:fixed; z-index:30; display:none; background:var(--panel);
         border:1px solid var(--border); border-radius:9px; padding:7px 10px;
         font-size:12px; pointer-events:none; box-shadow:var(--shadow2);
         max-width:340px; }
  #tip .tip-h { color:var(--muted); margin-bottom:3px; font-size:11.5px; }
  #tip .tip-row { display:flex; align-items:center; gap:6px; padding:1px 0; }
  #tip .tip-v { font-weight:600; font-variant-numeric:tabular-nums; }
  #tip .tip-l { color:var(--text2); overflow:hidden; text-overflow:ellipsis;
                white-space:nowrap; }

  #lightbox { position:fixed; inset:0; z-index:20; background:rgba(0,0,0,.66);
              backdrop-filter:blur(3px); display:none; align-items:flex-start;
              justify-content:center; overflow-y:auto; padding:30px 20px; }
  #lightbox.open { display:flex; }
  #lb-inner { background:var(--page); border-radius:14px; padding:16px 20px;
              max-width:min(1500px, 96vw); width:100%; box-shadow:var(--shadow2); }
  #lb-head { display:flex; justify-content:space-between; align-items:center;
             margin-bottom:12px; }
  #lb-head h3 { margin:0; font-size:15px; }
  #lb-head button { font:inherit; font-size:12px; color:var(--accent); background:none;
                    border:1px solid var(--border); border-radius:999px; padding:3px 12px;
                    cursor:pointer; }
  #lb-head button:hover { background:var(--wash); border-color:var(--accent); }
  #lb-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr));
             gap:14px; }
  #lb-grid video { width:100%; border-radius:9px; background:#000; }
  #lb-grid .vlabel { font-size:12.5px; margin-bottom:5px; display:flex; gap:6px;
                     align-items:center; }

  .emptynote { font-size:12.5px; color:var(--muted); font-style:italic; }
  #nobanner { background:var(--panel); border:1px solid var(--border); border-radius:11px;
              padding:14px 18px; margin-bottom:20px; font-size:13.5px; display:none;
              box-shadow:var(--shadow); }

  @media print {
    :root, :root[data-theme="dark"] {
      color-scheme: light;
      --page:#ffffff; --panel:#ffffff; --border:rgba(0,0,0,.18); --rowhover:transparent;
      --text:#000000; --text2:#333330; --muted:#55534e;
      --grid:#dddcd6; --axis:#999790; --accent:#2159a8; --wash:transparent;
      --shadow:none; --shadow2:none;
      --s0:#2a78d6; --s1:#008300; --s2:#e87ba4; --s3:#eda100;
      --s4:#1baf7a; --s5:#eb6834; --s6:#4a3aa7; --s7:#e34948;
      --rz-fell:#d03b3b; --rz-off:#6c5fc7; --rz-far:#fab219; --rz-done:#d9d8d1;
      --rz-x0:#00868b; --rz-x1:#a1568c; --rz-x2:#7a6a3a;
    }
    #sidebar, #dragbar, .filterrow, #tip, #lightbox, #nobanner { display:none !important; }
    #layout { display:block; }
    #main { padding:0; }
    .panel, table.summary { break-inside:avoid; box-shadow:none; }
    section { margin-bottom:18px; }
  }
</style>
</head>
<body>
<noscript><p style="padding:20px">This report needs JavaScript (all data is embedded, nothing is fetched).</p></noscript>
<div id="layout">
  <nav id="sidebar">
    <h1><span class="brandmark"></span>sim2sim benchmark</h1>
    <button id="themebtn" title="cycle color theme">theme: auto</button>
    <h2>Experiments <span><button id="btn-all">all</button> <button id="btn-none">none</button></span></h2>
    <div id="runboxes"></div>
    <div class="note">1-8 toggle &middot; shift+digit solo &middot; 0 all &middot; click a name to solo</div>
    <h2>Sections</h2>
    <a class="navlink" href="#sec-summary">Summary</a>
    <a class="navlink" href="#sec-signif">Significance</a>
    <a class="navlink" href="#sec-robustness">Robustness</a>
    <a class="navlink" href="#sec-corner">Corner turn</a>
    <a class="navlink" href="#sec-human">Human dribble</a>
    <a class="navlink" href="#sec-uturn">U-turn</a>
    <a class="navlink" href="#sec-speed">Speed</a>
    <a class="navlink" href="#sec-traces">Control traces</a>
    <a class="navlink" href="#sec-videos">Videos</a>
    <h2>Run info</h2>
    <div id="runinfo" class="note"></div>
  </nav>
  <div id="dragbar" role="separator" aria-orientation="vertical" tabindex="0"
       title="drag to resize the sidebar (double-click to reset)"></div>
  <main id="main">
    <header id="pagehead">
      <div>
        <div class="eyebrow">sim2sim benchmark</div>
        <h1>Checkpoint comparison report</h1>
      </div>
      <div class="headmeta" id="headmeta"></div>
    </header>
    <div id="nobanner">No experiments selected &mdash; enable one in the sidebar.</div>
    <div id="cmpbanner"></div>
    <div id="trainbox"></div>

    <section id="sec-summary"><h2><span class="eyebrow">overview</span>Summary</h2>
      <div class="note">Headline numbers per run; <b>best of the selected runs</b> is underlined,
        small &Delta; is vs the first selected run. Values carry &plusmn;1 SE; a &Delta; is
        <b>coloured only when its 95&nbsp;% bootstrap CI excludes zero</b> (shown in
        brackets) &mdash; grey means the gap is inside the noise. Continuous
        metrics are survivors-only, with their own n.
        <b>survival</b> asks only whether the robot stayed upright;
        <b>training-faithful survival</b> also counts a lost ball as an ended
        episode, which is training's own done-set (<code>fall</code> OR
        <code>ball_lost</code> OR <code>time_out</code>), so the gap between the
        two rows is the episodes that stayed up while the ball rolled away. It
        reads &ndash; on runs recorded before the foot-to-ball lost criterion
        existed. Note the two lost-ball thresholds: <b>training-faithful
        survival</b> uses the strict 0.5&nbsp;m foot-surface distance training
        terminates on, while the <b>possession</b> row uses a looser eval
        threshold (0.8&nbsp;m) &mdash; a brief kick past the dribble pocket is
        not yet "lost".
        See <a href="#sec-signif">Significance</a> for every condition.</div>
      <div id="summary-host" style="overflow-x:auto"></div></section>

    <section id="sec-signif"><h2><span class="eyebrow">statistics</span>Significant differences</h2>
      <div class="note">Where the selected runs actually differ. Each cell is one
        condition; it is coloured only when the 95&nbsp;% bootstrap CI on the
        difference clears zero &mdash; at n&nbsp;=&nbsp;48 per condition the noise
        floor alone is ~7&nbsp;points, so an uncoloured cell means "we cannot
        tell these apart", not "they are equal". Episodes are paired on
        (condition,&nbsp;rep) &mdash; the same route in both runs; pairing removes
        route difficulty but <i>not</i> the shared-<code>mjData</code> slot draw,
        so it tightens the intervals only modestly.</div>
      <div class="filterrow" id="signif-controls"></div>
      <div id="signif-host"></div>
    </section>

    <section id="sec-robustness"><h2><span class="eyebrow">robustness</span>Perturbation axes &mdash; nominal human routes (20 s)</h2>
      <div class="note">Each axis perturbs the nominal route bank; dotted line = that run's unperturbed
        baseline. Shaded bands = &plusmn;1 binomial SE. Y scales are shared across axes per metric.</div>
      <div class="legend" id="rob-legend"></div>
      <div class="filterrow" id="rob-filter"></div>
      <div id="rob-host"></div></section>

    <section id="sec-corner"><h2><span class="eyebrow">capability</span>Corner turn</h2>
      <div class="note">150&ndash;180&deg; arc, 12 s, fail-fast; success = finished the turn, no fall,
        ball kept. Turn radius = 1/&kappa;.</div>
      <div class="legend" id="corner-legend"></div>
      <div class="grid" id="corner-grid"></div></section>

    <section id="sec-human"><h2><span class="eyebrow">capability</span>Human dribble</h2>
      <div class="note">Human-route generator with curvature capped at &kappa;<sub>cap</sub> (larger
        = sharper routes), 20 s fail-fast; success = kept control for 20 s.</div>
      <div class="legend" id="human-legend"></div>
      <div class="grid" id="human-grid"></div></section>

    <section id="sec-uturn"><h2><span class="eyebrow">capability</span>U-turn about-face</h2>
      <div class="note">Run-in + 160&ndash;200&deg; turn, radius 1/&kappa;, 10 s, fail-fast.</div>
      <div class="legend" id="uturn-legend"></div>
      <div class="grid" id="uturn-grid"></div></section>

    <section id="sec-speed"><h2><span class="eyebrow">capability</span>Speed</h2>
      <div class="note">Straight-line max speed (10 s, fail-fast) + controllability on human routes
        (trained command distribution; band = &plusmn;1 sd per bin).</div>
      <div class="legend" id="speed-legend"></div>
      <div id="track-badges" class="note"></div>
      <div class="grid" id="speed-grid"></div></section>

    <section id="sec-traces"><h2><span class="eyebrow">diagnostics</span>Control traces &mdash; speed_tracking episodes</h2>
      <div class="note">Ball speed along the command direction (0.5 s smoothed) vs the commanded
        target. &mu; = per-run mean ball speed over the episode.</div>
      <div class="legend" id="traces-legend"></div>
      <div class="grid" id="traces-grid"></div></section>

    <section id="sec-videos"><h2><span class="eyebrow">media</span>Per-condition videos</h2>
      <div class="note">One mp4 per condition (rep-0 episode, chase camera). Click to compare the
        selected runs side by side; middle-click opens the raw file.</div>
      <div id="videos-host"></div></section>
  </main>
</div>

<div id="lightbox">
  <div id="lb-inner">
    <div id="lb-head"><h3 id="lb-title"></h3><button id="lb-close">close (esc)</button></div>
    <div id="lb-grid"></div>
  </div>
</div>
<div id="tip"></div>

<script>
"use strict";
const DATA = __DATA__;
const META = __META__;
const ROB_GROUPS = __ROB_GROUPS__;
const REAL_WORLD = __REAL_WORLD__;
const ROB_METRICS = __ROB_METRICS__;
const CAP_METRICS = __CAP_METRICS__;

const DIRTXT = {up: "↑ better", down: "↓ better", one: "→ 1 ideal",
                zero: "→ 0 ideal"};
const DASH = {r: "6,3", ref: "2,3", cmd: "7,4", base: "1,3"};
// Discovered from the data (see reason_legend) so a new fail_reason string
// cannot silently vanish and leave the stacked bars not summing to 100%.
const REASONS = __REASONS__;
// {"i>j": {metric: [{cond, group, x, delta, lo, hi, sig, paired, n}]}} -- only
// i<j is stored; getDiffs negates for the other direction.
const DIFFS = __DIFFS__;
const DIFF_METRICS = __DIFF_METRICS__;
// axis labels for the difference map: robustness groups come from the data-driven
// ROB_GROUPS, capability groups are named here (they have their own sections)
const DIFF_FULL = __DIFF_FULL__;
// Embedded, not spelled out in the string below: an undefined identifier here
// only blows up on the >MAX_DIFF_RUNS branch, i.e. exactly the case nobody
// renders while developing, and the throw takes every section after
// Significance down with it.
const MAX_DIFF_RUNS = __MAX_DIFF_RUNS__;
const ROB_LABEL = Object.assign(Object.fromEntries(ROB_GROUPS), {
  baseline: "nominal (unperturbed)",
  straight_speed: "straight, commanded speed (m/s)",
  corner_turn: "corner turn |\u03ba| (1/m)",
  u_turn: "u-turn |\u03ba| (1/m)",
  human_dribble: "human dribble \u03ba-cap (1/m)",
  speed_tracking: "speed tracking",
});

function getDiffs(baseIdx, cmpIdx, metric) {
  if (baseIdx === cmpIdx) return null;
  const lo = Math.min(baseIdx, cmpIdx), hi = Math.max(baseIdx, cmpIdx);
  const block = DIFFS[`${lo}>${hi}`];
  if (!block || !block[metric]) return null;
  const flip = baseIdx > cmpIdx;
  return block[metric].map(e => flip
    ? {...e, delta: -e.delta, lo: e.hi == null ? null : -e.hi,
       hi: e.lo == null ? null : -e.lo}
    : e);
}

function nominalDiff(baseIdx, cmpIdx, metric) {
  const all = getDiffs(baseIdx, cmpIdx, metric);
  if (!all) return null;
  return all.find(e => e.cond === "nominal") || null;
}

const state = {
  on: DATA.map(() => true),
  robMetrics: new Set(ROB_METRICS.slice(0, 4).map(m => m[0])),
  robReasons: false,
  turn: {corner: {L: true, R: true}, uturn: {L: true, R: true}},
};
let prevOn = null;

function sv(i) { return `var(--s${i % 8})`; }
function visible() { return DATA.map((r, i) => ({r, i})).filter(o => state.on[o.i]); }

function h(tag, cls, text, parent) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function el(tag, attrs, parent) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function makeSVG(w, hh) {
  const s = el("svg", {viewBox: `0 0 ${w} ${hh}`, width: "100%", class: "chart"});
  return s;
}
function fmtVal(v) {
  if (v == null || !isFinite(v)) return "–";
  const a = Math.abs(v);
  const d = a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : 3;
  return String(+v.toFixed(d));
}
function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) hi = lo + 1;
  const span = hi - lo, step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || mag * 10;
  const t0 = Math.ceil(lo / step) * step, ticks = [];
  for (let t = t0; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(10));
  return {ticks, step};
}
function tickFmt(step) {
  let dec = 0;
  while (dec < 6 && Math.abs(Math.round(step * 10 ** dec) - step * 10 ** dec) > 1e-9) dec++;
  return t => t.toFixed(dec);
}

// ---- tooltip -------------------------------------------------------------
const tip = document.getElementById("tip");
function moveTip(ev) {
  tip.style.left = "0px"; tip.style.top = "0px";
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 12;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 10;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 10;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function keySVG(cvar, dash, parent) {
  const s = el("svg", {viewBox: "0 0 20 8", width: 20, height: 8}, parent);
  el("line", {x1: 1, x2: 19, y1: 4, y2: 4, "stroke-width": 2,
              style: `stroke:${cvar}`, ...(dash ? {"stroke-dasharray": dash} : {})}, s);
}
function hideTip() { tip.style.display = "none"; }

// ---- line chart ----------------------------------------------------------
// series: {x[], y[], se[]?, n[]?, label, cvar, dash?, sw?, runIdx?, ref?}
// opts: {yDomain, xLabel, yLabel, hlines:[{y,cvar,label}], zeroBase(!==false), xFmt,
//        vlines:[{x,band:[lo,hi],title}]}   // x/band may each be null
function lineChart(host, seriesList, opts = {}) {
  const W = 340, H = 210, m = {l: 46, r: 10, t: 8, b: opts.xLabel ? 32 : 20};
  const svg = makeSVG(W, H);
  host.appendChild(svg);
  const data = seriesList.filter(s => !s.ref);
  const flat = [];
  for (const s of data)
    for (let i = 0; i < s.x.length; i++)
      if (s.y[i] != null && isFinite(s.y[i]))
        flat.push([s.x[i], s.y[i], (s.se && s.se[i]) || 0]);
  if (!flat.length) {
    el("text", {x: W / 2, y: H / 2, "text-anchor": "middle"}, svg)
      .textContent = "no data for selected runs";
    return;
  }
  let xlo, xhi;
  if (opts.xDomain) { [xlo, xhi] = opts.xDomain; }
  else { xlo = Math.min(...flat.map(p => p[0])); xhi = Math.max(...flat.map(p => p[0])); }
  if (xlo === xhi) { xlo -= 0.5; xhi += 0.5; }
  let ylo, yhi;
  if (opts.yDomain) { [ylo, yhi] = opts.yDomain; }
  else {
    let lo = Math.min(...flat.map(p => p[1] - p[2]));
    let hi = Math.max(...flat.map(p => p[1] + p[2]));
    for (const hl of opts.hlines || [])
      if (hl.y != null) { lo = Math.min(lo, hl.y); hi = Math.max(hi, hl.y); }
    if (opts.zeroBase !== false) lo = Math.min(0, lo);
    let span = hi - lo;
    if (span <= 0) span = Math.abs(hi) || 1;
    yhi = hi + 0.06 * span;
    ylo = (opts.zeroBase !== false && lo === 0) ? 0 : lo - 0.06 * span;
  }
  const X = v => m.l + (v - xlo) / (xhi - xlo) * (W - m.l - m.r);
  const Y = v => H - m.b - (v - ylo) / (yhi - ylo) * (H - m.t - m.b);
  const yt = niceTicks(ylo, yhi), yf = tickFmt(yt.step);
  for (const t of yt.ticks) {
    el("line", {x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "gridline"}, svg);
    el("text", {x: m.l - 5, y: Y(t) + 3, "text-anchor": "end"}, svg).textContent = yf(t);
  }
  const xt = niceTicks(xlo, xhi), xf = opts.xFmt || tickFmt(xt.step);
  for (const t of xt.ticks)
    el("text", {x: X(t), y: H - m.b + 13, "text-anchor": "middle"}, svg).textContent = xf(t);
  el("line", {x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axisline"}, svg);
  el("line", {x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, class: "axisline"}, svg);
  if (opts.xLabel)
    el("text", {x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle"}, svg)
      .textContent = opts.xLabel;
  if (opts.yLabel)
    el("text", {x: 11, y: (m.t + H - m.b) / 2, "text-anchor": "middle",
                transform: `rotate(-90 11 ${(m.t + H - m.b) / 2})`}, svg)
      .textContent = opts.yLabel;

  const clampY = v => Math.max(ylo, Math.min(yhi, v));
  for (const s of data) {                       // SE / sd bands first
    if (!s.se) continue;
    let seg = [];
    const flush = () => {
      if (seg.length > 1) {
        const up = seg.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(clampY(p[1] + p[2])).toFixed(1)}`).join("");
        const dn = seg.slice().reverse().map(p => `L${X(p[0]).toFixed(1)},${Y(clampY(p[1] - p[2])).toFixed(1)}`).join("");
        el("path", {d: up + dn + "Z", "fill-opacity": 0.13, stroke: "none",
                    style: `fill:${s.cvar}`, "pointer-events": "none",
                    ...(s.runIdx != null ? {"data-run": s.runIdx} : {})}, svg);
      }
      seg = [];
    };
    for (let i = 0; i < s.x.length; i++) {
      if (s.y[i] == null || s.se[i] == null) flush();
      else seg.push([s.x[i], s.y[i], s.se[i]]);
    }
    flush();
  }
  for (const vl of opts.vlines || []) {         // measured REAL hardware value
    const cx = v => Math.max(m.l, Math.min(W - m.r, X(v)));
    if (vl.band) {
      const [a, b] = [cx(vl.band[0]), cx(vl.band[1])];
      if (b > a)
        el("rect", {x: a, y: m.t, width: b - a, height: H - m.b - m.t,
                    style: "fill:var(--text)", "fill-opacity": 0.07, stroke: "none",
                    "pointer-events": "none"}, svg);
    }
    if (vl.x != null && vl.x >= xlo && vl.x <= xhi) {
      el("line", {x1: X(vl.x), x2: X(vl.x), y1: m.t, y2: H - m.b,
                  "stroke-dasharray": "5 3", "stroke-width": 1.4, opacity: 0.7,
                  style: "stroke:var(--text)"}, svg);
      el("text", {x: X(vl.x) + 3, y: m.t + 9, "font-size": 9, opacity: 0.75,
                  style: "fill:var(--text)"}, svg).textContent = "real";
    }
  }
  for (const hl of opts.hlines || []) {         // per-run reference levels
    if (hl.y == null || hl.y < ylo || hl.y > yhi) continue;
    el("line", {x1: m.l, x2: W - m.r, y1: Y(hl.y), y2: Y(hl.y),
                "stroke-dasharray": DASH.base, "stroke-width": 1.4, opacity: 0.65,
                style: `stroke:${hl.cvar}`,
                ...(hl.runIdx != null ? {"data-run": hl.runIdx} : {})}, svg);
  }
  for (const s of seriesList.filter(q => q.ref)) {   // y=x style references
    const d = s.x.map((x, i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(s.y[i]).toFixed(1)}`).join("");
    el("path", {d, fill: "none", "stroke-width": 1.2, "stroke-dasharray": DASH.ref,
                style: "stroke:var(--muted)"}, svg);
  }
  for (const s of data) {
    const sw = s.sw || 2;
    let seg = [];
    const flush = () => {
      if (seg.length) {
        const d = seg.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
        el("path", {d, fill: "none", "stroke-width": sw, "stroke-linejoin": "round",
                    "stroke-linecap": "round", style: `stroke:${s.cvar}`,
                    ...(s.dash ? {"stroke-dasharray": DASH[s.dash] || s.dash} : {}),
                    ...(s.runIdx != null ? {"data-run": s.runIdx} : {})}, svg);
      }
      seg = [];
    };
    for (let i = 0; i < s.x.length; i++) {
      if (s.y[i] == null || !isFinite(s.y[i])) flush();
      else seg.push([s.x[i], s.y[i]]);
    }
    flush();
    if (s.x.length <= 40) {
      for (let i = 0; i < s.x.length; i++) {
        if (s.y[i] == null || !isFinite(s.y[i])) continue;
        el("circle", {cx: X(s.x[i]), cy: Y(s.y[i]), r: 3, "stroke-width": 1.5,
                      style: `fill:${s.cvar};stroke:var(--panel)`,
                      ...(s.runIdx != null ? {"data-run": s.runIdx} : {})}, svg);
      }
    }
  }

  // crosshair + shared tooltip listing every series at the snapped x
  const cross = el("line", {y1: m.t, y2: H - m.b, class: "crossline",
                            style: "display:none"}, svg);
  const xsU = [...new Set(data.flatMap(s => s.x))].sort((a, b) => a - b);
  const maps = data.map(s => {
    const mp = new Map();
    s.x.forEach((x, i) => { if (s.y[i] != null && isFinite(s.y[i])) mp.set(x, i); });
    return mp;
  });
  const hit = el("rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b,
                          fill: "transparent"}, svg);
  hit.addEventListener("pointermove", ev => {
    const box = svg.getBoundingClientRect();
    const px = (ev.clientX - box.left) * (W / box.width);
    let best = xsU[0], bd = Infinity;
    for (const x of xsU) {
      const d = Math.abs(X(x) - px);
      if (d < bd) { bd = d; best = x; }
    }
    cross.setAttribute("x1", X(best)); cross.setAttribute("x2", X(best));
    cross.style.display = "";
    tip.textContent = "";
    h("div", "tip-h", `${opts.xLabel || "x"} = ${fmtVal(best)}`, tip);
    const rows = [];
    data.forEach((s, si) => {
      const i = maps[si].get(best);
      if (i != null) rows.push({s, y: s.y[i], se: s.se ? s.se[i] : null,
                                n: s.n ? s.n[i] : null});
    });
    if (!rows.length) { hideTip(); return; }
    rows.sort((a, b) => b.y - a.y);
    for (const r of rows) {
      const row = h("div", "tip-row", null, tip);
      keySVG(r.s.cvar, r.s.dash ? (DASH[r.s.dash] || r.s.dash) : null, row);
      h("span", "tip-v", fmtVal(r.y) + (r.se != null ? ` ±${fmtVal(r.se)}` : ""), row);
      h("span", "tip-l", r.s.label + (r.n != null ? ` (n=${r.n})` : ""), row);
    }
    tip.style.display = "block";
    moveTip(ev);
  });
  hit.addEventListener("pointerleave", () => { cross.style.display = "none"; hideTip(); });
}

// ---- failure-mode stacked bars --------------------------------------------
// perRun: [{label, runIdx, pts:[{x, reasons, n}]}]
function reasonChart(host, perRun, opts = {}) {
  const W = 340, H = 210, m = {l: 46, r: 10, t: 8, b: opts.xLabel ? 32 : 20};
  const svg = makeSVG(W, H);
  host.appendChild(svg);
  const cats = [...new Set(perRun.flatMap(p => p.pts.map(q => q.x)))].sort((a, b) => a - b);
  if (!cats.length) {
    el("text", {x: W / 2, y: H / 2, "text-anchor": "middle"}, svg)
      .textContent = "no data for selected runs";
    return;
  }
  const plotW = W - m.l - m.r, band = plotW / cats.length;
  const nRuns = perRun.length;
  const barW = Math.min(22, Math.max(3, band * 0.72 / nRuns - 2));
  const Y = v => H - m.b - v / 100 * (H - m.t - m.b);
  for (const t of [0, 25, 50, 75, 100]) {
    el("line", {x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "gridline"}, svg);
    el("text", {x: m.l - 5, y: Y(t) + 3, "text-anchor": "end"}, svg).textContent = t;
  }
  el("line", {x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axisline"}, svg);
  el("line", {x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, class: "axisline"}, svg);
  cats.forEach((c, ci) => {
    el("text", {x: m.l + (ci + 0.5) * band, y: H - m.b + 13, "text-anchor": "middle"}, svg)
      .textContent = fmtVal(c);
  });
  if (opts.xLabel)
    el("text", {x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle"}, svg)
      .textContent = opts.xLabel;
  cats.forEach((c, ci) => {
    const x0 = m.l + (ci + 0.5) * band - (nRuns * (barW + 2) - 2) / 2;
    perRun.forEach((p, pi) => {
      const pt = p.pts.find(q => q.x === c);
      if (!pt || !pt.n) return;
      const bx = x0 + pi * (barW + 2);
      let yCur = 0;
      for (const [key, , cvar] of REASONS) {
        const cnt = pt.reasons[key] || 0;
        if (!cnt) continue;
        const hh = cnt / pt.n * 100;
        const y1 = Y(yCur + hh), y2 = Y(yCur);
        el("rect", {x: bx, y: y1 + 0.5, width: barW, height: Math.max(0.5, y2 - y1 - 1),
                    style: `fill:${cvar}`,
                    ...(p.runIdx != null ? {"data-run": p.runIdx} : {})}, svg);
        yCur += hh;
      }
      el("rect", {x: bx, y: H - m.b + 1.5, width: barW, height: 3,
                  style: `fill:${sv(p.runIdx)}`,
                  ...(p.runIdx != null ? {"data-run": p.runIdx} : {})}, svg);
      const hitr = el("rect", {x: bx - 1, y: m.t, width: barW + 2, height: H - m.t - m.b,
                               fill: "transparent"}, svg);
      hitr.addEventListener("pointermove", ev => {
        tip.textContent = "";
        h("div", "tip-h", `${p.label} — ${opts.xLabel || "x"} = ${fmtVal(c)} (n=${pt.n})`, tip);
        for (const [key, lbl, cvar] of [...REASONS].reverse()) {
          const cnt = pt.reasons[key] || 0;
          if (!cnt) continue;
          const row = h("div", "tip-row", null, tip);
          const sq = h("span", null, null, row);
          sq.style.cssText = `width:10px;height:10px;border-radius:2px;background:${cvar}`;
          h("span", "tip-v", `${Math.round(cnt / pt.n * 100)}%`, row);
          h("span", "tip-l", `${lbl} (${cnt})`, row);
        }
        tip.style.display = "block";
        moveTip(ev);
      });
      hitr.addEventListener("pointerleave", hideTip);
    });
  });
}

// ---- helpers ---------------------------------------------------------------
function panel(host, title, dir) {
  const d = h("div", "panel", null, host);
  const t = h("h3", null, title, d);
  if (dir) h("span", "dir", DIRTXT[dir], t);
  return d;
}
function runSeries(run, i, pts, metric, extra = {}) {
  return {
    x: pts.map(p => p.x), y: pts.map(p => p[metric] == null ? null : p[metric]),
    // band whenever the metric ships an SE, not just for the rate metrics:
    // speed_ratio / cross_track / ach_speed carry a SEM too, and a bare mean
    // line reads as far more certain than n=48 episodes justify
    se: pts.some(p => p[metric + "_se"] != null)
        ? pts.map(p => p[metric + "_se"]) : null,
    // per-metric sample size: cross_track / speed_ratio / ach_speed are
    // survivors-only, so the condition's episode count p.n overstates them
    n: pts.map(p => p[metric + "_n"] != null ? p[metric + "_n"] : p.n),
    label: run.label + (extra.suffix || ""), cvar: sv(i), runIdx: i,
    dash: extra.dash,
  };
}
function legendChips(host, items) {
  host.textContent = "";
  for (const it of items) {
    const c = h("span", "chip", null, host);
    if (it.toggle) {
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.checked = it.toggle.get();
      cb.addEventListener("change", () => it.toggle.set(cb.checked));
      c.appendChild(cb);
    }
    if (it.square) {
      const sq = h("span", null, null, c);
      sq.style.cssText = `width:10px;height:10px;border-radius:2px;background:${it.cvar}`;
    } else {
      keySVG(it.cvar, it.dash, c);
    }
    h("span", null, it.label, c);
    if (it.runIdx != null) {
      c.addEventListener("mouseenter", () => highlightRun(it.runIdx));
      c.addEventListener("mouseleave", () => highlightRun(null));
    }
  }
}
function runChips(extra = []) {
  return visible().map(({r, i}) => ({cvar: sv(i), label: r.label, runIdx: i})).concat(extra);
}
function highlightRun(idx) {
  document.querySelectorAll("#main [data-run]").forEach(e => {
    e.style.opacity = (idx == null || +e.dataset.run === idx) ? "" : "0.12";
  });
}
function reasonPts(pts) {
  return pts.map(p => ({x: p.x, reasons: p.reasons || {}, n: p.n}));
}
function mergeLR(d) {
  const by = new Map();
  for (const side of ["L", "R"]) {
    for (const p of d[side] || []) {
      const cur = by.get(p.x) || {x: p.x, reasons: {}, n: 0};
      cur.n += p.n || 0;
      for (const k in p.reasons || {}) cur.reasons[k] = (cur.reasons[k] || 0) + p.reasons[k];
      by.set(p.x, cur);
    }
  }
  return [...by.values()].sort((a, b) => a.x - b.x);
}

// ---- summary table ---------------------------------------------------------
const SGROUPS = [
  ["run data", [
    ["episodes (rob + cap)", null, r => null, r => `${r.info.n_rob} + ${r.info.n_cap}`],
  ]],
  ["nominal — unperturbed human routes", [
    ["survival (%)", "up", r => r.nominal && r.nominal.survival,
     r => r.nominal ? `${fmtVal(r.nominal.survival)} ±${fmtVal(r.nominal.survival_se)}` : null,
     "survival"],
    // sits directly under `survival` on purpose: the gap between the two rows
    // IS the episodes that kept the robot upright and lost the ball, which
    // training would have ended.
    ["training-faithful survival (%)", "up",
     r => r.nominal && r.nominal.train_survival,
     r => r.nominal && r.nominal.train_survival != null
       ? `${fmtVal(r.nominal.train_survival)} ±${fmtVal(r.nominal.train_survival_se)}`
       : null,
     "train_survival"],
    ["robot-ball dist (m, p90)", "down", r => r.nominal && r.nominal.ball_dist_p90,
     r => r.nominal && r.nominal.ball_dist_p90 != null
       ? `${fmtVal(r.nominal.ball_dist_p90)}` : null],
    ["possession (%)", "up", r => r.nominal && r.nominal.possession,
     r => r.nominal ? `${fmtVal(r.nominal.possession)} ±${fmtVal(r.nominal.possession_se)}` : null],
    ["speed ratio (survivors)", "one", r => r.nominal && r.nominal.speed_ratio,
     r => r.nominal && r.nominal.speed_ratio != null
       ? `${fmtVal(r.nominal.speed_ratio)} ±${fmtVal(r.nominal.speed_ratio_se)}`
         + ` (n=${r.nominal.speed_ratio_n})` : null],
    ["cross-track (m)", "down", r => r.nominal && r.nominal.cross_track,
     r => r.nominal && r.nominal.cross_track != null
       ? `${fmtVal(r.nominal.cross_track)} ±${fmtVal(r.nominal.cross_track_se)}`
         + ` (n=${r.nominal.cross_track_n})` : null,
     "cross_track"],
  ]],
  ["speed & controllability", [
    ["max straight speed @≥50% success (m/s)", "up", r => r.top.max_speed],
    ["controllability pooled r", "up", r => r.pairs && r.pairs.r],
    ["cmd→ball speed slope", "one", r => r.pairs && r.pairs.slope],
    ["tracking mean per-episode r", "up", r => r.top.tracking_r],
    ["tracking slope (actual/cmd)", "one", r => r.top.tracking_slope],
    ["tracking bias (m/s)", "zero", r => r.top.tracking_bias],
    ["tracking residual (m/s)", "down", r => r.top.tracking_resid],
  ]],
  ["turning", [
    ["corner max |κ| left (1/m)", "up", r => r.top.corner_L],
    ["corner max |κ| right (1/m)", "up", r => r.top.corner_R],
    ["u-turn max |κ| left (1/m)", "up", r => r.top.uturn_L],
    ["u-turn max |κ| right (1/m)", "up", r => r.top.uturn_R],
    ["human κ-cap @≥50% success (1/m)", "up", r => r.top.human_cap],
  ]],
];
function betterOf(a, b, dir) {
  if (dir === "one") return Math.abs(a - 1) < Math.abs(b - 1);
  if (dir === "zero") return Math.abs(a) < Math.abs(b);
  return dir === "down" ? a < b : a > b;
}
function renderSummary() {
  const host = document.getElementById("summary-host");
  host.textContent = "";
  const vis = visible();
  if (!vis.length) return;
  const tb = h("table", "summary", null, host);
  const hr = h("tr", null, null, h("thead", null, null, tb));
  h("th", "mname", "metric", hr);
  for (const {r, i} of vis) {
    const th = h("th", null, null, hr);
    const wrap = h("span", "runth", null, th);
    const sw = h("span", "swatch", null, wrap);
    sw.style.background = sv(i);
    h("span", null, r.label, wrap);
  }
  const body = h("tbody", null, null, tb);
  for (const [gLabel, rows] of SGROUPS) {
    const gtr = h("tr", "sgroup", null, body);
    const gtd = h("td", null, gLabel, gtr);
    gtd.colSpan = vis.length + 1;
    for (const [label, dir, get, fmt, dkey] of rows) {
    const tr = h("tr", null, null, body);
    const nm = h("td", "mname", label, tr);
    if (dir) h("span", "dir", DIRTXT[dir], nm);
    const vals = vis.map(({r}) => get(r));
    let bestIdx = -1;
    if (dir && vis.length > 1) {
      vals.forEach((v, k) => {
        if (v == null) return;
        if (bestIdx < 0 || betterOf(v, vals[bestIdx], dir)) bestIdx = k;
      });
      if (bestIdx >= 0 && vals.some((v, j) =>       // no winner on a tie
          j !== bestIdx && v != null &&
          !betterOf(vals[bestIdx], v, dir) && !betterOf(v, vals[bestIdx], dir)))
        bestIdx = -1;
    }
    const ref = vals[0];
    vis.forEach(({r}, k) => {
      const td = h("td", null, null, tr);
      const txt = fmt ? fmt(r) : null;
      h("span", bestIdx === k ? "best" : null,
        txt != null ? txt : fmtVal(vals[k]), td);
      if (dir && k > 0 && vals[k] != null && ref != null) {
        const dv = vals[k] - ref;
        // Colour ONLY when the 95% bootstrap CI on the difference excludes
        // zero. The measured noise floor at n=48 is ~7 points, so painting
        // every non-zero delta red/green (what this did before) reports noise
        // as regression. Metrics with no CI available stay neutral.
        const ci = dkey ? nominalDiff(vis[0].i, vis[k].i, dkey) : null;
        if (Math.abs(dv) < 1e-9 && !ci) h("span", "delta", "±0", td);
        else {
          const sig = ci ? ci.sig : null;
          const cls = sig === true
            ? (betterOf(vals[k], ref, dir) ? "dgood" : "dbad") : "dnull";
          const sp = h("span", "delta " + cls,
                       (dv >= 0 ? "+" : "") + fmtVal(dv), td);
          if (ci) {
            sp.title = `95% bootstrap CI on the difference: `
              + `[${fmtVal(ci.lo)}, ${fmtVal(ci.hi)}]`
              + (ci.paired ? ` — paired on route (n=${ci.n})` : ` — unpaired (n=${ci.n})`)
              + (sig ? "" : " — includes 0, not significant");
            h("span", "ci", ` [${fmtVal(ci.lo)}, ${fmtVal(ci.hi)}]`, td);
          }
        }
      }
    });
    }
  }
}

// ---- significance ------------------------------------------------------------
// Default view is a DIFFERENCE MAP: one row per perturbation group, one cell
// per axis level, coloured only where the 95% CI clears zero. 134 forest rows
// is a wall nobody reads; ~10 rows of coloured cells answers "where does this
// checkpoint differ, and which way" at a glance. Expanding a group reveals the
// per-condition intervals underneath.
// baseline = null -> the first selected run. Made explicit (and pickable)
// because with 3+ checkpoints "whatever happens to be first" is not an answer.
const signifState = {metric: "survival", baseline: null, expanded: new Set(),
                     fold: new Map()};

function betterDir(delta, dir) { return dir === "down" ? delta < 0 : delta > 0; }

function renderSignificance() {
  const controls = document.getElementById("signif-controls");
  const host = document.getElementById("signif-host");
  controls.textContent = ""; host.textContent = "";
  const vis = visible();
  if (vis.length < 2) {
    h("div", "note", "Select at least two experiments to compare.", host);
    return;
  }
  // eligible baselines: every visible run when all pairs were computed,
  // otherwise only run 0 (the only one with precomputed comparisons)
  const eligible = vis.filter(o => DIFF_FULL || o.i === 0);
  let base = eligible.find(o => o.i === signifState.baseline) || eligible[0] || vis[0];
  const others = vis.filter(o => o.i !== base.i);
  const dir = (DIFF_METRICS.find(m => m[0] === signifState.metric) || [])[2] || "up";

  const mrow = h("div", "ctlrow", null, controls);
  h("span", "ctllabel", "metric", mrow);
  for (const [key, label] of DIFF_METRICS) {
    const lab = h("label", "chip" + (signifState.metric === key ? " on" : ""), null, mrow);
    const rb = h("input", null, null, lab);
    rb.type = "radio"; rb.name = "signifmetric";
    rb.checked = signifState.metric === key;
    rb.addEventListener("change", () => {
      signifState.metric = key; signifState.expanded.clear(); renderSignificance();
    });
    h("span", null, label, lab);
  }
  if (vis.length > 2 || eligible.length > 1) {
    const brow = h("div", "ctlrow", null, controls);
    // "compare against" read as "this one is the yardstick, colour the others"
    // -- the exact inverse of what the picker does now.
    h("span", "ctllabel", "subject — green = this run is better", brow);
    for (const o of eligible) {
      const lab = h("label", "chip" + (o.i === base.i ? " on" : ""), null, brow);
      const rb = h("input", null, null, lab);
      rb.type = "radio"; rb.name = "signifbase";
      rb.checked = o.i === base.i;
      rb.addEventListener("change", () => {
        signifState.baseline = o.i; signifState.expanded.clear(); renderSignificance();
      });
      const sw = h("span", "swatch", null, lab);
      sw.style.background = sv(o.i);
      h("span", null, o.r.label, lab);
    }
    if (!DIFF_FULL)
      h("span", "ctlnote", `— only ${vis[0].r.label} can be the subject: with `
        + `more than ${MAX_DIFF_RUNS} runs the report precomputes comparisons `
        + `against the first run only`, brow);
  }

  // With many runs the stacked maps become the same wall this view replaced,
  // so EVERY comparison is collapsible -- the first two used to be rendered
  // bare, which made the two biggest blocks the only ones you could not get out
  // of the way. They still start open so the page is useful on arrival.
  const openUntil = 2;
  others.forEach((other, oi) => {
    // Oriented SUBJECT-first: deltas are base - other, so a green cell means
    // "the run you picked is better". getDiffs negates and swaps the CI ends
    // for us, so every downstream reader (counts, forest, tooltips) follows.
    // The other orientation is the trap this view kept walking into: you pick
    // a checkpoint to look at, and green then meant its RIVAL won.
    const entries = getDiffs(other.i, base.i, signifState.metric);
    const sig0 = (entries || []).filter(e => e.sig);
    const det = h("details", "foldcard", null, host);
    // Remember the user's fold state: every group expand re-renders this whole
    // section, and reading `oi < openUntil` fresh each time would silently
    // re-open a card they had just closed.
    det.open = signifState.fold.has(other.i)
      ? signifState.fold.get(other.i) : oi < openUntil;
    det.addEventListener("toggle", () => signifState.fold.set(other.i, det.open));
    // The summary IS the card title -- a separate heading inside would just
    // repeat it on every open card.
    h("summary", null,
      `${base.r.label} vs ${other.r.label}`
      + (entries && entries.length
         ? ` — ${sig0.length} of ${entries.length} conditions differ`
         : " — no comparable conditions"), det);
    const card = h("div", "card", null, det);
    // NO run-colour swatch here on purpose: the cells below are coloured by
    // BETTER/WORSE, and a run swatch in the same block invites reading a green
    // cell as "this run's colour" instead of "this run is better" -- which is
    // exactly the collision the series palette makes easy (slot 1 IS green).
    if (!entries || !entries.length) {
      h("div", "note", "no comparable conditions for this metric", card);
      return;
    }
    const sig = entries.filter(e => e.sig);
    const better = sig.filter(e => betterDir(e.delta, dir)).length;

    // headline verdict, in words
    const verdict = h("div", "verdict", null, card);
    h("span", "vgood", `${better} better`, verdict);
    h("span", "vsep", "·", verdict);
    h("span", "vbad", `${sig.length - better} worse`, verdict);
    h("span", "vsep", "·", verdict);
    h("span", "vnull", `${entries.length - sig.length} indistinguishable`, verdict);
    h("span", "vnote", ` — of ${entries.length} conditions, at 95 % CI`, verdict);

    // spell out what the two cell colours mean, BY RUN NAME
    const ckey = h("div", "colorkey", null, card);
    const chip = (cls, text) => {
      const w = h("span", "keyitem", null, ckey);
      h("span", "keyswatch " + cls, null, w);
      h("span", null, text, w);
    };
    chip("cgood", `${base.r.label} better than ${other.r.label}`);
    chip("cbad", `${base.r.label} worse`);
    chip("cnull", "cannot tell apart");

    // group rows, most-affected first
    const groups = new Map();
    for (const e of entries) {
      if (!groups.has(e.group)) groups.set(e.group, []);
      groups.get(e.group).push(e);
    }
    const ordered = [...groups.entries()].sort((a, b) =>
      b[1].filter(e => e.sig).length - a[1].filter(e => e.sig).length
      || a[0].localeCompare(b[0]));

    for (const [group, list] of ordered) {
      list.sort((a, b) => a.x - b.x);
      const nsig = list.filter(e => e.sig).length;
      const key = `${other.i}|${group}`;
      const row = h("div", "maprow" + (nsig ? "" : " quiet"), null, card);
      const nameCell = h("div", "mapname", null, row);
      h("span", "mapcaret", signifState.expanded.has(key) ? "▾" : "▸", nameCell);
      h("span", null, ROB_LABEL[group] || group, nameCell);
      const strip = h("div", "mapstrip", null, row);
      for (const e of list) {
        const cell = h("div", "mapcell", null, strip);
        if (e.sig) cell.classList.add(betterDir(e.delta, dir) ? "cgood" : "cbad");
        // saturation carries |delta| relative to the biggest gap in this group
        const peak = Math.max(...list.map(v => Math.abs(v.delta))) || 1;
        if (e.sig) cell.style.opacity = (0.45 + 0.55 * Math.abs(e.delta) / peak).toFixed(2);
        cell.title = `${group} = ${fmtVal(e.x)}\n`
          + `${e.delta >= 0 ? "+" : ""}${fmtVal(e.delta)} `
          + `[${fmtVal(e.lo)}, ${fmtVal(e.hi)}]  n=${e.n}`
          + (e.sig ? "" : "  (includes 0)");
      }
      const tally = h("div", "maptally", nsig ? `${nsig}/${list.length}` : "—", row);
      if (nsig) tally.classList.add("on");
      nameCell.style.cursor = strip.style.cursor = "pointer";
      const toggle = () => {
        if (signifState.expanded.has(key)) signifState.expanded.delete(key);
        else signifState.expanded.add(key);
        renderSignificance();
      };
      nameCell.addEventListener("click", toggle);
      strip.addEventListener("click", toggle);
      if (signifState.expanded.has(key)) drawForest(card, list, other.i, dir);
    }
    h("div", "note", "Each cell is one condition along that axis, left to right. "
      + "Colour only where the 95 % CI clears zero; intensity tracks the size of "
      + "the gap. Click a row for the per-condition intervals.", card);
  });
}

// per-condition intervals for ONE group -- the detail behind a difference-map row
function drawForest(host, list, runIdx, dir) {
  const wrap = h("div", "forestwrap", null, host);
  let span = 0;
  for (const e of list)
    for (const v of [e.delta, e.lo, e.hi])
      if (v != null && isFinite(v)) span = Math.max(span, Math.abs(v));
  span = span || 1;
  const W = 300, PAD = 6;
  const xs = v => PAD + (W - 2 * PAD) * (v + span) / (2 * span);
  for (const e of list) {
    const row = h("div", "forestrow", null, wrap);
    h("div", "fname", e.cond, row);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${W} 13`);
    svg.setAttribute("preserveAspectRatio", "none");
    svg.style.width = "100%"; svg.style.height = "13px";
    const mk = (tag, attrs) => {
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      for (const k in attrs) el.setAttribute(k, attrs[k]);
      svg.appendChild(el); return el;
    };
    mk("line", {x1: xs(0), x2: xs(0), y1: 0, y2: 13,
                stroke: "var(--axis)", "stroke-width": 1});
    const col = e.sig ? (betterDir(e.delta, dir) ? "var(--dgood)" : "var(--dbad)")
                      : "var(--muted)";
    if (e.lo != null && e.hi != null)
      mk("line", {x1: xs(e.lo), x2: xs(e.hi), y1: 6.5, y2: 6.5, stroke: col,
                  "stroke-width": e.sig ? 2.5 : 1.5, opacity: e.sig ? 0.9 : 0.4});
    mk("circle", {cx: xs(e.delta), cy: 6.5, r: 2.8,
                  fill: e.paired ? col : "var(--bg)", stroke: col,
                  "stroke-width": 1.4, opacity: e.sig ? 1 : 0.5});
    row.appendChild(svg);
    const num = h("div", "fnum",
      `${e.delta >= 0 ? "+" : ""}${fmtVal(e.delta)}`
      + (e.lo != null ? ` [${fmtVal(e.lo)}, ${fmtVal(e.hi)}]` : ""), row);
    num.title = `n=${e.n}${e.paired ? " paired on route" : " unpaired"}`;
    if (e.sig) num.style.color = "var(--fg)";
  }
}

// ---- sections ---------------------------------------------------------------
function robDomains(vis) {
  // speed_ratio floors at 0 and reserves room to 1.15, but GROWS past it: a
  // fixed [0, 1.15] silently clipped any run that overshoots the command, which
  // reads as "no data up there" rather than ">1"
  const dom = {survival: [0, 102], possession: [0, 102]};
  let ct = 0, du = 0, sr = 1.15, bd = 0, fb = 0, mzlo = 9, mzhi = 0;
  for (const {r} of vis) {
    const sets = [...ROB_GROUPS.map(([g]) => r.robustness[g] || []),
                  r.nominal ? [r.nominal] : []];
    for (const set of sets)
      for (const p of set) {
        if (p.cross_track != null) ct = Math.max(ct, p.cross_track);
        if (p.mean_duration != null) du = Math.max(du, p.mean_duration);
        if (p.ball_dist_p90 != null) bd = Math.max(bd, p.ball_dist_p90);
        if (p.foot_ball_dist_p90 != null) fb = Math.max(fb, p.foot_ball_dist_p90);
        if (p.min_pelvis_z_p5 != null) {
          mzlo = Math.min(mzlo, p.min_pelvis_z_p5); mzhi = Math.max(mzhi, p.min_pelvis_z_p5);
        }
        // include the SE so a band is never squashed against the ceiling
        if (p.speed_ratio != null)
          sr = Math.max(sr, p.speed_ratio + (p.speed_ratio_se || 0));
      }
  }
  dom.cross_track = [0, ct * 1.08 + 1e-6];
  dom.mean_duration = [0, du * 1.08 + 1e-6];
  dom.ball_dist_p90 = [0, bd * 1.08 + 1e-6];
  dom.foot_ball_dist_p90 = [0, fb * 1.08 + 1e-6];
  // pelvis height sits in a narrow band just above the fall threshold, so a
  // 0-based axis would flatten the whole signal
  dom.min_pelvis_z_p5 = mzhi > 0 ? [Math.max(0, mzlo - 0.03), mzhi + 0.03] : [0, 1];
  dom.speed_ratio = [0, sr * 1.02];
  return dom;
}
function renderRobustness() {
  const host = document.getElementById("rob-host");
  const open = {};
  host.querySelectorAll("details").forEach(d => { open[d.dataset.g] = d.open; });
  host.textContent = "";
  const vis = visible();
  legendChips(document.getElementById("rob-legend"),
              runChips([{cvar: "var(--muted)", dash: DASH.base, label: "nominal baseline"},
                        {cvar: "var(--text)", dash: "5 3",
                         label: "real hardware value (shaded = measured spread)"}]));
  const dom = robDomains(vis);
  for (const [group, gLabel] of ROB_GROUPS) {
    const det = document.createElement("details");
    det.className = "robgroup"; det.dataset.g = group;
    det.open = open[group] !== undefined ? open[group] : true;
    host.appendChild(det);
    const real = REAL_WORLD[group];
    const measured = real && (real.nominal != null || real.band != null);
    const sum = h("summary", null, gLabel, det);
    if (measured) {
      const txt = real.nominal != null
        ? `real ${fmtVal(real.nominal)}` + (real.band ? ` [${fmtVal(real.band[0])}, ${fmtVal(real.band[1])}]` : "")
        : `real [${fmtVal(real.band[0])}, ${fmtVal(real.band[1])}]`;
      const tag = h("span", "realtag", txt, sum);
      tag.title = real.note || "";
    }
    // the marker is the sim2real overlay: where the deployment hardware actually
    // sits on this axis. Unmeasured channels get none (see real_world.py).
    const vlines = measured ? [{x: real.nominal, band: real.band}] : [];
    const row = h("div", "robrow", null, det);
    // full axis range even where a metric has no survivors (null points)
    const gxs = vis.flatMap(({r}) => (r.robustness[group] || []).map(p => p.x));
    const xDomain = gxs.length ? [Math.min(...gxs), Math.max(...gxs)] : null;
    for (const [metric, mLabel, dir] of ROB_METRICS) {
      if (!state.robMetrics.has(metric)) continue;
      const p = panel(row, mLabel, dir);
      const series = [], hlines = [];
      for (const {r, i} of vis) {
        series.push(runSeries(r, i, r.robustness[group] || [], metric));
        if (r.nominal && r.nominal[metric] != null)
          hlines.push({y: r.nominal[metric], cvar: sv(i), runIdx: i});
      }
      lineChart(p, series, {yDomain: dom[metric], xLabel: gLabel, hlines, xDomain, vlines});
    }
    if (state.robReasons) {
      const p = panel(row, "failure modes (share of episodes)");
      reasonChart(p, vis.map(({r, i}) =>
        ({label: r.label, runIdx: i, pts: reasonPts(r.robustness[group] || [])})),
        {xLabel: gLabel});
    }
  }
}
function capDomains(vis) {
  const dom = {success: [0, 102], survival: [0, 102], possession: [0, 102]};
  let ct = 0;
  for (const {r} of vis)
    for (const pts of [r.corner.L, r.corner.R, r.human, r.uturn.L, r.uturn.R])
      for (const p of pts || [])
        if (p.cross_track != null) ct = Math.max(ct, p.cross_track);
  dom.cross_track = [0, ct * 1.08 + 1e-6];
  return dom;
}
function renderTurns(gridId, key, xLabel, dom, legendId) {
  const g = document.getElementById(gridId);
  g.textContent = "";
  const vis = visible();
  const t = state.turn[key];
  const extra = t ? [
    {cvar: "var(--text2)", label: "left (solid)",
     toggle: {get: () => t.L, set: v => { t.L = v; renderTurns(gridId, key, xLabel, dom, legendId); }}},
    {cvar: "var(--text2)", dash: DASH.r, label: "right (dashed)",
     toggle: {get: () => t.R, set: v => { t.R = v; renderTurns(gridId, key, xLabel, dom, legendId); }}},
  ] : [];
  legendChips(document.getElementById(legendId), runChips(extra));
  const gxs = vis.flatMap(({r}) => {
    const d = r[key];
    return Array.isArray(d) ? d.map(p => p.x)
                            : [...(d.L || []), ...(d.R || [])].map(p => p.x);
  });
  const xDomain = gxs.length ? [Math.min(...gxs), Math.max(...gxs)] : null;
  for (const [metric, mLabel, dir] of CAP_METRICS) {
    const p = panel(g, mLabel, dir);
    const series = [];
    for (const {r, i} of vis) {
      const d = r[key];
      if (Array.isArray(d)) {
        series.push(runSeries(r, i, d, metric));
      } else if (d) {
        if (t.L) series.push(runSeries(r, i, d.L, metric, {suffix: " L"}));
        if (t.R) series.push(runSeries(r, i, d.R, metric, {suffix: " R", dash: "r"}));
      }
    }
    lineChart(p, series, {yDomain: dom[metric], xLabel, xDomain});
  }
  const p = panel(g, "failure modes (share of episodes)");
  reasonChart(p, vis.map(({r, i}) => {
    const d = r[key];
    return {label: r.label, runIdx: i,
            pts: Array.isArray(d) ? reasonPts(d) : mergeLR(d)};
  }), {xLabel});
}
function renderSpeed() {
  const g = document.getElementById("speed-grid");
  g.textContent = "";
  const vis = visible();
  legendChips(document.getElementById("speed-legend"),
              runChips([{cvar: "var(--muted)", dash: DASH.ref, label: "achieved = commanded"}]));
  for (const [metric, mLabel, dir] of [["success", "max speed: success rate (%)", "up"],
                                       ["survival", "max speed: survival rate (%)", "up"]]) {
    const p = panel(g, mLabel, dir);
    const series = vis.map(({r, i}) => runSeries(r, i, r.straight, metric));
    lineChart(p, series, {yDomain: [0, 102], xLabel: "commanded speed (m/s), straight"});
  }
  {
    const p = panel(g, "achieved vs commanded: the plateau = measured max", "up");
    const series = vis.map(({r, i}) => runSeries(r, i, r.straight, "ach_speed"));
    const xs = series.flatMap(s => s.x);
    if (xs.length) {
      const lo = Math.min(...xs), hi = Math.max(...xs);
      series.push({x: [lo, hi], y: [lo, hi], ref: true});
    }
    lineChart(p, series, {xLabel: "commanded speed (m/s), straight",
                          yLabel: "achieved ball speed (m/s)", zeroBase: false});
  }
  {
    const p = panel(g, "controllability: binned cmd vs actual (human routes)", "up");
    const series = [];
    for (const {r, i} of vis) {
      if (!r.pairs) continue;
      series.push({x: r.pairs.points.map(q => q.x), y: r.pairs.points.map(q => q.y),
                   se: r.pairs.points.map(q => q.sd), label: r.label,
                   cvar: sv(i), runIdx: i});
    }
    const xs = series.flatMap(s => s.x);
    if (xs.length) {
      const lo = Math.min(...xs), hi = Math.max(...xs);
      series.push({x: [lo, hi], y: [lo, hi], ref: true});
    }
    lineChart(p, series, {xLabel: "commanded speed (m/s), human routes",
                          yLabel: "ball speed along cmd (m/s)", zeroBase: false});
  }
  {
    const p = panel(g, "failure modes: straight max speed");
    reasonChart(p, vis.map(({r, i}) =>
      ({label: r.label, runIdx: i, pts: reasonPts(r.straight)})),
      {xLabel: "commanded speed (m/s)"});
  }
  const b = document.getElementById("track-badges");
  b.textContent = "";
  for (const {r, i} of vis) {
    const chip = h("span", "chip", null, b);
    chip.style.marginRight = "16px";
    const sw = h("span", "swatch", null, chip);
    sw.style.cssText += `;width:10px;height:10px;background:${sv(i)}`;
    const track = r.tracking && r.tracking[0];
    h("span", null,
      `${r.label}: pooled r ${r.pairs ? fmtVal(r.pairs.r) : "–"}` +
      `, slope ${r.pairs ? fmtVal(r.pairs.slope) : "–"}` +
      `${r.pairs ? ` (n=${r.pairs.n})` : ""}` +
      ` · tracking: surv ${track ? fmtVal(track.survival) : "–"}%` +
      `, poss ${track ? fmtVal(track.possession) : "–"}%` +
      `, mean r ${fmtVal(r.top.tracking_r)}`, chip);
  }
}
function renderTraces() {
  const g = document.getElementById("traces-grid");
  g.textContent = "";
  const vis = visible();
  legendChips(document.getElementById("traces-legend"),
              runChips([{cvar: "var(--text2)", dash: DASH.cmd, label: "commanded"}]));
  const keys = [...new Set(vis.flatMap(({r}) => r.traces ? Object.keys(r.traces) : []))]
    .sort((a, b) => +a - +b);
  if (!keys.length) {
    h("div", "emptynote", "no speed traces recorded for the selected runs", g);
    return;
  }
  for (const key of keys) {
    const runsWith = vis.filter(({r}) => r.traces && r.traces[key]);
    if (!runsWith.length) continue;
    const p = panel(g, `episode ${key}`);
    h("div", "panelsub",
      runsWith.map(({r}) => `${r.label} μ=${fmtVal(r.traces[key].mean_act)}`).join(" · ")
      + " m/s", p);
    const longest = runsWith.reduce((a, b) =>
      b.r.traces[key].cmd.length > a.r.traces[key].cmd.length ? b : a);
    const cmdTr = longest.r.traces[key];
    const series = [{
      x: cmdTr.cmd.map((_, i) => +(i * cmdTr.dt).toFixed(2)), y: cmdTr.cmd,
      label: "commanded", cvar: "var(--text2)", dash: "cmd", sw: 1.6,
    }];
    for (const {r, i} of runsWith) {
      const tr = r.traces[key];
      series.push({x: tr.act.map((_, j) => +(j * tr.dt).toFixed(2)), y: tr.act,
                   label: r.label, cvar: sv(i), runIdx: i, sw: 1.6});
    }
    lineChart(p, series, {xLabel: "t (s)", yLabel: "m/s"});
  }
}

// ---- videos -----------------------------------------------------------------
function natCmp(a, b) {
  const split = s => s.split(/(\d+\.?\d*)/).filter(t => t !== "");
  const ka = split(a), kb = split(b);
  for (let i = 0; i < Math.max(ka.length, kb.length); i++) {
    if (ka[i] === undefined) return -1;
    if (kb[i] === undefined) return 1;
    const na = parseFloat(ka[i]), nb = parseFloat(kb[i]);
    if (!isNaN(na) && !isNaN(nb)) { if (na !== nb) return na - nb; }
    else if (ka[i] !== kb[i]) return ka[i] < kb[i] ? -1 : 1;
  }
  return 0;
}
function openLightbox(test, cond) {
  const grid = document.getElementById("lb-grid");
  grid.textContent = "";
  document.getElementById("lb-title").textContent = `${test} / ${cond}`;
  for (const {r, i} of visible()) {
    const v = r.videos[test] && r.videos[test][cond];
    if (!v) continue;
    const card = h("div", null, null, grid);
    const lbl = h("div", "vlabel", null, card);
    const sw = h("span", "swatch", null, lbl);
    sw.style.background = sv(i);
    h("span", null, r.label, lbl);
    const vid = document.createElement("video");
    vid.controls = true; vid.preload = "metadata";
    vid.src = encodeURI(v);
    card.appendChild(vid);
  }
  document.getElementById("lightbox").classList.add("open");
}
function closeLightbox() {
  document.getElementById("lightbox").classList.remove("open");
  document.querySelectorAll("#lb-grid video").forEach(v => v.pause());
}
function renderVideos() {
  const host = document.getElementById("videos-host");
  host.textContent = "";
  const vis = visible();
  const tests = [...new Set(vis.flatMap(({r}) => Object.keys(r.videos)))].sort(natCmp);
  if (!tests.length) {
    h("div", "emptynote",
      "no videos found for the selected runs (record with --videos)", host);
    return;
  }
  // load first frames only when scrolled into view (there can be 100+ mp4s)
  const lazy = new IntersectionObserver(entries => {
    for (const e of entries)
      if (e.isIntersecting) {
        e.target.preload = "metadata";
        e.target.load();
        lazy.unobserve(e.target);
      }
  }, {rootMargin: "300px"});
  for (const test of tests) {
    const runsWith = vis.filter(({r}) => r.videos[test]);
    if (!runsWith.length) continue;
    h("h3", null, test, host);
    const conds = [...new Set(runsWith.flatMap(({r}) => Object.keys(r.videos[test])))]
      .sort(natCmp);
    // split at the last "_": corner_L_0.4 -> corner_L / 0.4, dr_x0.25 -> dr / x0.25;
    // a tail without digits (or no "_") keeps the whole name as its own category
    const cats = new Map();
    for (const c of conds) {
      const idx = c.lastIndexOf("_");
      const tail = idx >= 0 ? c.slice(idx + 1) : "";
      const split = /\d/.test(tail);
      const cat = split ? c.slice(0, idx) : c;
      if (!cats.has(cat)) cats.set(cat, []);
      cats.get(cat).push({cond: c, val: split ? tail : c});
    }
    for (const [cat, items] of cats) {
      for (const {r, i} of runsWith) {
        const has = items.filter(it => r.videos[test][it.cond]);
        if (!has.length) continue;
        const catDiv = h("div", "vcat", null, host);
        const head = h("h4", null, null, catDiv);
        if (runsWith.length > 1) {
          const sw = h("span", "swatch", null, head);
          sw.style.cssText = `width:10px;height:10px;background:${sv(i)}`;
          h("span", null, `${cat} — ${r.label}`, head);
        } else {
          h("span", null, cat, head);
        }
        const strip = h("div", "vstrip", null, catDiv);
        for (const it of has) {
          const tile = h("div", "vtile", null, strip);
          const vid = document.createElement("video");
          vid.preload = "none";
          vid.muted = true;
          vid.playsInline = true;
          vid.src = encodeURI(r.videos[test][it.cond]);
          vid.title = `${it.cond} — click to play/pause`;
          vid.addEventListener("click", () => {
            if (vid.paused) { vid.controls = true; vid.play(); }
            else vid.pause();
          });
          tile.appendChild(vid);
          lazy.observe(vid);
          const cap = h("div", "vcap", null, tile);
          h("span", null, it.val, cap);
          const big = h("button", null, "⛶", cap);
          big.title = "open large / compare runs";
          big.addEventListener("click", () => openLightbox(test, it.cond));
        }
      }
    }
  }
}

// ---- sidebar / state ----------------------------------------------------------
const DEFAULT_RM = ROB_METRICS.slice(0, 4).map(m => m[0]);
function saveHash() {
  const on = DATA.filter((_, i) => state.on[i]).map(r => encodeURIComponent(r.label));
  const parts = [];
  if (on.length < DATA.length) parts.push("on=" + on.join(","));
  const rmDefault = state.robMetrics.size === DEFAULT_RM.length &&
                    DEFAULT_RM.every(k => state.robMetrics.has(k));
  if (!rmDefault)
    parts.push("rm=" + ROB_METRICS.map(m => m[0])
                        .filter(k => state.robMetrics.has(k)).join(","));
  if (state.robReasons) parts.push("fm=1");
  try {
    history.replaceState(null, "", parts.length ? "#" + parts.join("&")
                                                : location.href.split("#")[0]);
  } catch (e) { /* file:// restrictions in some browsers */ }
}
function loadHash() {
  if (!location.hash) return;
  // parse raw: values were encodeURIComponent'd, so split BEFORE decoding
  // (URLSearchParams would decode first and corrupt labels with ',' or '%')
  const q = {};
  for (const part of location.hash.slice(1).split("&")) {
    const eq = part.indexOf("=");
    if (eq > 0) q[part.slice(0, eq)] = part.slice(eq + 1);
  }
  const dec = s => { try { return decodeURIComponent(s); } catch (e) { return s; } };
  if (q.on != null) {
    const labels = new Set(q.on.split(",").map(dec));
    DATA.forEach((r, i) => { state.on[i] = labels.has(r.label); });
    if (!state.on.some(Boolean)) state.on = DATA.map(() => true);
  }
  if (q.rm != null) {
    const keys = new Set(q.rm.split(",").map(dec));
    const valid = ROB_METRICS.map(m => m[0]).filter(k => keys.has(k));
    if (valid.length) state.robMetrics = new Set(valid);
  }
  if (q.fm === "1") state.robReasons = true;
}
function syncBoxes() {
  document.querySelectorAll("#runboxes input").forEach((cb, i) => {
    cb.checked = state.on[i];
  });
}
function solo(i) {
  const soloedMe = state.on[i] && state.on.filter(Boolean).length === 1;
  if (soloedMe && prevOn) {
    state.on = prevOn.slice();
    prevOn = null;
  } else {
    // keep the original multi-selection across solo-to-solo switches
    if (state.on.filter(Boolean).length !== 1) prevOn = state.on.slice();
    state.on = DATA.map((_, j) => j === i);
  }
  syncBoxes();
  renderAll();
}
function buildSidebar() {
  const boxes = document.getElementById("runboxes");
  DATA.forEach((run, i) => {
    const row = h("div", "runrow", null, boxes);
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = state.on[i];
    cb.addEventListener("change", () => { state.on[i] = cb.checked; renderAll(); });
    row.appendChild(cb);
    const sw = h("span", "swatch", null, row);
    sw.style.background = sv(i);
    const name = h("button", "runname", run.label, row);
    name.type = "button";
    name.title = `${run.info.dir}\nclick to solo`;
    name.addEventListener("click", () => solo(i));
    name.addEventListener("focus", () => highlightRun(i));
    name.addEventListener("blur", () => highlightRun(null));
    h("span", "runn", `${run.info.n_rob + run.info.n_cap}`, row);
    row.addEventListener("mouseenter", () => highlightRun(i));
    row.addEventListener("mouseleave", () => highlightRun(null));
  });
  document.getElementById("btn-all").addEventListener("click", () => {
    state.on = DATA.map(() => true); syncBoxes(); renderAll();
  });
  document.getElementById("btn-none").addEventListener("click", () => {
    state.on = DATA.map(() => false); syncBoxes(); renderAll();
  });
  const info = document.getElementById("runinfo");
  DATA.forEach((run, i) => {
    const d = h("div", null, null, info);
    d.style.cssText = "margin-bottom:6px";
    const sw = h("span", null, null, d);
    sw.style.cssText = `display:inline-block;width:9px;height:9px;border-radius:2px;` +
                       `background:${sv(i)};margin-right:5px`;
    h("span", null, `${run.label} — ${run.info.dir}`, d);
    h("div", null, `${run.info.n_rob} rob + ${run.info.n_cap} cap episodes` +
      (run.info.data_time ? `, data ${run.info.data_time}` : ""), d);
  });
  h("div", null, `report generated ${META.generated}`, info).style.marginTop = "8px";
  const cmd = h("div", null, META.cmd, info);
  cmd.style.cssText = "word-break:break-all;opacity:.8";
}
function buildRobFilter() {
  const host = document.getElementById("rob-filter");
  for (const [key, label] of ROB_METRICS) {
    const lb = h("label", null, null, host);
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = state.robMetrics.has(key);
    cb.addEventListener("change", () => {
      cb.checked ? state.robMetrics.add(key) : state.robMetrics.delete(key);
      renderRobustness(); saveHash();
    });
    lb.appendChild(cb);
    h("span", null, label, lb);
  }
  const lb = h("label", null, null, host);
  const cb = document.createElement("input");
  cb.type = "checkbox"; cb.checked = state.robReasons;
  cb.addEventListener("change", () => {
    state.robReasons = cb.checked; renderRobustness(); saveHash();
  });
  lb.appendChild(cb);
  h("span", null, "failure modes", lb);
}
function buildDragbar() {
  const bar = document.getElementById("dragbar");
  const sb = document.getElementById("sidebar");
  const saved = parseInt(localStorage.getItem("s2s-sbw"), 10);
  if (saved >= 200 && saved <= 560) sb.style.width = saved + "px";
  const setW = w => {
    w = Math.max(200, Math.min(560, w));
    sb.style.width = w + "px";
    localStorage.setItem("s2s-sbw", String(w));
  };
  bar.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    bar.setPointerCapture(ev.pointerId);
    bar.classList.add("dragging");
    document.body.classList.add("resizing");
  });
  bar.addEventListener("pointermove", ev => {
    if (bar.classList.contains("dragging")) setW(ev.clientX);
  });
  for (const type of ["pointerup", "pointercancel"])
    bar.addEventListener(type, () => {
      bar.classList.remove("dragging");
      document.body.classList.remove("resizing");
    });
  bar.addEventListener("dblclick", () => {
    sb.style.width = "";
    localStorage.removeItem("s2s-sbw");
  });
  bar.addEventListener("keydown", ev => {
    if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
    ev.preventDefault();
    setW(sb.getBoundingClientRect().width + (ev.key === "ArrowRight" ? 16 : -16));
  });
}
function buildHeader() {
  const hm = document.getElementById("headmeta");
  const eps = DATA.reduce((a, r) => a + r.info.n_rob + r.info.n_cap, 0);
  h("span", "mchip", `${DATA.length} experiment${DATA.length === 1 ? "" : "s"}`, hm);
  h("span", "mchip", `${eps.toLocaleString("en-US")} episodes`, hm);
  h("span", "mchip", `generated ${META.generated}`, hm);
  if (META.live) h("span", "mchip live", "live — refresh re-reads runs", hm);
}
function buildTheme() {
  const btn = document.getElementById("themebtn");
  const seq = ["auto", "light", "dark"];
  let cur = localStorage.getItem("s2s-theme") || "auto";
  const apply = () => {
    if (cur === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = cur;
    btn.textContent = "theme: " + cur;
  };
  btn.addEventListener("click", () => {
    cur = seq[(seq.indexOf(cur) + 1) % seq.length];
    localStorage.setItem("s2s-theme", cur);
    apply();
  });
  apply();
}
function buildScrollSpy() {
  const links = [...document.querySelectorAll(".navlink")];
  const byId = Object.fromEntries(links.map(l => [l.getAttribute("href").slice(1), l]));
  const sections = [...document.querySelectorAll("#main section")];
  const inview = new Set();
  const obs = new IntersectionObserver(entries => {
    for (const e of entries)
      e.isIntersecting ? inview.add(e.target.id) : inview.delete(e.target.id);
    const top = sections.find(s => inview.has(s.id));
    if (top) {
      links.forEach(l => l.classList.remove("active"));
      byId[top.id].classList.add("active");
    }
  }, {rootMargin: "-10% 0px -55% 0px"});
  sections.forEach(s => obs.observe(s));
}
document.addEventListener("keydown", ev => {
  if (ev.key === "Escape") { closeLightbox(); return; }   // works even from <video>
  if (/^(INPUT|TEXTAREA|VIDEO|SELECT|BUTTON)$/.test(ev.target.tagName)) return;
  if (!ev.code || !ev.code.startsWith("Digit")) return;
  const d = +ev.code.slice(5);              // ev.code survives shift ('!' etc.)
  if (d === 0) {
    state.on = DATA.map(() => true); syncBoxes(); renderAll(); return;
  }
  if (d >= 1 && d <= DATA.length) {
    if (ev.shiftKey) solo(d - 1);
    else { state.on[d - 1] = !state.on[d - 1]; syncBoxes(); renderAll(); }
  }
});
document.getElementById("lb-close").addEventListener("click", closeLightbox);
document.getElementById("lightbox").addEventListener("click", ev => {
  if (ev.target.id === "lightbox") closeLightbox();
});

// Comparability: two runs are only directly comparable if their condition
// tables (fingerprint) match. Previously plot.py warned on stdout and the HTML
// report -- the artifact people actually compare in -- said nothing.
function renderComparability() {
  const banner = document.getElementById("cmpbanner");
  const box = document.getElementById("trainbox");
  banner.style.display = "none"; banner.textContent = ""; box.textContent = "";
  const vis = visible();
  if (vis.length < 2) return;

  const bad = [];
  for (const test of ["robustness", "capability"]) {
    const seen = new Map();
    for (const {r} of vis) {
      const fp = ((r.prov || {}).fingerprints || {})[test];
      if (fp) seen.set(fp, [...(seen.get(fp) || []), r.label]);
    }
    if (seen.size > 1) bad.push([test, [...seen.entries()]]);
  }
  if (bad.length) {
    banner.style.display = "block";
    h("b", null, "Condition tables differ. ", banner);
    h("span", null, "These runs were recorded against different condition "
      + "tables, so their curves share an x axis only where values happen to "
      + "coincide and are not paired. Re-run with the same --dr-from to compare "
      + "fairly.", banner);
    for (const [test, groups] of bad) {
      const line = h("div", null, null, banner);
      h("span", null, `${test}: `, line);
      h("span", null, groups.map(([fp, labels]) =>
        `${labels.join(", ")} → ${fp}`).join("   |   "), line);
    }
  }

  // training DR per run -- the "match ITS training params" check
  const rows = [["policy", r => (r.prov.train || {}).onnx],
                ["ball mass DR", r => fmtRange((r.prov.train || {}).ball_mass)],
                ["ball radius DR", r => fmtRange((r.prov.train || {}).ball_radius)],
                ["ball fric DR", r => fmtRange((r.prov.train || {}).ball_friction)],
                ["foot fric DR", r => fmtRange((r.prov.train || {}).foot_friction)],
                ["ball damping c", r => fmtVal((r.prov.train || {}).ball_damping)],
                ["obs lag (steps)", r => fmtRange((r.prov.train || {}).obs_delay)],
                ["act lag (ms)", r => fmtRange((r.prov.train || {}).act_delay)],
                ["push robot dv", r => fmtVal((r.prov.train || {}).push_robot)],
                ["push ball dv", r => fmtVal((r.prov.train || {}).push_ball)]];
  if (!vis.some(({r}) => r.prov && r.prov.train)) return;
  const det = h("details", null, null, box);
  h("summary", null, "training DR each checkpoint was actually trained with", det);
  const tb = h("table", null, null, det);
  const hr = h("tr", null, null, h("thead", null, null, tb));
  h("th", null, "channel", hr);
  for (const {r} of vis) h("th", null, r.label, hr);
  const body = h("tbody", null, null, tb);
  for (const [label, get] of rows) {
    const tr = h("tr", null, null, body);
    h("th", null, label, tr);
    const vals = vis.map(({r}) => get(r));
    const differ = new Set(vals.map(String)).size > 1;
    for (const v of vals) h("td", differ ? "mismatch" : null, v == null ? "–" : v, tr);
  }
}

function fmtRange(pair) {
  if (!pair) return "not randomized";
  return `[${fmtVal(pair[0])}, ${fmtVal(pair[1])}]`;
}

function renderAll() {
  document.getElementById("nobanner").style.display =
    state.on.some(Boolean) ? "none" : "block";
  const dom = capDomains(visible());
  renderComparability();
  renderSummary();
  renderSignificance();
  renderRobustness();
  renderTurns("corner-grid", "corner", "|κ| (1/m)", dom, "corner-legend");
  renderTurns("human-grid", "human", "κ-cap (1/m)", dom, "human-legend");
  renderTurns("uturn-grid", "uturn", "|κ| (1/m)", dom, "uturn-legend");
  renderSpeed();
  renderTraces();
  renderVideos();
  saveHash();
}

loadHash();
buildDragbar();
buildHeader();
buildSidebar();
buildRobFilter();
buildTheme();
buildScrollSpy();
renderAll();
</script>
</body>
</html>
"""


def js_embed(obj):
    """JSON for inline <script> embedding: no bare NaN, no '</script>' escape."""
    return json.dumps(obj, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")


def resolve_runs(args, quiet=False):
    if args.run_dirs is not None:
        labels = args.labels or [os.path.basename(os.path.normpath(d))
                                 for d in args.run_dirs]
        if len(args.run_dirs) != len(labels):
            raise RuntimeError("--labels must match --run-dirs")
        return args.run_dirs, labels
    run_dirs = sorted(
        os.path.join(args.runs_root, d) for d in os.listdir(args.runs_root)
        if os.path.exists(os.path.join(args.runs_root, d, "robustness.csv"))
        or os.path.exists(os.path.join(args.runs_root, d, "capability.csv")))
    if not run_dirs:
        raise RuntimeError(f"no runs with CSVs found under {args.runs_root}")
    if not quiet:
        print(f"[html_report] discovered {len(run_dirs)} runs under {args.runs_root}")
    return run_dirs, [os.path.basename(os.path.normpath(d)) for d in run_dirs]


def generate(args, live=False, quiet=False):
    """Aggregate the CSVs of every (re-)discovered run into the report HTML."""
    run_dirs, labels = resolve_runs(args, quiet=quiet)
    report_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(report_dir, exist_ok=True)
    # parse once, then take the UNION of robustness groups across runs so a group
    # only one run swept still gets a panel (the old fixed ROB_GROUPS list both
    # hid new groups and rendered empty ones for groups nobody swept)
    parsed = [(read_rows(os.path.join(d, "robustness.csv")),
               read_rows(os.path.join(d, "capability.csv"))) for d in run_dirs]
    rob_groups = robustness_groups(*[r for pair in parsed for r in pair])
    runs = [collect_run(d, lab, i, report_dir, rob_groups, rows)
            for i, (d, lab, rows) in enumerate(zip(run_dirs, labels, parsed))]
    title = "sim2sim benchmark — " + (
        ", ".join(labels) if len(labels) <= 5 else f"{len(labels)} runs")
    meta = dict(
        generated=datetime.datetime.now().isoformat(timespec="minutes", sep=" "),
        live=live,
        cmd="python -m sim2sim_benchmark.html_report "
            + " ".join(shlex.quote(a) for a in sys.argv[1:]))
    payload = {
        "__TITLE__": html_lib.escape(title),
        "__ROB_GROUPS__": js_embed(rob_groups),
        "__REASONS__": js_embed(reason_legend(runs)),
        "__DIFFS__": js_embed(condition_diffs(parsed, labels)),
        "__DIFF_METRICS__": js_embed([(k, lab, d) for k, lab, _, _, _, d in DIFF_METRICS]),
        "__DIFF_FULL__": js_embed(len(run_dirs) <= MAX_DIFF_RUNS),
        "__MAX_DIFF_RUNS__": js_embed(MAX_DIFF_RUNS),
        "__REAL_WORLD__": js_embed(REAL_WORLD),
        "__ROB_METRICS__": js_embed(ROB_METRICS),
        "__CAP_METRICS__": js_embed(CAP_METRICS),
        "__META__": js_embed(meta),
        "__DATA__": js_embed(runs),
    }
    # single pass so payload content can never corrupt a later substitution
    return re.sub("|".join(map(re.escape, payload)),
                  lambda mo: payload[mo.group(0)], HTML_TEMPLATE)


def serve(args):
    """Live mode: every page refresh re-discovers runs and re-aggregates the
    CSVs server-side. Static assets (videos, CSVs) come straight from disk,
    so the report's relative video links keep working."""
    import functools
    import http.server

    root = os.getcwd()
    out_abs = os.path.abspath(args.out)
    rel = os.path.relpath(out_abs, root)
    report_url = None if rel.startswith("..") else "/" + rel.replace(os.sep, "/")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            try:
                self._get()
            except (BrokenPipeError, ConnectionResetError):
                pass    # client hung up early — normal for video previews

        def _get(self):
            path = self.path.split("?", 1)[0]
            if path == "/" and report_url:
                self.send_response(302)
                self.send_header("Location", report_url)
                self.end_headers()
                return
            if path == report_url or (path == "/" and not report_url):
                try:
                    html = generate(args, live=True, quiet=True)
                except Exception as exc:
                    self.send_error(500, f"report generation failed: {exc}")
                    return
                try:
                    with open(out_abs, "w") as f:   # keep the snapshot fresh too
                        f.write(html)
                except OSError:
                    pass
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.headers.get("Range"):
                self._serve_range(self.headers["Range"])
                return
            super().do_GET()

        def _serve_range(self, range_header):
            """Minimal byte-range support so <video> seeking works (the stdlib
            handler always sends whole files and cannot resume)."""
            try:
                f = open(self.translate_path(self.path), "rb")
            except OSError:
                self.send_error(404)
                return
            with f:
                size = os.fstat(f.fileno()).st_size
                m = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
                if not m or (not m.group(1) and not m.group(2)):
                    self.send_error(416)
                    return
                if not m.group(1):                     # suffix form: last N bytes
                    start = max(0, size - int(m.group(2)))
                    end = size - 1
                else:
                    start = int(m.group(1))
                    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                self.send_response(206)
                self.send_header("Content-Type", self.guess_type(self.translate_path(self.path)))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def log_message(self, fmt, *fargs):
            if self.path.split("?", 1)[0] in ("/", report_url):
                sys.stderr.write(f"[html_report] {fmt % fargs}\n")

    srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", args.port), functools.partial(Handler, directory=root))
    url = f"http://127.0.0.1:{args.port}" + (report_url or "/")
    print(f"[html_report] live report at {url} — every refresh re-reads "
          f"{args.runs_root if args.run_dirs is None else 'the given run dirs'}; "
          f"Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[html_report] stopped")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dirs", nargs="+", default=None,
                    help="experiment dirs, each holding robustness.csv / capability.csv "
                         "(+ optional *_speed_pairs/_traces CSVs and videos/). Omit to "
                         "auto-discover every run under --runs-root (tensorboard-style: "
                         "include everything, choose what to view in the browser)")
    ap.add_argument("--runs-root", default="sim2sim_eval_results/runs",
                    help="scanned when --run-dirs is omitted")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="one per run dir; defaults to the dir basenames")
    ap.add_argument("--out", default="sim2sim_eval_results/compare/report.html")
    ap.add_argument("--serve", action="store_true",
                    help="serve the report over localhost instead of only writing the "
                         "file: every browser refresh re-discovers runs and re-reads "
                         "the CSVs (still also refreshes the --out snapshot)")
    ap.add_argument("--port", type=int, default=8000, help="port for --serve")
    args = ap.parse_args()
    if args.serve:
        serve(args)
        return
    try:
        html = generate(args)
    except RuntimeError as e:
        ap.error(str(e))
    with open(args.out, "w") as f:
        f.write(html)
    print(f"[html_report] wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
