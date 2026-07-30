"""Self-contained interactive HTML comparison report (tensorboard-style).

  python -m sim2sim_benchmark.html_report \
      --run-dirs sim2sim_eval_results/runs/m80000 sim2sim_eval_results/runs/m90000 \
      --labels iter80000 iter90000 \
      --out sim2sim_eval_results/compare/report.html

  python -m sim2sim_benchmark.html_report --serve      # live mode: refresh (F5)
      # re-discovers runs and rebuilds whatever changed since the last load
      # (unchanged -> served from memory; see the "aggregation cache" section)

One HTML file, no external assets: experiment checkboxes in the sidebar select
which runs are drawn. A topline summary table leads, then a DIFFERENCE MAP that
answers "where do these runs actually differ, and is it real" -- every condition
carries a 95 % bootstrap CI on the difference (paired on route via
(condition, rep)), and nothing is coloured unless that interval clears zero.
Resolution scales as 1/sqrt(n): a single condition (n~48) needs a ~20-point gap
to clear 95 %, one axis pooled (~480) ~6 points, the whole table (~3500) ~2
points. So an uncoloured single-condition cell means "cannot tell apart" at that
scale, NOT "equal" -- read the axis trend or raise --reps.

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
import hashlib
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
               ("possession", "ball possession (%, survivors)", "up"),
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
    "joint_friction": "leg+waist joint friction (N*m)",
    "dr_scale": "DR scale alpha",   # legacy CSVs only
}
# groups that belong to the capability sections, never the robustness grid
CAP_GROUPS = {"baseline", "straight_speed", "corner_turn", "u_turn",
              "human_dribble", "speed_tracking"}
CAP_METRICS = [("success", "strict completion success (%)", "up"),
               ("success_route", "route-control success (%)", "up"),
               ("success_possession", "upright + ball at termination (%)", "up"),
               ("cross_track_success", "cross-track (m, strict successes)", "down"),
               ("survival", "survival rate (%)", "up"),
               ("progress", "progress before termination (m)", "up"),
               ("ball_dist_p90", "robot-ball distance (m, p90)", "down"),
               ("cross_track", "cross-track (m, upright; fail-fast censored)", "down")]


# fail_reason -> (legend label, CSS colour var). "timeout"/"completed"/
# "incomplete" are the three readings of an empty fail_reason (see
# condition_stats): ran the clock out with no fail-fast, finished cleanly, or
# never finished the route geometry (scored success=0, so NOT a success colour).
REASON_STYLE = {
    "completed": ("completed", "var(--rz-done)"),
    "timeout": ("ran full clock", "var(--rz-done)"),
    "incomplete": ("route unfinished", "var(--rz-x0)"),
    "ball_far": ("ball lost", "var(--rz-far)"),
    "off_route": ("off route", "var(--rz-off)"),
    "fell": ("fell", "var(--rz-fell)"),
}
REASON_ORDER = ["completed", "timeout", "incomplete", "ball_far", "off_route", "fell"]
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
# statistic, is_rate, direction). "down" metrics are handled in the JS by flipping the
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
    ("success", "strict completion success (%)", lambda r: r["success"],
     stats.rate_stat, True, "up"),
    # the two looser readings of the same episode (engine.episode_metrics):
    # possession >= route >= strict `success`. None on pre-2026-07-22 runs, which
    # drops them from the pair rather than comparing against a missing column.
    ("success_possession", "possession success (%)", lambda r: r["success_possession"],
     stats.rate_stat, True, "up"),
    ("success_route", "route-adherence success (%)", lambda r: r["success_route"],
     stats.rate_stat, True, "up"),
    ("cross_track_success", "cross-track (m, strict successes)",
     lambda r: r["ct"] if r["success"] is not None and r["success"] > 0.5 else None,
     stats.mean_stat, False, "down"),
    ("cross_track", "cross-track (m)", lambda r: r["ct"] if r["fell"] < 0.5 else None,
     stats.mean_stat, False, "down"),
    ("foot_ball_dist", "foot-ball distance (m)", lambda r: r["foot_ball_dist"],
     stats.mean_stat, False, "down"),
    ("ball_dist", "robot-ball distance (m)", lambda r: r["ball_dist"],
     stats.mean_stat, False, "down"),
    ("progress", "progress (m)", lambda r: r["progress"], stats.mean_stat, False, "up"),
    # plastic_turf's headline: every episode is MEANT to end in a failure, so the
    # question is how long the policy lasted, not whether it survived a fixed
    # budget. Meaningful on the other tables too, just less interesting there --
    # their budgets are short enough that most episodes hit the cap.
    ("duration", "task survival (s)", lambda r: r["duration"],
     stats.mean_stat, False, "up"),
]
# Any run can be the significance subject, at any run count. The report used to
# precompute {every pair} x {every metric} x {every condition} and cap the run
# list at 8 to keep that affordable -- 55 pairs at 11 runs is already ~2.5 min,
# and it grows as n^2 (1225 pairs at 50 checkpoints, ~53 min and ~200 MB).
#
# Nothing needs that cross product. You look at ONE (subject, metric) at a time,
# which is n-1 blocks, and the two views want very different slices:
#   FULL_SCOPE     every condition of one metric   ~200 bootstraps  (~0.26 s)
#   NOMINAL_SCOPE  one condition of every metric   ~1 bootstrap     (~1.4 ms)
# So blocks are computed per (pair, metric, scope) on demand and cached forever
# (see AggCache). Cost tracks what you actually open, not n^2. What ships inline
# is the bounded O(n) slice that makes the page useful on arrival -- and is the
# only thing an offline snapshot can show, since file:// cannot fetch.
# Metrics whose FULL condition sweep ships inline. Two, not one: `survival` is the
# arrival metric for the perturbation axes and `duration` is the field trial's
# headline, and an offline snapshot that cannot fetch has to answer both. Each
# costs (n-1) blocks of ~16 kB, so this stays O(n).
INLINE_DIFF_METRICS = ("survival", "duration")
NOMINAL_COND = "nominal"
FULL_SCOPE, NOMINAL_SCOPE = "full", "nom"
DIFF_ENDPOINT = "/_s2s/diff"        # --serve only; None on a static snapshot

# Every per-episode table a run dir can hold, in the order collect_run and
# pair_diffs walk them. plastic_turf joined 2026-07-29; a run predating it just
# yields [] for that slot, which every consumer already handles.
EPISODE_TABLES = ("robustness.csv", "capability.csv", "plastic_turf.csv")
TURF_GROUP = "plastic_turf"
# (condition name, what it is). No comparability flag: there is one point, and the
# thresholds are stated authoritatively by the parameter panel, not by a chip that
# has to be kept in sync by hand.
TURF_POINTS = [
    ("turf_harsh", "the field recipe — EDU torque ceiling, ±7° frame error, "
                   "heavy pile, trained-envelope pushes"),
]
# Dropped 2026-07-29 along with their episodes (turf_mild = probe round g,
# turf_max = round d). Named here only so a stale CSV cannot quietly resurrect
# them through turf_series' unknown-condition fallback.
TURF_RETIRED = ("turf_mild", "turf_max")

# What the field-trial parameter panel shows, grouped the way you reason about the
# deployment: (category, [(condition key, label, unit)]). Read from the run's own
# <title>.conditions.json, never re-derived from today's code -- a run's record of
# what it tested has to survive the code moving on. `dr.*` and `run.*` reach into
# the nested dr dict and the run-level block.
TURF_PARAM_GROUPS = [
    ("route & task", [
        ("route_mode", "route generator", ""),
        ("route_vmax", "commanded speed cap", "m/s"),
        ("route_v2_vel", "v2 training speed chain", ""),
        ("route_len_m", "route length", "m"),
        ("episode_s", "episode budget", "s"),
    ]),
    ("termination", [
        ("ball_far_fail_m", "ball lost beyond", "m"),
        ("offroute_fail_m", "off route beyond", "m"),
    ]),
    ("ball", [
        ("dr.mass", "mass", "kg"),
        ("dr.radius", "radius", "m"),
        ("dr.ball", "surface friction", ""),
        ("ball_damping", "roll brake c", ""),
        ("ball_roll_fric", "roll friction", ""),
        ("ball_bounce_dampratio", "bounce damp ratio", ""),
    ]),
    ("ground & turf", [
        ("dr.foot", "foot friction", ""),
        ("ground_solref_tc", "contact time constant", "s"),
        ("pile_drag", "pile drag", "N·s/m"),
        ("pile_height", "pile height", "m"),
    ]),
    ("actuation", [
        ("motor_curve", "torque-speed ceiling", ""),
        ("motor_peak_scale", "peak torque scale (EDU)", "×URDF"),
        ("motor_vel_scale", "zero-torque speed scale", "×URDF"),
        ("joint_friction", "joint friction", "N·m"),
    ]),
    ("observation", [
        ("obs_noise_scale", "noise", "×trained"),
        ("ball_obs_bias", "ball position bias", "m"),
        ("obs_frame_rpy_bias_deg", "frame bias roll/pitch/yaw", "deg"),
        ("ball_obs_delay_steps", "ball obs lag", "steps @50 Hz"),
        ("action_delay_ms", "action lag", "ms"),
    ]),
    ("disturbance", [
        ("push_dv", "base push", "×trained envelope"),
        ("ball_push_dv", "ball push", "m/s"),
        ("push_interval_s", "push interval", "s"),
        ("tether_back_n", "tether backward", "N"),
        ("tether_down_n", "tether downward", "N"),
    ]),
    ("episode start", [
        ("reset_ball_dist", "ball distance", "m"),
        ("reset_ball_bearing", "ball bearing", "deg"),
        ("reset_jitter", "pose jitter", ""),
        ("standby_hold_s", "standby hold before hand-off", "s"),
        ("run.settle_range", "policy takeover window", "s"),
    ]),
    ("run level (not condition keys)", [
        ("run.hybridfoot", "hybrid foot collision", ""),
        ("run.robots", "parallel robots", ""),
        ("run.route_bank", "distinct routes", ""),
        ("run.reps", "reps per route", ""),
    ]),
]
# The conditions the SUMMARY table quotes a CI for, i.e. what NOMINAL_SCOPE
# computes. One bootstrap each, ~1.4 ms -- cheap enough to cover the field-trial
# rows too, and without them those rows could only show a bare delta.
HEADLINE_CONDS = (NOMINAL_COND,) + tuple(p[0] for p in TURF_POINTS)
assert not set(TURF_RETIRED) & set(HEADLINE_CONDS)


def _diff_rows(rows, extract):
    """Episode rows reduced to {pair, value} for one diff metric."""
    out = []
    for r in rows:
        value = extract(r)
        out.append(dict(pair=stats.pair_key(r),
                        v=float("nan") if value is None else float(value)))
    return out


def _by_condition(rows):
    out = {}
    for r in rows:
        out.setdefault(r["condition"], []).append(r)
    return out


def pair_diffs(rows_a, rows_b, metrics=None, conditions=None):
    """Bootstrap CIs on the shared conditions of ONE pair of runs.

    Returns {metric: [{cond, group, x, delta, lo, hi, sig, paired, n}]}, with
    delta oriented as b - a. The (rob, cap) rows are grouped by condition ONCE
    and reused across metrics -- regrouping per metric re-walked ~10k rows ten
    times per pair for nothing.

    `metrics` and `conditions` restrict the work, and the difference between the
    two views is 200x: the summary table wants ONE condition across every metric
    (~10 bootstraps), the difference map wants every condition of ONE metric
    (~200). Computing the whole cross product for every pair up front is what
    forced the old run-count cap."""
    wanted = None if metrics is None else set(metrics)
    per_metric = {}
    tables = [(_by_condition(a), _by_condition(b))
              for a, b in zip(rows_a, rows_b)]      # robustness, capability
    for key, _label, extract, stat_fn, is_rate, _dir in DIFF_METRICS:
        if wanted is not None and key not in wanted:
            continue
        entries = []
        for by_cond_a, by_cond_b in tables:
            shared = set(by_cond_a) & set(by_cond_b)
            if conditions is not None:
                shared &= set(conditions)
            for cond in sorted(shared):
                a = _diff_rows(by_cond_a[cond], extract)
                b = _diff_rows(by_cond_b[cond], extract)
                if not any(np.isfinite(r["v"]) for r in a):
                    continue              # metric undefined for this condition
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
    return per_metric


def diff_block(get_rows, cache, i, j, metric, scope):
    """One (pair, metric, scope) block of CIs, oriented delta = hi - lo.

    Always keyed on the ordered pair, so both reading directions share one
    computation and the JS negates for the reverse -- see getDiffs().

    This is THE view the report was missing: the summary table used to colour
    any non-zero delta green/red, while a single condition (n~48) needs a
    ~20-point gap to clear 95% (per-condition SE ~7 pts). Here a delta is only
    marked significant when the 95% bootstrap CI on the difference excludes zero.
    """
    lo, hi = (i, j) if i < j else (j, i)
    entries = cache.get_block(lo, hi, metric, scope) if cache else None
    if entries is None and scope == NOMINAL_SCOPE and cache:
        # the full sweep already contains the headline rows; never re-bootstrap them
        full = cache.get_block(lo, hi, metric, FULL_SCOPE)
        if full is not None:
            entries = [e for e in full if e["cond"] in HEADLINE_CONDS]
    if entries is not None:
        if cache:
            cache.hits += 1
        return entries
    entries = pair_diffs(
        get_rows(lo), get_rows(hi), [metric],
        set(HEADLINE_CONDS) if scope == NOMINAL_SCOPE else None).get(metric, [])
    if cache:
        cache.misses += 1
        cache.put_block(lo, hi, metric, scope, entries)
    return entries


def diff_rows_for(get_rows, cache, pairs, metrics, scope):
    """[{metric: entries}] for a batch of index pairs -- one request's worth."""
    return [{m: diff_block(get_rows, cache, i, j, m, scope) for m in metrics}
            for i, j in pairs]


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


def order_rob_groups(present):
    """Group names -> ordered (group, label) pairs.

    Known groups keep their curated order and label; anything else is appended
    alphabetically under its raw name. Groups with no episodes are dropped, so
    the report no longer renders empty panels for channels this checkpoint never
    swept (ball_damping / dr_scale on the current runs)."""
    present = set(present) - CAP_GROUPS - {""}
    known = [g for g in ROB_GROUP_LABEL if g in present]
    unknown = sorted(present - set(known))
    return [(g, ROB_GROUP_LABEL.get(g, g)) for g in known + unknown]


def robustness_groups(*row_lists):
    """Ordered (group, label) pairs actually present in `row_lists`."""
    return order_rob_groups(r["group"] for rows in row_lists for r in rows)


def _f(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def nested_strict_success(raw_success, route_success, completed):
    """Repair/derive strict success from the two constraints it must contain.

    Protocol-2 CSVs already carry ``success_route`` and ``completed``, but their
    raw ``success`` column was computed independently and can therefore exceed
    route success. ``completed is None`` means the route has no finite arc and
    reaching the full episode budget is the completion criterion.
    """
    if route_success is None:
        return raw_success
    return 1.0 if (route_success > 0.5 and completed != 0.0) else 0.0


def read_rows(path):
    if not os.path.exists(path):
        return []
    out, bad, repaired = [], 0, 0
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
            raw_success = _f(r.get("success"))
            completed = _f(r.get("completed"))
            route_success = _f(r.get("success_route"))
            strict_success = nested_strict_success(raw_success, route_success, completed)
            if (raw_success is not None and strict_success is not None
                    and raw_success != strict_success):
                repaired += 1
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
                success=strict_success,
                # `completed` distinguishes "ran clean to the end of the budget"
                # from "never finished the arc"; the three nested verdicts and the
                # raw failure quantities are absent on pre-2026-07-22 runs, so
                # every one of these is a .get that may stay None
                completed=completed,
                success_possession=_f(r.get("success_possession")),
                success_route=route_success,
                max_ct=_f(r.get("max_cross_track_m")),
                terminal_ct=_f(r.get("terminal_cross_track_m")),
                off_route_t=_f(r.get("off_route_t_s")),
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
    if repaired:
        print(f"[html_report] {path}: repaired strict-success nesting in "
              f"{repaired} legacy rows from success_route + completed")
    return out


def finite(values):
    return [v for v in values if v is not None and np.isfinite(v)]


def condition_stats(rows, fail_fast=None):
    """rows of one condition -> point stats used by every panel.

    `fail_fast` (None = infer) decides how an empty fail_reason is labelled.
    engine.episode_metrics writes `success` ONLY when a fail-fast criterion is
    armed, so a finite success column is an exact marker for it -- no group-name
    list to keep in sync.

    Most continuous metrics are SURVIVORS-ONLY (`alive`). An episode truncated by a
    fall covers its route distance in less wall time, so an unfiltered
    ach_speed / speed_ratio mean RISES as the condition gets harder -- it moves
    opposite to what it claims to measure. cross-track was already filtered this
    way; ach/ratio now match it, and every panel reports the sample it used.
    Capability additionally reports ``cross_track_success`` on strict successes
    only. This is the CT companion to strict success rate; the survivor CT is
    retained and explicitly labelled as fail-fast-censored diagnostic data.

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
    # possession at the MAIN (eval) threshold, SURVIVORS-ONLY: P(kept ball |
    # stayed up). Conditioning on survival is deliberate -- an all-rows
    # possession counts a fall where the ball happened to stop near the feet (or
    # a fall before first-touch, so the lost-flag never armed) as "kept", i.e. a
    # FALLEN robot "possessing", which is nonsense. Survivors-only drops the
    # fallen episodes entirely. This is a CONDITIONAL rate on a different
    # denominator than survival, so like survivors-only cross-track it is NOT
    # meant to be compared to survival numerically -- it can sit well above it
    # (a policy whose only failure mode is falling keeps the ball ~100% of the
    # time it stays up). The combined "upright AND kept" number that IS <=
    # survival is train_survival below (fall OR lost, at the 0.5 m flag).
    poss_p = (1.0 - float(np.mean([r["ball_lost"] for r in alive]))
              if alive else None)
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

    def rate(key):
        """Mean of one of the looser success verdicts, None when the run predates
        the column (so the report shows a gap, not a fake 0%)."""
        vals = finite([r.get(key) for r in rows])
        if not vals:
            return None, None
        p = float(np.mean(vals))
        return round(100.0 * p, 2), rate_se(p, len(vals))

    poss_succ, poss_succ_se = rate("success_possession")
    route_succ, route_succ_se = rate("success_route")
    ratios = [r["ach"] / r["cmd"] for r in alive
              if r["ach"] is not None and r["cmd"] is not None and r["cmd"] > 0.05]
    ct_vals = finite([r["ct"] for r in alive])
    strict_success_rows = [r for r in rows
                           if r["success"] is not None and r["success"] > 0.5]
    ct_success_vals = finite([r["ct"] for r in strict_success_rows])
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
        #
        # `completed == 0` is the third case and used to be swallowed by the
        # first: an arc route the robot never finished. engine.episode_metrics
        # scores those success=0, so painting them "completed" put failures in
        # the success colour (55 episodes on human_dr_m80000).
        key = r["reason"] or ("completed" if fail_fast else "timeout")
        if not r["reason"] and fail_fast and r.get("completed") == 0.0:
            key = "incomplete"
        reasons[key] = reasons.get(key, 0) + 1
    return dict(
        n=n, n_alive=len(alive),
        survival=round(100.0 * surv_p, 2), survival_se=rate_se(surv_p, n),
        train_survival=(None if train_surv_p is None
                        else round(100.0 * train_surv_p, 2)),
        train_survival_se=(None if train_surv_p is None
                           else rate_se(train_surv_p, n)),
        possession=(None if poss_p is None else round(100.0 * poss_p, 2)),
        possession_se=(None if poss_p is None else rate_se(poss_p, len(alive))),
        success=None if succ_p is None else round(100.0 * succ_p, 2),
        success_se=None if succ_p is None else rate_se(succ_p, len(succ_vals)),
        success_possession=poss_succ, success_possession_se=poss_succ_se,
        success_route=route_succ, success_route_se=route_succ_se,
        speed_ratio=round(float(np.mean(ratios)), 4) if ratios else None,
        speed_ratio_se=mean_se(ratios),
        speed_ratio_n=len(ratios),
        cross_track=round(float(np.mean(ct_vals)), 4) if ct_vals else None,
        cross_track_se=mean_se(ct_vals),
        cross_track_n=len(ct_vals),      # survivors only -- smaller than n
        cross_track_success=(round(float(np.mean(ct_success_vals)), 4)
                             if ct_success_vals else None),
        cross_track_success_se=mean_se(ct_success_vals),
        cross_track_success_n=len(ct_success_vals),
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
        # time-to-failure is right-skewed (one lucky episode drags the mean), so
        # the field-trial view reads the median beside it
        duration_p50=round(float(np.median(dur_vals)), 2) if dur_vals else None,
        duration_n=len(dur_vals),
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
        """Largest x whose strict success rate still clears the threshold.

        This used to stop at the FIRST point below the threshold and report the
        top of the initial contiguous run, which is only the same number when the
        curve degrades monotonically -- and real curves do not. On
        human_dr_m80000 the straight-speed sweep reads 1.0 -> 40%, 1.25 -> 54%,
        1.5 -> 58%, so the old rule broke at 1.0 and reported "no passing point"
        for a policy that clears 50% at 1.5 m/s.

        `contiguous` (the old number) is kept alongside: where the two disagree
        the curve is non-monotonic, which is itself worth showing rather than
        silently picking one.
        """
        best = contiguous = None
        broken = False
        for p in series:                     # sorted by x
            if p.get("success") is None:
                continue
            if p["success"] >= threshold:
                best = p["x"]
                if not broken:
                    contiguous = p["x"]
            else:
                broken = True
        return best, contiguous

    track = [r for r in cap_rows if r["group"] == "speed_tracking"]
    tr = finite([r["r"] for r in track])

    def avg(key, digits=3):
        vals = finite([r.get(key) for r in track])
        return round(float(np.mean(vals)), digits) if vals else None

    top = {}
    for key, series in (("max_speed", straight), ("corner_L", corner["L"]),
                        ("corner_R", corner["R"]), ("uturn_L", uturn["L"]),
                        ("uturn_R", uturn["R"]), ("human_cap", human)):
        best, contiguous = passing(series)
        top[key] = best
        top[f"{key}_contiguous"] = contiguous
        if contiguous != best:
            # the curve dips below the threshold and comes back: the headline
            # number is the largest passing point, but the policy is NOT reliable
            # everywhere below it, and that is worth saying out loud
            print(f"[html_report] {key}: non-monotonic sweep — clears the threshold "
                  f"up to {best}, but not continuously from the start "
                  f"(contiguous limit {contiguous})")
    return dict(
        tracking_slope=avg("slope"), tracking_bias=avg("bias"),
        tracking_resid=avg("resid"), **top,
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


def turf_series(turf_rows):
    """plastic_turf rows -> one entry per severity, in severity order.

    Not group_series: the points are NAMED (their axis index is just an ordering)
    and the page has to show which of them share a measurement scale, so the name
    rides along with the stats."""
    by_name = _by_condition(turf_rows)
    out = []
    for name, label in TURF_POINTS:
        rows = by_name.get(name)
        if not rows:
            continue
        out.append(dict(name=name, label=label,
                        x=float(rows[0]["axis"]), **condition_stats(rows)))
    known = {p[0] for p in TURF_POINTS} | set(TURF_RETIRED)
    for name in sorted(set(by_name) - known):
        rows = by_name[name]
        out.append(dict(name=name, label=name,
                        x=float(rows[0]["axis"]), **condition_stats(rows)))
    return out


def turf_params(run_dir):
    """The plastic_turf condition values THIS run recorded, or None.

    Reads the run's own dump rather than re-deriving from conditions.py: the point
    of the panel is what was tested, and code moves on."""
    path = os.path.join(run_dir, "plastic_turf.conditions.json")
    try:
        with open(path) as f:
            blob = json.load(f)
    except (OSError, ValueError):
        return None
    live = {p[0] for p in TURF_POINTS}
    cond = next((c for c in blob.get("conditions") or [] if c.get("name") in live), None)
    if cond is None:
        return None
    out = {}
    for _group, items in TURF_PARAM_GROUPS:
        for key, _label, _unit in items:
            if key.startswith("run."):
                out[key] = (blob.get("run_level") or {}).get(key[4:])
            elif key.startswith("dr."):
                out[key] = (cond.get("dr") or {}).get(key[3:])
            else:
                out[key] = cond.get(key)
    return out


def collect_run(run_dir, label, index, report_dir, rob_groups=None, rows=None):
    """One run -> the JSON blob the page draws. `rob_groups` is the union of
    robustness groups across ALL selected runs (so a group only one run has still
    gets a panel); `rows` reuses an already-parsed (rob, cap, turf) triple."""
    rob, cap, turf = rows if rows is not None else tuple(
        read_rows(os.path.join(run_dir, f)) for f in EPISODE_TABLES)
    if rob_groups is None:
        rob_groups = robustness_groups(rob, cap)
    nominal = [r for r in rob if r["group"] == "baseline"]
    corner = group_series(cap, "corner_turn", split_sign=True)
    human = group_series(cap, "human_dribble")
    uturn = group_series(cap, "u_turn", split_sign=True)
    straight = group_series(cap, "straight_speed")
    csv_paths = [os.path.join(run_dir, f) for f in
                 EPISODE_TABLES + ("capability_speed_pairs.csv",
                                   "capability_speed_traces.csv")]
    mtimes = [os.path.getmtime(p) for p in csv_paths if os.path.exists(p)]
    pairs = binned_pairs(os.path.join(run_dir, "capability_speed_pairs.csv"))
    return dict(
        label=label, color=index % 8,
        prov=run_provenance(run_dir),
        info=dict(dir=os.path.relpath(run_dir), n_rob=len(rob), n_cap=len(cap),
                  n_turf=len(turf),
                  data_time=datetime.datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d %H:%M")
                  if mtimes else None),
        nominal=condition_stats(nominal) if nominal else None,
        turf=turf_series(turf), turf_params=turf_params(run_dir),
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
  /* field trial: a ranked bar per run. A table beats a grouped bar chart here --
     the interesting read is the ORDER over checkpoints, and it stays legible at
     any run count instead of squeezing n bars into one axis. */
  .turfcard { margin:10px 0 16px; }
  .turfhead { display:flex; gap:8px; align-items:baseline; margin:0 0 3px;
              font-size:13.5px; font-weight:600; flex-wrap:wrap; }
  .turfhead .turfscale { font-weight:400; font-size:11.5px; color:var(--muted); }
  .turfrow { display:grid; grid-template-columns:250px 1fr 132px 116px 42px;
             gap:10px; align-items:center; padding:2px 4px; border-radius:5px;
             font-size:12px; }
  .turfrow:hover { background:var(--wash); }
  .turfname { display:flex; gap:6px; align-items:center; white-space:nowrap;
              overflow:hidden; text-overflow:ellipsis; }
  .turfbarwrap { position:relative; height:16px; border-radius:3px;
                 background:color-mix(in srgb, var(--muted) 12%, transparent); }
  .turfbar { position:absolute; left:0; top:0; bottom:0; border-radius:3px; }
  /* the whisker has to read ON TOP of a saturated bar, which a thin dark line
     does not -- the SE is the whole point here (whether the field recipe can
     separate checkpoints at all), so it gets a light rule with end caps */
  .turferr { position:absolute; top:50%; height:2px; transform:translateY(-50%);
             background:var(--panel); box-shadow:0 0 0 .5px rgba(0,0,0,.35); }
  .turferr::before, .turferr::after { content:""; position:absolute; top:-3px;
             width:2px; height:8px; background:var(--panel);
             box-shadow:0 0 0 .5px rgba(0,0,0,.35); }
  .turferr::before { left:0; }
  .turferr::after { right:0; }
  .turfval { text-align:right; font-variant-numeric:tabular-nums; }
  .turfval .turfsd { color:var(--muted); }
  /* how the episodes ENDED, inline on the row. reasonChart is for a swept axis;
     with one axis value per card it drew a full-size plot per severity and
     buried the ranking it was meant to annotate. */
  .turfmix { display:flex; height:11px; border-radius:2px; overflow:hidden;
             background:color-mix(in srgb, var(--muted) 12%, transparent); }
  .turfmix > span { display:block; }
  .turfn { text-align:right; color:var(--muted); font-size:11px; }
  .turfempty { color:var(--muted); font-size:12px; padding:4px; }
  /* the parameter panel sits BELOW the ranking: you come here for the numbers,
     then ask what produced them */
  .turfparams { margin-top:14px; }
  .turfparams > summary { cursor:pointer; font-size:12.5px; color:var(--text2);
                          padding:4px 2px; }
  .turfparams > summary:hover { color:var(--fg); }
  .turfpgrid { display:grid; grid-template-columns:repeat(auto-fit, minmax(310px, 1fr));
               gap:4px 26px; margin-top:6px; }
  .turfpgroup { break-inside:avoid; }
  .turfpgroup h5 { margin:8px 0 3px; font-size:11px; letter-spacing:.06em;
                   text-transform:uppercase; color:var(--accent); font-weight:650; }
  .turfprow { display:grid; grid-template-columns:1fr auto; gap:12px;
              font-size:12px; padding:1.5px 0; align-items:baseline; }
  .turfprow .turfpk { color:var(--text2); }
  .turfprow .turfpv { font-variant-numeric:tabular-nums; text-align:right;
                      white-space:nowrap; }
  .turfprow .turfpu { color:var(--muted); font-size:11px; }
  .turfprow.differ .turfpv { color:var(--dbad); font-weight:650; }
  .turfprow.unset { opacity:.45; }
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
    <a class="navlink" href="#sec-turf">Field trial</a>
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

    <section id="sec-turf"><h2><span class="eyebrow">deployment</span>Field trial &mdash; the joint real-world distribution</h2>
      <div class="note">The one table where every channel is off-nominal AT ONCE,
        because that is what the field is: turf underfoot, an imperfect ball, a
        mocap frame a few degrees out, firmware torque limits, a safety rope and a
        hand-off from standby. Every other section sweeps ONE channel off a clean
        base and answers "what breaks it"; this one answers <b>"how long does it
        last out there"</b>.
        Termination is TASK-level &mdash; fall, ball lost, or off route &mdash; so a
        policy cannot score by abandoning the ball and staying upright. The
        headline is therefore <b>mean task-survival seconds</b>, not a survival
        rate at a fixed budget: every episode is meant to end in a failure, and
        the question is how long it took.
        <b>Push magnitude is pinned to the trained envelope</b>
        (<code>push_dv</code>&nbsp;0.5, i.e. &plusmn;0.5&nbsp;m/s and
        &plusmn;0.78&nbsp;rad/s of yaw, exactly the <code>env.yaml</code> range).
        The recipe first shoved at 4&times; that; a one-channel-at-a-time ablation
        (<code>turf_harsh_ablation_20260729</code>) showed it alone drove upright
        survival from 94&nbsp;% to 25&nbsp;%, made 75&nbsp;% of episodes end in a
        fall, held strict success at 0&nbsp;% everywhere, and squeezed the spread
        over ten checkpoints to 0.65&nbsp;s. Neither the &plusmn;7&deg; frame error
        nor the EDU torque ceiling moved falls at all.
        Episodes end at a 1.5&nbsp;m lost-ball distance or a 600&nbsp;s budget the
        ceiling never reaches, and the measured ball is pinned
        (0.39&ndash;0.40&nbsp;kg, 0.095&ndash;0.105&nbsp;m) so every checkpoint
        meets the same one &mdash; only its friction follows each checkpoint's own
        training DR.
        <b>Foot geometry differs from the rest of the report</b>: this table runs
        on the <code>--hybridfoot</code> deploy geometry (capsules carry the floor
        contact, the ankle_roll STL carries the ball), because that is what the
        hardware has, while the robustness and capability sections above were
        recorded on the 7-capsule feet. A capsule-trained policy meeting mesh feet
        here is a sim2real finding, not a misconfiguration.</div>
      <div class="legend" id="turf-legend"></div>
      <div id="turf-host"></div></section>

    <section id="sec-signif"><h2><span class="eyebrow">statistics</span>Significant differences</h2>
      <div class="note">Where the selected runs actually differ. Each cell is one
        condition; it is coloured only when the 95&nbsp;% bootstrap CI on the
        difference clears zero. Read resolution by the SCALE you are looking at
        (rate noise scales as 1/&radic;n, so it shrinks as you pool):
        <b>one condition</b> (n&nbsp;&asymp;&nbsp;48) &mdash; a gap must exceed
        <b>~20&nbsp;points</b> to clear 95&nbsp;% (per-condition SE ~7&nbsp;pts,
        the difference of two ~10&nbsp;pts);
        <b>one axis</b> (~10 levels pooled, n&nbsp;&asymp;&nbsp;480) &mdash;
        <b>~6&nbsp;points</b>, and a monotone trend is itself signal without any
        single level clearing;
        <b>the whole table</b> (n&nbsp;&asymp;&nbsp;3500) &mdash;
        <b>~2&nbsp;points</b> (measured &plusmn;1.9 paired).
        So an uncoloured single-condition cell means "we cannot tell these apart",
        not "they are equal" &mdash; look at the axis trend, or raise
        <code>--reps</code>. Episodes are paired on (condition,&nbsp;rep) &mdash;
        the same route in both runs; pairing removes route difficulty but
        <i>not</i> the shared-<code>mjData</code> slot draw, so it tightens the
        per-condition intervals only modestly.</div>
      <div class="filterrow" id="signif-controls"></div>
      <div id="signif-host"></div>
    </section>

    <section id="sec-robustness"><h2><span class="eyebrow">robustness</span>Perturbation axes &mdash; nominal human routes</h2>
      <div class="note">Each axis perturbs the nominal route bank; dotted line = that run's unperturbed
        baseline. Shaded bands = &plusmn;1 binomial SE. Y scales are shared across axes per metric.</div>
      <div class="legend" id="rob-legend"></div>
      <div class="filterrow" id="rob-filter"></div>
      <div id="rob-host"></div></section>

    <section id="sec-corner"><h2><span class="eyebrow">capability</span>Corner turn</h2>
      <div class="note">150&ndash;180&deg; arc, fail-fast. Strict success = route-control success
        plus finishing the turn; turn radius = 1/&kappa;. Cross-track is shown both on strict
        successes and, separately, on all upright but potentially early-truncated episodes.</div>
      <div class="legend" id="corner-legend"></div>
      <div class="grid" id="corner-grid"></div></section>

    <section id="sec-human"><h2><span class="eyebrow">capability</span>Human dribble</h2>
      <div class="note">Human-route generator with curvature capped at &kappa;<sub>cap</sub> (larger
        = sharper routes), fail-fast for the configured episode budget. Three verdicts expose
        upright+ball at termination, route control, and strict full-budget success separately.</div>
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
// {"i>j|scope": {metric: [{cond, group, x, delta, lo, hi, sig, paired, n}]}}.
// Only i<j is stored; getDiffs negates for the other direction. What ships here
// is the bounded arrival slice (pairs against run 0) -- everything else is
// fetched from DIFF_ENDPOINT, see DiffStore.
const DIFFS = __DIFFS__;
const DIFF_METRICS = __DIFF_METRICS__;
// null when the report was built as a static file. The protocol check matters
// too: --serve also writes the snapshot to --out, so that file carries an
// endpoint it cannot reach once you open it from disk.
// which run the inline slice is anchored on: the most complete one, so the
// arrival view is not blank for a table the alphabetically-first run happens to
// be missing. Also the default significance subject, for the same reason.
const INLINE_ANCHOR = __INLINE_ANCHOR__;
const DIFF_ENDPOINT = __DIFF_ENDPOINT__;
const CAN_FETCH = !!DIFF_ENDPOINT
  && !(typeof location !== "undefined" && location.protocol === "file:");
const NOMINAL_COND = __NOMINAL_COND__;
const [FULL_SCOPE, NOM_SCOPE] = __SCOPES__;
const TURF_PARAM_GROUPS = __TURF_PARAM_GROUPS__;
// axis labels for the difference map: robustness groups come from the data-driven
// ROB_GROUPS, capability groups are named here (they have their own sections)
const ROB_LABEL = Object.assign(Object.fromEntries(ROB_GROUPS), {
  baseline: "nominal (unperturbed)",
  plastic_turf: "field trial — deployment hypothesis",
  straight_speed: "straight, commanded speed (m/s)",
  corner_turn: "corner turn |\u03ba| (1/m)",
  u_turn: "u-turn |\u03ba| (1/m)",
  human_dribble: "human dribble \u03ba-cap (1/m)",
  speed_tracking: "speed tracking",
});

// ---- difference blocks ------------------------------------------------------
// Every block is one (ordered pair, metric, scope) and is fetched at most once.
// LOADED holds what we have, PENDING de-dupes concurrent asks, and a block that
// comes back unavailable is stored as null so a re-render cannot loop on it.
const LOADED = new Map(), PENDING = new Map();
function blockKey(lo, hi, metric, scope) { return `${lo}>${hi}|${scope}|${metric}`; }
for (const [pk, byMetric] of Object.entries(DIFFS)) {
  const [pair, scope] = pk.split("|");
  const [lo, hi] = pair.split(">");
  for (const [metric, entries] of Object.entries(byMetric))
    LOADED.set(blockKey(+lo, +hi, metric, scope), entries);
}

function negate(e) {
  return {...e, delta: -e.delta, lo: e.hi == null ? null : -e.hi,
          hi: e.lo == null ? null : -e.lo};
}

// entries oriented delta = cmp - base, or undefined when not loaded yet
function peekDiffs(baseIdx, cmpIdx, metric, scope) {
  if (baseIdx === cmpIdx) return null;
  const lo = Math.min(baseIdx, cmpIdx), hi = Math.max(baseIdx, cmpIdx);
  let entries = LOADED.get(blockKey(lo, hi, metric, scope));
  if (entries === undefined && scope === NOM_SCOPE) {
    const full = LOADED.get(blockKey(lo, hi, metric, FULL_SCOPE));
    if (full) entries = full.filter(e => e.cond === NOMINAL_COND);
  }
  if (!entries) return entries === undefined ? undefined : null;
  return baseIdx > cmpIdx ? entries.map(negate) : entries;
}

function getDiffs(baseIdx, cmpIdx, metric) {
  return peekDiffs(baseIdx, cmpIdx, metric, FULL_SCOPE) || null;
}

// CI for ONE named condition. The summary table quotes the unperturbed `nominal`
// row for most metrics and a turf severity for the field-trial rows, so the
// condition has to be a parameter -- hardcoding NOMINAL_COND here quietly
// attached the clean-route interval to the field-trial delta.
function condDiff(baseIdx, cmpIdx, metric, cond) {
  const all = peekDiffs(baseIdx, cmpIdx, metric, NOM_SCOPE);
  return all ? all.find(e => e.cond === (cond || NOMINAL_COND)) || null : null;
}

function turfPoint(run, name) {
  return (run.turf || []).find(p => p.name === name) || null;
}

// One round trip for a whole row: subject vs every other run currently on
// screen. Returns a promise that settles once every requested block is either
// loaded or marked unavailable; resolves to false when there was nothing to do.
function loadDiffs(pairs, metrics, scope) {
  const want = [], keys = [];
  for (const [a, b] of pairs) {
    const lo = Math.min(a, b), hi = Math.max(a, b);
    for (const metric of metrics) {
      const k = blockKey(lo, hi, metric, scope);
      if (LOADED.has(k) || PENDING.has(k) || keys.includes(k)) continue;
      if (scope === NOM_SCOPE && LOADED.has(blockKey(lo, hi, metric, FULL_SCOPE)))
        continue;                       // derivable from the full sweep
      keys.push(k);
      if (!want.some(p => p[0] === lo && p[1] === hi)) want.push([lo, hi]);
    }
  }
  if (!keys.length) return Promise.resolve(false);
  if (!CAN_FETCH) {                     // offline snapshot: nothing to fetch from
    keys.forEach(k => LOADED.set(k, null));
    return Promise.resolve(true);
  }
  const req = fetch(DIFF_ENDPOINT, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      pairs: want.map(([lo, hi]) => [DATA[lo].label, DATA[hi].label]),
      metrics, scope}),
  }).then(r => r.ok ? r.json() : Promise.reject(new Error(r.status)))
    .then(({blocks}) => {
      want.forEach(([lo, hi], k) => {
        const byMetric = blocks[k] || {};
        for (const metric of metrics)
          LOADED.set(blockKey(lo, hi, metric, scope), byMetric[metric] || null);
      });
    })
    .catch(() => { keys.forEach(k => { if (!LOADED.has(k)) LOADED.set(k, null); }); })
    .then(() => { keys.forEach(k => PENDING.delete(k)); return true; });
  keys.forEach(k => PENDING.set(k, req));
  return req;
}

// Sections render from whatever is loaded and re-render once the rest lands.
// Tokens are PER SECTION: a shared counter would let the summary's batch cancel
// the significance redraw, which is the bug where half the page stays "loading".
const diffTokens = {};
function afterDiffs(who, pairs, metrics, scope, redraw) {
  const mine = diffTokens[who] = (diffTokens[who] || 0) + 1;
  loadDiffs(pairs, metrics, scope).then(changed => {
    if (changed && mine === diffTokens[who]) redraw();
  });
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
    // speed_ratio / cross-track variants / ach_speed carry a SEM too, and a
    // bare mean line reads as far more certain than n=48 episodes justify
    se: pts.some(p => p[metric + "_se"] != null)
        ? pts.map(p => p[metric + "_se"]) : null,
    // per-metric sample size: cross-track variants / speed_ratio / ach_speed use
    // conditioned subsets, so the condition's episode count p.n overstates them
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
    ["episodes (rob + cap + turf)", null, r => null,
     r => `${r.info.n_rob} + ${r.info.n_cap}`
          + (r.info.n_turf ? ` + ${r.info.n_turf}` : " + —")],
  ]],
  // The field trial LEADS the table on purpose: it is the one number that answers
  // "how does this checkpoint do in the real environment". Under the clean-route
  // rows it read as an afterthought.
  ["field trial — joint real-world distribution", [
    ["task survival (s)", "up",
     r => { const p = turfPoint(r, "turf_harsh"); return p && p.mean_duration; },
     r => { const p = turfPoint(r, "turf_harsh");
            return p && p.mean_duration != null
              ? `${fmtVal(p.mean_duration)} ±${fmtVal(p.mean_duration_se)}`
                + (p.duration_p50 != null ? ` [${fmtVal(p.duration_p50)}]` : "")
              : null; },
     "duration", "turf_harsh"],
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
    ["possession (%, survivors)", "up", r => r.nominal && r.nominal.possession,
     r => r.nominal && r.nominal.possession != null
       ? `${fmtVal(r.nominal.possession)} ±${fmtVal(r.nominal.possession_se)}` : null],
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
    ["max straight speed @≥50% strict success (m/s)", "up", r => r.top.max_speed],
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
    ["human κ-cap @≥50% strict success (1/m)", "up", r => r.top.human_cap],
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
  // the Δ column needs one condition (nominal) of every metric against the
  // first selected run -- n-1 blocks at ~1.4 ms each, so the whole row is one
  // cheap batch. Cells render neutral until it lands.
  if (vis.length > 1)
    afterDiffs("summary", vis.slice(1).map(o => [vis[0].i, o.i]),
               DIFF_METRICS.map(m => m[0]), NOM_SCOPE, renderSummary);
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
    for (const [label, dir, get, fmt, dkey, dcond] of rows) {
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
        // zero. A single condition (n~48) needs a ~20-point gap to clear 95%,
        // so painting every non-zero delta red/green (what this did before)
        // reports noise as regression. Metrics with no CI available stay neutral.
        const ci = dkey ? condDiff(vis[0].i, vis[k].i, dkey, dcond) : null;
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
const signifState = {metric: "survival", baseline: INLINE_ANCHOR,
                     expanded: new Set(), fold: new Map()};

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
  // ANY visible run can be the subject at any run count: the blocks this view
  // needs are the n-1 pairs against `base` for ONE metric, fetched on demand and
  // cached server-side forever. The old build precomputed every pair x every
  // metric up front, which is n^2 and is why the picker used to be pinned to the
  // first run past 8 checkpoints.
  let base = vis.find(o => o.i === signifState.baseline) || vis[0];
  const others = vis.filter(o => o.i !== base.i);
  const dir = (DIFF_METRICS.find(m => m[0] === signifState.metric) || [])[2] || "up";
  afterDiffs("signif", others.map(o => [base.i, o.i]), [signifState.metric],
             FULL_SCOPE, renderSignificance);

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
  if (vis.length > 1) {       // at two runs it still says WHICH ONE green means
    const brow = h("div", "ctlrow", null, controls);
    // "compare against" read as "this one is the yardstick, colour the others"
    // -- the exact inverse of what the picker does now.
    h("span", "ctllabel", "subject — green = this run is better", brow);
    for (const o of vis) {
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
    // three states, and they must read differently: still computing, computed
    // and empty, or unreachable. Collapsing "loading" into "no data" is how a
    // 20 s cold bootstrap looks like a broken report.
    const entries = peekDiffs(other.i, base.i, signifState.metric, FULL_SCOPE);
    const loading = entries === undefined;
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
      + (loading ? " — computing…"
         : entries && entries.length
         ? ` — ${sig0.length} of ${entries.length} conditions differ`
         : " — no comparable conditions"), det);
    const card = h("div", "card", null, det);
    // NO run-colour swatch here on purpose: the cells below are coloured by
    // BETTER/WORSE, and a run swatch in the same block invites reading a green
    // cell as "this run's colour" instead of "this run is better" -- which is
    // exactly the collision the series palette makes easy (slot 1 IS green).
    if (loading) {
      h("div", "note", `bootstrapping ${signifState.metric} for this pair — `
        + "about a second, then it is cached for good", card);
      return;
    }
    if (entries === null) {
      h("div", "note", CAN_FETCH
        ? "this comparison could not be computed — see the server log"
        : "this comparison is not in the offline snapshot: it ships only the "
          + `${DATA[0].label} row. Run --serve to pick any subject or metric.`,
        card);
      return;
    }
    if (!entries.length) {
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
// ---- field trial -------------------------------------------------------------
// One card per condition, runs ranked by mean task-survival. Bars normalise
// WITHIN a card so a second point could never be compared to this one by length.
function renderTurf() {
  const host = document.getElementById("turf-host");
  host.textContent = "";
  const vis = visible();
  // the reason colours, NOT the run palette: the run colour is already on every
  // row's swatch, and the only unexplained colours in this section are the
  // outcome-mix segments
  legendChips(document.getElementById("turf-legend"),
              REASONS.map(([, label, cvar]) => ({cvar, label}))
                     .concat([{cvar: "var(--fg)", label: "whisker = ±1 SE"}]));
  const withTurf = vis.filter(({r}) => (r.turf || []).length);
  if (!withTurf.length) {
    h("div", "turfempty", vis.length
      ? "none of the selected runs has plastic_turf.csv — run the benchmark with "
        + "--plastic-turf to fill it in"
      : "select at least one experiment", host);
    return;
  }
  // order comes from turf_series, which emits TURF_POINTS order -- the live point
  // first, retired ones after. NOT sorted by axis index: turf_harsh's index is 1
  // for historical reasons and sorting on it would bury the only live point.
  const points = [];
  for (const {r, i} of withTurf)
    for (const p of r.turf)
      if (!points.some(q => q.name === p.name))
        points.push({name: p.name, label: p.label, x: p.x});

  for (const pt of points) {
    const rows = withTurf
      .map(({r, i}) => ({i, label: r.label, p: (r.turf || []).find(q => q.name === pt.name)}))
      .filter(o => o.p && o.p.mean_duration != null)
      .sort((a, b) => b.p.mean_duration - a.p.mean_duration);
    const card = h("div", "turfcard", null, host);
    const head = h("div", "turfhead", null, card);
    h("span", null, pt.name, head);
    // no threshold chip here: the parameter panel below is the authority, and a
    // hand-written one went stale the moment the recipe changed
    h("span", "turfscale", pt.label, head);
    if (!rows.length) {
      h("div", "turfempty", "no episodes for this severity in the selected runs", card);
      continue;
    }
    // normalise WITHIN the card: a shared axis across incomparable points is
    // exactly the misread the warning above is trying to prevent
    const peak = Math.max(...rows.map(o => o.p.mean_duration + (o.p.mean_duration_se || 0)));
    for (const {i, label, p} of rows) {
      const row = h("div", "turfrow", null, card);
      const nm = h("div", "turfname", null, row);
      const sw = h("span", "swatch", null, nm);
      sw.style.background = sv(i);
      h("span", null, label, nm);
      const wrap = h("div", "turfbarwrap", null, row);
      const frac = peak > 0 ? p.mean_duration / peak : 0;
      const bar = h("div", "turfbar", null, wrap);
      bar.style.width = `${(100 * frac).toFixed(1)}%`;
      bar.style.background = sv(i);
      bar.dataset.run = i;
      if (p.mean_duration_se) {
        const lo = Math.max(0, p.mean_duration - p.mean_duration_se) / peak;
        const hi = Math.min(1, (p.mean_duration + p.mean_duration_se) / peak);
        const err = h("div", "turferr", null, wrap);
        err.style.left = `${(100 * lo).toFixed(1)}%`;
        err.style.width = `${(100 * (hi - lo)).toFixed(1)}%`;
      }
      const val = h("div", "turfval", null, row);
      h("span", null, `${fmtVal(p.mean_duration)} s`, val);
      h("span", "turfsd", ` ±${fmtVal(p.mean_duration_se)}`
        + (p.duration_p50 != null ? ` [${fmtVal(p.duration_p50)}]` : ""), val);
      // outcome mix, inline: WHY the episodes ended, right next to how long
      // they lasted. That pairing is the whole read -- 4 s of fell is a
      // different finding from 4 s of ball-lost.
      const mix = h("div", "turfmix", null, row);
      const tot = Object.values(p.reasons || {}).reduce((a, b) => a + b, 0);
      const share = [];
      for (const [key, lbl, cvar] of REASONS) {
        const cnt = (p.reasons || {})[key] || 0;
        if (!cnt) continue;
        const seg = h("span", null, null, mix);
        seg.style.cssText = `width:${(100 * cnt / tot).toFixed(2)}%;background:${cvar}`;
        share.push(`${lbl} ${Math.round(100 * cnt / tot)}%`);
      }
      mix.title = share.join(" · ") || "no outcomes recorded";
      h("div", "turfn", `n=${p.n}`, row);
      row.title = `${label} — ${pt.name}\n`
        + `ended: ${share.join(", ")}\n`
        + `mean ${fmtVal(p.mean_duration)} ±${fmtVal(p.mean_duration_se)} s`
        + (p.duration_p50 != null ? `, median ${fmtVal(p.duration_p50)} s` : "")
        + `\nfell ${fmtVal(100 - p.survival)}% of ${p.n} episodes`
        + (p.progress != null ? `\nprogress ${fmtVal(p.progress)} m` : "")
        + (p.cross_track != null ? `\ncross-track ${fmtVal(p.cross_track)} m` : "");
    }
  }
  renderTurfParams(host, withTurf);
}

// What produced the numbers above. One value per row when every selected run
// tested the same thing (the normal case); flagged red and listed per run when
// they disagree, because then the ranking above is comparing two experiments.
function renderTurfParams(host, withTurf) {
  const have = withTurf.filter(({r}) => r.turf_params);
  if (!have.length) {
    h("div", "turfempty", "no recorded parameters for the selected runs — the "
      + "run dirs predate plastic_turf.conditions.json; re-run the benchmark to "
      + "record what was tested", host);
    return;
  }
  const det = h("details", "turfparams", null, host);
  const missing = withTurf.length - have.length;
  h("summary", null, `parameters actually tested (${have.length} run`
    + `${have.length > 1 ? "s" : ""}${missing ? `, ${missing} without a record` : ""})`,
    det);
  const grid = h("div", "turfpgrid", null, det);
  for (const [group, items] of TURF_PARAM_GROUPS) {
    const box = h("div", "turfpgroup", null, grid);
    h("h5", null, group, box);
    for (const [key, label, unit] of items) {
      const vals = have.map(({r}) => JSON.stringify(r.turf_params[key] ?? null));
      const uniq = [...new Set(vals)];
      const row = h("div", "turfprow", null, box);
      if (uniq.length > 1) row.classList.add("differ");
      if (uniq.length === 1 && JSON.parse(uniq[0]) === null) row.classList.add("unset");
      h("span", "turfpk", label, row);
      const vd = h("span", "turfpv", null, row);
      h("span", null, uniq.length > 1 ? `${uniq.length} values` : fmtParam(JSON.parse(uniq[0])), vd);
      if (unit) h("span", "turfpu", " " + unit, vd);
      row.title = uniq.length > 1
        ? have.map(({r}) => `${r.label}: ${fmtParam(r.turf_params[key] ?? null)}`).join("\n")
        : `${key} = ${fmtParam(JSON.parse(uniq[0]))}${unit ? " " + unit : ""}`;
    }
  }
}

function fmtParam(v) {
  if (v === null || v === undefined) return "—";
  if (v === true) return "on";
  if (v === false) return "off";
  if (Array.isArray(v))
    return Array.isArray(v[0]) ? v.map(fmtParam).join(" / ")
                               : v.map(x => fmtVal(x)).join(" … ");
  return typeof v === "number" ? fmtVal(v) : String(v);
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
  const dom = {success: [0, 102], success_route: [0, 102],
               success_possession: [0, 102], survival: [0, 102], possession: [0, 102]};
  let ct = 0, ctSuccess = 0;
  for (const {r} of vis)
    for (const pts of [r.corner.L, r.corner.R, r.human, r.uturn.L, r.uturn.R])
      for (const p of pts || []) {
        if (p.cross_track != null) ct = Math.max(ct, p.cross_track);
        if (p.cross_track_success != null)
          ctSuccess = Math.max(ctSuccess, p.cross_track_success);
      }
  dom.cross_track = [0, ct * 1.08 + 1e-6];
  dom.cross_track_success = [0, ctSuccess * 1.08 + 1e-6];
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
  for (const [metric, mLabel, dir] of [["success", "max speed: strict success (%)", "up"],
                                       ["success_possession", "upright + ball at termination (%)", "up"],
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
        // ALWAYS name the checkpoint. This used to be dropped when only one run
        // had videos for the test, on the theory that there was nothing to
        // disambiguate -- but that is exactly the case where you cannot tell:
        // 11 checkpoints selected, one of them recorded this, and the heading
        // said "straight".
        const sw = h("span", "swatch", null, head);
        sw.style.cssText = `width:10px;height:10px;background:${sv(i)}`;
        h("span", null, `${cat} — ${r.label}`, head);
        const strip = h("div", "vstrip", null, catDiv);
        for (const it of has) {
          const tile = h("div", "vtile", null, strip);
          const vid = document.createElement("video");
          vid.preload = "none";
          vid.muted = true;
          vid.playsInline = true;
          vid.src = encodeURI(r.videos[test][it.cond]);
          // the run name again on the tile itself: once you have scrolled the
          // heading off the top, a hover is the only thing left to ask
          vid.title = `${r.label}\n${test} / ${it.cond}\nclick to play/pause`;
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
  renderTurf();
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


# ---- aggregation cache -------------------------------------------------------
# --serve used to rebuild the entire report on every F5. Measured 2026-07-27 on
# 11 runs / 110 928 episodes: 3.3 s re-reading the CSVs, 2.9 s re-aggregating,
# 28.6 s re-bootstrapping the pairwise CIs -- 35 s to reproduce numbers that had
# not changed. Two on-disk caches under <report_dir>/.cache/<code>/ fix that:
#   run/<h>.json    one run's aggregated blob, keyed on that run dir alone
#   diff/<h>.json   one pair's bootstrap CIs, keyed on both runs' two CSVs
# so adding a run only costs that run plus its own pairs, and a refresh that
# changes nothing costs a stat() walk.
#
# STALENESS IS THE ONLY REAL RISK HERE -- a cached number that no longer matches
# the code is worse than a slow report. So <code> is `code_fingerprint`: a hash
# of stats.py plus everything in THIS file above HTML_TEMPLATE, i.e. all the
# aggregation and bootstrap code. Touch any of it and the whole generation is
# abandoned (and swept on the next run); the JS/CSS below the template can still
# be edited for free. There is nothing to remember to bump.
CACHE_DIRNAME = ".cache"

_CODE_FP = None


def _sha(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
        h.update(b"\0")
    return h.hexdigest()[:20]


def code_fingerprint():
    """Hash of the numeric code. None when the sources cannot be read, which
    disables caching rather than risking a key that never invalidates."""
    global _CODE_FP
    if _CODE_FP is None:
        try:
            with open(__file__, "rb") as f:
                src = f.read().split(b'HTML_TEMPLATE = r"""', 1)[0]
            with open(stats.__file__, "rb") as f:
                src += f.read()
        except OSError:
            return None
        _CODE_FP = _sha(src)
    return _CODE_FP


def _stat_fp(paths):
    """(size, mtime_ns) over `paths`; missing files still contribute a slot so
    deleting one moves the fingerprint."""
    parts = []
    for p in paths:
        try:
            st = os.stat(p)
            parts.append(f"{os.path.basename(p)}|{st.st_size}|{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{os.path.basename(p)}|-")
    return _sha(*parts)


def csv_fingerprint(run_dir):
    """Just the episode tables -- all `pair_diffs` reads. Keeping videos out
    means recording a clip does not invalidate 3 s of bootstrap per pair."""
    return _stat_fp([os.path.join(run_dir, f) for f in EPISODE_TABLES])


def tree_fingerprint(run_dir):
    """(relpath, size, mtime_ns) over every file in the run dir -- `collect_run`
    also reads the speed CSVs, params/ provenance and videos/. stat-only, so it
    stays cheap on run dirs carrying hundreds of MB of video."""
    parts = []
    for root, dirs, files in os.walk(run_dir):
        dirs.sort()
        for name in sorted(files):
            p = os.path.join(root, name)
            try:
                st = os.stat(p)
            except OSError:
                continue
            parts.append(f"{os.path.relpath(p, run_dir)}|{st.st_size}|{st.st_mtime_ns}")
    return _sha(*parts)


class AggCache:
    """Content-addressed JSON cache for run blobs and pairwise diffs."""

    def __init__(self, report_dir, run_dirs, enabled=True):
        self.root = os.path.join(report_dir, CACHE_DIRNAME)
        self.code = code_fingerprint() if enabled else None
        self.enabled = self.code is not None
        self.report_dir = report_dir
        self.run_dirs = run_dirs
        self._tree, self._csv = {}, {}
        self.hits = self.misses = 0
        if self.enabled:
            self._drop_old_generations()

    # both fingerprints are lazy: the /_s2s/diff endpoint only ever needs the
    # two CSVs, and walking every run's videos/ for it would be pure waste
    def tree_fp(self, i):
        if i not in self._tree:
            self._tree[i] = tree_fingerprint(self.run_dirs[i])
        return self._tree[i]

    def tree_fps(self):
        return [self.tree_fp(i) for i in range(len(self.run_dirs))]

    def _drop_old_generations(self):
        """Entries live under .cache/<code fingerprint>/, so an edit to the
        aggregation code strands a whole generation at once. Delete the stale
        ones here rather than let a month of iteration silently pile up."""
        import shutil
        try:
            names = os.listdir(self.root)
        except OSError:
            return
        for name in names:
            if name == self.code or not re.fullmatch(r"[0-9a-f]{20}", name):
                continue        # never touch anything we did not write
            shutil.rmtree(os.path.join(self.root, name), ignore_errors=True)

    def csv_fp(self, i):
        if i not in self._csv:
            self._csv[i] = csv_fingerprint(self.run_dirs[i])
        return self._csv[i]

    def _path(self, kind, key):
        return os.path.join(self.root, self.code, kind, key + ".json")

    # hits/misses are counted by the CALLER, not here: a nominal block derived
    # from an already-cached full sweep does two reads and recomputes nothing,
    # so counting raw reads reported a permanent miss on every warm run
    def _read(self, kind, key):
        if not self.enabled:
            return None
        try:
            with open(self._path(kind, key)) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _write(self, kind, key, obj):
        if not self.enabled:
            return
        path = self._path(kind, key)
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(obj, f, separators=(",", ":"), allow_nan=False)
            os.replace(tmp, path)   # atomic: a killed writer leaves no half file
        except (OSError, ValueError):
            try:
                os.remove(tmp)
            except OSError:
                pass

    # run blobs are cached WITHOUT the cross-run context (label, colour slot and
    # the union of robustness groups all depend on which runs are selected);
    # `generate` re-applies those to the loaded blob, which is pure dict work.
    def _run_key(self, i):
        return _sha(self.tree_fp(i), self.report_dir)

    def get_run(self, i):
        return self._read("run", self._run_key(i))

    def put_run(self, i, blob):
        self._write("run", self._run_key(i), blob)

    # one file per (ordered pair, metric, scope): the unit the page actually
    # asks for, so nothing is computed to satisfy a request for something else
    def _block_key(self, i, j, metric, scope):
        return _sha(self.csv_fp(i), self.csv_fp(j), metric, scope,
                    stats.DEFAULT_N_BOOT, stats.DEFAULT_ALPHA)

    def get_block(self, i, j, metric, scope):
        return self._read("diff", self._block_key(i, j, metric, scope))

    def put_block(self, i, j, metric, scope, entries):
        self._write("diff", self._block_key(i, j, metric, scope), entries)


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


def generate(args, live=False, quiet=False, memo=None):
    """Aggregate the CSVs of every (re-)discovered run into the report HTML.

    `memo` is an optional dict the caller keeps across invocations (--serve):
    when nothing under any run dir has moved, the previously built HTML is
    returned verbatim and no CSV is opened at all."""
    run_dirs, labels = resolve_runs(args, quiet=quiet)
    report_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(report_dir, exist_ok=True)
    cache = AggCache(report_dir, run_dirs, enabled=not getattr(args, "no_cache", False))

    # the whole page is a pure function of (code, run set, run contents), so an
    # untouched runs/ tree can be answered from the last build. --no-cache (and
    # an unreadable source tree, which leaves us unable to detect a code edit)
    # turns this off along with the disk cache.
    key = None
    if memo is not None and cache.enabled:
        key = _sha(cache.code, *labels, *cache.tree_fps(), live)
        if memo.get("key") == key:
            return memo["html"]

    # CSVs are read at most once per run, and only for a run whose blob or one
    # of whose pairs missed the cache
    parsed = {}

    def get_rows(i):
        if i not in parsed:
            parsed[i] = tuple(read_rows(os.path.join(run_dirs[i], f))
                              for f in EPISODE_TABLES)
        return parsed[i]

    # cached per run in ISOLATION (its own robustness groups only), then widened
    # below to the UNION across runs so a group only one run swept still gets a
    # panel (the old fixed ROB_GROUPS list both hid new groups and rendered empty
    # ones for groups nobody swept)
    runs = []
    for i, d in enumerate(run_dirs):
        blob = cache.get_run(i)
        if blob is None:
            blob = collect_run(d, labels[i], i, report_dir, None, get_rows(i))
            cache.misses += 1
            cache.put_run(i, blob)
        else:
            cache.hits += 1
        runs.append(blob)
    rob_groups = order_rob_groups(g for r in runs for g in r["robustness"])
    for i, r in enumerate(runs):
        r["label"] = labels[i]          # label/colour depend on the selection,
        r["color"] = i % 8              # not on the run, so they stay outside the key
        r["robustness"] = {g: r["robustness"].get(g, []) for g, _ in rob_groups}
    # INLINE SLICE, O(n): every pair against ONE anchor run, one metric at full
    # sweep plus the headline row of every metric (what the summary table reads).
    # That is the arrival view and, on a file:// snapshot, the only view --
    # everything else is fetched from /_s2s/diff. n-1 pairs, not n(n-1)/2.
    #
    # The anchor is the most COMPLETE run, not run 0. Anchoring on the first run
    # alphabetically meant that if it happened to be missing a table, every inline
    # pair was missing it too -- which is exactly what happened to the field trial
    # the day it landed (the alphabetically-first checkpoint has no exported ONNX,
    # so no plastic_turf.csv, so the arrival view showed zero field-trial rows).
    anchor = max(range(len(runs)),
                 key=lambda i: (len(runs[i]["turf"]),
                                runs[i]["info"]["n_rob"] + runs[i]["info"]["n_cap"],
                                -i)) if runs else 0
    inline_pairs = [(anchor, j) for j in range(len(run_dirs)) if j != anchor]
    inline = {}
    for (i, j), blocks in zip(
            inline_pairs,
            diff_rows_for(get_rows, cache, inline_pairs,
                          list(INLINE_DIFF_METRICS), FULL_SCOPE)):
        inline[f"{min(i, j)}>{max(i, j)}|{FULL_SCOPE}"] = blocks
    for (i, j), blocks in zip(
            inline_pairs,
            diff_rows_for(get_rows, cache, inline_pairs,
                          [m[0] for m in DIFF_METRICS], NOMINAL_SCOPE)):
        inline[f"{min(i, j)}>{max(i, j)}|{NOMINAL_SCOPE}"] = blocks

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
        "__DIFFS__": js_embed(inline),
        "__DIFF_METRICS__": js_embed([(k, lab, d) for k, lab, _, _, _, d in DIFF_METRICS]),
        "__INLINE_ANCHOR__": js_embed(anchor),
        "__DIFF_ENDPOINT__": js_embed(DIFF_ENDPOINT if live else None),
        "__NOMINAL_COND__": js_embed(NOMINAL_COND),
        "__TURF_PARAM_GROUPS__": js_embed(TURF_PARAM_GROUPS),
        "__SCOPES__": js_embed([FULL_SCOPE, NOMINAL_SCOPE]),
        "__REAL_WORLD__": js_embed(REAL_WORLD),
        "__ROB_METRICS__": js_embed(ROB_METRICS),
        "__CAP_METRICS__": js_embed(CAP_METRICS),
        "__META__": js_embed(meta),
        "__DATA__": js_embed(runs),
    }
    # single pass so payload content can never corrupt a later substitution
    html = re.sub("|".join(map(re.escape, payload)),
                  lambda mo: payload[mo.group(0)], HTML_TEMPLATE)
    # also reported under --serve's quiet: a miss is the only case worth a line,
    # and it explains where the seconds went
    if cache.enabled and (not quiet or cache.misses):
        print(f"[html_report] cache {cache.hits} served / {cache.misses} computed "
              f"({os.path.join(report_dir, CACHE_DIRNAME)})", file=sys.stderr)
    if memo is not None and key is not None:
        memo.update(key=key, html=html)
    return html


def serve(args):
    """Live mode: every page refresh re-discovers runs and re-aggregates what
    actually changed. Static assets (videos, CSVs) come straight from disk, so
    the report's relative video links keep working.

    `memo` makes an unchanged refresh cost one stat() walk instead of a full
    rebuild; `AggCache` makes a changed one cost only the runs and pairs that
    moved. Both are keyed on file fingerprints, so editing a CSV by hand or
    dropping in a new run dir is picked up the same as a fresh eval."""
    import functools
    import http.server
    import threading
    import time

    root = os.getcwd()
    out_abs = os.path.abspath(args.out)
    rel = os.path.relpath(out_abs, root)
    report_url = None if rel.startswith("..") else "/" + rel.replace(os.sep, "/")
    memo, lock = {}, threading.Lock()

    def diff_api(req):
        """{pairs:[[labelA,labelB],...], metrics:[...], scope} -> [{metric: [...]}].

        Pairs the page cannot resolve (a run dir that vanished between load and
        click) come back as null rather than shifting the rest of the batch."""
        run_dirs, labels = resolve_runs(args, quiet=True)
        report_dir = os.path.dirname(os.path.abspath(args.out)) or "."
        cache = AggCache(report_dir, run_dirs,
                         enabled=not getattr(args, "no_cache", False))
        idx = {lab: i for i, lab in enumerate(labels)}
        parsed = {}

        def get_rows(i):
            if i not in parsed:
                parsed[i] = tuple(read_rows(os.path.join(run_dirs[i], f))
                                  for f in EPISODE_TABLES)
            return parsed[i]

        known = {m[0] for m in DIFF_METRICS}
        metrics = [m for m in req.get("metrics") or [] if m in known]
        scope = req.get("scope")
        if scope not in (FULL_SCOPE, NOMINAL_SCOPE) or not metrics:
            raise ValueError("bad scope or metrics")
        out = []
        for pair in req.get("pairs") or []:
            i, j = (idx.get(pair[0]), idx.get(pair[1])) if len(pair) == 2 else (None, None)
            out.append(None if i is None or j is None or i == j else
                       {m: diff_block(get_rows, cache, i, j, m, scope) for m in metrics})
        return out

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_POST(self):
            if self.path.split("?", 1)[0] != DIFF_ENDPOINT:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > 1 << 20:            # a batch is a few KB of labels
                    self.send_error(413)
                    return
                req = json.loads(self.rfile.read(length) or b"{}")
                t0 = time.perf_counter()
                body = json.dumps({"blocks": diff_api(req)},
                                  separators=(",", ":"), allow_nan=False).encode()
            except (ValueError, KeyError, TypeError) as exc:
                self.send_error(400, f"bad diff request: {exc}")
                return
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:
                self.send_error(500, f"diff computation failed: {exc}")
                return
            took = time.perf_counter() - t0
            if took > 0.5:      # a cold batch; say so rather than look hung
                sys.stderr.write(f"[html_report] diff batch "
                                 f"{len(req.get('pairs') or [])} pairs x "
                                 f"{len(req.get('metrics') or [])} metrics "
                                 f"({req.get('scope')}) in {took:.1f}s\n")
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

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
                # ThreadingHTTPServer: serialise so two tabs refreshing at once
                # rebuild once, not twice
                with lock:
                    before = memo.get("key")
                    t0 = time.perf_counter()
                    try:
                        html = generate(args, live=True, quiet=True, memo=memo)
                    except Exception as exc:
                        self.send_error(500, f"report generation failed: {exc}")
                        return
                    rebuilt = memo.get("key") != before
                    if rebuilt:
                        sys.stderr.write(
                            f"[html_report] rebuilt in {time.perf_counter() - t0:.1f}s\n")
                        try:
                            with open(out_abs, "w") as f:   # keep the snapshot fresh
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
            if self.path.split("?", 1)[0] in ("/", report_url, DIFF_ENDPOINT):
                sys.stderr.write(f"[html_report] {fmt % fargs}\n")

    srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", args.port), functools.partial(Handler, directory=root))
    url = f"http://127.0.0.1:{args.port}" + (report_url or "/")
    print(f"[html_report] live report at {url} — every refresh re-checks "
          f"{args.runs_root if args.run_dirs is None else 'the given run dirs'} and "
          f"rebuilds only what moved; Ctrl-C to stop")
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
                         "file: every browser refresh re-discovers runs and rebuilds "
                         "whatever changed (the --out snapshot is refreshed on each "
                         "actual rebuild)")
    ap.add_argument("--port", type=int, default=8000, help="port for --serve")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore <report_dir>/.cache/ and re-aggregate everything "
                         "from the CSVs. The cache keys already carry a hash of the "
                         "aggregation and bootstrap code, so this is a debugging "
                         "escape hatch, not something a code change needs")
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
