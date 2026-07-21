"""Comparison figures from benchmark CSVs (engine-independent; numpy+matplotlib only).

  python -m sim2sim_benchmark.plot \
      --run-dirs eval_result/m80000 eval_result/m90000 \
      --labels iter80000 iter90000 --out-dir eval_result

Each run dir is one EXPERIMENT (one color) holding robustness.csv and/or
capability.csv (+ capability_speed_pairs.csv). Three output figures:

  robustness_compare.png  perturbation axes x survival/possession/speed/tracking
  speed_compare.png       SPEED capability: max speed (success + achieved vs
                          commanded on the straight) and controllability
                          (cmd-vs-actual over the human-route vmax sweep, pooled r)
  route_compare.png       ROUTE capability: corner turn success + cross-track
                          over |kappa| (solid = left, dashed = right)
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib

from .real_world import real_marker

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _route_engine():
    """Import the engine for ROUTE GEOMETRY only (the schematic panels). The
    sim deps are stubbed when absent, so plotting works on CSV-only machines."""
    import types
    for name in ("mujoco", "onnxruntime"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                sys.modules[name] = types.ModuleType(name)
    if not hasattr(sys.modules["mujoco"], "viewer"):
        try:
            __import__("mujoco.viewer")
        except ImportError:
            sys.modules["mujoco.viewer"] = types.ModuleType("mujoco.viewer")
            sys.modules["mujoco"].viewer = sys.modules["mujoco.viewer"]
    from . import engine
    return engine


def _example_route(kappa=None, lead_m=1.0, angle_deg=180.0, human_seed=None,
                   route_len=20.0, human_kappa_cap=None, vmax_range=None):
    """One example route polyline from the REAL benchmark route generator —
    the schematic shows exactly the geometry the test commands."""
    engine = _route_engine()
    cfg = dict(engine.ROUTE_CFG)
    cfg["routeLength"] = route_len
    if human_kappa_cap is not None:
        cfg["routeHumanKappaCap"] = human_kappa_cap
    route = engine.Route(cfg, 0)
    route.rng = np.random.Generator(np.random.PCG64(0 if human_seed is None else human_seed))
    if vmax_range is not None:
        # consume the same per-episode cruise draw as the real condition, so the
        # geometry draws line up with the actual test routes
        route.vmax_range = tuple(vmax_range)
    if kappa is not None:
        route.const_kappa = kappa
        route.lead_range = (lead_m, lead_m)
        route.arc_deg = (angle_deg, angle_deg)
    route.reset(np.zeros(2), np.array([1.0, 0.0]), 4 if human_seed is not None else 1)
    while route.filled < len(route.speed):
        route.last_seg = max(0, route.filled - 1)
        route._extend()
    return route.points[: route.filled + 1]


def _draw_route_examples(panel, routes, title):
    """routes: list of (points, label, color, linestyle)."""
    for points, label, color, linestyle in routes:
        panel.plot(points[:, 0], points[:, 1], color=color, ls=linestyle,
                   lw=1.8, label=label)
    panel.plot(0, 0, marker="o", ms=7, color="black")
    panel.annotate("start (ball)", (0, 0), textcoords="offset points",
                   xytext=(6, -14), fontsize=8, color="0.4")
    panel.set_aspect("equal")
    panel.grid(alpha=0.25)
    panel.set_title(title, fontsize=11)
    panel.set_xlabel("x (m)")
    panel.set_ylabel("y (m)")
    panel.legend(fontsize=7.5, loc="best")


SCHEMATIC_COLORS = ["#9dc3e6", "#4f81bd", "#1f3864"]

ROBUSTNESS_GROUPS = ["ball_mass", "ball_radius", "foot_friction", "ball_friction",
                     "ball_damping", "base_push", "ball_push", "obs_latency", "act_latency"]
GROUP_LABEL = {
    "ball_mass": "ball mass (kg)",
    "ball_radius": "ball radius (m)",
    "foot_friction": "foot friction",
    "ball_friction": "ball friction",
    "ball_damping": "ball roll brake c (1 m/s rolls 3.5/c m)",
    "dr_scale": "DR scale alpha",   # legacy CSVs only
    "base_push": "base push dv (m/s)",
    "ball_push": "ball push dv (m/s)",
    "obs_latency": "ball-obs latency (steps)",
    "act_latency": "action latency (ms)",
    "straight_speed": "cmd speed, straight (m/s)",
    "corner_turn": "|kappa|, corner turn (1/m)",
}
EXPERIMENT_COLORS = ["tab:red", "tab:blue", "0.45", "tab:green", "tab:purple", "tab:orange"]


def load_rows(csv_path):
    rows = []
    with open(csv_path) as f:
        for record in csv.DictReader(f):
            def num(key):
                value = record.get(key)
                return float(value) if value not in (None, "") else float("nan")
            rows.append(dict(group=record["group"], axis=float(record["axis_value"]),
                             condition=record.get("condition", ""),
                             rep=record.get("rep", ""),
                             fell=float(record["fell"]), lost=float(record["ball_lost"]),
                             cross_track=num("cross_track_m"), ach_speed=num("ach_speed_mps"),
                             cmd_speed=num("cmd_speed_mps"), success=num("success"),
                             speed_corr_r=num("speed_corr_r"),
                             progress=num("progress_m"), duration=num("duration_s"),
                             ball_dist=num("ball_dist_m"),
                             reason=(record.get("fail_reason") or "").strip()))
    return rows


def load_speed_pairs(csv_path):
    """(cmd, actual) arrays from a *_speed_pairs.csv."""
    cmd, actual = [], []
    with open(csv_path) as f:
        for record in csv.DictReader(f):
            cmd.append(float(record["cmd_speed_mps"]))
            actual.append(float(record["ball_speed_mps"]))
    return np.asarray(cmd), np.asarray(actual)


def load_speed_traces(csv_path):
    """{(vmax, episode): (cmd, v_along_cmd, v_abs) 50 Hz arrays} from
    *_speed_traces.csv."""
    by_key = {}
    with open(csv_path) as f:
        for record in csv.DictReader(f):
            key = (float(record["axis_value"]), int(record.get("episode", 0) or 0))
            by_key.setdefault(key, []).append(
                (int(record["step"]), float(record["cmd_speed_mps"]),
                 float(record["ball_speed_along_cmd_mps"]),
                 float(record["ball_speed_abs_mps"])))
    traces = {}
    for key, steps in by_key.items():
        steps.sort()
        arr = np.asarray(steps, float)
        traces[key] = (arr[:, 1], arr[:, 2], arr[:, 3])
    return traces


def smooth(values, window=25):
    """0.5 s moving average at the 50 Hz policy rate."""
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="same")


def level_stats(rows):
    """axis value -> survival/possession/success rates (+binomial SE), speed ratio,
    survivor cross-track, ball-distance percentiles, progress.

    EVERY continuous metric here is SURVIVORS-ONLY. An episode truncated by a
    fall covers its route distance in less wall time, so an unfiltered
    ach_speed / speed_ratio mean RISES as the condition gets harder — the metric
    would move opposite to the thing it is supposed to measure. Filtering also
    matches cross-track, which was already survivors-only.

    ball_dist_p50/p90 are the continuous form of possession: the binary
    `ball_lost` flag is near-degenerate in practice (1 of 3504 episodes on the
    2026-07-20 runs), so the rate panel is a flat 100% line while the underlying
    distance carries a clean monotone signal.
    """
    by_axis = {}
    for row in rows:
        by_axis.setdefault(row["axis"], []).append(row)
    stats = {}
    for axis_value, level_rows in sorted(by_axis.items()):
        n = len(level_rows)
        alive = [r for r in level_rows if r["fell"] < 0.5]

        def rate(values):
            values = [v for v in values if np.isfinite(v)]
            if not values:
                return float("nan"), float("nan")
            p = 1.0 - float(np.mean(values))
            return p, np.sqrt(max(p * (1 - p), 1e-4 / len(values)) / len(values))

        def mean_of(key, source):
            values = [r[key] for r in source if np.isfinite(r.get(key, float("nan")))]
            return float(np.mean(values)) if values else float("nan")

        def pct_of(key, q, source):
            values = [r[key] for r in source if np.isfinite(r.get(key, float("nan")))]
            return float(np.percentile(values, q)) if values else float("nan")

        survival, survival_se = rate([r["fell"] for r in level_rows])
        possession, possession_se = rate([r["lost"] for r in level_rows])
        success, success_se = rate([1.0 - r["success"] for r in level_rows
                                    if np.isfinite(r["success"])] or [float("nan")])
        speed_ratios = [r["ach_speed"] / r["cmd_speed"] for r in alive
                        if np.isfinite(r["ach_speed"]) and np.isfinite(r["cmd_speed"])
                        and r["cmd_speed"] > 0.05]
        survivor_ct = [r["cross_track"] for r in alive if np.isfinite(r["cross_track"])]
        stats[axis_value] = dict(
            n=n, n_alive=len(alive),
            survival=survival, survival_se=survival_se,
            possession=possession, possession_se=possession_se,
            success=success, success_se=success_se,
            speed_ratio=float(np.mean(speed_ratios)) if speed_ratios else float("nan"),
            ach_speed=mean_of("ach_speed", alive),
            cross_track=float(np.mean(survivor_ct)) if survivor_ct else float("nan"),
            ball_dist_p50=pct_of("ball_dist", 50, level_rows),
            ball_dist_p90=pct_of("ball_dist", 90, level_rows),
            progress=mean_of("progress", level_rows),
            duration=mean_of("duration", level_rows))
    return stats


def draw_series(panel, stats, x_values, metric, color, linestyle, label=None, use_abs=False):
    x_plot = np.abs(np.asarray(x_values, float)) if use_abs else np.asarray(x_values, float)
    y = np.array([stats[v][metric] for v in x_values])
    if metric in ("survival", "possession", "success"):
        se = np.array([stats[v][f"{metric}_se"] for v in x_values])
        panel.plot(x_plot, 100 * y, marker="o", ms=3.5, lw=1.7,
                   color=color, ls=linestyle, label=label)
        panel.fill_between(x_plot, 100 * (y - se), 100 * (y + se), color=color, alpha=0.15)
    else:
        panel.plot(x_plot, y, marker="o", ms=3.5, lw=1.7,
                   color=color, ls=linestyle, label=label)


REAL_COLOR = "#111111"


def _draw_real_marker(panel, group, annotate):
    """Overlay where the DEPLOYMENT HARDWARE sits on this axis (real_world table):
    shaded span = measurement spread, dashed line = the measured nominal. Channels
    that have not been measured get nothing — see the real_world docstring on why
    an unmarked panel is preferable to a guessed one. Returns True if drawn."""
    marker = real_marker(group)
    if marker is None:
        return False
    nominal, band, _ = marker
    if band is not None:
        panel.axvspan(band[0], band[1], color=REAL_COLOR, alpha=0.07, lw=0, zorder=0)
    if nominal is not None:
        panel.axvline(nominal, color=REAL_COLOR, ls="--", lw=1.3, alpha=0.7,
                      zorder=1, label="real (hardware)" if annotate else None)
    if annotate:
        x = nominal if nominal is not None else 0.5 * (band[0] + band[1])
        panel.annotate("real", xy=(x, 1.0), xycoords=("data", "axes fraction"),
                       xytext=(3, -10), textcoords="offset points",
                       fontsize=8, color=REAL_COLOR, alpha=0.8)
    return True


def _style_panel(panel, row_metric, group, is_top, is_bottom, is_left, row_label):
    panel.grid(alpha=0.3)
    if row_metric in ("survival", "possession", "success"):
        panel.set_ylim(-5, 105)
    elif row_metric == "speed_ratio":
        # floor at 0 and reserve room up to 1.15, but let the axis GROW past it:
        # a hard set_ylim(0, 1.15) silently clipped any run that overshoots the
        # commanded speed, which reads as "no data up there" rather than ">1"
        low, high = panel.get_ylim()
        panel.set_ylim(0, max(1.15, high))
    if is_top:
        panel.set_title(GROUP_LABEL.get(group, group), fontsize=10.5)
    if is_bottom:
        panel.set_xlabel(GROUP_LABEL.get(group, group), fontsize=9)
    if is_left:
        panel.set_ylabel(row_label, fontsize=10)


def _collect_legend(fig, top_row_panels, n_labels):
    seen = {}
    for panel in top_row_panels:
        for handle, label in zip(*panel.get_legend_handles_labels()):
            seen.setdefault(label, handle)
    fig.legend(list(seen.values()), list(seen.keys()), loc="lower center",
               ncol=max(2, n_labels + 1), fontsize=10, frameon=False)


def robustness_figure(experiments, labels, out_path):
    """Rows: survival / ball distance p90 / speed ratio / cross-track. Dotted
    horizontal lines = each experiment's nominal baseline.

    The p90 ball-distance row replaced the old binary "ball possession" row: the
    `ball_lost` flag fires on ~1 episode in 3500, so that panel was a flat 100%
    line in every group while the distance it thresholds degrades cleanly."""
    metrics = [("survival", "survival rate (%)"),
               ("ball_dist_p90", "robot-ball distance (m, p90)"),
               ("speed_ratio", "speed ratio (achieved/cmd, survivors)"),
               ("cross_track", "cross-track (m, survivors)")]
    groups = [g for g in ROBUSTNESS_GROUPS if any(r["group"] == g for e in experiments for r in e)]
    groups += sorted({r["group"] for e in experiments for r in e} - set(groups)
                     - {"baseline", "straight_speed", "corner_turn", "u_turn",
                        "human_dribble", "speed_tracking"})
    if not groups:
        return False
    # metric grid + one extra column: the shared route-bank schematic (top half)
    fig = plt.figure(figsize=(3.4 * (len(groups) + 1) + 1, 12))
    grid = fig.add_gridspec(len(metrics), len(groups) + 1)
    panels = np.empty((len(metrics), len(groups)), dtype=object)
    for row in range(len(metrics)):
        for col in range(len(groups)):
            panels[row, col] = fig.add_subplot(
                grid[row, col], sharey=None if col == 0 else panels[row, 0])
    schematic_panel = fig.add_subplot(grid[0:2, len(groups)])
    _draw_route_examples(schematic_panel, [
        (_example_route(human_seed=seed, route_len=30.0), f"route seed {seed}", color, "-")
        for seed, color in zip((0, 1, 2), SCHEMATIC_COLORS)
    ], "test routes: nominal human bank\n(every condition cycles the same seeds)")
    labeled = set()
    for col, group in enumerate(groups):
        for exp_index, (rows, label) in enumerate(zip(experiments, labels)):
            color = EXPERIMENT_COLORS[exp_index % len(EXPERIMENT_COLORS)]
            group_rows = [r for r in rows if r["group"] == group]
            if not group_rows:
                continue
            stats = level_stats(group_rows)
            x_values = sorted(stats)
            for row, (metric, _) in enumerate(metrics):
                series_label = None
                if row == 0 and exp_index not in labeled:
                    series_label = label; labeled.add(exp_index)
                draw_series(panels[row, col], stats, x_values, metric, color, "-", series_label)
            baseline = [r for r in rows if r["group"] == "baseline"]
            if baseline:
                base = list(level_stats(baseline).values())[0]
                for row, (metric, _) in enumerate(metrics):
                    value = base[metric]
                    if np.isfinite(value):
                        scale = 100 if metric in ("survival", "possession") else 1
                        panels[row, col].axhline(scale * value, color=color, ls=":",
                                                 lw=0.9, alpha=0.55)
        for row, (metric, row_label) in enumerate(metrics):
            # marker AFTER the series so the axis limits are already settled;
            # only the top panel contributes the legend entry / "real" text
            _draw_real_marker(panels[row, col], group, annotate=(row == 0))
            _style_panel(panels[row, col], metric, group, row == 0,
                         row == len(metrics) - 1, col == 0, row_label)
    _collect_legend(fig, [panels[0, c] for c in range(len(groups))], len(labels))
    fig.suptitle("Sim2sim benchmark — ROBUSTNESS: perturbation axes on nominal routes; "
                 "one color per experiment, dotted = its nominal baseline, "
                 "dashed black + shading = the measured REAL hardware value", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_path, dpi=115)
    print(f"saved {out_path}")
    return True


def speed_figure(experiments, labels, pair_data, out_path):
    """The SPEED capability figure — three panels:
      1. max speed, success rate vs commanded speed (straight route);
      2. max speed, achieved vs commanded (straight; the plateau off the y=x
         line is the measured max dribble speed);
      3. controllability: binned commanded-vs-actual over the human-route vmax
         sweep, identity line, pooled Pearson r per experiment in the legend.
    pair_data: {label: (cmd array, actual array)} from capability_speed_pairs.csv."""
    straight = [(rows, label) for rows, label in zip(experiments, labels)
                if any(r["group"] == "straight_speed" for r in rows)]
    if not straight and not pair_data:
        return False
    fig, (success_panel, achieved_panel, tracking_panel, schematic_panel) = plt.subplots(
        1, 4, figsize=(22, 5.6))

    for exp_index, (rows, label) in enumerate(zip(experiments, labels)):
        color = EXPERIMENT_COLORS[exp_index % len(EXPERIMENT_COLORS)]
        group_rows = [r for r in rows if r["group"] == "straight_speed"]
        if not group_rows:
            continue
        stats = level_stats(group_rows)
        x_values = sorted(stats)
        draw_series(success_panel, stats, x_values, "success", color, "-", label)
        survival_pct = 100 * np.array([stats[v]["survival"] for v in x_values])
        success_panel.plot(np.asarray(x_values, float), survival_pct,
                           ls=":", lw=1.5, color=color)
        draw_series(achieved_panel, stats, x_values, "ach_speed", color, "-")
    success_panel.set_ylim(-5, 105)
    success_panel.set_xlabel("commanded speed (m/s), straight route")
    success_panel.set_ylabel("rate (%)")
    success_panel.set_title("max speed: success (solid)\n/ survival = no fall (dotted)",
                            fontsize=11)

    straight_axes = [r["axis"] for e in experiments for r in e if r["group"] == "straight_speed"]
    if straight_axes:
        lo, hi = min(straight_axes), max(straight_axes)
        achieved_panel.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1.2,
                            label="achieved = commanded")
        achieved_panel.legend(fontsize=8.5, loc="upper left")
    achieved_panel.set_xlabel("commanded speed (m/s), straight route")
    achieved_panel.set_ylabel("achieved ball speed (m/s)")
    achieved_panel.set_title("max speed: the plateau = measured max", fontsize=11)

    if pair_data:
        cmd_lo = min(cmd.min() for cmd, _ in pair_data.values()) - 0.1
        cmd_hi = max(cmd.max() for cmd, _ in pair_data.values()) + 0.1
        tracking_panel.plot([cmd_lo, cmd_hi], [cmd_lo, cmd_hi], color="gray", ls="--",
                            lw=1.2, label="actual = commanded")
        bins = np.arange(cmd_lo, cmd_hi + 0.05, 0.05)
        for exp_index, label in enumerate(labels):
            if label not in pair_data:
                continue
            cmd, actual = pair_data[label]
            color = EXPERIMENT_COLORS[exp_index % len(EXPERIMENT_COLORS)]
            centers, means, stds = [], [], []
            for b0, b1 in zip(bins[:-1], bins[1:]):
                in_bin = (cmd >= b0) & (cmd < b1)
                if in_bin.sum() >= 20:
                    centers.append(0.5 * (b0 + b1))
                    means.append(float(actual[in_bin].mean()))
                    stds.append(float(actual[in_bin].std()))
            centers, means, stds = map(np.asarray, (centers, means, stds))
            pooled_r = (float(np.corrcoef(cmd, actual)[0, 1])
                        if cmd.std() > 1e-3 else float("nan"))
            tracking_panel.plot(centers, means, marker="o", ms=4, lw=1.8, color=color,
                                label=f"{label} (r={pooled_r:.2f})")
            tracking_panel.fill_between(centers, means - stds, means + stds,
                                        color=color, alpha=0.15)
        tracking_panel.legend(fontsize=8.5, loc="upper left")
    tracking_panel.set_xlabel("commanded speed (m/s), human routes")
    tracking_panel.set_ylabel("actual ball speed along command (m/s, 0.5 s smoothed)")
    tracking_panel.set_title("controllability: cmd vs actual, pooled Pearson r", fontsize=11)

    _draw_route_examples(schematic_panel, [
        (np.array([[0.0, 0.0], [12.0, 0.0]]), "straight (max speed)", "black", "-"),
        (_example_route(human_seed=5, route_len=25.0), "human route (tracking)",
         "#4f81bd", "-"),
        (_example_route(human_seed=11, route_len=25.0), None, "#9dc3e6", "-"),
    ], "test routes")

    for panel in (success_panel, achieved_panel, tracking_panel):
        panel.grid(alpha=0.3)
    fig.suptitle("Sim2sim benchmark — SPEED: max speed (straight, 10 s, fail-fast) "
                 "+ controllability (human routes, trained command distribution)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=115)
    print(f"saved {out_path}")
    return True


def speed_control_traces_figure(traces, label, out_path):
    """Per-episode control traces for ONE experiment (one panel per traced
    episode): gray = raw ball |v|, red = |v| smoothed 0.5 s, blue =
    v-along-command smoothed 0.5 s, black dashed = commanded speed."""
    if not traces:
        return False
    keys = sorted(traces)
    n_cols = min(4, len(keys))
    n_rows = -(-len(keys) // n_cols)
    fig, panels = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.6 * n_rows),
                               sharey=True, squeeze=False)
    for index, key in enumerate(keys):
        _, episode = key
        panel = panels[index // n_cols, index % n_cols]
        cmd, along_cmd, speed_abs = traces[key]
        t = np.arange(len(cmd)) / 50.0
        panel.plot(t, speed_abs, color="0.8", lw=0.6,
                   label="ball |v| raw" if index == 0 else None)
        panel.plot(t, smooth(speed_abs), color="tab:red", lw=1.6,
                   label="ball |v| smooth 0.5s" if index == 0 else None)
        panel.plot(t, smooth(along_cmd), color="tab:blue", lw=1.6,
                   label="ball v-along-cmd smooth 0.5s" if index == 0 else None)
        panel.plot(t, cmd, color="black", ls="--", lw=1.3,
                   label="cmd target" if index == 0 else None)
        panel.set_title(f"env{episode}  mean v-cmd={np.mean(along_cmd):.2f}  "
                        f"mean cmd={np.mean(cmd):.2f} m/s", fontsize=10)
        panel.grid(alpha=0.25)
        panel.set_xlabel("t (s)")
        if index % n_cols == 0:
            panel.set_ylabel("m/s")
        # bottom-right inset: the ACTUAL route of this episode — env i replays
        # route-bank seed i with the same per-episode cruise draw as the real
        # speed_tracking condition, so this is the very polyline it followed
        inset = panel.inset_axes([0.60, 0.05, 0.38, 0.40])
        points = _example_route(human_seed=episode, route_len=45.0,
                                vmax_range=(1.2, 2.0))
        inset.plot(points[:, 0], points[:, 1], color="#4f81bd", lw=1.0)
        inset.plot(0, 0, marker="o", ms=3, color="black")
        inset.set_aspect("equal")
        inset.tick_params(labelsize=6, length=2, pad=1)
        inset.set_title("route (m)", fontsize=7, pad=2)
    for index in range(len(keys), n_rows * n_cols):
        panels[index // n_cols, index % n_cols].axis("off")
    panels[0, 0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle(f"Speed control traces — {label}: ball speed vs cmd target "
                 "(human routes, smoothed 0.5 s MA)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=115)
    print(f"saved {out_path}")
    return True


def _draw_turn_row(panels, experiments, labels, group, test_name,
                   schematic_kappas, schematic_lead_m, schematic_angle_deg):
    """One turn-test row (corner_turn / u_turn) over |kappa|: success rate,
    survival rate (no fall before termination, whatever ended the episode),
    survivor cross-track, and an example-routes schematic;
    solid = left turns, dashed = right turns."""
    success_panel, survival_panel, ct_panel, schematic_panel = panels
    labeled = set()
    for exp_index, (rows, label) in enumerate(zip(experiments, labels)):
        color = EXPERIMENT_COLORS[exp_index % len(EXPERIMENT_COLORS)]
        group_rows = [r for r in rows if r["group"] == group]
        if not group_rows:
            continue
        for sign, linestyle, tag in ((1, "-", "L"), (-1, "--", "R")):
            side_rows = [r for r in group_rows if (r["axis"] >= 0) == (sign > 0)]
            if not side_rows:
                continue
            stats = level_stats(side_rows)
            x_values = sorted(stats)
            series_label = None
            if (exp_index, tag) not in labeled:
                series_label = f"{label} ({tag})"; labeled.add((exp_index, tag))
            draw_series(success_panel, stats, x_values, "success", color, linestyle,
                        series_label, use_abs=True)
            draw_series(survival_panel, stats, x_values, "survival", color, linestyle,
                        use_abs=True)
            draw_series(ct_panel, stats, x_values, "cross_track", color, linestyle,
                        use_abs=True)
    success_panel.set_ylim(-5, 105)
    success_panel.set_ylabel("success rate (%)")
    success_panel.set_title(f"{test_name}: success (finished the turn,\n"
                            "no fall, ball kept)", fontsize=11)
    survival_panel.set_ylim(-5, 105)
    survival_panel.set_ylabel("survival rate (%)")
    survival_panel.set_title(f"{test_name}: survival\n(no fall before termination)",
                             fontsize=11)
    ct_panel.set_ylabel("cross-track (m, survivors)")
    ct_panel.set_title(f"{test_name}: tracking error", fontsize=11)
    for panel in (success_panel, survival_panel, ct_panel):
        panel.set_xlabel("|kappa| (1/m)")
        panel.grid(alpha=0.3)
    success_panel.legend(fontsize=8.5)
    # example routes drawn from the real generator; the turn radius label is the
    # EFFECTIVE one (the back-solved kappa of the ds=0.25 m command polyline)
    segment_len = _route_engine().ROUTE_CFG["routeSegmentLength"]
    theta = np.deg2rad(schematic_angle_deg)
    example_routes = []
    # long exit tail: drawn at the same wide proportions as the training
    # route-design figures (the tight cap reads as an arc at this zoom)
    route_len = schematic_lead_m + theta / min(schematic_kappas) + 8.0
    for kappa, color in zip(schematic_kappas, SCHEMATIC_COLORS):
        n_segments = max(3, int(np.ceil(theta / (kappa * segment_len))))
        effective_radius = (n_segments * segment_len) / theta
        example_routes.append((_example_route(kappa=kappa, lead_m=schematic_lead_m,
                                              angle_deg=schematic_angle_deg,
                                              route_len=route_len),
                               f"|kappa|={kappa:g} (turn R~{effective_radius:.2f} m)",
                               color, "-"))
    mid_kappa = schematic_kappas[len(schematic_kappas) // 2]
    example_routes.append((_example_route(kappa=-mid_kappa, lead_m=schematic_lead_m,
                                          angle_deg=schematic_angle_deg,
                                          route_len=route_len),
                           f"right turn (kappa=-{mid_kappa:g})", SCHEMATIC_COLORS[1], "--"))
    _draw_route_examples(schematic_panel, example_routes,
                         f"test routes: lead-in + {schematic_angle_deg:.0f} deg turn + exit\n"
                         f"(command polyline, ds={segment_len:g} m)")


def _draw_human_dribble_row(panels, experiments, labels):
    """The human-dribble row of the route figure: success / survival /
    cross-track over the route_human_kappa_cap sweep + example routes."""
    success_panel, survival_panel, ct_panel, schematic_panel = panels
    for exp_index, (rows, label) in enumerate(zip(experiments, labels)):
        color = EXPERIMENT_COLORS[exp_index % len(EXPERIMENT_COLORS)]
        group_rows = [r for r in rows if r["group"] == "human_dribble"]
        if not group_rows:
            continue
        stats = level_stats(group_rows)
        x_values = sorted(stats)
        draw_series(success_panel, stats, x_values, "success", color, "-", label)
        draw_series(survival_panel, stats, x_values, "survival", color, "-")
        draw_series(ct_panel, stats, x_values, "cross_track", color, "-")
    success_panel.set_ylim(-5, 105)
    success_panel.set_ylabel("success rate (%)")
    success_panel.set_title("human dribble: success\n(kept control for 20 s)", fontsize=11)
    survival_panel.set_ylim(-5, 105)
    survival_panel.set_ylabel("survival rate (%)")
    survival_panel.set_title("human dribble: survival\n(no fall before termination)",
                             fontsize=11)
    ct_panel.set_ylabel("cross-track (m, survivors)")
    ct_panel.set_title("human dribble: tracking error", fontsize=11)
    for panel in (success_panel, survival_panel, ct_panel):
        panel.set_xlabel("route_human_kappa_cap (1/m)")
        panel.grid(alpha=0.3)
    success_panel.legend(fontsize=8.5)
    example_routes = []
    for cap, color, seed in ((0.3, SCHEMATIC_COLORS[0], 7), (0.7, SCHEMATIC_COLORS[1], 7),
                             (1.1, SCHEMATIC_COLORS[2], 7)):
        example_routes.append((_example_route(human_seed=seed, route_len=30.0,
                                              human_kappa_cap=cap),
                               f"kappa cap {cap:g}", color, "-"))
    _draw_route_examples(schematic_panel, example_routes,
                         "human routes, same seed:\nturn aggressiveness scales with the cap")


def route_figure(experiments, labels, out_path):
    """The ROUTE figure: corner-turn row on top, human-dribble row below."""
    has_corner = any(r["group"] == "corner_turn" for e in experiments for r in e)
    has_human = any(r["group"] == "human_dribble" for e in experiments for r in e)
    if not (has_corner or has_human):
        return False
    fig, panels = plt.subplots(2, 4, figsize=(22.5, 10.4))
    _draw_turn_row(panels[0], experiments, labels, "corner_turn", "corner turn",
                   schematic_kappas=(0.4, 0.7, 1.0), schematic_lead_m=2.75,
                   schematic_angle_deg=165.0)
    _draw_human_dribble_row(panels[1], experiments, labels)
    fig.suptitle("Sim2sim benchmark — ROUTE: corner turn (150-180 deg arc, 12 s; "
                 "solid = L, dashed = R) + human dribble (kappa-cap sweep, 20 s)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=115)
    print(f"saved {out_path}")
    return True


def uturn_figure(experiments, labels, out_path):
    if not any(r["group"] == "u_turn" for e in experiments for r in e):
        return False
    fig, panels = plt.subplots(1, 4, figsize=(22.5, 5.4))
    _draw_turn_row(panels, experiments, labels, "u_turn", "u-turn",
                   schematic_kappas=(2.0, 3.0, 4.0), schematic_lead_m=2.75,
                   schematic_angle_deg=180.0)
    fig.suptitle("Sim2sim benchmark — U-TURN: about-face drill (run-in + 160-200 deg turn, "
                 "radius 1/kappa, 10 s); solid = L, dashed = R", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=115)
    print(f"saved {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser(prog="sim2sim_benchmark.plot", description=__doc__)
    ap.add_argument("--run-dirs", nargs="+", required=True,
                    help="one benchmark output dir per experiment "
                         "(holding robustness.csv / capability.csv)")
    ap.add_argument("--labels", nargs="*", default=None, help="one label per run dir")
    ap.add_argument("--out-dir", default=".", help="where the comparison figures go")
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.run_dirs]
    assert len(labels) == len(args.run_dirs), "--labels must match --run-dirs"
    os.makedirs(args.out_dir, exist_ok=True)

    # robustness figure (robustness.csv per run dir)
    experiments, kept_labels, fingerprints = [], [], {}
    for run_dir, label in zip(args.run_dirs, labels):
        csv_path = os.path.join(run_dir, "robustness.csv")
        if os.path.exists(csv_path):
            experiments.append(load_rows(csv_path)); kept_labels.append(label)
            fp_path = os.path.join(run_dir, "robustness.fingerprint.json")
            if os.path.exists(fp_path):
                import json
                fingerprints[label] = json.load(open(fp_path))["fingerprint"]
        else:
            print(f"[plot] {csv_path} missing -> skipping {label} in the robustness figure")
    if len(set(fingerprints.values())) > 1:
        print("[plot] WARNING: run dirs were recorded with DIFFERENT robustness condition "
              "tables (fingerprints differ) — curves share axes only where values overlap "
              "and are NOT route-paired; re-run with a common --dr-from for a fair comparison:")
        for label, fp in fingerprints.items():
            print(f"[plot]   {label}: {fp[:12]}")
    if experiments:
        robustness_figure(experiments, kept_labels,
                          os.path.join(args.out_dir, "robustness_compare.png"))

    # speed + route figures (capability.csv and the speed-pairs sidecar)
    experiments, kept_labels, pair_data = [], [], {}
    for run_dir, label in zip(args.run_dirs, labels):
        csv_path = os.path.join(run_dir, "capability.csv")
        if not os.path.exists(csv_path):
            print(f"[plot] {csv_path} missing -> skipping {label} in the capability figures")
            continue
        experiments.append(load_rows(csv_path)); kept_labels.append(label)
        pairs_path = os.path.join(run_dir, "capability_speed_pairs.csv")
        if os.path.exists(pairs_path):
            pair_data[label] = load_speed_pairs(pairs_path)
    if experiments:
        speed_figure(experiments, kept_labels, pair_data,
                     os.path.join(args.out_dir, "speed_compare.png"))
        route_figure(experiments, kept_labels,
                     os.path.join(args.out_dir, "route_compare.png"))
        uturn_figure(experiments, kept_labels,
                     os.path.join(args.out_dir, "uturn_compare.png"))

    # per-experiment control trace figures (capability_speed_traces.csv)
    for run_dir, label in zip(args.run_dirs, labels):
        traces_path = os.path.join(run_dir, "capability_speed_traces.csv")
        if os.path.exists(traces_path):
            safe = "".join(ch if ch.isalnum() else "_" for ch in label)
            speed_control_traces_figure(load_speed_traces(traces_path), label,
                                        os.path.join(args.out_dir,
                                                     f"speed_traces_{safe}.png"))


if __name__ == "__main__":
    main()
