"""Comparison figures from benchmark CSVs (engine-independent; numpy+matplotlib only).

  python -m sim2sim_benchmark.plot \
      --run-dirs eval_result/m80000 eval_result/m90000 \
      --labels iter80000 iter90000 --out-dir eval_result

Each run dir is one EXPERIMENT (one color) holding robustness.csv and/or
capability.csv. Outputs robustness_compare.png / capability_compare.png:
columns = test axes, rows = metrics, experiments overlaid as colored curves
(corner turns: solid = left, dashed = right; dotted = nominal baseline).
"""
import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROBUSTNESS_GROUPS = ["dr_scale", "base_push", "ball_push", "obs_latency", "act_latency"]
CAPABILITY_GROUPS = ["straight_speed", "corner_turn"]
GROUP_LABEL = {
    "dr_scale": "DR scale alpha",
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
                             fell=float(record["fell"]), lost=float(record["ball_lost"]),
                             cross_track=num("cross_track_m"), ach_speed=num("ach_speed_mps"),
                             cmd_speed=num("cmd_speed_mps"), success=num("success")))
    return rows


def level_stats(rows):
    """axis value -> survival/possession/success rates (+binomial SE), speed ratio,
    survivor cross-track."""
    by_axis = {}
    for row in rows:
        by_axis.setdefault(row["axis"], []).append(row)
    stats = {}
    for axis_value, level_rows in sorted(by_axis.items()):
        n = len(level_rows)

        def rate(values):
            values = [v for v in values if np.isfinite(v)]
            if not values:
                return float("nan"), float("nan")
            p = 1.0 - float(np.mean(values))
            return p, np.sqrt(max(p * (1 - p), 1e-4 / len(values)) / len(values))

        survival, survival_se = rate([r["fell"] for r in level_rows])
        possession, possession_se = rate([r["lost"] for r in level_rows])
        success, success_se = rate([1.0 - r["success"] for r in level_rows
                                    if np.isfinite(r["success"])] or [float("nan")])
        speed_ratios = [r["ach_speed"] / r["cmd_speed"] for r in level_rows
                        if np.isfinite(r["ach_speed"]) and np.isfinite(r["cmd_speed"])
                        and r["cmd_speed"] > 0.05]
        survivor_ct = [r["cross_track"] for r in level_rows
                       if r["fell"] < 0.5 and np.isfinite(r["cross_track"])]
        stats[axis_value] = dict(
            n=n, survival=survival, survival_se=survival_se,
            possession=possession, possession_se=possession_se,
            success=success, success_se=success_se,
            speed_ratio=float(np.mean(speed_ratios)) if speed_ratios else float("nan"),
            cross_track=float(np.mean(survivor_ct)) if survivor_ct else float("nan"))
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


def _style_panel(panel, row_metric, group, is_top, is_bottom, is_left, row_label):
    panel.grid(alpha=0.3)
    if row_metric in ("survival", "possession", "success"):
        panel.set_ylim(-5, 105)
    elif row_metric == "speed_ratio":
        panel.set_ylim(0, 1.15)
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
    """Rows: survival / possession / speed ratio / cross-track. Dotted horizontal
    lines = each experiment's nominal baseline."""
    metrics = [("survival", "survival rate (%)"), ("possession", "ball possession (%)"),
               ("speed_ratio", "speed ratio (achieved/cmd)"),
               ("cross_track", "cross-track (m, survivors)")]
    groups = [g for g in ROBUSTNESS_GROUPS if any(r["group"] == g for e in experiments for r in e)]
    groups += sorted({r["group"] for e in experiments for r in e}
                     - set(groups) - {"baseline"} - set(CAPABILITY_GROUPS))
    if not groups:
        return False
    fig, panels = plt.subplots(len(metrics), len(groups),
                               figsize=(3.4 * len(groups) + 1, 12), sharey="row", squeeze=False)
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
            _style_panel(panels[row, col], metric, group, row == 0,
                         row == len(metrics) - 1, col == 0, row_label)
    _collect_legend(fig, [panels[0, c] for c in range(len(groups))], len(labels))
    fig.suptitle("Sim2sim benchmark — ROBUSTNESS: perturbation axes on nominal routes; "
                 "one color per experiment, dotted = its nominal baseline", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_path, dpi=115)
    print(f"saved {out_path}")
    return True


def capability_figure(experiments, labels, out_path):
    """Rows: success rate / speed ratio / cross-track. Corner turns: solid = left,
    dashed = right."""
    metrics = [("success", "success rate (%)"), ("speed_ratio", "speed ratio (achieved/cmd)"),
               ("cross_track", "cross-track (m, survivors)")]
    groups = [g for g in CAPABILITY_GROUPS if any(r["group"] == g for e in experiments for r in e)]
    if not groups:
        return False
    fig, panels = plt.subplots(len(metrics), len(groups),
                               figsize=(4.6 * len(groups) + 1, 9.5), sharey="row", squeeze=False)
    labeled = set()
    for col, group in enumerate(groups):
        for exp_index, (rows, label) in enumerate(zip(experiments, labels)):
            color = EXPERIMENT_COLORS[exp_index % len(EXPERIMENT_COLORS)]
            group_rows = [r for r in rows if r["group"] == group]
            if not group_rows:
                continue
            if group == "corner_turn":
                directions = ((1, "-", "L"), (-1, "--", "R"))
            else:
                directions = ((1, "-", ""),)
            for sign, linestyle, tag in directions:
                side_rows = [r for r in group_rows if (r["axis"] >= 0) == (sign > 0)] \
                    if group == "corner_turn" else group_rows
                if not side_rows:
                    continue
                stats = level_stats(side_rows)
                x_values = sorted(stats)
                for row, (metric, _) in enumerate(metrics):
                    series_label = None
                    if row == 0 and (exp_index, tag) not in labeled:
                        series_label = f"{label} ({tag})" if tag else label
                        labeled.add((exp_index, tag))
                    draw_series(panels[row, col], stats, x_values, metric, color,
                                linestyle, series_label, use_abs=(group == "corner_turn"))
        for row, (metric, row_label) in enumerate(metrics):
            _style_panel(panels[row, col], metric, group, row == 0,
                         row == len(metrics) - 1, col == 0, row_label)
    _collect_legend(fig, [panels[0, c] for c in range(len(groups))], len(labels))
    fig.suptitle("Sim2sim benchmark — CAPABILITY (10 s, fail if ball >0.8 m off route "
                 "or >1.2 m from robot); corner: solid = L, dashed = R", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
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

    for test, figure in (("robustness", robustness_figure), ("capability", capability_figure)):
        experiments, kept_labels = [], []
        for run_dir, label in zip(args.run_dirs, labels):
            csv_path = os.path.join(run_dir, f"{test}.csv")
            if os.path.exists(csv_path):
                experiments.append(load_rows(csv_path)); kept_labels.append(label)
            else:
                print(f"[plot] {csv_path} missing -> skipping {label} in the {test} figure")
        if experiments:
            figure(experiments, kept_labels, os.path.join(args.out_dir, f"{test}_compare.png"))
        else:
            print(f"[plot] no {test}.csv anywhere -> no {test} figure")


if __name__ == "__main__":
    main()
