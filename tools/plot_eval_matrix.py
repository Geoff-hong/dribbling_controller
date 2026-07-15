#!/usr/bin/env python3
"""Plot the eval matrix from one or more `dribble_pysim_multi.py --matrix` CSVs.

Layout (the agreed format): columns = X axes (condition groups), rows = Y
metrics (survival, ball possession, speed ratio, cross-track). Each CSV is one
EXPERIMENT and gets one color; its curves are overlaid in every panel, so
experiments are compared directly inside the same axes.

  python tools/plot_eval_matrix.py \
      --csv eval_result/m80000/matrix.csv eval_result/m90000/matrix.csv \
      --labels iter80000 iter90000 --out eval_result/matrix_compare.png

The arc_kappa group encodes turn direction in the sign of axis_value: left
(+kappa) is drawn solid, right (-kappa) dashed, sharing the experiment color.
The baseline group is drawn as a horizontal dotted reference line per metric.
No pandas dependency; numpy + matplotlib only.
"""
import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUP_ORDER = ["dr_scale", "base_push", "ball_push", "obs_latency", "act_latency",
               "straight_speed", "arc_kappa"]
GROUP_LABEL = {
    "dr_scale": "DR scale alpha",
    "base_push": "base push dv (m/s)",
    "ball_push": "ball push dv (m/s)",
    "obs_latency": "ball-obs latency (steps)",
    "act_latency": "action latency (ms)",
    "straight_speed": "cmd speed v, STRAIGHT (m/s)",
    "arc_kappa": "|kappa|, corner turn 150-180deg (1/m)",
}
CAPABILITY = {"straight_speed", "arc_kappa"}
ROWS = ["survival % (arc: success %)", "ball possession (%)", "speed ratio (achieved/cmd)",
        "cross-track (m, survivors)"]
COLORS = ["tab:red", "tab:blue", "0.45", "tab:green", "tab:purple", "tab:orange"]


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            def num(key):
                v = r.get(key)
                return float(v) if v not in (None, "") else float("nan")
            rows.append(dict(group=r["group"], axis=float(r["axis_value"]),
                             fell=float(r["fell"]), lost=float(r["ball_lost"]),
                             ct=num("cross_track_m"),
                             ach=num("ach_speed_mps"), cmd=num("cmd_speed_mps"),
                             succ=num("success")))
    return rows


def level_stats(rows):
    """-> {axis_value: dict(n, surv, surv_se, poss, poss_se, ratio, ct)} (ct over survivors)."""
    by = {}
    for r in rows:
        by.setdefault(r["axis"], []).append(r)
    out = {}
    for ax_v, rs in sorted(by.items()):
        n = len(rs)
        surv = 1.0 - np.mean([r["fell"] for r in rs])
        poss = 1.0 - np.mean([r["lost"] for r in rs])
        ratios = [r["ach"] / r["cmd"] for r in rs
                  if np.isfinite(r["ach"]) and np.isfinite(r["cmd"]) and r["cmd"] > 0.05]
        ratio = float(np.mean(ratios)) if ratios else float("nan")
        alive = [r["ct"] for r in rs if r["fell"] < 0.5 and np.isfinite(r["ct"])]
        succs = [r["succ"] for r in rs if np.isfinite(r["succ"])]
        succ = float(np.mean(succs)) if succs else float("nan")
        ns = max(1, len(succs))
        out[ax_v] = dict(
            n=n, surv=surv, poss=poss, ratio=ratio,
            surv_se=np.sqrt(max(surv * (1 - surv), 1e-4 / n) / n),
            poss_se=np.sqrt(max(poss * (1 - poss), 1e-4 / n) / n),
            succ=succ, succ_se=np.sqrt(max(succ * (1 - succ), 1e-4 / ns) / ns) if succs else float("nan"),
            ct=float(np.mean(alive)) if alive else np.nan)
    return out


def draw(ax_panel, row, x, st, color, ls, label=None, use_abs=False, row0="surv"):
    x = list(x)
    if row == 0:
        y = 100 * np.array([st[v][row0] for v in x]); se = 100 * np.array([st[v][f"{row0}_se"] for v in x])
    elif row == 1:
        y = 100 * np.array([st[v]["poss"] for v in x]); se = 100 * np.array([st[v]["poss_se"] for v in x])
    elif row == 2:
        y = np.array([st[v]["ratio"] for v in x]); se = None
    else:
        y = np.array([st[v]["ct"] for v in x]); se = None
    xs = np.abs(np.asarray(x, float)) if use_abs else np.asarray(x, float)
    ax_panel.plot(xs, y, marker="o", ms=3.5, lw=1.7, color=color, ls=ls, label=label)
    if se is not None:
        ax_panel.fill_between(xs, y - se, y + se, color=color, alpha=0.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="one matrix CSV per experiment")
    ap.add_argument("--labels", nargs="*", default=None, help="one label per CSV (default: dirname)")
    ap.add_argument("--out", default="eval_matrix.png")
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(os.path.dirname(os.path.abspath(p))) or p for p in args.csv]
    assert len(labels) == len(args.csv), "--labels must match --csv"

    exps = [load(p) for p in args.csv]
    groups = [g for g in GROUP_ORDER if any(r["group"] == g for e in exps for r in e)]
    extra = sorted({r["group"] for e in exps for r in e} - set(groups) - {"baseline"})
    groups += extra
    if not groups:
        raise SystemExit("no plottable groups found in the CSVs")

    fig, axg = plt.subplots(4, len(groups), figsize=(3.4 * len(groups) + 1, 12),
                            sharey="row", squeeze=False)
    labeled = set()   # (experiment idx, style tag) already carries a legend label
    for c, g in enumerate(groups):
        for ei, (rows, lab) in enumerate(zip(exps, labels)):
            color = COLORS[ei % len(COLORS)]
            sub = [r for r in rows if r["group"] == g]
            base = [r for r in rows if r["group"] == "baseline"]
            if not sub:
                continue
            if g == "arc_kappa":
                for sgn, ls, tag in ((1, "-", "L"), (-1, "--", "R")):
                    ss = [r for r in sub if (r["axis"] >= 0) == (sgn > 0)]
                    if not ss:
                        continue
                    # corner-turn conditions carry a success verdict -> row 0 shows
                    # success rate; older CSVs without it fall back to survival
                    row0 = "succ" if any(np.isfinite(r["succ"]) for r in ss) else "surv"
                    st = level_stats(ss); x = sorted(st)
                    for row in range(4):
                        lbl = None
                        if row == 0 and (ei, tag) not in labeled:
                            lbl = f"{lab} ({tag})"; labeled.add((ei, tag))
                        draw(axg[row, c], row, x, st, color, ls, label=lbl, use_abs=True, row0=row0)
            else:
                st = level_stats(sub); x = sorted(st)
                for row in range(4):
                    lbl = None
                    if row == 0 and (ei, "") not in labeled:
                        lbl = lab; labeled.add((ei, ""))
                    draw(axg[row, c], row, x, st, color, "-", label=lbl)
            # baseline (nominal) reference per experiment
            if base:
                bs = level_stats(base); bv = list(bs.values())[0]
                for row, val in ((0, 100 * bv["surv"]), (1, 100 * bv["poss"]), (3, bv["ct"])):
                    axg[row, c].axhline(val, color=color, ls=":", lw=0.9, alpha=0.55)
        for row in range(4):
            ax = axg[row, c]
            ax.grid(alpha=0.3)
            if row in (0, 1):
                ax.set_ylim(-5, 105)
            elif row == 2:
                ax.set_ylim(0, 1.15)
            if row == 0:
                ax.set_title(GROUP_LABEL.get(g, g), fontsize=10.5,
                             color="tab:red" if g in CAPABILITY else "black")
            if row == 3:
                ax.set_xlabel(GROUP_LABEL.get(g, g), fontsize=8.5)
            if c == 0:
                ax.set_ylabel(ROWS[row], fontsize=10)
    seen = {}   # label -> handle, deduped across the whole top row
    for c in range(len(groups)):
        for h, l in zip(*axg[0, c].get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(list(seen.values()), list(seen.keys()), loc="lower center",
               ncol=max(2, len(labels) + 2), fontsize=10, frameon=False)
    fig.suptitle("Eval matrix — columns = X axes, rows = Y metrics; one color per experiment "
                 "(dotted = that experiment's nominal baseline; arc: solid = left, dashed = right)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(args.out, dpi=115)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
