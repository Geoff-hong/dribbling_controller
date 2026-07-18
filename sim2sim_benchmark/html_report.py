"""Self-contained interactive HTML comparison report (tensorboard-style).

  python -m sim2sim_benchmark.html_report \
      --run-dirs sim2sim_eval_results/runs/m80000 sim2sim_eval_results/runs/m90000 \
      --labels iter80000 iter90000 \
      --out sim2sim_eval_results/compare/report.html

One HTML file, no external assets: experiment checkboxes in the sidebar select
which runs are drawn; sections mirror the PNG figures (robustness axes, corner
turn, human dribble, u-turn, speed, control traces) plus a per-condition video
index (links resolve relative to the report location, so keep the report under
sim2sim_eval_results/compare/ with runs under sim2sim_eval_results/runs/).
All aggregation happens here in Python; the embedded JS only toggles and draws.
"""
import argparse
import csv
import json
import os

import numpy as np

PALETTE = ["#d62728", "#1f77b4", "#555555", "#2ca02c", "#9467bd", "#ff7f0e",
           "#8c564b", "#e377c2"]

ROB_METRICS = [("survival", "survival rate (%)"), ("possession", "ball possession (%)"),
               ("speed_ratio", "speed ratio (achieved/cmd)"), ("cross_track", "cross-track (m, survivors)")]
ROB_GROUPS = [("dr_scale", "DR scale alpha"), ("base_push", "base push dv (m/s)"),
              ("ball_push", "ball push dv (m/s)"), ("obs_latency", "ball-obs latency (steps)"),
              ("act_latency", "action latency (ms)")]
CAP_METRICS = [("success", "success rate (%)"), ("survival", "survival rate (%)"),
               ("cross_track", "cross-track (m, survivors)")]


def read_rows(path):
    if not os.path.exists(path):
        return []
    out = []
    for r in csv.DictReader(open(path)):
        out.append(dict(
            condition=r["condition"], group=r["group"], axis=float(r["axis_value"]),
            fell=float(r["fell"]), ball_lost=float(r["ball_lost"]),
            success=float(r["success"]) if r["success"] else None,
            ach=float(r["ach_speed_mps"]) if r["ach_speed_mps"] else None,
            cmd=float(r["cmd_speed_mps"]) if r["cmd_speed_mps"] else None,
            ct=float(r["cross_track_m"]) if r["cross_track_m"] else None,
            r=float(r["speed_corr_r"]) if r["speed_corr_r"] else None))
    return out


def finite(values):
    return [v for v in values if v is not None and np.isfinite(v)]


def condition_stats(rows):
    """rows of one condition -> point stats used by every panel."""
    surv = 100.0 * (1.0 - np.mean([r["fell"] for r in rows]))
    poss = 100.0 * (1.0 - np.mean([r["ball_lost"] for r in rows]))
    succ_vals = finite([r["success"] for r in rows])
    succ = 100.0 * np.mean(succ_vals) if succ_vals else None
    ratios = [r["ach"] / r["cmd"] for r in rows
              if r["ach"] is not None and r["cmd"] not in (None, 0) and r["cmd"] > 0.05]
    ratio = float(np.mean(ratios)) if ratios else None
    ct_vals = finite([r["ct"] for r in rows if r["fell"] < 0.5])
    ct = float(np.mean(ct_vals)) if ct_vals else None
    ach_vals = finite([r["ach"] for r in rows])
    ach = float(np.mean(ach_vals)) if ach_vals else None
    return dict(survival=round(surv, 2), possession=round(poss, 2),
                success=None if succ is None else round(succ, 2),
                speed_ratio=None if ratio is None else round(ratio, 4),
                cross_track=None if ct is None else round(ct, 4),
                ach_speed=None if ach is None else round(ach, 4))


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
    """capability_speed_pairs.csv -> binned cmd-vs-actual curve + pooled r."""
    if not os.path.exists(path):
        return None
    cmd, act = [], []
    for r in csv.DictReader(open(path)):
        cmd.append(float(r["cmd_speed_mps"])); act.append(float(r["ball_speed_mps"]))
    cmd = np.array(cmd); act = np.array(act)
    if len(cmd) < 100 or cmd.std() < 1e-3:
        return None
    r = float(np.corrcoef(cmd, act)[0, 1])
    edges = np.linspace(cmd.min(), cmd.max(), nbins + 1)
    pts = []
    for i in range(nbins):
        m = (cmd >= edges[i]) & (cmd < edges[i + 1] if i < nbins - 1 else cmd <= edges[i + 1])
        if m.sum() >= 20:
            pts.append(dict(x=round(float(0.5 * (edges[i] + edges[i + 1])), 4),
                            y=round(float(act[m].mean()), 4)))
    return dict(r=round(r, 3), points=pts)


def traces(path, smooth_steps=25, keep_every=5):
    """capability_speed_traces.csv -> per-episode downsampled cmd + smoothed
    along-command speed (50 Hz -> 10 Hz after a 0.5 s moving average)."""
    if not os.path.exists(path):
        return None
    eps = {}
    for r in csv.DictReader(open(path)):
        eps.setdefault(int(r["episode"]), []).append(
            (int(r["step"]), float(r["cmd_speed_mps"]), float(r["ball_speed_along_cmd_mps"])))
    out = {}
    for ep, items in sorted(eps.items()):
        items.sort()
        cmd = np.array([i[1] for i in items]); along = np.array([i[2] for i in items])
        k = np.ones(smooth_steps) / smooth_steps
        along_s = np.convolve(along, k, mode="same")
        out[str(ep)] = dict(
            dt=0.02 * keep_every,
            cmd=[round(float(v), 3) for v in cmd[::keep_every]],
            act=[round(float(v), 3) for v in along_s[::keep_every]])
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


def collect_run(run_dir, label, color, report_dir):
    rob = read_rows(os.path.join(run_dir, "robustness.csv"))
    cap = read_rows(os.path.join(run_dir, "capability.csv"))
    nominal = [r for r in rob if r["group"] == "baseline"]
    data = dict(
        label=label, color=color,
        nominal=condition_stats(nominal) if nominal else None,
        robustness={g: group_series(rob, g) for g, _ in ROB_GROUPS},
        straight=group_series(cap, "straight_speed"),
        corner=group_series(cap, "corner_turn", split_sign=True),
        human=group_series(cap, "human_dribble"),
        uturn=group_series(cap, "u_turn", split_sign=True),
        tracking=group_series(cap, "speed_tracking"),
        pairs=binned_pairs(os.path.join(run_dir, "capability_speed_pairs.csv")),
        traces=traces(os.path.join(run_dir, "capability_speed_traces.csv")),
        videos=video_index(run_dir, report_dir))
    return data


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>sim2sim benchmark report</title>
<style>
  :root { --bg:#fafafa; --panel:#ffffff; --border:#dddddd; --text:#222222; --muted:#777777; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
         background:var(--bg); color:var(--text); }
  #layout { display:flex; min-height:100vh; }
  #sidebar { width:250px; padding:14px; border-right:1px solid var(--border);
             background:var(--panel); position:sticky; top:0; height:100vh; overflow-y:auto;
             flex-shrink:0; }
  #sidebar h1 { font-size:15px; margin:0 0 12px; }
  #sidebar h2 { font-size:12px; text-transform:uppercase; color:var(--muted); margin:16px 0 6px; }
  .runrow { display:flex; align-items:center; gap:7px; padding:3px 0; font-size:13px; cursor:pointer; }
  .swatch { width:14px; height:14px; border-radius:3px; flex-shrink:0; }
  .navlink { display:block; font-size:13px; color:#2159a8; text-decoration:none; padding:2px 0; }
  #main { flex:1; padding:18px 24px; min-width:0; }
  section { margin-bottom:34px; }
  section > h2 { font-size:17px; border-bottom:1px solid var(--border); padding-bottom:6px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(340px, 1fr)); gap:14px; }
  .panel { background:var(--panel); border:1px solid var(--border); border-radius:6px; padding:8px 10px 4px; }
  .panel h3 { font-size:12.5px; margin:2px 0 4px; font-weight:600; color:#333; }
  svg text { font-size:10px; fill:#555; }
  svg .axisline { stroke:#999; stroke-width:1; }
  svg .gridline { stroke:#eee; stroke-width:1; }
  table.videos { border-collapse:collapse; font-size:12.5px; }
  table.videos th, table.videos td { border:1px solid var(--border); padding:3px 8px; text-align:left; }
  table.videos th { background:#f0f0f0; position:sticky; top:0; }
  table.videos a { color:#2159a8; text-decoration:none; }
  .note { font-size:12px; color:var(--muted); margin:4px 0 10px; }
  .rbadge { display:inline-block; padding:1px 7px; border-radius:9px; color:#fff; font-size:11.5px; margin-right:6px;}
</style>
</head>
<body>
<div id="layout">
  <nav id="sidebar">
    <h1>sim2sim benchmark</h1>
    <h2>Experiments</h2>
    <div id="runboxes"></div>
    <h2>Sections</h2>
    <a class="navlink" href="#sec-robustness">Robustness</a>
    <a class="navlink" href="#sec-corner">Corner turn</a>
    <a class="navlink" href="#sec-human">Human dribble</a>
    <a class="navlink" href="#sec-uturn">U-turn</a>
    <a class="navlink" href="#sec-speed">Speed</a>
    <a class="navlink" href="#sec-traces">Control traces</a>
    <a class="navlink" href="#sec-videos">Videos</a>
    <h2>Episodes</h2>
    <div class="note">__META__</div>
  </nav>
  <main id="main">
    <section id="sec-robustness"><h2>Robustness — perturbation axes, nominal human routes (20 s)</h2>
      <div class="note">dotted horizontal line = each experiment's nominal baseline</div>
      <div class="grid" id="rob-grid"></div></section>
    <section id="sec-corner"><h2>Capability — corner turn (solid = L, dashed = R)</h2>
      <div class="grid" id="corner-grid"></div></section>
    <section id="sec-human"><h2>Capability — human dribble (kappa-cap sweep, 20 s fail-fast)</h2>
      <div class="grid" id="human-grid"></div></section>
    <section id="sec-uturn"><h2>Capability — u-turn about-face (solid = L, dashed = R)</h2>
      <div class="grid" id="uturn-grid"></div></section>
    <section id="sec-speed"><h2>Capability — speed</h2>
      <div id="track-badges" class="note"></div>
      <div class="grid" id="speed-grid"></div></section>
    <section id="sec-traces"><h2>Control traces — speed_tracking rep 0-7 (dashed = commanded)</h2>
      <div class="grid" id="traces-grid"></div></section>
    <section id="sec-videos"><h2>Per-condition videos</h2>
      <div class="note">one mp4 per condition: the rep-0 episode, chase camera (links open the local file)</div>
      <div id="videos-host"></div></section>
  </main>
</div>
<script>
const DATA = __DATA__;
const ROB_GROUPS = __ROB_GROUPS__;
const ROB_METRICS = __ROB_METRICS__;
const CAP_METRICS = __CAP_METRICS__;
const enabled = DATA.map(() => true);

function visibleRuns() { return DATA.filter((_, i) => enabled[i]); }

function makeSVG(w, h) {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", `0 0 ${w} ${h}`);
  s.setAttribute("width", "100%");
  return s;
}
function el(tag, attrs, parent) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) { hi = lo + 1; }
  const span = hi - lo, step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || mag * 10;
  const t0 = Math.ceil(lo / step) * step, out = [];
  for (let t = t0; t <= hi + 1e-9; t += step) out.push(+t.toFixed(10));
  return out;
}

// series: [{x[], y[], color, dash, label}], opts: {yDomain, xLabel, hlines:[{y,color}]}
function lineChart(host, seriesList, opts = {}) {
  const W = 340, H = 200, m = {l: 42, r: 10, t: 8, b: 30};
  const svg = makeSVG(W, H);
  const xs = seriesList.flatMap(s => s.x), ysAll = seriesList.flatMap(s => s.y).filter(v => v != null);
  if (!xs.length || !ysAll.length) { host.appendChild(svg); return; }
  let xlo = Math.min(...xs), xhi = Math.max(...xs);
  if (xlo === xhi) { xlo -= 0.5; xhi += 0.5; }
  let [ylo, yhi] = opts.yDomain || [Math.min(0, Math.min(...ysAll)), Math.max(...ysAll) * 1.08 + 1e-6];
  const X = v => m.l + (v - xlo) / (xhi - xlo) * (W - m.l - m.r);
  const Y = v => H - m.b - (v - ylo) / (yhi - ylo) * (H - m.t - m.b);
  for (const t of niceTicks(ylo, yhi)) {
    el("line", {x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "gridline"}, svg);
    const txt = el("text", {x: m.l - 5, y: Y(t) + 3, "text-anchor": "end"}, svg);
    txt.textContent = +t.toFixed(3);
  }
  for (const t of niceTicks(xlo, xhi)) {
    const txt = el("text", {x: X(t), y: H - m.b + 14, "text-anchor": "middle"}, svg);
    txt.textContent = +t.toFixed(3);
  }
  el("line", {x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axisline"}, svg);
  el("line", {x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, class: "axisline"}, svg);
  if (opts.xLabel) {
    const t = el("text", {x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle"}, svg);
    t.textContent = opts.xLabel;
  }
  for (const hl of opts.hlines || []) {
    if (hl.y == null || hl.y < ylo || hl.y > yhi) continue;
    el("line", {x1: m.l, x2: W - m.r, y1: Y(hl.y), y2: Y(hl.y), stroke: hl.color,
                "stroke-dasharray": "2,3", "stroke-width": 1, opacity: 0.7}, svg);
  }
  for (const s of seriesList) {
    const pts = s.x.map((x, i) => [x, s.y[i]]).filter(p => p[1] != null);
    if (!pts.length) continue;
    const d = pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    el("path", {d, fill: "none", stroke: s.color, "stroke-width": 1.8,
                ...(s.dash ? {"stroke-dasharray": "5,4"} : {})}, svg);
    for (const p of pts) el("circle", {cx: X(p[0]), cy: Y(p[1]), r: 2.4, fill: s.color}, svg);
  }
  host.appendChild(svg);
}

function panel(host, title) {
  const d = document.createElement("div");
  d.className = "panel";
  const h = document.createElement("h3");
  h.textContent = title;
  d.appendChild(h);
  host.appendChild(d);
  return d;
}
function pick(pointList, metric) {
  return {x: pointList.map(p => p.x), y: pointList.map(p => p[metric])};
}
function metricDomain(metric) {
  return (metric === "survival" || metric === "possession" || metric === "success")
    ? [0, 102] : null;
}

function renderRobustness() {
  const g = document.getElementById("rob-grid");
  g.innerHTML = "";
  for (const [group, gLabel] of ROB_GROUPS)
    for (const [metric, mLabel] of ROB_METRICS) {
      const p = panel(g, `${gLabel} — ${mLabel}`);
      const series = [], hlines = [];
      for (const run of visibleRuns()) {
        const pts = run.robustness[group] || [];
        series.push({...pick(pts, metric), color: run.color});
        if (run.nominal && run.nominal[metric] != null)
          hlines.push({y: run.nominal[metric], color: run.color});
      }
      lineChart(p, series, {yDomain: metricDomain(metric), xLabel: gLabel, hlines});
    }
}

function renderTurns(gridId, key, xLabel) {
  const g = document.getElementById(gridId);
  g.innerHTML = "";
  for (const [metric, mLabel] of CAP_METRICS) {
    const p = panel(g, mLabel);
    const series = [];
    for (const run of visibleRuns()) {
      const d = run[key];
      if (Array.isArray(d)) {                       // human dribble: plain series
        series.push({...pick(d, metric), color: run.color});
      } else if (d) {                               // corner / u-turn: L solid, R dashed
        series.push({...pick(d.L, metric), color: run.color});
        series.push({...pick(d.R, metric), color: run.color, dash: true});
      }
    }
    lineChart(p, series, {yDomain: metricDomain(metric), xLabel});
  }
}

function renderSpeed() {
  const g = document.getElementById("speed-grid");
  g.innerHTML = "";
  for (const [metric, mLabel] of [["success", "max speed: success rate (%)"],
                                  ["survival", "max speed: survival rate (%)"],
                                  ["ach_speed", "achieved vs commanded (m/s)"]]) {
    const p = panel(g, mLabel);
    const series = visibleRuns().map(run => ({...pick(run.straight, metric), color: run.color}));
    if (metric === "ach_speed") {
      const xs = series.flatMap(s => s.x);
      if (xs.length) series.push({x: [Math.min(...xs), Math.max(...xs)],
                                  y: [Math.min(...xs), Math.max(...xs)],
                                  color: "#aaa", dash: true});
    }
    lineChart(p, series, {yDomain: metricDomain(metric), xLabel: "commanded speed (m/s), straight"});
  }
  const p = panel(g, "controllability: binned cmd vs actual (human routes)");
  const series = [];
  for (const run of visibleRuns())
    if (run.pairs) series.push({x: run.pairs.points.map(q => q.x),
                                y: run.pairs.points.map(q => q.y), color: run.color});
  const xs = series.flatMap(s => s.x);
  if (xs.length) series.push({x: [Math.min(...xs), Math.max(...xs)],
                              y: [Math.min(...xs), Math.max(...xs)], color: "#aaa", dash: true});
  lineChart(p, series, {xLabel: "commanded speed (m/s)"});
  const b = document.getElementById("track-badges");
  b.innerHTML = "pooled Pearson r: " + visibleRuns().map(run =>
    `<span class="rbadge" style="background:${run.color}">${run.label}: ${run.pairs ? run.pairs.r : "-"}</span>`).join("");
}

function renderTraces() {
  const g = document.getElementById("traces-grid");
  g.innerHTML = "";
  for (let ep = 0; ep < 8; ep++) {
    const key = String(ep);
    const runsWith = visibleRuns().filter(r => r.traces && r.traces[key]);
    if (!runsWith.length) continue;
    const p = panel(g, `episode ${ep}`);
    const series = [];
    const first = runsWith[0].traces[key];
    const t = first.cmd.map((_, i) => +(i * first.dt).toFixed(2));
    series.push({x: t, y: first.cmd, color: "#333", dash: true});
    for (const run of runsWith) {
      const tr = run.traces[key];
      series.push({x: tr.act.map((_, i) => +(i * tr.dt).toFixed(2)), y: tr.act, color: run.color});
    }
    lineChart(p, series, {xLabel: "t (s)"});
  }
}

function renderVideos() {
  const host = document.getElementById("videos-host");
  host.innerHTML = "";
  const tests = [...new Set(visibleRuns().flatMap(r => Object.keys(r.videos)))].sort();
  for (const test of tests) {
    const runsWith = visibleRuns().filter(r => r.videos[test]);
    if (!runsWith.length) continue;
    const conds = [...new Set(runsWith.flatMap(r => Object.keys(r.videos[test])))].sort();
    const h = document.createElement("h3"); h.textContent = test; host.appendChild(h);
    const tb = document.createElement("table"); tb.className = "videos";
    tb.innerHTML = "<tr><th>condition</th>" +
      runsWith.map(r => `<th style="color:${r.color}">${r.label}</th>`).join("") + "</tr>" +
      conds.map(c => "<tr><td>" + c + "</td>" + runsWith.map(r => {
        const v = r.videos[test][c];
        return `<td>${v ? `<a href="${v}" target="_blank">▶ play</a>` : "-"}</td>`;
      }).join("") + "</tr>").join("");
    host.appendChild(tb);
  }
}

function renderAll() {
  renderRobustness();
  renderTurns("corner-grid", "corner", "|kappa| (1/m)");
  renderTurns("human-grid", "human", "route_human_kappa_cap");
  renderTurns("uturn-grid", "uturn", "|kappa| (1/m)");
  renderSpeed();
  renderTraces();
  renderVideos();
}

const boxes = document.getElementById("runboxes");
DATA.forEach((run, i) => {
  const row = document.createElement("label");
  row.className = "runrow";
  row.innerHTML = `<input type="checkbox" checked><span class="swatch" style="background:${run.color}"></span>${run.label}`;
  row.querySelector("input").addEventListener("change", e => { enabled[i] = e.target.checked; renderAll(); });
  boxes.appendChild(row);
});
renderAll();
</script>
</body>
</html>
"""


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
    args = ap.parse_args()
    if args.run_dirs is None:
        args.run_dirs = sorted(
            os.path.join(args.runs_root, d) for d in os.listdir(args.runs_root)
            if os.path.exists(os.path.join(args.runs_root, d, "robustness.csv"))
            or os.path.exists(os.path.join(args.runs_root, d, "capability.csv")))
        if not args.run_dirs:
            ap.error(f"no runs with CSVs found under {args.runs_root}")
        print(f"[html_report] discovered {len(args.run_dirs)} runs under {args.runs_root}")
    if args.labels is None:
        args.labels = [os.path.basename(os.path.normpath(d)) for d in args.run_dirs]
    if len(args.run_dirs) != len(args.labels):
        ap.error("--labels must match --run-dirs")

    report_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(report_dir, exist_ok=True)
    runs = [collect_run(d, lab, PALETTE[i % len(PALETTE)], report_dir)
            for i, (d, lab) in enumerate(zip(args.run_dirs, args.labels))]

    n_rob = sum(1 for _ in open(os.path.join(args.run_dirs[0], "robustness.csv"))) - 1 \
        if os.path.exists(os.path.join(args.run_dirs[0], "robustness.csv")) else 0
    meta = (f"{len(runs)} experiments; paired episodes per condition "
            f"(first run: {n_rob} robustness rows)")
    html = (HTML_TEMPLATE
            .replace("__DATA__", json.dumps(runs, separators=(",", ":")))
            .replace("__ROB_GROUPS__", json.dumps(ROB_GROUPS))
            .replace("__ROB_METRICS__", json.dumps(ROB_METRICS))
            .replace("__CAP_METRICS__", json.dumps(CAP_METRICS))
            .replace("__META__", meta))
    with open(args.out, "w") as f:
        f.write(html)
    print(f"[html_report] wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, "
          f"{len(runs)} experiments)")


if __name__ == "__main__":
    main()
