#!/usr/bin/env python3
"""Regenerate the demo figures in this folder.

The figures preview what a real benchmark run produces — SAME plotting code,
SAME layout/axes/legends — but the numbers are synthetic placeholders (smooth
sigmoids + sampling noise for two fake experiments), NOT policy results.

Needs only numpy + matplotlib (the MuJoCo/onnxruntime imports of the engine are
stubbed), so it runs anywhere:

  python sim2sim_benchmark/demo/make_demo_figures.py
"""
import os
import sys
import tempfile
import types

import numpy as np

# the plot module never touches the simulator, but conditions.py imports the
# engine for the condition schema -> stub the sim deps so this runs anywhere
for module_name in ("mujoco", "mujoco.viewer", "onnxruntime"):
    sys.modules.setdefault(module_name, types.ModuleType(module_name))
sys.modules["mujoco"].viewer = sys.modules["mujoco.viewer"]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from sim2sim_benchmark.conditions import robustness_conditions, capability_conditions
from sim2sim_benchmark.report import write_csv, write_speed_pairs_csv, write_speed_traces_csv
from sim2sim_benchmark import plot

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
EPISODES_PER_CONDITION = 24

# where each fake experiment's survival/success cliff sits per group, and how
# steep it is — chosen only so the curves LOOK like plausible benchmark output
CLIFF_CENTER = dict(baseline=99, dr_scale=1.5, base_push=0.8, ball_push=1.8,
                    obs_latency=4.5, act_latency=28, straight_speed=2.8,
                    corner_turn=0.75, u_turn=3.0, human_dribble=0.85, speed_tracking=99)
CLIFF_STEEPNESS = dict(baseline=1, dr_scale=3.5, base_push=4.5, ball_push=2.5,
                       obs_latency=1.6, act_latency=0.18, straight_speed=3.5,
                       corner_turn=7, u_turn=1.4, human_dribble=4.0, speed_tracking=1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fake_control_trace(rng, vmax, gain, noise, steps=1000):
    """One fake 20 s / 50 Hz control trace: piecewise-constant command from the
    curvature law clipped at vmax; actual = lagged gain * cmd + kick spikes."""
    cmd = np.empty(steps)
    i = 0
    while i < steps:
        segment = int(rng.uniform(50, 150))   # 1-3 s per curvature segment
        level = min(vmax, float(rng.choice([2.0, 1.94, 1.58, 1.37, 1.22])))
        cmd[i:i + segment] = level
        i += segment
    lag = np.convolve(cmd, np.ones(40) / 40, mode="same")   # sluggish response
    along_cmd = gain * lag + rng.normal(0, noise, steps)
    kicks = np.abs(np.sin(np.linspace(0, steps / 30.0 * np.pi, steps))) * rng.uniform(0.3, 0.8)
    speed_abs = np.abs(along_cmd) + kicks * rng.uniform(0.5, 1.0, steps)
    return cmd, along_cmd, speed_abs


def write_fake_run(run_dir, cliff_shift, tracking_gain, tracking_noise, make_traces=True):
    """One fake experiment: robustness.csv + capability.csv (+ pairs & traces)."""
    os.makedirs(run_dir, exist_ok=True)
    rng = np.random.default_rng(abs(hash(run_dir)) % 2**32)
    for test, table in (("robustness", robustness_conditions()),
                        ("capability", capability_conditions())):
        rows, pair_rows, trace_rows = [], [], []
        for condition in table:
            group = condition["group"]
            axis = abs(condition["axis"])
            ok_prob = (0.97 if group in ("baseline", "speed_tracking") else
                       0.97 * sigmoid(-CLIFF_STEEPNESS[group]
                                      * (axis - (CLIFF_CENTER[group] + cliff_shift))))
            if group in ("corner_turn", "u_turn") and condition["axis"] < 0:
                ok_prob *= 0.85   # fake left/right asymmetry
            lost_prob = min(1.0, (1 - ok_prob) * 1.3
                            + (0.5 if group == "ball_push" and axis >= 1 else 0.02))
            fail_fast = condition["offroute_fail_m"] is not None
            cmd_speed = (condition["route_vmax"] if group == "straight_speed"
                         else (min(2.0, np.sqrt(0.75 / axis))
                               if group in ("corner_turn", "u_turn") else 1.7))
            for rep in range(EPISODES_PER_CONDITION):
                if fail_fast:
                    success = float(rng.random() < ok_prob)
                    reason = "" if success else rng.choice(["off_route", "ball_far", "fell"])
                    fell = float(reason == "fell")
                else:
                    success = float("nan")
                    fell = float(rng.random() > ok_prob)
                    reason = "fell" if fell else ""
                duration = ((10.0 if fail_fast else 20.0) if not reason
                            else float(rng.uniform(2, 9)))
                achieved = (min(cmd_speed * rng.uniform(0.85, 0.95), 2.3)
                            if group == "straight_speed"
                            else cmd_speed * rng.uniform(0.8, 0.95))
                corr = float("nan")
                if group == "speed_tracking":
                    vmax_cfg = condition["route_vmax"] or 2.0
                    vmax = (rng.uniform(*vmax_cfg)
                            if isinstance(vmax_cfg, (list, tuple)) else float(vmax_cfg))
                    cmds = np.minimum(vmax, 1.22 + 0.78 * rng.random(190))
                    actual = tracking_gain * cmds + rng.normal(0, tracking_noise, 190)
                    corr = (float(np.corrcoef(cmds, actual)[0, 1])
                            if cmds.std() > 1e-2 else float("nan"))
                    pair_rows.extend((condition["axis"], rep, float(c), float(a))
                                     for c, a in zip(cmds[::5], actual[::5]))
                    if make_traces and rep < 8:
                        trace = fake_control_trace(rng, vmax, tracking_gain, tracking_noise)
                        trace_rows.extend(
                            (condition["axis"], rep, step, float(c), float(p), float(a))
                            for step, (c, p, a) in enumerate(zip(*trace)))
                rows.append(dict(condition=condition["name"], group=group,
                                 axis_value=condition["axis"], rep=rep,
                                 route_seed=rep % 12, mass=0.391, radius=0.10,
                                 foot=0.8, ball=0.5, fell=fell, fail_reason=reason,
                                 duration=duration,
                                 cross_track=float(rng.uniform(0.08, 0.35)),
                                 progress=achieved * duration, ach_speed=achieved,
                                 cmd_speed=cmd_speed,
                                 ball_lost=float(rng.random() < lost_prob),
                                 ball_dist=float(rng.uniform(0.4, 0.75)),
                                 completed=success, success=success,
                                 speed_corr_r=corr))
        write_csv(rows, os.path.join(run_dir, f"{test}.csv"))
        if pair_rows:
            write_speed_pairs_csv(pair_rows, os.path.join(run_dir, f"{test}_speed_pairs.csv"))
        if trace_rows:
            write_speed_traces_csv(trace_rows, os.path.join(run_dir, f"{test}_speed_traces.csv"))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        run_a = os.path.join(tmp, "demo_expA")
        run_b = os.path.join(tmp, "demo_expB")
        write_fake_run(run_a, cliff_shift=-0.1, tracking_gain=0.82, tracking_noise=0.18)
        # traces only for one mock experiment — the real run makes one per checkpoint
        write_fake_run(run_b, cliff_shift=+0.1, tracking_gain=0.90, tracking_noise=0.10,
                       make_traces=False)
        sys.argv = ["plot", "--run-dirs", run_a, run_b,
                    "--labels", "expA-mock", "expB-mock",
                    "--out-dir", DEMO_DIR]
        plot.main()
    print(f"demo figures regenerated under {DEMO_DIR}")


if __name__ == "__main__":
    main()
