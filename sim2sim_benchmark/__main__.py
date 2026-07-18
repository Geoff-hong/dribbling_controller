"""CLI entry: python -m sim2sim_benchmark --robustness --capability --onnx ... --reset ...

Each selected test runs its condition table to completion and writes
<out-dir>/robustness.csv / capability.csv (plus a console summary). Render the
comparison figures afterwards with `python -m sim2sim_benchmark.plot`.
"""
import argparse
import os
import sys

if "--videos" in sys.argv:
    # the offscreen renderer needs a headless GL backend; must be set before
    # the first mujoco import (pulled in by the engine below)
    os.environ.setdefault("MUJOCO_GL", "egl")

from . import engine
from .conditions import robustness_conditions, capability_conditions, load_conditions_json
from .runner import run_condition_table, record_condition_videos
from .report import report


def main():
    ap = argparse.ArgumentParser(prog="sim2sim_benchmark", description=__doc__)
    ap.add_argument("--robustness", action="store_true",
                    help="DR scale / base push / ball push / latency axes on nominal human "
                         "routes; metrics = survival / possession / speed / tracking")
    ap.add_argument("--capability", action="store_true",
                    help="straight-line max speed + corner-turn max curvature with fail-fast "
                         "control criteria; metric = success rate")
    ap.add_argument("--conditions", default="",
                    help="run a custom JSON condition table instead of the built-in ones")
    ap.add_argument("--onnx", default=engine.DEFAULT_ONNX, help="deployment policy ONNX")
    ap.add_argument("--reset", default=engine.DEFAULT_RESET,
                    help="reset-state file (standby-trained policies need the standby reset)")
    ap.add_argument("--robots", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--route-bank", type=int, default=12,
                    help="fixed route seeds every condition cycles through")
    ap.add_argument("--reps", type=int, default=4,
                    help="episodes per condition = --route-bank x this")
    ap.add_argument("--episode-s", type=float, default=20.0,
                    help="episode length for conditions that do not pin their own")
    ap.add_argument("--standby-hold-s", type=float, default=0.0)
    ap.add_argument("--out-dir", default="eval_result/benchmark",
                    help="output folder for the per-episode CSVs")
    ap.add_argument("--videos", action="store_true",
                    help="after each table's statistics, record one mp4 per condition "
                         "(the rep-0 route, chase camera) under <out-dir>/videos/<test>/")
    ap.add_argument("--shard", default="",
                    help="'i/n': run only every n-th condition starting at i (0-based) — "
                         "launch n parallel processes with the same out-dir to split a "
                         "table across cores. Per-episode seeding keeps the FULL-table "
                         "condition index, so the union of shards is bit-identical to an "
                         "unsharded run; each shard writes <test>.shardI.csv (merge: keep "
                         "one header, concatenate rows into <test>.csv)")
    args = ap.parse_args()

    shard_i, shard_n = 0, 1
    if args.shard:
        try:
            shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            ap.error("--shard must be 'i/n'")
        if not 0 <= shard_i < shard_n:
            ap.error("--shard must be 'i/n' with 0 <= i < n")

    tables = []
    if args.conditions:
        tables.append(("conditions", load_conditions_json(args.conditions)))
    if args.robustness:
        tables.append(("robustness", robustness_conditions()))
    if args.capability:
        tables.append(("capability", capability_conditions()))
    if not tables:
        ap.error("pick at least one of --robustness / --capability / --conditions")

    os.makedirs(args.out_dir, exist_ok=True)
    for title, table in tables:
        table = [{**c, "_seed_index": j} for j, c in enumerate(table)][shard_i::shard_n]
        episode_rows = []
        speed_pair_rows = []
        speed_trace_rows = []
        suffix = f".shard{shard_i}" if args.shard else ""
        csv_path = os.path.join(args.out_dir, f"{title}{suffix}.csv")
        try:
            run_condition_table(table, title, episode_rows,
                                onnx=args.onnx, reset_file=args.reset,
                                n_robots=args.robots, seed=args.seed,
                                route_bank=args.route_bank, reps=args.reps,
                                episode_s=args.episode_s,
                                standby_hold_s=args.standby_hold_s,
                                speed_pair_rows=speed_pair_rows,
                                speed_trace_rows=speed_trace_rows)
        finally:
            # partial results survive a crash / Ctrl-C — never lose completed episodes
            report(episode_rows, csv_path, title, speed_pair_rows, speed_trace_rows)
        if args.videos:
            record_condition_videos(table, title, os.path.join(args.out_dir, "videos", title),
                                    onnx=args.onnx, reset_file=args.reset, seed=args.seed,
                                    episode_s=args.episode_s,
                                    standby_hold_s=args.standby_hold_s)


if __name__ == "__main__":
    main()
