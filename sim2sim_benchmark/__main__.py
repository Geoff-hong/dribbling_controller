"""CLI entry: python -m sim2sim_benchmark --robustness --capability --onnx ... --reset ...

Each selected test runs its condition table to completion and writes
<out-dir>/robustness.csv / capability.csv (plus a console summary). Render the
comparison figures afterwards with `python -m sim2sim_benchmark.plot`.
"""
import argparse
import os

from . import engine
from .conditions import robustness_conditions, capability_conditions, load_conditions_json
from .runner import run_condition_table
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
    args = ap.parse_args()

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
        episode_rows = []
        speed_pair_rows = []
        speed_trace_rows = []
        csv_path = os.path.join(args.out_dir, f"{title}.csv")
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


if __name__ == "__main__":
    main()
