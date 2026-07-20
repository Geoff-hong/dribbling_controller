"""CLI entry: python -m sim2sim_benchmark --robustness --capability --onnx ... --reset ...

Each selected test runs its condition table to completion and writes
<out-dir>/robustness.csv / capability.csv (plus a console summary). Episodes
are flushed to the CSV as they complete, and re-running with the same out-dir
resumes: episodes already in the CSV are skipped (--fresh starts over). Render
the comparison figures afterwards with `python -m sim2sim_benchmark.plot`.
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
from .report import EpisodeStream, report_csv
from .train_dr import read_train_dr, describe


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
    ap.add_argument("--dr-from", default="",
                    help="checkpoint dir / env.yaml the test DR ranges are derived from "
                         "(default: the env.yaml next to --onnx). When COMPARING policies "
                         "trained with different DR, pass the same --dr-from to every run "
                         "so all runs share one condition table")
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
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing CSVs in --out-dir and start over (default: "
                         "resume — episodes already recorded are skipped)")
    args = ap.parse_args()

    shard_i, shard_n = 0, 1
    if args.shard:
        try:
            shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            ap.error("--shard must be 'i/n'")
        if not 0 <= shard_i < shard_n:
            ap.error("--shard must be 'i/n' with 0 <= i < n")

    # test ranges anchor on the DR the policy was actually trained with
    train = read_train_dr(args.dr_from or args.onnx)
    if args.dr_from and train is None:
        ap.error(f"--dr-from {args.dr_from}: no env.yaml found — an explicit DR source "
                 "must resolve (a typo here would silently rebase the whole table)")
    for line in describe(train):
        print(f"[train_dr] {line}")
    engine.configure_train_dr(train)

    tables = []
    if args.conditions:
        tables.append(("conditions", load_conditions_json(args.conditions)))
    if args.robustness:
        tables.append(("robustness", robustness_conditions(train)))
    if args.capability:
        tables.append(("capability", capability_conditions(train)))
    if not tables:
        ap.error("pick at least one of --robustness / --capability / --conditions")

    import glob as globmod
    import hashlib
    import json

    def table_fingerprint(table):
        # semantic identity of a run's conditions: the FULL table plus the engine
        # DR state its dr_scale/latency sampling depends on
        payload = json.dumps({"table": table, "dr": engine.DR,
                              "ball_delay": engine.BALL_DELAY_RANGE,
                              "act_delay": engine.ACT_DELAY_SUBSTEPS,
                              "act_zero": engine.ACT_DELAY_ZERO_PROB},
                             sort_keys=True, default=list)
        return hashlib.sha1(payload.encode()).hexdigest()

    os.makedirs(args.out_dir, exist_ok=True)
    # resume guard BEFORE anything is written: episodes are resumed by condition
    # name alone, so resuming onto a table with different semantics (other
    # --dr-from/--onnx DR, changed axis derivation) would silently mix data
    def read_fingerprint(fp_path):
        # parallel shard launches race reader vs writer; the write below is atomic
        # (os.replace), so a decode error can only be a half-written file from an
        # OLD code version — retry briefly, then fail rather than guess
        import time
        for _ in range(10):
            try:
                return json.load(open(fp_path))["fingerprint"]
            except (json.JSONDecodeError, KeyError):
                time.sleep(0.2)
        ap.error(f"{fp_path}: unreadable/corrupt — delete it (or use --fresh) and rerun")

    for title, table in tables:
        fp = table_fingerprint(table)
        fp_path = os.path.join(args.out_dir, f"{title}.fingerprint.json")
        has_csv = bool(globmod.glob(os.path.join(args.out_dir, f"{title}*.csv")))
        if not args.fresh and os.path.exists(fp_path):
            if read_fingerprint(fp_path) != fp:
                ap.error(f"{fp_path}: the existing {title} CSVs were recorded with a "
                         "DIFFERENT condition table (other --dr-from/--onnx DR or "
                         "changed axes) — resuming would silently mix incompatible "
                         "episodes. Use --fresh or a new --out-dir.")
        elif not args.fresh and has_csv:
            ap.error(f"{args.out_dir}: {title} CSVs exist but carry no table "
                     "fingerprint (recorded before DR-anchored tables) — resuming "
                     "would silently mix incompatible episodes. Use --fresh or a "
                     "new --out-dir.")
        tmp_path = f"{fp_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump({"fingerprint": fp, "conditions": [
                dict(name=c["name"], group=c["group"], axis=c["axis"]) for c in table]},
                f, indent=2)
        os.replace(tmp_path, fp_path)   # atomic: concurrent shards never see a partial file
    # provenance: the parsed training DR + the derived ranges this run tested with
    with open(os.path.join(args.out_dir, "train_dr.json"), "w") as f:
        json.dump({"train": train, "derived_dr": engine.DR,
                   "derived_sweep_ranges": engine.SWEEP_RANGES,
                   "onnx": args.onnx}, f, indent=2, default=list)
    for title, table in tables:
        table = [{**c, "_seed_index": j} for j, c in enumerate(table)][shard_i::shard_n]
        suffix = f".shard{shard_i}" if args.shard else ""
        csv_path = os.path.join(args.out_dir, f"{title}{suffix}.csv")
        stream = EpisodeStream(csv_path, fresh=args.fresh)
        try:
            run_condition_table(table, title, stream,
                                onnx=args.onnx, reset_file=args.reset,
                                n_robots=args.robots, seed=args.seed,
                                route_bank=args.route_bank, reps=args.reps,
                                episode_s=args.episode_s,
                                standby_hold_s=args.standby_hold_s)
        finally:
            # every completed episode is already on disk; just close and summarize
            stream.close()
            report_csv(csv_path, title)
        if args.videos:
            record_condition_videos(table, title, os.path.join(args.out_dir, "videos", title),
                                    onnx=args.onnx, reset_file=args.reset, seed=args.seed,
                                    episode_s=args.episode_s,
                                    standby_hold_s=args.standby_hold_s)


if __name__ == "__main__":
    main()
