"""CLI entry: python -m sim2sim_benchmark --robustness --capability --onnx ... --reset ...

Each selected test runs its condition table to completion and writes
<out-dir>/robustness.csv / capability.csv (plus a console summary). Episodes
are flushed to the CSV as they complete, and re-running with the same out-dir
resumes: episodes already in the CSV are skipped (--fresh starts over). Read the
results with `python -m sim2sim_benchmark.html_report` (the default interactive
report); `python -m sim2sim_benchmark.plot` renders the legacy static PNGs.
"""
import argparse
import csv
import os
import sys

if "--videos" in sys.argv:
    # the offscreen renderer needs a headless GL backend; must be set before
    # the first mujoco import (pulled in by the engine below)
    os.environ.setdefault("MUJOCO_GL", "egl")

from . import engine
from .conditions import robustness_conditions, capability_conditions, load_conditions_json
from .runner import run_condition_table, record_condition_videos
from .report import EpisodeStream, drop_conditions, report_csv
from . import topup
from .train_dr import read_train_dr, describe


def main():
    ap = argparse.ArgumentParser(prog="sim2sim_benchmark", description=__doc__)
    ap.add_argument("--robustness", action="store_true",
                    help="DR scale / base push / ball push / latency axes on nominal human "
                         "routes; metrics = survival / possession / speed / tracking")
    ap.add_argument("--capability", action="store_true",
                    help="straight-line max speed + corner-turn max curvature with fail-fast "
                         "control criteria; metrics = nested success rates + cross-track")
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
    ap.add_argument("--episode-s", type=float, default=12.0,
                    help="episode budget: the length for conditions that do not "
                         "pin their own, AND a cap on those that do. Default 12 s, "
                         "inside the final 15 s episode length of both current "
                         "lineages' training curricula (5 -> 10 -> 15 s); evaluating "
                         "longer than training measures survival at a duration the "
                         "policy never saw")
    ap.add_argument("--standby-hold-s", type=float, default=0.0,
                    help="whole-run FALLBACK stiff-standby hold (s) for conditions "
                         "that do not pin their own; the robustness 'handover' axis "
                         "sweeps it per-condition, so leave this 0 for a normal run")
    ap.add_argument("--settle-s", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="training settle_time_range_s: per-episode standby-PD "
                         "takeover window U(LO,HI) s on the policy's OWN gains "
                         "(soft, action overridden to the default pose). Default: "
                         "read from the checkpoint's env.yaml (all current "
                         "checkpoints trained with settle OFF -> 0)")
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
    ap.add_argument("--apply-topup", action="store_true",
                    help="the table changed: DROP the recorded episodes of every "
                         "condition whose semantics changed, then continue. Run this "
                         "ONCE without --shard before launching the shards; the shards "
                         "then simply resume and fill the gaps. Conditions whose "
                         "fingerprint is unchanged keep their episodes")
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

    # training settle window: --settle-s overrides; else the checkpoint's own
    # settle_time_range_s (None/(0,0) -> off, which every current checkpoint is)
    settle_range = None
    if args.settle_s is not None:
        settle_range = tuple(sorted(args.settle_s))
    elif train and train.get("settle_time_range_s") and train["settle_time_range_s"][1] > 0:
        settle_range = tuple(train["settle_time_range_s"])
    if settle_range:
        print(f"[settle] policy takeover window U{settle_range} s (soft gains, "
              "action = default pose) on every episode")

    tables = []
    if args.conditions:
        tables.append(("conditions", load_conditions_json(args.conditions)))
    if args.robustness:
        tables.append(("robustness", robustness_conditions(train)))
    if args.capability:
        tables.append(("capability", capability_conditions(train)))
    if not tables:
        ap.error("pick at least one of --robustness / --capability / --conditions")

    # --episode-s is a CAP, not just the fallback for unpinned conditions: a
    # condition that pins a longer budget (human_dribble and speed_tracking pin
    # 20 s) would otherwise evaluate the policy past any length it ever saw in
    # training, and survival falls monotonically with episode length, so the
    # extra seconds are a silent handicap rather than a harder test of the same
    # thing. Shorter pinned budgets (straight 10 s, most turn budgets) are left
    # alone -- they are fail-fast time limits, not exposure. NOTE: at the 12 s
    # default the gentlest corner_turn budgets (kappa 0.2 -> 15 s, 0.3 -> 13 s)
    # are also capped, which eats their turn_budget slack margin; a timeout
    # there can be budget-clipping, not lost control (--episode-s 15 restores).
    capped = 0
    for _title, table in tables:
        for c in table:
            if c.get("episode_s") and float(c["episode_s"]) > args.episode_s:
                c["episode_s"] = args.episode_s
                capped += 1
    if capped:
        print(f"[conditions] capped {capped} condition budgets to "
              f"--episode-s {args.episode_s:g}s")

    import glob as globmod
    import hashlib
    import json

    os.makedirs(args.out_dir, exist_ok=True)
    # Everything configure_train_dr() resolves from the checkpoint's env.yaml.
    # The ball channels were covered; the ROBOT ones were not, so two runs of
    # different checkpoints could share a fingerprint while sampling different
    # gain / payload / CoM / encoder / joint-friction ranges. Route geometry and
    # the MJCF are in for the same reason: neither is hashed by the source
    # fingerprint, and both change what an episode is.
    engine_state = {"dr": engine.DR, "ball_delay": engine.BALL_DELAY_RANGE,
                    "act_delay": engine.ACT_DELAY_SUBSTEPS,
                    "act_zero": engine.ACT_DELAY_ZERO_PROB,
                    "ball_damping": engine.BALL_DAMPING,
                    "obs_noise": engine.OBS_NOISE,
                    "actuator_gain": engine.ACTUATOR_GAIN_RANGE,
                    "actuator_damping": engine.ACTUATOR_DAMPING_RANGE,
                    "payload_kg": engine.PAYLOAD_KG_RANGE,
                    "base_com": engine.BASE_COM_RANGE,
                    "joint_offset": engine.JOINT_OFFSET_RANGE,
                    "joint_friction": engine.JOINT_FRICTION_RANGE,
                    "reset_ball_dist": engine.RESET_BALL_DIST_RANGE,
                    "reset_ball_bearing": engine.RESET_BALL_BEARING_DEG,
                    "route_cfg": engine.ROUTE_CFG,
                    "mjcf": topup.file_sha(engine.SINGLE_MJCF)}
    # NOT including reps: extra reps are extra episodes of the SAME condition, so
    # raising it is itself a free top-up that plain resume already handles.
    # onnx/reset are hashed by CONTENT: every checkpoint dir names its policy
    # `softtouch_dribble_deploy.onnx`, so the basename identified six different
    # policies as one. `robots` is in because slot count changes the numerical
    # noise (see runner's 29/46/46 note), not the expectation.
    run_params = {"seed": args.seed, "route_bank": args.route_bank,
                  "episode_s": args.episode_s, "standby_hold_s": args.standby_hold_s,
                  "settle_range": list(settle_range) if settle_range else None,
                  "robots": args.robots,
                  "onnx": os.path.basename(args.onnx), "reset": os.path.basename(args.reset),
                  "onnx_sha": topup.file_sha(args.onnx, (args.onnx + ".data",)),
                  "reset_sha": topup.file_sha(args.reset)}
    manifest_path = os.path.join(args.out_dir, "conditions.manifest.json")
    new_manifest = topup.build_manifest(tables, engine_state, run_params)
    old_manifest = None if args.fresh else topup.load_manifest(manifest_path)

    # Per-condition resume guard, BEFORE anything is written. Episodes resume by
    # condition name, so a name whose semantics changed must have its old rows
    # dropped or the two protocols get averaged together.
    plan = {}
    for title, table in tables:
        plan[title] = topup.classify(new_manifest, old_manifest, title)
    # episodes actually on disk, per condition — "reusable" is a statement about
    # the FINGERPRINT, not about coverage, and conflating the two would read as
    # "already measured" for a condition that was never run
    def recorded_counts(title):
        counts = {}
        for path in (globmod.glob(os.path.join(args.out_dir, f"{title}.csv"))
                     + globmod.glob(os.path.join(args.out_dir, f"{title}.shard*.csv"))):
            if "_speed_" in os.path.basename(path):
                continue
            try:
                with open(path, newline="") as f:
                    for row in list(csv.reader(f))[1:]:
                        if row:
                            counts[row[0]] = counts.get(row[0], 0) + 1
            except OSError:
                pass
        return counts

    has_rows = any(recorded_counts(title) for title, _ in tables)
    for line in topup.describe_drift(old_manifest, new_manifest):
        print(f"[topup] WARNING: {line}")
    if old_manifest and topup.describe_drift(old_manifest, new_manifest):
        print("[topup] WARNING: reused episodes came from that other build — only "
              "you can judge whether the change was physics-neutral (stripping "
              "render-only meshes was; moving the obs anchor was not).")
    if topup.legacy_seeding(old_manifest, has_rows):
        print("[topup] NOTE: episodes on disk predate name-hash seeding. They are "
              "kept per condition (never mixed within one), but this dir is no "
              "longer reproducible from the table alone.")
    for title, (reusable, changed, fresh, stale) in plan.items():
        counts = recorded_counts(title)
        covered = sum(1 for n in reusable if counts.get(n))
        print(f"[topup] {title}: {len(reusable)} reusable "
              f"({covered} with episodes on disk, {len(reusable) - covered} not yet run), "
              f"{len(changed)} changed, {len(fresh)} new, {len(stale)} stale")
        for label, names in (("changed", changed), ("new", fresh), ("stale", stale)):
            if names:
                print(f"[topup]   {label}: {', '.join(names[:8])}"
                      f"{f' (+{len(names) - 8} more)' if len(names) > 8 else ''}")

    needs_drop = {t: c for t, (_, c, _, _) in plan.items() if c}
    if needs_drop and not args.apply_topup:
        ap.error("condition semantics changed for: "
                 + "; ".join(f"{t}: {', '.join(n)}" for t, n in needs_drop.items())
                 + ". Their recorded episodes describe a different experiment. Re-run "
                   "ONCE with --apply-topup (single process, before launching shards) "
                   "to drop those rows, or use --fresh / a new --out-dir.")
    if args.apply_topup:
        if args.shard:
            ap.error("--apply-topup rewrites the shard CSVs; run it once WITHOUT "
                     "--shard before launching the shards")
        for title, names in needs_drop.items():
            for path in sorted(globmod.glob(os.path.join(args.out_dir, f"{title}.csv"))
                               + globmod.glob(os.path.join(args.out_dir, f"{title}.shard*.csv"))):
                if "_speed_" in os.path.basename(path):
                    continue          # sidecars are handled by drop_conditions itself
                n = drop_conditions(path, names)
                if n:
                    print(f"[topup] dropped {n} episodes from {os.path.basename(path)}")
    topup.save_manifest(manifest_path, new_manifest)
    # provenance: the parsed training DR + the derived ranges this run tested with
    with open(os.path.join(args.out_dir, "train_dr.json"), "w") as f:
        json.dump({"train": train, "derived_dr": engine.DR,
                   "derived_sweep_ranges": engine.SWEEP_RANGES,
                   "onnx": args.onnx}, f, indent=2, default=list)
    for title, table in tables:
        # seed index from the condition NAME, not its position: inserting or
        # reordering an axis must not perturb any other condition's draws
        table = [{**c, "_seed_index": topup.seed_index(c["name"])}
                 for c in table][shard_i::shard_n]
        suffix = f".shard{shard_i}" if args.shard else ""
        csv_path = os.path.join(args.out_dir, f"{title}{suffix}.csv")
        stream = EpisodeStream(csv_path, fresh=args.fresh)
        try:
            run_condition_table(table, title, stream,
                                onnx=args.onnx, reset_file=args.reset,
                                n_robots=args.robots, seed=args.seed,
                                route_bank=args.route_bank, reps=args.reps,
                                episode_s=args.episode_s,
                                standby_hold_s=args.standby_hold_s,
                                settle_range=settle_range)
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
