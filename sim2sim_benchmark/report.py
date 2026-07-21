"""Per-episode CSV and console summary for a benchmark run.

Episodes are streamed to disk as they complete (EpisodeStream): the CSV itself
is the progress record, so a killed run loses nothing and a restart with the
same out-dir resumes where it stopped.
"""
import csv
import os

import numpy as np

CSV_COLUMNS = ["condition", "group", "axis_value", "rep", "route_seed",
               "ball_mass", "ball_radius", "foot_fric", "ball_fric",
               "fell", "fail_reason", "duration_s", "cross_track_m", "progress_m",
               "ach_speed_mps", "cmd_speed_mps", "ball_lost", "ball_dist_m",
               # foot_ball_dist_m is the TRAINING measure of possession (nearest
               # foot to ball surface); ball_dist_m stays pelvis-to-centre because
               # the ball_far fail-fast threshold is calibrated on it
               # `ball_lost` above is the MAIN threshold (engine.LOST_BALL_MAIN,
               # 0.8 m); ball_lost_t_s is its first-loss time.
               "foot_ball_dist_m", "ball_lost_t_s",
               "completed", "success", "speed_corr_r",
               # r is scale/offset invariant (a constant 0.5x cmd scores r=1), so
               # the regression of actual on commanded is what measures tracking
               "speed_slope", "speed_bias", "speed_resid_mps",
               # raw fall-criterion quantities: any threshold can be re-derived
               # from these, so changing FALL_Z stays auditable rather than a
               # silent re-ruling of past runs
               "min_pelvis_z", "max_tilt_gvec_z",
               # sticky lost-ball flag at the OTHER grid thresholds: 0.5 m =
               # training-faithful (feeds train_survival), 1.0 m = looser "really
               # gone" read. The MAIN threshold's flag is `ball_lost`.
               "ball_lost_05", "ball_lost_10"]
PAIRS_COLUMNS = ["axis_value", "rep", "cmd_speed_mps", "ball_speed_mps"]
TRACES_COLUMNS = ["axis_value", "episode", "step", "cmd_speed_mps",
                  "ball_speed_along_cmd_mps", "ball_speed_abs_mps"]


def _flag(value):
    return "" if not np.isfinite(value) else int(value)


def _num(value, fmt="{:.4f}"):
    return "" if not np.isfinite(value) else fmt.format(value)


def format_episode_row(row):
    return [row["condition"], row["group"], f"{row['axis_value']:.4f}",
            row["rep"], row["route_seed"],
            f"{row['mass']:.5f}", f"{row['radius']:.5f}",
            f"{row['foot']:.4f}", f"{row['ball']:.4f}",
            int(row["fell"]), row["fail_reason"],
            f"{row['duration']:.3f}", f"{row['cross_track']:.5f}",
            f"{row['progress']:.3f}", f"{row['ach_speed']:.4f}",
            f"{row['cmd_speed']:.4f}", int(row["ball_lost"]),
            f"{row['ball_dist']:.4f}",
            f"{row['foot_ball_dist']:.4f}", _num(row["ball_lost_t"], "{:.3f}"),
            _flag(row["completed"]), _flag(row["success"]),
            _num(row["speed_corr_r"], "{:.3f}"),
            _num(row["speed_slope"], "{:.4f}"), _num(row["speed_bias"], "{:.4f}"),
            _num(row["speed_resid"], "{:.4f}"),
            _num(row["min_pelvis_z"], "{:.4f}"), _num(row["max_tilt_gvec_z"], "{:.4f}"),
            # ball_lost_grid = [0.5, 0.8(main), 1.0]; the 0.8 is already `ball_lost`
            int(row["ball_lost_grid"][0]), int(row["ball_lost_grid"][2])]


def write_csv(episode_rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for row in episode_rows:
            writer.writerow(format_episode_row(row))


def _read_valid_rows(path, columns):
    """Data rows with exactly len(columns) fields from a possibly truncated CSV.
    Returns (rows, dirty); dirty = the file holds anything else (partial last
    line from a hard kill, legacy header without the `rep` column) and must be
    rewritten before appending."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return [], False
    with open(path, newline="") as f:
        parsed = list(csv.reader(f))
    if not parsed:
        return [], False
    header, body = parsed[0], parsed[1:]
    dirty = False
    if header != columns:
        if "rep" in columns and header == [c for c in columns if c != "rep"]:
            pad = columns.index("rep")                 # legacy pairs schema
            body = [r[:pad] + [""] + r[pad:] for r in body]
            dirty = True
        else:
            return [], True                            # unknown schema: start over
    rows = [r for r in body if len(r) == len(columns)]
    return rows, dirty or len(rows) != len(body)


def _rewrite(path, columns, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    os.replace(tmp, path)


def drop_conditions(csv_path, names):
    """Delete every episode of `names` from a main CSV and its aux sidecars.

    Used by the top-up planner when a condition's semantics changed: its old
    episodes describe a different experiment, so leaving them in would silently
    average two protocols together. Returns the number of episodes removed.

    The aux CSVs key on axis_value, not condition name, so they are filtered by
    the (axis_value, rep) pairs the dropped episodes owned — the same key
    EpisodeStream uses to reconcile them on resume.
    """
    rows, _ = _read_valid_rows(csv_path, CSV_COLUMNS)
    if not rows:
        return 0
    names = set(names)
    keep = [r for r in rows if r[0] not in names]
    if len(keep) == len(rows):
        return 0
    dropped_keys = {(r[2], r[3]) for r in rows if r[0] in names}
    _rewrite(csv_path, CSV_COLUMNS, keep)
    for path, columns in ((csv_path.replace(".csv", "_speed_pairs.csv"), PAIRS_COLUMNS),
                          (csv_path.replace(".csv", "_speed_traces.csv"), TRACES_COLUMNS)):
        aux, _ = _read_valid_rows(path, columns)
        if not aux:
            continue
        kept = [r for r in aux if (r[0], r[1]) not in dropped_keys]
        if len(kept) != len(aux):
            _rewrite(path, columns, kept)
    return len(rows) - len(keep)


class EpisodeStream:
    """Append-and-flush one episode at a time; the main CSV is the resume state.

    Aux rows (speed pairs/traces) are written BEFORE the episode's main row, so
    the main row acts as the commit marker: on restart, episodes present in the
    main CSV are skipped (`is_done`), a partial last line from a hard kill is
    dropped, and aux rows of episodes that never committed are removed."""

    def __init__(self, csv_path, fresh=False):
        self.csv_path = csv_path
        self.pairs_path = csv_path.replace(".csv", "_speed_pairs.csv")
        self.traces_path = csv_path.replace(".csv", "_speed_traces.csv")
        self.done = set()          # (condition name, rep)
        self._done_axis = set()    # (axis_value string, rep) — keys the aux rows
        self._files = {}
        if fresh:
            for path in (self.csv_path, self.pairs_path, self.traces_path):
                if os.path.exists(path):
                    os.remove(path)
        else:
            self._load_and_repair()

    def _load_and_repair(self):
        main_rows, dirty = _read_valid_rows(self.csv_path, CSV_COLUMNS)
        for row in main_rows:
            self.done.add((row[0], int(row[3])))
            self._done_axis.add((row[2], int(row[3])))
        if dirty:
            _rewrite(self.csv_path, CSV_COLUMNS, main_rows)
        for path, columns in ((self.pairs_path, PAIRS_COLUMNS),
                              (self.traces_path, TRACES_COLUMNS)):
            rows, dirty = _read_valid_rows(path, columns)
            # legacy rows (rep == "") predate commit tracking — keep them
            kept = [r for r in rows if r[1] == "" or (r[0], int(r[1])) in self._done_axis]
            if dirty or len(kept) != len(rows):
                _rewrite(path, columns, kept)

    def _writer(self, path, columns):
        if path not in self._files:
            new = not os.path.exists(path) or os.path.getsize(path) == 0
            f = open(path, "a", newline="")
            writer = csv.writer(f)
            if new:
                writer.writerow(columns)
            self._files[path] = (f, writer)
        return self._files[path]

    def is_done(self, condition_name, rep):
        return (condition_name, rep) in self.done

    def write_episode(self, row, pair_rows=(), trace_rows=()):
        """pair_rows: (axis_value, rep, cmd, actual); trace_rows: (axis_value,
        episode, step, cmd, along_cmd, speed_abs)."""
        if trace_rows:
            f, writer = self._writer(self.traces_path, TRACES_COLUMNS)
            writer.writerows([f"{a:.4f}", episode, step, f"{c:.4f}",
                              f"{p:.4f}", f"{s:.4f}"]
                             for a, episode, step, c, p, s in trace_rows)
            f.flush()
        if pair_rows:
            f, writer = self._writer(self.pairs_path, PAIRS_COLUMNS)
            writer.writerows([f"{a:.4f}", rep, f"{c:.4f}", f"{v:.4f}"]
                             for a, rep, c, v in pair_rows)
            f.flush()
        f, writer = self._writer(self.csv_path, CSV_COLUMNS)
        writer.writerow(format_episode_row(row))
        f.flush()
        self.done.add((row["condition"], row["rep"]))

    def close(self):
        for f, _ in self._files.values():
            f.close()
        self._files = {}


def _rate_pm(flags):
    """(rate %, standard error %) for a 0/1 outcome. The SE is binomial: episodes
    within a condition are independent draws, since the trajectory depends on
    which robot slot the episode landed on (see the runner's assign_next_episode
    note) and that assignment is effectively random. Report this ALWAYS — at the
    default reps 4 (n=48) it is ~7 points near p=0.5, wide enough that most
    single-axis wiggles in these tables are noise."""
    if not len(flags):
        return float("nan"), float("nan")
    p = float(np.mean(flags))
    return 100.0 * p, 100.0 * np.sqrt(max(p * (1.0 - p), 0.0) / len(flags))


def _mean_pm(values):
    """(mean, standard error of the mean) for a continuous metric."""
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _pct(values, q):
    values = [v for v in values if np.isfinite(v)]
    return float(np.percentile(values, q)) if values else float("nan")


def print_summary(episode_rows, title):
    by_condition = {}
    for row in episode_rows:
        key = (row["group"], row["axis_value"], row["condition"])
        by_condition.setdefault(key, []).append(row)
    print(f"\n=== {title.upper()}: {len(episode_rows)} episodes "
          f"| {len(by_condition)} conditions ===")
    print("every +/- is one standard error (rates binomial, others SEM); two "
          "conditions differ only when the gap exceeds ~2x the larger SE")
    print("v / v/cmd / ct are SURVIVORS-ONLY: an episode truncated by a fall "
          "covers its distance in less time, so an unfiltered mean rises as the "
          "condition gets harder. poss% uses the TRAINING lost-ball criterion "
          "(nearest foot to ball surface > 0.5 m for 0.1 s, after first touch); "
          "bd90 = 90th pct robot-ball distance, its continuous form.")
    print(f"{'condition':<18}{'n':>4}{'surv%':>14}{'succ%':>14}{'poss%':>7}"
          f"{'v(m/s)':>14}{'v/cmd':>7}{'r':>7}{'ct(m)':>14}{'bd90':>7}")
    for (group, axis, name) in sorted(by_condition, key=lambda k: (k[0], k[1])):
        rows = by_condition[(group, axis, name)]
        # survivors define every continuous metric below -- see the note above
        alive = [r for r in rows if r["fell"] < 0.5]
        survival, survival_se = _rate_pm([1.0 - r["fell"] for r in rows])
        possession, _ = _rate_pm([1.0 - r["ball_lost"] for r in rows])
        successes = [r["success"] for r in rows if np.isfinite(r["success"])]
        if successes:
            success, success_se = _rate_pm(successes)
            success_txt = f"{success:>8.0f}+-{success_se:<4.0f}"
        else:
            success_txt = f"{'-':>14}"
        ach_speed, ach_speed_se = _mean_pm([r["ach_speed"] for r in alive])
        speed_ratio, _ = _mean_pm([r["ach_speed"] / r["cmd_speed"] for r in alive
                                   if np.isfinite(r["ach_speed"]) and np.isfinite(r["cmd_speed"])
                                   and r["cmd_speed"] > 0.05])
        corr_values = [r["speed_corr_r"] for r in rows if np.isfinite(r["speed_corr_r"])]
        corr_txt = f"{np.mean(corr_values):>7.2f}" if corr_values else f"{'-':>7}"
        # cross-track is survivors-only, so its n is not len(rows) -- _mean_pm
        # divides by the actual survivor count, which is what the SE must use
        cross_track, cross_track_se = _mean_pm([r["cross_track"] for r in alive])
        ball_dist_p90 = _pct([r["ball_dist"] for r in rows], 90)
        print(f"{name:<18}{len(rows):>4}"
              f"{survival:>8.0f}+-{survival_se:<4.0f}{success_txt}"
              f"{possession:>7.0f}"
              f"{ach_speed:>8.2f}+-{ach_speed_se:<4.2f}{speed_ratio:>7.2f}{corr_txt}"
              f"{cross_track:>8.3f}+-{cross_track_se:<5.3f}{ball_dist_p90:>7.2f}")


def write_speed_pairs_csv(speed_pair_rows, csv_path):
    """(axis_value, rep, cmd, actual) rows from the speed-controllability test."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(PAIRS_COLUMNS)
        for axis_value, rep, cmd, actual in speed_pair_rows:
            writer.writerow([f"{axis_value:.4f}", rep, f"{cmd:.4f}", f"{actual:.4f}"])


def write_speed_traces_csv(speed_trace_rows, csv_path):
    """Full-rate (50 Hz) per-step traces of the first episodes of each
    speed-tracking condition, for the control trace plots."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TRACES_COLUMNS)
        for axis_value, episode, step, cmd, along_cmd, speed_abs in speed_trace_rows:
            writer.writerow([f"{axis_value:.4f}", episode, step, f"{cmd:.4f}",
                             f"{along_cmd:.4f}", f"{speed_abs:.4f}"])


def load_summary_rows(csv_path):
    """Episode rows back from disk, in the dict form print_summary expects."""
    def num(value):
        return float(value) if value not in ("", None) else float("nan")
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(condition=r["condition"], group=r["group"],
                             axis_value=float(r["axis_value"]),
                             rep=r.get("rep", ""), route_seed=r.get("route_seed", ""),
                             fell=num(r["fell"]), ball_lost=num(r["ball_lost"]),
                             ball_lost_05=num(r.get("ball_lost_05", "")),
                             ball_lost_10=num(r.get("ball_lost_10", "")),
                             success=num(r["success"]), ach_speed=num(r["ach_speed_mps"]),
                             cmd_speed=num(r["cmd_speed_mps"]),
                             speed_corr_r=num(r["speed_corr_r"]),
                             cross_track=num(r["cross_track_m"]),
                             progress=num(r.get("progress_m", "")),
                             duration=num(r.get("duration_s", "")),
                             ball_dist=num(r.get("ball_dist_m", "")),
                             foot_ball_dist=num(r.get("foot_ball_dist_m", "")),
                             ball_lost_t=num(r.get("ball_lost_t_s", "")),
                             speed_slope=num(r.get("speed_slope", "")),
                             speed_bias=num(r.get("speed_bias", "")),
                             speed_resid=num(r.get("speed_resid_mps", "")),
                             min_pelvis_z=num(r.get("min_pelvis_z", "")),
                             max_tilt_gvec_z=num(r.get("max_tilt_gvec_z", ""))))
    return rows


def report_csv(csv_path, title):
    """Console summary from the on-disk CSV (previous + newly streamed rows)."""
    if not os.path.exists(csv_path):
        print(f"[{title}] no episodes completed")
        return
    rows = load_summary_rows(csv_path)
    if not rows:
        print(f"[{title}] no episodes completed")
        return
    print(f"[{title}] {csv_path}: {len(rows)} rows")
    print_summary(rows, title)
