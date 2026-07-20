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
               "completed", "success", "speed_corr_r"]
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
            _flag(row["completed"]), _flag(row["success"]),
            _num(row["speed_corr_r"], "{:.3f}")]


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


def print_summary(episode_rows, title):
    by_condition = {}
    for row in episode_rows:
        key = (row["group"], row["axis_value"], row["condition"])
        by_condition.setdefault(key, []).append(row)
    print(f"\n=== {title.upper()}: {len(episode_rows)} episodes "
          f"| {len(by_condition)} conditions ===")
    print(f"{'condition':<18}{'n':>4}{'surv%':>7}{'succ%':>7}{'poss%':>7}"
          f"{'v(m/s)':>8}{'v/cmd':>7}{'r':>7}{'ct(m)':>8}")
    for (group, axis, name) in sorted(by_condition, key=lambda k: (k[0], k[1])):
        rows = by_condition[(group, axis, name)]
        survival = 100.0 * (1.0 - np.mean([r["fell"] for r in rows]))
        possession = 100.0 * (1.0 - np.mean([r["ball_lost"] for r in rows]))
        successes = [r["success"] for r in rows if np.isfinite(r["success"])]
        success_txt = f"{100.0 * np.mean(successes):>7.0f}" if successes else f"{'-':>7}"
        ach_speeds = [r["ach_speed"] for r in rows if np.isfinite(r["ach_speed"])]
        ach_speed = np.mean(ach_speeds) if ach_speeds else float("nan")
        speed_ratios = [r["ach_speed"] / r["cmd_speed"] for r in rows
                        if np.isfinite(r["ach_speed"]) and np.isfinite(r["cmd_speed"])
                        and r["cmd_speed"] > 0.05]
        speed_ratio = np.mean(speed_ratios) if speed_ratios else float("nan")
        corr_values = [r["speed_corr_r"] for r in rows if np.isfinite(r["speed_corr_r"])]
        corr_txt = f"{np.mean(corr_values):>7.2f}" if corr_values else f"{'-':>7}"
        survivor_ct = [r["cross_track"] for r in rows
                       if r["fell"] < 0.5 and np.isfinite(r["cross_track"])]
        cross_track = np.mean(survivor_ct) if survivor_ct else float("nan")
        print(f"{name:<18}{len(rows):>4}{survival:>7.0f}{success_txt}{possession:>7.0f}"
              f"{ach_speed:>8.2f}{speed_ratio:>7.2f}{corr_txt}{cross_track:>8.3f}")


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
                             fell=num(r["fell"]), ball_lost=num(r["ball_lost"]),
                             success=num(r["success"]), ach_speed=num(r["ach_speed_mps"]),
                             cmd_speed=num(r["cmd_speed_mps"]),
                             speed_corr_r=num(r["speed_corr_r"]),
                             cross_track=num(r["cross_track_m"])))
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
