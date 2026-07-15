"""Per-episode CSV and console summary for a benchmark run."""
import csv

import numpy as np

CSV_COLUMNS = ["condition", "group", "axis_value", "rep", "route_seed",
               "ball_mass", "ball_radius", "foot_fric", "ball_fric",
               "fell", "fail_reason", "duration_s", "cross_track_m", "progress_m",
               "ach_speed_mps", "cmd_speed_mps", "ball_lost", "ball_dist_m",
               "completed", "success"]


def _flag(value):
    return "" if not np.isfinite(value) else int(value)


def write_csv(episode_rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for row in episode_rows:
            writer.writerow([row["condition"], row["group"], f"{row['axis_value']:.4f}",
                             row["rep"], row["route_seed"],
                             f"{row['mass']:.5f}", f"{row['radius']:.5f}",
                             f"{row['foot']:.4f}", f"{row['ball']:.4f}",
                             int(row["fell"]), row["fail_reason"],
                             f"{row['duration']:.3f}", f"{row['cross_track']:.5f}",
                             f"{row['progress']:.3f}", f"{row['ach_speed']:.4f}",
                             f"{row['cmd_speed']:.4f}", int(row["ball_lost"]),
                             f"{row['ball_dist']:.4f}",
                             _flag(row["completed"]), _flag(row["success"])])


def print_summary(episode_rows, title):
    by_condition = {}
    for row in episode_rows:
        key = (row["group"], row["axis_value"], row["condition"])
        by_condition.setdefault(key, []).append(row)
    print(f"\n=== {title.upper()}: {len(episode_rows)} episodes "
          f"| {len(by_condition)} conditions ===")
    print(f"{'condition':<18}{'n':>4}{'surv%':>7}{'succ%':>7}{'poss%':>7}{'v/cmd':>7}{'ct(m)':>8}")
    for (group, axis, name) in sorted(by_condition, key=lambda k: (k[0], k[1])):
        rows = by_condition[(group, axis, name)]
        survival = 100.0 * (1.0 - np.mean([r["fell"] for r in rows]))
        possession = 100.0 * (1.0 - np.mean([r["ball_lost"] for r in rows]))
        successes = [r["success"] for r in rows if np.isfinite(r["success"])]
        success_txt = f"{100.0 * np.mean(successes):>7.0f}" if successes else f"{'-':>7}"
        speed_ratios = [r["ach_speed"] / r["cmd_speed"] for r in rows
                        if np.isfinite(r["ach_speed"]) and np.isfinite(r["cmd_speed"])
                        and r["cmd_speed"] > 0.05]
        speed_ratio = np.mean(speed_ratios) if speed_ratios else float("nan")
        survivor_ct = [r["cross_track"] for r in rows
                       if r["fell"] < 0.5 and np.isfinite(r["cross_track"])]
        cross_track = np.mean(survivor_ct) if survivor_ct else float("nan")
        print(f"{name:<18}{len(rows):>4}{survival:>7.0f}{success_txt}{possession:>7.0f}"
              f"{speed_ratio:>7.2f}{cross_track:>8.3f}")


def report(episode_rows, csv_path, title):
    if not episode_rows:
        print(f"[{title}] no episodes completed")
        return
    if csv_path:
        write_csv(episode_rows, csv_path)
        print(f"[{title}] saved {csv_path} ({len(episode_rows)} rows)")
    print_summary(episode_rows, title)
