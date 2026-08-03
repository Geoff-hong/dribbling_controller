#!/usr/bin/env python3
"""Plot deployed SoftTouch command routes against recorded real-ball tracks.

The active episodes are recovered from the 50 Hz policy topic.  For each
episode, the plot uses:

* `/softtouch/dribble/markers`: the exact route and current/preview command
  arrows produced by the deployed C++ controller;
* `/softtouch/mocap/ball/pose`: the high-rate recorded ball position;
* `/softtouch/policy/observation`: command speed and ball-velocity diagnostics.

The first marker after each controller activation is dropped because it can
still contain the previous controller instance's route.  A ball-position jump
larger than 0.5 m marks the remaining track as unreliable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mcap_ros2.reader import read_ros2_messages


DEFAULT_LOG_ROOT = Path(
    "/home/alden/Desktop/sim2real_logs/2026-7-27/"
    "softtouch_deploy_logs_20260727_night_20260728"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/alden/Desktop/dribbling_controller/analysis/softtouch_20260727_night"
)


@dataclass
class MarkerSample:
    time: float
    ball_xy: np.ndarray
    route_xy: np.ndarray
    current_arrow: np.ndarray
    preview_arrow: np.ndarray


@dataclass
class Episode:
    run: str
    number: int
    start: float
    end: float
    ball_time: np.ndarray
    ball_xy: np.ndarray
    markers: list[MarkerSample]
    command_speed: np.ndarray
    velocity_spike_time: np.ndarray
    bad_time: float | None
    ct_time: np.ndarray
    ct: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.run} / E{self.number:02d}"

    @property
    def reliable_mask(self) -> np.ndarray:
        if self.bad_time is None:
            return np.ones(len(self.ball_time), dtype=bool)
        return self.ball_time < self.bad_time

    @property
    def reliable_ct_mask(self) -> np.ndarray:
        if self.bad_time is None:
            return np.ones(len(self.ct_time), dtype=bool)
        return self.ct_time < self.bad_time

    @property
    def ct_mean(self) -> float:
        values = self.ct[self.reliable_ct_mask]
        return float(np.mean(values)) if len(values) else float("nan")

    @property
    def ct_final(self) -> float:
        values = self.ct[self.reliable_ct_mask]
        return float(values[-1]) if len(values) else float("nan")

    @property
    def route_for_plot(self) -> np.ndarray:
        eligible = [
            sample
            for sample in self.markers
            if self.bad_time is None or sample.time < self.bad_time
        ]
        return eligible[-1].route_xy if eligible else self.markers[-1].route_xy


def configure_matplotlib() -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    if font_path.exists():
        mpl.font_manager.fontManager.addfont(font_path)
        mpl.rcParams["font.family"] = "Noto Sans CJK JP"
    mpl.rcParams.update(
        {
            "axes.unicode_minus": False,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#94a3b8",
            "grid.color": "#cbd5e1",
            "grid.alpha": 0.45,
        }
    )


def stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def nearest_polyline_distance(point: np.ndarray, line: np.ndarray) -> float:
    starts = line[:-1]
    delta = line[1:] - starts
    denom = np.sum(delta * delta, axis=1)
    u = np.divide(
        np.sum((point - starts) * delta, axis=1),
        denom,
        out=np.zeros_like(denom),
        where=denom > 1.0e-12,
    )
    projection = starts + np.clip(u, 0.0, 1.0)[:, None] * delta
    return float(np.min(np.linalg.norm(projection - point, axis=1)))


def load_schema(mcap_path: Path) -> dict[str, Any]:
    for record in read_ros2_messages(
        mcap_path, topics=["/softtouch/policy/observation_schema"]
    ):
        return json.loads(record.ros_msg.data)
    raise RuntimeError(f"No observation schema in {mcap_path}")


def latest_term_slice(schema: dict[str, Any], term_name: str) -> slice:
    names = schema["observation_names"]
    dims = schema["observation_dims"]
    history = schema["history_lengths"]
    cursor = 0
    for name, dim, length in zip(names, dims, history):
        width = int(dim) * int(length)
        if name == term_name:
            start = cursor + width - int(dim)
            return slice(start, start + int(dim))
        cursor += width
    raise KeyError(term_name)


def active_ranges(mcap_path: Path) -> list[tuple[float, float]]:
    times = np.asarray(
        [
            record.log_time_ns * 1.0e-9
            for record in read_ros2_messages(
                mcap_path, topics=["/softtouch/policy/joint_target"]
            )
        ],
        dtype=float,
    )
    if not len(times):
        return []
    cuts = np.r_[0, np.where(np.diff(times) > 0.2)[0] + 1, len(times)]
    return [(float(times[a]), float(times[b - 1])) for a, b in zip(cuts[:-1], cuts[1:])]


def load_episode(
    mcap_path: Path,
    run: str,
    number: int,
    start: float,
    end: float,
    speed_slice: slice,
    ball_velocity_slice: slice,
) -> Episode:
    margin = 0.03
    kwargs = {
        "start_time": int((start - margin) * 1.0e9),
        "end_time": int((end + margin) * 1.0e9),
    }

    ball_rows: list[tuple[float, float, float]] = []
    for record in read_ros2_messages(
        mcap_path, topics=["/softtouch/mocap/ball/pose"], **kwargs
    ):
        position = record.ros_msg.pose.position
        ball_rows.append(
            (record.log_time_ns * 1.0e-9, float(position.x), float(position.y))
        )
    ball = np.asarray(ball_rows, dtype=float)
    if not len(ball):
        raise RuntimeError(f"No ball poses for {run} episode {number}")

    marker_rows: list[MarkerSample] = []
    for record in read_ros2_messages(
        mcap_path, topics=["/softtouch/dribble/markers"], **kwargs
    ):
        by_id = {marker.id: marker for marker in record.ros_msg.markers}
        if not all(marker_id in by_id for marker_id in (0, 1, 2, 3)):
            continue
        ball_marker = by_id[0].pose.position
        route = np.asarray(
            [(point.x, point.y) for point in by_id[1].points], dtype=float
        )
        current_arrow = np.asarray(
            [(point.x, point.y) for point in by_id[2].points], dtype=float
        )
        preview_arrow = np.asarray(
            [(point.x, point.y) for point in by_id[3].points], dtype=float
        )
        if len(route) >= 2 and current_arrow.shape == (2, 2):
            marker_rows.append(
                MarkerSample(
                    time=record.log_time_ns * 1.0e-9,
                    ball_xy=np.asarray([ball_marker.x, ball_marker.y]),
                    route_xy=route,
                    current_arrow=current_arrow,
                    preview_arrow=preview_arrow,
                )
            )
    if len(marker_rows) < 2:
        raise RuntimeError(f"Insufficient route markers for {run} episode {number}")
    marker_rows = marker_rows[1:]

    command_speed: list[float] = []
    spike_times: list[float] = []
    for record in read_ros2_messages(
        mcap_path, topics=["/softtouch/policy/observation"], **kwargs
    ):
        observation = np.asarray(record.ros_msg.data, dtype=float)
        command_speed.append(float(observation[speed_slice][0]))
        if float(np.linalg.norm(observation[ball_velocity_slice])) > 5.0:
            spike_times.append(record.log_time_ns * 1.0e-9)

    position_steps = np.linalg.norm(np.diff(ball[:, 1:3], axis=0), axis=1)
    bad_indices = np.where(position_steps > 0.5)[0]
    bad_time = float(ball[bad_indices[0] + 1, 0]) if len(bad_indices) else None

    ct_time = np.asarray([sample.time for sample in marker_rows], dtype=float)
    ct = np.asarray(
        [
            nearest_polyline_distance(sample.ball_xy, sample.route_xy)
            for sample in marker_rows
        ],
        dtype=float,
    )
    return Episode(
        run=run,
        number=number,
        start=start,
        end=end,
        ball_time=ball[:, 0],
        ball_xy=ball[:, 1:3],
        markers=marker_rows,
        command_speed=np.asarray(command_speed),
        velocity_spike_time=np.asarray(spike_times),
        bad_time=bad_time,
        ct_time=ct_time,
        ct=ct,
    )


def load_all_episodes(log_root: Path) -> list[Episode]:
    bag_root = log_root / "softtouch_bags"
    requested = {
        "205524": bag_root
        / "softtouch_real_20260727_205524"
        / "softtouch_real_20260727_205524_0.mcap",
        "224829": bag_root
        / "softtouch_real_20260727_224829"
        / "softtouch_real_20260727_224829_0.mcap",
    }
    episodes: list[Episode] = []
    for run, mcap_path in requested.items():
        if not mcap_path.exists():
            raise FileNotFoundError(mcap_path)
        schema = load_schema(mcap_path)
        speed_slice = latest_term_slice(schema, "target_speed")
        ball_velocity_slice = latest_term_slice(schema, "ball_lin_vel_b")
        for number, (start, end) in enumerate(active_ranges(mcap_path), 1):
            episodes.append(
                load_episode(
                    mcap_path,
                    run,
                    number,
                    start,
                    end,
                    speed_slice,
                    ball_velocity_slice,
                )
            )
    if len(episodes) != 16:
        raise RuntimeError(f"Expected 16 active episodes, found {len(episodes)}")
    return episodes


def colored_track(
    ax: plt.Axes,
    xy: np.ndarray,
    mask: np.ndarray,
    *,
    linewidth: float,
    zorder: int = 4,
) -> None:
    reliable = xy[mask]
    if len(reliable) < 2:
        return
    points = reliable[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    progress = np.linspace(0.0, 1.0, len(segments))
    collection = LineCollection(
        segments,
        cmap="viridis",
        norm=mpl.colors.Normalize(0.0, 1.0),
        linewidth=linewidth,
        zorder=zorder,
    )
    collection.set_array(progress)
    ax.add_collection(collection)


def spaced_markers(
    episode: Episode, interval_s: float
) -> list[MarkerSample]:
    selected: list[MarkerSample] = []
    next_time = -np.inf
    for sample in episode.markers:
        if episode.bad_time is not None and sample.time >= episode.bad_time:
            break
        if sample.time >= next_time:
            selected.append(sample)
            next_time = sample.time + interval_s
    return selected


def add_command_arrows(
    ax: plt.Axes,
    episode: Episode,
    *,
    interval_s: float,
    show_preview: bool,
) -> None:
    samples = spaced_markers(episode, interval_s)
    if not samples:
        return
    current = np.asarray([sample.current_arrow for sample in samples])
    current_delta = current[:, 1] - current[:, 0]
    ax.quiver(
        current[:, 0, 0],
        current[:, 0, 1],
        current_delta[:, 0],
        current_delta[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.006,
        headwidth=4.0,
        headlength=5.0,
        color="#0891b2",
        alpha=0.78,
        zorder=5,
    )
    if show_preview:
        preview = np.asarray([sample.preview_arrow for sample in samples])
        preview_delta = preview[:, 1] - preview[:, 0]
        ax.quiver(
            preview[:, 0, 0],
            preview[:, 0, 1],
            preview_delta[:, 0],
            preview_delta[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
            headwidth=4.0,
            headlength=5.0,
            color="#f59e0b",
            alpha=0.58,
            zorder=4,
        )


def add_time_labels(ax: plt.Axes, episode: Episode, every_s: float = 2.0) -> None:
    relative = episode.ball_time - episode.start
    reliable = episode.reliable_mask
    for target in np.arange(every_s, max(relative[reliable], default=0.0), every_s):
        valid_indices = np.where(reliable)[0]
        index = valid_indices[int(np.argmin(np.abs(relative[reliable] - target)))]
        point = episode.ball_xy[index]
        ax.text(
            point[0],
            point[1],
            f"{target:.0f}s",
            fontsize=7,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.7},
            zorder=7,
        )


def plot_episode_xy(
    ax: plt.Axes,
    episode: Episode,
    *,
    arrow_interval_s: float,
    show_preview: bool,
    show_time_labels: bool,
) -> None:
    route = episode.route_for_plot
    ax.plot(
        route[:, 0],
        route[:, 1],
        color="#111827",
        linewidth=1.6,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    reliable = episode.reliable_mask
    colored_track(ax, episode.ball_xy, reliable, linewidth=2.6)
    if not np.all(reliable):
        invalid = episode.ball_xy[~reliable]
        if len(invalid):
            previous = episode.ball_xy[np.where(reliable)[0][-1] : np.where(reliable)[0][-1] + 1]
            invalid = np.vstack([previous, invalid])
            ax.plot(
                invalid[:, 0],
                invalid[:, 1],
                color="#dc2626",
                linewidth=2.0,
                linestyle="--",
                zorder=6,
            )
            ax.scatter(
                invalid[1, 0],
                invalid[1, 1],
                marker="x",
                s=50,
                linewidth=2.0,
                color="#dc2626",
                zorder=7,
            )

    start_index = np.where(reliable)[0][0]
    end_index = np.where(reliable)[0][-1]
    ax.scatter(
        episode.ball_xy[start_index, 0],
        episode.ball_xy[start_index, 1],
        s=34,
        marker="o",
        facecolor="#22c55e",
        edgecolor="white",
        linewidth=0.8,
        zorder=8,
    )
    ax.scatter(
        episode.ball_xy[end_index, 0],
        episode.ball_xy[end_index, 1],
        s=38,
        marker="s",
        facecolor="#ef4444",
        edgecolor="white",
        linewidth=0.8,
        zorder=8,
    )
    add_command_arrows(
        ax,
        episode,
        interval_s=arrow_interval_s,
        show_preview=show_preview,
    )
    if show_time_labels:
        add_time_labels(ax, episode)

    all_xy = np.vstack([route, episode.ball_xy])
    minimum = np.min(all_xy, axis=0)
    maximum = np.max(all_xy, axis=0)
    center = 0.5 * (minimum + maximum)
    span = max(float(np.max(maximum - minimum)), 0.7)
    pad = 0.10 * span
    ax.set_xlim(center[0] - 0.5 * span - pad, center[0] + 0.5 * span + pad)
    ax.set_ylim(center[1] - 0.5 * span - pad, center[1] + 0.5 * span + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    speed = float(np.median(episode.command_speed))
    bad_suffix = "  [position jump]" if episode.bad_time is not None else ""
    ax.set_title(
        f"{episode.label}  mean CT={episode.ct_mean*100:.1f}cm"
        f"  final={episode.ct_final*100:.1f}cm\n"
        f"v_cmd={speed:.2f}m/s  duration={episode.end-episode.start:.1f}s{bad_suffix}"
    )


def save_all_episode_map(episodes: list[Episode], output_dir: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(18, 17))
    for ax, episode in zip(axes.flat, episodes):
        plot_episode_xy(
            ax,
            episode,
            arrow_interval_s=1.3,
            show_preview=False,
            show_time_labels=False,
        )
    for row in range(4):
        axes[row, 0].set_ylabel("world y (m)")
    for col in range(4):
        axes[-1, col].set_xlabel("world x (m)")
    fig.suptitle(
        "2026-07-27 SoftTouch: Deployed Command Route vs. Recorded Ball Trajectory "
        "(16 Active Episodes)\n"
        "Trajectory color: purple -> yellow = early -> late; arrows = current command direction",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    handles = [
        Line2D([0], [0], color="#111827", linestyle="--", linewidth=1.8, label="Commanded route"),
        Line2D([0], [0], color="#0891b2", linewidth=2.2, label="Current command direction"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#22c55e", label="Ball start"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#ef4444", label="Reliable endpoint"),
        Line2D([0], [0], color="#dc2626", linestyle="--", linewidth=2.0, label="Unreliable mocap segment"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.925, bottom=0.065, hspace=0.32, wspace=0.22)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"cmd_route_vs_ball_all_episodes.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def save_ct_timeline(episodes: list[Episode], output_dir: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(18, 13), sharex=True, sharey=True)
    for ax, episode in zip(axes.flat, episodes):
        relative = episode.ct_time - episode.start
        reliable = episode.reliable_ct_mask
        ax.axhspan(0.0, 0.2, color="#22c55e", alpha=0.06)
        ax.axhspan(0.2, 0.5, color="#f59e0b", alpha=0.06)
        ax.axhspan(0.5, 1.25, color="#ef4444", alpha=0.045)
        ax.plot(relative[reliable], episode.ct[reliable], color="#2563eb", linewidth=2.0)
        if not np.all(reliable):
            ax.plot(
                relative[~reliable],
                episode.ct[~reliable],
                color="#dc2626",
                linestyle="--",
                linewidth=1.8,
            )
            ax.axvline(
                episode.bad_time - episode.start,
                color="#dc2626",
                linestyle=":",
                linewidth=1.4,
            )
        for spike_time in episode.velocity_spike_time:
            index = int(np.argmin(np.abs(episode.ct_time - spike_time)))
            ax.scatter(
                relative[index],
                episode.ct[index],
                marker="^",
                s=32,
                facecolor="#d946ef",
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
            )
        ax.axhline(0.2, color="#16a34a", linewidth=0.8, linestyle=":")
        ax.axhline(0.5, color="#d97706", linewidth=0.8, linestyle=":")
        ax.axhline(0.8, color="#dc2626", linewidth=0.8, linestyle=":")
        ax.grid(True)
        spike_suffix = (
            f"  velocity spikes x{len(episode.velocity_spike_time)}"
            if len(episode.velocity_spike_time)
            else ""
        )
        ax.set_title(
            f"{episode.label}  mean={episode.ct_mean*100:.1f}cm"
            f"  final={episode.ct_final*100:.1f}cm{spike_suffix}"
        )
    for row in range(4):
        axes[row, 0].set_ylabel("CT (m)")
    for col in range(4):
        axes[-1, col].set_xlabel("Time since activation (s)")
    axes[0, 0].set_xlim(0.0, 9.5)
    axes[0, 0].set_ylim(0.0, 1.25)
    fig.suptitle(
        "Cross-Track Error over Time "
        "(purple triangles: ball-speed observations > 5 m/s received by the policy)",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    handles = [
        Line2D([0], [0], color="#2563eb", linewidth=2.0, label="Reliable CT"),
        Line2D([0], [0], color="#dc2626", linestyle="--", linewidth=2.0, label="Unreliable mocap segment"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#d946ef", label="Ball-speed observation spike"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.925, bottom=0.08, hspace=0.34, wspace=0.18)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"cross_track_timeline_all_episodes.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def save_focus_map(episodes: list[Episode], output_dir: Path) -> None:
    wanted = [
        ("205524", 2),
        ("224829", 1),
        ("224829", 2),
        ("224829", 3),
        ("224829", 5),
        ("224829", 8),
    ]
    index = {(episode.run, episode.number): episode for episode in episodes}
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    captions = [
        "Moderate late drift",
        "Late ball escape + chest-mocap dropout",
        "Best CT; data trusted only before 1.456 m jump",
        "Robot and ball drift together",
        "Largest sustained route drift",
        "Low-CT reference",
    ]
    for ax, key, caption in zip(axes.flat, wanted, captions):
        episode = index[key]
        plot_episode_xy(
            ax,
            episode,
            arrow_interval_s=0.8,
            show_preview=True,
            show_time_labels=True,
        )
        ax.text(
            0.02,
            0.02,
            caption,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
            bbox={"boxstyle": "round,pad=0.25", "fc": "#f8fafc", "ec": "#cbd5e1", "alpha": 0.92},
            zorder=10,
        )
    axes[0, 0].set_ylabel("world y (m)")
    axes[1, 0].set_ylabel("world y (m)")
    for ax in axes[-1]:
        ax.set_xlabel("world x (m)")
    fig.suptitle(
        "Representative Episodes: Commanded Route, Current/Preview Directions, "
        "and Recorded Ball Trajectory",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    handles = [
        Line2D([0], [0], color="#111827", linestyle="--", linewidth=1.8, label="Commanded route"),
        Line2D([0], [0], color="#0891b2", linewidth=2.2, label="Current direction"),
        Line2D([0], [0], color="#f59e0b", linewidth=2.0, label="Preview direction"),
        Line2D([0], [0], color="#dc2626", linestyle="--", linewidth=2.0, label="Mocap anomaly"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.075, hspace=0.28, wspace=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"cmd_route_vs_ball_focus.{suffix}",
            dpi=240 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def print_summary(episodes: list[Episode], output_dir: Path) -> None:
    all_ct = np.concatenate(
        [episode.ct[episode.reliable_ct_mask] for episode in episodes]
    )
    print(f"wrote plots to {output_dir}")
    print(
        "CT point-weighted mean/median/RMS/P90 = "
        f"{np.mean(all_ct):.4f} / {np.median(all_ct):.4f} / "
        f"{np.sqrt(np.mean(all_ct * all_ct)):.4f} / {np.quantile(all_ct, .9):.4f} m"
    )
    for episode in episodes:
        print(
            f"{episode.label}: mean={episode.ct_mean:.3f} m "
            f"final={episode.ct_final:.3f} m "
            f"v_cmd={np.median(episode.command_speed):.3f} m/s "
            f"velocity_spikes={len(episode.velocity_spike_time)} "
            f"position_jump={'yes' if episode.bad_time is not None else 'no'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_all_episodes(args.log_root)
    save_all_episode_map(episodes, args.output_dir)
    save_ct_timeline(episodes, args.output_dir)
    save_focus_map(episodes, args.output_dir)
    print_summary(episodes, args.output_dir)


if __name__ == "__main__":
    main()
