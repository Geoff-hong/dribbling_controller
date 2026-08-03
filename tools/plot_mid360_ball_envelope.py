#!/usr/bin/env python3
"""Analyze trusted humanoid-relative ball motion for MID-360 placement.

The script uses the 16 active episodes in the two valid 2026-07-27 hardware
bags.  Ball samples after a mocap position jump larger than 0.5 m are excluded.
Chest poses are interpolated at the ball mocap timestamps, and the result is
expressed in the policy observation frame (+x forward, +y left, +z up).

The MID-360 origin and orientation are read from the robot_description stored
in the bag.  Its origin is converted from torso_link into the observation frame
using the ONNX observation schema stored in the same bag.  Mount-pitch coverage
uses the official native vertical FOV of -7 to +52 degrees.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
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

BALL_RADIUS_M = 0.10
MID360_FOV_LOW_DEG = -7.0
MID360_FOV_HIGH_DEG = 52.0
SYNC_LIMIT_S = 0.03
ROBUST_TAIL_FRACTION = 0.005


@dataclass(frozen=True)
class SensorGeometry:
    observation_body: str
    observation_offset_body: np.ndarray
    sensor_parent_body: str
    sensor_origin_body: np.ndarray
    sensor_origin_observation: np.ndarray
    sensor_rotation_body: np.ndarray
    sensor_rpy_body: np.ndarray

    @property
    def current_down_pitch_deg(self) -> float:
        forward = self.sensor_rotation_body[:, 0]
        return float(
            np.degrees(
                np.arctan2(
                    -forward[2],
                    np.hypot(forward[0], forward[1]),
                )
            )
        )

    @property
    def is_inverted(self) -> bool:
        return bool(self.sensor_rotation_body[2, 2] < 0.0)


@dataclass
class RelativeEpisode:
    run: str
    number: int
    start: float
    end: float
    log_time: np.ndarray
    relative_time: np.ndarray
    ball_world: np.ndarray
    ball_observation: np.ndarray
    ball_sensor_observation: np.ndarray
    sensor_range: np.ndarray
    robot_azimuth_deg: np.ndarray
    angular_radius_deg: np.ndarray
    position_jump_m: float | None
    rejected_sync_samples: int
    chest_position_jumps: int
    chest_orientation_jumps: int

    @property
    def label(self) -> str:
        return f"{self.run} / E{self.number:02d}"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
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
    return [
        (float(times[first]), float(times[last - 1]))
        for first, last in zip(cuts[:-1], cuts[1:])
    ]


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def inverted_mount_rotation(down_pitch_deg: float) -> np.ndarray:
    """Sensor-to-body rotation for the logged inverted, forward-facing layout."""
    down_pitch = np.radians(down_pitch_deg)
    return rpy_matrix(np.asarray([0.0, np.pi - down_pitch, np.pi]))


def load_sensor_geometry(mcap_path: Path) -> SensorGeometry:
    schema_record = next(
        read_ros2_messages(
            mcap_path, topics=["/softtouch/policy/observation_schema"]
        )
    )
    schema = json.loads(schema_record.ros_msg.data)
    observation_body = str(schema["obs_frame_body"])
    observation_offset = np.asarray(schema["obs_frame_offset"], dtype=float)

    description_record = next(
        read_ros2_messages(mcap_path, topics=["/robot_description"])
    )
    robot = ET.fromstring(description_record.ros_msg.data)
    joint = robot.find("./joint[@name='mid360_joint']")
    if joint is None:
        raise RuntimeError("mid360_joint is absent from robot_description")
    parent = joint.find("parent")
    origin = joint.find("origin")
    if parent is None or origin is None:
        raise RuntimeError("mid360_joint has incomplete parent/origin data")
    sensor_parent = str(parent.attrib["link"])
    sensor_xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    sensor_rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
    if observation_body != sensor_parent:
        raise RuntimeError(
            "Observation frame and MID-360 do not share a parent body: "
            f"{observation_body} vs {sensor_parent}"
        )
    return SensorGeometry(
        observation_body=observation_body,
        observation_offset_body=observation_offset,
        sensor_parent_body=sensor_parent,
        sensor_origin_body=sensor_xyz,
        sensor_origin_observation=sensor_xyz - observation_offset,
        sensor_rotation_body=rpy_matrix(sensor_rpy),
        sensor_rpy_body=sensor_rpy,
    )


def read_pose_rows(
    mcap_path: Path,
    topic: str,
    start: float,
    end: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    log_time: list[float] = []
    header_time: list[float] = []
    position: list[tuple[float, float, float]] = []
    quaternion: list[tuple[float, float, float, float]] = []
    kwargs = {
        "start_time": int((start - 0.05) * 1.0e9),
        "end_time": int((end + 0.05) * 1.0e9),
    }
    for record in read_ros2_messages(mcap_path, topics=[topic], **kwargs):
        pose = record.ros_msg.pose
        log_time.append(record.log_time_ns * 1.0e-9)
        header_time.append(stamp_to_seconds(record.ros_msg.header.stamp))
        position.append(
            (
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            )
        )
        quaternion.append(
            (
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            )
        )
    if not log_time:
        raise RuntimeError(f"No {topic} messages in requested range")
    return (
        np.asarray(log_time),
        np.asarray(header_time),
        np.asarray(position),
        np.asarray(quaternion),
    )


def quaternion_step_deg(quaternion: np.ndarray) -> np.ndarray:
    if len(quaternion) < 2:
        return np.empty(0)
    dot = np.abs(np.sum(quaternion[:-1] * quaternion[1:], axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def interpolate_frame_pose(
    sample_time: np.ndarray,
    frame_time: np.ndarray,
    frame_position: np.ndarray,
    frame_quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(frame_time)
    frame_time = frame_time[order]
    frame_position = frame_position[order]
    frame_quaternion = frame_quaternion[order]
    unique = np.r_[True, np.diff(frame_time) > 1.0e-9]
    frame_time = frame_time[unique]
    frame_position = frame_position[unique]
    frame_quaternion = frame_quaternion[unique]

    upper = np.searchsorted(frame_time, sample_time, side="right")
    lower = upper - 1
    valid = (lower >= 0) & (upper < len(frame_time))
    safe_lower = np.clip(lower, 0, len(frame_time) - 1)
    safe_upper = np.clip(upper, 0, len(frame_time) - 1)
    t0 = frame_time[safe_lower]
    t1 = frame_time[safe_upper]
    span = t1 - t0
    valid &= span > 1.0e-9
    valid &= np.maximum(sample_time - t0, t1 - sample_time) <= SYNC_LIMIT_S
    weight = np.divide(
        sample_time - t0,
        span,
        out=np.zeros_like(sample_time),
        where=span > 1.0e-9,
    )
    weight = np.clip(weight, 0.0, 1.0)

    position = (
        (1.0 - weight[:, None]) * frame_position[safe_lower]
        + weight[:, None] * frame_position[safe_upper]
    )
    q0 = frame_quaternion[safe_lower]
    q1 = frame_quaternion[safe_upper].copy()
    flip = np.sum(q0 * q1, axis=1) < 0.0
    q1[flip] *= -1.0
    quaternion = (1.0 - weight[:, None]) * q0 + weight[:, None] * q1
    norm = np.linalg.norm(quaternion, axis=1)
    valid &= norm > 1.0e-9
    quaternion = np.divide(
        quaternion,
        norm[:, None],
        out=np.zeros_like(quaternion),
        where=norm[:, None] > 1.0e-9,
    )
    return position, quaternion, valid


def rotate_world_to_frame(
    frame_quaternion_xyzw: np.ndarray, vector_world: np.ndarray
) -> np.ndarray:
    inverse_xyz = -frame_quaternion_xyzw[:, :3]
    inverse_w = frame_quaternion_xyzw[:, 3]
    twice_cross = 2.0 * np.cross(inverse_xyz, vector_world)
    return (
        vector_world
        + inverse_w[:, None] * twice_cross
        + np.cross(inverse_xyz, twice_cross)
    )


def load_relative_episode(
    mcap_path: Path,
    run: str,
    number: int,
    start: float,
    end: float,
    geometry: SensorGeometry,
) -> RelativeEpisode:
    ball_log, ball_header, ball_position, _ = read_pose_rows(
        mcap_path, "/softtouch/mocap/ball/pose", start, end
    )
    chest_log, chest_header, chest_position, chest_quaternion = read_pose_rows(
        mcap_path, "/softtouch/mocap/chest/pose", start, end
    )

    active_ball = (ball_log >= start) & (ball_log <= end)
    ball_log = ball_log[active_ball]
    ball_header = ball_header[active_ball]
    ball_position = ball_position[active_ball]

    ball_step = np.linalg.norm(np.diff(ball_position, axis=0), axis=1)
    bad_indices = np.where(ball_step > 0.5)[0]
    position_jump = float(ball_step[bad_indices[0]]) if len(bad_indices) else None
    if len(bad_indices):
        reliable = np.arange(len(ball_log)) <= bad_indices[0]
        ball_log = ball_log[reliable]
        ball_header = ball_header[reliable]
        ball_position = ball_position[reliable]

    chest_dt = np.diff(chest_header)
    chest_step = np.linalg.norm(np.diff(chest_position, axis=0), axis=1)
    chest_angle = quaternion_step_deg(chest_quaternion)
    quick = (chest_dt > 1.0e-6) & (chest_dt <= 0.05)
    chest_position_jumps = int(np.count_nonzero(quick & (chest_step > 0.12)))
    chest_orientation_jumps = int(np.count_nonzero(quick & (chest_angle > 35.0)))

    frame_position, frame_quaternion, synced = interpolate_frame_pose(
        ball_header,
        chest_header,
        chest_position,
        chest_quaternion,
    )
    rejected_sync_samples = int(np.count_nonzero(~synced))
    ball_log = ball_log[synced]
    ball_position = ball_position[synced]
    frame_position = frame_position[synced]
    frame_quaternion = frame_quaternion[synced]

    ball_observation = rotate_world_to_frame(
        frame_quaternion,
        ball_position - frame_position,
    )
    ball_sensor = ball_observation - geometry.sensor_origin_observation
    sensor_range = np.linalg.norm(ball_sensor, axis=1)
    robot_azimuth = np.degrees(
        np.arctan2(ball_sensor[:, 1], ball_sensor[:, 0])
    )
    angular_radius = np.degrees(
        np.arcsin(np.clip(BALL_RADIUS_M / sensor_range, 0.0, 1.0))
    )
    return RelativeEpisode(
        run=run,
        number=number,
        start=start,
        end=end,
        log_time=ball_log,
        relative_time=ball_log - start,
        ball_world=ball_position,
        ball_observation=ball_observation,
        ball_sensor_observation=ball_sensor,
        sensor_range=sensor_range,
        robot_azimuth_deg=robot_azimuth,
        angular_radius_deg=angular_radius,
        position_jump_m=position_jump,
        rejected_sync_samples=rejected_sync_samples,
        chest_position_jumps=chest_position_jumps,
        chest_orientation_jumps=chest_orientation_jumps,
    )


def load_all(
    log_root: Path,
) -> tuple[list[RelativeEpisode], SensorGeometry]:
    bag_root = log_root / "softtouch_bags"
    requested = {
        "205524": bag_root
        / "softtouch_real_20260727_205524"
        / "softtouch_real_20260727_205524_0.mcap",
        "224829": bag_root
        / "softtouch_real_20260727_224829"
        / "softtouch_real_20260727_224829_0.mcap",
    }
    geometry: SensorGeometry | None = None
    episodes: list[RelativeEpisode] = []
    for run, mcap_path in requested.items():
        if not mcap_path.exists():
            raise FileNotFoundError(mcap_path)
        run_geometry = load_sensor_geometry(mcap_path)
        if geometry is None:
            geometry = run_geometry
        else:
            if not np.allclose(
                geometry.sensor_origin_observation,
                run_geometry.sensor_origin_observation,
                atol=1.0e-9,
            ) or not np.allclose(
                geometry.sensor_rotation_body,
                run_geometry.sensor_rotation_body,
                atol=1.0e-9,
            ):
                raise RuntimeError("MID-360 geometry differs between bags")
        for number, (start, end) in enumerate(active_ranges(mcap_path), 1):
            episodes.append(
                load_relative_episode(
                    mcap_path,
                    run,
                    number,
                    start,
                    end,
                    run_geometry,
                )
            )
    if geometry is None:
        raise RuntimeError("No geometry loaded")
    if len(episodes) != 16:
        raise RuntimeError(f"Expected 16 active episodes, found {len(episodes)}")
    return episodes, geometry


def native_elevation_deg(
    vector_body: np.ndarray, sensor_rotation_body: np.ndarray
) -> np.ndarray:
    vector_sensor = vector_body @ sensor_rotation_body
    return np.degrees(
        np.arctan2(
            vector_sensor[:, 2],
            np.hypot(vector_sensor[:, 0], vector_sensor[:, 1]),
        )
    )


def vertical_margin_deg(
    vector_body: np.ndarray,
    angular_radius_deg: np.ndarray,
    sensor_rotation_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    elevation = native_elevation_deg(vector_body, sensor_rotation_body)
    lower_margin = (
        elevation - angular_radius_deg - MID360_FOV_LOW_DEG
    )
    upper_margin = (
        MID360_FOV_HIGH_DEG - elevation - angular_radius_deg
    )
    return elevation, np.minimum(lower_margin, upper_margin)


def optimize_inverted_pitch(
    vector_body: np.ndarray, angular_radius_deg: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    pitch_grid = np.linspace(-5.0, 45.0, 1001)
    coverage = np.empty_like(pitch_grid)
    for index, pitch in enumerate(pitch_grid):
        _, margin = vertical_margin_deg(
            vector_body,
            angular_radius_deg,
            inverted_mount_rotation(float(pitch)),
        )
        coverage[index] = 100.0 * np.mean(margin >= 0.0)
    best = float(pitch_grid[int(np.argmax(coverage))])
    practical = round(best * 2.0) / 2.0
    return practical, pitch_grid, coverage


def sample_arrays(
    episodes: list[RelativeEpisode],
) -> dict[str, np.ndarray]:
    return {
        "ball_observation": np.concatenate(
            [episode.ball_observation for episode in episodes]
        ),
        "ball_sensor": np.concatenate(
            [episode.ball_sensor_observation for episode in episodes]
        ),
        "range": np.concatenate([episode.sensor_range for episode in episodes]),
        "azimuth": np.concatenate(
            [episode.robot_azimuth_deg for episode in episodes]
        ),
        "angular_radius": np.concatenate(
            [episode.angular_radius_deg for episode in episodes]
        ),
    }


def latest_term_slice(schema: dict[str, Any], term_name: str) -> slice:
    cursor = 0
    for name, dimension, history in zip(
        schema["observation_names"],
        schema["observation_dims"],
        schema["history_lengths"],
    ):
        width = int(dimension) * int(history)
        if name == term_name:
            start = cursor + width - int(dimension)
            return slice(start, start + int(dimension))
        cursor += width
    raise KeyError(term_name)


def validate_against_policy_observation(
    episodes: list[RelativeEpisode], log_root: Path
) -> dict[str, Any]:
    bag_root = log_root / "softtouch_bags"
    paths = {
        "205524": bag_root
        / "softtouch_real_20260727_205524"
        / "softtouch_real_20260727_205524_0.mcap",
        "224829": bag_root
        / "softtouch_real_20260727_224829"
        / "softtouch_real_20260727_224829_0.mcap",
    }
    errors: list[np.ndarray] = []
    time_delta: list[np.ndarray] = []
    for run, path in paths.items():
        schema_record = next(
            read_ros2_messages(
                path, topics=["/softtouch/policy/observation_schema"]
            )
        )
        schema = json.loads(schema_record.ros_msg.data)
        ball_slice = latest_term_slice(schema, "ball_pos_b")
        for episode in [item for item in episodes if item.run == run]:
            observation_time: list[float] = []
            observed_ball: list[np.ndarray] = []
            for record in read_ros2_messages(
                path,
                topics=["/softtouch/policy/observation"],
                start_time=int(episode.start * 1.0e9),
                end_time=int(episode.end * 1.0e9),
            ):
                observation_time.append(record.log_time_ns * 1.0e-9)
                observed_ball.append(
                    np.asarray(record.ros_msg.data, dtype=float)[ball_slice]
                )
            if not observation_time:
                continue
            observation_time_array = np.asarray(observation_time)
            observed_ball_array = np.asarray(observed_ball)
            upper = np.searchsorted(
                episode.log_time, observation_time_array, side="left"
            )
            lower = np.clip(upper - 1, 0, len(episode.log_time) - 1)
            upper = np.clip(upper, 0, len(episode.log_time) - 1)
            lower_delta = np.abs(
                episode.log_time[lower] - observation_time_array
            )
            upper_delta = np.abs(
                episode.log_time[upper] - observation_time_array
            )
            choose_upper = upper_delta < lower_delta
            nearest = np.where(choose_upper, upper, lower)
            delta = np.minimum(lower_delta, upper_delta)
            valid = delta <= SYNC_LIMIT_S
            errors.append(
                observed_ball_array[valid]
                - episode.ball_observation[nearest[valid]]
            )
            time_delta.append(delta[valid])
    error = np.concatenate(errors)
    delta = np.concatenate(time_delta)
    error_norm = np.linalg.norm(error, axis=1)
    return {
        "paired_policy_ticks": int(len(error)),
        "nearest_mocap_time_delta_ms": {
            "median": float(1000.0 * np.median(delta)),
            "q99": float(1000.0 * np.quantile(delta, 0.99)),
        },
        "ball_pos_b_reconstruction_error_m": {
            "median_norm": float(np.median(error_norm)),
            "q95_norm": float(np.quantile(error_norm, 0.95)),
            "q95_abs_xyz": np.quantile(np.abs(error), 0.95, axis=0).tolist(),
        },
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    levels = {
        "min": 0.0,
        "q00_5": 0.005,
        "q01": 0.01,
        "q05": 0.05,
        "median": 0.5,
        "q95": 0.95,
        "q99": 0.99,
        "q99_5": 0.995,
        "max": 1.0,
    }
    return {
        name: float(np.quantile(values, level))
        for name, level in levels.items()
    }


def coverage_stats(
    episodes: list[RelativeEpisode],
    sensor_rotation: np.ndarray,
) -> dict[str, Any]:
    per_episode: list[dict[str, Any]] = []
    all_elevation: list[np.ndarray] = []
    all_margin: list[np.ndarray] = []
    for episode in episodes:
        elevation, margin = vertical_margin_deg(
            episode.ball_sensor_observation,
            episode.angular_radius_deg,
            sensor_rotation,
        )
        center_inside = (
            (elevation >= MID360_FOV_LOW_DEG)
            & (elevation <= MID360_FOV_HIGH_DEG)
        )
        full_inside = margin >= 0.0
        per_episode.append(
            {
                "episode": episode.label,
                "center_coverage_percent": float(100.0 * np.mean(center_inside)),
                "full_ball_coverage_percent": float(100.0 * np.mean(full_inside)),
                "minimum_margin_deg": float(np.min(margin)),
                "q00_5_margin_deg": float(
                    np.quantile(margin, ROBUST_TAIL_FRACTION)
                ),
            }
        )
        all_elevation.append(elevation)
        all_margin.append(margin)
    elevation = np.concatenate(all_elevation)
    margin = np.concatenate(all_margin)
    return {
        "center_coverage_percent": float(
            100.0
            * np.mean(
                (elevation >= MID360_FOV_LOW_DEG)
                & (elevation <= MID360_FOV_HIGH_DEG)
            )
        ),
        "full_ball_coverage_percent": float(100.0 * np.mean(margin >= 0.0)),
        "elevation_deg": quantiles(elevation),
        "vertical_margin_deg": quantiles(margin),
        "per_episode": per_episode,
    }


def colored_relative_track(ax: plt.Axes, episode: RelativeEpisode) -> None:
    xy = episode.ball_observation[:, :2]
    if len(xy) < 2:
        return
    points = xy[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    collection = LineCollection(
        segments,
        cmap="viridis",
        norm=mpl.colors.Normalize(0.0, 1.0),
        linewidth=2.0,
        alpha=0.9,
        zorder=3,
    )
    collection.set_array(np.linspace(0.0, 1.0, len(segments)))
    ax.add_collection(collection)
    ax.scatter(
        xy[0, 0],
        xy[0, 1],
        marker="o",
        s=28,
        facecolor="#22c55e",
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )
    ax.scatter(
        xy[-1, 0],
        xy[-1, 1],
        marker="s",
        s=30,
        facecolor="#ef4444",
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )


def save_relative_trajectories(
    episodes: list[RelativeEpisode],
    geometry: SensorGeometry,
    output_dir: Path,
) -> None:
    all_xy = np.concatenate(
        [episode.ball_observation[:, :2] for episode in episodes]
    )
    low = np.quantile(all_xy, 0.002, axis=0)
    high = np.quantile(all_xy, 0.998, axis=0)
    span = high - low
    pad = np.maximum(0.12 * span, 0.12)
    fig, axes = plt.subplots(4, 4, figsize=(16, 15), sharex=True, sharey=True)
    for ax, episode in zip(axes.flat, episodes):
        colored_relative_track(ax, episode)
        ax.scatter(
            0.0,
            0.0,
            marker="+",
            s=70,
            linewidth=1.8,
            color="#111827",
            zorder=6,
        )
        ax.scatter(
            geometry.sensor_origin_observation[0],
            geometry.sensor_origin_observation[1],
            marker="D",
            s=28,
            facecolor="#7c3aed",
            edgecolor="white",
            linewidth=0.6,
            zorder=6,
        )
        x = episode.ball_observation[:, 0]
        y = episode.ball_observation[:, 1]
        ax.set_title(
            f"{episode.label}  duration={episode.relative_time[-1]:.1f}s\n"
            f"x50={np.median(x):.2f}m  |y|95={np.quantile(np.abs(y), .95):.2f}m"
        )
        ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
        ax.set_ylim(low[1] - pad[1], high[1] + pad[1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
    for row in range(4):
        axes[row, 0].set_ylabel("Left y in humanoid frame (m)")
    for column in range(4):
        axes[-1, column].set_xlabel("Forward x in humanoid frame (m)")
    fig.suptitle(
        "Trusted Ball Trajectories Relative to the Humanoid "
        "(purple -> yellow = early -> late)",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="+",
            color="#111827",
            linestyle="none",
            markersize=9,
            label="Humanoid observation-frame origin",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="#7c3aed",
            markersize=6,
            label="MID-360 origin",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#22c55e",
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#ef4444",
            label="Reliable endpoint",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(
        left=0.065,
        right=0.99,
        top=0.925,
        bottom=0.065,
        hspace=0.30,
        wspace=0.16,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"humanoid_ball_relative_trajectories.{suffix}",
            dpi=230 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def save_mount_coverage(
    episodes: list[RelativeEpisode],
    geometry: SensorGeometry,
    recommended_pitch_deg: float,
    pitch_grid: np.ndarray,
    pitch_coverage: np.ndarray,
    output_dir: Path,
) -> None:
    arrays = sample_arrays(episodes)
    ball_sensor = arrays["ball_sensor"]
    angular_radius = arrays["angular_radius"]
    current_rotation = geometry.sensor_rotation_body
    recommended_rotation = inverted_mount_rotation(recommended_pitch_deg)
    current_elevation, current_margin = vertical_margin_deg(
        ball_sensor, angular_radius, current_rotation
    )
    recommended_elevation, recommended_margin = vertical_margin_deg(
        ball_sensor, angular_radius, recommended_rotation
    )
    current_full = current_margin >= 0.0
    recommended_full = recommended_margin >= 0.0

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5))
    top, current_ax, recommended_ax, coverage_ax = axes.flat

    plot_stride = max(1, len(ball_sensor) // 7000)
    shown = slice(None, None, plot_stride)
    scatter = top.scatter(
        ball_sensor[shown, 0],
        ball_sensor[shown, 1],
        c=np.linalg.norm(ball_sensor[shown], axis=1),
        cmap="viridis",
        s=5,
        alpha=0.42,
        linewidth=0.0,
    )
    top.scatter(
        0.0,
        0.0,
        marker="D",
        s=65,
        facecolor="#7c3aed",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="MID-360 origin",
    )
    az_low, az_high = np.quantile(
        arrays["azimuth"], [ROBUST_TAIL_FRACTION, 1.0 - ROBUST_TAIL_FRACTION]
    )
    radial_high = float(np.quantile(np.hypot(ball_sensor[:, 0], ball_sensor[:, 1]), 0.998))
    for angle in (az_low, az_high):
        direction = np.asarray([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        top.plot(
            [0.0, radial_high * direction[0]],
            [0.0, radial_high * direction[1]],
            color="#dc2626",
            linestyle="--",
            linewidth=1.4,
        )
    top.set_aspect("equal", adjustable="box")
    top.set_xlabel("Forward from MID-360 (m)")
    top.set_ylabel("Left from MID-360 (m)")
    top.set_title(
        "Top View: Trusted Ball Envelope\n"
        f"99% center azimuth: {az_low:.1f} deg to {az_high:.1f} deg"
    )
    top.grid(True)
    colorbar = fig.colorbar(scatter, ax=top, pad=0.02)
    colorbar.set_label("3D range (m)")

    def elevation_panel(
        ax: plt.Axes,
        elevation: np.ndarray,
        margin: np.ndarray,
        title: str,
    ) -> None:
        inside = margin >= 0.0
        ax.scatter(
            arrays["azimuth"][shown][inside[shown]],
            elevation[shown][inside[shown]],
            s=5,
            color="#2563eb",
            alpha=0.32,
            linewidth=0.0,
            label="Full ball inside vertical FOV",
        )
        ax.scatter(
            arrays["azimuth"][shown][~inside[shown]],
            elevation[shown][~inside[shown]],
            s=7,
            color="#dc2626",
            alpha=0.48,
            linewidth=0.0,
            label="Ball disk clipped by FOV",
        )
        ax.axhspan(
            MID360_FOV_LOW_DEG,
            MID360_FOV_HIGH_DEG,
            color="#22c55e",
            alpha=0.07,
        )
        ax.axhline(
            MID360_FOV_LOW_DEG,
            color="#16a34a",
            linestyle="--",
            linewidth=1.0,
        )
        ax.axhline(
            MID360_FOV_HIGH_DEG,
            color="#16a34a",
            linestyle="--",
            linewidth=1.0,
        )
        ax.set_xlabel("Ball azimuth in humanoid frame (deg)")
        ax.set_ylabel("Ball-center elevation in native MID-360 frame (deg)")
        ax.set_title(title)
        ax.grid(True)

    elevation_panel(
        current_ax,
        current_elevation,
        current_margin,
        "Logged Mount\n"
        f"inverted, {geometry.current_down_pitch_deg:.1f} deg down; "
        f"full-ball coverage={100*np.mean(current_full):.1f}%",
    )
    elevation_panel(
        recommended_ax,
        recommended_elevation,
        recommended_margin,
        "Recommended Mount\n"
        f"inverted, {recommended_pitch_deg:.1f} deg down; "
        f"full-ball coverage={100*np.mean(recommended_full):.1f}%",
    )

    labels = [episode.label.replace(" / ", "\n") for episode in episodes]
    current_rates: list[float] = []
    recommended_rates: list[float] = []
    for episode in episodes:
        _, margin_current = vertical_margin_deg(
            episode.ball_sensor_observation,
            episode.angular_radius_deg,
            current_rotation,
        )
        _, margin_recommended = vertical_margin_deg(
            episode.ball_sensor_observation,
            episode.angular_radius_deg,
            recommended_rotation,
        )
        current_rates.append(100.0 * float(np.mean(margin_current >= 0.0)))
        recommended_rates.append(
            100.0 * float(np.mean(margin_recommended >= 0.0))
        )
    index = np.arange(len(episodes))
    width = 0.38
    coverage_ax.bar(
        index - width / 2,
        current_rates,
        width,
        color="#ef4444",
        alpha=0.78,
        label=f"Logged {geometry.current_down_pitch_deg:.1f} deg",
    )
    coverage_ax.bar(
        index + width / 2,
        recommended_rates,
        width,
        color="#2563eb",
        alpha=0.82,
        label=f"Recommended {recommended_pitch_deg:.1f} deg",
    )
    coverage_ax.set_xticks(index)
    coverage_ax.set_xticklabels(labels, rotation=0, fontsize=7)
    coverage_ax.set_ylim(0.0, 105.0)
    coverage_ax.set_ylabel("Full-ball vertical-FOV coverage (%)")
    coverage_ax.set_title(
        "Conservative Coverage by Episode\n"
        "(entire 0.10 m-radius angular disk inside -7 deg to +52 deg)"
    )
    coverage_ax.grid(True, axis="y")
    coverage_ax.legend(loc="lower right")

    score_ax = coverage_ax.inset_axes([0.08, 0.18, 0.35, 0.30])
    score_ax.plot(pitch_grid, pitch_coverage, color="#7c3aed", linewidth=1.5)
    score_ax.axvline(
        geometry.current_down_pitch_deg,
        color="#ef4444",
        linestyle=":",
        linewidth=1.0,
    )
    score_ax.axvline(
        recommended_pitch_deg,
        color="#2563eb",
        linestyle=":",
        linewidth=1.0,
    )
    score_ax.set_xlabel("Down pitch (deg)", fontsize=7)
    score_ax.set_ylabel("Full-ball coverage (%)", fontsize=7)
    score_ax.tick_params(labelsize=6)
    score_ax.grid(True)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#2563eb",
            markersize=6,
            label="Full ball inside vertical FOV",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#dc2626",
            markersize=6,
            label="Ball disk clipped by FOV",
        ),
        Line2D(
            [0],
            [0],
            color="#16a34a",
            linestyle="--",
            label="Native vertical-FOV edges (-7 deg, +52 deg)",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "MID-360 Mount-Angle Coverage from Trusted Hardware Ball Motion",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.91,
        bottom=0.075,
        hspace=0.31,
        wspace=0.18,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"mid360_mount_coverage.{suffix}",
            dpi=230 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_summary(
    episodes: list[RelativeEpisode],
    geometry: SensorGeometry,
    recommended_pitch_deg: float,
    policy_validation: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    arrays = sample_arrays(episodes)
    current = coverage_stats(episodes, geometry.sensor_rotation_body)
    recommended_rotation = inverted_mount_rotation(recommended_pitch_deg)
    recommended = coverage_stats(episodes, recommended_rotation)
    ball_observation = arrays["ball_observation"]
    summary: dict[str, Any] = {
        "trusted_samples": int(len(arrays["range"])),
        "active_episode_count": len(episodes),
        "trusted_duration_sum_s": float(
            sum(
                episode.relative_time[-1] - episode.relative_time[0]
                for episode in episodes
            )
        ),
        "filters": {
            "active_policy_segments_only": True,
            "post_ball_position_jump_excluded": True,
            "ball_position_jump_threshold_m": 0.5,
            "chest_ball_sync_limit_s": SYNC_LIMIT_S,
            "robust_tail_fraction": ROBUST_TAIL_FRACTION,
        },
        "coordinate_convention": {
            "frame": "policy observation frame rigidly attached to torso_link",
            "x": "forward",
            "y": "left",
            "z": "up",
        },
        "geometry_from_bag": {
            "observation_body": geometry.observation_body,
            "observation_offset_body_m": geometry.observation_offset_body.tolist(),
            "mid360_parent_body": geometry.sensor_parent_body,
            "mid360_origin_body_m": geometry.sensor_origin_body.tolist(),
            "mid360_origin_observation_m": geometry.sensor_origin_observation.tolist(),
            "mid360_rpy_body_rad": geometry.sensor_rpy_body.tolist(),
            "mid360_is_inverted": geometry.is_inverted,
            "current_forward_axis_down_pitch_deg": geometry.current_down_pitch_deg,
        },
        "trusted_ball_position_in_humanoid_frame_m": {
            "forward_x": quantiles(ball_observation[:, 0]),
            "left_y": quantiles(ball_observation[:, 1]),
            "up_z": quantiles(ball_observation[:, 2]),
        },
        "trusted_ball_position_from_mid360": {
            "range_m": quantiles(arrays["range"]),
            "humanoid_azimuth_deg": quantiles(arrays["azimuth"]),
            "angular_radius_deg_for_0_10m_ball": quantiles(
                arrays["angular_radius"]
            ),
        },
        "official_mid360_vertical_fov_deg": [
            MID360_FOV_LOW_DEG,
            MID360_FOV_HIGH_DEG,
        ],
        "current_mount": current,
        "recommended_mount": {
            "layout": "inverted, forward-facing",
            "forward_axis_down_pitch_deg": recommended_pitch_deg,
            "additional_down_pitch_from_logged_mount_deg": (
                recommended_pitch_deg - geometry.current_down_pitch_deg
            ),
            **recommended,
        },
        "data_quality": {
            "episodes_with_ball_position_jump": [
                {
                    "episode": episode.label,
                    "jump_m": episode.position_jump_m,
                }
                for episode in episodes
                if episode.position_jump_m is not None
            ],
            "rejected_unsynchronized_samples": int(
                sum(episode.rejected_sync_samples for episode in episodes)
            ),
            "detected_chest_position_jumps": int(
                sum(episode.chest_position_jumps for episode in episodes)
            ),
            "detected_chest_orientation_jumps": int(
                sum(episode.chest_orientation_jumps for episode in episodes)
            ),
            "policy_frame_validation": policy_validation,
        },
    }
    summary_path = output_dir / "mid360_mount_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    position = summary["trusted_ball_position_in_humanoid_frame_m"]
    sensor = summary["trusted_ball_position_from_mid360"]
    current = summary["current_mount"]
    recommended = summary["recommended_mount"]
    geometry = summary["geometry_from_bag"]
    quality = summary["data_quality"]
    validation = quality["policy_frame_validation"]
    print(
        f"trusted samples/episodes/duration = "
        f"{summary['trusted_samples']} / {summary['active_episode_count']} / "
        f"{summary['trusted_duration_sum_s']:.1f} s"
    )
    print(
        "humanoid-frame robust 99% x/y/z = "
        f"[{position['forward_x']['q00_5']:.3f}, {position['forward_x']['q99_5']:.3f}] / "
        f"[{position['left_y']['q00_5']:.3f}, {position['left_y']['q99_5']:.3f}] / "
        f"[{position['up_z']['q00_5']:.3f}, {position['up_z']['q99_5']:.3f}] m"
    )
    print(
        "MID-360 robust 99% range/azimuth = "
        f"[{sensor['range_m']['q00_5']:.3f}, {sensor['range_m']['q99_5']:.3f}] m / "
        f"[{sensor['humanoid_azimuth_deg']['q00_5']:.1f}, "
        f"{sensor['humanoid_azimuth_deg']['q99_5']:.1f}] deg"
    )
    print(
        "logged mount = "
        f"inverted, {geometry['current_forward_axis_down_pitch_deg']:.2f} deg down; "
        f"center/full-ball coverage = "
        f"{current['center_coverage_percent']:.1f}% / "
        f"{current['full_ball_coverage_percent']:.1f}%"
    )
    print(
        "recommended mount = "
        f"inverted, {recommended['forward_axis_down_pitch_deg']:.1f} deg down "
        f"(additional {recommended['additional_down_pitch_from_logged_mount_deg']:.1f} deg); "
        f"center/full-ball coverage = "
        f"{recommended['center_coverage_percent']:.1f}% / "
        f"{recommended['full_ball_coverage_percent']:.1f}%"
    )
    print(
        "quality: post-jump episodes/rejected sync/chest pos jumps/chest ori jumps = "
        f"{len(quality['episodes_with_ball_position_jump'])} / "
        f"{quality['rejected_unsynchronized_samples']} / "
        f"{quality['detected_chest_position_jumps']} / "
        f"{quality['detected_chest_orientation_jumps']}"
    )
    print(
        "policy ball_pos_b reconstruction median/p95 error = "
        f"{1000.0 * validation['ball_pos_b_reconstruction_error_m']['median_norm']:.1f} / "
        f"{1000.0 * validation['ball_pos_b_reconstruction_error_m']['q95_norm']:.1f} mm"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes, geometry = load_all(args.log_root)
    arrays = sample_arrays(episodes)
    recommended_pitch, pitch_grid, pitch_coverage = optimize_inverted_pitch(
        arrays["ball_sensor"],
        arrays["angular_radius"],
    )
    save_relative_trajectories(episodes, geometry, args.output_dir)
    save_mount_coverage(
        episodes,
        geometry,
        recommended_pitch,
        pitch_grid,
        pitch_coverage,
        args.output_dir,
    )
    policy_validation = validate_against_policy_observation(
        episodes, args.log_root
    )
    summary = write_summary(
        episodes,
        geometry,
        recommended_pitch,
        policy_validation,
        args.output_dir,
    )
    print_summary(summary)
    print(f"wrote MID-360 analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
