#!/usr/bin/env python3
"""Build a presentation-ready report for the six selected hardware episodes.

The report deliberately excludes the MID-360 placement study.  It combines:

* ball and humanoid cross-track error against the deployed route markers;
* trusted ball/chest mocap and data-quality checks;
* actual humanoid speed relative to the 0.4 m/s command;
* foot-sole clearance reconstructed from chest mocap, robot TF, and the exact
  G1 ankle-roll STL meshes;
* the effective phase alignment between 50 Hz joint position targets and the
  measured joint positions;
* relevant facts from the archived training configuration used by the
  deployed m55000 checkpoint.

The foot analysis uses the lower foot as a local support-plane reference.
This cancels the absolute mocap/world-height calibration error.  A swing is
flagged as "low clearance" when the complete rendered foot mesh never clears
that plane by 30 mm.  This is an engineering screening threshold, not a force
sensor measurement of physical ground contact.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from mcap_ros2.reader import read_ros2_messages
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import analyze_mid360_foot_contact_visibility as robot_geometry  # noqa: E402
import plot_mid360_ball_envelope as relative_ball  # noqa: E402
import plot_softtouch_cmd_ball as route_analysis  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "softtouch_20260727_night"
DEFAULT_LOG_ROOT = route_analysis.DEFAULT_LOG_ROOT
CHECKPOINT_DIR = Path(
    "/home/alden/Desktop/SoftTouch_dribble-sim/checkpoints/"
    "g1_dribble_s3_bodyframe_v2vel_4096_net512_mesh_s2r_iter59999"
)
MESH_ROOT = (
    Path(
        "/home/alden/Desktop/SoftTouch_dribble-sim/source/multiagent_sim/"
        "multiagent_sim/assets/unitree_description/meshes/g1"
    )
)

FOCUS_KEYS = [
    ("205524", 2),
    ("224829", 1),
    ("224829", 2),
    ("224829", 3),
    ("224829", 5),
    ("224829", 8),
]

RUN_PATHS = {
    run: DEFAULT_LOG_ROOT
    / "softtouch_bags"
    / f"softtouch_real_20260727_{run}"
    / f"softtouch_real_20260727_{run}_0.mcap"
    for run in ("205524", "224829")
}
CONTROLLER_LOGS = {
    "205524": DEFAULT_LOG_ROOT
    / "softtouch_logs"
    / "run_20260727_205523"
    / "controller_manual.log",
    "224829": DEFAULT_LOG_ROOT
    / "softtouch_logs"
    / "run_20260727_224829"
    / "controller_manual.log",
}

POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]
LEG_JOINT_INDICES = [
    index
    for index, name in enumerate(POLICY_JOINT_NAMES)
    if any(term in name for term in ("hip", "knee", "ankle"))
]

LOW_CLEARANCE_M = 0.030
SEVERE_CLEARANCE_M = 0.020
SWING_START_CLEARANCE_M = 0.003
SWING_MIN_DURATION_S = 0.18
RESAMPLE_PERIOD_S = 0.02
VELOCITY_AVERAGING_HALF_WINDOW_S = 0.20

EPISODE_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
]
LEFT_COLOR = "#0891b2"
RIGHT_COLOR = "#d946ef"
BODY_COLOR = "#f97316"
BALL_COLOR = "#2563eb"


@dataclass
class SwingEvent:
    side: str
    start_s: float
    end_s: float
    peak_time_s: float
    peak_clearance_m: float
    toe_clearance_at_peak_m: float
    heel_clearance_at_peak_m: float
    foot_pitch_down_at_peak_deg: float

    @property
    def limiting_end(self) -> str:
        return (
            "toe"
            if self.toe_clearance_at_peak_m <= self.heel_clearance_at_peak_m
            else "heel"
        )

    @property
    def low_clearance(self) -> bool:
        return self.peak_clearance_m < LOW_CLEARANCE_M

    @property
    def severe_clearance(self) -> bool:
        return self.peak_clearance_m < SEVERE_CLEARANCE_M


@dataclass
class FootDiagnostics:
    time_s: np.ndarray
    left_clearance_m: np.ndarray
    right_clearance_m: np.ndarray
    chest_drop_m: np.ndarray
    chest_tilt_deg: np.ndarray
    events: list[SwingEvent]
    chest_drop_q95_m: float
    chest_drop_max_m: float
    chest_tilt_q95_deg: float
    tf_samples: int
    max_chest_tf_sync_ms: float


@dataclass
class EpisodeDiagnostics:
    run: str
    number: int
    label: str
    active_duration_s: float
    trusted_duration_s: float
    ct_samples: int
    ball_ct_mean_m: float
    ball_ct_median_m: float
    ball_ct_rms_m: float
    ball_ct_p90_m: float
    ball_ct_p95_m: float
    ball_ct_max_m: float
    ball_ct_final_m: float
    humanoid_ct_mean_m: float
    humanoid_ct_final_m: float
    sustained_ct_020_time_s: float | None
    sustained_ct_050_time_s: float | None
    commanded_speed_mps: float
    humanoid_speed_median_mps: float
    humanoid_speed_p90_mps: float
    direction_error_median_deg: float
    direction_error_p90_deg: float
    lateral_speed_median_mps: float
    low_clearance_swings: int
    severe_clearance_swings: int
    total_swings: int
    low_clearance_toe_limited: int
    swing_peak_clearance_median_m: float
    chest_drop_q95_m: float
    chest_drop_max_m: float
    chest_tilt_q95_deg: float
    ball_position_jump_m: float | None
    ball_position_jump_time_s: float | None
    rejected_ball_chest_sync_samples: int
    ball_velocity_spikes_over_5mps: int
    policy_ticks: int
    policy_dt_p99_ms: float
    policy_dt_max_ms: float
    runtime_overruns: int
    runtime_missed_cycles: int
    runtime_max_loop_ms: float
    runtime_mocap_stale_events: int
    note: str


@dataclass
class PlotSeries:
    marker_time_s: np.ndarray
    ball_ct_m: np.ndarray
    humanoid_ct_m: np.ndarray
    marker_chest_xy: np.ndarray
    chest_track_xy: np.ndarray
    foot: FootDiagnostics
    episode: Any


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#f8fafc",
            "axes.edgecolor": "#94a3b8",
            "grid.color": "#cbd5e1",
            "grid.alpha": 0.48,
        }
    )


def quantile_or_nan(values: np.ndarray, level: float) -> float:
    return float(np.quantile(values, level)) if len(values) else float("nan")


def first_sustained_time(
    time_s: np.ndarray,
    values: np.ndarray,
    threshold: float,
    samples: int = 3,
) -> float | None:
    for index in range(max(0, len(values) - samples + 1)):
        if np.all(values[index : index + samples] > threshold):
            return float(time_s[index])
    return None


def line_segments(
    ax: plt.Axes,
    xy: np.ndarray,
    *,
    cmap: str = "viridis",
    linewidth: float = 2.4,
    alpha: float = 0.9,
    zorder: int = 4,
) -> None:
    if len(xy) < 2:
        return
    points = xy[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=mpl.colors.Normalize(0.0, 1.0),
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    collection.set_array(np.linspace(0.0, 1.0, len(segments)))
    ax.add_collection(collection)


def read_chest_pose(
    mcap_path: Path, start: float, end: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    log_time, header_time, position, quaternion = relative_ball.read_pose_rows(
        mcap_path,
        "/softtouch/mocap/chest/pose",
        start - 0.30,
        end + 0.30,
    )
    return log_time, header_time, position, quaternion


def interpolate_columns(
    query_time: np.ndarray,
    reference_time: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(query_time, reference_time, values[:, column])
            for column in range(values.shape[1])
        ]
    )


def extend_polyline_endpoints(
    line: np.ndarray, extension_m: float = 2.0
) -> np.ndarray:
    """Extend a route tangent at both ends for humanoid lateral-CT checks.

    The deployed ball route begins at the initial ball center.  The humanoid
    correctly starts about 0.5 m behind that point, so distance to the finite
    route would otherwise count normal along-track separation as cross-track
    error.
    """
    if len(line) < 2:
        return line
    first_delta = line[1] - line[0]
    last_delta = line[-1] - line[-2]
    first_norm = float(np.linalg.norm(first_delta))
    last_norm = float(np.linalg.norm(last_delta))
    if first_norm <= 1.0e-9 or last_norm <= 1.0e-9:
        return line
    first = line[0] - extension_m * first_delta / first_norm
    last = line[-1] + extension_m * last_delta / last_norm
    return np.vstack([first, line, last])


def mesh_min_world_z(
    world_from_link_rotation: np.ndarray,
    world_from_link_translation: np.ndarray,
    vertices: np.ndarray,
    *,
    chunk_size: int = 192,
) -> np.ndarray:
    output = np.empty(len(world_from_link_rotation), dtype=float)
    for first in range(0, len(output), chunk_size):
        last = min(first + chunk_size, len(output))
        projected = (
            world_from_link_rotation[first:last, 2, :] @ vertices.T
            + world_from_link_translation[first:last, 2, None]
        )
        output[first:last] = np.min(projected, axis=1)
    return output


def bridge_short_false_gaps(mask: np.ndarray, max_gap_samples: int) -> np.ndarray:
    bridged = mask.copy()
    true_indices = np.flatnonzero(bridged)
    for left, right in zip(true_indices[:-1], true_indices[1:]):
        if right - left - 1 <= max_gap_samples:
            bridged[left : right + 1] = True
    return bridged


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    cuts = np.r_[
        0,
        np.flatnonzero(np.diff(mask.astype(np.int8)) != 0) + 1,
        len(mask),
    ]
    return [
        (int(first), int(last))
        for first, last in zip(cuts[:-1], cuts[1:])
        if mask[first]
    ]


def detect_swing_events(
    time_s: np.ndarray,
    clearance: np.ndarray,
    toe_clearance: np.ndarray,
    heel_clearance: np.ndarray,
    pitch_down_deg: np.ndarray,
) -> list[SwingEvent]:
    events: list[SwingEvent] = []
    min_samples = int(round(SWING_MIN_DURATION_S / RESAMPLE_PERIOD_S))
    boundary_samples = int(round(0.10 / RESAMPLE_PERIOD_S))
    bridge_samples = int(round(0.08 / RESAMPLE_PERIOD_S))
    for side_index, side in enumerate(("left", "right")):
        mask = clearance[side_index] > SWING_START_CLEARANCE_M
        mask = bridge_short_false_gaps(mask, bridge_samples)
        for first, last in true_runs(mask):
            if last - first < min_samples:
                continue
            if first < boundary_samples or last > len(mask) - boundary_samples:
                continue
            local_peak = int(np.argmax(clearance[side_index, first:last]))
            peak = first + local_peak
            events.append(
                SwingEvent(
                    side=side,
                    start_s=float(time_s[first]),
                    end_s=float(time_s[last - 1]),
                    peak_time_s=float(time_s[peak]),
                    peak_clearance_m=float(clearance[side_index, peak]),
                    toe_clearance_at_peak_m=float(
                        toe_clearance[side_index, peak]
                    ),
                    heel_clearance_at_peak_m=float(
                        heel_clearance[side_index, peak]
                    ),
                    foot_pitch_down_at_peak_deg=float(
                        pitch_down_deg[side_index, peak]
                    ),
                )
            )
    return sorted(events, key=lambda item: item.peak_time_s)


def build_foot_diagnostics(
    mcap_path: Path,
    start: float,
    end: float,
    observation_offset: np.ndarray,
    mesh_vertices: dict[str, dict[str, np.ndarray]],
) -> FootDiagnostics:
    (
        tf_time,
        torso_from_left,
        torso_from_right,
        _,
    ) = robot_geometry.collect_foot_poses(mcap_path, start, end)
    _, chest_header, chest_position, chest_quaternion = read_chest_pose(
        mcap_path, start, end
    )
    observation_position, observation_quaternion, valid = (
        relative_ball.interpolate_frame_pose(
            tf_time,
            chest_header,
            chest_position,
            chest_quaternion,
        )
    )
    tf_time = tf_time[valid]
    observation_position = observation_position[valid]
    observation_quaternion = observation_quaternion[valid]
    torso_transforms = [
        torso_from_left[valid],
        torso_from_right[valid],
    ]
    if len(tf_time) < 20:
        raise RuntimeError("Insufficient synchronized chest/TF samples")

    world_from_observation_rotation = Rotation.from_quat(
        observation_quaternion
    ).as_matrix()
    world_from_torso_translation = observation_position - np.einsum(
        "nij,j->ni",
        world_from_observation_rotation,
        observation_offset,
    )

    raw_min_z: dict[str, dict[str, np.ndarray]] = {}
    raw_origin: list[np.ndarray] = []
    raw_pitch_down: list[np.ndarray] = []
    for side, torso_from_foot in zip(
        ("left", "right"), torso_transforms
    ):
        world_from_foot_rotation = np.einsum(
            "nij,njk->nik",
            world_from_observation_rotation,
            torso_from_foot[:, :3, :3],
        )
        world_from_foot_translation = (
            world_from_torso_translation
            + np.einsum(
                "nij,nj->ni",
                world_from_observation_rotation,
                torso_from_foot[:, :3, 3],
            )
        )
        raw_min_z[side] = {
            region: mesh_min_world_z(
                world_from_foot_rotation,
                world_from_foot_translation,
                vertices,
            )
            for region, vertices in mesh_vertices[side].items()
        }
        raw_origin.append(world_from_foot_translation)
        raw_pitch_down.append(
            np.degrees(
                np.arctan2(
                    -world_from_foot_rotation[:, 2, 0],
                    np.hypot(
                        world_from_foot_rotation[:, 0, 0],
                        world_from_foot_rotation[:, 1, 0],
                    ),
                )
            )
        )

    query_time = np.arange(
        max(start, tf_time[0]),
        min(end, tf_time[-1]),
        RESAMPLE_PERIOD_S,
    )
    relative_time = query_time - start
    min_z = np.asarray(
        [
            np.interp(query_time, tf_time, raw_min_z[side]["all"])
            for side in ("left", "right")
        ]
    )
    toe_z = np.asarray(
        [
            np.interp(query_time, tf_time, raw_min_z[side]["toe"])
            for side in ("left", "right")
        ]
    )
    heel_z = np.asarray(
        [
            np.interp(query_time, tf_time, raw_min_z[side]["heel"])
            for side in ("left", "right")
        ]
    )
    pitch_down = np.asarray(
        [
            np.interp(query_time, tf_time, raw_pitch_down[index])
            for index in range(2)
        ]
    )
    for array in (min_z, toe_z, heel_z, pitch_down):
        if array.shape[1] >= 5:
            for side_index in range(2):
                array[side_index] = savgol_filter(
                    array[side_index], 5, 2, mode="interp"
                )

    support_plane_z = np.minimum(min_z[0], min_z[1])
    clearance = min_z - support_plane_z
    toe_clearance = toe_z - support_plane_z
    heel_clearance = heel_z - support_plane_z

    chest_z = np.interp(query_time, chest_header, chest_position[:, 2])
    baseline_mask = relative_time <= 0.50
    chest_baseline = float(np.median(chest_z[baseline_mask]))
    chest_drop = np.maximum(0.0, chest_baseline - chest_z)
    chest_quaternion_query = np.column_stack(
        [
            np.interp(query_time, chest_header, chest_quaternion[:, column])
            for column in range(4)
        ]
    )
    chest_quaternion_query /= np.linalg.norm(
        chest_quaternion_query, axis=1
    )[:, None]
    chest_rotation = Rotation.from_quat(chest_quaternion_query).as_matrix()
    chest_tilt = np.degrees(
        np.arccos(np.clip(chest_rotation[:, 2, 2], -1.0, 1.0))
    )
    events = detect_swing_events(
        relative_time,
        clearance,
        toe_clearance,
        heel_clearance,
        pitch_down,
    )

    nearest_chest_index, chest_delta = robot_geometry.nearest_indices(
        chest_header, tf_time
    )
    del nearest_chest_index
    return FootDiagnostics(
        time_s=relative_time,
        left_clearance_m=clearance[0],
        right_clearance_m=clearance[1],
        chest_drop_m=chest_drop,
        chest_tilt_deg=chest_tilt,
        events=events,
        chest_drop_q95_m=float(np.quantile(chest_drop, 0.95)),
        chest_drop_max_m=float(np.max(chest_drop)),
        chest_tilt_q95_deg=float(np.quantile(chest_tilt, 0.95)),
        tf_samples=int(len(tf_time)),
        max_chest_tf_sync_ms=float(1000.0 * np.max(chest_delta)),
    )


def route_and_body_metrics(
    episode: Any,
    mcap_path: Path,
) -> tuple[dict[str, float], PlotSeries, np.ndarray]:
    chest_log, _, chest_position, _ = read_chest_pose(
        mcap_path, episode.start, episode.end
    )
    marker_time = np.asarray([sample.time for sample in episode.markers])
    marker_relative = marker_time - episode.start
    marker_chest_xy = interpolate_columns(
        marker_time, chest_log, chest_position[:, :2]
    )
    humanoid_ct = np.asarray(
        [
            route_analysis.nearest_polyline_distance(
                point, extend_polyline_endpoints(sample.route_xy)
            )
            for point, sample in zip(marker_chest_xy, episode.markers)
        ]
    )

    half_window = VELOCITY_AVERAGING_HALF_WINDOW_S
    before = interpolate_columns(
        marker_time - half_window,
        chest_log,
        chest_position[:, :2],
    )
    after = interpolate_columns(
        marker_time + half_window,
        chest_log,
        chest_position[:, :2],
    )
    body_velocity = (after - before) / (2.0 * half_window)
    command_vector = np.asarray(
        [
            sample.current_arrow[1] - sample.current_arrow[0]
            for sample in episode.markers
        ]
    )
    command_norm = np.linalg.norm(command_vector, axis=1)
    command_direction = np.divide(
        command_vector,
        command_norm[:, None],
        out=np.zeros_like(command_vector),
        where=command_norm[:, None] > 1.0e-9,
    )
    along = np.sum(body_velocity * command_direction, axis=1)
    lateral = np.abs(
        body_velocity[:, 0] * command_direction[:, 1]
        - body_velocity[:, 1] * command_direction[:, 0]
    )
    speed = np.linalg.norm(body_velocity, axis=1)
    direction_error = np.degrees(np.arctan2(lateral, along))

    reliable = episode.reliable_ct_mask
    speed_valid = (
        reliable
        & (marker_relative >= 0.50)
        & (marker_time <= episode.end - half_window)
    )
    trusted_marker_time = marker_relative[reliable]
    metrics = {
        "humanoid_ct_mean_m": float(np.mean(humanoid_ct[reliable])),
        "humanoid_ct_final_m": float(humanoid_ct[reliable][-1]),
        "humanoid_speed_median_mps": float(np.median(speed[speed_valid])),
        "humanoid_speed_p90_mps": float(
            np.quantile(speed[speed_valid], 0.90)
        ),
        "direction_error_median_deg": float(
            np.median(direction_error[speed_valid])
        ),
        "direction_error_p90_deg": float(
            np.quantile(direction_error[speed_valid], 0.90)
        ),
        "lateral_speed_median_mps": float(
            np.median(lateral[speed_valid])
        ),
    }
    chest_track_mask = (
        (chest_log >= episode.start)
        & (chest_log <= episode.end)
        & (
            True
            if episode.bad_time is None
            else chest_log < episode.bad_time
        )
    )
    chest_track = chest_position[chest_track_mask, :2]
    # Foot data is attached later to avoid reading TF twice.
    placeholder = FootDiagnostics(
        time_s=np.empty(0),
        left_clearance_m=np.empty(0),
        right_clearance_m=np.empty(0),
        chest_drop_m=np.empty(0),
        chest_tilt_deg=np.empty(0),
        events=[],
        chest_drop_q95_m=float("nan"),
        chest_drop_max_m=float("nan"),
        chest_tilt_q95_deg=float("nan"),
        tf_samples=0,
        max_chest_tf_sync_ms=float("nan"),
    )
    series = PlotSeries(
        marker_time_s=trusted_marker_time,
        ball_ct_m=episode.ct[reliable],
        humanoid_ct_m=humanoid_ct[reliable],
        marker_chest_xy=marker_chest_xy[reliable],
        chest_track_xy=chest_track,
        foot=placeholder,
        episode=episode,
    )
    return metrics, series, speed[speed_valid]


def parse_controller_log(path: Path) -> list[tuple[float, str]]:
    timestamp_pattern = re.compile(r"\[(\d{10}\.\d+)\]")
    rows: list[tuple[float, str]] = []
    for line in path.read_text(errors="replace").splitlines():
        match = timestamp_pattern.search(line)
        if match:
            rows.append((float(match.group(1)), line))
    return rows


def controller_events(
    rows: list[tuple[float, str]], start: float, end: float
) -> dict[str, Any]:
    active = [
        (time, line)
        for time, line in rows
        if start - 0.01 <= time <= end + 0.01
    ]
    runtime = [
        (time, line)
        for time, line in active
        if time >= start + 0.10
    ]
    overrun_pattern = re.compile(
        r"loop took ([0-9.]+) ms \(missed cycles : (\d+)\)"
    )
    overruns: list[tuple[float, float, int]] = []
    for time, line in runtime:
        if "Overrun detected" not in line:
            continue
        match = overrun_pattern.search(line)
        if match:
            overruns.append(
                (time - start, float(match.group(1)), int(match.group(2)))
            )
    stale_times = [
        time - start
        for time, line in runtime
        if "missing or stale" in line
    ]
    return {
        "overruns": overruns,
        "overrun_count": len(overruns),
        "missed_cycles": int(sum(item[2] for item in overruns)),
        "max_loop_ms": float(
            max((item[1] for item in overruns), default=0.0)
        ),
        "mocap_stale_times_s": stale_times,
        "mocap_stale_events": len(stale_times),
    }


def policy_timing(
    mcap_path: Path, start: float, end: float
) -> dict[str, float | int]:
    times = np.asarray(
        [
            record.log_time_ns * 1.0e-9
            for record in read_ros2_messages(
                mcap_path,
                topics=["/softtouch/policy/joint_target"],
                start_time=int(start * 1.0e9),
                end_time=int((end + 0.001) * 1.0e9),
            )
        ]
    )
    delta_ms = 1000.0 * np.diff(times)
    return {
        "ticks": int(len(times)),
        "dt_p99_ms": quantile_or_nan(delta_ms, 0.99),
        "dt_max_ms": float(np.max(delta_ms)),
    }


def load_joint_tracking_episode(
    mcap_path: Path, start: float, end: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_time: list[float] = []
    target_position: list[np.ndarray] = []
    for record in read_ros2_messages(
        mcap_path,
        topics=["/softtouch/policy/joint_target"],
        start_time=int(start * 1.0e9),
        end_time=int((end + 0.001) * 1.0e9),
    ):
        target_time.append(record.log_time_ns * 1.0e-9)
        target_position.append(np.asarray(record.ros_msg.data, dtype=float))

    state_time: list[float] = []
    state_position: list[np.ndarray] = []
    state_names: list[str] | None = None
    for record in read_ros2_messages(
        mcap_path,
        topics=["/joint_states"],
        start_time=int((start - 0.02) * 1.0e9),
        end_time=int((end + 0.32) * 1.0e9),
    ):
        state_time.append(record.log_time_ns * 1.0e-9)
        state_position.append(np.asarray(record.ros_msg.position, dtype=float))
        state_names = list(record.ros_msg.name)
    if state_names is None:
        raise RuntimeError("No joint states")
    policy_order = [
        state_names.index(name) for name in POLICY_JOINT_NAMES
    ]
    return (
        np.asarray(target_time),
        np.asarray(target_position),
        np.asarray(state_time),
        np.asarray(state_position)[:, policy_order],
    )


def effective_joint_response(
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> dict[str, Any]:
    lag_grid = np.arange(0.0, 0.3001, 0.005)
    rmse = np.empty_like(lag_grid)
    for lag_index, lag in enumerate(lag_grid):
        errors: list[np.ndarray] = []
        for target_time, target, state_time, state in records:
            valid = target_time + lag <= state_time[-1]
            actual = np.column_stack(
                [
                    np.interp(
                        target_time[valid] + lag,
                        state_time,
                        state[:, joint],
                    )
                    for joint in range(len(POLICY_JOINT_NAMES))
                ]
            )
            errors.append(
                (actual - target[valid])[:, LEG_JOINT_INDICES]
            )
        error = np.concatenate(errors)
        rmse[lag_index] = np.sqrt(np.mean(error * error))
    best_index = int(np.argmin(rmse))
    best_lag = float(lag_grid[best_index])

    shifted_errors: list[np.ndarray] = []
    for target_time, target, state_time, state in records:
        valid = target_time + best_lag <= state_time[-1]
        actual = np.column_stack(
            [
                np.interp(
                    target_time[valid] + best_lag,
                    state_time,
                    state[:, joint],
                )
                for joint in range(len(POLICY_JOINT_NAMES))
            ]
        )
        shifted_errors.append(
            (actual - target[valid])[:, LEG_JOINT_INDICES]
        )
    shifted_error = np.concatenate(shifted_errors)
    return {
        "lag_grid_ms": (1000.0 * lag_grid).tolist(),
        "leg_rmse_deg": np.degrees(rmse).tolist(),
        "best_alignment_ms": 1000.0 * best_lag,
        "rmse_at_zero_lag_deg": float(np.degrees(rmse[0])),
        "rmse_at_best_lag_deg": float(
            np.degrees(rmse[best_index])
        ),
        "median_abs_error_at_best_lag_deg": float(
            np.degrees(np.median(np.abs(shifted_error)))
        ),
        "q95_abs_error_at_best_lag_deg": float(
            np.degrees(np.quantile(np.abs(shifted_error), 0.95))
        ),
        "interpretation": (
            "Best-fit phase alignment between commanded PD equilibrium "
            "positions and measured leg joints. It includes actuator/plant "
            "dynamics and is not a direct network-latency measurement."
        ),
    }


def classify_episode(
    episode: Any,
    humanoid_ct_final: float,
    stale_events: int,
) -> str:
    if episode.run == "224829" and episode.number == 1:
        suffix = " + late mocap dropout" if stale_events else ""
        return "Late ball escape" + suffix
    if episode.run == "224829" and episode.number == 2:
        return "Best CT; truncate before mocap jump"
    if episode.ct_final >= 0.50 and humanoid_ct_final >= 0.45:
        return "Ball and humanoid drift together"
    if episode.ct_final >= 0.50:
        return "Severe late route divergence"
    if episode.ct_final >= 0.25:
        return "Moderate late drift"
    return "Low-CT reference"


def build_diagnostics(
    episodes: list[Any],
    relative_episodes: dict[tuple[str, int], Any],
) -> tuple[
    list[EpisodeDiagnostics],
    dict[tuple[str, int], PlotSeries],
    dict[str, Any],
]:
    mesh_vertices: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        vertices = np.asarray(
            trimesh.load_mesh(
                MESH_ROOT / f"{side}_ankle_roll_link.STL",
                process=True,
            ).vertices
        )
        mesh_vertices[side] = {
            "all": vertices,
            "toe": vertices[vertices[:, 0] >= 0.080],
            "heel": vertices[vertices[:, 0] <= 0.0],
        }

    log_rows = {
        run: parse_controller_log(path)
        for run, path in CONTROLLER_LOGS.items()
    }
    geometry_by_run = {
        run: relative_ball.load_sensor_geometry(path)
        for run, path in RUN_PATHS.items()
    }
    diagnostics: list[EpisodeDiagnostics] = []
    plot_series: dict[tuple[str, int], PlotSeries] = {}
    tracking_records: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []

    for episode in episodes:
        key = (episode.run, episode.number)
        mcap_path = RUN_PATHS[episode.run]
        body_metrics, series, _ = route_and_body_metrics(
            episode, mcap_path
        )
        foot = build_foot_diagnostics(
            mcap_path,
            episode.start,
            episode.end
            if episode.bad_time is None
            else min(episode.end, episode.bad_time),
            geometry_by_run[episode.run].observation_offset_body,
            mesh_vertices,
        )
        series.foot = foot
        plot_series[key] = series

        events = foot.events
        low_events = [event for event in events if event.low_clearance]
        severe_events = [
            event for event in events if event.severe_clearance
        ]
        event_peak = np.asarray(
            [event.peak_clearance_m for event in events], dtype=float
        )
        controller = controller_events(
            log_rows[episode.run],
            episode.start,
            episode.end,
        )
        timing = policy_timing(
            mcap_path, episode.start, episode.end
        )
        relative = relative_episodes[key]
        trusted_duration = float(
            relative.relative_time[-1]
            if len(relative.relative_time)
            else 0.0
        )
        reliable_ct = episode.ct[episode.reliable_ct_mask]
        reliable_ct_time = (
            episode.ct_time[episode.reliable_ct_mask] - episode.start
        )
        note = classify_episode(
            episode,
            body_metrics["humanoid_ct_final_m"],
            controller["mocap_stale_events"],
        )
        diagnostic = EpisodeDiagnostics(
            run=episode.run,
            number=episode.number,
            label=episode.label,
            active_duration_s=float(episode.end - episode.start),
            trusted_duration_s=trusted_duration,
            ct_samples=int(len(reliable_ct)),
            ball_ct_mean_m=float(np.mean(reliable_ct)),
            ball_ct_median_m=float(np.median(reliable_ct)),
            ball_ct_rms_m=float(
                np.sqrt(np.mean(reliable_ct * reliable_ct))
            ),
            ball_ct_p90_m=float(np.quantile(reliable_ct, 0.90)),
            ball_ct_p95_m=float(np.quantile(reliable_ct, 0.95)),
            ball_ct_max_m=float(np.max(reliable_ct)),
            ball_ct_final_m=float(reliable_ct[-1]),
            humanoid_ct_mean_m=body_metrics["humanoid_ct_mean_m"],
            humanoid_ct_final_m=body_metrics["humanoid_ct_final_m"],
            sustained_ct_020_time_s=first_sustained_time(
                reliable_ct_time, reliable_ct, 0.20
            ),
            sustained_ct_050_time_s=first_sustained_time(
                reliable_ct_time, reliable_ct, 0.50
            ),
            commanded_speed_mps=float(
                np.median(episode.command_speed)
            ),
            humanoid_speed_median_mps=body_metrics[
                "humanoid_speed_median_mps"
            ],
            humanoid_speed_p90_mps=body_metrics[
                "humanoid_speed_p90_mps"
            ],
            direction_error_median_deg=body_metrics[
                "direction_error_median_deg"
            ],
            direction_error_p90_deg=body_metrics[
                "direction_error_p90_deg"
            ],
            lateral_speed_median_mps=body_metrics[
                "lateral_speed_median_mps"
            ],
            low_clearance_swings=len(low_events),
            severe_clearance_swings=len(severe_events),
            total_swings=len(events),
            low_clearance_toe_limited=sum(
                event.limiting_end == "toe" for event in low_events
            ),
            swing_peak_clearance_median_m=float(
                np.median(event_peak) if len(event_peak) else np.nan
            ),
            chest_drop_q95_m=foot.chest_drop_q95_m,
            chest_drop_max_m=foot.chest_drop_max_m,
            chest_tilt_q95_deg=foot.chest_tilt_q95_deg,
            ball_position_jump_m=relative.position_jump_m,
            ball_position_jump_time_s=(
                None
                if episode.bad_time is None
                else float(episode.bad_time - episode.start)
            ),
            rejected_ball_chest_sync_samples=int(
                relative.rejected_sync_samples
            ),
            ball_velocity_spikes_over_5mps=int(
                len(episode.velocity_spike_time)
            ),
            policy_ticks=int(timing["ticks"]),
            policy_dt_p99_ms=float(timing["dt_p99_ms"]),
            policy_dt_max_ms=float(timing["dt_max_ms"]),
            runtime_overruns=int(controller["overrun_count"]),
            runtime_missed_cycles=int(controller["missed_cycles"]),
            runtime_max_loop_ms=float(controller["max_loop_ms"]),
            runtime_mocap_stale_events=int(
                controller["mocap_stale_events"]
            ),
            note=note,
        )
        diagnostics.append(diagnostic)
        tracking_records.append(
            load_joint_tracking_episode(
                mcap_path, episode.start, episode.end
            )
        )

    return diagnostics, plot_series, effective_joint_response(
        tracking_records
    )


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    pdf_pages: PdfPages | None = None,
) -> None:
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
    )
    if pdf_pages is not None:
        pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def save_route_map(
    diagnostics: list[EpisodeDiagnostics],
    series_by_key: dict[tuple[str, int], PlotSeries],
    output_dir: Path,
    pdf_pages: PdfPages,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17.2, 10.2))
    for ax, diagnostic in zip(axes.flat, diagnostics):
        series = series_by_key[(diagnostic.run, diagnostic.number)]
        episode = series.episode
        route = episode.route_for_plot
        ax.plot(
            route[:, 0],
            route[:, 1],
            color="#111827",
            linestyle=(0, (4, 3)),
            linewidth=1.7,
            zorder=2,
        )
        ball = episode.ball_xy[episode.reliable_mask]
        line_segments(ax, ball, linewidth=2.7, zorder=5)
        chest_stride = max(1, len(series.chest_track_xy) // 600)
        ax.plot(
            series.chest_track_xy[::chest_stride, 0],
            series.chest_track_xy[::chest_stride, 1],
            color=BODY_COLOR,
            linewidth=2.0,
            alpha=0.82,
            zorder=4,
        )
        route_analysis.add_command_arrows(
            ax,
            episode,
            interval_s=1.0,
            show_preview=False,
        )
        ax.scatter(
            ball[0, 0],
            ball[0, 1],
            s=34,
            marker="o",
            facecolor="#22c55e",
            edgecolor="white",
            linewidth=0.7,
            zorder=8,
        )
        ax.scatter(
            ball[-1, 0],
            ball[-1, 1],
            s=36,
            marker="s",
            facecolor="#ef4444",
            edgecolor="white",
            linewidth=0.7,
            zorder=8,
        )
        all_xy = np.vstack([route, ball, series.chest_track_xy])
        minimum = np.min(all_xy, axis=0)
        maximum = np.max(all_xy, axis=0)
        center = 0.5 * (minimum + maximum)
        span = max(float(np.max(maximum - minimum)), 0.8)
        ax.set_xlim(center[0] - 0.60 * span, center[0] + 0.60 * span)
        ax.set_ylim(center[1] - 0.60 * span, center[1] + 0.60 * span)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_title(
            f"{diagnostic.label} · {diagnostic.note}\n"
            f"Ball CT mean/final: {diagnostic.ball_ct_mean_m:.3f}/"
            f"{diagnostic.ball_ct_final_m:.3f} m · "
            f"Humanoid final: {diagnostic.humanoid_ct_final_m:.3f} m"
        )
    for row in range(2):
        axes[row, 0].set_ylabel("World y [m]")
    for ax in axes[-1]:
        ax.set_xlabel("World x [m]")
    fig.suptitle(
        "FOCUSED HARDWARE EPISODES · ROUTE, BALL, AND HUMANOID",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    handles = [
        Line2D(
            [0], [0], color="#111827", linestyle="--",
            linewidth=1.8, label="Commanded route"
        ),
        Line2D(
            [0], [0], color=BALL_COLOR,
            linewidth=2.8, label="Ball track (early → late)"
        ),
        Line2D(
            [0], [0], color=BODY_COLOR,
            linewidth=2.2, label="Humanoid chest track"
        ),
        Line2D(
            [0], [0], color="#0891b2",
            linewidth=2.2, label="Command direction (1.0 s spacing)"
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        top=0.91,
        bottom=0.075,
        hspace=0.28,
        wspace=0.18,
    )
    save_figure(
        fig,
        output_dir,
        "softtouch_focus_route_map",
        pdf_pages=pdf_pages,
    )


def save_ct_timeline(
    diagnostics: list[EpisodeDiagnostics],
    series_by_key: dict[tuple[str, int], PlotSeries],
    output_dir: Path,
    pdf_pages: PdfPages,
) -> None:
    fig, axes = plt.subplots(
        2, 3, figsize=(17.2, 9.1), sharex=True, sharey=True
    )
    for ax, diagnostic in zip(axes.flat, diagnostics):
        series = series_by_key[(diagnostic.run, diagnostic.number)]
        ax.axhspan(0.0, 0.20, color="#22c55e", alpha=0.07)
        ax.axhspan(0.20, 0.50, color="#f59e0b", alpha=0.07)
        ax.axhspan(0.50, 1.00, color="#ef4444", alpha=0.06)
        ax.plot(
            series.marker_time_s,
            series.ball_ct_m,
            color=BALL_COLOR,
            linewidth=2.3,
            label="Ball CT",
        )
        ax.plot(
            series.marker_time_s,
            series.humanoid_ct_m,
            color=BODY_COLOR,
            linewidth=1.9,
            alpha=0.88,
            label="Humanoid CT",
        )
        if diagnostic.ball_position_jump_m is not None:
            ax.axvline(
                diagnostic.trusted_duration_s,
                color="#dc2626",
                linestyle="--",
                linewidth=1.4,
            )
        ax.axhline(
            0.20, color="#16a34a", linewidth=0.8, linestyle=":"
        )
        ax.axhline(
            0.50, color="#dc2626", linewidth=0.8, linestyle=":"
        )
        ax.grid(True)
        ax.set_title(
            f"{diagnostic.label} · {diagnostic.note}\n"
            f"Ball mean/RMS/P90/final = "
            f"{diagnostic.ball_ct_mean_m:.3f}/"
            f"{diagnostic.ball_ct_rms_m:.3f}/"
            f"{diagnostic.ball_ct_p90_m:.3f}/"
            f"{diagnostic.ball_ct_final_m:.3f} m"
        )
    axes[0, 0].set_xlim(0.0, 9.5)
    axes[0, 0].set_ylim(0.0, 1.00)
    for row in range(2):
        axes[row, 0].set_ylabel("Cross-track error [m]")
    for ax in axes[-1]:
        ax.set_xlabel("Time since activation [s]")
    fig.suptitle(
        "CROSS-TRACK ERROR · BALL VS. HUMANOID",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    handles = [
        Line2D(
            [0], [0], color=BALL_COLOR,
            linewidth=2.4, label="Ball CT"
        ),
        Line2D(
            [0], [0], color=BODY_COLOR,
            linewidth=2.2, label="Humanoid CT"
        ),
        Line2D(
            [0], [0], color="#dc2626", linestyle="--",
            linewidth=1.4, label="Trusted-data cutoff"
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        top=0.90,
        bottom=0.085,
        hspace=0.31,
        wspace=0.16,
    )
    save_figure(
        fig,
        output_dir,
        "softtouch_focus_ct_timeline",
        pdf_pages=pdf_pages,
    )


def save_scrape_timeline(
    diagnostics: list[EpisodeDiagnostics],
    series_by_key: dict[tuple[str, int], PlotSeries],
    output_dir: Path,
    pdf_pages: PdfPages,
) -> None:
    fig, axes = plt.subplots(
        2, 3, figsize=(17.2, 9.4), sharex=True, sharey=True
    )
    for ax, diagnostic in zip(axes.flat, diagnostics):
        foot = series_by_key[
            (diagnostic.run, diagnostic.number)
        ].foot
        ax.axhspan(
            0.0,
            100.0 * LOW_CLEARANCE_M,
            color="#ef4444",
            alpha=0.075,
        )
        ax.plot(
            foot.time_s,
            100.0 * foot.left_clearance_m,
            color=LEFT_COLOR,
            linewidth=1.6,
            label="Left foot",
        )
        ax.plot(
            foot.time_s,
            100.0 * foot.right_clearance_m,
            color=RIGHT_COLOR,
            linewidth=1.6,
            label="Right foot",
        )
        for event in foot.events:
            color = LEFT_COLOR if event.side == "left" else RIGHT_COLOR
            marker = "x" if event.low_clearance else "o"
            ax.scatter(
                event.peak_time_s,
                100.0 * event.peak_clearance_m,
                marker=marker,
                s=34 if event.low_clearance else 18,
                color=color,
                linewidth=1.2,
                zorder=6,
            )
        drop_ax = ax.twinx()
        drop_ax.plot(
            foot.time_s,
            100.0 * foot.chest_drop_m,
            color="#475569",
            linestyle="--",
            linewidth=1.3,
            alpha=0.70,
        )
        drop_ax.set_ylim(0.0, 22.0)
        drop_ax.tick_params(
            axis="y", labelsize=7, colors="#64748b"
        )
        if ax not in axes[:, -1]:
            drop_ax.set_yticklabels([])
        else:
            drop_ax.set_ylabel(
                "Chest drop [cm]", color="#64748b", fontsize=8
            )
        ax.axhline(
            100.0 * LOW_CLEARANCE_M,
            color="#dc2626",
            linewidth=0.9,
            linestyle=":",
        )
        ax.set_ylim(0.0, 18.0)
        ax.grid(True)
        ax.set_title(
            f"{diagnostic.label} · low swings "
            f"{diagnostic.low_clearance_swings}/"
            f"{diagnostic.total_swings} "
            f"(severe <2 cm: {diagnostic.severe_clearance_swings})\n"
            f"Low swings toe-limited: "
            f"{diagnostic.low_clearance_toe_limited}/"
            f"{diagnostic.low_clearance_swings} · "
            f"max chest drop: {100*diagnostic.chest_drop_max_m:.1f} cm"
        )
    axes[0, 0].set_xlim(0.0, 9.5)
    for row in range(2):
        axes[row, 0].set_ylabel(
            "Conservative swing-foot clearance [cm]"
        )
    for ax in axes[-1]:
        ax.set_xlabel("Time since activation [s]")
    fig.suptitle(
        "GROUND-SCRAPE EVIDENCE · FULL-FOOT MESH CLEARANCE",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    handles = [
        Line2D(
            [0], [0], color=LEFT_COLOR,
            linewidth=2.0, label="Left full-foot clearance"
        ),
        Line2D(
            [0], [0], color=RIGHT_COLOR,
            linewidth=2.0, label="Right full-foot clearance"
        ),
        Line2D(
            [0], [0], color="#475569", linestyle="--",
            linewidth=1.5, label="Chest drop from activation height"
        ),
        Line2D(
            [0], [0], marker="x", linestyle="none",
            color="#dc2626", label="Swing peak below 3 cm"
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False
    )
    fig.text(
        0.055,
        0.028,
        "Support-plane reference = lower foot at the same instant. "
        "This cancels absolute mocap height bias. A <3 cm swing is a "
        "scrape-risk flag, not a force-sensor contact label.",
        fontsize=8.5,
        color="#475569",
    )
    fig.subplots_adjust(
        left=0.060,
        right=0.945,
        top=0.89,
        bottom=0.105,
        hspace=0.33,
        wspace=0.20,
    )
    save_figure(
        fig,
        output_dir,
        "softtouch_focus_scrape_diagnostics",
        pdf_pages=pdf_pages,
    )


def save_summary_dashboard(
    diagnostics: list[EpisodeDiagnostics],
    series_by_key: dict[tuple[str, int], PlotSeries],
    joint_response: dict[str, Any],
    output_dir: Path,
    pdf_pages: PdfPages,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 11.0))
    labels = [
        diagnostic.label.replace(" / ", "\n")
        for diagnostic in diagnostics
    ]
    x = np.arange(len(diagnostics))

    ax = axes[0, 0]
    width = 0.25
    ax.bar(
        x - width,
        [item.ball_ct_mean_m for item in diagnostics],
        width,
        color="#60a5fa",
        label="Mean",
    )
    ax.bar(
        x,
        [item.ball_ct_rms_m for item in diagnostics],
        width,
        color="#2563eb",
        label="RMS",
    )
    ax.bar(
        x + width,
        [item.ball_ct_final_m for item in diagnostics],
        width,
        color="#ef4444",
        label="Final",
    )
    ax.axhline(0.20, color="#16a34a", linestyle=":", linewidth=1.0)
    ax.axhline(0.50, color="#dc2626", linestyle=":", linewidth=1.0)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Ball cross-track error [m]")
    ax.set_title("Route performance")
    ax.legend(frameon=False, ncol=3)
    ax.grid(True, axis="y")

    ax = axes[0, 1]
    actual_speed = np.asarray(
        [item.humanoid_speed_median_mps for item in diagnostics]
    )
    bars = ax.bar(
        x,
        actual_speed,
        color=EPISODE_COLORS,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axhline(
        0.40,
        color="#111827",
        linestyle="--",
        linewidth=1.5,
        label="Deployed command: 0.40 m/s",
    )
    ax.axhspan(
        0.50,
        1.10,
        color="#f59e0b",
        alpha=0.10,
        label="Training slow-cruise range",
    )
    for bar, value in zip(bars, actual_speed):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.018,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 0.82)
    ax.set_ylabel("Median humanoid speed [m/s]")
    ax.set_title("Sustained 0.4 m/s command is out of training cruise range")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y")

    ax = axes[1, 0]
    for episode_index, diagnostic in enumerate(diagnostics):
        events = series_by_key[
            (diagnostic.run, diagnostic.number)
        ].foot.events
        jitter = np.linspace(-0.13, 0.13, max(len(events), 1))
        for event_index, event in enumerate(events):
            marker = "v" if event.limiting_end == "toe" else "^"
            color = (
                "#dc2626"
                if event.low_clearance
                else EPISODE_COLORS[episode_index]
            )
            ax.scatter(
                episode_index + jitter[event_index],
                100.0 * event.peak_clearance_m,
                color=color,
                marker=marker,
                s=28,
                alpha=0.85,
                linewidth=0.0,
            )
    ax.axhspan(0.0, 3.0, color="#ef4444", alpha=0.08)
    ax.axhline(3.0, color="#dc2626", linestyle=":", linewidth=1.2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 18.0)
    ax.set_ylabel("Peak full-foot swing clearance [cm]")
    ax.set_title(
        "Low-clearance swings are systematic (▼ toe-limited, ▲ heel-limited)"
    )
    ax.grid(True, axis="y")

    ax = axes[1, 1]
    lag_ms = np.asarray(joint_response["lag_grid_ms"])
    rmse_deg = np.asarray(joint_response["leg_rmse_deg"])
    best = joint_response["best_alignment_ms"]
    ax.plot(lag_ms, rmse_deg, color="#7c3aed", linewidth=2.5)
    ax.axvline(
        best,
        color="#dc2626",
        linestyle="--",
        linewidth=1.4,
        label=f"Best phase alignment: {best:.0f} ms",
    )
    ax.axvspan(
        0.0,
        20.0,
        color="#22c55e",
        alpha=0.10,
        label="Explicit delay DR only: 0–20 ms",
    )
    ax.set_xlabel("Measured-joint time shift after target [ms]")
    ax.set_ylabel("Leg target-to-state RMSE [deg]")
    ax.set_title("Effective real-plant phase response")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True)

    pooled_ct = np.concatenate(
        [
            series_by_key[(item.run, item.number)].ball_ct_m
            for item in diagnostics
        ]
    )
    total_low = sum(item.low_clearance_swings for item in diagnostics)
    total_swings = sum(item.total_swings for item in diagnostics)
    fig.suptitle(
        "SOFTTOUCH HARDWARE FINDINGS · SIX SELECTED EPISODES",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.055,
        0.022,
        f"Pooled ball CT: mean {np.mean(pooled_ct):.3f} m · "
        f"RMS {np.sqrt(np.mean(pooled_ct**2)):.3f} m · "
        f"P90 {np.quantile(pooled_ct, .90):.3f} m.  "
        f"Low-clearance swings: {total_low}/{total_swings}.  "
        "Core interpretation: low toe clearance is real and systematic, "
        "while route failures split into ball escape and whole-body drift.",
        fontsize=9.2,
        color="#334155",
    )
    fig.subplots_adjust(
        left=0.070,
        right=0.985,
        top=0.91,
        bottom=0.080,
        hspace=0.33,
        wspace=0.20,
    )
    save_figure(
        fig,
        output_dir,
        "softtouch_focus_summary",
        pdf_pages=pdf_pages,
    )


def serializable_episode(diagnostic: EpisodeDiagnostics) -> dict[str, Any]:
    result = asdict(diagnostic)
    for key, value in list(result.items()):
        if isinstance(value, float) and not np.isfinite(value):
            result[key] = None
    return result


def build_summary(
    diagnostics: list[EpisodeDiagnostics],
    series_by_key: dict[tuple[str, int], PlotSeries],
    joint_response: dict[str, Any],
) -> dict[str, Any]:
    pooled_ct = np.concatenate(
        [
            series_by_key[(item.run, item.number)].ball_ct_m
            for item in diagnostics
        ]
    )
    episode_means = np.asarray(
        [item.ball_ct_mean_m for item in diagnostics]
    )
    all_events = [
        event
        for item in diagnostics
        for event in series_by_key[(item.run, item.number)].foot.events
    ]
    low_events = [event for event in all_events if event.low_clearance]
    severe_events = [
        event for event in all_events if event.severe_clearance
    ]
    return {
        "scope": {
            "episodes": [
                f"{run}/E{number:02d}" for run, number in FOCUS_KEYS
            ],
            "active_duration_sum_s": float(
                sum(item.active_duration_s for item in diagnostics)
            ),
            "trusted_duration_sum_s": float(
                sum(item.trusted_duration_s for item in diagnostics)
            ),
            "definition_of_ct": (
                "Euclidean nearest distance from the recorded ball center "
                "to the deployed commanded-route polyline at each route-marker "
                "timestamp."
            ),
        },
        "ball_cross_track_m": {
            "point_weighted_samples": int(len(pooled_ct)),
            "point_weighted_mean": float(np.mean(pooled_ct)),
            "point_weighted_median": float(np.median(pooled_ct)),
            "point_weighted_rms": float(
                np.sqrt(np.mean(pooled_ct * pooled_ct))
            ),
            "point_weighted_p90": float(np.quantile(pooled_ct, 0.90)),
            "point_weighted_p95": float(np.quantile(pooled_ct, 0.95)),
            "maximum": float(np.max(pooled_ct)),
            "episode_balanced_mean": float(np.mean(episode_means)),
            "episode_balanced_std": float(
                np.std(episode_means, ddof=1)
            ),
        },
        "ground_scrape_screen": {
            "method": (
                "Chest mocap + time-aligned robot TF + exact ankle-roll STL. "
                "At every instant, the lower foot defines the local support "
                "plane. Swing phases are continuous intervals with >3 mm "
                "relative clearance for >=0.18 s."
            ),
            "low_clearance_threshold_m": LOW_CLEARANCE_M,
            "severe_clearance_threshold_m": SEVERE_CLEARANCE_M,
            "swings": len(all_events),
            "low_clearance_swings": len(low_events),
            "severe_clearance_swings": len(severe_events),
            "low_clearance_fraction_percent": float(
                100.0 * len(low_events) / len(all_events)
            ),
            "low_clearance_toe_limited": int(
                sum(event.limiting_end == "toe" for event in low_events)
            ),
            "peak_clearance_m": {
                "min": float(
                    min(event.peak_clearance_m for event in all_events)
                ),
                "median": float(
                    np.median(
                        [event.peak_clearance_m for event in all_events]
                    )
                ),
                "q25": float(
                    np.quantile(
                        [event.peak_clearance_m for event in all_events],
                        0.25,
                    )
                ),
            },
            "caveat": (
                "This establishes low geometric clearance and scrape risk. "
                "The logs contain no foot force/contact topic, so it does not "
                "label exact physical scrape instants."
            ),
        },
        "speed_tracking": {
            "deployed_command_mps": 0.4,
            "episode_median_humanoid_speed_mps": [
                item.humanoid_speed_median_mps for item in diagnostics
            ],
            "episodes_above_command_by_25pct": int(
                sum(
                    item.humanoid_speed_median_mps > 0.50
                    for item in diagnostics
                )
            ),
            "training_slow_cruise_range_mps": [0.5, 1.1],
            "training_slow_cruise_probability": 0.25,
            "training_primary_cruise_range_mps": [1.1, 2.0],
            "training_route_start_speed_mps": 0.6,
        },
        "joint_response": joint_response,
        "data_quality": {
            "ball_velocity_spikes_over_5mps": int(
                sum(
                    item.ball_velocity_spikes_over_5mps
                    for item in diagnostics
                )
            ),
            "mocap_position_jump": {
                "episode": "224829/E02",
                "magnitude_m": next(
                    item.ball_position_jump_m
                    for item in diagnostics
                    if item.run == "224829" and item.number == 2
                ),
                "detected_time_since_activation_s": next(
                    item.ball_position_jump_time_s
                    for item in diagnostics
                    if item.run == "224829" and item.number == 2
                ),
                "last_trusted_sample_time_s": next(
                    item.trusted_duration_s
                    for item in diagnostics
                    if item.run == "224829" and item.number == 2
                ),
            },
            "late_chest_mocap_dropout": {
                "episode": "224829/E01",
                "events": next(
                    item.runtime_mocap_stale_events
                    for item in diagnostics
                    if item.run == "224829" and item.number == 1
                ),
            },
            "policy_rate": (
                "50 Hz is stable in five episodes. E01 has one 39.98 ms "
                "interval coincident with the late chest-mocap dropout."
            ),
        },
        "deployed_checkpoint_context": {
            "checkpoint": "softtouch_dribble_deploy_m55000.onnx",
            "archived_training_config": str(CHECKPOINT_DIR / "env.yaml"),
            "archived_training_command": str(CHECKPOINT_DIR / "command.txt"),
            "training_ball_angular_damping_per_s": 4.0,
            "hardware_calibration_noted_in_checkpoint_readme_per_s": 0.9,
            "explicit_action_delay_randomization_ms": [0.0, 20.0],
            "foot_clearance_or_slip_reward_present": False,
            "feet_excluded_from_undesired_contact_penalty": True,
            "actor_observes_base_height": False,
            "actor_observes_base_linear_velocity": False,
            "actor_observes_foot_contact": False,
        },
        "episodes": [
            serializable_episode(item) for item in diagnostics
        ],
        "interpretation": {
            "primary_ground_scrape_cause": (
                "The deployed motion repeatedly provides only 1–3 cm of "
                "full-foot swing clearance, usually toe-limited. The archived "
                "training objective has no clearance/sliding cost and excludes "
                "the feet from undesired-contact penalties, so the optimizer "
                "has no direct reason to preserve a real-hardware toe margin."
            ),
            "amplifiers": [
                (
                    "Sustained 0.4 m/s deployment is below the checkpoint's "
                    "training cruise range; five episodes move >25% faster "
                    "than commanded."
                ),
                (
                    "Measured leg response best aligns roughly 0.14 s after "
                    "the PD target, increasing phase error during toe-off."
                ),
                (
                    "Chest height drops about 9–10 cm at the 95th percentile "
                    "and up to about 20 cm in the worst episode."
                ),
            ],
            "route_failure_modes": [
                (
                    "224829/E01: the ball leaves the route while the humanoid "
                    "remains much closer, consistent with ball escape."
                ),
                (
                    "224829/E03 and E05: ball and humanoid CT rise together, "
                    "so whole-body route following is the dominant failure."
                ),
                (
                    "Low-CT E02/E08 also contain low-clearance swings; scrape "
                    "risk is systematic but is not sufficient by itself to "
                    "explain every CT failure."
                ),
            ],
        },
    }


def save_csv(
    diagnostics: list[EpisodeDiagnostics], output_dir: Path
) -> None:
    rows = [serializable_episode(item) for item in diagnostics]
    path = output_dir / "softtouch_focus_episode_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt_optional(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def save_markdown(
    diagnostics: list[EpisodeDiagnostics],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    ct = summary["ball_cross_track_m"]
    scrape = summary["ground_scrape_screen"]
    response = summary["joint_response"]
    rows = []
    for item in diagnostics:
        trusted_suffix = (
            f"{item.trusted_duration_s:.2f}"
            if item.ball_position_jump_m is not None
            else f"{item.active_duration_s:.2f}"
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    item.label,
                    trusted_suffix,
                    f"{item.ball_ct_mean_m:.3f}",
                    f"{item.ball_ct_rms_m:.3f}",
                    f"{item.ball_ct_p90_m:.3f}",
                    f"{item.ball_ct_final_m:.3f}",
                    f"{item.humanoid_ct_final_m:.3f}",
                    f"{item.humanoid_speed_median_mps:.2f}",
                    (
                        f"{item.low_clearance_swings}/"
                        f"{item.total_swings}"
                    ),
                    f"{100*item.chest_drop_max_m:.1f}",
                    item.note,
                ]
            )
            + " |"
        )

    text = f"""# SoftTouch Hardware Analysis — Selected Episodes

## Scope and headline result

Episodes: `205524/E02`, `224829/E01`, `E02`, `E03`, `E05`, `E08`.

CT is the nearest 2-D distance from the recorded ball center to the deployed
commanded-route polyline at each route-marker timestamp. Only trusted samples
before the first >0.5 m mocap jump are included.

- Point-weighted CT: **mean {ct['point_weighted_mean']:.3f} m**, median
  {ct['point_weighted_median']:.3f} m, RMS {ct['point_weighted_rms']:.3f} m,
  P90 {ct['point_weighted_p90']:.3f} m, maximum {ct['maximum']:.3f} m.

- Episode-balanced mean CT: **{ct['episode_balanced_mean']:.3f} ±
  {ct['episode_balanced_std']:.3f} m**.

- Three late failures are qualitatively different: E01 is mainly **ball
  escape**, while E03/E05 are mainly **whole-body route drift**.

## Episode table

| Episode | Trusted s | Ball CT mean | RMS | P90 | Final | Humanoid final CT | Humanoid speed | Low swings | Max chest drop cm | Diagnosis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

The speed command is 0.40 m/s in every episode. The displayed humanoid speed is
a 0.4 s centered displacement estimate, which filters normal gait oscillation.

## Why the feet scrape the ground

The evidence supports a **systematic low-clearance, toe-limited gait**, rather
than one isolated sensor failure:

- Exact foot STL + time-aligned robot TF reconstruction finds
  **{scrape['low_clearance_swings']}/{scrape['swings']} swing phases** whose
  complete foot mesh never clears the support-foot plane by 3 cm;
  {scrape['severe_clearance_swings']} remain below 2 cm.

- **{scrape['low_clearance_toe_limited']}/{scrape['low_clearance_swings']}**
  low-clearance swings are toe-limited. This matches the observed toe/forefoot
  rubbing: the heel can rise while the toe remains close to the floor.

- The chest drops roughly 9–10 cm at the 95th percentile after activation,
  and the worst selected episode reaches about
  {100*max(item.chest_drop_max_m for item in diagnostics):.1f} cm.

- Leg states best align with their changing position targets about
  **{response['best_alignment_ms']:.0f} ms later**. This is an effective
  plant/actuator phase response, not a claim of {response['best_alignment_ms']:.0f}
  ms network delay.

The checkpoint configuration explains why this behavior can survive training:

1. The archived reward set has **no foot-clearance or foot-sliding term**.

2. Both ankle-roll links are excluded from the undesired-contact penalty.

3. Positive terms reward fast foot motion near/at ball contact.

4. The actor observes neither base height, base linear velocity, nor foot
   contact state, so it cannot directly detect sag or scraping.

5. Sustained deployment at 0.40 m/s is below the training slow-cruise range
   (0.5–1.1 m/s; only 25% of training cruise) and far below the primary range
   (1.1–2.0 m/s). Five of six episodes still move faster than 0.50 m/s.

**Conclusion:** scraping is not caused by a single corrupt mocap sample. The
primary design cause is that the learned gait is allowed to use a very small
toe margin; slow-command out-of-distribution operation, body sag, and real
joint phase response consume the remaining margin. Scraping then adds
unmodeled friction/impulses and weakens directional control. It is systematic,
but it is not the only cause of high CT because low-CT E02/E08 also scrape.

## Other major findings

- **Ball dynamics mismatch:** the deployed checkpoint was trained with ball
  angular damping 4.0 s⁻¹, while its archived README identifies 0.9 s⁻¹ as the
  hardware calibration. E01 ends with ball CT 0.888 m while humanoid CT is
  0.290 m, consistent with a ball that escapes farther than the body.

- **Data corruption in 224829/E02:** ball position jumps
  {next(item.ball_position_jump_m for item in diagnostics if item.run == '224829' and item.number == 2):.3f}
  m at about
  {next(item.ball_position_jump_time_s for item in diagnostics if item.run == '224829' and item.number == 2):.2f}
  s. The last trusted synchronized sample is at
  {next(item.trusted_duration_s for item in diagnostics if item.run == '224829' and item.number == 2):.2f}
  s, and all reported E02 metrics stop there.

- **Late mocap dropout in 224829/E01:** one 39.98 ms policy interval occurs
  near 8.04 s when the chest pose goes stale. CT had already begun diverging
  at 6.40 s, so the dropout amplifies the ending but does not initiate it.

- **Real-time overruns:** E03 contains two in-episode 500 Hz overruns
  (maximum 9.44 ms, 7 missed cycles). One occurs immediately before sustained
  CT exceeds 0.20 m. This is a secondary risk, not a session-wide explanation.

- **No ball-velocity spike issue in the selected trusted data:** there are zero
  policy observations above 5 m/s.

## Recommended order of fixes

1. Add swing-toe/full-foot clearance and stance-foot slip penalties; retain a
   positive margin (for example 4–6 cm) through the central swing phase.

2. Fine-tune with sustained 0.35–0.60 m/s commands and a direct root-velocity
   vector tracking term. Validate speed and turning before returning to ball
   dribbling.

3. Fit sim actuator dynamics to the measured target/state phase response and
   recheck ankle-pitch/knee behavior under the real gains.

4. Match ball damping to hardware (or randomize around it), then rerun E01-like
   ball-escape tests.

5. Eliminate chest-mocap dropouts and enable deterministic real-time scheduling;
   these are not the primary cause but can turn a recoverable drift into a
   failure.

## Figures

- `softtouch_focus_summary.png`: one-slide summary.

- `softtouch_focus_ct_timeline.png`: ball versus humanoid CT.

- `softtouch_focus_scrape_diagnostics.png`: reconstructed foot clearance and
  chest drop.

- `softtouch_focus_route_map.png`: route, ball, humanoid, and commands. Command
  arrows are sampled at equal 1.0 s intervals.

## Method limits

There is no recorded foot force/contact topic. Therefore the report proves
low geometric clearance and identifies scrape-risk phases, but exact physical
contact instants still require foot force, motor-current residual, or
high-speed video confirmation. The support-foot reference cancels absolute
mocap height bias; it does not estimate floor compliance.

The 140 ms joint value is a best-fit phase alignment of changing PD equilibrium
targets and measured joints. It includes the real plant response and must not
be compared one-to-one with the explicit 0–20 ms action-delay randomization;
the simulator's own actuator dynamics can also create phase lag.

Finally, these are six deliberately selected episodes, not a random sample of
all deployments. The statistics are descriptive; `±` above is the standard
deviation across six episode means, not a population confidence interval.
"""
    (output_dir / "softtouch_focus_report.md").write_text(
        text, encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    args = parser.parse_args()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_route_episodes = route_analysis.load_all_episodes(DEFAULT_LOG_ROOT)
    episode_index = {
        (episode.run, episode.number): episode
        for episode in all_route_episodes
    }
    episodes = [episode_index[key] for key in FOCUS_KEYS]

    all_relative_episodes, _ = relative_ball.load_all(DEFAULT_LOG_ROOT)
    relative_index = {
        (episode.run, episode.number): episode
        for episode in all_relative_episodes
    }
    diagnostics, plot_series, joint_response = build_diagnostics(
        episodes, relative_index
    )
    summary = build_summary(
        diagnostics, plot_series, joint_response
    )

    with (
        args.output_dir / "softtouch_focus_presentation.pdf"
    ).open("wb") as handle:
        with PdfPages(handle) as pdf_pages:
            save_summary_dashboard(
                diagnostics,
                plot_series,
                joint_response,
                args.output_dir,
                pdf_pages,
            )
            save_ct_timeline(
                diagnostics,
                plot_series,
                args.output_dir,
                pdf_pages,
            )
            save_scrape_timeline(
                diagnostics,
                plot_series,
                args.output_dir,
                pdf_pages,
            )
            save_route_map(
                diagnostics,
                plot_series,
                args.output_dir,
                pdf_pages,
            )

    (args.output_dir / "softtouch_focus_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_csv(diagnostics, args.output_dir)
    save_markdown(diagnostics, summary, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "ct": summary["ball_cross_track_m"],
                "scrape": summary["ground_scrape_screen"],
                "joint_response": {
                    key: value
                    for key, value in joint_response.items()
                    if key not in ("lag_grid_ms", "leg_rmse_deg")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
