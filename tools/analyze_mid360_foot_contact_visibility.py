#!/usr/bin/env python3
"""Check proposed upper-chest MID-360 visibility near G1 foot contact.

This combines trusted mocap ball/chest data with the time-aligned robot TF
kinematics.  A ball sample is considered a conservative foot-contact candidate
when the ball sphere is within 15 mm of either rendered foot mesh.

The output separates:
  1. vertical-FOV inclusion, which is deterministic from the mount geometry;
  2. possible self-occlusion, estimated with exact center-ray intersections
     against the two foot triangle meshes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mcap_ros2.reader import read_ros2_messages
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import plot_mid360_ball_envelope as base  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "analysis" / "softtouch_20260727_night"
ASSET_ROOT = Path(
    "/home/alden/Desktop/SoftTouch_dribble-sim/source/multiagent_sim/"
    "multiagent_sim/assets/unitree_description"
)
PROPOSED_SENSOR_ORIGIN_TORSO = np.array([0.125, 0.0, 0.248])
PROPOSED_SENSOR_ORIGIN_OBSERVATION = np.array([0.048, 0.0, 0.100])
PROPOSED_DOWN_PITCH_DEG = 35.5
CONTACT_GAP_TOLERANCE_M = 0.015
NEAR_FOOT_GAP_TOLERANCE_M = 0.080
BALL_RADIUS_M = 0.10
TF_SYNC_LIMIT_S = 0.010

LEG_CHAINS = {
    "left": [
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
    ],
    "right": [
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
    ],
}
TORSO_CHAIN = ["waist_yaw_link", "waist_roll_link", "torso_link"]


@dataclass
class EpisodeContactData:
    label: str
    time: np.ndarray
    ball_observation: np.ndarray
    ball_torso: np.ndarray
    ball_sensor: np.ndarray
    sensor_range: np.ndarray
    angular_radius_deg: np.ndarray
    vertical_elevation_deg: np.ndarray
    vertical_margin_deg: np.ndarray
    left_surface_distance: np.ndarray
    right_surface_distance: np.ndarray
    nearest_foot: np.ndarray
    nearest_surface_distance: np.ndarray
    tf_time_delta: np.ndarray
    left_torso_from_foot: np.ndarray
    right_torso_from_foot: np.ndarray
    torso_from_leg_links: dict[str, np.ndarray]


def stamp_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def quaternion_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    norm = float(np.dot(q, q))
    if norm < 1.0e-15:
        return np.eye(3)
    q *= np.sqrt(2.0 / norm)
    outer = np.outer(q, q)
    return np.array(
        [
            [
                1.0 - outer[1, 1] - outer[2, 2],
                outer[0, 1] - outer[2, 3],
                outer[0, 2] + outer[1, 3],
            ],
            [
                outer[0, 1] + outer[2, 3],
                1.0 - outer[0, 0] - outer[2, 2],
                outer[1, 2] - outer[0, 3],
            ],
            [
                outer[0, 2] - outer[1, 3],
                outer[1, 2] + outer[0, 3],
                1.0 - outer[0, 0] - outer[1, 1],
            ],
        ]
    )


def transform_matrix(transform: Any) -> np.ndarray:
    translation = transform.translation
    rotation = transform.rotation
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_matrix_xyzw(
        np.array([rotation.x, rotation.y, rotation.z, rotation.w])
    )
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


def chain_transform(
    transforms_by_child: dict[str, np.ndarray], chain: list[str]
) -> np.ndarray:
    result = np.eye(4)
    for child in chain:
        result = result @ transforms_by_child[child]
    return result


def collect_foot_poses(
    mcap_path: Path, start: float, end: float
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    times: list[float] = []
    torso_from_left: list[np.ndarray] = []
    torso_from_right: list[np.ndarray] = []
    torso_from_leg_links: dict[str, list[np.ndarray]] = {
        link: []
        for chain in LEG_CHAINS.values()
        for link in chain
    }
    required = set(TORSO_CHAIN)
    for chain in LEG_CHAINS.values():
        required.update(chain)

    for record in read_ros2_messages(
        mcap_path,
        topics=["/tf"],
        start_time=int((start - 0.05) * 1.0e9),
        end_time=int((end + 0.05) * 1.0e9),
    ):
        transforms = {
            item.child_frame_id: transform_matrix(item.transform)
            for item in record.ros_msg.transforms
            if item.child_frame_id in required
        }
        if not required.issubset(transforms):
            continue
        stamps = [
            stamp_seconds(item.header.stamp)
            for item in record.ros_msg.transforms
            if item.child_frame_id in required
        ]
        pelvis_from_torso = chain_transform(transforms, TORSO_CHAIN)
        torso_from_pelvis = np.linalg.inv(pelvis_from_torso)
        pelvis_from_foot: dict[str, np.ndarray] = {}
        for side, chain in LEG_CHAINS.items():
            pelvis_from_link = np.eye(4)
            for link in chain:
                pelvis_from_link = pelvis_from_link @ transforms[link]
                torso_from_leg_links[link].append(
                    torso_from_pelvis @ pelvis_from_link
                )
            pelvis_from_foot[side] = pelvis_from_link
        times.append(float(np.median(stamps)))
        torso_from_left.append(
            torso_from_pelvis @ pelvis_from_foot["left"]
        )
        torso_from_right.append(
            torso_from_pelvis @ pelvis_from_foot["right"]
        )

    if not times:
        raise RuntimeError(f"No complete leg TF records in {start}..{end}")
    order = np.argsort(times)
    time_array = np.asarray(times)[order]
    left_array = np.asarray(torso_from_left)[order]
    right_array = np.asarray(torso_from_right)[order]
    unique = np.r_[True, np.diff(time_array) > 1.0e-9]
    leg_arrays = {
        link: np.asarray(values)[order][unique]
        for link, values in torso_from_leg_links.items()
    }
    return (
        time_array[unique],
        left_array[unique],
        right_array[unique],
        leg_arrays,
    )


def physical_ball_times(mcap_path: Path, episode: Any) -> np.ndarray:
    raw_log, raw_header, _, _ = base.read_pose_rows(
        mcap_path,
        "/softtouch/mocap/ball/pose",
        episode.start,
        episode.end,
    )
    upper = np.searchsorted(raw_log, episode.log_time, side="left")
    lower = np.clip(upper - 1, 0, len(raw_log) - 1)
    upper = np.clip(upper, 0, len(raw_log) - 1)
    choose_upper = (
        np.abs(raw_log[upper] - episode.log_time)
        < np.abs(raw_log[lower] - episode.log_time)
    )
    nearest = np.where(choose_upper, upper, lower)
    mismatch = np.abs(raw_log[nearest] - episode.log_time)
    if float(np.max(mismatch)) > 1.0e-6:
        raise RuntimeError(
            f"Could not map filtered ball times for {episode.label}: "
            f"{np.max(mismatch):.6f}s"
        )
    return raw_header[nearest]


def nearest_indices(
    reference_time: np.ndarray, query_time: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    upper = np.searchsorted(reference_time, query_time, side="left")
    lower = np.clip(upper - 1, 0, len(reference_time) - 1)
    upper = np.clip(upper, 0, len(reference_time) - 1)
    lower_delta = np.abs(reference_time[lower] - query_time)
    upper_delta = np.abs(reference_time[upper] - query_time)
    choose_upper = upper_delta < lower_delta
    index = np.where(choose_upper, upper, lower)
    return index, np.minimum(lower_delta, upper_delta)


def points_in_link_frame(
    point_torso: np.ndarray, torso_from_link: np.ndarray
) -> np.ndarray:
    rotation = torso_from_link[:, :3, :3]
    translation = torso_from_link[:, :3, 3]
    return np.einsum(
        "nji,nj->ni", rotation, point_torso - translation
    )


def load_mesh(side: str) -> tuple[trimesh.Trimesh, cKDTree]:
    path = ASSET_ROOT / "meshes" / "g1" / f"{side}_ankle_roll_link.STL"
    mesh = trimesh.load_mesh(path, process=True)
    return mesh, cKDTree(np.asarray(mesh.vertices))


def load_leg_meshes() -> dict[str, trimesh.Trimesh]:
    mesh_root = ASSET_ROOT / "meshes" / "g1"
    return {
        link: trimesh.load_mesh(mesh_root / f"{link}.STL", process=True)
        for chain in LEG_CHAINS.values()
        for link in chain
    }


def episode_data(
    episode: Any,
    mcap_path: Path,
    left_tree: cKDTree,
    right_tree: cKDTree,
) -> EpisodeContactData:
    ball_time = physical_ball_times(mcap_path, episode)
    (
        tf_time,
        torso_from_left,
        torso_from_right,
        torso_from_leg_links,
    ) = collect_foot_poses(mcap_path, episode.start, episode.end)
    tf_index, tf_delta = nearest_indices(tf_time, ball_time)
    valid = tf_delta <= TF_SYNC_LIMIT_S
    if not np.all(valid):
        # Keep the trusted mocap subset strict here too; missing TF must not be
        # counted as a visibility failure or success.
        ball_time = ball_time[valid]
        tf_index = tf_index[valid]
        tf_delta = tf_delta[valid]
        ball_observation = episode.ball_observation[valid]
    else:
        ball_observation = episode.ball_observation
    torso_from_left = torso_from_left[tf_index]
    torso_from_right = torso_from_right[tf_index]
    torso_from_leg_links = {
        link: transforms[tf_index]
        for link, transforms in torso_from_leg_links.items()
    }

    ball_torso = ball_observation + base.load_sensor_geometry(
        mcap_path
    ).observation_offset_body
    ball_left = points_in_link_frame(ball_torso, torso_from_left)
    ball_right = points_in_link_frame(ball_torso, torso_from_right)
    left_distance = left_tree.query(ball_left, workers=-1)[0]
    right_distance = right_tree.query(ball_right, workers=-1)[0]
    nearest_foot = np.where(left_distance <= right_distance, 0, 1)
    nearest_distance = np.minimum(left_distance, right_distance)

    ball_sensor = ball_observation - PROPOSED_SENSOR_ORIGIN_OBSERVATION
    sensor_range = np.linalg.norm(ball_sensor, axis=1)
    angular_radius = np.degrees(
        np.arcsin(
            np.clip(BALL_RADIUS_M / sensor_range, 0.0, 1.0)
        )
    )
    elevation, margin = base.vertical_margin_deg(
        ball_sensor,
        angular_radius,
        base.inverted_mount_rotation(PROPOSED_DOWN_PITCH_DEG),
    )
    return EpisodeContactData(
        label=episode.label,
        time=ball_time,
        ball_observation=ball_observation,
        ball_torso=ball_torso,
        ball_sensor=ball_sensor,
        sensor_range=sensor_range,
        angular_radius_deg=angular_radius,
        vertical_elevation_deg=elevation,
        vertical_margin_deg=margin,
        left_surface_distance=left_distance,
        right_surface_distance=right_distance,
        nearest_foot=nearest_foot,
        nearest_surface_distance=nearest_distance,
        tf_time_delta=tf_delta,
        left_torso_from_foot=torso_from_left,
        right_torso_from_foot=torso_from_right,
        torso_from_leg_links=torso_from_leg_links,
    )


def ray_intersects_mesh_before_ball(
    sensor_torso: np.ndarray,
    ball_torso: np.ndarray,
    torso_from_link: np.ndarray,
    triangles: np.ndarray,
) -> bool:
    ray = ball_torso - sensor_torso
    center_distance = float(np.linalg.norm(ray))
    if center_distance <= BALL_RADIUS_M:
        return True
    direction = ray / center_distance
    max_distance = center_distance - BALL_RADIUS_M
    return ray_intersects_mesh_before_distance(
        sensor_torso,
        direction,
        max_distance,
        torso_from_link,
        triangles,
    )

def ray_intersects_mesh_before_distance(
    sensor_torso: np.ndarray,
    direction_torso: np.ndarray,
    max_distance: float,
    torso_from_link: np.ndarray,
    triangles: np.ndarray,
) -> bool:
    rotation = torso_from_link[:3, :3]
    translation = torso_from_link[:3, 3]
    origin = rotation.T @ (sensor_torso - translation)
    direction = rotation.T @ direction_torso
    v0 = triangles[:, 0]
    edge1 = triangles[:, 1] - v0
    edge2 = triangles[:, 2] - v0
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(determinant) > 1.0e-10
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    s = origin - v0
    u = inverse * np.einsum("ij,ij->i", s, h)
    valid &= (u >= 0.0) & (u <= 1.0)
    q = np.cross(s, edge1)
    v = inverse * (q @ direction)
    valid &= (v >= 0.0) & ((u + v) <= 1.0)
    distance = inverse * np.einsum("ij,ij->i", edge2, q)
    valid &= (distance > 1.0e-6) & (distance < max_distance)
    return bool(np.any(valid))


def apparent_ball_rays(
    sensor_torso: np.ndarray, ball_torso: np.ndarray
) -> list[tuple[np.ndarray, float]]:
    """Sample rays across the apparent disk and their sphere-hit distances."""
    center_vector = ball_torso - sensor_torso
    center_distance = float(np.linalg.norm(center_vector))
    center_direction = center_vector / center_distance
    reference = (
        np.array([0.0, 0.0, 1.0])
        if abs(center_direction[2]) < 0.92
        else np.array([0.0, 1.0, 0.0])
    )
    basis_u = np.cross(center_direction, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(center_direction, basis_u)
    angular_radius = float(
        np.arcsin(np.clip(BALL_RADIUS_M / center_distance, 0.0, 1.0))
    )
    samples: list[tuple[float, float]] = [(0.0, 0.0)]
    for radius_fraction, count in (
        (0.32, 8),
        (0.62, 12),
        (0.90, 16),
    ):
        samples.extend(
            (
                radius_fraction,
                2.0 * np.pi * index / count,
            )
            for index in range(count)
        )

    rays: list[tuple[np.ndarray, float]] = []
    for radius_fraction, angle in samples:
        offset_direction = (
            np.cos(angle) * basis_u + np.sin(angle) * basis_v
        )
        offset_angle = radius_fraction * angular_radius
        direction = (
            np.cos(offset_angle) * center_direction
            + np.sin(offset_angle) * offset_direction
        )
        projection = float(np.dot(center_vector, direction))
        discriminant = (
            projection * projection
            - (center_distance * center_distance - BALL_RADIUS_M**2)
        )
        if discriminant < 0.0:
            continue
        hit_distance = projection - float(np.sqrt(discriminant))
        rays.append((direction, hit_distance))
    return rays


def event_ball_surface_visibility(
    data: list[EpisodeContactData],
    event_mask_parts: list[np.ndarray],
    leg_meshes: dict[str, trimesh.Trimesh],
    sensor_origin_torso: np.ndarray,
    down_pitch_deg: float,
) -> dict[str, Any]:
    rotation = base.inverted_mount_rotation(down_pitch_deg)
    visible_fractions: list[float] = []
    fov_fractions: list[float] = []
    clear_fractions: list[float] = []
    ray_counts: list[int] = []
    per_episode_visible: dict[str, list[float]] = {}
    per_episode_fov: dict[str, list[float]] = {}
    per_episode_clear: dict[str, list[float]] = {}
    for item, item_event in zip(data, event_mask_parts):
        for index in np.flatnonzero(item_event):
            rays = apparent_ball_rays(
                sensor_origin_torso,
                item.ball_torso[index],
            )
            visible = 0
            inside_fov = 0
            clear = 0
            for direction, hit_distance in rays:
                elevation = float(
                    base.native_elevation_deg(
                        direction[None, :], rotation
                    )[0]
                )
                ray_inside = (
                    base.MID360_FOV_LOW_DEG
                    <= elevation
                    <= base.MID360_FOV_HIGH_DEG
                )
                if ray_inside:
                    inside_fov += 1
                ray_blocked = False
                for link, mesh in leg_meshes.items():
                    if ray_intersects_mesh_before_distance(
                        sensor_origin_torso,
                        direction,
                        hit_distance,
                        item.torso_from_leg_links[link][index],
                        np.asarray(mesh.triangles),
                    ):
                        ray_blocked = True
                        break
                if not ray_blocked:
                    clear += 1
                if ray_inside and not ray_blocked:
                    visible += 1
            count = len(rays)
            ray_counts.append(count)
            visible_fractions.append(visible / count)
            fov_fractions.append(inside_fov / count)
            clear_fractions.append(clear / count)
            per_episode_visible.setdefault(item.label, []).append(
                visible / count
            )
            per_episode_fov.setdefault(item.label, []).append(
                inside_fov / count
            )
            per_episode_clear.setdefault(item.label, []).append(
                clear / count
            )

    visible_array = np.asarray(visible_fractions)
    fov_array = np.asarray(fov_fractions)
    clear_array = np.asarray(clear_fractions)
    return {
        "sampled_rays_per_ball_disk": sorted(set(ray_counts)),
        "events": int(len(visible_array)),
        "events_with_any_visible_sampled_ball_surface": int(
            np.count_nonzero(visible_array > 0.0)
        ),
        "events_with_no_visible_sampled_ball_surface": int(
            np.count_nonzero(visible_array == 0.0)
        ),
        "events_with_at_least_25pct_visible_disk_samples": int(
            np.count_nonzero(visible_array >= 0.25)
        ),
        "events_with_at_least_50pct_visible_disk_samples": int(
            np.count_nonzero(visible_array >= 0.50)
        ),
        "visible_disk_sample_fraction_percent": {
            "min": float(100.0 * np.min(visible_array)),
            "q10": float(100.0 * np.quantile(visible_array, 0.10)),
            "median": float(100.0 * np.median(visible_array)),
            "max": float(100.0 * np.max(visible_array)),
        },
        "inside_vertical_fov_disk_sample_fraction_percent": {
            "min": float(100.0 * np.min(fov_array)),
            "median": float(100.0 * np.median(fov_array)),
        },
        "clear_of_leg_meshes_disk_sample_fraction_percent": {
            "min": float(100.0 * np.min(clear_array)),
            "q10": float(100.0 * np.quantile(clear_array, 0.10)),
            "median": float(100.0 * np.median(clear_array)),
        },
        "per_episode": {
            label: {
                "events": len(values),
                "events_with_any_visible_surface": int(
                    np.count_nonzero(np.asarray(values) > 0.0)
                ),
                "events_with_at_least_25pct_visible_surface": int(
                    np.count_nonzero(np.asarray(values) >= 0.25)
                ),
                "visible_surface_fraction_percent": {
                    "min": float(100.0 * np.min(values)),
                    "median": float(100.0 * np.median(values)),
                },
                "inside_vertical_fov_fraction_percent": {
                    "min": float(
                        100.0 * np.min(per_episode_fov[label])
                    ),
                    "median": float(
                        100.0 * np.median(per_episode_fov[label])
                    ),
                },
                "clear_of_leg_meshes_fraction_percent": {
                    "min": float(
                        100.0 * np.min(per_episode_clear[label])
                    ),
                    "median": float(
                        100.0 * np.median(per_episode_clear[label])
                    ),
                },
            }
            for label, values in per_episode_visible.items()
        },
        "interpretation": (
            "37 deterministic rays sample each apparent ball disk. This is "
            "a geometry visibility estimate, not a guarantee that the "
            "non-repetitive MID-360 scan produces enough returns in one frame."
        ),
    }


def event_count(mask: np.ndarray, time: np.ndarray, max_gap_s: float = 0.08) -> int:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return 0
    gaps = np.diff(time[indices])
    return int(1 + np.count_nonzero(gaps > max_gap_s))


def closest_sample_per_event_mask(
    distance: np.ndarray,
    contact_mask: np.ndarray,
    time: np.ndarray,
    max_gap_s: float = 0.08,
) -> np.ndarray:
    """Select the minimum foot-distance sample from each contact window."""
    selected = np.zeros(len(contact_mask), dtype=bool)
    indices = np.flatnonzero(contact_mask)
    if len(indices) == 0:
        return selected
    split_at = np.flatnonzero(np.diff(time[indices]) > max_gap_s) + 1
    for group in np.split(indices, split_at):
        selected[group[int(np.argmin(distance[group]))]] = True
    return selected


def concat(data: list[EpisodeContactData], name: str) -> np.ndarray:
    return np.concatenate([getattr(item, name) for item in data])


def pitch_sweep(
    ball_sensor: np.ndarray,
    angular_radius_deg: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    grid = np.linspace(20.0, 55.0, 701)
    result: dict[str, Any] = {}
    all_coverage: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        center_coverage = np.empty_like(grid)
        full_coverage = np.empty_like(grid)
        for index, pitch in enumerate(grid):
            elevation, margin = base.vertical_margin_deg(
                ball_sensor,
                angular_radius_deg,
                base.inverted_mount_rotation(float(pitch)),
            )
            center_inside = (
                (elevation >= base.MID360_FOV_LOW_DEG)
                & (elevation <= base.MID360_FOV_HIGH_DEG)
            )
            center_coverage[index] = 100.0 * np.mean(center_inside[mask])
            full_coverage[index] = 100.0 * np.mean(margin[mask] >= 0.0)
        all_coverage[name] = full_coverage
        best_index = int(np.argmax(full_coverage))
        result[name] = {
            "best_full_ball_pitch_deg": float(grid[best_index]),
            "best_full_ball_coverage_percent": float(
                full_coverage[best_index]
            ),
            "center_coverage_at_that_pitch_percent": float(
                center_coverage[best_index]
            ),
            "coverage_by_pitch": {
                f"{pitch:.1f}_deg": {
                    "center_percent": float(
                        center_coverage[
                            int(np.argmin(np.abs(grid - pitch)))
                        ]
                    ),
                    "full_ball_percent": float(
                        full_coverage[
                            int(np.argmin(np.abs(grid - pitch)))
                        ]
                    ),
                }
                for pitch in (35.5, 38.0, 40.0, 42.0, 45.0)
            },
        }

    # If foot contact is primary, find a practical compromise: retain at least
    # 97% full-ball coverage over all trusted data, then maximize coverage of
    # the closest sample from every inferred contact event.
    feasible = all_coverage["all_samples"] >= 97.0
    event_scores = all_coverage["contact_event_closest_samples"].copy()
    event_scores[~feasible] = -np.inf
    compromise_index = int(np.argmax(event_scores))
    result["contact_priority_compromise"] = {
        "criterion": (
            "maximize closest-contact-event full-ball coverage while "
            "retaining at least 97% coverage over all trusted samples"
        ),
        "pitch_deg": float(grid[compromise_index]),
        "contact_event_full_ball_coverage_percent": float(
            all_coverage["contact_event_closest_samples"][compromise_index]
        ),
        "all_samples_full_ball_coverage_percent": float(
            all_coverage["all_samples"][compromise_index]
        ),
        "near_foot_full_ball_coverage_percent": float(
            all_coverage["near_foot"][compromise_index]
        ),
    }
    return result


def origin_pitch_summary(
    ball_observation: np.ndarray,
    event_mask: np.ndarray,
    near_mask: np.ndarray,
    origin_torso: np.ndarray,
) -> dict[str, Any]:
    origin_observation = (
        origin_torso
        - np.array([0.077, 0.0, 0.148])
    )
    ball_sensor = ball_observation - origin_observation
    sensor_range = np.linalg.norm(ball_sensor, axis=1)
    angular_radius = np.degrees(
        np.arcsin(
            np.clip(BALL_RADIUS_M / sensor_range, 0.0, 1.0)
        )
    )
    grid = np.linspace(25.0, 55.0, 601)
    all_full = np.empty_like(grid)
    event_full = np.empty_like(grid)
    near_full = np.empty_like(grid)
    for index, pitch in enumerate(grid):
        _, margin = base.vertical_margin_deg(
            ball_sensor,
            angular_radius,
            base.inverted_mount_rotation(float(pitch)),
        )
        full = margin >= 0.0
        all_full[index] = 100.0 * np.mean(full)
        event_full[index] = 100.0 * np.mean(full[event_mask])
        near_full[index] = 100.0 * np.mean(full[near_mask])
    best_overall = int(np.argmax(all_full))
    feasible = all_full >= 97.0
    contact_score = event_full.copy()
    contact_score[~feasible] = -np.inf
    compromise = (
        int(np.argmax(contact_score))
        if np.any(feasible)
        else best_overall
    )
    return {
        "best_overall_pitch_deg": float(grid[best_overall]),
        "best_overall_full_ball_coverage_percent": float(
            all_full[best_overall]
        ),
        "contact_priority_pitch_with_all_coverage_at_least_97_deg": float(
            grid[compromise]
        ),
        "contact_event_full_ball_coverage_percent": float(
            event_full[compromise]
        ),
        "near_foot_full_ball_coverage_percent": float(
            near_full[compromise]
        ),
        "all_samples_full_ball_coverage_percent": float(
            all_full[compromise]
        ),
    }


def mask_stats(
    mask: np.ndarray,
    elevation: np.ndarray,
    margin: np.ndarray,
    sensor_range: np.ndarray,
    ball_observation: np.ndarray,
) -> dict[str, Any]:
    center_inside = (
        (elevation >= base.MID360_FOV_LOW_DEG)
        & (elevation <= base.MID360_FOV_HIGH_DEG)
    )
    full_inside = margin >= 0.0
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {"samples": 0}
    return {
        "samples": count,
        "center_vertical_fov_coverage_percent": float(
            100.0 * np.mean(center_inside[mask])
        ),
        "full_ball_vertical_fov_coverage_percent": float(
            100.0 * np.mean(full_inside[mask])
        ),
        "sensor_range_m": {
            "min": float(np.min(sensor_range[mask])),
            "median": float(np.median(sensor_range[mask])),
            "max": float(np.max(sensor_range[mask])),
        },
        "ball_observation_xy_m": {
            "x_min": float(np.min(ball_observation[mask, 0])),
            "x_median": float(np.median(ball_observation[mask, 0])),
            "x_max": float(np.max(ball_observation[mask, 0])),
            "abs_y_max": float(
                np.max(np.abs(ball_observation[mask, 1]))
            ),
        },
        "vertical_margin_deg": {
            "min": float(np.min(margin[mask])),
            "q01": float(np.quantile(margin[mask], 0.01)),
            "median": float(np.median(margin[mask])),
        },
    }


def configure_plot() -> None:
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "figure.facecolor": "#10151d",
            "axes.facecolor": "#171e28",
            "savefig.facecolor": "#10151d",
            "text.color": "#edf3fa",
            "axes.labelcolor": "#a8b5c7",
            "xtick.color": "#a8b5c7",
            "ytick.color": "#a8b5c7",
            "axes.edgecolor": "#3b4655",
            "grid.color": "#3b4655",
            "grid.alpha": 0.38,
            "font.family": "DejaVu Sans",
        }
    )


def save_plot(
    data: list[EpisodeContactData],
    contact_mask: np.ndarray,
    near_mask: np.ndarray,
    possible_occlusion: np.ndarray,
    output_path: Path,
) -> None:
    configure_plot()
    ball = concat(data, "ball_observation")
    margin = concat(data, "vertical_margin_deg")
    proximity_gap = concat(data, "nearest_surface_distance") - BALL_RADIUS_M

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.4))
    ax = axes[0]
    shown = near_mask
    scatter = ax.scatter(
        ball[shown, 0],
        ball[shown, 1],
        c=np.clip(proximity_gap[shown] * 1000.0, -10.0, 80.0),
        cmap="viridis",
        s=15,
        alpha=0.75,
        linewidth=0.0,
    )
    ax.scatter(
        ball[contact_mask, 0],
        ball[contact_mask, 1],
        facecolor="none",
        edgecolor="#ffad33",
        s=38,
        linewidth=0.9,
        label="Contact candidate (≤15 mm gap)",
    )
    ax.set_title("Near-foot ball positions")
    ax.set_xlabel("Forward x from observation frame [m]")
    ax.set_ylabel("Left y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(frameon=False, fontsize=8)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Ball-to-foot surface gap [mm]")

    ax = axes[1]
    full = margin >= 0.0
    ax.scatter(
        ball[contact_mask & full, 0],
        margin[contact_mask & full],
        color="#31d6a2",
        s=17,
        alpha=0.72,
        label="Full ball inside FOV",
    )
    ax.scatter(
        ball[contact_mask & ~full, 0],
        margin[contact_mask & ~full],
        color="#ff5d73",
        s=25,
        alpha=0.88,
        label="Clipped by vertical FOV",
    )
    ax.axhline(0.0, color="#edf3fa", linewidth=1.0, linestyle="--")
    ax.set_title("Contact-candidate vertical-FOV margin")
    ax.set_xlabel("Forward x from observation frame [m]")
    ax.set_ylabel("Full-ball margin [deg]")
    ax.grid(True)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    labels = ["Clear center ray", "Foot blocks center ray"]
    counts = [
        int(np.count_nonzero(contact_mask & ~possible_occlusion)),
        int(np.count_nonzero(contact_mask & possible_occlusion)),
    ]
    bars = ax.bar(
        labels,
        counts,
        color=["#31d6a2", "#ffad33"],
        edgecolor="#dfe8f2",
        linewidth=0.7,
    )
    for bar, value in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + max(counts + [1]) * 0.025,
            str(value),
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )
    ax.set_title("Conservative foot self-occlusion check")
    ax.set_ylabel("Contact-candidate samples")
    ax.tick_params(axis="x", rotation=10)
    ax.grid(True, axis="y")

    fig.suptitle(
        "PROPOSED UPPER-CHEST MID-360 · FOOT-CONTACT VISIBILITY",
        x=0.04,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.04,
        0.025,
        "Trusted mocap + time-aligned robot TF. Contact candidates use the "
        "actual foot STL with a 15 mm tolerance. Center-ray occlusion does "
        "not mean the whole ball is invisible.",
        color="#a8b5c7",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.055, right=0.985, top=0.84, bottom=0.17, wspace=0.28
    )
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode",
        action="append",
        default=[],
        help=(
            "Restrict analysis to RUN/EXX, for example 224829/E03. "
            "Repeat for multiple episodes."
        ),
    )
    parser.add_argument(
        "--output-stem",
        default="mid360_foot_contact_visibility",
        help="Basename for the output JSON and PNG.",
    )
    args = parser.parse_args()
    if Path(args.output_stem).name != args.output_stem:
        parser.error("--output-stem must be a basename, not a path")

    all_episodes, _ = base.load_all(base.DEFAULT_LOG_ROOT)
    if args.episode:
        requested = {
            value.strip().replace(" / ", "/").upper()
            for value in args.episode
        }
        available = {
            item.label.replace(" / ", "/").upper(): item
            for item in all_episodes
        }
        missing = sorted(requested - set(available))
        if missing:
            parser.error(
                "unknown episode(s): "
                + ", ".join(missing)
                + "; available: "
                + ", ".join(sorted(available))
            )
        episodes = [
            item
            for item in all_episodes
            if item.label.replace(" / ", "/").upper() in requested
        ]
    else:
        episodes = all_episodes
    run_paths = {
        run: base.DEFAULT_LOG_ROOT
        / "softtouch_bags"
        / f"softtouch_real_20260727_{run}"
        / f"softtouch_real_20260727_{run}_0.mcap"
        for run in ("205524", "224829")
    }
    left_mesh, left_tree = load_mesh("left")
    right_mesh, right_tree = load_mesh("right")
    leg_meshes = load_leg_meshes()

    data = [
        episode_data(
            episode,
            run_paths[episode.run],
            left_tree,
            right_tree,
        )
        for episode in episodes
    ]
    elevation = concat(data, "vertical_elevation_deg")
    margin = concat(data, "vertical_margin_deg")
    sensor_range = concat(data, "sensor_range")
    ball_observation = concat(data, "ball_observation")
    nearest_distance = concat(data, "nearest_surface_distance")
    tf_delta = concat(data, "tf_time_delta")
    contact_mask = (
        nearest_distance <= BALL_RADIUS_M + CONTACT_GAP_TOLERANCE_M
    )
    near_mask = (
        nearest_distance <= BALL_RADIUS_M + NEAR_FOOT_GAP_TOLERANCE_M
    )
    event_mask_parts = []
    for item in data:
        item_contact = (
            item.nearest_surface_distance
            <= BALL_RADIUS_M + CONTACT_GAP_TOLERANCE_M
        )
        event_mask_parts.append(
            closest_sample_per_event_mask(
                item.nearest_surface_distance,
                item_contact,
                item.time,
            )
        )
    event_mask = np.concatenate(event_mask_parts)

    # Exact center-ray intersections are only needed for the conservative
    # contact subset, so this remains inexpensive despite the dense STL.
    possible_occlusion_parts: list[np.ndarray] = []
    possible_leg_occlusion_parts: list[np.ndarray] = []
    occluding_leg_links: dict[str, int] = {
        link: 0 for link in leg_meshes
    }
    for item in data:
        item_contact = (
            item.nearest_surface_distance
            <= BALL_RADIUS_M + CONTACT_GAP_TOLERANCE_M
        )
        item_event = closest_sample_per_event_mask(
            item.nearest_surface_distance,
            item_contact,
            item.time,
        )
        occluded = np.zeros(len(item.time), dtype=bool)
        leg_occluded = np.zeros(len(item.time), dtype=bool)
        for index in np.flatnonzero(item_contact):
            occluded[index] = (
                ray_intersects_mesh_before_ball(
                    PROPOSED_SENSOR_ORIGIN_TORSO,
                    item.ball_torso[index],
                    item.left_torso_from_foot[index],
                    np.asarray(left_mesh.triangles),
                )
                or ray_intersects_mesh_before_ball(
                    PROPOSED_SENSOR_ORIGIN_TORSO,
                    item.ball_torso[index],
                    item.right_torso_from_foot[index],
                    np.asarray(right_mesh.triangles),
                )
            )
        for index in np.flatnonzero(item_event):
            for link, mesh in leg_meshes.items():
                if ray_intersects_mesh_before_ball(
                    PROPOSED_SENSOR_ORIGIN_TORSO,
                    item.ball_torso[index],
                    item.torso_from_leg_links[link][index],
                    np.asarray(mesh.triangles),
                ):
                    leg_occluded[index] = True
                    occluding_leg_links[link] += 1
                    break
        possible_occlusion_parts.append(occluded)
        possible_leg_occlusion_parts.append(leg_occluded)
    possible_occlusion = np.concatenate(possible_occlusion_parts)
    possible_leg_occlusion = np.concatenate(
        possible_leg_occlusion_parts
    )

    forward_origin_sweep: dict[str, Any] = {}
    for origin_x in (0.125, 0.150, 0.175, 0.200, 0.225, 0.250):
        origin = np.array([origin_x, 0.0, 0.248])
        occluded_count = 0
        occluding_links: dict[str, int] = {
            link: 0 for link in leg_meshes
        }

        for item, item_event in zip(data, event_mask_parts):
            for index in np.flatnonzero(item_event):
                for link, mesh in leg_meshes.items():
                    if ray_intersects_mesh_before_ball(
                        origin,
                        item.ball_torso[index],
                        item.torso_from_leg_links[link][index],
                        np.asarray(mesh.triangles),
                    ):
                        occluded_count += 1
                        occluding_links[link] += 1
                        break
        forward_origin_sweep[f"x_{origin_x:.3f}_m"] = {
            "origin_torso_m": origin.tolist(),
            "forward_from_anchor_plane_mm": float(
                1000.0 * (origin_x - 0.0039563)
            ),
            "ahead_of_torso_shell_at_this_height_mm_approx": float(
                1000.0 * (origin_x - 0.0718)
            ),
            "contact_event_center_ray_clear_percent": float(
                100.0
                * (np.count_nonzero(event_mask) - occluded_count)
                / np.count_nonzero(event_mask)
            ),
            "contact_event_center_ray_occluded_count": occluded_count,
            "first_occluding_link_counts": {
                link: count
                for link, count in occluding_links.items()
                if count
            },
            **origin_pitch_summary(
                ball_observation,
                event_mask,
                near_mask,
                origin,
            ),
        }

    event_surface_visibility = {
        f"{pitch:.1f}_deg": event_ball_surface_visibility(
            data,
            event_mask_parts,
            leg_meshes,
            PROPOSED_SENSOR_ORIGIN_TORSO,
            pitch,
        )
        for pitch in (35.5,)
    }

    per_episode_contact_summary: dict[str, Any] = {}
    for item, item_event, item_leg_occlusion in zip(
        data,
        event_mask_parts,
        possible_leg_occlusion_parts,
    ):
        def event_fov_at_pitch(pitch: float) -> dict[str, float]:
            elevation_at_pitch, margin_at_pitch = base.vertical_margin_deg(
                item.ball_sensor,
                item.angular_radius_deg,
                base.inverted_mount_rotation(pitch),
            )
            center_inside = (
                (elevation_at_pitch >= base.MID360_FOV_LOW_DEG)
                & (elevation_at_pitch <= base.MID360_FOV_HIGH_DEG)
            )
            return {
                "center_percent": float(
                    100.0 * np.mean(center_inside[item_event])
                ),
                "full_ball_percent": float(
                    100.0 * np.mean(margin_at_pitch[item_event] >= 0.0)
                ),
            }

        per_episode_contact_summary[item.label] = {
            "contact_events": int(np.count_nonzero(item_event)),
            "fov_at_35_5_deg": event_fov_at_pitch(35.5),
            "fov_at_38_0_deg": event_fov_at_pitch(38.0),
            "fov_at_41_1_deg": event_fov_at_pitch(41.1),
            "fov_at_42_0_deg": event_fov_at_pitch(42.0),
            "center_ray_clear_of_all_leg_meshes_percent": float(
                100.0 * np.mean(~item_leg_occlusion[item_event])
            ),
        }

    episode_events = {
        item.label: event_count(
            item.nearest_surface_distance
            <= BALL_RADIUS_M + CONTACT_GAP_TOLERANCE_M,
            item.time,
        )
        for item in data
    }
    contact_stats = mask_stats(
        contact_mask,
        elevation,
        margin,
        sensor_range,
        ball_observation,
    )
    near_stats = mask_stats(
        near_mask,
        elevation,
        margin,
        sensor_range,
        ball_observation,
    )
    event_stats = mask_stats(
        event_mask,
        elevation,
        margin,
        sensor_range,
        ball_observation,
    )
    contact_count = int(np.count_nonzero(contact_mask))
    contact_stats.update(
        {
            "candidate_events": int(sum(episode_events.values())),
            "candidate_events_per_episode": episode_events,
            "center_ray_clear_of_both_feet_percent": float(
                100.0
                * np.mean(~possible_occlusion[contact_mask])
                if contact_count
                else 0.0
            ),
            "center_ray_intersects_a_foot_percent": float(
                100.0
                * np.mean(possible_occlusion[contact_mask])
                if contact_count
                else 0.0
            ),
        }
    )
    event_stats.update(
        {
            "center_ray_clear_of_both_feet_percent": float(
                100.0 * np.mean(~possible_occlusion[event_mask])
                if np.any(event_mask)
                else 0.0
            ),
            "center_ray_intersects_a_foot_percent": float(
                100.0 * np.mean(possible_occlusion[event_mask])
                if np.any(event_mask)
                else 0.0
            ),
            "center_ray_clear_of_all_leg_meshes_percent": float(
                100.0 * np.mean(~possible_leg_occlusion[event_mask])
                if np.any(event_mask)
                else 0.0
            ),
            "center_ray_intersects_a_leg_mesh_percent": float(
                100.0 * np.mean(possible_leg_occlusion[event_mask])
                if np.any(event_mask)
                else 0.0
            ),
            "first_occluding_link_counts": {
                link: count
                for link, count in occluding_leg_links.items()
                if count
            },
        }
    )
    ball_sensor = concat(data, "ball_sensor")
    angular_radius = concat(data, "angular_radius_deg")
    pitch_results = pitch_sweep(
        ball_sensor,
        angular_radius,
        {
            "all_samples": np.ones(len(ball_sensor), dtype=bool),
            "near_foot": near_mask,
            "contact_candidates": contact_mask,
            "contact_event_closest_samples": event_mask,
        },
    )
    result = {
        "selected_episodes": [item.label for item in episodes],
        "method": {
            "trusted_samples_with_synchronized_robot_tf": int(
                len(nearest_distance)
            ),
            "max_tf_sync_error_ms": float(1000.0 * np.max(tf_delta)),
            "ball_radius_m": BALL_RADIUS_M,
            "contact_candidate_definition": (
                "ball sphere within 15 mm of either exact foot STL"
            ),
            "near_foot_definition": (
                "ball sphere within 80 mm of either exact foot STL"
            ),
            "occlusion_definition": (
                "center rays are checked against the exact feet for all "
                "contact candidates and against all 12 leg-link meshes at "
                "the closest sample of each contact event"
            ),
        },
        "mount": {
            "origin_torso_m": PROPOSED_SENSOR_ORIGIN_TORSO.tolist(),
            "inverted": True,
            "down_pitch_deg": PROPOSED_DOWN_PITCH_DEG,
        },
        "contact_candidates": contact_stats,
        "contact_event_closest_samples": event_stats,
        "near_foot": near_stats,
        "pitch_sweep": pitch_results,
        "forward_origin_sweep": forward_origin_sweep,
        "contact_event_ball_surface_visibility": (
            event_surface_visibility
        ),
        "per_episode_contact_summary": per_episode_contact_summary,
        "interpretation": (
            "Vertical-FOV coverage answers whether the ball lies inside the "
            "specified MID-360 view. A foot-center-ray intersection is a "
            "conservative warning, not proof that the entire ball is hidden."
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"{args.output_stem}.json"
    png_path = OUTPUT_DIR / f"{args.output_stem}.png"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    save_plot(data, contact_mask, near_mask, possible_occlusion, png_path)
    print(json.dumps(result, indent=2))
    print(png_path)
    print(json_path)


if __name__ == "__main__":
    main()
