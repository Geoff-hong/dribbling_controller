#!/usr/bin/env python3
"""Export a SoftTouch motion-clip reset state for the ROS2 MuJoCo plugin."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import onnx


DEFAULT_SOFTTOUCH_DIR = Path.home() / "SoftTouch"
DEFAULT_POLICY = str(DEFAULT_SOFTTOUCH_DIR / "checkpoints" / "g1_dribble_s3_human_iter35000" / "softtouch_dribble_deploy.onnx")
DEFAULT_MOTION_FILES = [
    str(DEFAULT_SOFTTOUCH_DIR / "data" / "dribble" / "npz" / "DribbleF_L_normalized.npz"),
    str(DEFAULT_SOFTTOUCH_DIR / "data" / "dribble" / "npz" / "DribbleF_R_normalized.npz"),
    str(DEFAULT_SOFTTOUCH_DIR / "data" / "dribble" / "npz" / "JogF_LF_normalized.npz"),
    str(DEFAULT_SOFTTOUCH_DIR / "data" / "dribble" / "npz" / "JogF_RF_normalized.npz"),
    str(DEFAULT_SOFTTOUCH_DIR / "data" / "dribble" / "npz" / "WalkF_LF_normalized.npz"),
    str(DEFAULT_SOFTTOUCH_DIR / "data" / "dribble" / "npz" / "WalkF_RF_normalized.npz"),
]
DEFAULT_OUTPUT = "config/g1/softtouch_mujoco_reset_jogf_lf_frame0.txt"
BALL_RADIUS = 0.09
RESET_BALL_FORWARD_M = 0.65


def load_metadata(onnx_path: Path) -> dict[str, str]:
    model = onnx.load(str(onnx_path))
    return {entry.key: entry.value for entry in model.metadata_props}


def parse_csv(meta: dict[str, str], key: str) -> list[str]:
    value = meta[key]
    if not value:
        return []
    return [item for item in value.split(",") if item]


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_to_mat(q).T @ v


def normalize_quat(q: np.ndarray) -> np.ndarray:
    return q / max(float(np.linalg.norm(q)), 1.0e-9)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=np.float64)


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rotate_xy(v: np.ndarray, yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    out = np.asarray(v, dtype=np.float64).copy()
    x = out[0]
    y = out[1]
    out[0] = c * x - s * y
    out[1] = s * x + c * y
    return out


def fmt_vec(values: np.ndarray | list[float]) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def build_reset_state(
    motion_path: Path,
    frame: int,
    *,
    target_yaw: float | None,
    joint_noise: float,
    seed: int,
) -> dict[str, np.ndarray | int | Path]:
    data = np.load(motion_path)
    length = int(data["joint_pos"].shape[0])
    if length <= 0:
        raise ValueError(f"Motion clip has no frames: {motion_path}")
    frame = int(np.clip(frame, 0, length - 1))

    root_pos_all = np.asarray(data["body_pos_w"][:, 0], dtype=np.float64).copy()
    all_body_z = np.asarray(data["body_pos_w"][..., 2], dtype=np.float64)
    ground_offset = float(all_body_z.min()) - 0.02
    if ground_offset > 0.01:
        root_pos_all[:, 2] -= ground_offset

    root_pos = root_pos_all[frame].copy()
    root_quat = normalize_quat(np.asarray(data["body_quat_w"][frame, 0], dtype=np.float64))
    root_lin_vel = np.asarray(data["body_lin_vel_w"][frame, 0], dtype=np.float64).copy()
    root_ang_vel_world = np.asarray(data["body_ang_vel_w"][frame, 0], dtype=np.float64).copy()
    joint_pos = np.asarray(data["joint_pos"][frame], dtype=np.float64).copy()
    joint_vel = np.asarray(data["joint_vel"][frame], dtype=np.float64).copy()

    if target_yaw is not None:
        delta_yaw = target_yaw - yaw_from_quat(root_quat)
        root_quat = normalize_quat(quat_mul(yaw_quat(delta_yaw), root_quat))
        root_lin_vel = rotate_xy(root_lin_vel, delta_yaw)
        root_ang_vel_world = rotate_xy(root_ang_vel_world, delta_yaw)
    root_pos[:2] = 0.0

    if joint_noise > 0.0:
        rng = np.random.default_rng(seed)
        joint_pos += rng.uniform(-joint_noise, joint_noise, size=joint_pos.shape)

    forward = quat_to_mat(root_quat)[:2, 0]
    forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
    ball_pos = np.array(
        [
            root_pos[0] + RESET_BALL_FORWARD_M * forward[0],
            root_pos[1] + RESET_BALL_FORWARD_M * forward[1],
            BALL_RADIUS,
        ],
        dtype=np.float64,
    )

    return {
        "motion_path": motion_path,
        "frame": frame,
        "root_pos": root_pos,
        "root_quat": root_quat,
        "root_lin_vel": root_lin_vel,
        "root_ang_vel_body": rotate_inverse(root_quat, root_ang_vel_world),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "ball_pos": ball_pos,
        "ball_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ball_lin_vel": np.zeros(3, dtype=np.float64),
        "ball_ang_vel": np.zeros(3, dtype=np.float64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--motion-files", nargs="*", default=DEFAULT_MOTION_FILES)
    parser.add_argument("--motion-index", type=int, default=2)
    parser.add_argument("--motion-frame", type=int, default=0)
    parser.add_argument("--reset-yaw-deg", type=float, default=-15.0)
    parser.add_argument("--preserve-motion-yaw", action="store_true")
    parser.add_argument("--joint-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root-joint", default="floating_base_joint")
    parser.add_argument("--ball-joint", default="softtouch_ball_freejoint")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    policy_path = Path(args.policy).expanduser().resolve()
    metadata = load_metadata(policy_path)
    joint_names = parse_csv(metadata, "joint_names")
    if len(joint_names) != 29:
        raise RuntimeError(f"Expected 29 policy joints, got {len(joint_names)} from {policy_path}.")

    motion_files = [Path(path).expanduser().resolve() for path in args.motion_files]
    if not motion_files:
        raise ValueError("--motion-files is empty.")
    if args.motion_index < 0 or args.motion_index >= len(motion_files):
        raise IndexError(f"--motion-index {args.motion_index} is out of range for {len(motion_files)} motion files.")

    target_yaw = None if args.preserve_motion_yaw else math.radians(float(args.reset_yaw_deg))
    reset = build_reset_state(
        motion_files[int(args.motion_index)],
        int(args.motion_frame),
        target_yaw=target_yaw,
        joint_noise=float(args.joint_noise),
        seed=int(args.seed),
    )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# softtouch_mujoco_reset_state_v1",
        f"source_policy {policy_path}",
        f"source_motion {reset['motion_path']}",
        f"motion_frame {int(reset['frame'])}",
        f"target_yaw_deg {'motion' if target_yaw is None else f'{float(args.reset_yaw_deg):.17g}'}",
        f"joint_noise {float(args.joint_noise):.17g}",
        f"root_joint {args.root_joint}",
        f"ball_joint {args.ball_joint}",
        f"root_pos {fmt_vec(reset['root_pos'])}",
        f"root_quat {fmt_vec(reset['root_quat'])}",
        f"root_lin_vel {fmt_vec(reset['root_lin_vel'])}",
        f"root_ang_vel_body {fmt_vec(reset['root_ang_vel_body'])}",
        f"ball_pos {fmt_vec(reset['ball_pos'])}",
        f"ball_quat {fmt_vec(reset['ball_quat'])}",
        f"ball_lin_vel {fmt_vec(reset['ball_lin_vel'])}",
        f"ball_ang_vel {fmt_vec(reset['ball_ang_vel'])}",
        "joint_names " + " ".join(joint_names),
        f"joint_pos {fmt_vec(reset['joint_pos'])}",
        f"joint_vel {fmt_vec(reset['joint_vel'])}",
    ]
    output.write_text("\n".join(lines) + "\n")

    print(f"OK - wrote SoftTouch MuJoCo reset state: {output}")
    print(f"motion: {reset['motion_path']} frame={int(reset['frame'])}")
    print(f"root_pos: {fmt_vec(reset['root_pos'])}")
    print(f"ball_pos: {fmt_vec(reset['ball_pos'])}")


if __name__ == "__main__":
    main()
