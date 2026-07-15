#!/usr/bin/env python3
"""Export a SoftTouch Stage-3 dribble checkpoint for deployment.

The Stage-3 SoftTouch policy is not a plain whole-body joint policy. The PPO
actor outputs an 8-D latent action. The frozen Stage-2 VAE prior/decoder then
turns that latent into a 29-D raw joint-position action:

    latent = actor(norm(actor_obs))
    z = mu_p(decoder_state) + lab_lambda * sigma_p(decoder_state) * tanh(latent)
    raw_joint_action = decoder(decoder_state, z)

This exporter packs that whole chain into one ONNX graph. The graph has one
input named "obs":

    obs[:, 0:90]    = SoftTouch Stage-3 actor observation
    obs[:, 90:180]  = Stage-2 decoder state_only observation

The first output is named "actions" and is the 29-D raw joint action. It should
be scaled and offset by the controller using the metadata fields action_scale
and default_joint_pos, exactly like BeyondMimic's motion-tracking exporter.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable

import onnx
import torch
import torch.nn as nn


DEFAULT_CHECKPOINT_DIR = Path.home() / "SoftTouch" / "checkpoints" / "g1_dribble_s3_human_iter35000"

JOINT_NAMES = [
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

DEFAULT_JOINT_POS = [
    -0.312,
    -0.312,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.669,
    0.669,
    0.2,
    0.2,
    -0.363,
    -0.363,
    0.2,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.6,
    0.6,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 2.0 * 3.141592653589793 * 10.0
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

ACTION_SCALE_PATTERNS = {
    ".*_hip_yaw_joint": 0.25 * 88.0 / STIFFNESS_7520_14,
    ".*_hip_roll_joint": 0.25 * 139.0 / STIFFNESS_7520_22,
    ".*_hip_pitch_joint": 0.25 * 88.0 / STIFFNESS_7520_14,
    ".*_knee_joint": 0.25 * 139.0 / STIFFNESS_7520_22,
    ".*_ankle_pitch_joint": 0.25 * 50.0 / (2.0 * STIFFNESS_5020),
    ".*_ankle_roll_joint": 0.25 * 50.0 / (2.0 * STIFFNESS_5020),
    "waist_roll_joint": 0.25 * 50.0 / (2.0 * STIFFNESS_5020),
    "waist_pitch_joint": 0.25 * 50.0 / (2.0 * STIFFNESS_5020),
    "waist_yaw_joint": 0.25 * 88.0 / STIFFNESS_7520_14,
    ".*_shoulder_pitch_joint": 0.25 * 25.0 / STIFFNESS_5020,
    ".*_shoulder_roll_joint": 0.25 * 25.0 / STIFFNESS_5020,
    ".*_shoulder_yaw_joint": 0.25 * 25.0 / STIFFNESS_5020,
    ".*_elbow_joint": 0.25 * 25.0 / STIFFNESS_5020,
    ".*_wrist_roll_joint": 0.25 * 25.0 / STIFFNESS_5020,
    ".*_wrist_pitch_joint": 0.25 * 5.0 / STIFFNESS_4010,
    ".*_wrist_yaw_joint": 0.25 * 5.0 / STIFFNESS_4010,
}

STIFFNESS_PATTERNS = {
    ".*_hip_pitch_joint": STIFFNESS_7520_14,
    ".*_hip_roll_joint": STIFFNESS_7520_22,
    ".*_hip_yaw_joint": STIFFNESS_7520_14,
    ".*_knee_joint": STIFFNESS_7520_22,
    ".*_ankle_pitch_joint": 2.0 * STIFFNESS_5020,
    ".*_ankle_roll_joint": 2.0 * STIFFNESS_5020,
    "waist_roll_joint": 2.0 * STIFFNESS_5020,
    "waist_pitch_joint": 2.0 * STIFFNESS_5020,
    "waist_yaw_joint": STIFFNESS_7520_14,
    ".*_shoulder_pitch_joint": STIFFNESS_5020,
    ".*_shoulder_roll_joint": STIFFNESS_5020,
    ".*_shoulder_yaw_joint": STIFFNESS_5020,
    ".*_elbow_joint": STIFFNESS_5020,
    ".*_wrist_roll_joint": STIFFNESS_5020,
    ".*_wrist_pitch_joint": STIFFNESS_4010,
    ".*_wrist_yaw_joint": STIFFNESS_4010,
}

DAMPING_PATTERNS = {
    ".*_hip_pitch_joint": DAMPING_7520_14,
    ".*_hip_roll_joint": DAMPING_7520_22,
    ".*_hip_yaw_joint": DAMPING_7520_14,
    ".*_knee_joint": DAMPING_7520_22,
    ".*_ankle_pitch_joint": 2.0 * DAMPING_5020,
    ".*_ankle_roll_joint": 2.0 * DAMPING_5020,
    "waist_roll_joint": 2.0 * DAMPING_5020,
    "waist_pitch_joint": 2.0 * DAMPING_5020,
    "waist_yaw_joint": DAMPING_7520_14,
    ".*_shoulder_pitch_joint": DAMPING_5020,
    ".*_shoulder_roll_joint": DAMPING_5020,
    ".*_shoulder_yaw_joint": DAMPING_5020,
    ".*_elbow_joint": DAMPING_5020,
    ".*_wrist_roll_joint": DAMPING_5020,
    ".*_wrist_pitch_joint": DAMPING_4010,
    ".*_wrist_yaw_joint": DAMPING_4010,
}

ARMATURE_PATTERNS = {
    ".*_hip_pitch_joint": ARMATURE_7520_14,
    ".*_hip_roll_joint": ARMATURE_7520_22,
    ".*_hip_yaw_joint": ARMATURE_7520_14,
    ".*_knee_joint": ARMATURE_7520_22,
    ".*_ankle_pitch_joint": 2.0 * ARMATURE_5020,
    ".*_ankle_roll_joint": 2.0 * ARMATURE_5020,
    "waist_roll_joint": 2.0 * ARMATURE_5020,
    "waist_pitch_joint": 2.0 * ARMATURE_5020,
    "waist_yaw_joint": ARMATURE_7520_14,
    ".*_shoulder_pitch_joint": ARMATURE_5020,
    ".*_shoulder_roll_joint": ARMATURE_5020,
    ".*_shoulder_yaw_joint": ARMATURE_5020,
    ".*_elbow_joint": ARMATURE_5020,
    ".*_wrist_roll_joint": ARMATURE_5020,
    ".*_wrist_pitch_joint": ARMATURE_4010,
    ".*_wrist_yaw_joint": ARMATURE_4010,
}

# 2026-06-17 DR run dropped all world-frame terms (cmd_dir_w, next_cmd_dir_w,
# pelvis_pos_xy_w, pelvis_yaw_cossin_w) -> yaw/position-invariant actor obs = 82.
ACTOR_OBSERVATION_NAMES = [
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "last_latent_action",
    "ball_pos_b",
    "ball_lin_vel_b",
    "target_dir_b",
    "target_speed",
    "next_target_speed",
]

DEPLOY_OBSERVATION_NAMES = ACTOR_OBSERVATION_NAMES + [
    "decoder_base_ang_vel",
    "decoder_joint_pos",
    "decoder_joint_vel",
    "last_decoded_action",
]

DEPLOY_OBSERVATION_DIMS = [
    3,
    3,
    29,
    29,
    8,
    3,
    3,
    2,
    1,
    1,
    3,
    29,
    29,
    29,
]


def resolve_patterns(patterns: dict[str, float], names: Iterable[str]) -> list[float]:
    values = []
    for name in names:
        matches = [float(v) for pat, v in patterns.items() if re.fullmatch(pat, name)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one match for {name}, got {len(matches)}")
        values.append(matches[0])
    return values


def resolve_action_scale_like_isaaclab(patterns: dict[str, float], names: Iterable[str]) -> list[float]:
    """Resolve action scale exactly like the trained Isaac action term did.

    ``LatentJointPositionAction`` calls IsaacLab's
    ``resolve_matching_names_values(..., preserve_order=True)`` and then uses
    the returned values directly as a dense per-joint tensor. In the IsaacLab
    version used here, that return value is grouped by regex key order rather
    than target joint order. The Stage-3 checkpoint was trained/evaluated with
    that behavior, so deployment metadata must preserve it for compatibility.
    """
    names = list(names)
    matched_by_name: dict[str, str] = {}
    grouped_values: list[float] = []
    for pattern, value in patterns.items():
        group = [name for name in names if re.fullmatch(pattern, name)]
        if not group:
            raise RuntimeError(f"Action scale pattern {pattern!r} did not match any joint")
        for name in group:
            if name in matched_by_name:
                raise RuntimeError(
                    f"Action scale joint {name!r} matched both {matched_by_name[name]!r} and {pattern!r}"
                )
            matched_by_name[name] = pattern
            grouped_values.append(float(value))
    missing = [name for name in names if name not in matched_by_name]
    if missing:
        raise RuntimeError(f"Action scale is missing joints: {missing}")
    return grouped_values


def csv(values: Iterable[object], decimals: int = 10) -> str:
    out = []
    for value in values:
        if isinstance(value, float):
            out.append(f"{value:.{decimals}g}")
        else:
            out.append(str(value))
    return ",".join(out)


# --- v2 obs layout (2026-06-21 run) -----------------------------------------
# Adds a single-frame `ball_radius` term (r-0.10) and a 10-frame flattened actor
# history (isaaclab policy group history_length=10). Single frame = 83 dims, so
# the actor input is 83*10 = 830. Decoder state_only is unchanged (single frame).
ACTOR_SINGLE_FRAME_NAMES_V2 = [
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "last_latent_action",
    "ball_pos_b",
    "ball_lin_vel_b",
    "ball_radius",
    "target_dir_b",
    "target_speed",
    "next_target_speed",
]
ACTOR_SINGLE_FRAME_DIMS_V2 = [3, 3, 29, 29, 8, 3, 3, 1, 2, 1, 1]  # sum = 83
DECODER_OBSERVATION_NAMES = [
    "decoder_base_ang_vel",
    "decoder_joint_pos",
    "decoder_joint_vel",
    "last_decoded_action",
]
DECODER_OBSERVATION_DIMS = [3, 29, 29, 29]  # sum = 90
ACTOR_HISTORY_LENGTH_V2 = 10


class SoftTouchActor(nn.Module):
    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        super().__init__()
        # Infer the actor input width from the checkpoint so this serves both the
        # 82-dim single-frame policy and the 830-dim (83x10 history) policy.
        in_dim = int(state_dict["actor.0.weight"].shape[1])
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 8),
        )
        self.net.load_state_dict(
            {
                key.removeprefix("actor."): value
                for key, value in state_dict.items()
                if key.startswith("actor.")
            }
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def build_mlp(in_dim: int, hidden: list[int], out_dim: int | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for width in hidden:
        layers.append(nn.Linear(last, width))
        layers.append(nn.SiLU())
        last = width
    if out_dim is not None:
        layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class SoftTouchVaeCore(nn.Module):
    def __init__(self, artifact: dict) -> None:
        super().__init__()
        arch = artifact["architecture"]
        self.state_only_dim = int(arch["state_only_dim"])
        self.action_dim = int(arch["action_dim"])
        self.latent_dim = int(arch["latent_dim"])
        self.logvar_min = float(arch["logvar_min"])
        self.logvar_max = float(arch["logvar_max"])

        prior_hidden = [int(x) for x in arch["prior_hidden"]]
        decoder_hidden = [int(x) for x in arch["decoder_hidden"]]

        self.prior_mlp = build_mlp(self.state_only_dim, prior_hidden)
        self.prior_mu = nn.Linear(prior_hidden[-1], self.latent_dim)
        self.prior_logvar = nn.Linear(prior_hidden[-1], self.latent_dim)
        self.decoder_mlp = build_mlp(self.state_only_dim + self.latent_dim, decoder_hidden)
        self.action_head = nn.Linear(decoder_hidden[-1], self.action_dim)

        full_state = artifact["student_state_dict"]
        self.load_state_dict(
            {
                key: value
                for key, value in full_state.items()
                if key.startswith("prior_")
                or key.startswith("decoder_")
                or key.startswith("action_head")
            }
        )

    def prior(self, state_only: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.prior_mlp(state_only)
        mu = self.prior_mu(h)
        logvar = self.prior_logvar(h).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar

    def decode(self, state_only: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_mlp(torch.cat([state_only, z], dim=-1))
        return self.action_head(h)


class SoftTouchDribbleDeployPolicy(nn.Module):
    def __init__(self, checkpoint: dict, artifact: dict, lab_lambda: float) -> None:
        super().__init__()
        self.actor = SoftTouchActor(checkpoint["model_state_dict"])
        # actor input width = single_frame_dim * history_length (830 for the v2
        # history policy, 82 for the original single-frame policy)
        self.actor_obs_dim = int(self.actor.net[0].in_features)
        self.decoder_state_dim = int(artifact["architecture"]["state_only_dim"])
        self.lab_lambda = float(lab_lambda)
        self.vae = SoftTouchVaeCore(artifact)

        norm = checkpoint["obs_norm_state_dict"]
        self.register_buffer("obs_mean", norm["_mean"].float())
        self.register_buffer("obs_std", norm["_std"].float())

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_obs = obs[:, : self.actor_obs_dim]
        decoder_state = obs[:, self.actor_obs_dim : self.actor_obs_dim + self.decoder_state_dim]
        norm_actor_obs = (actor_obs - self.obs_mean) / (self.obs_std + 1.0e-2)
        latent_action = self.actor(norm_actor_obs)
        mu_p, logvar_p = self.vae.prior(decoder_state)
        sigma_p = torch.exp(0.5 * logvar_p)
        z = mu_p + self.lab_lambda * sigma_p * torch.tanh(latent_action)
        raw_joint_action = self.vae.decode(decoder_state, z)
        return raw_joint_action, latent_action, z, mu_p, logvar_p


def attach_metadata(onnx_path: Path, checkpoint_path: Path, artifact_path: Path, lab_lambda: float,
                    obs_layout: str = "v1") -> None:
    model = onnx.load(str(onnx_path))
    if obs_layout == "v2":
        actor_names = ACTOR_SINGLE_FRAME_NAMES_V2
        actor_dims = ACTOR_SINGLE_FRAME_DIMS_V2
        actor_history = ACTOR_HISTORY_LENGTH_V2
    else:
        actor_names = ACTOR_OBSERVATION_NAMES
        actor_dims = DEPLOY_OBSERVATION_DIMS[: len(ACTOR_OBSERVATION_NAMES)]
        actor_history = 1
    obs_names = actor_names + DECODER_OBSERVATION_NAMES
    obs_dims = list(actor_dims) + DECODER_OBSERVATION_DIMS
    history_lengths = [actor_history] * len(actor_names) + [1] * len(DECODER_OBSERVATION_NAMES)
    actor_obs_dim = sum(actor_dims) * actor_history  # 830 for v2, 82 for v1
    metadata = {
        "policy_kind": "softtouch_dribble_latent_v1",
        "checkpoint_path": str(checkpoint_path),
        "artifact_path": str(artifact_path),
        "lab_lambda": lab_lambda,
        "joint_names": JOINT_NAMES,
        "joint_stiffness": resolve_patterns(STIFFNESS_PATTERNS, JOINT_NAMES),
        "joint_damping": resolve_patterns(DAMPING_PATTERNS, JOINT_NAMES),
        "joint_armature": resolve_patterns(ARMATURE_PATTERNS, JOINT_NAMES),
        "default_joint_pos": DEFAULT_JOINT_POS,
        "action_scale": resolve_action_scale_like_isaaclab(ACTION_SCALE_PATTERNS, JOINT_NAMES),
        "command_names": ["dribble_route"],
        "observation_names": obs_names,
        "observation_dims": obs_dims,
        "observation_history_lengths": history_lengths,
        "actor_observation_dim": actor_obs_dim,
        "decoder_state_dim": 90,
        "latent_dim": 8,
        "action_dim": 29,
        "obs_normalizer_eps": 1.0e-2,
        "onnx_input_layout": (
            f"obs = actor_obs[{actor_obs_dim}] + decoder_state_only[90]"
        ),
        "actor_obs_names": actor_names,
        "actor_history_length": actor_history,
        "route_defaults": [
            "reset_ball_forward_m=0.65",
            "route_seg_len_m=0.25",
            "route_lookahead_m=0.8",
            "route_preview_arc_m=1.0",
            "route_human_kappa_cap=0.5",
            "route_human_weave_mag=0.4:1.0",
            "route_lazy_extend=true",
            "route_init_segments=9",
            "route_extend_ahead_margin_segs=10",
            "route_extend_chunk=1",
            "cmd_mode_probs=0:0:0:0:1",
        ],
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = key
        if isinstance(value, list):
            entry.value = csv(value)
        else:
            entry.value = str(value)
        model.metadata_props.append(entry)
    onnx.save(model, str(onnx_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_DIR / "model_35000.pt"),
    )
    parser.add_argument(
        "--artifact",
        default=str(DEFAULT_CHECKPOINT_DIR / "stage2_decoder_dim8.pt"),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_CHECKPOINT_DIR / "softtouch_dribble_deploy.onnx"),
    )
    parser.add_argument("--lab-lambda", type=float, default=2.5)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--obs-layout", choices=["v1", "v2"], default="v1",
                        help="v1 = 82-dim single-frame actor; v2 = 83-dim single frame "
                             "(+ball_radius) x 10-frame history = 830-dim actor (2026-06-21 run)")
    args = parser.parse_args()

    checkpoint_path = Path(os.path.expanduser(args.checkpoint)).resolve()
    artifact_path = Path(os.path.expanduser(args.artifact)).resolve()
    output_path = Path(os.path.expanduser(args.output)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    policy = SoftTouchDribbleDeployPolicy(checkpoint, artifact, args.lab_lambda)
    policy.eval()

    obs_dim = policy.actor_obs_dim + policy.decoder_state_dim
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        outputs = policy(dummy_obs)
    print(f"[export] obs_dim={obs_dim}")
    print(f"[export] output shapes={[tuple(o.shape) for o in outputs]}")

    torch.onnx.export(
        policy,
        dummy_obs,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        input_names=["obs"],
        output_names=["actions", "latent_action", "latent_z", "prior_mu", "prior_logvar"],
        dynamic_axes={},
    )
    attach_metadata(output_path, checkpoint_path, artifact_path, args.lab_lambda, args.obs_layout)
    onnx.checker.check_model(str(output_path))
    print(f"[export] wrote {output_path}")


if __name__ == "__main__":
    main()
