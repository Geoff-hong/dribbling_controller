#!/usr/bin/env python3
"""Standalone MuJoCo dribble sim (single robot, Phase 1).

Runs the exported SoftTouch dribble ONNX policy directly (onnxruntime) inside a
plain MuJoCo sim — no ros2_control. Ports the deployment's obs assembly, route
generator and PD controller from the C++ (SoftTouchDribbleCommon.cpp /
SoftTouchDribbleObservation.cpp) so the Python rollout matches the C++ one.

Phase 1 goal: validate the port. Use --headless to step without a viewer and
print base height / forward progress (robot should stay up ~0.7 m and move).
Use the default (viewer) mode to watch it.
"""
import argparse
import re
import time
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort

# ---- route config (mirrors config/g1/softtouch_dribble_controllers.yaml route:) ----
ROUTE_CFG = dict(
    routeLength=20.0, routeSegmentLength=0.25, routeLookahead=0.8, routePreviewArc=1.0,
    routeCurvatureMin=0.0, routeCurvatureMax=0.0, routeSFlipArc=2.5,
    routeHumanKappaCap=0.5, routeHumanPersist=0.6, routeHumanWeaveMin=0.4, routeHumanWeaveMax=1.0,
    routeHumanBigProbability=0.09, routeHumanBigAngleMinDeg=40.0, routeHumanBigAngleMaxDeg=180.0,
    routeKvScale=0.75, routeVmax=2.0, routeLazyExtend=True, routeInitSegments=9,
    routeExtendChunk=1, routeExtendAheadMarginSegments=10,
)
CMD_MODE = 4
RESET_BALL_FORWARD = 0.65
JOINT_LIMIT_FACTOR = 0.9
DECIMATION = 4          # policy at 50 Hz, sim at 200 Hz
EFFORT_LIMIT = np.array([88., 88., 88., 139., 139., 50., 88., 88., 50., 139., 139.,
                         25., 25., 50., 50., 25., 25., 50., 50., 25., 25., 25., 25.,
                         25., 25., 5., 5., 5., 5.])


# --------------------------- quaternion helpers (wxyz) ---------------------------
def world_to_body(quat_wxyz, vec):
    negq = np.zeros(4); res = np.zeros(3)
    mujoco.mju_negQuat(negq, np.ascontiguousarray(quat_wxyz, dtype=np.float64))
    mujoco.mju_rotVecQuat(res, np.ascontiguousarray(vec, dtype=np.float64), negq)
    return res


def yaw_from_quat(q):
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# --------------------------- route generator (port) ---------------------------
class Route:
    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.rng = np.random.Generator(np.random.PCG64(seed))
        n = max(1, int(round(cfg["routeLength"] / max(cfg["routeSegmentLength"], 1e-9))))
        self.points = np.zeros((n + 1, 2))
        self.speed = np.zeros(n)
        self.filled = 0
        self.end_heading = 0.0
        self.last_seg = -1
        self.h_sign = 1.0; self.big_remain = 0.0; self.big_sign = 1.0

    def _u(self, lo, hi):
        return self.rng.uniform(lo, hi)

    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v)
        return np.array([1.0, 0.0]) if n < 1e-9 else v / n

    def reset(self, origin, forward, cmd_mode):
        self.cmd_mode = cmd_mode
        self.cmd_sign = -1.0 if cmd_mode == 2 else 1.0
        self.last_seg = -1
        max_seg = len(self.speed)
        init = (np.clip(self.cfg["routeInitSegments"], 1, max_seg)
                if self.cfg["routeLazyExtend"] else max_seg)
        self._build(int(init), True, np.asarray(origin, float), self._unit(np.asarray(forward, float)))
        return self.update(origin)

    def update(self, ball_xy):
        self._extend()
        ball_xy = np.asarray(ball_xy, float)
        filled = max(1, self.filled)
        best_d2 = np.inf; best_t = 0.0; best_seg = 0; best_proj = self.points[0]
        for i in range(filled):
            a = self.points[i]; b = self.points[i + 1]; ab = b - a
            ab2 = max(ab @ ab, 1e-9)
            t = np.clip((ball_xy - a) @ ab / ab2, 0.0, 1.0)
            proj = a + t * ab
            d2 = (ball_xy - proj) @ (ball_xy - proj)
            if d2 < best_d2:
                best_d2, best_t, best_seg, best_proj = d2, t, i, proj
        self.last_seg = best_seg
        s_star = (best_seg + best_t) * self.cfg["routeSegmentLength"]
        nsi = int(np.clip(np.floor((s_star + self.cfg["routeLookahead"]) / self.cfg["routeSegmentLength"]),
                          0, len(self.speed) - 1))
        return dict(
            target_speed=self.speed[best_seg],
            next_target_speed=self.speed[nsi],
            target_dir=self._unit(self._point_at(s_star + self.cfg["routeLookahead"]) - ball_xy),
            next_target_dir=self._unit(
                self._point_at(s_star + self.cfg["routeLookahead"] + self.cfg["routePreviewArc"]) - ball_xy),
        )

    def _point_at(self, arc):
        max_f = max(0.0, self.filled - 1e-4)
        f = min(max(arc / self.cfg["routeSegmentLength"], 0.0), max_f)
        i = int(np.clip(int(f), 0, len(self.points) - 2))
        frac = f - i
        return self.points[i] + frac * (self.points[i + 1] - self.points[i])

    def _build(self, num, init, origin=None, forward=None):
        ds = self.cfg["routeSegmentLength"]
        seg_off = 0 if init else self.filled
        org = origin if init else self.points[seg_off]
        theta = self.end_heading
        if init:
            theta = np.arctan2(forward[1], forward[0])
            self.h_sign = 1.0 if self._u(0, 1) < 0.5 else -1.0
            self.big_remain = 0.0; self.big_sign = 1.0
            self.points[0] = org
        if self.cmd_mode == 4:
            kappa = self._human_kappa(num)
        else:
            kappa = np.zeros(num)  # cmd_mode 4 only in deploy; others unused here
        heading = theta; point = np.asarray(org, float).copy()
        for i in range(num):
            point = point + np.array([np.cos(heading), np.sin(heading)]) * ds
            self.points[seg_off + 1 + i] = point
            kabs = max(abs(kappa[i]), 1e-3)
            self.speed[seg_off + i] = min(self.cfg["routeVmax"], np.sqrt(self.cfg["routeKvScale"] / kabs))
            heading += kappa[i] * ds
        self.end_heading = heading
        self.filled = seg_off + num

    def _human_kappa(self, num):
        cap = self.cfg["routeHumanKappaCap"]; ds = self.cfg["routeSegmentLength"]
        amin = np.deg2rad(self.cfg["routeHumanBigAngleMinDeg"]); amax = np.deg2rad(self.cfg["routeHumanBigAngleMaxDeg"])
        out = np.zeros(num)
        for i in range(num):
            in_big = self.big_remain > 0.0
            if not in_big and self._u(0, 1) < self.cfg["routeHumanBigProbability"]:
                angle = self._u(amin, amax)
                self.big_remain = max(2.0, np.ceil(angle / (cap * ds)))
                self.big_sign = 1.0 if self._u(0, 1) < 0.5 else -1.0
                in_big = True
            if self._u(0, 1) > self.cfg["routeHumanPersist"]:
                self.h_sign = -self.h_sign
            mag = self._u(self.cfg["routeHumanWeaveMin"], self.cfg["routeHumanWeaveMax"]) * cap
            out[i] = self.big_sign * cap if in_big else self.h_sign * mag
            if in_big:
                self.big_remain -= 1.0
        return out

    def _extend(self):
        if not self.cfg["routeLazyExtend"] or self.last_seg < 0:
            return
        max_seg = len(self.speed)
        if self.filled >= max_seg or (self.filled - self.last_seg) >= self.cfg["routeExtendAheadMarginSegments"]:
            return
        num = min(self.cfg["routeExtendChunk"], max_seg - self.filled)
        if num > 0:
            self._build(num, False)


# --------------------------- metadata parsing ---------------------------
def csv_floats(s):
    return np.array([float(x) for x in re.split(r"[,\s]+", s.strip()) if x != ""])


def load_policy(onnx_path):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta = {p.key: p.value for p in sess.get_modelmeta().custom_metadata_map.items()} \
        if hasattr(sess.get_modelmeta(), "custom_metadata_map") else {}
    md = sess.get_modelmeta().custom_metadata_map
    return sess, md


# --------------------------- main sim ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", default="/home/aldebaran/Desktop/dribbling_controller/mjcf/g1_softtouch_dribble.xml")
    ap.add_argument("--onnx", default="/home/aldebaran/Desktop/SoftTouch-multiagent/logs/rsl_rl/g1_dribble/2026-06-17_16-44-29/softtouch_dribble_deploy_iter50000.onnx")
    ap.add_argument("--reset", default="/home/aldebaran/Desktop/dribbling_controller/config/g1/softtouch_mujoco_reset_walkf_rf_frame0.txt")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    md = sess.get_modelmeta().custom_metadata_map
    jnames = [s for s in md["joint_names"].split(",")]
    kp = csv_floats(md["joint_stiffness"]); kd = csv_floats(md["joint_damping"])
    action_scale = csv_floats(md["action_scale"]); default_q = csv_floats(md["default_joint_pos"])
    nj = len(jnames)
    assert nj == 29, nj

    # index maps: policy joint -> mjModel qpos/dof/actuator
    qadr = np.zeros(nj, int); vadr = np.zeros(nj, int); aadr = np.zeros(nj, int)
    for i, nm in enumerate(jnames):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        qadr[i] = model.jnt_qposadr[jid]; vadr[i] = model.jnt_dofadr[jid]
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
        aadr[i] = aid
    # joint clip box from config joint_limit_lower/upper (policy order) * factor (matches C++)
    jlo = np.array([-2.5307, -2.5307, -2.618, -0.5236, -2.9671, -0.52, -2.7576, -2.7576, -0.52,
                    -0.087267, -0.087267, -3.0892, -3.0892, -0.87267, -0.87267, -1.5882, -2.2515,
                    -0.2618, -0.2618, -2.618, -2.618, -1.0472, -1.0472, -1.97222, -1.97222,
                    -1.61443, -1.61443, -1.61443, -1.61443])
    jhi = np.array([2.8798, 2.8798, 2.618, 2.9671, 0.5236, 0.52, 2.7576, 2.7576, 0.52, 2.8798, 2.8798,
                    2.6704, 2.6704, 0.5236, 0.5236, 2.2515, 1.5882, 0.2618, 0.2618, 2.618, 2.618,
                    2.0944, 2.0944, 1.97222, 1.97222, 1.61443, 1.61443, 1.61443, 1.61443])
    jc = 0.5 * (jlo + jhi); jhw = 0.5 * (jhi - jlo) * JOINT_LIMIT_FACTOR

    base_j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    base_q = model.jnt_qposadr[base_j]; base_v = model.jnt_dofadr[base_j]
    ball_j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "softtouch_ball_freejoint")
    ball_q = model.jnt_qposadr[ball_j]; ball_v = model.jnt_dofadr[ball_j]
    # ball damping is applied at runtime by the bridge plugin, NOT in the MJCF -> replicate it
    model.dof_damping[ball_v:ball_v + 3] = 0.0          # translational
    model.dof_damping[ball_v + 3:ball_v + 6] = 0.006256  # angular (= 4*I for r=0.10/m=0.391)

    # ---- reset from the deployment reset-state file ----
    rs = {}
    for line in open(args.reset):
        t = line.split()
        if not t or t[0].startswith("#"):
            continue
        rs[t[0]] = t[1:]
    mujoco.mj_resetData(model, data)
    data.qpos[base_q:base_q + 3] = [float(x) for x in rs["root_pos"]]
    data.qpos[base_q + 3:base_q + 7] = [float(x) for x in rs["root_quat"]]
    data.qvel[base_v:base_v + 3] = [float(x) for x in rs["root_lin_vel"]]
    data.qvel[base_v + 3:base_v + 6] = [float(x) for x in rs["root_ang_vel_body"]]
    rj_names = rs["joint_names"]; rj_pos = [float(x) for x in rs["joint_pos"]]; rj_vel = [float(x) for x in rs["joint_vel"]]
    name2idx = {nm: k for k, nm in enumerate(rj_names)}
    for i, nm in enumerate(jnames):
        k = name2idx[nm]
        data.qpos[qadr[i]] = rj_pos[k]; data.qvel[vadr[i]] = rj_vel[k]
    data.qpos[ball_q:ball_q + 3] = [float(x) for x in rs["ball_pos"]]
    data.qpos[ball_q + 3:ball_q + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    # ---- route init (anchored at ball, heading = pelvis forward) ----
    base_quat = data.qpos[base_q + 3:base_q + 7].copy()
    fwd3 = np.zeros(3); mujoco.mju_rotVecQuat(fwd3, np.array([1.0, 0, 0]), base_quat)
    ball_xy = data.qpos[ball_q:ball_q + 2].copy()
    route = Route(ROUTE_CFG, args.seed)
    cmd = route.reset(ball_xy, fwd3[:2], CMD_MODE)

    # route visualization dots (mocap spheres in the MJCF) -> drive them along the route
    dot_mocap = []
    for k in range(40):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"route_dot_{k}")
        dot_mocap.append(int(model.body_mocapid[bid]) if bid >= 0 else -1)

    def update_route_dots():
        npts = route.filled + 1
        nd = len(dot_mocap)
        for i, m in enumerate(dot_mocap):
            if m < 0:
                continue
            if npts < 2:
                data.mocap_pos[m] = [0.0, 0.0, -1.0]
                continue
            idx = (i * (npts - 1) + (nd - 1) // 2) // (nd - 1) if nd > 1 else 0
            p = route.points[idx]
            data.mocap_pos[m] = [p[0], p[1], 0.10]

    prev_latent = np.zeros(8, np.float32)
    prev_decoded = np.zeros(nj, np.float32)
    DECODER_DIM = 90

    def build_obs():
        bq = data.qpos[base_q + 3:base_q + 7]
        pelvis = data.qpos[base_q:base_q + 3]
        base_angvel = data.qvel[base_v + 3:base_v + 6]          # body frame (free joint)
        proj_grav = world_to_body(bq, [0, 0, -1.0])
        q = data.qpos[qadr] - default_q
        qd = data.qvel[vadr]
        ball_w = data.qpos[ball_q:ball_q + 3]
        ball_vw = data.qvel[ball_v:ball_v + 3]
        ball_b = world_to_body(bq, ball_w - pelvis)
        ball_vb = world_to_body(bq, ball_vw)
        tdir_b = world_to_body(bq, [cmd["target_dir"][0], cmd["target_dir"][1], 0.0])[:2]
        actor = np.concatenate([
            base_angvel, proj_grav, q, qd, prev_latent, ball_b, ball_vb,
            tdir_b, [cmd["target_speed"]], [cmd["next_target_speed"]],
        ])
        decoder = np.concatenate([base_angvel, q, qd, prev_decoded])
        assert len(actor) == 82 and len(decoder) == DECODER_DIM, (len(actor), len(decoder))
        return np.concatenate([actor, decoder]).astype(np.float32)[None, :]

    target = data.qpos[qadr].copy()
    ts = model.opt.timestep
    nsteps = int(args.seconds / ts)
    zmin, zmax = 99.0, -99.0
    x0 = data.qpos[base_q]

    def policy_step():
        nonlocal cmd, prev_latent, prev_decoded, target
        ball_xy = data.qpos[ball_q:ball_q + 2].copy()
        cmd = route.update(ball_xy)
        update_route_dots()
        obs = build_obs()
        actions, latent, *_ = sess.run(None, {"obs": obs})
        prev_decoded = actions[0].copy()
        prev_latent = latent[0].copy()
        tgt = default_q + action_scale * actions[0]
        target = np.clip(tgt, jc - jhw, jc + jhw)

    def control_torque():
        q = data.qpos[qadr]; qd = data.qvel[vadr]
        tau = kp * (target - q) - kd * qd
        return np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)

    def step_once(i):
        if i % DECIMATION == 0:
            policy_step()
        data.ctrl[aadr] = control_torque()
        mujoco.mj_step(model, data)

    if args.headless:
        for i in range(nsteps):
            step_once(i)
            z = data.qpos[base_q + 2]; zmin = min(zmin, z); zmax = max(zmax, z)
            if not np.all(np.isfinite(data.qpos)):
                print(f"[pysim] DIVERGED at step {i} (t={i*ts:.2f}s)"); return
        dx = data.qpos[base_q] - x0
        print(f"[pysim] survived {args.seconds:.0f}s | base z range [{zmin:.3f},{zmax:.3f}] "
              f"(fell if <~0.4) | forward dx={dx:.2f} m | final ball-robot gap="
              f"{np.linalg.norm(data.qpos[ball_q:ball_q+2]-data.qpos[base_q:base_q+2]):.2f} m")
    else:
        with mujoco.viewer.launch_passive(model, data) as v:
            i = 0
            while v.is_running():
                t0 = time.time()
                step_once(i); i += 1
                if i % DECIMATION == 0:
                    v.sync()
                dt = model.opt.timestep - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)  # pace to real time


if __name__ == "__main__":
    main()
