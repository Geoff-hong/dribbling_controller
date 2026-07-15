#!/usr/bin/env python3
"""Standalone ball-roll test for sim2sim damping calibration.

Replicates ONLY the ball + floor + contact params from
mjcf/g1_softtouch_dribble.xml (no robot), so we can measure:
  1. numerical stability of the ball under a given integrator + angular damping
  2. roll distance and velocity-decay time constant tau for a ball launched
     rolling at v0 = 1 m/s.

PhysX/training reference (from SoftTouch-multiagent docs):
  real grass roll  ~0.72 m @ v=1 m/s
  PhysX sim roll   ~0.88 m @ v=1 m/s, tau ~0.87 s
Note: for exponential decay, roll_distance ~= v0 * tau.

The ball angular damping in the real deploy is applied at runtime by
SoftTouchMujocoBallBridgePlugin onto dof_damping[rot] of the ball freejoint.
Here we set it directly to mirror that.
"""

import argparse
import numpy as np
import mujoco

BALL_RADIUS = 0.09
BALL_MASS = 0.34
V0 = 1.0  # initial rolling speed (m/s)

MJCF_TEMPLATE = """
<mujoco model="ball_roll_test">
  <option timestep="0.005" integrator="{integrator}"/>
  <worldbody>
    <geom name="floor" size="0 0 0.01" type="plane"/>
    <body name="ball" pos="0 0 {radius}">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="{radius}" mass="{mass}"
            condim="6" friction="0.5 0.005 0.0001"/>
    </body>
  </worldbody>
  <contact>
    <pair name="floor_ball" geom1="floor" geom2="ball_geom" condim="6"
          solref="0.01 1" friction="0.5 0.5 0.005 0.0001 0.0001"/>
  </contact>
</mujoco>
"""


def run_case(integrator, angular_damping, v0=V0, max_time=8.0, verbose=False):
    xml = MJCF_TEMPLATE.format(integrator=integrator, radius=BALL_RADIUS, mass=BALL_MASS)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    dofadr = model.jnt_dofadr[jid]
    # translational dofs 0..2 -> 0 damping; rotational dofs 3..5 -> angular_damping
    model.dof_damping[dofadr + 0:dofadr + 3] = 0.0
    model.dof_damping[dofadr + 3:dofadr + 6] = angular_damping

    # report the ball's actual rotational inertia (so we can check 4*I etc.)
    mujoco.mj_forward(model, data)
    Iyy = model.body_inertia[model.jnt_bodyid[jid]][1]

    # initial state: resting on floor, rolling in +x without slip:
    # v = (v0,0,0), omega = (0, v0/r, 0)
    data.qpos[:3] = [0.0, 0.0, BALL_RADIUS]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[0:3] = [v0, 0.0, 0.0]
    data.qvel[3:6] = [0.0, v0 / BALL_RADIUS, 0.0]

    ts = model.opt.timestep
    x0 = data.qpos[0]
    times, vxs, zs = [], [], []
    stable = True
    reason = "stopped"
    t = 0.0
    while t < max_time:
        mujoco.mj_step(model, data)
        t += ts
        vx = data.qvel[0]
        z = data.qpos[2]
        times.append(t); vxs.append(vx); zs.append(z)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            stable = False; reason = "NaN/Inf"; break
        if abs(z - BALL_RADIUS) > 0.05:  # launched off / sank into floor
            stable = False; reason = f"z-deviation {z:.3f}"; break
        if vx < 0.01:
            reason = "stopped (vx<0.01)"; break

    times = np.array(times); vxs = np.array(vxs); zs = np.array(zs)
    distance = data.qpos[0] - x0
    # effective tau: time for vx to fall to v0/e
    tau = float("nan")
    target = v0 / np.e
    idx = np.where(vxs <= target)[0]
    if len(idx) > 0:
        tau = times[idx[0]]
    z_dev = float(np.max(np.abs(zs - BALL_RADIUS))) if len(zs) else float("nan")

    return dict(integrator=integrator, d=angular_damping, Iyy=float(Iyy),
                distance=float(distance), tau=tau, t_stop=float(times[-1]) if len(times) else 0.0,
                stable=stable, reason=reason, z_dev=z_dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrators", nargs="+", default=["implicitfast", "Euler"])
    ap.add_argument("--dampings", nargs="+", type=float,
                    default=[4.0, 0.044, 0.0044064, 0.0])
    args = ap.parse_args()

    I_expected = 0.4 * BALL_MASS * BALL_RADIUS ** 2
    print(f"ball solid-sphere inertia I = 2/5*m*r^2 = {I_expected:.7f} kg*m^2")
    print(f"4*I (PhysX angular_damping=4.0 equivalent) = {4*I_expected:.7f}")
    print(f"reference: PhysX roll ~0.88 m, tau ~0.87 s @ v0=1 m/s\n")
    print(f"{'integrator':<14}{'damping':>12}{'roll[m]':>10}{'tau[s]':>9}"
          f"{'t_stop[s]':>10}{'z_dev[m]':>10}  stable / reason")
    print("-" * 90)
    for integ in args.integrators:
        for d in args.dampings:
            r = run_case(integ, d)
            print(f"{r['integrator']:<14}{r['d']:>12.7f}{r['distance']:>10.3f}"
                  f"{r['tau']:>9.3f}{r['t_stop']:>10.3f}{r['z_dev']:>10.4f}  "
                  f"{'OK' if r['stable'] else 'UNSTABLE':<9}{r['reason']}")
        print()


if __name__ == "__main__":
    main()
