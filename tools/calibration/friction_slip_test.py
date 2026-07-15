#!/usr/bin/env python3
"""Quantify foot-ground slip vs MuJoCo contact-solver settings (impratio, cone).

Motivation: the foot-floor friction COEFFICIENT (0.8) is in the training range,
but MuJoCo's soft friction constraint with the default impratio=1 lets a loaded
contact CREEP even when the tangential force is below the Coulomb limit. PhysX
(training) friction is effectively rigid -> negligible stance slip. impratio has
no closed-form match to PhysX, so we pick it empirically: the smallest value that
brings stance slip down into the "rigid" range.

Test: a box of mass m rests on the floor (normal load N = m*g). We apply a constant
horizontal force F = frac * mu * N (frac<1, i.e. BELOW the slip limit, so the ideal
rigid answer is ZERO motion). We measure how far it creeps over one stance duration.
Sweep impratio and cone. Also include frac>1 as a sanity check that it DOES slide.
"""

import argparse
import numpy as np
import mujoco

G = 9.81
MU = 0.8                  # foot-floor slide friction (matches MJCF foot-floor pair)
MASS = 35.0               # ~G1 body mass carried on one stance foot (worst case)
STANCE_T = 0.3            # one stance phase ~0.3 s
SLIP_OK_MM = 1.0          # criterion: stance creep below this is "rigid enough"

MJCF = """
<mujoco model="friction_slip_test">
  <option timestep="0.005" integrator="implicitfast" cone="{cone}" impratio="{impratio}"/>
  <worldbody>
    <geom name="floor" size="0 0 0.01" type="plane"/>
    <body name="box" pos="0 0 0.025">
      <freejoint name="box_free"/>
      <geom name="box_geom" type="box" size="0.09 0.03 0.025" mass="{mass}"
            condim="4" friction="{mu} {mu} 0.04"/>
    </body>
  </worldbody>
</mujoco>
"""


def run_case(impratio, cone, frac, settle_t=0.4, push_t=STANCE_T):
    xml = MJCF.format(cone=cone, impratio=impratio, mass=MASS, mu=MU)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ts = model.opt.timestep

    N = MASS * G
    F = frac * MU * N  # horizontal force along +x

    # 1) settle under gravity (no push) so the contact is established
    for _ in range(int(settle_t / ts)):
        mujoco.mj_step(model, data)

    x_start = data.qpos[0]
    vx_settle = abs(data.qvel[0])

    # 2) apply constant horizontal force for one stance duration
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    nstep = int(push_t / ts)
    for _ in range(nstep):
        data.xfrc_applied[:] = 0.0
        data.xfrc_applied[body_id, 0] = F  # world-frame force in +x
        mujoco.mj_step(model, data)

    slip = data.qpos[0] - x_start
    vx_end = data.qvel[0]
    return dict(impratio=impratio, cone=cone, frac=frac,
                slip_mm=slip * 1e3, vx_end=vx_end, vx_settle=vx_settle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impratios", nargs="+", type=float, default=[1, 5, 10, 20, 50, 100])
    ap.add_argument("--cones", nargs="+", default=["pyramidal", "elliptic"])
    args = ap.parse_args()

    N = MASS * G
    print(f"box mass={MASS} kg -> normal load N={N:.0f} N, mu={MU}, "
          f"Coulomb limit mu*N={MU*N:.0f} N")
    print(f"stance push duration={STANCE_T}s; criterion: below-limit creep < {SLIP_OK_MM} mm\n")

    # frac=0.95 -> just below slip limit (the demanding stance case; ideal=0 motion)
    # frac=1.20 -> above limit (sanity: must slide)
    for frac in (0.95, 1.20):
        tag = "BELOW limit (ideal: ~0 slip)" if frac < 1 else "ABOVE limit (sanity: must slide)"
        print(f"=== F = {frac}*mu*N   [{tag}] ===")
        print(f"{'cone':<11}{'impratio':>9}{'stance_slip[mm]':>17}{'vx_end[m/s]':>13}")
        print("-" * 52)
        for cone in args.cones:
            for ip in args.impratios:
                r = run_case(ip, cone, frac)
                flag = ""
                if frac < 1:
                    flag = "  <- OK" if abs(r["slip_mm"]) < SLIP_OK_MM else ""
                print(f"{cone:<11}{ip:>9.0f}{r['slip_mm']:>17.3f}{r['vx_end']:>13.4f}{flag}")
            print()


if __name__ == "__main__":
    main()
