#!/usr/bin/env python3
"""Generate domain-randomized MuJoCo model variants for C++ deployment-stack
robustness testing.

The ROS 2 sim2sim path loads a single MJCF. To probe robustness we bake the DR
into physical MJCF variants and sweep them with a bash loop (dr_robustness_sweep.sh),
so NO controller/plugin code changes are needed. Ranges mirror the training DR
used in tools/dribble_pysim_multi.py (verified against dribble_env_cfg.py):

    ball_mass      [0.352, 0.430] kg   (0.391 x [0.9, 1.1])   -> ball geom mass
    foot_friction  [0.50, 1.00]        (foot/floor dynamic)   -> foot-floor pairs
    ball_friction  [0.475, 0.525]                             -> ball-contact pairs
    ball_radius    [0.09, 0.11] m      (NOT trained; +/-10%)  -> ball geom size

ball_angular_damping is NOT an independent knob: it is 4*I = 4*(2/5*m*r^2), so it
is recomputed per variant and written to the manifest. The sweep passes it to the
bridge via softtouch_ball_angular_damping so the ball spin-decay stays physically
consistent with the randomized mass/radius. (The ball_radius *observation* stays
at nominal 0.10 since we do not touch C++, so radius DR also injects a small obs
error -- an intentional bonus robustness probe.)

Variant 000 is always the nominal (un-randomized) baseline.
"""
import argparse
import csv
import random
import re
from pathlib import Path

# Training DR ranges (match tools/dribble_pysim_multi.py DR dict).
DR = dict(
    ball_mass=(0.352, 0.430),
    ball_radius=(0.09, 0.11),
    foot_friction=(0.50, 1.00),
    ball_friction=(0.475, 0.525),
)
NOMINAL = dict(ball_mass=0.391, ball_radius=0.10, foot_friction=0.8, ball_friction=0.5)

# Exact friction strings in the base MJCF (all foot-floor / all ball pairs share one).
FOOT_FLOOR_FRICTION = 'friction="0.8 0.8 0.04"'
BALL_PAIR_FRICTION = 'friction="0.5 0.5 0.005 0.0001 0.0001"'


def angular_damping(mass: float, radius: float) -> float:
    """4 * I for a solid sphere: I = 2/5 * m * r^2."""
    return 4.0 * (0.4 * mass * radius * radius)


def make_variant(base_text: str, mass: float, radius: float, foot: float, ball: float) -> str:
    text = base_text

    # ball geom: size + mass (only inside the softtouch_ball_geom element)
    m = re.search(r'<geom name="softtouch_ball_geom".*?/>', text, re.DOTALL)
    if m is None:
        raise RuntimeError("softtouch_ball_geom not found in base MJCF")
    block = m.group(0)
    new_block = re.sub(r'size="[0-9.]+"', f'size="{radius:.5f}"', block, count=1)
    new_block = re.sub(r'mass="[0-9.]+"', f'mass="{mass:.5f}"', new_block, count=1)
    text = text.replace(block, new_block, 1)

    # foot-floor pairs (14, identical string) -> foot friction on the two tangential dims
    n_foot = text.count(FOOT_FLOOR_FRICTION)
    text = text.replace(FOOT_FLOOR_FRICTION, f'friction="{foot:.4f} {foot:.4f} 0.04"')

    # ball-contact pairs (floor_ball + 14 foot_ball, identical string) -> ball friction
    n_ball = text.count(BALL_PAIR_FRICTION)
    text = text.replace(BALL_PAIR_FRICTION, f'friction="{ball:.4f} {ball:.4f} 0.005 0.0001 0.0001"')

    if n_foot == 0 or n_ball == 0:
        raise RuntimeError(f"friction patterns not found (foot={n_foot}, ball={n_ball}); "
                           "base MJCF layout changed")
    return text


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(here / "mjcf" / "g1_softtouch_dribble.xml"),
                    help="base MJCF to randomize")
    ap.add_argument("--out-dir", default=str(here / "mjcf" / "dr_variants"),
                    help="output dir for variants (kept inside the package so ROS can load it)")
    ap.add_argument("--n", type=int, default=16,
                    help="TOTAL number of variants (variant 000 is the nominal baseline; "
                         "the remaining n-1 are randomized). n=1 -> nominal only.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible sampling")
    args = ap.parse_args()

    base_text = Path(args.base).read_text()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("g1_dr_*.xml"):  # drop variants from a previous (larger) run
        stale.unlink()
    rng = random.Random(args.seed)

    rows = []
    # variant 000 = nominal baseline
    specs = [("000", NOMINAL["ball_mass"], NOMINAL["ball_radius"],
              NOMINAL["foot_friction"], NOMINAL["ball_friction"])]
    for i in range(1, max(1, args.n)):
        specs.append((
            f"{i:03d}",
            rng.uniform(*DR["ball_mass"]),
            rng.uniform(*DR["ball_radius"]),
            rng.uniform(*DR["foot_friction"]),
            rng.uniform(*DR["ball_friction"]),
        ))

    for vid, mass, radius, foot, ball in specs:
        text = make_variant(base_text, mass, radius, foot, ball)
        path = out_dir / f"g1_dr_{vid}.xml"
        path.write_text(text)
        damp = angular_damping(mass, radius)
        rows.append(dict(variant=vid, mjcf=str(path), rel_model_file=f"/mjcf/dr_variants/g1_dr_{vid}.xml",
                         ball_mass=round(mass, 5), ball_radius=round(radius, 5),
                         foot_friction=round(foot, 4), ball_friction=round(ball, 4),
                         ball_angular_damping=round(damp, 6), route_seed=int(vid)))

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"[gen_dr_mjcf] wrote {len(rows)} variants (incl. nominal 000) to {out_dir}")
    print(f"[gen_dr_mjcf] manifest: {manifest}")


if __name__ == "__main__":
    main()
