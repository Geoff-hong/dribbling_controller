#!/usr/bin/env python3
"""Generate the ball-contact collision shells used by the hybridfoot MJCF.

Sources (Unitree G1 visual STLs, ROS install):
  /opt/ros/jazzy/share/unitree_description/meshes/g1/{left,right}_ankle_roll_link.STL
  /opt/ros/jazzy/share/unitree_description/meshes/g1/{left,right}_knee_link.STL

Foot shell : ankle_roll STL cropped at z <= +0.005 (removes the ankle-mount
             boss so the convex hull does not bulge above the real shoe top),
             then convex hull.
Shin shell : knee_link STL convex hull (the tapered shin is near-convex, no
             crop needed).

Outputs, per side and part:
  --emit inline : vertex list ("x y z x y z ...") for MJCF <mesh vertex="..."/>
                  (what mjcf/g1_softtouch_dribble_hybridfoot.xml embeds)
  --emit stl    : binary STL of the hull, for the training-side URDF/Isaac.
                  Use --max-verts 64 there: PhysX convex meshes are capped at
                  64 vertices, the grid is coarsened until the hull fits.

Benchmark-side settings (committed XML): --grid 0.0005, no --max-verts.
"""
import argparse
import os
import struct

import numpy as np
from scipy.spatial import ConvexHull

MESH_DIR = "/opt/ros/jazzy/share/unitree_description/meshes/g1"
FOOT_CUT = 0.005      # keep z <= cut; boss starts above the shoe top here
SIDES = ("left", "right")


def load_stl(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    rec = np.frombuffer(b[84:84 + n * 50], dtype=np.uint8).reshape(n, 50)
    return rec[:, 12:48].copy().view("<f4").reshape(n, 3, 3)


def crop_z(tri, cut):
    """Vertices below the cut plane plus exact edge/plane intersections."""
    v = tri.reshape(-1, 3)
    pts = [v[v[:, 2] <= cut]]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        p, q = tri[:, a, :], tri[:, b, :]
        crossing = (p[:, 2] - cut) * (q[:, 2] - cut) < 0
        t = (cut - p[crossing, 2]) / (q[crossing, 2] - p[crossing, 2])
        pts.append(p[crossing] + t[:, None] * (q[crossing] - p[crossing]))
    return np.vstack(pts)


def hull_verts(points, grid, max_verts=None):
    hv = points[ConvexHull(points).vertices]
    while True:
        snapped = np.unique(np.round(hv / grid) * grid, axis=0)
        out = snapped[ConvexHull(snapped).vertices]
        if max_verts is None or len(out) <= max_verts:
            return out, grid
        grid *= 1.3


def write_stl(path, verts):
    hull = ConvexHull(verts)
    tris = []
    center = verts.mean(axis=0)
    for simplex, eq in zip(hull.simplices, hull.equations):
        a, b, c = verts[simplex]
        n = eq[:3]
        if np.dot(np.cross(b - a, c - a), n) < 0:      # outward winding
            b, c = c, b
        tris.append((n, a, b, c))
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for n, a, b, c in tris:
            f.write(np.asarray([n, a, b, c], dtype="<f4").tobytes())
            f.write(b"\0\0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=float, default=0.0005,
                    help="vertex snap grid in m (coarsened if --max-verts binds)")
    ap.add_argument("--max-verts", type=int, default=None,
                    help="cap hull vertex count (PhysX convex limit: 64)")
    ap.add_argument("--emit", choices=("inline", "stl"), default="inline")
    ap.add_argument("--out-dir", default=".", help="output dir for --emit stl")
    args = ap.parse_args()

    parts = []
    for side in SIDES:
        foot = load_stl(f"{MESH_DIR}/{side}_ankle_roll_link.STL")
        parts.append((f"{side}_footshell", crop_z(foot, FOOT_CUT)))
        shin = load_stl(f"{MESH_DIR}/{side}_knee_link.STL").reshape(-1, 3)
        parts.append((f"{side}_shinshell", shin))

    for name, pts in parts:
        hv, used_grid = hull_verts(pts, args.grid, args.max_verts)
        if args.emit == "inline":
            vtx = " ".join(f"{c:.4f}" for c in hv.ravel())
            print(f'<mesh name="{name}" vertex="{vtx}"/>')
            print(f"<!-- {name}: {len(hv)} verts, grid {used_grid:.4g} -->")
        else:
            path = os.path.join(args.out_dir, f"{name}.STL")
            write_stl(path, hv)
            print(f"{path}: {len(hv)} verts (grid {used_grid:.4g})")


if __name__ == "__main__":
    main()
