#!/usr/bin/env python3
"""Draw the proposed G1 upper-chest MID-360 installation.

The torso outline is projected directly from Unitree's torso_link STL.  All
mount coordinates are expressed in torso_link coordinates (+X forward, +Y
left, +Z up).  The MID-360 housing is a 65 x 65 x 60 mm schematic envelope;
the white point is the optical origin used by the coverage calculation.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon
from scipy.spatial import ConvexHull


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "analysis" / "softtouch_20260727_night"
TORSO_STL = Path(
    "/opt/ros/jazzy/share/unitree_description/meshes/g1/"
    "torso_link_rev_1_0.STL"
)

ANCHOR_LEFT = np.array([0.0039563, 0.10022, 0.24778])
ANCHOR_RIGHT = np.array([0.0039563, -0.10022, 0.24778])
BRACKET_RAIL_X = 0.090
SENSOR_ORIGIN = np.array([0.125, 0.0, 0.248])
SENSOR_SIZE = np.array([0.065, 0.065, 0.060])
DOWN_PITCH_DEG = 35.5
PRACTICAL_PITCH_RANGE_DEG = (33.0, 37.25)
FOV_NATIVE_DEG = (-7.0, 52.0)

# Recomputed from the 23,075 trusted samples at SENSOR_ORIGIN.
CENTER_COVERAGE_PCT = 98.7302
FULL_BALL_COVERAGE_PCT = 97.4085
RANGE_99_M = (0.868, 1.836)
AZIMUTH_99_DEG = (-81.85, 76.63)

BG = "#10151d"
PANEL = "#171e28"
GRID = "#34404f"
TEXT = "#edf3fa"
MUTED = "#9caabd"
TORSO = "#557fae"
TORSO_EDGE = "#9fc7ed"
MOUNT = "#ff9f1c"
SENSOR = "#29bde9"
FOV = "#2ed4a7"
ANGLE = "#ffd166"
ORIGIN = "#ffffff"


def configure_matplotlib() -> None:
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "grid.color": GRID,
            "grid.alpha": 0.35,
        }
    )


def load_binary_stl_vertices(path: Path) -> np.ndarray:
    """Return every triangle vertex from a binary STL."""
    triangle_dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]
    )
    with path.open("rb") as stream:
        stream.seek(80)
        triangle_count = struct.unpack("<I", stream.read(4))[0]
        triangles = np.fromfile(
            stream, dtype=triangle_dtype, count=triangle_count
        )
    if triangles.size != triangle_count:
        raise ValueError(f"Incomplete binary STL: {path}")
    return triangles["vertices"].reshape(-1, 3).astype(float)


def rotation_y(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def sensor_rotation() -> np.ndarray:
    """Inverted MID-360, with its +X axis pitched down in torso coordinates."""
    rz_pi = np.diag([-1.0, -1.0, 1.0])
    return rz_pi @ rotation_y(180.0 - DOWN_PITCH_DEG)


def box_corners(center: np.ndarray, size: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    signs = np.array(
        [
            [sx, sy, sz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    )
    local = signs * size / 2.0
    return center + local @ rotation.T


def convex_polygon(points_2d: np.ndarray) -> np.ndarray:
    unique = np.unique(np.round(points_2d, decimals=7), axis=0)
    hull = ConvexHull(unique)
    return unique[hull.vertices]


def add_hull(ax: plt.Axes, points: np.ndarray, dims: tuple[int, int]) -> None:
    polygon = convex_polygon(points[:, list(dims)])
    ax.add_patch(
        Polygon(
            polygon,
            closed=True,
            facecolor=TORSO,
            edgecolor=TORSO_EDGE,
            linewidth=1.4,
            alpha=0.58,
            zorder=1,
        )
    )


def plot_polyline(
    ax: plt.Axes,
    points: np.ndarray,
    dims: tuple[int, int],
    *,
    width: float = 4.0,
) -> None:
    values = points[:, list(dims)]
    ax.plot(
        values[:, 0],
        values[:, 1],
        color=MOUNT,
        linewidth=width,
        solid_capstyle="round",
        zorder=5,
    )


def add_sensor_polygon(
    ax: plt.Axes, corners: np.ndarray, dims: tuple[int, int]
) -> None:
    polygon = convex_polygon(corners[:, list(dims)])
    ax.add_patch(
        Polygon(
            polygon,
            closed=True,
            facecolor=SENSOR,
            edgecolor="#d7f7ff",
            linewidth=1.6,
            alpha=0.86,
            zorder=6,
        )
    )


def add_origin(ax: plt.Axes, xy: tuple[float, float]) -> None:
    ax.scatter(
        [xy[0]],
        [xy[1]],
        s=42,
        facecolor=ORIGIN,
        edgecolor=BG,
        linewidth=1.2,
        zorder=9,
    )


def style_axes(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def draw_front(
    ax: plt.Axes, torso_vertices: np.ndarray, sensor_corners: np.ndarray
) -> None:
    ax.set_title("FRONT VIEW  ·  looking rearward along +X")
    add_hull(ax, torso_vertices, (1, 2))

    # Two forward rails overlap the anchor centers in this projection; the
    # orange bridge shows how the load is shared across both upper anchors.
    ax.plot(
        [ANCHOR_RIGHT[1], ANCHOR_LEFT[1]],
        [SENSOR_ORIGIN[2], SENSOR_ORIGIN[2]],
        color=MOUNT,
        linewidth=5.0,
        solid_capstyle="round",
        zorder=5,
    )
    for anchor, side, alignment in (
        (ANCHOR_LEFT, "LEFT ANCHOR", "right"),
        (ANCHOR_RIGHT, "RIGHT ANCHOR", "left"),
    ):
        ax.scatter(
            [anchor[1]],
            [anchor[2]],
            s=105,
            facecolor=MOUNT,
            edgecolor="#ffe2b8",
            linewidth=1.2,
            zorder=8,
        )
        offset = -0.010 if side.startswith("LEFT") else 0.010
        ax.annotate(
            side,
            (anchor[1], anchor[2]),
            xytext=(anchor[1] + offset, anchor[2] + 0.033),
            color=TEXT,
            fontsize=8.4,
            ha=alignment,
            arrowprops=dict(arrowstyle="-", color=MOUNT, linewidth=1.0),
            zorder=10,
        )

    add_sensor_polygon(ax, sensor_corners, (1, 2))
    add_origin(ax, (SENSOR_ORIGIN[1], SENSOR_ORIGIN[2]))
    ax.annotate(
        "MID-360 origin\ncentered at y = 0",
        (SENSOR_ORIGIN[1], SENSOR_ORIGIN[2]),
        xytext=(0.0, 0.335),
        ha="center",
        color=TEXT,
        fontsize=9,
        arrowprops=dict(arrowstyle="-|>", color=SENSOR, linewidth=1.2),
        zorder=10,
    )
    ax.axvline(0.0, color=MUTED, linestyle="--", linewidth=0.8, alpha=0.5)
    style_axes(
        ax,
        "torso +Y left  [m]",
        "torso +Z up  [m]",
        (-0.175, 0.175),
        (-0.025, 0.365),
    )
    # Positive +Y must appear on the viewer's left in a front view.
    ax.invert_xaxis()


def draw_side(
    ax: plt.Axes, torso_vertices: np.ndarray, sensor_corners: np.ndarray
) -> None:
    ax.set_title("LEFT-SIDE VIEW  ·  exact pitch and vertical FOV")
    add_hull(ax, torso_vertices, (0, 2))

    anchor_mid = (ANCHOR_LEFT + ANCHOR_RIGHT) / 2.0
    route = np.array(
        [
            anchor_mid,
            [BRACKET_RAIL_X, 0.0, SENSOR_ORIGIN[2]],
            SENSOR_ORIGIN,
        ]
    )
    plot_polyline(ax, route, (0, 2), width=5.0)
    ax.scatter(
        [anchor_mid[0]],
        [anchor_mid[2]],
        s=105,
        facecolor=MOUNT,
        edgecolor="#ffe2b8",
        linewidth=1.2,
        zorder=8,
    )
    add_sensor_polygon(ax, sensor_corners, (0, 2))
    add_origin(ax, (SENSOR_ORIGIN[0], SENSOR_ORIGIN[2]))

    # Official native -7...+52 degree vertical FOV, transformed through
    # the inverted 35.5-degree-down installation.
    elevations = np.linspace(FOV_NATIVE_DEG[0], FOV_NATIVE_DEG[1], 80)
    global_angles = -np.deg2rad(DOWN_PITCH_DEG + elevations)
    radius = 0.65
    wedge = np.column_stack(
        (
            SENSOR_ORIGIN[0] + radius * np.cos(global_angles),
            SENSOR_ORIGIN[2] + radius * np.sin(global_angles),
        )
    )
    fov_polygon = np.vstack(
        ([SENSOR_ORIGIN[0], SENSOR_ORIGIN[2]], wedge)
    )
    ax.add_patch(
        Polygon(
            fov_polygon,
            closed=True,
            facecolor=FOV,
            edgecolor=FOV,
            linewidth=1.1,
            alpha=0.15,
            zorder=2,
        )
    )
    for elevation in FOV_NATIVE_DEG:
        global_angle = -np.deg2rad(DOWN_PITCH_DEG + elevation)
        end = SENSOR_ORIGIN[[0, 2]] + radius * np.array(
            [np.cos(global_angle), np.sin(global_angle)]
        )
        ax.plot(
            [SENSOR_ORIGIN[0], end[0]],
            [SENSOR_ORIGIN[2], end[1]],
            color=FOV,
            linewidth=1.5,
            zorder=3,
        )

    # Torso forward reference and sensor +X direction.
    torso_forward_end = SENSOR_ORIGIN[[0, 2]] + np.array([0.27, 0.0])
    sensor_forward_angle = -np.deg2rad(DOWN_PITCH_DEG)
    sensor_forward_end = SENSOR_ORIGIN[[0, 2]] + 0.31 * np.array(
        [np.cos(sensor_forward_angle), np.sin(sensor_forward_angle)]
    )
    ax.add_patch(
        FancyArrowPatch(
            SENSOR_ORIGIN[[0, 2]],
            torso_forward_end,
            arrowstyle="-|>",
            mutation_scale=12,
            color=MUTED,
            linewidth=1.4,
            linestyle="--",
            zorder=7,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            SENSOR_ORIGIN[[0, 2]],
            sensor_forward_end,
            arrowstyle="-|>",
            mutation_scale=14,
            color=SENSOR,
            linewidth=2.4,
            zorder=8,
        )
    )
    arc_radius = 0.17
    ax.add_patch(
        Arc(
            SENSOR_ORIGIN[[0, 2]],
            2 * arc_radius,
            2 * arc_radius,
            theta1=-DOWN_PITCH_DEG,
            theta2=0.0,
            color=ANGLE,
            linewidth=2.0,
            zorder=8,
        )
    )
    half_angle = -np.deg2rad(DOWN_PITCH_DEG / 2.0)
    ax.text(
        SENSOR_ORIGIN[0] + 0.19 * np.cos(half_angle),
        SENSOR_ORIGIN[2] + 0.19 * np.sin(half_angle),
        "35.5° DOWN",
        color=ANGLE,
        fontsize=9.5,
        weight="bold",
        ha="left",
        va="center",
        zorder=10,
    )
    ax.text(
        SENSOR_ORIGIN[0] + 0.37,
        SENSOR_ORIGIN[2] - 0.50,
        "native vertical FOV\n−7° … +52°",
        color=FOV,
        fontsize=9,
        ha="center",
    )
    ax.text(
        torso_forward_end[0] - 0.01,
        torso_forward_end[1] + 0.018,
        "torso +X",
        color=MUTED,
        fontsize=8.5,
        ha="right",
    )
    ax.text(
        sensor_forward_end[0] + 0.012,
        sensor_forward_end[1],
        "MID-360 +X",
        color=SENSOR,
        fontsize=8.5,
        va="center",
    )

    # Dimension: sensor origin is 121 mm ahead of the anchor x-plane.
    dim_z = 0.365
    ax.plot(
        [ANCHOR_LEFT[0], ANCHOR_LEFT[0]],
        [ANCHOR_LEFT[2], dim_z + 0.01],
        color=MUTED,
        linewidth=0.8,
    )
    ax.plot(
        [SENSOR_ORIGIN[0], SENSOR_ORIGIN[0]],
        [SENSOR_ORIGIN[2], dim_z + 0.01],
        color=MUTED,
        linewidth=0.8,
    )
    ax.annotate(
        "",
        xy=(ANCHOR_LEFT[0], dim_z),
        xytext=(SENSOR_ORIGIN[0], dim_z),
        arrowprops=dict(arrowstyle="<->", color=TEXT, linewidth=1.1),
    )
    ax.text(
        0.5 * (ANCHOR_LEFT[0] + SENSOR_ORIGIN[0]),
        dim_z + 0.015,
        "121 mm",
        color=TEXT,
        ha="center",
        fontsize=9,
        weight="bold",
    )
    ax.annotate(
        "origin  (0.125, 0, 0.248) m",
        SENSOR_ORIGIN[[0, 2]],
        xytext=(0.20, 0.31),
        color=TEXT,
        fontsize=8.8,
        arrowprops=dict(arrowstyle="-|>", color=SENSOR, linewidth=1.1),
        zorder=10,
    )
    style_axes(
        ax,
        "torso +X forward  [m]",
        "torso +Z up  [m]",
        (-0.10, 0.82),
        (-0.40, 0.41),
    )


def draw_top(
    ax: plt.Axes, torso_vertices: np.ndarray, sensor_corners: np.ndarray
) -> None:
    ax.set_title("TOP VIEW  ·  horizontal coverage")
    # Plot horizontal +Y and vertical +X, so forward is up on the page.
    add_hull(ax, torso_vertices, (1, 0))

    for anchor in (ANCHOR_LEFT, ANCHOR_RIGHT):
        rail = np.array(
            [anchor, [BRACKET_RAIL_X, anchor[1], SENSOR_ORIGIN[2]]]
        )
        plot_polyline(ax, rail, (1, 0), width=4.0)
        ax.scatter(
            [anchor[1]],
            [anchor[0]],
            s=85,
            facecolor=MOUNT,
            edgecolor="#ffe2b8",
            linewidth=1.1,
            zorder=8,
        )
    bridge = np.array(
        [
            [BRACKET_RAIL_X, ANCHOR_RIGHT[1], SENSOR_ORIGIN[2]],
            [BRACKET_RAIL_X, ANCHOR_LEFT[1], SENSOR_ORIGIN[2]],
        ]
    )
    tongue = np.array(
        [
            [BRACKET_RAIL_X, 0.0, SENSOR_ORIGIN[2]],
            SENSOR_ORIGIN,
        ]
    )
    plot_polyline(ax, bridge, (1, 0), width=4.0)
    plot_polyline(ax, tongue, (1, 0), width=4.0)
    add_sensor_polygon(ax, sensor_corners, (1, 0))
    add_origin(ax, (SENSOR_ORIGIN[1], SENSOR_ORIGIN[0]))

    # MID-360 is 360 degrees horizontally. Show a small full ring plus the
    # central 99% ball-azimuth envelope observed in the trusted deployment data.
    ax.add_patch(
        Circle(
            (SENSOR_ORIGIN[1], SENSOR_ORIGIN[0]),
            0.43,
            fill=False,
            edgecolor=FOV,
            linewidth=1.1,
            linestyle=(0, (4, 3)),
            alpha=0.65,
            zorder=2,
        )
    )
    azimuths = np.linspace(AZIMUTH_99_DEG[0], AZIMUTH_99_DEG[1], 100)
    radius = 0.67
    azimuth_rad = np.deg2rad(azimuths)
    sector_edge = np.column_stack(
        (
            SENSOR_ORIGIN[1] + radius * np.sin(azimuth_rad),
            SENSOR_ORIGIN[0] + radius * np.cos(azimuth_rad),
        )
    )
    sector = np.vstack(
        ([SENSOR_ORIGIN[1], SENSOR_ORIGIN[0]], sector_edge)
    )
    ax.add_patch(
        Polygon(
            sector,
            closed=True,
            facecolor=FOV,
            edgecolor=FOV,
            linewidth=1.0,
            alpha=0.13,
            zorder=2,
        )
    )
    for azimuth in AZIMUTH_99_DEG:
        angle = np.deg2rad(azimuth)
        end = np.array(
            [
                SENSOR_ORIGIN[1] + radius * np.sin(angle),
                SENSOR_ORIGIN[0] + radius * np.cos(angle),
            ]
        )
        ax.plot(
            [SENSOR_ORIGIN[1], end[0]],
            [SENSOR_ORIGIN[0], end[1]],
            color=FOV,
            linewidth=1.2,
            zorder=3,
        )
    ax.text(
        0.0,
        0.61,
        "trusted ball azimuth (central 99%)\n−81.9° … +76.6°",
        color=FOV,
        fontsize=9,
        ha="center",
    )
    ax.text(
        -0.33,
        -0.22,
        "MID-360 horizontal FOV: 360°",
        color=MUTED,
        fontsize=8.8,
        rotation=31,
    )
    ax.annotate(
        "+X FORWARD",
        xy=(0.0, 0.78),
        xytext=(0.0, 0.48),
        ha="center",
        color=TEXT,
        fontsize=9,
        weight="bold",
        arrowprops=dict(arrowstyle="-|>", color=TEXT, linewidth=1.4),
    )
    style_axes(
        ax,
        "torso +Y left  [m]",
        "torso +X forward  [m]",
        (-0.78, 0.78),
        (-0.32, 0.84),
    )
    ax.invert_xaxis()


def write_mount_json(path: Path) -> None:
    data = {
        "frame": "torso_link",
        "coordinate_convention": {
            "x": "forward",
            "y": "left",
            "z": "up",
        },
        "upper_anchor_centers_m": {
            "left": ANCHOR_LEFT.tolist(),
            "right": ANCHOR_RIGHT.tolist(),
        },
        "proposed_mid360": {
            "origin_m": SENSOR_ORIGIN.tolist(),
            "orientation": "inverted",
            "forward_axis_down_pitch_deg": DOWN_PITCH_DEG,
            "yaw_deg": 0.0,
            "practical_pitch_range_deg": list(PRACTICAL_PITCH_RANGE_DEG),
            "schematic_housing_size_m": SENSOR_SIZE.tolist(),
        },
        "bracket": {
            "crossbar_x_m": BRACKET_RAIL_X,
            "origin_extension_from_anchor_x_plane_mm": float(
                1000.0 * (SENSOR_ORIGIN[0] - ANCHOR_LEFT[0])
            ),
            "origin_ahead_of_torso_shell_at_anchor_height_mm": 53.0,
        },
        "trusted_data": {
            "samples": 23075,
            "active_episodes": 16,
            "duration_s": 102.5195,
            "center_coverage_percent": CENTER_COVERAGE_PCT,
            "full_20cm_ball_coverage_percent": FULL_BALL_COVERAGE_PCT,
            "central_99_percent_range_m": list(RANGE_99_M),
            "central_99_percent_azimuth_deg": list(AZIMUTH_99_DEG),
        },
        "official_mid360_fov_deg": {
            "horizontal": 360.0,
            "vertical": list(FOV_NATIVE_DEG),
        },
        "engineering_note": (
            "The housing envelope is schematic. Verify optical-origin and "
            "fastener geometry against the official MID-360 CAD before "
            "manufacturing the bracket."
        ),
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    torso_vertices = load_binary_stl_vertices(TORSO_STL)
    sensor_corners = box_corners(
        SENSOR_ORIGIN, SENSOR_SIZE, sensor_rotation()
    )

    fig, axes = plt.subplots(1, 3, figsize=(18.2, 8.8))
    fig.subplots_adjust(
        left=0.045, right=0.985, bottom=0.145, top=0.82, wspace=0.21
    )
    draw_front(axes[0], torso_vertices, sensor_corners)
    draw_side(axes[1], torso_vertices, sensor_corners)
    draw_top(axes[2], torso_vertices, sensor_corners)

    fig.suptitle(
        "PROPOSED G1 UPPER-CHEST MID-360 INSTALLATION",
        x=0.045,
        y=0.955,
        ha="left",
        fontsize=21,
        color=TEXT,
        weight="bold",
    )
    fig.text(
        0.045,
        0.902,
        "torso_link coordinates  ·  optical origin (0.125, 0, 0.248) m"
        "  ·  inverted  ·  yaw 0°  ·  35.5° down",
        color=SENSOR,
        fontsize=12.5,
        weight="bold",
    )
    fig.text(
        0.045,
        0.865,
        "Mount at the height of the two upper-chest anchors, centered on the "
        "sagittal plane, and extend slightly ahead of the torso shell.",
        color=MUTED,
        fontsize=10.7,
    )
    fig.text(
        0.045,
        0.065,
        "Trusted deployment data: 23,075 samples / 16 active episodes / "
        "102.5 s  ·  coverage at this origin: "
        f"{CENTER_COVERAGE_PCT:.1f}% ball center, "
        f"{FULL_BALL_COVERAGE_PCT:.1f}% full 20 cm ball  ·  "
        "practical pitch tolerance: 33°–37° down",
        color=TEXT,
        fontsize=10.2,
        weight="bold",
    )
    fig.text(
        0.045,
        0.028,
        "Blue-gray outline: exact torso_link STL projection.  Cyan housing: "
        "65 × 65 × 60 mm schematic envelope; verify the official CAD origin "
        "and fastener interface before fabrication.",
        color=MUTED,
        fontsize=9.2,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "mid360_upper_chest_mount.png"
    pdf_path = OUTPUT_DIR / "mid360_upper_chest_mount.pdf"
    json_path = OUTPUT_DIR / "mid360_upper_chest_mount.json"
    fig.savefig(png_path, dpi=190)
    fig.savefig(pdf_path)
    plt.close(fig)
    write_mount_json(json_path)
    print(png_path)
    print(pdf_path)
    print(json_path)


if __name__ == "__main__":
    main()
