# SoftTouch Dribbling Controller

This repository is a dribbling-oriented fork of the BeyondMimic
`motion_tracking_controller`.  The original controller runs whole-body motion
tracking policies with legged_control2.  This fork keeps that path and adds a
SoftTouch dribbling deployment path:

- a ROS 2 controller for the SoftTouch Stage-3 dribble policy,
- ball pose/twist inputs for MuJoCo, mocap, or future perception pipelines,
- route command generation matching the SoftTouch dribble task,
- a MuJoCo physics bridge for ball state publishing and motion-frame reset,
- launch files for dribble sim2sim and mocap-based sim2real tests.

The controller is still a whole-body controller.  The main difference from
single-motion tracking is that the policy also observes ball state and route
commands.

## Requirements

The tested setup is:

- Ubuntu 24.04
- ROS 2 Jazzy
- legged_control2 and its Unitree/MuJoCo packages
- a G1 whole-body tracking or SoftTouch policy exported to ONNX

If ROS 1 is installed on the same machine, use a clean shell for ROS 2:

```bash
env -u ROS_DISTRO -u ROS_ROOT -u ROS_PACKAGE_PATH bash --noprofile --norc
source /opt/ros/jazzy/setup.bash
```

## Installation

Install ROS 2 Jazzy first by following the official ROS 2 documentation.  Then
install legged_control2 following the recommended Debian source installation:

https://qiayuanl.github.io/legged_control2_doc/installation.html#debian-source-recommended

The following packages are expected to be available after that setup:

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-unitree-description \
  ros-jazzy-unitree-systems \
  ros-jazzy-mujoco-ros2-control \
  ros-jazzy-rviz2 \
  ros-dev-tools \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool
```

Create a workspace and clone this fork:

```bash
mkdir -p ~/softtouch_ros2_ws/src
cd ~/softtouch_ros2_ws/src
git clone https://github.com/qiayuanl/unitree_bringup.git
git clone https://github.com/Geoff-hong/dribbling_controller.git motion_tracking_controller
cd ~/softtouch_ros2_ws
```

Install dependencies and build:

```bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo --packages-up-to motion_tracking_controller
source install/setup.bash
```

## SoftTouch Policy Artifact

The dribble launch files default to:

```text
~/SoftTouch/checkpoints/g1_dribble_s3_human_iter35000/softtouch_dribble_deploy.onnx
```

If that ONNX file already exists, no export step is needed.  If you have the
SoftTouch checkpoint and Stage-2 decoder but not the deployment ONNX, export it
from the SoftTouch Python environment:

```bash
cd ~/softtouch_ros2_ws/src/motion_tracking_controller
python tools/onnx_export/export_softtouch_dribble_policy.py \
  --checkpoint ~/SoftTouch/checkpoints/g1_dribble_s3_human_iter35000/model_35000.pt \
  --artifact ~/SoftTouch/checkpoints/g1_dribble_s3_human_iter35000/stage2_decoder_dim8.pt \
  --output ~/SoftTouch/checkpoints/g1_dribble_s3_human_iter35000/softtouch_dribble_deploy.onnx
```

The exported ONNX packs the Stage-3 actor and frozen Stage-2 decoder into one
deployment graph.  It expects the SoftTouch dribble observation layout described
in `config/g1/softtouch_dribble_controllers.yaml`.

Newer policies use a different observation layout, selected with `--obs-layout`:

- `--obs-layout v1` (default): 82-D single-frame actor + 90-D decoder = 172-D.
- `--obs-layout v2` (e.g. `g1_dribble_s3_human_dr_iter80000`): adds a
  `ball_radius` term and a 10-frame flattened actor history, giving an
  83 x 10 = 830-D actor + 90-D decoder = 920-D observation.

The exporter writes `observation_names`, `observation_dims`,
`observation_history_lengths` and `actor_history_length` into the ONNX metadata;
both the C++ controller and the Python sim read these, so the same deployment
code path serves either layout (see *Observation layouts* below).

## Sim2Sim Usage

Run the default dribble MuJoCo sim2sim:

```bash
env -u ROS_DISTRO -u ROS_ROOT -u ROS_PACKAGE_PATH bash --noprofile --norc
source /opt/ros/jazzy/setup.bash
source ~/softtouch_ros2_ws/install/setup.bash

ros2 launch motion_tracking_controller softtouch_dribble_mujoco.launch.py
```

Useful overrides:

```bash
# Use a non-default exported policy
ros2 launch motion_tracking_controller softtouch_dribble_mujoco.launch.py \
  policy_path:=/absolute/path/to/softtouch_dribble_deploy.onnx

# Show route and ball markers in RViz
ros2 launch motion_tracking_controller softtouch_dribble_mujoco.launch.py \
  launch_rviz:=true

# Use direct MuJoCo base truth for source comparison
ros2 launch motion_tracking_controller softtouch_dribble_mujoco.launch.py \
  softtouch_base_state_source:=topic

# Fall back to the original legged_control2 action path
ros2 launch motion_tracking_controller softtouch_dribble_mujoco.launch.py \
  softtouch_action_command_mode:=rl_controller

# Change the route (different but reproducible route per seed)
ros2 launch motion_tracking_controller softtouch_dribble_mujoco.launch.py \
  softtouch_seed:=7

# Shorter route so the robot reaches the end sooner (~18 m ~= 10 s at vmax 2 m/s)
ros2 launch motion_tracking_controller softtouch_dribble_mujoco.launch.py \
  softtouch_route_length_m:=18

# Override the ball bridge angular damping (= 4*I; used by the DR sweep)
ros2 launch motion_tracking_controller softtouch_dribble_mujoco.launch.py \
  softtouch_ball_angular_damping:=0.006256
```

An empty value means "use the YAML default", so pass these only when overriding.

The default dribble sim2sim launch uses:

- `mjcf/g1_softtouch_dribble.xml`,
- `config/g1/softtouch_dribble_controllers.yaml`,
- `config/g1/softtouch_mujoco_reset_walkf_rf_frame0.txt`,
- `SoftTouchMujocoBallBridgePlugin` to publish `/softtouch/ball/pose` and
  `/softtouch/ball/twist`,
- `position_target` action mode, which sends absolute joint targets with PD
  gains from the deployment metadata.

### Observation layouts (v1 single-frame vs v2 history)

The C++ controller drives its observation schema from the ONNX metadata, so both
policy versions run without recompiling:

- v1 (e.g. `iter35000`, `iter60000`): 172-D single-frame observation
  (82-D actor + 90-D decoder state).
- v2 (e.g. `iter80000`): 920-D observation — a `ball_radius` term is added and
  the 11 actor terms are stacked into a 10-frame history (83 x 10 = 830), plus
  the 90-D decoder state.  `observation_history_lengths` from the metadata and
  the per-term history buffering (isaaclab flatten order, oldest→newest per term)
  are handled by `legged_rl_controllers::ObservationManager`; the controller only
  reads the metadata and calls `setHistoryLengths`.  It also fails fast at
  configure time if the assembled observation width does not match the ONNX
  `obs` input.

Export the v2 ONNX with `--obs-layout v2`, and launch with the reset state the
policy was trained on.  The `iter80000` v2 policy was trained with standby-pose
reset mixing, so it needs the standby reset — the default walk-frame reset makes
it fall immediately:

```bash
ros2 launch motion_tracking_controller softtouch_dribble_mujoco.launch.py \
  policy_path:=/abs/path/g1_dribble_s3_human_dr_iter80000/softtouch_dribble_deploy.onnx \
  mujoco_reset_state_file:=/abs/path/config/g1/softtouch_mujoco_reset_standby.txt
```

## Python Batch Benchmark (`benchmark` branch)

The standalone Python sim (pysim) and the sim2sim standard benchmark live in
the top-level `sim2sim_benchmark/` package on the `benchmark` branch, together
with `tests/` and `tools/calibration/`.  That branch carries this whole deploy
tree plus the eval package (the engine loads `mjcf/g1_softtouch_dribble.xml`
and the `config/g1/` reset states), and picks up deploy changes via one-way
`git merge main`:

```bash
git checkout benchmark
```

See `sim2sim_benchmark/README.md` on that branch for usage.


## C++ Deployment-Stack DR Robustness Sweep

Unlike the pysim path, this sweeps domain randomization through the **actual ROS 2
deployment stack** (the C++ `SoftTouchDribbleController` in `ros2_control` +
`mujoco_sim_ros2`), to check the deployment stack — not just the policy — stays up
under perturbation.  DR is baked into physical MJCF variants (no controller code
changes): each variant randomizes ball mass, ball radius, and foot/ball friction
over the training DR ranges, with the ball angular damping (`= 4*I`) recomputed to
match.

Tools (`tools/ros2_dr_sweep/`):

- `gen_dr_mjcf.py` — writes N MJCF variants + `manifest.csv` into
  `mjcf/dr_variants/` (git-ignored).  `--n` is the TOTAL count; variant `000` is
  the un-randomized nominal baseline, so `--n 1` = nominal only.
- `fall_monitor.py` — subscribes to `/softtouch/base/pose` and judges fell /
  stayed-up + survival time over a fixed dribble window.
- `dr_robustness_sweep.sh` — for each variant: launch the sim2sim (with that model
  + matching angular damping + a per-variant route seed), record for `DUR` seconds,
  kill, then write `eval_result/dr_robustness/summary.csv` (stayed-up / fell /
  no-data per variant).
- `kill_sim.sh` — kills every process the sim spawns (ros2 launch's child nodes do
  NOT die on Ctrl+C; run this to clean up).

Run it (from a clean shell, in your terminal — not headless-safe, a MuJoCo window
opens per variant):

```bash
POLICY=/abs/path/g1_dribble_s3_human_dr_iter80000/softtouch_dribble_deploy.onnx \
RESET=/abs/path/config/g1/softtouch_mujoco_reset_standby.txt \
N=16 DUR=10 ./tools/ros2_dr_sweep/dr_robustness_sweep.sh
```

Env knobs: `N` (total variants), `DUR` (dribble seconds recorded, counted from the
first base-pose message), `STARTUP` (boot wait), `ROUTE_LEN` (empty = long default
route; the episode is bounded by `DUR` via kill, so the robot stops early on a long
route rather than running to the route end), `SEED` (DR sampling seed), `OUT`.

Notes:

- DR ranges match the training DR (see the DR table in
  `sim2sim_benchmark/README.md` on the `benchmark` branch); `ball_radius` is
  the one term outside the trained envelope, and since the `ball_radius`
  observation is held at nominal it also injects a small obs-error probe.
- This is a sequential robustness probe (stayed-up / fell), not a statistical
  fall-rate; use `python -m sim2sim_benchmark` (benchmark branch) for
  statistics.
- If a run is interrupted and leaves stray windows, run
  `./tools/ros2_dr_sweep/kill_sim.sh`.

## Real Robot / Mocap Usage

Real robot experiments are dangerous and are entirely at your own risk.  Start
with low-risk mocap wiring tests and verify all topics before activating the
policy on hardware.

The SoftTouch controller expects ball state on:

```text
/softtouch/ball/pose   geometry_msgs/PoseStamped, ball center xyz in world frame
/softtouch/ball/twist  geometry_msgs/TwistStamped, ball center linear velocity in world frame
```

If mocap publishes a `PoseStamped`, bridge it into the SoftTouch topic contract:

```bash
source /opt/ros/jazzy/setup.bash
source ~/softtouch_ros2_ws/install/setup.bash

ros2 run motion_tracking_controller bridge_softtouch_mocap_ball.py \
  --input-pose /mocap/ball/pose \
  --output-pose /softtouch/ball/pose \
  --output-twist /softtouch/ball/twist
```

Then launch the dribble real stack:

```bash
ros2 launch motion_tracking_controller softtouch_dribble_real.launch.py \
  network_interface:=<robot_network_interface>
```

Optional:

```bash
ros2 launch motion_tracking_controller softtouch_dribble_real.launch.py \
  network_interface:=<robot_network_interface> \
  policy_path:=/absolute/path/to/softtouch_dribble_deploy.onnx \
  enable_rosbag:=true
```

## Single Motion Tracking

The original BeyondMimic motion tracking path is still available:

```bash
ros2 launch motion_tracking_controller mujoco.launch.py policy_path:=/absolute/path/to/policy.onnx
ros2 launch motion_tracking_controller real.launch.py network_interface:=<robot_network_interface> policy_path:=/absolute/path/to/policy.onnx
```

The fork adds optional motion controls useful for expert policies:

- `motion_length`
- `motion_loop`
- `motion_time_step_stride`
- `motion_action_command_mode`

Leaving `motion_action_command_mode` empty uses the original
`RlController::update()` path.

## Code Structure

- `MotionTrackingController.*`
  Original whole-body tracking controller, extended with optional reset,
  motion timing, and explicit position/effort action modes.

- `MotionOnnxPolicy.*`
  ONNX wrapper for motion tracking policies.  It now supports motion length,
  looping, and time-step stride.

- `SoftTouchDribbleController.*`
  ROS 2 lifecycle controller for the SoftTouch dribble policy.

- `SoftTouchDribbleOnnxPolicy.*`
  Deployment ONNX wrapper.  It maintains latent/raw/decoded action history and
  converts raw policy output into absolute joint targets.

- `SoftTouchDribbleCommand.*`
  Ball state, optional base-state topic input, route command generation, and
  reset-time fallback state.

- `SoftTouchDribbleObservation.*`
  Observation terms matching the SoftTouch dribble task.

- `SoftTouchMujocoBallBridgePlugin.*`
  MuJoCo physics plugin for ball/base state publishing, ball damping, and
  reset-state application.

- `tools/ros2_dr_sweep/`
  C++ deployment-stack DR robustness sweep (`gen_dr_mjcf.py`,
  `dr_robustness_sweep.sh`, `fall_monitor.py`, `kill_sim.sh`): generate DR MJCF
  variants, run each through the ROS 2 sim2sim, judge stayed-up/fell, and clean
  up spawned processes (see *C++ Deployment-Stack DR Robustness Sweep*).

- `tools/onnx_export/export_softtouch_dribble_policy.py`
  Exports the SoftTouch Stage-3 actor plus Stage-2 decoder into one deployment
  ONNX.

- `tools/reset_export/`
  Reset-state file generators: `export_softtouch_mujoco_reset_state.py` (from
  SoftTouch motion clips) and `export_standby_reset_state.py` (from the
  StandbyController default pose).

- `tools/ros_utils/`
  Helpers run alongside experiments: `bridge_softtouch_mocap_ball.py` (mocap
  ball pose -> SoftTouch topic contract), `base_tf_broadcaster.py`
  (world->pelvis TF for RViz), `record_mujoco.sh` (x11grab screen capture).

## Notes

- `joint order mapping` is intentional.  Policy metadata order, ROS 2 control
  command order, and MuJoCo actuator order should not be assumed to match.
- `position_target` is not a torque policy.  It sends absolute joint targets,
  stiffness, damping, and zero effort.
- `effort_pd` is available for comparison, but it is not the default real robot
  path.
- MuJoCo sim2sim ball state is simulator truth.  For sim2real, replace it with
  mocap or perception estimates that obey the same topic contract.
