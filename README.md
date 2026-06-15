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
python tools/export_softtouch_dribble_policy.py \
  --checkpoint ~/SoftTouch/checkpoints/g1_dribble_s3_human_iter35000/model_35000.pt \
  --artifact ~/SoftTouch/checkpoints/g1_dribble_s3_human_iter35000/stage2_decoder_dim8.pt \
  --output ~/SoftTouch/checkpoints/g1_dribble_s3_human_iter35000/softtouch_dribble_deploy.onnx
```

The exported ONNX packs the Stage-3 actor and frozen Stage-2 decoder into one
deployment graph.  It expects the SoftTouch dribble observation layout described
in `config/g1/softtouch_dribble_controllers.yaml`.

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
```

The default dribble sim2sim launch uses:

- `mjcf/g1_softtouch_dribble.xml`,
- `config/g1/softtouch_dribble_controllers.yaml`,
- `config/g1/softtouch_mujoco_reset_walkf_rf_frame0.txt`,
- `SoftTouchMujocoBallBridgePlugin` to publish `/softtouch/ball/pose` and
  `/softtouch/ball/twist`,
- `position_target` action mode, which sends absolute joint targets with PD
  gains from the deployment metadata.

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

- `tools/bridge_softtouch_mocap_ball.py`
  Converts mocap ball pose into SoftTouch ball pose/twist topics.

- `tools/export_softtouch_dribble_policy.py`
  Exports the SoftTouch Stage-3 actor plus Stage-2 decoder into one deployment
  ONNX.

- `tools/export_softtouch_mujoco_reset_state.py`
  Generates reset-state text files from SoftTouch motion clips.

## Notes

- `joint order mapping` is intentional.  Policy metadata order, ROS 2 control
  command order, and MuJoCo actuator order should not be assumed to match.
- `position_target` is not a torque policy.  It sends absolute joint targets,
  stiffness, damping, and zero effort.
- `effort_pd` is available for comparison, but it is not the default real robot
  path.
- MuJoCo sim2sim ball state is simulator truth.  For sim2real, replace it with
  mocap or perception estimates that obey the same topic contract.
