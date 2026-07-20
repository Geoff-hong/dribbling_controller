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

## Standalone Python Sim (pysim): batch eval, DR, latency

For fast policy evaluation there is a standalone MuJoCo sim that loads the
exported ONNX directly with `onnxruntime` — no ROS 2, no ros2_control.  It ports
the deployment obs assembly, route generator and PD controller from the C++
(`SoftTouchDribbleCommon.cpp` / `SoftTouchDribbleObservation.cpp`) so the Python
rollout matches the C++ one, and it reads the same ONNX metadata, so one code
path serves both the v1 (172-D) and v2 history (920-D) policies.

This is the *policy-transfer* flavor of sim2sim (an Isaac/PhysX-trained policy
run in a different engine, MuJoCo).  It does **not** exercise the ROS 2
deployment stack (controller, StateEstimator, real-time loop) — for that, use the
ros2_control sim2sim above.  Run it from the SoftTouch Python env (torch / onnx /
mujoco), not the ROS 2 env.

The simulation core lives in `sim2sim_benchmark/engine.py` (Route / Robot /
multi-robot model composition); `python -m sim2sim_benchmark.pysim` is the
interactive CLI on top of it: live viewer, `--record` mp4, `--headless` smoke
test, `--eval` random-DR Monte Carlo and the `--sweep` single-param DR
diagnostic.

Common `sim2sim_benchmark.pysim` flags:

```text
--onnx PATH        policy ONNX to evaluate
--reset PATH       reset-state file (must match how the policy was trained;
                   standby-trained policies need the standby reset)
--robots N         number of robots simulated in parallel
--eval             batch random-DR eval (no window), prints stats + CSV
--sweep            systematic 1-param-at-a-time DR sweep on fixed routes
--latency          replicate the v2 training-time latency DR (see below)
--seconds S        wall-clock eval duration; --episode-s = per-episode length
--out-dir DIR      folder for all outputs; --csv / --plot / --record name them
--record FILE.mp4  offscreen N-up demo video (no window)
--headless         step without a viewer
```

Single-run knobs (also usable with the viewer for eyeballing): `--cmd-mode 0`
(straight route), `--route-kappa K` (constant-curvature arc, signed; speed
follows the trained law `min(vmax, sqrt(0.75/|kappa|))`), `--arc-angle-deg MIN
MAX` (one finite turn instead of endless circles), `--route-vmax`, `--push-dv` /
`--ball-push-dv` / `--push-interval-s` (velocity kicks), `--ball-delay-steps` /
`--act-delay-ms` (pin latency), `--offroute-fail-m` / `--ball-far-fail-m`
(fail-fast criteria), `--jitter` (reset noise).

Domain randomization (`--eval`, matches the training DR):

```text
ball_mass      [0.352, 0.430] kg   (0.391 x [0.9, 1.1])
ball_friction  [0.475, 0.525]
foot_friction  [0.50, 1.00]        (body/foot dynamic friction)
ball_radius    [0.09, 0.11] m      (NOT randomized in training; +/-10% band)
```

`--sweep` uses 1.5x the training range (centered), probing just past the trained
envelope.  `--latency` (v2 policies) adds, per episode: a ball-observation lag of
1-3 policy steps, and an action lag of 0-4 sim sub-steps (0-20 ms at dt=0.005),
with 30% of episodes forced to zero action lag.

## Sim2Sim Standard Benchmark (`sim2sim_benchmark/`)

The standard batch evaluation lives in the top-level `sim2sim_benchmark/`
package (the pysim engine above is only the simulator it drives).  It has two
separate tests, each a fixed condition table run to completion (queue-based, no
truncation bias) with one CSV row per episode:

- **Robustness** — perturb the environment, keep the nominal command (human
  routes, fixed route bank).  Axes: `dr_scale` (all DR params jointly, centered
  training ranges x alpha), `base_push` / `ball_push` (velocity kicks every 5 s,
  random direction/phase), `obs_latency` (ball-obs lag, steps), `act_latency`
  (action lag, ms).  Metrics: survival rate, ball possession (ball >1.5 m from
  the pelvis for >2 s = lost), achieved/commanded speed ratio, cross-track.
- **Capability** — clean nominal env (+ small reset jitter), extreme commands,
  fail-fast control criteria: the episode FAILS the moment the ball is >0.8 m
  off the route or >1.2 m from the robot; 10 s budget; metric = SUCCESS RATE.
  `straight_speed` sweeps the commanded speed on a straight route (success =
  kept control for the whole 10 s).  `corner_turn` is the turn-into-corner
  test: a random straight lead-in (1.5-4 m), ONE arc of 150-180 deg (random) at
  constant kappa, then a straight exit, both turn directions, speed following
  the trained law `min(2, sqrt(0.75/|kappa|))`; success additionally requires
  finishing the turn; 12 s budget.  kappa < 0.4 is not swept (the arc alone
  cannot finish in time at the trained speed law).  `human_dribble` runs the
  nominal task itself as a route test: human-dribble routes with the turn
  aggressiveness swept via `route_human_kappa_cap` (0.3-1.1; old policies
  trained at 0.5, the new command generator uses 1.0), 20 s fail-fast episodes,
  success = kept control the whole episode; drawn as the second row of the
  route figure.  `u_turn` is the about-face drill
  matching the training u_turn mode (run-in 1.5-4 m, ONE 160-200 deg turn,
  kappa swept 1.5-4.0 = turn radius down to 0.25 m, both directions; same
  fail-fast/success semantics, its own figure).  `speed_tracking` measures speed
  CONTROLLABILITY on nominal human-dribble routes with the TRAINING command
  distribution: the cruise pace is sampled per episode from U(1.2, 2.0) m/s
  (matching the training-side `ROUTE_CRUISE_RANGE`) and route curvature
  modulates it further, over route-bank x reps episodes.
  Per-step (commanded, actual) speed pairs are recorded — actual = ball
  velocity projected on the commanded direction, smoothed over 0.5 s — the
  per-episode Pearson r goes into the CSV (`speed_corr_r`), the 10 Hz pairs
  into `capability_speed_pairs.csv`, and the first 8 episodes dump full-rate
  traces into `capability_speed_traces.csv`; no fail-fast, 20 s episodes.

All conditions pin latency to the deployment nominal (ball lag 2 steps, action
lag 10 ms) unless the axis varies it.  Episodes per condition = `--route-bank`
(12) x `--reps` (4).  Every per-episode random draw — route geometry, cruise
pace, corner lead/angle, DR sampling, reset jitter, push phases — is a pure
function of (benchmark seed, condition, rep), so independently-run experiments
compare on IDENTICAL paired episodes; pick any set of run dirs to merge at
plot time.  A custom table can be run with `--conditions table.json`.

Benchmark outputs live under `sim2sim_eval_results/`: `runs/<node>/` holds one
checkpoint's eval artifacts (CSVs, logs, `videos/<test>/`), `compare/` holds
cross-experiment artifacts (the HTML report, PNG figures).

```bash
# from the repo root, SoftTouch python env
$PY -m sim2sim_benchmark --robustness --capability \
    --onnx "$ONNX" --reset "$RESET" --robots 32 --out-dir sim2sim_eval_results/runs/m80000
# -> runs/m80000/robustness.csv + capability.csv (+ console summaries)
# add --videos to also record one mp4 per condition (rep-0 route, chase camera)
# under runs/m80000/videos/<test>/ (offscreen; needs MUJOCO_GL=egl-capable box)
# add --shard i/n to split a table across n parallel processes (same out-dir;
# per-episode seeding keeps FULL-table condition indices, so the union of shards
# is the same paired episode set; merge: one header + concatenated shard rows)
# each finished episode is flushed to the CSV immediately, and the CSV is the
# progress record: re-running with the same out-dir (and shard layout) RESUMES,
# skipping episodes already recorded — a killed run loses nothing; --fresh
# ignores the existing CSVs and starts over

# interactive single-file HTML report (tensorboard-style): experiment checkboxes,
# robustness/capability panels, control traces, per-condition video index
$PY -m sim2sim_benchmark.html_report \
    --run-dirs sim2sim_eval_results/runs/m80000 sim2sim_eval_results/runs/m90000 \
    --labels iter80000 iter90000 --out sim2sim_eval_results/compare/report.html

# static comparison figures (PNG): one color per experiment
$PY -m sim2sim_benchmark.plot --run-dirs sim2sim_eval_results/runs/m80000 \
    sim2sim_eval_results/runs/m90000 \
    --labels iter80000 iter90000 --out-dir sim2sim_eval_results/compare
# -> robustness_compare.png                     (perturbation axes)
#    speed_compare.png                          (max speed + controllability, pooled r)
#    route_compare.png                          (corner turn)
#    uturn_compare.png                          (u-turn about-face drill)
#    speed_traces_<label>.png per experiment    (ball speed vs cmd target traces)
```

Preview what the figures look like (mock data, real plotting code):
`sim2sim_benchmark/demo/`.

Example — random-DR eval + DR sweep + demo video for the `iter80000` v2 policy
(the defaults already point at `checkpoints/g1_dribble_s3_human_dr_iter80000` +
the standby reset, so `--onnx/--reset` are only needed for other policies):

```bash
# from the repo root, SoftTouch python env (e.g. conda multiagentsim), NOT the ROS 2 env
PY=~/miniconda3/envs/multiagentsim/bin/python
OUT=eval_result/m80000

$PY -m sim2sim_benchmark.pysim --eval  --latency \
    --robots 32 --seconds 300 --out-dir "$OUT" --csv eval.csv --plot eval_plot.png
$PY -m sim2sim_benchmark.pysim --sweep --latency \
    --robots 32 --out-dir "$OUT" --csv sweep.csv --plot sweep.png
MUJOCO_GL=glx $PY -m sim2sim_benchmark.pysim --latency \
    --robots 4 --seconds 35 --out-dir "$OUT" --record dribble_4up.mp4
```

Outputs land under `eval_result/<run>/` (per-episode CSV, DR-sweep plots, demo
mp4).  That folder is git-ignored.

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

- DR ranges match the training DR (see the pysim table above); `ball_radius` is
  the one term outside the trained envelope, and since the `ball_radius`
  observation is held at nominal it also injects a small obs-error probe.
- This is a sequential robustness probe (stayed-up / fell), not a statistical
  fall-rate; use `python -m sim2sim_benchmark` for statistics.
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

- `sim2sim_benchmark/`
  The sim2sim standard benchmark package: simulation engine (`engine.py` —
  Route/Robot/multi-robot model composition), condition tables
  (`conditions.py`), queue runner (`runner.py`), CSV/console report
  (`report.py`), comparison figures (`plot.py`), and the interactive/legacy CLI
  (`pysim.py` — viewer, `--record`, `--headless`, `--eval`, `--sweep`).  See
  *Sim2Sim Standard Benchmark* and *Standalone Python Sim*.

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

- `tools/calibration/`
  One-off physics-calibration experiments whose conclusions are baked into the
  MJCF: `ball_roll_test.py` (ball angular damping vs the PhysX roll/decay
  reference) and `friction_slip_test.py` (contact-solver impratio/cone vs
  stance creep).

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
