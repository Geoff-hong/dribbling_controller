#!/usr/bin/env bash
set -euo pipefail

LOG_BASE="${SOFTTOUCH_LOG_BASE:-/home/chuye/softtouch_logs}"
BAG_BASE="${SOFTTOUCH_BAG_BASE:-/home/chuye/softtouch_bags}"
RUN_DIR="${SOFTTOUCH_RUN_DIR:-}"
RUN_DIR_FILE="${LOG_BASE}/latest_run_dir"
NETWORK_INTERFACE="${SOFTTOUCH_NETWORK_INTERFACE:-enp7s0}"
POLICY_PATH="${SOFTTOUCH_POLICY_PATH:-/home/chuye/SoftTouch/checkpoints/g1_dribble_s3_net512_iter20000/softtouch_dribble_deploy_m19999.onnx}"
MOCAP_REFERENCE_POSE_TOPIC="${SOFTTOUCH_MOCAP_REFERENCE_POSE_TOPIC:-/softtouch/mocap/chest/pose}"
MOCAP_REFERENCE_TWIST_TOPIC="${SOFTTOUCH_MOCAP_REFERENCE_TWIST_TOPIC:-/softtouch/mocap/chest/twist}"
MOCAP_REFERENCE_CHILD_FRAME="${SOFTTOUCH_MOCAP_REFERENCE_CHILD_FRAME:-softtouch_mocap_chest}"
MOCAP_REFERENCE_OFFSET_BODY="${SOFTTOUCH_MOCAP_REFERENCE_OFFSET_BODY:-0 0 0}"
BALL_RADIUS_M="${SOFTTOUCH_BALL_RADIUS_M:-0.10}"
OBS_DUMP_PATH="${SOFTTOUCH_OBS_DUMP_PATH:-}"
ROS1_MASTER_URI="${ROS_MASTER_URI:-http://192.168.1.135:11311}"
ROS1_IP="${ROS_IP:-192.168.1.135}"

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  bridge       Start ROS1->UDP senders, ROS2 mocap receivers, mocap markers, and torso-reference TF.
  controller   Start real robot controller launch in standby. Does not start rosbag.
  bag          Start a SoftTouch whitelist rosbag.
  rviz         Start SoftTouch real RViz.
  prep         Start bridge, whitelist rosbag, and RViz. Does not start the controller.
  all          Start bridge, controller, bag, and RViz.
  status       Show latest run dir and matching processes.
  stop         Stop helper processes: bridges, mocap viz, RViz, and SoftTouch whitelist bag.
  stop-all     Stop helpers and the real controller launch/control nodes.
EOF
}

ensure_run_dir() {
  mkdir -p "${LOG_BASE}" "${BAG_BASE}"
  if [[ -z "${RUN_DIR}" ]]; then
    if [[ -f "${RUN_DIR_FILE}" ]]; then
      RUN_DIR="$(cat "${RUN_DIR_FILE}")"
    else
      RUN_DIR="${LOG_BASE}/run_$(date +%Y%m%d_%H%M%S)"
      mkdir -p "${RUN_DIR}"
      echo "${RUN_DIR}" > "${RUN_DIR_FILE}"
    fi
  fi
  mkdir -p "${RUN_DIR}"
  echo "${RUN_DIR}" > "${RUN_DIR_FILE}"
}

new_run_dir() {
  mkdir -p "${LOG_BASE}" "${BAG_BASE}"
  RUN_DIR="${LOG_BASE}/run_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${RUN_DIR}"
  echo "${RUN_DIR}" > "${RUN_DIR_FILE}"
}

start_bg() {
  local name="$1"
  local cmd="$2"
  ensure_run_dir
  local wrapper="${RUN_DIR}/${name}.sh"
  cat > "${wrapper}" <<EOF
#!/usr/bin/env bash
set -eo pipefail
${cmd}
EOF
  chmod +x "${wrapper}"
  echo "[softtouch] start ${name}; log=${RUN_DIR}/${name}.log"
  setsid bash "${wrapper}" > "${RUN_DIR}/${name}.log" 2>&1 < /dev/null &
  echo "$!" > "${RUN_DIR}/${name}.pid"
}

ros1_prefix() {
  cat <<EOF
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/noetic/setup.bash
source /home/chuye/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=${ROS1_MASTER_URI}
export ROS_IP=${ROS1_IP}
EOF
}

ros2_prefix() {
  cat <<EOF
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_PACKAGE_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/jazzy/setup.bash
source /home/chuye/softtouch_ros2_ws/install/local_setup.bash
EOF
}

stop_helpers() {
  python3 - <<'PY'
import os
import signal
import subprocess
import time

patterns = (
    "ros1_mocap_udp_sender.py",
    "ros1_unlabeled_ball_udp_sender.py",
    "ros2_mocap_udp_receiver.py",
    "ros2_mocap_pose_viz.py",
    "rviz2 -d",
    "softtouch_real_dribble.rviz",
    "ros2 bag record -s mcap -o /home/chuye/softtouch_bags/softtouch_real_",
)
self_pid = os.getpid()
parent_pid = os.getppid()
rows = subprocess.run(["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=True).stdout.splitlines()
pids = []
for row in rows:
    row = row.strip()
    if not row:
        continue
    pid_text, _, args = row.partition(" ")
    try:
        pid = int(pid_text)
    except ValueError:
        continue
    if pid in (self_pid, parent_pid):
        continue
    if any(pattern in args for pattern in patterns):
        pids.append(pid)

for pid in pids:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
time.sleep(0.5)
for pid in pids:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        continue
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
PY
}

stop_all() {
  stop_helpers
  pkill -f 'softtouch_dribble_real.launch.py' 2>/dev/null || true
  pkill -f 'ros2_control_node' 2>/dev/null || true
  pkill -f 'robot_state_publisher' 2>/dev/null || true
  pkill -f 'joy_teleop' 2>/dev/null || true
  pkill -f 'joy_linux_node' 2>/dev/null || true
}

start_bridge() {
  start_bg ros1_torso_sender "$(ros1_prefix)
/home/chuye/motion_tracking_controller/tools/ros1_mocap_udp_sender.py --topic /mocap_output1 --vehicle-id 1 --host 127.0.0.1 --port 15551"

  start_bg ros1_ball_sender "$(ros1_prefix)
/home/chuye/motion_tracking_controller/tools/ros1_unlabeled_ball_udp_sender.py --topic /mocap_unlabeled_marker --host 127.0.0.1 --port 15552 --expected-z 0.10 --min-z 0.04 --max-z 0.35 --debug-topic /softtouch/mocap/ball/point_ros1 --max-predict-time 0.20 --max-hold-time 1.00"

  start_bg ros2_torso_receiver "$(ros2_prefix)
ros2 run motion_tracking_controller ros2_mocap_udp_receiver.py --port 15551 --pose-topic ${MOCAP_REFERENCE_POSE_TOPIC} --twist-topic ${MOCAP_REFERENCE_TWIST_TOPIC} --frame-id world --position-offset-body ${MOCAP_REFERENCE_OFFSET_BODY} --max-position-step-m 0.12 --max-orientation-step-deg 35.0 --jump-check-max-dt 0.05"

  start_bg ros2_ball_receiver "$(ros2_prefix)
ros2 run motion_tracking_controller ros2_mocap_udp_receiver.py --port 15552 --pose-topic /softtouch/mocap/ball/pose --twist-topic /softtouch/mocap/ball/twist --frame-id world --max-position-step-m 0.25 --jump-check-max-dt 0.05"

  start_bg chest_viz_tf "$(ros2_prefix)
ros2 run motion_tracking_controller ros2_mocap_pose_viz.py --ros-args -r __node:=softtouch_mocap_reference_viz -p pose_topic:=${MOCAP_REFERENCE_POSE_TOPIC} -p child_frame_id:=${MOCAP_REFERENCE_CHILD_FRAME} -p marker_namespace:=softtouch_mocap_reference -p marker_topic:=/softtouch/mocap/chest/markers -p path_topic:=/softtouch/mocap/chest/path -p show_arrow:=true -p sphere_diameter:=0.08"

  start_bg ball_viz "$(ros2_prefix)
ros2 run motion_tracking_controller ros2_mocap_pose_viz.py --ros-args -r __node:=softtouch_mocap_ball_viz -p pose_topic:=/softtouch/mocap/ball/pose -p child_frame_id:=softtouch_mocap_ball -p marker_namespace:=softtouch_mocap_ball -p marker_topic:=/softtouch/mocap/ball/markers -p path_topic:=/softtouch/mocap/ball/path -p show_arrow:=false -p sphere_diameter:=0.22 -p sphere_color_r:=1.0 -p sphere_color_g:=0.1 -p sphere_color_b:=0.1 -p sphere_color_a:=0.95"
}

start_controller() {
  ensure_run_dir
  local obs_dump_path="${OBS_DUMP_PATH:-${RUN_DIR}/policy_obs_dump.txt}"
  start_bg controller "$(ros2_prefix)
ros2 launch motion_tracking_controller softtouch_dribble_real.launch.py network_interface:=${NETWORK_INTERFACE} policy_path:=${POLICY_PATH} enable_rosbag:=false softtouch_ball_radius_m:=${BALL_RADIUS_M} softtouch_obs_frame_source:=topic softtouch_obs_frame_pose_topic:=${MOCAP_REFERENCE_POSE_TOPIC} softtouch_obs_dump_path:=${obs_dump_path}"
}

start_bag() {
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  start_bg bag "$(ros2_prefix)
mkdir -p ${BAG_BASE}
ros2 bag record -s mcap -o ${BAG_BASE}/softtouch_real_${stamp} --topics \
  ${MOCAP_REFERENCE_POSE_TOPIC} \
  ${MOCAP_REFERENCE_TWIST_TOPIC} \
  /softtouch/mocap/ball/pose \
  /softtouch/mocap/ball/twist \
  /softtouch/policy/observation \
  /softtouch/policy/observation_schema \
  /softtouch/policy/raw_action \
  /softtouch/policy/latent_action \
  /softtouch/policy/joint_target \
  /softtouch/dribble/markers \
  /softtouch/mocap/chest/markers \
  /softtouch/mocap/ball/markers \
  /controller_manager/activity \
  /robot_description \
  /joint_states \
  /joy \
  /tf \
  /tf_static \
  2> >(grep -v \"Topic '/lf/sportmodestate'\" >&2)"
}

start_rviz() {
  start_bg rviz "$(ros2_prefix)
rviz2 -d /home/chuye/softtouch_ros2_ws/install/motion_tracking_controller/share/motion_tracking_controller/rviz/softtouch_real_dribble.rviz"
}

status() {
  echo "[softtouch] latest_run_dir=$(cat "${RUN_DIR_FILE}" 2>/dev/null || true)"
  pgrep -af 'ros1_mocap_udp_sender.py|ros1_unlabeled_ball_udp_sender.py|ros2_mocap_udp_receiver.py|ros2_mocap_pose_viz.py|softtouch_dribble_real.launch.py|ros2_control_node|ros2 bag record .*softtouch_real_|rviz2 .*softtouch_real_dribble.rviz|joy_teleop|joy_linux_node|robot_state_publisher' || true
}

cmd="${1:-}"
case "${cmd}" in
  bridge)
    ensure_run_dir
    start_bridge
    ;;
  controller)
    ensure_run_dir
    start_controller
    ;;
  bag)
    ensure_run_dir
    start_bag
    ;;
  rviz)
    ensure_run_dir
    start_rviz
    ;;
  prep)
    new_run_dir
    stop_helpers
    start_bridge
    start_bag
    start_rviz
    ;;
  all)
    new_run_dir
    stop_helpers
    start_bridge
    start_controller
    start_bag
    start_rviz
    ;;
  status)
    status
    ;;
  stop)
    stop_helpers
    ;;
  stop-all)
    stop_all
    ;;
  *)
    usage
    exit 2
    ;;
esac
