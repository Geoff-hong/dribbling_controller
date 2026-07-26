#!/usr/bin/env bash
# One-shot: kill every process the SoftTouch dribble MuJoCo sim2sim spawns.
# ros2 launch's child nodes do NOT die on Ctrl+C, so run this to clean up.
#   ./tools/ros2_dr_sweep/kill_sim.sh
for pat in \
  'ros2 launch motion_tracking_controller' \
  'mujoco_sim' \
  'robot_state_publisher' \
  'controller_manager/spawner' \
  'dr_robustness_sweep' \
  'fall_monitor.py' ; do
  pkill -9 -f "$pat" 2>/dev/null
done
sleep 1
# second pass by explicit PID in case any survived
pids=$(pgrep -f 'ros2 launch motion_tracking_controller|mujoco_sim|robot_state_publisher' | grep -v $$)
[[ -n "$pids" ]] && kill -9 $pids 2>/dev/null
sleep 1
if pgrep -f 'mujoco_sim|ros2 launch motion_tracking_controller|robot_state_publisher' >/dev/null; then
  echo "[kill_sim] WARNING: some processes survived:"
  pgrep -af 'mujoco_sim|ros2 launch motion_tracking_controller|robot_state_publisher' | grep -v pgrep
else
  echo "[kill_sim] all sim processes cleared."
fi
