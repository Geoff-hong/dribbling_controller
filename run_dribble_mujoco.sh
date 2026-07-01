#!/usr/bin/env bash
# One-shot: launch the SoftTouch dribble MuJoCo sim AND screen-record just the
# MuJoCo window to mp4. Ctrl+C stops everything and finalizes the video.
#
#   ./run_dribble_mujoco.sh
#   RECORD=0 ./run_dribble_mujoco.sh                 # run without recording
#   OUT_VIDEO=~/clip.mp4 FPS=60 ./run_dribble_mujoco.sh
#   POLICY=/path/to.onnx ./run_dribble_mujoco.sh
# NOTE: no `set -u` — ROS setup.bash references unbound vars (AMENT_TRACE_SETUP_FILES).
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

POLICY="${POLICY:-/home/aldebaran/Desktop/SoftTouch-multiagent/logs/rsl_rl/g1_dribble/2026-06-17_16-44-29/softtouch_dribble_deploy_iter50000.onnx}"
WS="${WS:-$HOME/softtouch_ros2_ws}"
RECORD="${RECORD:-1}"
FPS="${FPS:-30}"
DURATION="${DURATION:-}"   # seconds; empty = record until Ctrl+C
OUT_VIDEO="${OUT_VIDEO:-$SCRIPT_DIR/dribble_$(date +%Y%m%d_%H%M%S).mp4}"
DISP="${DISPLAY:-:0}"

# conda's python shadows ROS python (rclpy import fails) -> drop it
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
  conda deactivate 2>/dev/null || true
  conda deactivate 2>/dev/null || true
fi

source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"

[[ -f "$POLICY" ]] || { echo "[run] ERROR: policy not found: $POLICY"; exit 1; }

pids=()
FFMPEG_PID=""
cleanup() {
  echo; echo "[run] stopping..."
  # finalize the mp4 first: SIGINT ffmpeg and give it a moment to flush the trailer
  if [[ -n "$FFMPEG_PID" ]] && kill -0 "$FFMPEG_PID" 2>/dev/null; then
    kill -INT "$FFMPEG_PID" 2>/dev/null
    for _ in $(seq 1 25); do kill -0 "$FFMPEG_PID" 2>/dev/null || break; sleep 0.2; done
    kill "$FFMPEG_PID" 2>/dev/null
  fi
  for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null; done
  # last of the last: tell the user where the video landed
  if [[ "$RECORD" == "1" ]]; then
    if [[ -f "$OUT_VIDEO" ]]; then
      echo "==================================================="
      echo "[run] video saved: $OUT_VIDEO  ($(du -h "$OUT_VIDEO" 2>/dev/null | cut -f1))"
      echo "==================================================="
    else
      echo "[run] no video file was written ($OUT_VIDEO)."
    fi
  fi
}
trap cleanup EXIT INT TERM

# diagnostics after the stack comes up (CM node is 'mujoco_sim_ros2_node', so we
# diagnose via topics + timeout instead of `ros2 control list_controllers`).
( sleep 16
  echo "================= DIAG ================="
  echo "  -- /softtouch/dribble/markers (want 'average rate ~16-20Hz' = route OK) --"
  timeout 5 ros2 topic hz /softtouch/dribble/markers 2>/dev/null
  echo "  (any 'average rate' line above = the route IS publishing)"
  echo "========================================" ) &
pids+=($!)

# Route is drawn inside the MuJoCo viewer (route_dot mocap bodies) -> no RViz.
echo "[run] launching MuJoCo sim ... (Ctrl+C to stop & finalize video)"
ros2 launch motion_tracking_controller softtouch_dribble_mujoco.launch.py \
  policy_path:="$POLICY" launch_rviz:=false &
LAUNCH_PID=$!
pids+=("$LAUNCH_PID")

# Auto-record JUST the MuJoCo window (best effort): find it with xdotool, take its
# absolute geometry with xwininfo, capture with the SYSTEM ffmpeg (has x11grab;
# conda's does not).
if [[ "$RECORD" == "1" ]]; then
  (
    WID=""
    for _ in $(seq 1 60); do
      WID=$(xdotool search --name "MuJoCo" 2>/dev/null | tail -1)
      [[ -n "$WID" ]] && break
      sleep 0.5
    done
    if [[ -z "$WID" ]]; then echo "[rec] MuJoCo window not found; running without video."; exit 0; fi
    sleep 1.0  # let the viewer settle to its final size
    INFO=$(xwininfo -id "$WID" 2>/dev/null)
    X=$(awk '/Absolute upper-left X/{print $NF}' <<<"$INFO")
    Y=$(awk '/Absolute upper-left Y/{print $NF}' <<<"$INFO")
    W=$(awk '/Width:/{print $NF}'  <<<"$INFO")
    H=$(awk '/Height:/{print $NF}' <<<"$INFO")
    W=$(( W - W % 2 )); H=$(( H - H % 2 ))
    DUR_ARGS=(); [[ -n "$DURATION" ]] && DUR_ARGS=(-t "$DURATION")
    echo "[rec] recording ${W}x${H} at +${X},${Y} -> ${OUT_VIDEO} ${DURATION:+(${DURATION}s)}"
    exec /usr/bin/ffmpeg -y -f x11grab -framerate "$FPS" -video_size "${W}x${H}" \
      -i "${DISP}+${X},${Y}" "${DUR_ARGS[@]}" -c:v libx264 -preset veryfast -pix_fmt yuv420p "$OUT_VIDEO"
  ) &
  FFMPEG_PID=$!
fi

wait "$LAUNCH_PID"
