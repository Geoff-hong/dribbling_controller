#!/usr/bin/env bash
# Record a window to mp4 via ffmpeg x11grab. Click the MuJoCo window when prompted.
#
#   tools/ros_utils/record_mujoco.sh [output.mp4] [fps]
#
# Run this in your own terminal (so it has your X session DISPLAY/xauth), in a
# second terminal while the sim is running. Press q (in this terminal) or Ctrl+C
# to stop. NOTE: uses /usr/bin/ffmpeg — conda's ffmpeg has no x11grab.
set -o pipefail

OUT="${1:-$HOME/dribble_$(date +%Y%m%d_%H%M%S).mp4}"
FPS="${2:-30}"
DISP="${DISPLAY:-:0}"
FFMPEG=/usr/bin/ffmpeg

command -v xwininfo >/dev/null || { echo "[rec] need xwininfo"; exit 1; }
[[ -x "$FFMPEG" ]] || { echo "[rec] $FFMPEG missing"; exit 1; }

echo "[rec] Click the MuJoCo window to record it..."
INFO="$(xwininfo)" || { echo "[rec] xwininfo failed"; exit 1; }
X=$(awk '/Absolute upper-left X/{print $NF}' <<<"$INFO")
Y=$(awk '/Absolute upper-left Y/{print $NF}' <<<"$INFO")
W=$(awk '/Width:/{print $NF}'  <<<"$INFO")
H=$(awk '/Height:/{print $NF}' <<<"$INFO")
W=$(( W - W % 2 )); H=$(( H - H % 2 ))   # libx264 needs even dimensions

echo "[rec] ${W}x${H} at +${X},${Y} on ${DISP} -> ${OUT} @ ${FPS}fps"
echo "[rec] press q here (or Ctrl+C) to stop."
"$FFMPEG" -y -f x11grab -framerate "$FPS" -video_size "${W}x${H}" -i "${DISP}+${X},${Y}" \
  -c:v libx264 -preset veryfast -pix_fmt yuv420p "$OUT"
echo "[rec] saved: ${OUT}"
