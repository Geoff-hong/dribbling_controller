#!/usr/bin/env bash
# DR robustness sweep for the C++ ROS 2 deployment stack.
#
# For each domain-randomized MJCF variant (from tools/gen_dr_mjcf.py) this:
#   1. launches the SoftTouch dribble MuJoCo sim2sim with that model + a matching
#      ball_angular_damping (= 4*I) + a per-variant route seed,
#   2. runs tools/fall_monitor.py for a fixed window to judge stayed-up vs fell,
#   3. tears the sim down and moves on,
# then writes eval_result/<out>/summary.csv joining DR params with the verdict.
#
# NOTE: mujoco_sim_ros2 has no headless mode here, so a MuJoCo window pops per run
# (it closes on teardown). That is expected. This is a sequential robustness probe,
# not a large Monte-Carlo eval (use tools/dribble_pysim_multi.py for statistics).
#
#   POLICY=/abs/....onnx RESET=/abs/..._standby.txt ./tools/dr_robustness_sweep.sh
#
# Env overrides: N (variants, default 16), DUR (monitor seconds, 30), STARTUP (8),
#                OUT (eval_result/dr_robustness), SEED (sampling seed, 0).
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

POLICY="${POLICY:-/home/aldebaran/Desktop/SoftTouch-multiagent/checkpoints/g1_dribble_s3_human_dr_iter80000/softtouch_dribble_deploy.onnx}"
RESET="${RESET:-$HOME/softtouch_ros2_ws/install/motion_tracking_controller/share/motion_tracking_controller/config/g1/softtouch_mujoco_reset_standby.txt}"
N="${N:-16}"
DUR="${DUR:-10}"        # recorded dribble window, counted from the first base-pose msg
# Small head-start only: the monitor then BLOCKS until the first base-pose message
# and records DUR from there, so the robot's dribble episode ~= DUR (not STARTUP+DUR).
# Safe to keep small now that teardown is robust (pre-flight clean + pkill-by-name) and
# the monitor never kills mid-boot (it waits up to --max-wait for data).
STARTUP="${STARTUP:-4}"
ROUTE_LEN="${ROUTE_LEN:-}"   # empty = keep the long default route (100 m). The episode is
                             # bounded by DUR (monitor records DUR seconds, THEN the sim is
                             # killed), so the robot stops early on a long route rather than
                             # running to the route end. Set a value only to also cap the route.
SEED="${SEED:-0}"
OUT="${OUT:-eval_result/dr_robustness}"
WS="${WS:-$HOME/softtouch_ros2_ws}"

# conda python shadows ROS python -> drop it
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
  conda deactivate 2>/dev/null || true; conda deactivate 2>/dev/null || true
fi
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"

[[ -f "$POLICY" ]] || { echo "[sweep] ERROR: policy not found: $POLICY"; exit 1; }
[[ -f "$RESET"  ]] || { echo "[sweep] ERROR: reset not found: $RESET";  exit 1; }

mkdir -p "$OUT"
RESULTS="$OUT/results"; mkdir -p "$RESULTS"

echo "[sweep] generating $N DR variants (seed $SEED)..."
python3 tools/gen_dr_mjcf.py --n "$N" --seed "$SEED" || { echo "[sweep] gen failed"; exit 1; }
MANIFEST="mjcf/dr_variants/manifest.csv"
[[ -f "$MANIFEST" ]] || { echo "[sweep] ERROR: manifest missing: $MANIFEST"; exit 1; }

# install/share/mjcf is per-file symlinked at build time, so freshly generated
# variants are NOT visible to the sim. Symlink the whole dr_variants dir into it
# (the launch resolves mujoco_model_file relative to the package share).
SHARE_MJCF="$WS/install/motion_tracking_controller/share/motion_tracking_controller/mjcf"
if [[ -d "$SHARE_MJCF" ]]; then
  ln -sfn "$SCRIPT_DIR/mjcf/dr_variants" "$SHARE_MJCF/dr_variants"
  echo "[sweep] linked variants into $SHARE_MJCF/dr_variants"
else
  echo "[sweep] WARNING: $SHARE_MJCF not found; is the workspace built?"
fi

clean_stack() {  # hard-kill every node the launch spawns so nothing lingers between runs
  pkill -9 -f 'mujoco_sim'              2>/dev/null
  pkill -9 -f 'ros2 launch motion_tracking_controller softtouch_dribble' 2>/dev/null
  pkill -9 -f 'spawner .*state_estimator'  2>/dev/null
  pkill -9 -f 'robot_state_publisher'  2>/dev/null
}

kill_run() {  # SIGKILL the sim (with its in-process controller) all at once so the window
              # closes instantly with the robot frozen in its last controlled pose. A SIGTERM
              # grace period would kill the controller first -> policy stops -> the humanoid
              # goes limp for a moment before the window disappears, which looks bad.
  clean_stack   # pkill -9 by name (mujoco_sim hosts the controller, so it dies atomically)
  sleep 2       # let processes exit / DDS discovery settle before the next launch
}

trap 'echo; echo "[sweep] interrupted"; kill_run "${CUR_PGID:-}"; exit 130' INT TERM

# Pre-flight: clear any stale sim/controller processes from a previous (killed) run
# so the very first variant configures cleanly.
echo "[sweep] pre-flight: clearing any stale sim processes..."
clean_stack
sleep 3

# manifest cols: variant,mjcf,rel_model_file,ball_mass,ball_radius,foot_friction,ball_friction,ball_angular_damping,route_seed
# tr -d '\r': the CSV has \r\n line endings, so strip CR or the last field (seed)
# would carry a trailing carriage return into the launch arg.
tail -n +2 "$MANIFEST" | tr -d '\r' | while IFS=, read -r variant mjcf rel mass radius foot ball damp seed; do
  echo "==================================================================="
  echo "[sweep] variant $variant  m=$mass r=$radius foot=$foot ball=$ball damp=$damp seed=$seed"
  LOG="$RESULTS/launch_$variant.log"
  # ros2 launch rejects an empty-value arg, so only pass route length when set.
  ROUTE_ARG=(); [[ -n "$ROUTE_LEN" ]] && ROUTE_ARG=(softtouch_route_length_m:="$ROUTE_LEN")
  setsid ros2 launch motion_tracking_controller softtouch_dribble_mujoco.launch.py \
    policy_path:="$POLICY" \
    mujoco_reset_state_file:="$RESET" \
    mujoco_model_file:="$rel" \
    softtouch_ball_angular_damping:="$damp" \
    softtouch_seed:="$seed" \
    "${ROUTE_ARG[@]}" \
    launch_rviz:=false > "$LOG" 2>&1 &
  CUR_PGID=$!   # setsid makes the child its own process-group leader (pgid == pid)
  sleep "$STARTUP"
  python3 tools/fall_monitor.py --seconds "$DUR" --out "$RESULTS/verdict_$variant.json" --label "$variant" \
      || echo "[sweep] monitor error on $variant"
  kill_run "$CUR_PGID"; CUR_PGID=""
done

echo "[sweep] aggregating..."
python3 - "$MANIFEST" "$RESULTS" "$OUT/summary.csv" <<'PY'
import csv, json, sys
from pathlib import Path
manifest, results_dir, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
rows = list(csv.DictReader(open(manifest)))
fields = ["variant", "ball_mass", "ball_radius", "foot_friction", "ball_friction",
          "ball_angular_damping", "route_seed", "got_data", "fell", "min_z", "last_z", "survival_s"]
out_rows, n_fell, n_ok, n_nodata = [], 0, 0, 0
for r in rows:
    vp = results_dir / f"verdict_{r['variant']}.json"
    v = json.loads(vp.read_text()) if vp.exists() else {}
    row = {k: r.get(k, "") for k in fields[:7]}
    row.update({k: v.get(k, "") for k in fields[7:]})
    out_rows.append(row)
    if not v.get("got_data", False): n_nodata += 1
    elif v.get("fell"): n_fell += 1
    else: n_ok += 1
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out_rows)
tot = len(out_rows)
print(f"\n=== DR robustness: {tot} variants | stayed-up {n_ok} | fell {n_fell} | no-data {n_nodata} ===")
for row in out_rows:
    tag = "no-data" if row["got_data"] in ("", False) else ("FELL " if row["fell"] else "ok   ")
    print(f"  {row['variant']}  {tag}  min_z={row['min_z']}  surv={row['survival_s']}s  "
          f"m={row['ball_mass']} r={row['ball_radius']} foot={row['foot_friction']} ball={row['ball_friction']}")
print(f"\nsummary -> {out}")
PY
echo "[sweep] done."
