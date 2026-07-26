# Sim2Sim Benchmark (pysim + standard benchmark)

This package lives on the `benchmark` branch only; `main` carries just the
C++/ROS 2 deploy stack.  The branch contains the full deploy tree (the engine
loads `mjcf/g1_softtouch_dribble.xml` and the `config/g1/` reset states) and
picks up deploy changes via one-way `git merge main` — never merge the other
way; promote validated assets to main by cherry-pick.

Package layout:

- `sim2sim_benchmark/` — simulation engine (`engine.py` — Route/Robot/
  multi-robot model composition), condition tables (`conditions.py`), queue
  runner (`runner.py`), CSV/console report (`report.py`), interactive HTML
  report (`html_report.py`), legacy figures (`plot.py`), and the interactive
  CLI (`pysim.py` — viewer, `--record`, `--headless`, `--eval`, `--sweep`).
- `tests/` — benchmark metric tests.
- `tools/calibration/` — one-off physics-calibration experiments whose
  conclusions are baked into the MJCF: `ball_roll_test.py` (ball angular
  damping vs the PhysX roll/decay reference) and `friction_slip_test.py`
  (contact-solver impratio/cone vs stance creep).

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
ros2_control sim2sim documented in the top-level README (main branch).  Run it from the SoftTouch Python env (torch / onnx /
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
  (action lag, ms).  Metrics: survival rate, ball possession (sticky nearest-foot
  to ball-surface threshold), achieved/commanded speed ratio, cross-track.
- **Capability** — clean nominal env (+ small reset jitter), extreme commands,
  fail-fast control criteria: the episode fails after the ball remains >0.8 m
  off the route for the configured dwell, or exceeds 1.2 m from the robot.
  The report shows three nested verdicts: upright+possession at termination,
  route-control success, and strict completion success. It also shows cross-track
  both on strict successes (the primary companion to success rate) and on all
  upright episodes (a diagnostic that may be censored by fail-fast).
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

All nominal conditions match the C++ sim2sim path's timing exactly: ball AND
base observations (obs frame, route input included) cross the 100 Hz bridge
topic hop, whose per-tick staleness was MEASURED on the C++ stack (stamps
logged in the controller, 2026-07-23) as 0 ms 22% / 5 ms 60% / 10 ms 18% —
the physics thread steps in bursts, so delivery often beats the tick. The
engine samples that distribution per tick (`bridge_delay_ms` = the publish
period); joint state is fresh and the action applies the same sim step. No
synthetic latency on top (ball lag 0 steps, action lag 0 ms). The latency axes
vary one synthetic channel at a time on top of that structural staleness; real
hardware latency remains unmeasured and is not guessed into the baseline.
Getting this staleness right matters: modeling it as a FIXED 10 ms cost 14 pts
of mean straight-speed success (58% vs 72%) and manufactured a spurious
low-speed off-route failure mode. Episodes per condition = `--route-bank`
(12) x `--reps` (4).  Every per-episode random draw — route geometry, cruise
pace, corner lead/angle, DR sampling, reset jitter, push phases — is a pure
function of (benchmark seed, condition, rep), so independently-run experiments
compare on IDENTICAL paired episodes; pick any set of run dirs to merge at
plot time.  A custom table can be run with `--conditions table.json`.

The current condition protocol is v4 (bridge-staleness timing parity, 12 s
default episode budget). It also matches the C++ human-route lazy extension and
`std::mt19937` float stream, and enforces the nested success invariant. Episode
rows from earlier protocols are not valid top-ups for v4; write a new run
directory (or explicitly apply the top-up migration) before comparing scores.

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

# DEFAULT REPORT: interactive single-file HTML (tensorboard-style): experiment
# checkboxes, robustness/capability panels, significance view, per-checkpoint
# training-DR table, control traces, per-condition video index
$PY -m sim2sim_benchmark.html_report \
    --run-dirs sim2sim_eval_results/runs/m80000 sim2sim_eval_results/runs/m90000 \
    --labels iter80000 iter90000 --out sim2sim_eval_results/compare/report.html
# or just: $PY -m sim2sim_benchmark.html_report   (auto-discovers runs/ , --serve for live)

# LEGACY static PNGs (optional; html_report supersedes these). Only for a
# browser-less export or the demo/ mock-ups:
$PY -m sim2sim_benchmark.plot --run-dirs sim2sim_eval_results/runs/m80000 \
    sim2sim_eval_results/runs/m90000 \
    --labels iter80000 iter90000 --out-dir sim2sim_eval_results/compare
# -> robustness_compare.png / speed_compare.png / route_compare.png /
#    uturn_compare.png / speed_traces_<label>.png per experiment
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
