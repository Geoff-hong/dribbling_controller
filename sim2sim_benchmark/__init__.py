"""Sim2sim standard benchmark for the SoftTouch dribble policy.

Two separate tests, run from the repo root:

  # robustness: perturb the environment, nominal command
  #   axes: DR scale | base push | ball push | obs latency | action latency
  #   metrics: survival / possession / speed ratio / cross-track
  # capability: clean env, extreme command, fail-fast control criteria
  #   axes: straight-line max speed | corner-turn max curvature
  #   metrics: nested possession/route/strict success + cross-track
  python -m sim2sim_benchmark --robustness --capability \
      --onnx /abs/path/softtouch_dribble_deploy.onnx \
      --reset config/g1/softtouch_mujoco_reset_standby.txt \
      --robots 32 --out-dir eval_result/m80000

  # interactive HTML report (DEFAULT — auto-discovers runs/, --serve for live)
  python -m sim2sim_benchmark.html_report --run-dirs eval_result/m80000 eval_result/m90000 \
      --labels iter80000 iter90000 --out eval_result/compare/report.html

Modules: engine (Route / Robot / MuJoCo model composition — the pysim core),
conditions (the tables), runner (queue execution on the engine), report (CSV +
console summary), html_report (the default interactive report), plot (LEGACY
static PNGs, superseded by html_report; kept for demo/ mock-ups), pysim (the
interactive / legacy CLI: viewer, --record, --headless, --eval, --sweep).

The runner drives the engine per-episode through Robot.reset(condition=...).
"""
