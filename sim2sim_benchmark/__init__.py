"""Sim2sim standard benchmark for the SoftTouch dribble policy.

Two separate tests, run from the repo root:

  # robustness: perturb the environment, nominal command
  #   axes: DR scale | base push | ball push | obs latency | action latency
  #   metrics: survival / possession / speed ratio / cross-track
  # capability: clean env, extreme command, fail-fast control criteria
  #   axes: straight-line max speed | corner-turn max curvature
  #   metric: success rate
  python -m sim2sim_benchmark --robustness --capability \
      --onnx /abs/path/softtouch_dribble_deploy.onnx \
      --reset config/g1/softtouch_mujoco_reset_standby.txt \
      --robots 32 --out-dir eval_result/m80000

  # comparison figures (one color per experiment)
  python -m sim2sim_benchmark.plot --run-dirs eval_result/m80000 eval_result/m90000 \
      --labels iter80000 iter90000 --out-dir eval_result

Modules: conditions (the tables), runner (queue execution on the pysim engine),
report (CSV + console summary), plot (comparison figures; engine-independent).

The simulation engine (Route / Robot / model composition) lives in
tools/dribble_pysim_multi.py; this package drives it per-episode through
Robot.reset(condition=...).
"""
import os
import sys

# the pysim engine is a standalone script under tools/, not an installed package
_TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
