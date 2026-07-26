## Alignment target: the C++ deploy stack, NOT Isaac

`sim2sim_benchmark` exists to predict what the **C++/ROS 2 deploy stack** does.
Isaac is the thing being checked — aligning the benchmark to Isaac would mean
validating a simulator against itself, which measures nothing.

- **Behaviour and implementation → match C++.** Control mode, PD execution,
  actuator type, timestep/integrator, target clipping, effort limits,
  observation construction, ball-state staleness handling, hand-off timing.
  When the benchmark and C++ differ, C++ wins — even where Isaac looks "more
  correct". The C++ sim2sim and the benchmark share `mjcf/g1_softtouch_dribble.xml`,
  so the MJCF is common ground: a change there hits both.
- **Numeric parameters → come from training.** DR ranges, PD gains, action
  scale, joint/geometry values, curriculum-derived anchors. These describe the
  checkpoint being evaluated, so they are read from its `env.yaml` (`train_dr.py`)
  or the training URDF.
- A discrepancy vs Isaac is a **finding about sim2real transfer**, not a
  benchmark bug. Do not "fix" the benchmark toward Isaac. Example: the benchmark
  applies PD as explicit torque on `<motor>` actuators, where Isaac uses an
  implicit PhysX drive; that costs ~15 pts of survival under latency, and it is
  CORRECT because the C++ system interface computes the same explicit torque
  against the same MJCF.

## Workflow
- **Eval artifacts: keep only valid runs.** If a play/eval turns out to be
  misconfigured (wrong friction/DR, wrong era, wrong mode mix — anything that
  invalidates the numbers), move its diagnostic dir and video to
  `~/Desktop/trash/` (descriptive subdir name) immediately after the corrected
  rerun exists — no need to ask. Never leave superseded/invalid eval outputs
  in run dirs; they pollute later comparisons.
- **Smoke tests are disposable.** Move smoke-test scripts/outputs to
  `~/Desktop/trash/` as soon as they've served their purpose — same session,
  not "later".
- **Eval videos go in the run, never elsewhere.** Recorded clips belong under
  `sim2sim_eval_results/runs/<run>/videos/<title>/<condition>.mp4` (the layout
  `record_condition_videos` already writes). Do NOT drop them on `~/Desktop/`,
  in the scratchpad, or any ad-hoc folder — put them next to the run whose
  policy they show.
- Before evaluating a checkpoint, read its own params/command.txt (or
  env.yaml/README) and match ITS training params — friction, ball damping,
  latency, reset distribution, curriculum state at that iteration. A
  checkpoint from an older code era may need era-only knobs disabled (e.g.
  --no_joint_friction_dr) or may not be fairly evaluable at current HEAD.
- After making changes, summarize what was modified and ask whether to commit
- Once the user approves the commit, push the same batch without asking again
- Do NOT auto-commit (wait for explicit approval on the commit itself)
- When proposing a task to run, state acceptance criteria upfront
- Provide exact commands when the user needs to run something

## Code Style
- All code comments in English
- Minimal comments, only where logic is non-obvious
- Follow Isaac Lab patterns: `@configclass`, ManagerBasedRLEnv, gym.register
- Reuse existing code: Stage-1 stack under `multiagent_sim/`; copy from PULSE/LATENT/goalkeeper for higher stages

### Git — Ask Before Commit
After every code change, ask the user whether to commit. On confirmation: `git add` → `git commit` (descriptive msg) → `git push` (no second confirmation; commit approval implies push for the same batch).

### What to commit
- `logs/` (training metadata, tfevents, videos, exported ONNX, git diffs) — always commit
- `*.pt` model checkpoints — excluded by `.gitignore`; do not force-add
- `reference/` — read-only, never commit changes under it
- `*.egg-info/` — build artifact, do not commit
- Anything under `data/` is gitignored — viewer/tool files belonging there should be moved to `tools/` if they need tracking
- Route/cmd design figures → top-level `route_design/` (conventions in its README), never `data/`

## Behavior
- Always verify current state (`git status`, read file) before asserting something. Never assume from earlier context.
- "Look at the data" = load the raw arrays. Filenames in this project mislabel content (AA Soccer's `Goal_Ball_*` aren't ball-aware, etc.).
- Before proposing to train/build, `ls logs/` and `ls data/` — assets often exist but aren't referenced by any script.
- Before quoting "N clips / teachers", count the directory. A script's list is usually a subset.
- When the user asks "did you actually look?", the answer is almost always no — stop and re-read from data, not from your prior summary.

## What NOT To Do
- Do not modify anything under `reference/`
- Do not re-introduce a runtime dependency on `whole_body_tracking` — Stage-1 code lives in `multiagent_sim/`
- Do not rewrite VAE / DAgger / reward structure from scratch — copy from PULSE / LATENT / goalkeeper
- Do not use DirectRLEnv — use ManagerBasedRLEnv
- Do not write custom PPO — use RSL-RL
- Do not add heavy config frameworks beyond what Isaac Lab uses internally
- Do not add TensorBoard/WandB until training actually runs
- Do not use IsaacGym Preview 4
- Do not commit dribble work to `kick-keep-sim` or keeper work to this branch
- Do not assume the AA Soccer pack contains ball data — it does not
