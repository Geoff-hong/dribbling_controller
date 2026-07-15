## Workflow
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
