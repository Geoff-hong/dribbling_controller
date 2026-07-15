# Benchmark figure previews (MOCK DATA)

What a benchmark run's figures look like — rendered by the REAL plotting code
(`sim2sim_benchmark.plot`) from synthetic placeholder numbers, so the layout,
axes and legends are exactly what `python -m sim2sim_benchmark.plot` produces,
while every curve is fake (two invented experiments, smooth sigmoids + noise).

- `robustness_compare.png` — columns = perturbation axes (DR scale, base push,
  ball push, obs latency, action latency); rows = survival %, ball possession %,
  speed ratio, cross-track; dotted line = each experiment's nominal baseline.
- `speed_compare.png` — the SPEED test: max-speed success rate and achieved-vs-
  commanded speed on the straight route (the plateau off the y = x line is the
  measured max dribble speed), plus controllability — binned commanded-vs-actual
  ball speed over the human-route vmax sweep (0.8-2.0 m/s) with the pooled
  Pearson r per experiment.
- `speed_traces_<label>.png` — per-experiment control traces, one panel per
  vmax: gray = raw ball |v|, red = |v| smoothed 0.5 s, blue = ball speed along
  the command smoothed 0.5 s, black dashed = the commanded speed steps.
- `route_compare.png` — the ROUTE test: corner-turn success rate and survivor
  cross-track over |kappa|; solid = left turns, dashed = right turns.

Regenerate (only numpy + matplotlib needed; the sim imports are stubbed):

```bash
python sim2sim_benchmark/demo/make_demo_figures.py
```
