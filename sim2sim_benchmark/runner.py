"""Queue-based executor: run a condition table on the pysim engine.

Every queued episode COMPLETES (no truncation bias). Every condition cycles the
same route-seed bank, so human routes AND the per-episode lead/angle draws of
corner turns are reproduced across conditions and experiments — comparisons are
made on paired draws.
"""
import numpy as np
import mujoco

from . import engine


def run_condition_table(table, title, episode_rows, *, onnx, reset_file,
                        n_robots=32, seed=0, route_bank=12, reps=4,
                        episode_s=20.0, standby_hold_s=0.0, speed_pair_rows=None):
    """Run every condition in `table` for route_bank x reps episodes, appending one
    metrics dict per completed episode into `episode_rows` (the caller owns the
    list so partial results survive a crash or Ctrl-C). Conditions that record
    speed pairs additionally append (axis_value, cmd, actual) rows, downsampled
    to 10 Hz, into `speed_pair_rows` when given."""
    # isolate the robots: no route may reach a neighbour's workspace mid-episode
    max_route_len = max([engine.ROUTE_CFG["routeLength"]]
                        + [c["route_len_m"] for c in table if c["route_len_m"]])
    spacing = 2.0 * max_route_len + 20.0
    model, data, robots = engine.build_world(n_robots, spacing, onnx, reset_file, seed)
    for rb in robots:
        rb.hold_s = standby_hold_s
        rb.episode_len_default = episode_s
    mujoco.mj_resetData(model, data)

    bank = max(1, route_bank)
    pending = [(index, episode, episode % bank)
               for index in range(len(table))
               for episode in range(bank * max(1, reps))]
    np.random.default_rng(seed).shuffle(pending)   # interleave over time/robots
    total = len(pending); next_pending = 0
    est_hours = sum(bank * max(1, reps) * float(c["episode_s"] or episode_s)
                    for c in table) / 3600.0
    print(f"[{title}] {len(table)} conditions x {bank * max(1, reps)} episodes "
          f"= {total} (~{est_hours:.1f} robot-hours, spacing {spacing:.0f} m)")

    def assign_next_episode(rb, t):
        nonlocal next_pending
        if next_pending < total:
            index, episode, route_seed = pending[next_pending]; next_pending += 1
            rb.assignment = (index, episode, route_seed)
            rb.reset(model, data, t, route_seed=route_seed, condition=table[index])
        else:
            rb.assignment = None
            rb.reset(model, data, t)

    for rb in robots:
        assign_next_episode(rb, 0.0)
    mujoco.mj_forward(model, data)

    done = 0
    while done < total:
        ended = engine.step_control_period(model, data, robots, standby_hold_s)
        for j in ended:
            rb = robots[j]
            if rb.assignment is not None:
                index, episode, route_seed = rb.assignment
                condition = table[index]
                episode_rows.append(dict(
                    condition=condition["name"], group=condition["group"],
                    axis_value=condition["axis"], rep=episode, route_seed=route_seed,
                    **rb.dr, **rb.episode_metrics(data, data.time)))
                if speed_pair_rows is not None:
                    pair_arrays = rb.speed_pair_arrays()
                    if pair_arrays is not None:
                        cmd, actual = pair_arrays
                        speed_pair_rows.extend(
                            (condition["axis"], float(c), float(a))
                            for c, a in zip(cmd[::5], actual[::5]))   # 50 Hz -> 10 Hz
                done += 1
            assign_next_episode(rb, data.time)
        if not np.all(np.isfinite(data.qpos)):
            print(f"\n[{title}] DIVERGED — aborting this table")
            break
        print(f"\r[{title}] {done}/{total} episodes", end="", flush=True)
    print()
    return episode_rows
