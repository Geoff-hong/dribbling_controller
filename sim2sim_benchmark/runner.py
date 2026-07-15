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
                        episode_s=20.0, standby_hold_s=0.0, speed_pair_rows=None,
                        speed_trace_rows=None):
    """Run every condition in `table` for route_bank x reps episodes, appending one
    metrics dict per completed episode into `episode_rows` (the caller owns the
    list so partial results survive a crash or Ctrl-C). Conditions that record
    speed pairs additionally append (axis_value, cmd, actual) rows, downsampled
    to 10 Hz, into `speed_pair_rows`, and — for the first EIGHT episodes of each
    such condition — full-rate (50 Hz) trace rows into `speed_trace_rows` for
    the per-episode control trace plots."""
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
            # EVERY random draw of this episode (DR sampling, reset jitter, push
            # phases/directions — on top of the already route-seeded geometry) is
            # a pure function of (benchmark seed, condition, rep): experiments run
            # independently yet compare on IDENTICAL, paired episodes, regardless
            # of robot count or queue timing.
            rb.rng = np.random.Generator(np.random.PCG64(
                np.random.SeedSequence((seed, index, episode))))
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
                if speed_trace_rows is not None and episode < 8:
                    trace = rb.speed_trace()
                    if trace is not None:
                        cmd, along_cmd, speed_abs = trace
                        speed_trace_rows.extend(
                            (condition["axis"], episode, step, float(c), float(p), float(a))
                            for step, (c, p, a) in enumerate(zip(cmd, along_cmd, speed_abs)))
                done += 1
            assign_next_episode(rb, data.time)
        if not np.all(np.isfinite(data.qpos)):
            print(f"\n[{title}] DIVERGED — aborting this table")
            break
        print(f"\r[{title}] {done}/{total} episodes", end="", flush=True)
    print()
    return episode_rows


def record_condition_videos(table, title, video_dir, *, onnx, reset_file, seed=0,
                            episode_s=20.0, standby_hold_s=0.0, fps=30,
                            width=640, height=360):
    """One mp4 per condition: replay the rep-0 episode (route seed 0 — the same
    route the statistics used) with a chase camera on a single robot.

    Runs offscreen (set MUJOCO_GL=egl on a headless box). One small world per
    table keeps the render cheap; the whole pass adds a few sim-minutes."""
    import imageio
    import os

    max_route_len = max([engine.ROUTE_CFG["routeLength"]]
                        + [c["route_len_m"] for c in table if c["route_len_m"]])
    model, data, robots = engine.build_world(1, 2.0 * max_route_len + 20.0,
                                             onnx, reset_file, seed)
    rb = robots[0]
    rb.hold_s = standby_hold_s
    rb.episode_len_default = episode_s
    mujoco.mj_resetData(model, data)
    os.makedirs(video_dir, exist_ok=True)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance = 4.5; camera.elevation = -25.0; camera.azimuth = 90.0
    period_dt = model.opt.timestep * engine.DECIMATION
    render_every = max(1, round((1.0 / fps) / period_dt))
    effective_fps = 1.0 / (period_dt * render_every)
    for index, condition in enumerate(table):
        # same per-episode stream as the statistics run -> the video is an exact
        # replay of the rep-0 episode in the CSV
        rb.rng = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence((seed, index, 0))))
        rb.reset(model, data, data.time, route_seed=0, condition=condition)
        mujoco.mj_forward(model, data)
        video_path = os.path.join(video_dir, f"{condition['name']}.mp4")
        writer = imageio.get_writer(video_path, fps=effective_fps, codec="libx264",
                                    quality=7, macro_block_size=None,
                                    ffmpeg_log_level="error")
        step = 0
        while True:
            ended = engine.step_control_period(model, data, robots, standby_hold_s)
            if step % render_every == 0:
                base = data.qpos[rb.bq:rb.bq + 3]
                camera.lookat = [float(base[0]), float(base[1]), 0.5]
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
            step += 1
            if 0 in ended or not np.all(np.isfinite(data.qpos)):
                break
        writer.close()
        print(f"\r[{title}] videos {index + 1}/{len(table)}", end="", flush=True)
    print()
    print(f"[{title}] saved {len(table)} videos under {video_dir}")
