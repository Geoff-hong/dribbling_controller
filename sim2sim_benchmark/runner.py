"""Queue-based executor: run a condition table on the pysim engine.

Every queued episode COMPLETES (no truncation bias). Every condition cycles the
same route-seed bank, so human routes AND the per-episode lead/angle draws of
corner turns are reproduced across conditions and experiments — comparisons are
made on paired draws.
"""
import numpy as np
import mujoco

from . import engine


def run_condition_table(table, title, stream, *, onnx, reset_file,
                        n_robots=32, seed=0, route_bank=12, reps=4,
                        episode_s=20.0, standby_hold_s=0.0):
    """Run every condition in `table` for route_bank x reps episodes. Each
    completed episode is written to `stream` (report.EpisodeStream) immediately —
    the CSV is the progress record, and episodes already in it are skipped, so a
    killed run resumes for free. Conditions that record speed pairs additionally
    stream (axis_value, rep, cmd, actual) rows downsampled to 10 Hz, and — for
    the first EIGHT episodes of each such condition — full-rate (50 Hz) trace
    rows for the per-episode control trace plots."""
    bank = max(1, route_bank)
    pending = [(index, episode, episode % bank)
               for index in range(len(table))
               for episode in range(bank * max(1, reps))]
    already_done = len(pending)
    pending = [p for p in pending
               if not stream.is_done(table[p[0]]["name"], p[1])]
    already_done -= len(pending)
    if already_done:
        print(f"[{title}] resume: {already_done} episodes already in {stream.csv_path}")
    if not pending:
        print(f"[{title}] nothing left to run")
        return
    np.random.default_rng(seed).shuffle(pending)   # interleave over time/robots
    total = len(pending); next_pending = 0
    est_hours = sum(float(table[index]["episode_s"] or episode_s)
                    for index, _, _ in pending) / 3600.0

    # isolate the robots: no route may reach a neighbour's workspace mid-episode
    max_route_len = max([engine.ROUTE_CFG["routeLength"]]
                        + [c["route_len_m"] for c in table if c["route_len_m"]])
    spacing = 2.0 * max_route_len + 20.0
    print(f"[{title}] {len(table)} conditions x {bank * max(1, reps)} episodes; "
          f"{total} to run (~{est_hours:.1f} robot-hours, spacing {spacing:.0f} m)")
    # visual=False: this world is never rendered, so drop the render-only meshes
    # (~11x less RSS at 32 robots). Physics is unchanged — see _single_robot_xml.
    model, data, robots = engine.build_world(n_robots, spacing, onnx, reset_file, seed,
                                             visual=False)
    for rb in robots:
        rb.hold_s = standby_hold_s
        rb.episode_len_default = episode_s
    mujoco.mj_resetData(model, data)

    def assign_next_episode(rb, t):
        nonlocal next_pending
        if next_pending < total:
            index, episode, route_seed = pending[next_pending]; next_pending += 1
            rb.assignment = (index, episode, route_seed)
            # EVERY random draw of this episode (DR sampling, reset jitter, push
            # phases/directions — on top of the already route-seeded geometry) is
            # a pure function of (benchmark seed, condition, rep). _seed_index
            # (set by --shard) keeps the FULL-table condition index.
            #
            # ⚠ This seeds the SETTINGS identically, NOT the trajectory. All robots
            # share one mjData on a spaced grid, so a slot's absolute world
            # coordinates (and its co-residents) perturb the last float bits, and
            # 20 s of contact-rich dribbling amplifies that chaotically. Measured
            # 2026-07-19 (m8500_bodyframe): three byte-identical conditions gave
            # 29/46/46 % survival, flipping ~half the per-episode fall outcomes at
            # the same (rep, route_seed); at --robots 1 the same three agreed. A
            # single process IS bit-reproducible run-to-run, so the slot draw acts
            # as legitimate randomization — rates stay unbiased, but conditions are
            # NOT route-paired and carry the full binomial SE (~7 pts at n=48).
            # Do not read differences below ~20 points as signal.
            rb.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                (seed, table[index].get("_seed_index", index), episode))))
            rb.reset(model, data, t, route_seed=route_seed, condition=table[index])
        else:
            rb.assignment = None
            rb.reset(model, data, t)

    needs_kinematics = any(rb.chest_body is not None for rb in robots)
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
                row = dict(
                    condition=condition["name"], group=condition["group"],
                    axis_value=condition["axis"], rep=episode, route_seed=route_seed,
                    **rb.dr, **rb.episode_metrics(data, data.time))
                pair_rows = []
                pair_arrays = rb.speed_pair_arrays()
                if pair_arrays is not None:
                    cmd, actual = pair_arrays
                    pair_rows = [(condition["axis"], episode, float(c), float(a))
                                 for c, a in zip(cmd[::5], actual[::5])]  # 50 Hz -> 10 Hz
                trace_rows = []
                if episode < 8:
                    trace = rb.speed_trace()
                    if trace is not None:
                        cmd, along_cmd, speed_abs = trace
                        trace_rows = [
                            (condition["axis"], episode, step, float(c), float(p), float(a))
                            for step, (c, p, a) in enumerate(zip(cmd, along_cmd, speed_abs))]
                stream.write_episode(row, pair_rows, trace_rows)
                done += 1
            assign_next_episode(rb, data.time)
        if ended and needs_kinematics:
            # chest-frame obs reads data.xpos/xquat, which a reset's qpos writes do
            # not refresh — the stale first frame would be tiled into every history
            # slot. Pure FK, so pelvis-frame runs stay bit-identical (hence the guard).
            mujoco.mj_kinematics(model, data)
        if not np.all(np.isfinite(data.qpos)):
            print(f"\n[{title}] DIVERGED — aborting this table")
            break
        print(f"\r[{title}] {done}/{total} episodes", end="", flush=True)
    print()


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
                                             onnx, reset_file, seed, visual=True)
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
        rb.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            (seed, condition.get("_seed_index", index), 0))))
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
