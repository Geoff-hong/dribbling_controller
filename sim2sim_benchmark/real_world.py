"""REAL-WORLD reference values for the robustness axes — the sim2real overlay.

The robustness axes themselves stay anchored on the checkpoint's TRAINING DR
(that is what makes a drop attributable to a parameter). This table is what gets
drawn ON TOP of them: every robustness panel marks where the deployment hardware
actually sits on its axis, so the read becomes "how much margin does the policy
have AT the real operating point", not just "how far past training does it
survive".

  nominal — the measured deploy value (drawn as a vertical marker)
  band    — the measurement spread (drawn as a shaded span)
  note    — provenance, shown in the report tooltip

nominal=None AND band=None -> the channel has not been measured yet and the
panel simply gets no marker. Fill these in from hardware measurements; do NOT
guess. An unmarked panel is honest; a wrong marker silently reframes every
conclusion drawn from the figure.

ball_damping is the one channel where the real value sits far OUTSIDE the
trained one: the checkpoints were calibrated on grass (c = 4.0, a 1 m/s ball
rolls 3.5/c = 0.88 m) while the hardware measures c = 0.9 (rolls 3.9 m) — a
4.4x gap. That is why it also gets its own real-anchored sweep axis
(conditions.ball_damping_axis), instead of only a marker.

Pure data: numpy/mujoco-free, so both conditions.py (engine side) and plot.py /
html_report.py (reporting side, documented as engine-independent) can import it.
"""

REAL_WORLD = {
    "ball_mass":     dict(nominal=None, band=None, note="TODO: weigh the deploy ball"),
    "ball_radius":   dict(nominal=None, band=None, note="TODO: measure the deploy ball"),
    "foot_friction": dict(nominal=None, band=(0.3, 0.6), note="measured, indoor test floor"),
    "ball_friction": dict(nominal=None, band=None,
                          note="TODO: SLIDING coefficient — dr['ball'] sets MuJoCo pair "
                               "friction[0:2] (slide), NOT the rolling term"),
    "ball_damping":  dict(nominal=0.9, band=(0.4, 2.0),
                          note="2026-07-17 hardware free-roll, indoor floor (k = 0.26/s); "
                               "band = measured decel spread 0.06-0.5 m/s^2"),
    "obs_latency":   dict(nominal=None, band=None, note="TODO: measure the deploy vision pipeline"),
    "act_latency":   dict(nominal=None, band=None, note="TODO: measure DDS + motor response"),
    "base_push":     dict(nominal=None, band=None, note=""),
    "ball_push":     dict(nominal=None, band=None, note=""),
}


def real_marker(group):
    """(nominal, band, note) for a robustness group, or None when unmeasured."""
    entry = REAL_WORLD.get(group)
    if not entry or (entry["nominal"] is None and entry["band"] is None):
        return None
    return entry["nominal"], entry["band"], entry["note"]
