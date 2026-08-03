# SoftTouch Hardware Analysis — Selected Episodes

## Scope and headline result

Episodes: `205524/E02`, `224829/E01`, `E02`, `E03`, `E05`, `E08`.

CT is the nearest 2-D distance from the recorded ball center to the deployed
commanded-route polyline at each route-marker timestamp. Only trusted samples
before the first >0.5 m mocap jump are included.

- Point-weighted CT: **mean 0.144 m**, median
  0.101 m, RMS 0.214 m,
  P90 0.346 m, maximum 0.916 m.

- Episode-balanced mean CT: **0.140 ±
  0.056 m**.

- Three late failures are qualitatively different: E01 is mainly **ball
  escape**, while E03/E05 are mainly **whole-body route drift**.

## Episode table

| Episode | Trusted s | Ball CT mean | RMS | P90 | Final | Humanoid final CT | Humanoid speed | Low swings | Max chest drop cm | Diagnosis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 205524 / E02 | 7.84 | 0.133 | 0.158 | 0.271 | 0.281 | 0.494 | 0.37 | 4/19 | 13.1 | Moderate late drift |
| 224829 / E01 | 8.40 | 0.195 | 0.303 | 0.463 | 0.888 | 0.290 | 0.62 | 5/21 | 12.4 | Late ball escape + late mocap dropout |
| 224829 / E02 | 6.04 | 0.075 | 0.092 | 0.143 | 0.140 | 0.071 | 0.63 | 6/15 | 11.0 | Best CT; truncate before mocap jump |
| 224829 / E03 | 9.24 | 0.142 | 0.211 | 0.424 | 0.509 | 0.510 | 0.53 | 3/22 | 15.7 | Ball and humanoid drift together |
| 224829 / E05 | 8.12 | 0.212 | 0.284 | 0.557 | 0.564 | 0.666 | 0.66 | 3/20 | 18.9 | Ball and humanoid drift together |
| 224829 / E08 | 7.60 | 0.082 | 0.101 | 0.178 | 0.120 | 0.252 | 0.65 | 1/17 | 9.5 | Low-CT reference |

The speed command is 0.40 m/s in every episode. The displayed humanoid speed is
a 0.4 s centered displacement estimate, which filters normal gait oscillation.

## Why the feet scrape the ground

The evidence supports a **systematic low-clearance, toe-limited gait**, rather
than one isolated sensor failure:

- Exact foot STL + time-aligned robot TF reconstruction finds
  **22/114 swing phases** whose
  complete foot mesh never clears the support-foot plane by 3 cm;
  7 remain below 2 cm.

- **20/22**
  low-clearance swings are toe-limited. This matches the observed toe/forefoot
  rubbing: the heel can rise while the toe remains close to the floor.

- The chest drops roughly 9–10 cm at the 95th percentile after activation,
  and the worst selected episode reaches about
  18.9 cm.

- Leg states best align with their changing position targets about
  **140 ms later**. This is an effective
  plant/actuator phase response, not a claim of 140
  ms network delay.

The checkpoint configuration explains why this behavior can survive training:

1. The archived reward set has **no foot-clearance or foot-sliding term**.

2. Both ankle-roll links are excluded from the undesired-contact penalty.

3. Positive terms reward fast foot motion near/at ball contact.

4. The actor observes neither base height, base linear velocity, nor foot
   contact state, so it cannot directly detect sag or scraping.

5. Sustained deployment at 0.40 m/s is below the training slow-cruise range
   (0.5–1.1 m/s; only 25% of training cruise) and far below the primary range
   (1.1–2.0 m/s). Five of six episodes still move faster than 0.50 m/s.

**Conclusion:** scraping is not caused by a single corrupt mocap sample. The
primary design cause is that the learned gait is allowed to use a very small
toe margin; slow-command out-of-distribution operation, body sag, and real
joint phase response consume the remaining margin. Scraping then adds
unmodeled friction/impulses and weakens directional control. It is systematic,
but it is not the only cause of high CT because low-CT E02/E08 also scrape.

## Other major findings

- **Ball dynamics mismatch:** the deployed checkpoint was trained with ball
  angular damping 4.0 s⁻¹, while its archived README identifies 0.9 s⁻¹ as the
  hardware calibration. E01 ends with ball CT 0.888 m while humanoid CT is
  0.290 m, consistent with a ball that escapes farther than the body.

- **Data corruption in 224829/E02:** ball position jumps
  1.456
  m at about
  6.09
  s. The last trusted synchronized sample is at
  6.04
  s, and all reported E02 metrics stop there.

- **Late mocap dropout in 224829/E01:** one 39.98 ms policy interval occurs
  near 8.04 s when the chest pose goes stale. CT had already begun diverging
  at 6.40 s, so the dropout amplifies the ending but does not initiate it.

- **Real-time overruns:** E03 contains two in-episode 500 Hz overruns
  (maximum 9.44 ms, 7 missed cycles). One occurs immediately before sustained
  CT exceeds 0.20 m. This is a secondary risk, not a session-wide explanation.

- **No ball-velocity spike issue in the selected trusted data:** there are zero
  policy observations above 5 m/s.

## Recommended order of fixes

1. Add swing-toe/full-foot clearance and stance-foot slip penalties; retain a
   positive margin (for example 4–6 cm) through the central swing phase.

2. Fine-tune with sustained 0.35–0.60 m/s commands and a direct root-velocity
   vector tracking term. Validate speed and turning before returning to ball
   dribbling.

3. Fit sim actuator dynamics to the measured target/state phase response and
   recheck ankle-pitch/knee behavior under the real gains.

4. Match ball damping to hardware (or randomize around it), then rerun E01-like
   ball-escape tests.

5. Eliminate chest-mocap dropouts and enable deterministic real-time scheduling;
   these are not the primary cause but can turn a recoverable drift into a
   failure.

## Figures

- `softtouch_focus_summary.png`: one-slide summary.

- `softtouch_focus_ct_timeline.png`: ball versus humanoid CT.

- `softtouch_focus_scrape_diagnostics.png`: reconstructed foot clearance and
  chest drop.

- `softtouch_focus_route_map.png`: route, ball, humanoid, and commands. Command
  arrows are sampled at equal 1.0 s intervals.

## Method limits

There is no recorded foot force/contact topic. Therefore the report proves
low geometric clearance and identifies scrape-risk phases, but exact physical
contact instants still require foot force, motor-current residual, or
high-speed video confirmation. The support-foot reference cancels absolute
mocap height bias; it does not estimate floor compliance.

The 140 ms joint value is a best-fit phase alignment of changing PD equilibrium
targets and measured joints. It includes the real plant response and must not
be compared one-to-one with the explicit 0–20 ms action-delay randomization;
the simulator's own actuator dynamics can also create phase lag.

Finally, these are six deliberately selected episodes, not a random sample of
all deployments. The statistics are descriptive; `±` above is the standard
deviation across six episode means, not a population confidence interval.
