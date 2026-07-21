"""Bootstrap confidence intervals for benchmark comparisons (numpy only).

Why bootstrap rather than a closed-form SE: the report mixes rates (survival,
success) with means (cross-track, achieved speed) and with survivor-conditioned
means, so one resampling path covers all of them, including the p=0/p=1 corners
where the binomial SE collapses to zero width and asserts false certainty.

Why PAIRED: the runner cycles a fixed route-seed bank, so two runs of the same
condition table saw the SAME routes. Differencing within a route removes route
difficulty from the comparison. Pairing is per (condition, rep): the runner
assigns route_seed = rep % bank deterministically, so the same (condition, rep)
is the same route in both runs.

Measured caveat (m16000 vs m80000, 2026-07-20): pairing tightens the survival CI
only slightly (+-1.9 pts paired vs +-2.2 unpaired over all 3504 episodes). The
dominant noise is NOT route difficulty but the shared-mjData robot-slot draw
documented in runner.assign_next_episode, which pairing cannot undo. Pairing is
a real but modest gain here; it would matter far more at --robots 1.

Statistics are vectorised over the whole (n_boot, n) resample matrix, which is
what makes a per-condition forest plot across every run pair affordable.
"""
import numpy as np

DEFAULT_N_BOOT = 2000
DEFAULT_ALPHA = 0.05


def _rng(seed):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def mean_stat(values):
    """NaN-ignoring mean over the last axis.

    Accepts a 1-D sample or a whole (n_boot, n) resample matrix; returning an
    array for the matrix case is what lets `bootstrap_*` skip a python loop."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    with np.errstate(invalid="ignore"):
        out = np.nanmean(values, axis=-1)
    return float(out) if np.ndim(out) == 0 else out


def rate_stat(values):
    """Percentage of the finite entries that are 1 (0/1 flags -> rate in %)."""
    out = mean_stat(values)
    return 100.0 * out if np.ndim(out) else 100.0 * float(out)


def _apply(stat_fn, matrix):
    """stat_fn over each row of `matrix`; falls back to a loop for a stat that
    is not vectorised (e.g. a percentile written scalar-only)."""
    out = np.asarray(stat_fn(matrix), dtype=float)
    if out.shape == (matrix.shape[0],):
        return out
    return np.array([float(np.asarray(stat_fn(row))) for row in matrix])


def bootstrap_ci(values, stat_fn=mean_stat, n_boot=DEFAULT_N_BOOT,
                 alpha=DEFAULT_ALPHA, seed=0):
    """Percentile CI of `stat_fn` over a resample of `values`.

    Returns (point, lo, hi); lo/hi are NaN when there is nothing to resample.
    A degenerate sample (all values identical, e.g. 48/48 survival) yields a
    zero-width interval -- correct for the percentile method, and the caller
    should widen it with `rule_of_three_pct` for 0/1 rates.
    """
    values = _finite(values)
    point = float(np.asarray(stat_fn(values))) if len(values) else float("nan")
    if len(values) < 2:
        return point, float("nan"), float("nan")
    idx = _rng(seed).integers(0, len(values), size=(n_boot, len(values)))
    draws = _apply(stat_fn, values[idx])
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 100 * alpha / 2)), \
        float(np.percentile(draws, 100 * (1 - alpha / 2)))


def rule_of_three_pct(n):
    """Upper bound (in points) on a rate whose observed count is 0 or n.

    The percentile bootstrap of an all-identical sample has zero width, which
    would draw 48/48 survival as a fact. The rule of three gives the 95% bound
    3/n, i.e. ~6 points at n=48 -- used as a floor on the CI half-width.
    """
    return 100.0 * 3.0 / n if n else float("nan")


def _aligned(rows_a, rows_b, value_key, pair_key):
    """(a_values, b_values) aligned on `pair_key`; keys present in exactly one
    run, or duplicated within a run, are dropped."""
    def index(rows):
        out = {}
        for row in rows:
            key = row.get(pair_key)
            if key is None:
                continue
            out.setdefault(key, []).append(row.get(value_key))
        return {k: v[0] for k, v in out.items() if len(v) == 1}

    ia, ib = index(rows_a), index(rows_b)
    shared = sorted(set(ia) & set(ib), key=str)
    a = np.array([ia[k] if ia[k] is not None else np.nan for k in shared], dtype=float)
    b = np.array([ib[k] if ib[k] is not None else np.nan for k in shared], dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def bootstrap_diff_ci(rows_a, rows_b, value_key, stat_fn=mean_stat,
                      pair_key=None, n_boot=DEFAULT_N_BOOT, alpha=DEFAULT_ALPHA,
                      seed=0, rate=False):
    """CI on stat(B) - stat(A). Positive delta = run B is higher.

    `pair_key` switches on paired resampling: the two runs are aligned on the
    key and the SAME resampled key set drives both arms, so route difficulty
    cancels. Falls back to independent resampling when fewer than 4 keys match.

    Returns dict(delta, lo, hi, paired, n_a, n_b, n_pairs, significant).
    `significant` is True only when the CI excludes zero. For rates the CI
    half-width is floored at the rule-of-three bound, so an arm sitting at 0% or
    100% never reports certainty.
    """
    va = _finite([r.get(value_key) for r in rows_a])
    vb = _finite([r.get(value_key) for r in rows_b])
    out = dict(delta=float("nan"), lo=float("nan"), hi=float("nan"),
               paired=False, n_a=int(len(va)), n_b=int(len(vb)), n_pairs=0,
               significant=False)
    if not len(va) or not len(vb):
        return out
    out["delta"] = float(np.asarray(stat_fn(vb))) - float(np.asarray(stat_fn(va)))

    pa = pb = None
    if pair_key is not None:
        pa, pb = _aligned(rows_a, rows_b, value_key, pair_key)
        if len(pa) < 4:
            pa = pb = None
    rng = _rng(seed)
    if pa is not None:
        out["paired"] = True
        out["n_pairs"] = int(len(pa))
        idx = rng.integers(0, len(pa), size=(n_boot, len(pa)))
        draws = _apply(stat_fn, pb[idx]) - _apply(stat_fn, pa[idx])
    else:
        ia = rng.integers(0, len(va), size=(n_boot, len(va)))
        ib = rng.integers(0, len(vb), size=(n_boot, len(vb)))
        draws = _apply(stat_fn, vb[ib]) - _apply(stat_fn, va[ia])
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return out
    lo = float(np.percentile(draws, 100 * alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    if rate:
        # a degenerate arm (0% or 100%) collapses its own resample to a point;
        # widen by the rule-of-three bound of the SMALLER arm rather than
        # claiming certainty
        floor = rule_of_three_pct(min(len(va), len(vb)))
        half = 0.5 * (hi - lo)
        if np.isfinite(floor) and half < floor:
            center = 0.5 * (hi + lo)
            lo, hi = center - floor, center + floor
    out.update(lo=lo, hi=hi, significant=bool(lo > 0.0 or hi < 0.0))
    return out


def pair_key(row):
    """Route identity of an episode. The runner sets route_seed = rep % bank, so
    (condition, rep) names the same route in every run of the same table."""
    return f"{row.get('condition', '')}|{row.get('rep', '')}"
