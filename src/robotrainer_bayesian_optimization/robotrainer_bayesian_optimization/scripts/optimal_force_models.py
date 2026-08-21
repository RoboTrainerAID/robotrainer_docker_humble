#!/usr/bin/env python3
"""Optimal disturbance force models for the RoboTrainer user study — PRODUCTION module.

Self-contained: numpy + scipy only, no pandas, no file I/O, no project imports. Drop it next to
your ROS 2 node and call one of the four functions after the assessment block.

    from optimal_force_models import (optimal_force_linear, optimal_force_quadratic,
                                      optimal_force_quadratic_maxforce,
                                      optimal_force_sensor_proxy, all_optimal_forces)

    tlx = {0.0: 70.0, 40.0: 60.0, 60.0: 50.0, 80.0: 40.0}      # force [N] -> raw TLX performance
    res = optimal_force_quadratic_maxforce(tlx, max_force_n=178.0)
    next_trial_force = res.optimal_force_n

Each function returns an OptimalForceResult; `optimal_force_n` is always a force that is safe to
command, i.e. inside [0, force_cap_n].

The four methods
----------------
1. LINEAR              y = a·F + b            a ≤ 0
2. QUADRATIC           y = a·F² + b·F + c     a ≤ 0, b ≤ 0
3. QUADRATIC_MAXFORCE  y = a·F² + b·F + c     a ≤ 0, b ≤ 0, y(F_max) ≤ 0
4. SENSOR_PROXY        frozen z-scored composite of three timeseries features — nothing is fitted

Methods 1-3 fit a NEW curve to THIS participant's four ratings; nothing is carried over from the
n=28 study except the model form. Method 4 applies 11 constants frozen from the n=28 study
(PROXY_* below) and fits nothing.

x* is the force at which the participant's performance curve crosses TARGET_PERFORMANCE_LEVEL on
the way DOWN. Where no such crossing exists inside the admissible range, the prescription is
clipped to the nearest edge and flagged — see `_clip_to_range`.

Assumed input orientation: raw NASA-TLX "eigene Leistung", 0-100, HIGHER = BETTER, so the curve
FALLS with rising force. If your questionnaire stores the inverted subscale, convert before
calling.

Force direction: the assessment tasks apply the disturbance to the LEFT, so the participant pushes
RIGHT and `max_force_n` must be the RoboTrainer *Right* max-push measurement. See MAX_FORCE_NOTE.

Runtime requirements
--------------------
Python >= 3.8, numpy, scipy. Nothing else — no pandas, no matplotlib, no file I/O.

Verified on ROS 2 Humble (Python 3.10.12, numpy 1.21.5, scipy 1.8.0), which is what the
`robotrainer_humble` image ships. A bare `ros:humble-ros-base` image has numpy but NOT scipy, so
there:

    apt-get install -y python3-scipy        # or: pip install scipy

scipy is needed for `lsq_linear`, which does the bound-constrained least squares that all three
questionnaire models rely on. Method 4 (the sensor proxy) is pure arithmetic and needs neither.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import lsq_linear

# ============================================================================================
# GENERAL CONSTANTS AND ASSUMPTIONS — tune here, nowhere else
# ============================================================================================

# --- the target level that defines x* -------------------------------------------------------
TARGET_PERFORMANCE_LEVEL = 48.5   # raw TLX "eigene Leistung" points; the literature target
RATING_MIN, RATING_MAX = 0.0, 100.0

# --- the assessment design ------------------------------------------------------------------
ASSESSMENT_FORCE_LEVELS = (0.0, 40.0, 60.0, 80.0)   # N, straight line, one scenario per level
MIN_FORCE_LEVELS = 3              # refuse to fit a quadratic on fewer points than parameters + 1

# --- device and participant limits ----------------------------------------------------------
DEVICE_FORCE_MIN_N = 0.0          # technical lower limit
DEVICE_FORCE_MAX_N = 300.0        # technical upper limit
# The prescription is additionally capped at the participant's own max push force: a force they
# cannot resist produces a failed run and a floored rating, which measures nothing. Set the factor
# to e.g. 0.9 to keep headroom below their true maximum.
MAX_FORCE_SAFETY_FACTOR = 1.0

# --- solver behaviour -----------------------------------------------------------------------
# A fit whose slope is pinned at the a <= 0 bound runs parallel to the target level, and
# (LEVEL - b)/a then explodes to ~1e13. Past this cutoff we call it what it is: no crossing.
PARALLEL_CUTOFF = 1e6
# Quadratic-MaxForce restricts the curve's END POINT: it must have reached zero performance by the
# participant's own maximum push force. False = the loose inequality y(F_max) <= 0 (recommended,
# and what the n=28 constants were derived with); True = the hard equality y(F_max) == 0.
MAXFORCE_HARD = False

MAX_FORCE_NOTE = ("The disturbance acts orthogonal to the path. Disturbance LEFT (the assessment "
                  "scenario, force_direction 2) => the participant pushes RIGHT => use the "
                  "RoboTrainer Right max-push value. Disturbance RIGHT => RoboTrainer Left.")

# --- Method 4: the 11 frozen constants ------------------------------------------------------
# Derived once from the n=28 study over the 26 users who answered the TLX, paths 4-7. Per user the
# feature is first averaged over the four assessment levels, THEN z-scored across users (that
# order matters). Signs are the direction of each feature's correlation with the
# Quadratic-MaxForce x* over the 24 users who had a valid one; alpha/beta moment-match the
# unitless composite onto that x* distribution. NEVER recompute these per participant.
PROXY_FEATURES = ('path_deviation_front_std',
                  'user_force_y_impulse',
                  'robot_vel_lin_mag_total_distance')

PROXY_FEATURE_MEANS = {                       # constants 4-6
    'path_deviation_front_std':          0.08251137791782269,
    'user_force_y_impulse':            -20.865736592950608,
    'robot_vel_lin_mag_total_distance':  8.399232967250104,
}
PROXY_FEATURE_SDS = {                         # constants 7-9  (sample SD, ddof=1)
    'path_deviation_front_std':          0.027198163163713453,
    'user_force_y_impulse':             40.337643210052526,
    'robot_vel_lin_mag_total_distance':  0.3952226819368701,
}
PROXY_FEATURE_SIGNS = {                       # constants 1-3
    'path_deviation_front_std':         -1.0,   # less deviation variability -> higher x*
    'user_force_y_impulse':             +1.0,
    'robot_vel_lin_mag_total_distance': -1.0,   # less distance travelled     -> higher x*
}
PROXY_ALPHA_N = 16.407182007584062            # constant 10: newtons per composite SD
PROXY_BETA_N = 70.86517585558724              # constant 11: offset [N]

# --- method identifiers ---------------------------------------------------------------------
LINEAR = 'Linear'
QUADRATIC = 'Quadratic'
QUADRATIC_MAXFORCE = 'Quadratic-MaxForce'
SENSOR_PROXY = 'Sensor-Proxy'
METHODS = (LINEAR, QUADRATIC, QUADRATIC_MAXFORCE, SENSOR_PROXY)

# --- result flags ---------------------------------------------------------------------------
FLAG_OK = 'ok'                       # crossing found analytically inside the admissible range
FLAG_OK_NUMERIC = 'ok_numeric'       # crossing found by bisection (root solver was degenerate)
FLAG_CLIPPED_LOW = 'clipped_low'     # curve already at/below target at 0 N  -> prescribe 0 N
FLAG_CLIPPED_HIGH = 'clipped_high'   # curve still above target at the cap   -> prescribe the cap

ForcePairs = Union[Mapping[float, float], Iterable[Sequence[float]]]


# ============================================================================================
# RESULT TYPE
# ============================================================================================

@dataclass(frozen=True)
class OptimalForceResult:
    """One method's prescription for one participant.

    optimal_force_n     the force to command in the next trial. ALWAYS inside [0, force_cap_n].
    flag                one of FLAG_*; anything other than 'ok'/'ok_numeric' is a censored
                        observation and must be recorded as such.
    raw_crossing_n      the unclipped analytic crossing, NaN when the curve never crosses. Log
                        this — it is what the offline analysis compares against.
    within_device_window  True when raw_crossing_n lies inside the technical 0-300 N window. This
                        is the validity definition used by method_comparison.py.
    coefficients        (a, b) for Linear, (a, b, c) for the quadratics, () for the proxy.
    fit_rmse            RMSE of the fit over the supplied force levels; NaN for the proxy.
    details             method-specific extras (composite value, z-scores, reason strings, ...).
    """
    method: str
    optimal_force_n: float
    flag: str
    force_cap_n: float
    raw_crossing_n: float = float('nan')
    within_device_window: bool = False
    coefficients: Tuple[float, ...] = ()
    fit_rmse: float = float('nan')
    max_force_n: float = float('nan')
    details: Dict[str, object] = field(default_factory=dict)

    @property
    def is_clipped(self) -> bool:
        return self.flag in (FLAG_CLIPPED_LOW, FLAG_CLIPPED_HIGH)

    def __str__(self) -> str:
        raw = 'none' if not math.isfinite(self.raw_crossing_n) else f'{self.raw_crossing_n:.2f} N'
        return (f'{self.method:<19} x* = {self.optimal_force_n:6.2f} N  [{self.flag}]  '
                f'raw crossing {raw}, cap {self.force_cap_n:.1f} N')


# ============================================================================================
# INPUT NORMALISATION
# ============================================================================================

def _as_force_value_arrays(pairs: ForcePairs, what: str) -> Tuple[np.ndarray, np.ndarray]:
    """Accept {force: value} or [(force, value), ...] and return sorted float arrays."""
    if isinstance(pairs, Mapping):
        items = list(pairs.items())
    else:
        items = []
        for item in pairs:
            if len(item) != 2:
                raise ValueError(f'{what}: expected (force, value) pairs, got {item!r}')
            items.append(tuple(item))
    if not items:
        raise ValueError(f'{what}: no data supplied')
    forces = np.array([float(f) for f, _ in items], float)
    values = np.array([float(v) for _, v in items], float)
    order = np.argsort(forces)
    forces, values = forces[order], values[order]
    if len(np.unique(forces)) != len(forces):
        raise ValueError(f'{what}: duplicate force levels {forces.tolist()}')
    if not np.all(np.isfinite(forces)) or not np.all(np.isfinite(values)):
        raise ValueError(f'{what}: non-finite entries in {list(zip(forces, values))}')
    return forces, values


def _check_ratings(forces: np.ndarray, ratings: np.ndarray, n_params: int) -> None:
    if len(forces) < max(n_params, MIN_FORCE_LEVELS):
        raise ValueError(f'need at least {max(n_params, MIN_FORCE_LEVELS)} force levels to fit '
                         f'{n_params} parameters, got {len(forces)}')
    bad = [(f, r) for f, r in zip(forces, ratings) if not RATING_MIN <= r <= RATING_MAX]
    if bad:
        raise ValueError(f'TLX performance ratings must lie in [{RATING_MIN:g}, {RATING_MAX:g}] '
                         f'(raw scale, higher = better); offending pairs: {bad}')
    missing = [f for f in ASSESSMENT_FORCE_LEVELS if not np.any(np.isclose(forces, f))]
    if missing:
        # not fatal: the fit works on whatever levels exist, but the caller should know
        import warnings
        warnings.warn(f'assessment levels {missing} N are missing; fitting on {forces.tolist()}',
                      RuntimeWarning, stacklevel=3)


def features_from_rows(rows: Iterable[Mapping[str, object]],
                       force_key: str = 'force_strength',
                       features: Sequence[str] = PROXY_FEATURES) -> Dict[str, Dict[float, float]]:
    """Convenience: build the sensor-proxy input from CSV-style row dicts.

    Each row is one assessment task and must carry `force_key` plus the feature columns, exactly as
    data/timeseries_features.csv stores them.

        rows = list(csv.DictReader(open('this_user_features.csv')))
        feats = features_from_rows(rows, force_key='force_strength')
        optimal_force_sensor_proxy(feats, max_force_n=178.0)
    """
    out: Dict[str, Dict[float, float]] = {f: {} for f in features}
    for row in rows:
        force = float(row[force_key])
        for name in features:
            if name not in row or row[name] in (None, ''):
                raise ValueError(f'row at {force:g} N is missing feature {name!r}')
            out[name][force] = float(row[name])
    return out


def _mean_feature(per_force: ForcePairs, name: str) -> float:
    """Average one feature over the assessment levels — the same aggregation the constants used."""
    forces, values = _as_force_value_arrays(per_force, f'feature {name!r}')
    return float(np.mean(values))


# ============================================================================================
# FITTING PRIMITIVES  (identical to optimal_difficulty/method_comparison.py)
# ============================================================================================

def _fit_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y = a·x + b with a <= 0, so the crossing is always reached on the way down."""
    return lsq_linear(np.column_stack([x, np.ones_like(x)]), y,
                      bounds=([-np.inf, -np.inf], [0.0, np.inf])).x


def _fit_quadratic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y = a·x² + b·x + c with a <= 0 and b <= 0.

    Together those put the vertex -b/(2a) at or before 0 N, so the curve falls monotonically over
    the whole positive range and its slope keeps steepening (y' <= 0, y'' <= 0).
    """
    return lsq_linear(np.column_stack([x ** 2, x, np.ones_like(x)]), y,
                      bounds=([-np.inf, -np.inf, -np.inf], [0.0, 0.0, np.inf])).x


def _bounded_lsq(design: np.ndarray, y: np.ndarray, lo, hi) -> np.ndarray:
    """lsq_linear with per-column scaling.

    The end-point design carries an x² - F_max² column reaching ~6e4 against a constant column of
    1, and that spread alone is enough to make the solver return NaN. Scaling each column to unit
    magnitude — and the matching bound with it, which preserves the sign constraints — fixes it.
    """
    scale = np.max(np.abs(design), axis=0)
    scale[scale == 0] = 1.0
    sol = lsq_linear(design / scale, y,
                     bounds=(np.asarray(lo, float) * scale, np.asarray(hi, float) * scale))
    return sol.x / scale


def _fit_quadratic_maxforce(x: np.ndarray, y: np.ndarray, fmax: float,
                            hard: bool = MAXFORCE_HARD) -> np.ndarray:
    """Quadratic that must be exhausted by the participant's own max push force: y(F_max) <= 0.

    An END-POINT restriction, not a curvature one: the curve is fitted freely subject to the usual
    a <= 0, b <= 0 and is only pulled where it would otherwise still show positive performance at a
    force the participant cannot even produce.

    Substituting c = -(a·F² + b·F) - s with slack s >= 0 turns the constraint into a plain box
    bound, so this stays a single bounded linear least-squares solve:
        y = a·(x² - F²) + b·(x - F) - s      and      y(F) = -s <= 0.
    """
    if not (np.isfinite(fmax) and fmax > 0):
        raise ValueError(f'Quadratic-MaxForce needs a positive finite max_force_n, got {fmax!r}. '
                         + MAX_FORCE_NOTE)
    f = float(fmax)
    if hard:
        a, b = _bounded_lsq(np.column_stack([x ** 2 - f ** 2, x - f]), y,
                            [-np.inf, -np.inf], [0.0, 0.0])
        slack = 0.0
    else:
        a, b, slack = _bounded_lsq(
            np.column_stack([x ** 2 - f ** 2, x - f, -np.ones_like(x)]), y,
            [-np.inf, -np.inf, 0.0], [0.0, 0.0, np.inf])
    return np.array([a, b, -(a * f ** 2 + b * f) - slack])


def curve_value(coeffs: Sequence[float], force) -> np.ndarray:
    """Evaluate a fitted performance curve at one or more forces."""
    return np.polyval(np.asarray(coeffs, float), np.asarray(force, float))


def _crossing(coeffs: Sequence[float]) -> float:
    """Force at which the fit crosses TARGET_PERFORMANCE_LEVEL downward; NaN if it never does.

    A line has a single root, taken as-is. For a quadratic only roots where the curve passes the
    level on the way DOWN qualify (2a·x + b < 0); of those the largest is used.
    """
    c = np.asarray(coeffs, float)
    if c.size == 0 or not np.all(np.isfinite(c)):
        return float('nan')
    if c.size == 2:
        a, b = c
        return float((TARGET_PERFORMANCE_LEVEL - b) / a) if a != 0 else float('nan')
    a, b, const = c
    roots = np.roots([a, b, const - TARGET_PERFORMANCE_LEVEL])
    ok = [r.real for r in roots
          if abs(r.imag) < 1e-9 and np.sign(2 * a * r.real + b) == -1]
    return float(max(ok)) if ok else float('nan')


def _fit_rmse(coeffs: Sequence[float], x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - curve_value(coeffs, x)) ** 2)))


# ============================================================================================
# RANGE HANDLING
# ============================================================================================

def force_cap(max_force_n: float = float('nan')) -> float:
    """The participant-specific upper limit: min(device limit, safety factor · own max push).

    Called with no argument (or NaN) it degrades to the pure technical limit — acceptable for the
    two methods that do not need F_max, but the safe cap is strongly preferred.
    """
    cap = DEVICE_FORCE_MAX_N
    if max_force_n is not None and np.isfinite(max_force_n) and max_force_n > 0:
        cap = min(cap, MAX_FORCE_SAFETY_FACTOR * float(max_force_n))
    return float(max(cap, DEVICE_FORCE_MIN_N))


def _in_device_window(value: float) -> bool:
    """method_comparison.py's validity test: finite, not a parallel-fit artefact, inside 0-300 N."""
    return bool(np.isfinite(value) and abs(value) <= PARALLEL_CUTOFF
                and DEVICE_FORCE_MIN_N <= value <= DEVICE_FORCE_MAX_N)


def _bisect_crossing(coeffs: Sequence[float], lo: float, hi: float,
                     tol: float = 1e-6) -> float:
    """Smallest force in [lo, hi] where the curve is at or below the target level."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(curve_value(coeffs, mid)) > TARGET_PERFORMANCE_LEVEL:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float(hi)


def _clip_to_range(raw: float, coeffs: Sequence[float], cap: float) -> Tuple[float, str]:
    """Turn a possibly-missing crossing into a commandable force plus a flag.

    Deliberately does NOT trust the root solver's failure mode: a quadratic whose curvature
    collapses to ~1e-16 yields a root at ~1e16 and looks like "no crossing" even though the curve
    does cross, below 0 N. The side is decided from the fitted curve at the range edges instead,
    with a bisection fallback for the degenerate cases.
    """
    if np.isfinite(raw) and abs(raw) <= PARALLEL_CUTOFF and DEVICE_FORCE_MIN_N <= raw <= cap:
        return float(raw), FLAG_OK
    if float(curve_value(coeffs, DEVICE_FORCE_MIN_N)) <= TARGET_PERFORMANCE_LEVEL:
        return DEVICE_FORCE_MIN_N, FLAG_CLIPPED_LOW          # already at/below target at 0 N
    if float(curve_value(coeffs, cap)) >= TARGET_PERFORMANCE_LEVEL:
        return cap, FLAG_CLIPPED_HIGH                        # still above target at the cap
    return _bisect_crossing(coeffs, DEVICE_FORCE_MIN_N, cap), FLAG_OK_NUMERIC


def _clip_plain(value: float, cap: float) -> Tuple[float, str]:
    """Clip a value that comes from no curve at all (the sensor proxy)."""
    if not np.isfinite(value):
        raise ValueError('sensor proxy produced a non-finite force')
    if value < DEVICE_FORCE_MIN_N:
        return DEVICE_FORCE_MIN_N, FLAG_CLIPPED_LOW
    if value > cap:
        return cap, FLAG_CLIPPED_HIGH
    return float(value), FLAG_OK


# ============================================================================================
# THE FOUR PUBLIC MODELS
# ============================================================================================

def _questionnaire_result(method: str, coeffs: np.ndarray, x: np.ndarray, y: np.ndarray,
                          max_force_n: float) -> OptimalForceResult:
    cap = force_cap(max_force_n)
    raw = _crossing(coeffs)
    value, flag = _clip_to_range(raw, coeffs, cap)
    return OptimalForceResult(
        method=method,
        optimal_force_n=value,
        flag=flag,
        force_cap_n=cap,
        raw_crossing_n=raw,
        within_device_window=_in_device_window(raw),
        coefficients=tuple(float(c) for c in coeffs),
        fit_rmse=_fit_rmse(coeffs, x, y),
        max_force_n=float(max_force_n) if max_force_n is not None else float('nan'),
        details={'force_levels_n': x.tolist(), 'ratings': y.tolist(),
                 'target_level': TARGET_PERFORMANCE_LEVEL},
    )


def optimal_force_linear(tlx_performance: ForcePairs,
                         max_force_n: float = float('nan')) -> OptimalForceResult:
    """METHOD 1 — straight line, y = a·F + b with a <= 0.

    tlx_performance : {force_n: raw performance rating} or [(force_n, rating), ...]
    max_force_n     : optional; used ONLY to cap the prescription at the participant's own limit.

    Falling, no acceleration. This is the method that fails most often: a participant who rates
    performance flat across all four levels drives `a` onto the bound, producing no crossing at
    all, and is then clipped and flagged.
    """
    x, y = _as_force_value_arrays(tlx_performance, 'tlx_performance')
    _check_ratings(x, y, n_params=2)
    return _questionnaire_result(LINEAR, _fit_linear(x, y), x, y, max_force_n)


def optimal_force_quadratic(tlx_performance: ForcePairs,
                            max_force_n: float = float('nan')) -> OptimalForceResult:
    """METHOD 2 — quadratic, y = a·F² + b·F + c with a <= 0 and b <= 0.

    tlx_performance : {force_n: raw performance rating} or [(force_n, rating), ...]
    max_force_n     : optional; used ONLY to cap the prescription.

    Monotone falling with steepening slope, but curvature is otherwise free, so for roughly half of
    participants it flattens toward a straight line and inherits Method 1's instability.
    """
    x, y = _as_force_value_arrays(tlx_performance, 'tlx_performance')
    _check_ratings(x, y, n_params=3)
    return _questionnaire_result(QUADRATIC, _fit_quadratic(x, y), x, y, max_force_n)


def optimal_force_quadratic_maxforce(tlx_performance: ForcePairs,
                                     max_force_n: float) -> OptimalForceResult:
    """METHOD 3 — quadratic plus the end-point restriction y(F_max) <= 0.  RECOMMENDED.

    tlx_performance : {force_n: raw performance rating} or [(force_n, rating), ...]
    max_force_n     : REQUIRED. The participant's max push force on the side opposing the
                      disturbance. For the assessment scenario (disturbance left) that is the
                      RoboTrainer *Right* measurement. See MAX_FORCE_NOTE.

    The F_max anchor is what rescues the flat raters and roughly halves the spread of x*; on n=28
    it was valid for 24/26 participants and was three times more stable under rating noise than
    Methods 1-2. By construction it can never prescribe more force than the participant can
    produce.
    """
    x, y = _as_force_value_arrays(tlx_performance, 'tlx_performance')
    _check_ratings(x, y, n_params=3)
    coeffs = _fit_quadratic_maxforce(x, y, max_force_n)
    return _questionnaire_result(QUADRATIC_MAXFORCE, coeffs, x, y, max_force_n)


def optimal_force_sensor_proxy(timeseries_features: Mapping[str, ForcePairs],
                               max_force_n: float = float('nan')) -> OptimalForceResult:
    """METHOD 4 — frozen sensor composite. No questionnaire, no fitting.

    timeseries_features : {feature_name: {force_n: value}} or {feature_name: [(force_n, value)]},
                          one entry per name in PROXY_FEATURES, values in the same units as
                          data/timeseries_features.csv. Use `features_from_rows` to build this from
                          CSV rows.
    max_force_n         : optional; used ONLY to cap the prescription.

    Each feature is averaged over the assessment levels, z-scored against the frozen cohort mean
    and SD, signed, and summed into the unitless composite C; then x* = alpha·C + beta. Every
    constant is frozen (PROXY_*) — nothing here is estimated from this participant.
    """
    missing = [f for f in PROXY_FEATURES if f not in timeseries_features]
    if missing:
        raise ValueError(f'sensor proxy needs features {list(PROXY_FEATURES)}; missing {missing}')

    means, z_scores, contributions = {}, {}, {}
    composite = 0.0
    for name in PROXY_FEATURES:
        value = _mean_feature(timeseries_features[name], name)
        z = (value - PROXY_FEATURE_MEANS[name]) / PROXY_FEATURE_SDS[name]
        signed = PROXY_FEATURE_SIGNS[name] * z
        means[name], z_scores[name], contributions[name] = value, z, signed
        composite += signed

    raw = PROXY_ALPHA_N * composite + PROXY_BETA_N
    cap = force_cap(max_force_n)
    value, flag = _clip_plain(raw, cap)
    return OptimalForceResult(
        method=SENSOR_PROXY,
        optimal_force_n=value,
        flag=flag,
        force_cap_n=cap,
        raw_crossing_n=raw,          # for the proxy this is the unclipped prediction
        within_device_window=_in_device_window(raw),
        coefficients=(),
        fit_rmse=float('nan'),       # nothing is fitted, so there is no fit error
        max_force_n=float(max_force_n) if max_force_n is not None else float('nan'),
        details={'feature_means': means, 'z_scores': z_scores,
                 'signed_contributions': contributions, 'composite_unitless': composite,
                 'alpha_N': PROXY_ALPHA_N, 'beta_N': PROXY_BETA_N},
    )


def all_optimal_forces(tlx_performance: ForcePairs,
                       timeseries_features: Mapping[str, ForcePairs],
                       max_force_n: float) -> Dict[str, OptimalForceResult]:
    """Run all four methods on one participant and return {method_name: result}.

    Call this once at the end of the assessment block (5b); the four forces are then presented in
    randomised order in block 5d.
    """
    return {
        LINEAR:             optimal_force_linear(tlx_performance, max_force_n),
        QUADRATIC:          optimal_force_quadratic(tlx_performance, max_force_n),
        QUADRATIC_MAXFORCE: optimal_force_quadratic_maxforce(tlx_performance, max_force_n),
        SENSOR_PROXY:       optimal_force_sensor_proxy(timeseries_features, max_force_n),
    }


__all__ = [
    'LINEAR', 'QUADRATIC', 'QUADRATIC_MAXFORCE', 'SENSOR_PROXY', 'METHODS',
    'FLAG_OK', 'FLAG_OK_NUMERIC', 'FLAG_CLIPPED_LOW', 'FLAG_CLIPPED_HIGH',
    'OptimalForceResult', 'optimal_force_linear', 'optimal_force_quadratic',
    'optimal_force_quadratic_maxforce', 'optimal_force_sensor_proxy', 'all_optimal_forces',
    'features_from_rows', 'force_cap', 'curve_value',
    'TARGET_PERFORMANCE_LEVEL', 'ASSESSMENT_FORCE_LEVELS',
    'DEVICE_FORCE_MIN_N', 'DEVICE_FORCE_MAX_N', 'MAX_FORCE_SAFETY_FACTOR',
    'PROXY_FEATURES', 'MAX_FORCE_NOTE',
]
