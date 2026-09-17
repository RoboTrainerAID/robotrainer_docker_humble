#!/usr/bin/env python3
"""ROS 1 bag -> the 3 sensor features Method 4 needs. Nothing is written to disk in between.

PRODUCTION module. numpy + rosbags only — no pandas, no project imports, no intermediate files.

    from sensor_feature_extraction import extract_features_from_bag, extract_user_features_from_bags
    from optimal_force_models import optimal_force_sensor_proxy

    feats = extract_features_from_bag('KATE_AA_U010_16_....bag')       # one assessment task

    features = extract_user_features_from_bags({0.0: bag0, 40.0: bag40,    # all four tasks
                                                60.0: bag60, 80.0: bag80})
    x4 = optimal_force_sensor_proxy(features, max_force_n=96.82)

`extract_user_features_from_bags` returns exactly the shape `optimal_force_sensor_proxy` takes, so
the two compose with no glue.

Why this exists
---------------
The research pipeline (`dataset_generation/`) turns 26 raw timeseries into 85 features for 560
paths, via an exported numpy dataset. Method 4 needs exactly THREE of those features, from FOUR raw
topics, on FOUR assessment tasks. This module is that subset read straight from the bag, and it
reproduces the original bit for bit — verified on all 560 n=28 paths and on the reference bag
(run this file to re-check).

What is kept from the full pipeline, and what is dropped
-------------------------------------------------------
KEPT (these change the numbers):
  * motion-start trimming — drop every sample before the robot first exceeds 0.01 m/s, using
    robot_vel_x as the reference clock, and do NOT shift relative timestamps
    (`TimeseriesPreprocessor._trim_path_data`)
  * the derived magnitude robot_vel_lin_mag = sqrt(vel_x^2 + vel_y^2), computed AFTER trimming
    (`TimeseriesDerivedTS._calculate_magnitudes`)
  * dt for the integral features = mean(diff(RAW timestamps)), not the nominal sample rate
    (`TimeseriesFeatureExtractor._compute_ts_features`)

DROPPED (verified not to touch these four series on the n=28 data):
  * HRV/PPG reformatting and zero-filtering — only apply to heart_rate / hrv / ppi / ppg_*
  * validation + imputation — fires only for missing or empty series; all 560 paths have all four
    series present, non-empty and correctly shaped
  * zero-disturbance imputation — only writes disturbance_force_*
  * the other 22 raw series, the other 5 derived series, and 82 of the 85 features

Requirements
------------
numpy and `rosbags` (`pip install rosbags`, Python >= 3.10; pulls apsw, lz4, ruamel.yaml,
zstandard). `rosbags` is pure Python and reads ROS 1 bags with NO ROS installation, so bags recorded
under Melodic load unchanged under ROS 2 Humble.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

# ============================================================================================
# CONSTANTS — all from dataset_generation/config.py
# ============================================================================================

# Series layout used throughout: (N, 3) with these columns.
COL_RAW_TS, COL_REL_TS, COL_VALUE = 0, 1, 2

# Motion start: the first sample whose |robot_vel_x| exceeds this is t0; everything before is cut.
MOTION_START_THRESHOLD = 0.01          # m/s

# The four raw series the three features are built from.
RAW_SERIES_NEEDED = ('path_deviation_front', 'user_force_y', 'robot_vel_x', 'robot_vel_y')

# Every one of these is in config.TIMESERIES_TRIMMED_BY_MOTION_START, so all four get trimmed.
TRIM_BY_MOTION_START = set(RAW_SERIES_NEEDED)

# The three features, in the order optimal_force_models.PROXY_FEATURES expects them.
PROXY_FEATURE_NAMES = ('path_deviation_front_std',
                       'user_force_y_impulse',
                       'robot_vel_lin_mag_total_distance')

# Which topic and field each raw series comes from. Verified against the numpy export for
# U010/path_16: message counts and every single value match bit for bit.
BAG_TOPICS = {
    'user_force_y': {
        'topic': '/base/output_data',                                    # geometry_msgs/WrenchStamped
        'fields': {'user_force_y': ('wrench', 'force', 'y')},
    },
    'path_deviation_front': {
        'topic': '/robotrainer_deviation/robotrainer_deviation',         # robotrainer_deviation/RobotrainerUserDeviation
        'fields': {'path_deviation_front': ('front',)},                  # msg is: float64 front/left/right
    },
    'robot_vel': {
        'topic': '/base/fts_adaptive_force_controller/debug/velocity_output',   # geometry_msgs/TwistStamped
        'fields': {'robot_vel_x': ('twist', 'linear', 'x'),
                   'robot_vel_y': ('twist', 'linear', 'y')},
    },
}

NS_PER_S = 1_000_000_000


# ============================================================================================
# READING THE BAG
# ============================================================================================

def _ros_time_to_sec(stamp_ns: int) -> float:
    """ROS 1 `Time.to_sec()`: seconds and nanoseconds combined SEPARATELY in float64.

    Not `stamp_ns / 1e9`. The two differ by up to one ULP (2.4e-7 s at a 2025 unix timestamp) and
    that is enough to change `dt = mean(diff(t))` in the last digits, so the integral features would
    stop being bit-exact against the numpy export. 761 of 2877 timestamps differ on the reference
    bag if you get this wrong.
    """
    return (stamp_ns // NS_PER_S) + (stamp_ns % NS_PER_S) / 1e9


def _nested(msg, field_path: Sequence[str]) -> float:
    for name in field_path:
        msg = getattr(msg, name)
    return float(msg)


def read_bag_series(bag_path: Path) -> Dict[str, np.ndarray]:
    """Read ONLY the topics the three features need out of a ROS 1 bag.

    Returns {series_name: (N, 3) array} with columns [raw_ts, rel_ts, value].

    `rosbags` reads the chunk index and seeks, so this touches only the few MB belonging to these
    three topics — a 6.8 GB bag full of camera and point-cloud data loads in well under a second.

    The custom `robotrainer_deviation/msg/RobotrainerUserDeviation` type is not in any typestore,
    but ROS 1 bags embed the message definition in the connection record, so it is registered on the
    fly from the bag itself. No .msg files and no ROS installation needed.

    `raw_ts` (column 0) is the BAG RECORD timestamp, not `header.stamp` — that is what the numpy
    export used. `rel_ts` (column 1) is filled as `raw_ts - bag_start`; it is never read by any of
    the three features, which use columns 0 and 2 only.
    """
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    bag_path = Path(bag_path)
    if not bag_path.exists():
        raise FileNotFoundError(f'bag not found: {bag_path}')

    wanted = {spec['topic']: spec['fields'] for spec in BAG_TOPICS.values()}
    typestore = get_typestore(Stores.ROS1_NOETIC)
    collected: Dict[str, list] = {name: [] for spec in BAG_TOPICS.values() for name in spec['fields']}

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic in wanted]
        missing = set(wanted) - {c.topic for c in connections}
        if missing:
            raise KeyError(f'{bag_path.name}: required topics missing from the bag: {sorted(missing)}')

        for conn in connections:                    # register custom types from the bag itself
            if conn.msgtype not in typestore.types:
                msgdef = conn.msgdef
                text = getattr(msgdef, 'data', msgdef)       # rosbags >=0.11 wraps it in an object
                typestore.register(get_types_from_msg(text, conn.msgtype))

        start_s = _ros_time_to_sec(reader.start_time)
        for conn, stamp_ns, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            raw_ts = _ros_time_to_sec(stamp_ns)
            for series_name, field_path in wanted[conn.topic].items():
                collected[series_name].append((raw_ts, raw_ts - start_s, _nested(msg, field_path)))

    out = {}
    for name, rows in collected.items():
        if not rows:
            raise ValueError(f'{bag_path.name}: no messages for {name}')
        out[name] = np.asarray(rows, float)
    return out


# ============================================================================================
# PREPROCESSING
# ============================================================================================

def motion_start_time(robot_vel_x: np.ndarray) -> Optional[float]:
    """Raw timestamp of the first sample exceeding MOTION_START_THRESHOLD, or None.

    None means "never moved", and the original pipeline then does NOT trim at all — it returns
    early, leaving every series untouched. Reproduced in `trim_to_motion_start`.
    """
    moving = np.abs(robot_vel_x[:, COL_VALUE]) > MOTION_START_THRESHOLD
    if not np.any(moving):
        return None
    return float(robot_vel_x[int(np.argmax(moving)), COL_RAW_TS])


def trim_to_motion_start(series: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Drop every sample before motion start. Relative timestamps are deliberately NOT shifted.

    Two quirks of the original that are load-bearing:
      * no robot_vel_x, or no sample above the threshold  -> nothing is trimmed at all;
      * if trimming would empty a series, that series keeps its UNTRIMMED data
        (`if trimmed_arr.size > 0` in the original).
    """
    if 'robot_vel_x' not in series:
        return dict(series)
    t0 = motion_start_time(series['robot_vel_x'])
    if t0 is None:
        return dict(series)

    out = dict(series)
    for name in TRIM_BY_MOTION_START:
        if name not in out:
            continue
        trimmed = out[name][out[name][:, COL_RAW_TS] >= t0]
        if trimmed.size > 0:
            out[name] = trimmed
    return out


def robot_vel_lin_mag(series: Mapping[str, np.ndarray]) -> Optional[np.ndarray]:
    """Derived series sqrt(vel_x^2 + vel_y^2), carrying robot_vel_x's timestamps.

    The components are assumed already synchronised; if they differ in length the original
    truncates both to the shorter one, so this does too.
    """
    vx, vy = series.get('robot_vel_x'), series.get('robot_vel_y')
    if vx is None or vy is None:
        return None
    n = min(len(vx), len(vy))
    magnitude = np.sqrt(vx[:n, COL_VALUE] ** 2 + vy[:n, COL_VALUE] ** 2)
    return np.column_stack((vx[:n, :2], magnitude))


# ============================================================================================
# THE THREE FEATURES
# ============================================================================================

def _dt(arr: np.ndarray) -> float:
    """Mean sample spacing from the RAW timestamps — what the integral features are scaled by."""
    return float(np.mean(np.diff(arr[:, COL_RAW_TS]))) if arr.shape[0] > 1 else 0.0


def feature_std(arr: np.ndarray) -> float:
    """`np.std(values)` — population SD, ddof = 0."""
    return float(np.std(arr[:, COL_VALUE]))


def feature_integral(arr: np.ndarray) -> float:
    """Signed integral `sum(values) * dt`. Called `impulse` for force, `total_distance` for speed.

    A rectangle rule, not a trapezoid, and 0.0 when there is only one sample — both as in the
    original, which is what makes the numbers reproduce exactly.
    """
    if arr.shape[0] <= 1:
        return 0.0
    return float(np.sum(arr[:, COL_VALUE]) * _dt(arr))


def features_from_series(raw: Mapping[str, np.ndarray]) -> Dict[str, float]:
    """trim -> derive -> three features. The core, kept separate so it can be tested on any source.

    `raw` must hold the four series of RAW_SERIES_NEEDED as (N, 3) arrays.
    """
    missing = [n for n in RAW_SERIES_NEEDED if n not in raw]
    if missing:
        raise KeyError(f'missing raw series: {missing}')

    series = trim_to_motion_start(raw)
    vel_mag = robot_vel_lin_mag(series)

    return {
        'path_deviation_front_std':         feature_std(series['path_deviation_front']),
        'user_force_y_impulse':             feature_integral(series['user_force_y']),
        'robot_vel_lin_mag_total_distance': feature_integral(vel_mag),
    }


# ============================================================================================
# PUBLIC API
# ============================================================================================

def extract_features_from_bag(bag_path: Path) -> Dict[str, float]:
    """THE single call: ROS 1 bag of one assessment task -> the three proxy features.

        feats = extract_features_from_bag('KATE_AA_U010_16_....bag')
        # {'path_deviation_front_std': ..., 'user_force_y_impulse': ...,
        #  'robot_vel_lin_mag_total_distance': ...}
    """
    return features_from_series(read_bag_series(bag_path))


def extract_user_features_from_bags(force_to_bag: Mapping[float, Path]
                                    ) -> Dict[str, Dict[float, float]]:
    """All four assessment tasks of ONE participant -> the input Method 4 takes.

        features = extract_user_features_from_bags({0.0: bag0, 40.0: bag40,
                                                    60.0: bag60, 80.0: bag80})
        x4 = optimal_force_sensor_proxy(features, max_force_n=...)

    The force levels are the keys, so the caller decides which bag was which task — the bag file
    name is not parsed for it.
    """
    out: Dict[str, Dict[float, float]] = {name: {} for name in PROXY_FEATURE_NAMES}
    for force_n, bag_path in sorted(force_to_bag.items()):
        for name, value in extract_features_from_bag(bag_path).items():
            out[name][float(force_n)] = value
    return out


__all__ = ['COL_RAW_TS', 'COL_REL_TS', 'COL_VALUE', 'MOTION_START_THRESHOLD',
           'RAW_SERIES_NEEDED', 'PROXY_FEATURE_NAMES', 'BAG_TOPICS',
           'read_bag_series', 'motion_start_time', 'trim_to_motion_start', 'robot_vel_lin_mag',
           'feature_std', 'feature_integral', 'features_from_series',
           'extract_features_from_bag', 'extract_user_features_from_bags']


# ============================================================================================
# SELF-TEST
# ============================================================================================
#
# Reference-precision note, because it decides what "same" can mean here: the reference CSV was
# written by pandas `to_csv` with default formatting, which stores 16 significant digits. float64
# needs up to 17 (repr(0.06882728959897137) has 17), so the file itself has already lost the last
# digit of some values. Agreement with the CSV is therefore capped at ~1e-15 relative and cannot be
# bit-exact no matter how correct the code is.
#
# Against the ORIGINAL CODE rather than its CSV output, this module is bit-exact: running
# dataset_generation's own TimeseriesLoader -> TimeseriesPreprocessor -> TimeseriesDerivedTS ->
# TimeseriesFeatureExtractor over data/timeseries_numpy and comparing float-by-float gives
# 560/560 identical values for all three features.
CSV_RTOL = 1e-12          # ~1000x looser than the 16-digit serialisation, far tighter than any bug
CSV_ATOL = 1e-9           # features span 0.05 .. 100, so this is negligible
PROXY_ATOL_N = 1e-6       # newtons

if __name__ == '__main__':
    import pandas as pd

    from optimal_force_models import optimal_force_sensor_proxy

    HERE = Path(__file__).resolve().parent
    REPO = HERE.parents[2]
    NUMPY_ROOT = REPO / 'data' / 'timeseries_numpy'
    REFERENCE_CSV = REPO / 'data' / 'timeseries_features.csv'
    BAG = REPO / 'data' / 'KATE_AA_U010_16_yellow_line_force_right_60-1_2025-08-07-18-58-52.bag'
    BAG_USER, BAG_PATH_ID = 10, 16        # the bag name encodes them: U010 -> user 10, _16_ -> path 16
    VERIFY_FORCE_TO_PATH = {0.0: 4, 40.0: 5, 60.0: 6, 80.0: 7}

    # --- VERIFICATION ONLY -------------------------------------------------------------------
    # The live session reads bags. These two helpers read the exported .npy dataset instead, and
    # exist purely so the regression tests below can cover all 560 n=28 paths — one bag would only
    # ever exercise one of them. They are NOT part of the production path.
    def _npy_series(path_dir: Path) -> Dict[str, np.ndarray]:
        out = {}
        for name in RAW_SERIES_NEEDED:
            arr = np.load(str(Path(path_dir) / f'{name}.npy'))
            if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
                raise ValueError(f'{path_dir}/{name}.npy: bad shape {arr.shape}')
            out[name] = np.asarray(arr, float)
        return out

    def _npy_user_features(user_dir: Path) -> Dict[str, Dict[float, float]]:
        out: Dict[str, Dict[float, float]] = {n: {} for n in PROXY_FEATURE_NAMES}
        for force_n, path_id in sorted(VERIFY_FORCE_TO_PATH.items()):
            feats = features_from_series(_npy_series(Path(user_dir) / f'path_{path_id}'))
            for name, value in feats.items():
                out[name][float(force_n)] = value
        return out
    # -----------------------------------------------------------------------------------------

    ok = True

    print('=' * 94)
    print('SELF-TEST 1 — the feature core vs the full pipeline output it must reproduce')
    print('=' * 94)
    print(f'  reference : {REFERENCE_CSV}\n')

    reference = pd.read_csv(REFERENCE_CSV).set_index(['user', 'path'])
    rows = []
    for user_dir in sorted(p for p in NUMPY_ROOT.iterdir() if p.is_dir() and p.name.startswith('U')):
        for path_dir in sorted((p for p in user_dir.iterdir() if p.is_dir()),
                               key=lambda p: int(p.name.split('_')[1])):
            rows.append({'user': int(user_dir.name[1:]), 'path': int(path_dir.name.split('_')[1]),
                         **features_from_series(_npy_series(path_dir))})
    mine = pd.DataFrame(rows).set_index(['user', 'path'])
    common = mine.index.intersection(reference.index)
    print(f'  {len(common)} paths compared\n')

    print(f'  {"feature":<36}{"max |diff|":>13}{"max rel":>11}{"scale":>12}  verdict')
    for name in PROXY_FEATURE_NAMES:
        a, b = mine.loc[common, name].astype(float), reference.loc[common, name].astype(float)
        diff = (a - b).abs()
        rel = (diff / b.abs().replace(0, np.nan)).max()
        match = bool(np.isclose(a, b, rtol=CSV_RTOL, atol=CSV_ATOL).all())
        ok &= match
        print(f'  {name:<36}{diff.max():>13.2e}{(0.0 if np.isnan(rel) else rel):>11.1e}'
              f'{b.abs().mean():>12.4g}  {"MATCH" if match else "DIFFERS"}')
    print("\n  (residual ~1e-15 is the CSV's 16-digit serialisation, not a computation difference)")

    print('\n' + '=' * 94)
    print('SELF-TEST 2 — close the loop: features -> Method 4 optimal force, all 26 participants')
    print('=' * 94)
    try:
        prescriptions = pd.read_csv(HERE / 'user_study_optimal_forces.csv')
        prescriptions = prescriptions[prescriptions['method'] == 'Sensor-Proxy'].set_index('user')
    except FileNotFoundError:
        print('  skipped — run evaluate_full_circle.py first')
        prescriptions = None

    if prescriptions is not None:
        worst, n_checked = 0.0, 0
        for user_id in sorted(prescriptions.index):
            user_dir = NUMPY_ROOT / f'U{user_id:03d}'
            if not user_dir.is_dir():
                continue
            f_max = float(prescriptions.at[user_id, 'max_push_force_N'])
            got = optimal_force_sensor_proxy(_npy_user_features(user_dir), f_max)
            worst = max(worst, abs(got.optimal_force_n - float(prescriptions.at[user_id,
                                                                               'optimal_force_N'])))
            n_checked += 1
        proxy_ok = worst <= PROXY_ATOL_N
        ok &= proxy_ok
        print(f'  {n_checked} participants · max |difference| in prescribed force: {worst:.3e} N  '
              f'({"PASS" if proxy_ok else "FAIL"})')

    print('\n' + '=' * 94)
    print('SELF-TEST 3 — the production path: read the ROS 1 bag directly')
    print('=' * 94)
    if not BAG.exists():
        print(f'  skipped — {BAG.name} not present')
    else:
        try:
            from_bag = extract_features_from_bag(BAG)
        except ImportError as exc:
            print(f'  skipped — {exc}  (pip install rosbags)')
        else:
            from_npy = features_from_series(_npy_series(NUMPY_ROOT / f'U{BAG_USER:03d}' /
                                                        f'path_{BAG_PATH_ID}'))
            ref_row = reference.loc[(BAG_USER, BAG_PATH_ID)]
            print(f'  bag   : {BAG.name}')
            print(f'  ref   : timeseries_features.csv  user={BAG_USER}  path={BAG_PATH_ID}\n')
            print(f'  {"feature":<36}{"from bag":>22}{"vs .npy":>12}{"rel vs CSV":>13}')
            bag_ok = True
            for name in PROXY_FEATURE_NAMES:
                b, n, c = from_bag[name], from_npy[name], float(ref_row[name])
                bit = (b == n)
                rel = abs(b - c) / abs(c) if c else abs(b - c)
                bag_ok &= bit and rel <= CSV_RTOL
                print(f'  {name:<36}{b:>22.15g}{"BIT-EXACT" if bit else "DIFFERS":>12}{rel:>13.1e}')
            ok &= bag_ok
            print(f'\n  ==> {"PASS" if bag_ok else "FAIL"} — the bag reproduces the numpy export '
                  f'bit for bit, and the CSV to its own precision')

    print(f'\n  ==> {"PASS" if ok else "FAIL"} — raw recordings reproduce the frozen Method 4 '
          f'prescription')
    raise SystemExit(0 if ok else 1)
