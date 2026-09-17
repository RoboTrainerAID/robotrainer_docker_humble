#!/usr/bin/env python3
"""Minimal usage example for optimal_force_models.py — Methods 1-4.

The sensor features are read straight out of a ROS 1 bag; nothing is written to disk in between.
Run it:  python example_call_optimal_force_models.py
"""

from pathlib import Path

from optimal_force_models import (all_optimal_forces, optimal_force_linear,
                                  optimal_force_quadratic, optimal_force_quadratic_maxforce,
                                  optimal_force_sensor_proxy)
from sensor_feature_extraction import extract_features_from_bag, extract_user_features_from_bags

DATA = Path(__file__).resolve().parents[3] / 'data'

# ---------------------------------------------------------------- what the session collected ----

# Block 3b — max push force on the side opposing the disturbance.
# The assessment disturbance goes LEFT, so the participant pushes RIGHT -> RoboTrainer Right.
MAX_FORCE_N = 178.0

# Block 5a — raw NASA-TLX "eigene Leistung" after each assessment task.
# 0-100, HIGHER = BETTER, so it falls as the force rises.
TLX_PERFORMANCE = {0.0: 70.0, 40.0: 60.0, 60.0: 50.0, 80.0: 40.0}     # {force [N]: rating}

# Block 5a — ONE ROS 1 BAG PER ASSESSMENT TASK.
#
#     BAG_PER_FORCE = {
#          0.0: DATA / 'KATE_AA_U0XX_4_..._force_left_0-1_....bag',    # path 4, no disturbance
#         40.0: DATA / 'KATE_AA_U0XX_5_..._force_left_40-1_....bag',   # path 5
#         60.0: DATA / 'KATE_AA_U0XX_6_..._force_left_60-1_....bag',   # path 6
#         80.0: DATA / 'KATE_AA_U0XX_7_..._force_left_80-1_....bag',   # path 7
#     }

BAG_60N = DATA / 'KATE_AA_U010_16_yellow_line_force_right_60-1_2025-08-07-18-58-52.bag'

BAG_PER_FORCE = {
     0.0: BAG_60N,     # TODO replace with the  0 N bag of this participant
    40.0: BAG_60N,     # TODO replace with the 40 N bag
    60.0: BAG_60N,     # the one real recording shipped here
    80.0: BAG_60N,     # TODO replace with the 80 N bag
}

# ---------------------------------------------------------------- bag -> the three features -----

# One call, one argument: the bag path. Reads only the three topics the features need
# (~5000 messages out of a 6.8 GB bag) and returns the features directly.
#### ONLY FOR TESTING: REMOVE FOR LIVE EXPEERIMENT
print(f'reading {BAG_60N.name}')
for name, value in extract_features_from_bag(BAG_60N).items():
    print(f'  {name:<36}{value:.12g}')

# The final call for all force levels
TIMESERIES_FEATURES = extract_user_features_from_bags(BAG_PER_FORCE)

# ---------------------------------------------------------------- block 5b: the four models -----

# One call per method. max_force_n is REQUIRED for Method 3 (it is part of the model) and optional
# for the others, where it only caps the prescription at what the participant can actually resist.
m1 = optimal_force_linear(TLX_PERFORMANCE, MAX_FORCE_N)
m2 = optimal_force_quadratic(TLX_PERFORMANCE, MAX_FORCE_N)
m3 = optimal_force_quadratic_maxforce(TLX_PERFORMANCE, MAX_FORCE_N)
m4 = optimal_force_sensor_proxy(TIMESERIES_FEATURES, MAX_FORCE_N)

print()
for result in (m1, m2, m3, m4):
    # .optimal_force_n is always safe to command: inside [0, force_cap_n].
    # .flag is 'ok' / 'ok_numeric' / 'clipped_low' / 'clipped_high' — anything but the first two is
    # a censored observation and must be logged with the trial.
    print(f'{result.method:<19} {result.optimal_force_n:6.2f} N   flag={result.flag}')

# Or all four in one call — this is what the ROS 2 node does at the end of block 5a.
print()
results = all_optimal_forces(TLX_PERFORMANCE, TIMESERIES_FEATURES, MAX_FORCE_N)
print('block 5d runs one trial at each of these, in random order:')
print('  ' + '  '.join(f'{r.optimal_force_n:.1f} N' for r in results.values()))
