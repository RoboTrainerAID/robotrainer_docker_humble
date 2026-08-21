#!/usr/bin/env python3
"""Minimal usage example for optimal_force_models.py — Methods 1-4.

One synthetic participant, hard-coded. Run it:  python example_call_optimal_force_models.py
"""

from optimal_force_models import (all_optimal_forces, optimal_force_linear,
                                  optimal_force_quadratic, optimal_force_quadratic_maxforce,
                                  optimal_force_sensor_proxy)

# ---------------------------------------------------------------- what the session collected ----

# Block 3b — max push force on the side opposing the disturbance.
# The assessment disturbance goes LEFT, so the participant pushes RIGHT -> RoboTrainer Right.
MAX_FORCE_N = 178.0

# Block 5a — raw NASA-TLX "eigene Leistung" after each assessment task.
# 0-100, HIGHER = BETTER, so it falls as the force rises.
TLX_PERFORMANCE = {0.0: 70.0, 40.0: 60.0, 60.0: 50.0, 80.0: 40.0}     # {force [N]: rating}

# Block 5a — the three proxy features per assessment task, in the units of
# data/timeseries_features.csv. Same four force levels as above.
TIMESERIES_FEATURES = {
    'path_deviation_front_std':         {0.0: 0.027, 40.0: 0.069, 60.0: 0.103, 80.0: 0.097},
    'user_force_y_impulse':             {0.0: -25.9, 40.0: -31.5, 60.0: -50.3, 80.0: -48.2},
    'robot_vel_lin_mag_total_distance': {0.0: 8.19,  40.0: 8.34,  60.0: 8.28,  80.0: 8.51},
}

# ---------------------------------------------------------------- block 5b: the four models -----

# One call per method. max_force_n is REQUIRED for Method 3 (it is part of the model) and optional
# for the others, where it only caps the prescription at what the participant can actually resist.
m1 = optimal_force_linear(TLX_PERFORMANCE, MAX_FORCE_N)
m2 = optimal_force_quadratic(TLX_PERFORMANCE, MAX_FORCE_N)
m3 = optimal_force_quadratic_maxforce(TLX_PERFORMANCE, MAX_FORCE_N)
m4 = optimal_force_sensor_proxy(TIMESERIES_FEATURES, MAX_FORCE_N)

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
