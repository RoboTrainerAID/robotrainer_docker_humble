#!/usr/bin/env python3
"""Minimal usage example for bayesian_optimization.py — block 5f, the ground truth.

One synthetic participant, hard-coded. Run it:  python example_call_bayesian_optimization.py
"""

import random

from bayesian_optimization import N_BO_TRIALS, bo_optimal_force, next_bo_force, run_bo_block

MAX_FORCE_N = 178.0            # block 3b, RoboTrainer Right (disturbance goes left)

# The 8 GP seed observations: (force [N], raw TLX performance rating).
# Four from the assessment tasks (5a) and four from the method trials (5d).
SEED_OBSERVATIONS = [
    (0.0, 72.0), (40.0, 66.0), (60.0, 53.0), (80.0, 37.0),        # block 5a
    (62.5, 55.0), (64.4, 50.0), (64.4, 52.0), (71.5, 45.0),       # block 5d, one per method
]


def run_trial_and_ask_tlx(force_n: float) -> float:
    """STAND-IN for the real thing: command the force, then read the participant's TLX rating.

    Replace this with your ROS 2 call. Here it fakes a participant whose performance curve is
    y = -0.0025*F^2 - 0.12*F + 72 (crossing 48.5 at ~76 N), plus 10 rating points of noise.
    Seeded by the force so the run is reproducible.
    """
    true_rating = -0.0025 * force_n ** 2 - 0.12 * force_n + 72.0
    noisy = true_rating + random.Random(round(force_n * 10)).gauss(0.0, 10.0)
    return min(100.0, max(0.0, noisy))          # the TLX scale is bounded 0-100


# ------------------------------------------------------ the 8-trial loop, one call per trial -----

observations = list(SEED_OBSERVATIONS)

for _ in range(N_BO_TRIALS):                       # 5 active-learning, then 3 level-targeting
    proposal = next_bo_force(observations, MAX_FORCE_N)
    rating = run_trial_and_ask_tlx(proposal.force_n)
    observations.append((proposal.force_n, rating))
    print(f'trial {proposal.trial_index + 1}/{N_BO_TRIALS}  {proposal.phase:<16} '
          f'{proposal.force_n:6.2f} N -> rating {rating:5.1f}')

# ------------------------------------------------------ the ground truth ------------------------

truth = bo_optimal_force(observations, MAX_FORCE_N)
print(f'\nx*_BO = {truth.optimal_force_n:.2f} N   '
      f'90% CI [{truth.credible_low_n:.1f}, {truth.credible_high_n:.1f}] N   '
      f'flag={truth.flag}')

# Store all of this per participant — the future proxy recalibration needs the whole posterior,
# not just the point estimate.
print(f'  from {truth.n_observations} observations, fitted '
      + '  '.join(f'{k}={v:.1f}' for k, v in truth.hyperparameters.items()))

# ------------------------------------------------------ or drive all 8 trials in one call --------

observations2, proposals = run_bo_block(SEED_OBSERVATIONS, MAX_FORCE_N, run_trial_and_ask_tlx)
print(f'\nrun_bo_block gave the same forces: '
      f'{[round(p.force_n, 1) for p in proposals]}')
