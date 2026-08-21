#!/usr/bin/env python3
"""Block 5f — Bayesian-optimization ground truth for the optimal disturbance force.

Same contract as `optimal_force_models.py`: hand it everything you have about the participant and
it hands back the next force to run.

    from bayesian_optimization import next_bo_force, bo_optimal_force

    obs = [(0.0, 70.0), (40.0, 60.0), (60.0, 50.0), (80.0, 40.0),      # block 5a
           (62.5, 55.0), (64.4, 50.0), (57.8, 60.0), (71.5, 45.0)]     # block 5d  -> 8 GP seeds

    for _ in range(8):                                   # 5 active-learning + 3 level-targeting
        p = next_bo_force(obs, max_force_n=178.0)
        rating = run_trial_and_ask_tlx(p.force_n)        # your ROS 2 side
        obs.append((p.force_n, rating))

    truth = bo_optimal_force(obs, max_force_n=178.0)     # x*_BO + credible interval

What is in here
---------------
§1  CANDIDATE SET   `F_cap` and the minimum-spacing rule — pure numpy, no torch needed. This is
                    where "the acquisition wants 300 N but the participant can only do 96 N" is
                    solved, and it is solved by construction rather than by a check afterwards.
§2  GP + ACQF CONFIG for Ax / BoTorch — the mean module, the priors, the fixed noise, and both
                    acquisition functions, spelled out. Torch is imported lazily, so this module
                    imports fine on a machine without it.
§3  NUMPY REFERENCE an independent implementation of the same GP and the same two acquisitions.
                    It is the executable spec for §2 and it runs anywhere numpy + scipy run, so
                    the candidate logic and the whole 8-trial loop can be tested without Ax.
§4  ONE-CALL API    `next_bo_force`, `run_bo_block`, `bo_optimal_force`.

Every prior below comes from §5 of OPTIMAL_FORCE_PIPELINE.md and is FROZEN from the n=28 study.
Nothing here is re-estimated per participant except the two kernel hyperparameters (§2.2).

Requirements
------------
§1, §3, §4  numpy + scipy only, Python >= 3.8. Tested on Ubuntu 22.04 / ROS 2 Humble
            (Python 3.10.12) with numpy 1.21-2.2 and pandas 1.3-2.3.

§2          Verified against the exact API of this stack, source by source, on 2026-08-21:

              ax-platform 1.0.0   requires Python >= 3.10 and PINS botorch == 0.14.0 exactly
              botorch     0.14.0  requires Python >= 3.10, PINS gpytorch == 1.14, torch >= 2.0.1
              numpy       1.24.3  fine — no removed aliases (np.float/int/bool) are used
              pandas      2.2.2   only used by the cohort check at the bottom of this file

            Python 3.10 is the MINIMUM for both ax 1.0.0 and botorch 0.14.0, and Humble ships
            3.10.12 — so there is no margin. §1/§3/§4 stay 3.9-clean; §2 does not have to.

            The code in §2 is unrun (torch is not installed on the authoring machine) but every
            call was checked against the released source of the pinned versions: signatures,
            defaults, keyword names, and the fixed-noise fantasize path. See the notes at each
            call site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from optimal_force_models import (DEVICE_FORCE_MIN_N, FLAG_CLIPPED_HIGH, FLAG_CLIPPED_LOW,
                                  FLAG_OK, RATING_MIN, TARGET_PERFORMANCE_LEVEL, force_cap)

# ============================================================================================
# FROZEN GP CONFIGURATION  (OPTIMAL_FORCE_PIPELINE.md §5)
# ============================================================================================

# Prior mean m(F): least squares through the four n=28 column means, highest power x² first so it
# drops straight into np.polyval / Horner. Crosses the target at 81.9 N.
PRIOR_MEAN_COEFFS = (-1.35490e-03, -0.083392, 64.4126)

PRIOR_AMPLITUDE = 14.2            # s   [rating points]; outputscale s^2 ~ 201
PRIOR_AMPLITUDE_LOGSD = 0.30      # lognormal width on s  (95%: x/÷1.8)
OBSERVATION_NOISE_SD = 10.0       # sigma_n [rating points] — FIXED, not fitted (§6.5)
PRIOR_LENGTHSCALE = 70.0          # l   [N]
PRIOR_LENGTHSCALE_LOGSD = 0.30
LENGTHSCALE_BOUNDS = (40.0, 150.0)   # hard interval; the weakest of the priors, so bound it
MATERN_NU = 2.5                   # Matern-5/2: smooth but not analytic

# --- trial budget -----------------------------------------------------------------------------
N_ACTIVE_LEARNING_TRIALS = 5      # phase A: learn the curve, ignore the target level
N_LEVEL_TARGETING_TRIALS = 3      # phase B: pin down the 48.5 crossing
N_BO_TRIALS = N_ACTIVE_LEARNING_TRIALS + N_LEVEL_TARGETING_TRIALS
PHASE_A, PHASE_B = 'active_learning', 'level_targeting'

STRADDLE_BETA = 1.96              # exploration weight; larger explores more
LEVEL_EI_SAMPLES = 512            # MC samples for the level-objective EI

# --- candidate set ----------------------------------------------------------------------------
GRID_RESOLUTION_N = 1.0           # the device is commanded in ~1 N steps; finer buys nothing
# Minimum spacing from every already-sampled force. Purpose: never spend a trial re-measuring a
# point you already know. The information scale for that is the lengthscale (two samples closer
# than ~l/8 are near-redundant), but a weak participant's whole admissible range can be shorter
# than l, so the rule has to scale with F_cap or the candidate set empties out. Measured on the
# n=28 F_max values: a fixed 8 N exhausts the set for 6 of 26 participants (user 12, F_cap 66.8 N,
# fits only 1 of 8 trials); the adaptive rule below fits all 8 for 26 of 26.
MIN_SPACING_FRACTION = 1.0 / 25.0     # of the admissible range
MIN_SPACING_BOUNDS = (3.0, 8.0)       # N, floor and ceiling
# Phase B needs a MUCH tighter rule than phase A, and this is not a detail — it is the difference
# between finding the contour and being pushed off it. The four method prescriptions of block 5d are
# all estimates of the same crossing, so the seeds cluster exactly where phase B wants to sample; at
# phase-A spacing that whole region is masked out and straddle is forced 15-20 N away from the
# contour it is supposed to bracket. The redundancy argument behind min-spacing only holds when the
# noise is small relative to the signal, and here it is not: with sigma_n = 10 and 16 observations a
# second sample next to an existing one is not redundant, it halves the effective noise there, which
# is exactly what tightens x*_BO.
MIN_SPACING_PHASE_B_FACTOR = 0.25
# Ladder used when the set still empties out; see `candidate_forces`.
SPACING_RELAXATION_LADDER = (1.0, 0.5, 0.0)   # multiples of the nominal spacing

# --- integration measure for the phase-A criterion --------------------------------------------
# qNegIntegratedPosteriorVariance integrates Var[f] over the measure represented by `mc_points`, so
# the choice of grid IS a choice of what "learning the curve" means. A uniform grid over [0, F_cap]
# rewards reducing variance where the rating has already hit the bottom of the 0-100 scale — a force
# so high that the participant scores 0 with certainty carries no information about the crossing, but
# a Gaussian posterior is unbounded and happily keeps paying to learn it. Truncate the measure to the
# forces where the rating is still on-scale.
RESTRICT_INTEGRATION_TO_MEASURABLE = True
MEASURABLE_MARGIN_SD = 2.0            # keep F while mu(F) + 2*sd(F) >= RATING_MIN
MIN_INTEGRATION_POINTS = 20

# --- reading x* off the posterior (§6.4) -------------------------------------------------------
POSTERIOR_PATHS = 2000            # sample paths for the credible interval
CREDIBLE_MASS = 0.90              # -> 5th / 95th percentiles

RUNG_NAMES = {0: 'nominal_spacing', 1: 'relaxed_spacing', 2: 'duplicates_only',
              3: 'deliberate_repeat'}


# ============================================================================================
# §1  CANDIDATE SET — the F_max limit and the minimum-spacing rule
# ============================================================================================

def min_spacing_for(cap_n: float, phase: str = PHASE_A) -> float:
    """Minimum distance a new trial must keep from every already-sampled force.

    Phase A: the adaptive rule (coverage matters, redundancy is waste).
    Phase B: a quarter of it, floored at the device resolution — see MIN_SPACING_PHASE_B_FACTOR.
    """
    lo, hi = MIN_SPACING_BOUNDS
    spacing = float(np.clip(cap_n * MIN_SPACING_FRACTION, lo, hi))
    if phase == PHASE_B:
        spacing = max(GRID_RESOLUTION_N, spacing * MIN_SPACING_PHASE_B_FACTOR)
    return spacing


def admissible_grid(max_force_n: float) -> np.ndarray:
    """Every force the participant may be asked to resist: [0, F_cap] on the device grid.

    `F_cap = min(300, MAX_FORCE_SAFETY_FACTOR * F_max)` — the same bound Methods 1-4 clip to, so
    the ground truth and the four prescriptions live on exactly the same interval.

    THIS is the answer to "what if the acquisition wants 300 N but the participant tops out at
    96 N": the grid never contains 300 N, so an inadmissible force cannot be proposed. It is not
    detected and corrected afterwards — it is unrepresentable. Two places have to be capped, not
    one:

      * the candidate set the acquisition is maximised over  (below), and
      * the integration grid of the phase-A criterion        (`ReferenceGP.alc` / `mc_points`).

    Capping only the first still wastes trials: the integrated-variance criterion would keep
    pushing candidates up against F_cap trying to reduce uncertainty in a region that can never be
    sampled, because the integral still rewards it there.
    """
    cap = force_cap(max_force_n)
    n = int(math.floor((cap - DEVICE_FORCE_MIN_N) / GRID_RESOLUTION_N)) + 1
    grid = DEVICE_FORCE_MIN_N + GRID_RESOLUTION_N * np.arange(n, dtype=float)
    if grid[-1] < cap - 1e-9:                 # keep the endpoint even off-grid
        grid = np.append(grid, cap)
    return grid


def candidate_forces(sampled_forces: Sequence[float], max_force_n: float,
                     phase: str = PHASE_A) -> Tuple[np.ndarray, float, int]:
    """Admissible forces that also respect the minimum spacing. Returns (candidates, spacing, rung).

    The spacing constraint is enforced by MASKING A DISCRETE GRID, not by a constrained continuous
    optimiser. In 1-D over at most ~300 points that is exact, ~free, and trivially debuggable —
    whereas `min |F - F_i| >= d` is a non-convex constraint with a disconnected feasible set, which
    is exactly what gradient-based `optimize_acqf` handles worst. Enumerate; do not optimise.

    The ladder, when the mask empties the set (a weak participant whose admissible range is short
    relative to the spacing):

        rung 0  nominal spacing
        rung 1  half the nominal spacing
        rung 2  exact duplicates only (spacing 0) — anything not already sampled is allowed
        rung 3  no free point at all: allow a DELIBERATE REPEAT of an existing force

    Rung 3 is not a wasted trial. With sigma_n fixed, a second observation at the same force still
    reduces the posterior variance there, and the pair is a test-retest measurement that feeds the
    cohort-level noise estimate (§6.5). Log the rung — a run that reached rung 2 or 3 sampled more
    densely than intended and the analysis should know.
    """
    grid = admissible_grid(max_force_n)
    spacing = min_spacing_for(grid[-1], phase)
    taken = np.asarray([f for f in sampled_forces if np.isfinite(f)], float)
    if taken.size == 0:
        return grid, spacing, 0

    distance = np.min(np.abs(grid[:, None] - taken[None, :]), axis=1)
    for rung, factor in enumerate(SPACING_RELAXATION_LADDER):
        threshold = spacing * factor
        free = grid[distance >= threshold] if threshold > 0 else grid[distance > 1e-9]
        if free.size:
            return free, threshold, rung
    return grid, 0.0, 3          # rung 3: every grid point is taken, allow a repeat


def integration_grid(gp: 'ReferenceGP', max_force_n: float) -> np.ndarray:
    """The measure the phase-A variance integral is taken over (`mc_points` in BoTorch).

    The admissible grid, truncated where the posterior says the rating has certainly floored at the
    bottom of the 0-100 scale. Since the curve falls monotonically the informative region is a
    prefix, so this is a single truncation rather than a mask.

    If you would rather spend the five phase-A trials near the crossing than spread over the whole
    curve, restrict the measure to a band instead of a prefix — one line, and a deliberate change of
    what phase A is for:

        band = grid[np.abs(grid - predicted_crossing) <= 2 * gp.lengthscale]

    The default is the wider measure because block 5f is meant to deliver a ground-truth *curve* as
    well as a ground-truth x*, and phase B already concentrates on the contour.
    """
    grid = admissible_grid(max_force_n)
    if not RESTRICT_INTEGRATION_TO_MEASURABLE:
        return grid
    mu, sd = gp.posterior(grid)
    floored = np.flatnonzero(mu + MEASURABLE_MARGIN_SD * sd < RATING_MIN)
    if floored.size == 0:
        return grid
    cut = max(int(floored[0]), MIN_INTEGRATION_POINTS)
    return grid[:cut]


def target_reachable(cap_n: float, posterior_mean_at_cap: float) -> bool:
    """False when the participant's curve is still above the target at their own limit.

    Then no crossing exists inside [0, F_cap] and phase B has nothing to bracket: it will pile its
    trials against F_cap, which is the right behaviour (it pins down how close they get) but x*_BO
    must be reported as censored high, exactly like `FLAG_CLIPPED_HIGH` in §3.2. Run phase B anyway
    so the protocol stays identical for everyone, and carry the flag into the analysis.
    """
    return bool(posterior_mean_at_cap <= TARGET_PERFORMANCE_LEVEL)


# ============================================================================================
# §2  GP AND ACQUISITION CONFIGURATION FOR Ax / BoTorch
# ============================================================================================
#
# Torch is imported inside the functions so this module works without it.
#
# ---------------------------------------------------------------------------------------------
# §2.1  The prior mean: use a real mean MODULE, not the residual trick
# ---------------------------------------------------------------------------------------------
# OPTIMAL_FORCE_PIPELINE.md §6.1 suggests fitting on residuals y - m(F) to avoid writing a mean
# module. That works for FITTING but it breaks LEVEL TARGETING, and the reason is worth stating:
# in residual space the crossing condition y(F) = 48.5 becomes r(F) = 48.5 - m(F), i.e. the target
# is no longer a constant but a function of the force. Both phase-B acquisitions would then have to
# recompute m(X) internally at every candidate, and `GenericMCObjective`'s callable only receives
# X in newer BoTorch versions. Ten lines of mean module removes the whole problem and keeps the
# posterior directly interpretable in rating points. Do that instead.

def make_prior_mean_module(coeffs: Sequence[float] = PRIOR_MEAN_COEFFS):
    """A FIXED (non-learnable) polynomial prior mean, highest power first.

    `register_buffer`, not `register_parameter`: the n=28 group curve is a prior, so the marginal
    likelihood must not be allowed to fit it away. Measured effect of getting this wrong — letting
    the mean be a fitted constant, which is BoTorch's default — is x*_BO RMSE 12.7 N instead of
    9.9 N (§5.1).
    """
    import torch
    from gpytorch.means import Mean

    class PolynomialPriorMean(Mean):
        def __init__(self, poly_coeffs):
            super().__init__()
            self.register_buffer('poly_coeffs',
                                 torch.as_tensor(poly_coeffs, dtype=torch.double))

        def forward(self, x):
            f = x.squeeze(-1)
            out = torch.zeros_like(f)
            for c in self.poly_coeffs:            # Horner
                out = out * f + c
            return out

    return PolynomialPriorMean(coeffs)


# ---------------------------------------------------------------------------------------------
# §2.2  The model
# ---------------------------------------------------------------------------------------------
def build_covar_module():
    """Matern-5/2 kernel with the frozen n=28 priors on amplitude and lengthscale.

    Split out so `AX_GENERATION_NOTES` can hand the same object to Ax's `ModelConfig.model_options`
    and get an identical model to the one `build_botorch_gp` builds.

    The amplitude prior is stated on the OUTPUTSCALE, which is s^2, not s: if
    log s ~ N(log s0, sd^2) then log s^2 = 2 log s ~ N(2 log s0, (2 sd)^2) — hence both factors of 2.
    Verified for gpytorch 1.14: `LogNormalPrior(loc, scale)`, `Interval(lower, upper)`,
    `MaternKernel(nu=, ard_num_dims=, lengthscale_prior=, lengthscale_constraint=)`,
    `ScaleKernel(base_kernel, outputscale_prior=)`.
    """
    from gpytorch.constraints import Interval
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.priors import LogNormalPrior

    covar = ScaleKernel(
        MaternKernel(
            nu=MATERN_NU,
            ard_num_dims=1,
            lengthscale_prior=LogNormalPrior(math.log(PRIOR_LENGTHSCALE),
                                             PRIOR_LENGTHSCALE_LOGSD),
            lengthscale_constraint=Interval(*LENGTHSCALE_BOUNDS),
        ),
        outputscale_prior=LogNormalPrior(2.0 * math.log(PRIOR_AMPLITUDE),
                                         2.0 * PRIOR_AMPLITUDE_LOGSD),
    )
    covar.base_kernel.lengthscale = PRIOR_LENGTHSCALE          # start at the prior median
    covar.outputscale = PRIOR_AMPLITUDE ** 2
    return covar


def build_botorch_gp(forces: Sequence[float], ratings: Sequence[float],
                     noise_sd: float = OBSERVATION_NOISE_SD,
                     fit_hyperparameters: bool = True):
    """The session GP, on the RAW RATING SCALE, with every prior set explicitly.

    Fixed:   the prior mean (§2.1) and sigma_n (below).
    Fitted:  outputscale s^2 and lengthscale l, by MAP under the lognormal priors.

    Three things that silently break the priors if you leave the defaults alone:

    1. `outcome_transform=None`. `SingleTaskGP` defaults to `Standardize(m=1)`, which rescales Y to
       zero mean / unit variance. Then `s = 14.2` and `sigma_n = 10` are no longer rating points and
       every prior here is meaningless. Standardising would also fight the fixed prior mean.
    2. `input_transform=None`. Default `Normalize` maps force to [0, 1], so `l = 70` would have to
       become `70 / F_cap` — a different prior per participant. Keep newtons.
    3. `train_Yvar`, not a fitted `GaussianLikelihood`. Supplying the variance makes it a
       fixed-noise model, which is what §6.5 asks for: with 16 points sigma_n is confounded with l
       (short l + small noise fits the same data as long l + large noise, and the two give
       materially different crossings), and a per-participant fitted sigma_n makes the ground truths
       non-comparable across the cohort.

    Verified against botorch 0.14.0 (the version ax-platform 1.0.0 pins). Its signature is

        SingleTaskGP(train_X, train_Y, train_Yvar=None, likelihood=None, covar_module=None,
                     mean_module=None, outcome_transform=DEFAULT, input_transform=None)

    so every keyword used below exists. `outcome_transform` defaults to the sentinel `DEFAULT`,
    which is resolved to `Standardize(...)`; passing `None` explicitly disables it, which is what
    we want. `input_transform` is already `None` by default — passed anyway so the intent is on the
    page. On a botorch older than ~0.12 the sentinel does not exist and `None` may be ignored, so
    if you ever unpin, assert `model.outcome_transform is None` after construction.

    `fit_gpytorch_mll` really does give MAP rather than ML: gpytorch's
    `ExactMarginalLogLikelihood.forward` calls `_add_other_terms`, which iterates
    `model.named_priors()` and adds each prior's log-probability to the objective. The priors
    registered on the kernel below are therefore part of what gets maximised. Without them this
    would be plain ML and the lengthscale would run to a bound.
    """
    import torch
    from botorch.models import SingleTaskGP

    X = torch.as_tensor(np.asarray(forces, float), dtype=torch.double).unsqueeze(-1)
    Y = torch.as_tensor(np.asarray(ratings, float), dtype=torch.double).unsqueeze(-1)
    Yvar = torch.full_like(Y, float(noise_sd) ** 2)          # sigma_n^2 = 100, known

    model = SingleTaskGP(
        train_X=X, train_Y=Y, train_Yvar=Yvar,
        mean_module=make_prior_mean_module(),
        covar_module=build_covar_module(),
        outcome_transform=None,        # see (1)
        input_transform=None,          # see (2)
    )
    if fit_hyperparameters:
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        # fit_gpytorch_mll maximises MLL + log priors, i.e. MAP, because the priors are registered
        # on the modules above. Without them this would be plain ML and l would run to a bound.
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


# ---------------------------------------------------------------------------------------------
# §2.3  Phase A acquisition — integrated posterior variance reduction
# ---------------------------------------------------------------------------------------------
def build_phase_a_acqf(model, max_force_n: float,
                       mc_points_grid: Optional[np.ndarray] = None):
    """qNegIntegratedPosteriorVariance == the ALC criterion, maximised.

        alpha_ALC(c) = integral[ sigma^2(F) - sigma^2_{|c}(F) ] dF
                     = integral[ k_post(F,c)^2 ] dF / ( sigma^2(c) + sigma_n^2 )

    Deterministic and closed-form: it does not depend on the unobserved rating.

    `mc_points` IS the integration domain, so it must be the admissible grid — see
    `admissible_grid`. Do NOT pass a 0-300 N grid when F_cap is 96 N: the criterion would then
    reward candidates for reducing variance in a region that can never be sampled, and push every
    phase-A trial up against F_cap.

    Why not plain `PosteriorStandardDeviation` (argmax sigma)? In 1-D it hugs the boundaries — with
    0 N already sampled it walks straight to F_cap and then bisects the largest gaps, which is a
    space-filling design, not an information-driven one.

    One thing worth knowing, because it is the single place this could have failed silently: in
    botorch 0.14.0 `qNegIntegratedPosteriorVariance.forward` evaluates the criterion by building a
    fantasy model with `self.model.fantasize(X=X, sampler=self.sampler)` and does NOT pass
    `observation_noise`. Our GP is a FIXED-noise model (`train_Yvar`), and `FantasizeMixin.fantasize`
    handles exactly that case: when `observation_noise is None` and the likelihood is a
    `FixedNoiseGaussianLikelihood` it takes the MEAN of the training noise. Every `train_Yvar` entry
    here is identically `sigma_n^2 = 100`, so the mean is exactly 100 and the fantasy model carries
    the intended noise. If you ever make the noise heteroscedastic, that mean stops being exact.

    Signature verified for botorch 0.14.0:
        qNegIntegratedPosteriorVariance(model, mc_points, sampler=None,
                                        posterior_transform=None, X_pending=None)
    """
    import torch
    from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance

    grid = admissible_grid(max_force_n) if mc_points_grid is None else np.asarray(mc_points_grid)
    mc_points = torch.as_tensor(grid, dtype=torch.double).unsqueeze(-1)
    return qNegIntegratedPosteriorVariance(model=model, mc_points=mc_points)


# ---------------------------------------------------------------------------------------------
# §2.4  Phase B acquisition — level targeting at 48.5
# ---------------------------------------------------------------------------------------------
# NEVER plain EI/UCB here: the curve falls monotonically, so its maximum is at F = 0 and all three
# remaining trials would go there.
#
# The transform is  g(y) = -|y - 48.5|,  applied to POSTERIOR SAMPLES INSIDE the acquisition — not
# to the training data. Transforming the data and refitting would destroy the sign (68.5 and 28.5
# both map to -20, so the model could no longer tell too-easy from too-hard) and make the likelihood
# a folded normal, biased by -0.798*sigma_n ~ -8 rating points exactly at the crossing. Inside the
# MC integral the sample is drawn first and transformed after, so the model stays Gaussian on the
# rating scale and the folded distribution is handled correctly by construction.

def build_phase_b_acqf(model, train_X, kind: str = 'qlognei'):
    """Level-targeting acquisition. `kind` is 'qlognei' (recommended) or 'straddle'.

    Only the acquisition changes between phases — same model, same priors, same accumulated
    observations. Nothing is refitted or reloaded.
    """
    import torch

    if kind == 'qlognei':
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.acquisition.objective import GenericMCObjective

        target = float(TARGET_PERFORMANCE_LEVEL)
        # Z: (sample_shape x batch x q x m) posterior samples on the RATING scale.
        level_objective = GenericMCObjective(
            lambda Z, X=None: -(Z.squeeze(-1) - target).abs())

        # Verified for botorch 0.14.0:
        #   qLogNoisyExpectedImprovement(model, X_baseline, sampler=None, objective=None,
        #       posterior_transform=None, X_pending=None, constraints=None, eta=1e-3, fat=True,
        #       prune_baseline=True, cache_root=True, tau_max=..., tau_relu=..., ...)
        # `prune_baseline` DEFAULTS TO TRUE in this version, hence the explicit False: pruning uses
        # the objective to drop baseline points that cannot be best, which buys nothing at 16 points
        # and is one more interaction with a non-monotone custom objective to reason about.
        # `GenericMCObjective` invokes its callable as `self.objective(samples, X=X)` — a keyword —
        # so the second parameter of the lambda above MUST be named `X`.
        return qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=torch.as_tensor(np.asarray(train_X, float),
                                       dtype=torch.double).reshape(-1, 1),
            objective=level_objective,
            prune_baseline=False,
            # cache_root=False,   # uncomment if you hit numerical trouble; the root caching is
            #                     # objective-agnostic in principle but cheap to switch off here
        )

    if kind == 'straddle':
        from botorch.acquisition.analytic import AnalyticAcquisitionFunction
        from botorch.utils.transforms import t_batch_mode_transform

        class Straddle(AnalyticAcquisitionFunction):
            """alpha(F) = beta*sigma(F) - |mu(F) - target|   (Bryan et al. 2005).

            Sample where the posterior mean is near the target AND the posterior is still
            uncertain. The exploration term is explicit, which is worth something with only three
            trials left.
            """

            def __init__(self, model, target=TARGET_PERFORMANCE_LEVEL, beta=STRADDLE_BETA,
                         posterior_transform=None, **kwargs):
                super().__init__(model=model, posterior_transform=posterior_transform)
                self.target, self.beta = float(target), float(beta)

            @t_batch_mode_transform(expected_q=1)
            def forward(self, X):
                mean, sigma = self._mean_and_sigma(X)      # latent SD — what straddle wants
                return self.beta * sigma - (mean - self.target).abs()

        return Straddle(model)

    raise ValueError(f"kind must be 'qlognei' or 'straddle', got {kind!r}")


# ---------------------------------------------------------------------------------------------
# §2.5  Optimising the acquisition over the masked candidate set
# ---------------------------------------------------------------------------------------------
def propose_next_force_botorch(forces: Sequence[float], ratings: Sequence[float],
                               max_force_n: float, phase: str,
                               phase_b_kind: str = 'qlognei') -> Tuple[float, Dict]:
    """One BoTorch proposal. Returns (force_n, diagnostics).

    `optimize_acqf_discrete` over the masked grid, NOT `optimize_acqf` over a box:

      * F_cap and the minimum spacing are both already baked into the candidate list, so no
        constraint machinery is needed and no proposal can ever be inadmissible;
      * `min |F - F_i| >= d` is non-convex with a disconnected feasible set — the one shape
        gradient-based acquisition optimisation handles worst;
      * 1-D, <=301 candidates: exhaustive evaluation is cheaper than L-BFGS restarts and exact.

    botorch 0.14.0's signature is
        optimize_acqf_discrete(acq_function, q, choices, max_batch_size=2048, unique=True,
                               X_avoid=None, inequality_constraints=None)
    returning `(q x d candidates, acquisition value)`. Note the native `X_avoid`: it excludes EXACT
    points, so it would cover "never repeat a force" but not "keep 7 N clear of one" — the spacing
    rule still needs the grid mask. `unique=True` is irrelevant at q=1.

    If you would rather stay inside Ax's generation-step machinery, see `AX_GENERATION_NOTES`.
    """
    import torch
    from botorch.optim import optimize_acqf_discrete

    candidates, spacing, rung = candidate_forces(forces, max_force_n, phase)
    model = build_botorch_gp(forces, ratings)
    if phase == PHASE_A:
        # same restricted measure as the reference implementation, so the two backends agree
        mc_grid = integration_grid(ReferenceGP(np.asarray(forces, float),
                                              np.asarray(ratings, float)).fit(), max_force_n)
        acqf = build_phase_a_acqf(model, max_force_n, mc_points_grid=mc_grid)
    else:
        acqf = build_phase_b_acqf(model, forces, kind=phase_b_kind)

    choices = torch.as_tensor(candidates, dtype=torch.double).unsqueeze(-1)
    best_X, best_value = optimize_acqf_discrete(acq_function=acqf, q=1, choices=choices)
    return float(best_X.item()), {
        'acquisition_value': float(best_value.item()),
        'n_candidates': int(candidates.size),
        'spacing_used_n': spacing,
        'spacing_rung': RUNG_NAMES[rung],
        'lengthscale_n': float(model.covar_module.base_kernel.lengthscale.item()),
        'amplitude': float(model.covar_module.outputscale.item()) ** 0.5,
        'backend': 'botorch',
    }


AX_GENERATION_NOTES = """
Driving this from Ax 1.0.0 instead of calling BoTorch directly
-------------------------------------------------------------
Everything above is per-participant and 1-D, so `propose_next_force_botorch` is less code than an
Ax Experiment. If you want Ax's machinery anyway, this is the form for **ax-platform 1.0.0**
(verified against the released source of that tag, not the current docs — see the rename warning at
the bottom):

    from ax.modelbridge.registry import Generators                    # `Models` is a deprecated shim
    from ax.models.torch.botorch_modular.surrogate import Surrogate, SurrogateSpec
    from ax.models.torch.botorch_modular.utils import ModelConfig
    from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    surrogate = Surrogate(
        surrogate_spec=SurrogateSpec(
            model_configs=[
                ModelConfig(
                    botorch_model_class=SingleTaskGP,
                    mll_class=ExactMarginalLogLikelihood,
                    model_options={                       # -> SingleTaskGP(**model_options)
                        "mean_module": make_prior_mean_module(),
                        "covar_module": build_covar_module(),   # the ScaleKernel of build_botorch_gp
                        "outcome_transform": None,
                    },
                    input_transform_classes=[],           # see (3)
                    outcome_transform_classes=[],         # see (3)
                )
            ]
        )
    )

    generator = Generators.BOTORCH_MODULAR(
        experiment=exp, data=data,
        surrogate=surrogate,
        botorch_acqf_class=qNegIntegratedPosteriorVariance,            # phase A
        acquisition_options={"mc_points": mc_points_for(max_force_n)},
    )
    # phase B: identical call, only these two change ->
    #   botorch_acqf_class=qLogNoisyExpectedImprovement,
    #   acquisition_options={"objective": level_objective, "prune_baseline": False}

Six points, all checked against the ax-platform 1.0.0 source:

1. `SurrogateSpec(model_configs=[ModelConfig(...)])`, NOT the flat kwargs. `Surrogate.__init__` in
   1.0.0 still accepts `botorch_model_class`, `model_options`, `mll_class`,
   `input_transform_classes`, `outcome_transform_classes` directly, but they now route through
   `_raise_deprecation_warning` — the docstring says "deprecated in favor of model_configs". Use the
   spec form so this does not have to be rewritten at the next minor release.

2. `Generators`, not `Models`. In 1.0.0 `ax/modelbridge/registry.py` defines
   `class Generators(ModelRegistryBase)` with `BOTORCH_MODULAR = "BoTorch"`, and keeps
   `class Models(metaclass=ModelsMetaClass)` whose docstring is literally "This is deprecated. Use
   Generators instead."

3. TRANSFORMS — the one that silently invalidates every prior in this file. `ModelConfig`'s
   `input_transform_classes` defaults to the `DEFAULT` sentinel (i.e. Ax picks `Normalize`), so it
   must be set to `[]` to keep the lengthscale in newtons; `outcome_transform_classes` defaults to
   `None`, but pass `[]` plus `model_options={"outcome_transform": None}` so nothing can slip a
   `Standardize` in and turn `s = 14.2` and `sigma_n = 10` into unitless numbers. After the first
   fit, assert on the constructed model rather than trusting this.

4. FIXED NOISE. Ax routes observation noise through the `sem` column of `Data`. Put
   `sem = OBSERVATION_NOISE_SD` (10.0) on every observation and Ax builds the fixed-noise model
   (`train_Yvar`); leave it empty or NaN and it infers the noise, which §6.5 explicitly does not
   want. This is easy to miss because nothing errors — you just get a different model.

5. SEARCH SPACE. `RangeParameter("force_n", lower=0.0, upper=F_cap)` per participant, so F_cap is
   enforced by the optimiser rather than checked afterwards. For the minimum spacing as well, use a
   `ChoiceParameter` over `candidate_forces(...)[0]` and REBUILD IT BEFORE EVERY TRIAL — the
   exclusion depends on what has already run, so it cannot be a static search space. Ax 1.0.0's
   public API exposes `RangeParameterConfig` / `ChoiceParameterConfig` via `ax.api.configs`.

6. RENAME WARNING for later versions. After 1.0.0 these modules were renamed:
   `ax.modelbridge` -> `ax.adapter`, `ax.models` -> `ax.generators`, and the tutorial form became
   `GeneratorSpec(generator_enum=Generators.BOTORCH_MODULAR, generator_kwargs={...})` from
   `ax.generation_strategy.generator_spec`. `ax.adapter.registry` does NOT exist in 1.0.0 — if you
   copy the current ax.dev tutorial it will ImportError against your pin.
"""


# ============================================================================================
# §3  NUMPY REFERENCE IMPLEMENTATION  (the executable spec — tested)
# ============================================================================================

def prior_mean(force) -> np.ndarray:
    """m(F), the frozen n=28 group curve."""
    return np.polyval(np.asarray(PRIOR_MEAN_COEFFS, float), np.asarray(force, float))


def _matern52(a: np.ndarray, b: np.ndarray, amplitude: float, lengthscale: float) -> np.ndarray:
    r = np.abs(np.asarray(a, float)[:, None] - np.asarray(b, float)[None, :]) / lengthscale
    s5 = math.sqrt(5.0)
    return amplitude ** 2 * (1.0 + s5 * r + 5.0 / 3.0 * r ** 2) * np.exp(-s5 * r)


@dataclass
class ReferenceGP:
    """The same GP as §2, in numpy: fixed prior mean, fixed noise, MAP over (s, l).

    Kept deliberately independent of torch so the candidate logic, the phase switch and the whole
    8-trial loop can be exercised in CI. `next_bo_force(..., backend='numpy')` uses it.
    """
    forces: np.ndarray
    ratings: np.ndarray
    noise_sd: float = OBSERVATION_NOISE_SD
    amplitude: float = PRIOR_AMPLITUDE
    lengthscale: float = PRIOR_LENGTHSCALE
    _chol: Optional[np.ndarray] = field(default=None, repr=False)
    _alpha: Optional[np.ndarray] = field(default=None, repr=False)

    # ---------------------------------------------------------------- fitting ----------------
    def _neg_log_posterior(self, theta: np.ndarray) -> float:
        s, ell = math.exp(theta[0]), math.exp(theta[1])
        K = _matern52(self.forces, self.forces, s, ell) + self.noise_sd ** 2 * np.eye(len(self.forces))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e12
        r = self.ratings - prior_mean(self.forces)
        a = np.linalg.solve(L.T, np.linalg.solve(L, r))
        mll = -0.5 * float(r @ a) - float(np.log(np.diag(L)).sum())
        # lognormal MAP terms: s^2 ~ logN(2 log s0, 2 sd), l ~ logN(log l0, sd)
        lp = (-0.5 * ((2 * theta[0] - 2 * math.log(PRIOR_AMPLITUDE))
                      / (2 * PRIOR_AMPLITUDE_LOGSD)) ** 2
              - 0.5 * ((theta[1] - math.log(PRIOR_LENGTHSCALE)) / PRIOR_LENGTHSCALE_LOGSD) ** 2)
        return -(mll + lp)

    def fit(self) -> 'ReferenceGP':
        """MAP over (log s, log l) under the lognormal priors, l bounded to LENGTHSCALE_BOUNDS."""
        bounds = [(math.log(1.0), math.log(60.0)),
                  (math.log(LENGTHSCALE_BOUNDS[0]), math.log(LENGTHSCALE_BOUNDS[1]))]
        x0 = np.array([math.log(PRIOR_AMPLITUDE), math.log(PRIOR_LENGTHSCALE)])
        res = minimize(self._neg_log_posterior, x0, method='L-BFGS-B', bounds=bounds)
        if res.success or np.isfinite(res.fun):
            self.amplitude, self.lengthscale = math.exp(res.x[0]), math.exp(res.x[1])
        return self._condition()

    def _condition(self) -> 'ReferenceGP':
        K = (_matern52(self.forces, self.forces, self.amplitude, self.lengthscale)
             + self.noise_sd ** 2 * np.eye(len(self.forces)))
        self._chol = np.linalg.cholesky(K)
        r = self.ratings - prior_mean(self.forces)
        self._alpha = np.linalg.solve(self._chol.T, np.linalg.solve(self._chol, r))
        return self

    # ---------------------------------------------------------------- posterior --------------
    def posterior(self, grid: np.ndarray, full_cov: bool = False):
        """(mu, sd) of the LATENT curve on `grid`; `full_cov` returns the covariance instead."""
        if self._chol is None:
            self._condition()
        Ks = _matern52(grid, self.forces, self.amplitude, self.lengthscale)
        mu = prior_mean(grid) + Ks @ self._alpha
        V = np.linalg.solve(self._chol, Ks.T)
        if full_cov:
            return mu, _matern52(grid, grid, self.amplitude, self.lengthscale) - V.T @ V
        var = self.amplitude ** 2 - np.sum(V ** 2, axis=0)
        return mu, np.sqrt(np.clip(var, 1e-12, None))

    # ---------------------------------------------------------------- acquisitions ----------
    def alc(self, candidates: np.ndarray, integration_grid: np.ndarray) -> np.ndarray:
        """Phase A: integral k_post(F,c)^2 dF / (sigma^2(c) + sigma_n^2).

        Mirrors qNegIntegratedPosteriorVariance. `integration_grid` must be the ADMISSIBLE grid.
        """
        if self._chol is None:
            self._condition()
        Kg = _matern52(integration_grid, self.forces, self.amplitude, self.lengthscale)
        Kc = _matern52(candidates, self.forces, self.amplitude, self.lengthscale)
        Vg = np.linalg.solve(self._chol, Kg.T)            # (n, n_grid)
        Vc = np.linalg.solve(self._chol, Kc.T)            # (n, n_cand)
        k_post = _matern52(integration_grid, candidates, self.amplitude, self.lengthscale) - Vg.T @ Vc
        var_c = np.clip(self.amplitude ** 2 - np.sum(Vc ** 2, axis=0), 1e-12, None)
        dF = float(np.mean(np.diff(integration_grid))) if integration_grid.size > 1 else 1.0
        return (k_post ** 2).sum(axis=0) * dF / (var_c + self.noise_sd ** 2)

    def straddle(self, candidates: np.ndarray, beta: float = STRADDLE_BETA) -> np.ndarray:
        """Phase B, analytic: beta*sigma(F) - |mu(F) - target|."""
        mu, sd = self.posterior(candidates)
        return beta * sd - np.abs(mu - TARGET_PERFORMANCE_LEVEL)

    def level_ei(self, candidates: np.ndarray, n_samples: int = LEVEL_EI_SAMPLES,
                 seed: int = 0) -> np.ndarray:
        """Phase B, MC: E[ max(0, g(f(c)) - best_g) ] with g(y) = -|y - target|.

        The numpy stand-in for qLogNEI + GenericMCObjective. Note the transform is applied to the
        SAMPLES, never to the training data — the whole point of §2.4.
        """
        mu_c, sd_c = self.posterior(candidates)
        mu_obs, _ = self.posterior(self.forces)
        best = float(np.max(-np.abs(mu_obs - TARGET_PERFORMANCE_LEVEL)))   # "noisy" baseline
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_samples, candidates.size))
        samples = mu_c[None, :] + sd_c[None, :] * z
        g = -np.abs(samples - TARGET_PERFORMANCE_LEVEL)
        return np.maximum(0.0, g - best).mean(axis=0)


# ============================================================================================
# §4  ONE-CALL API
# ============================================================================================

@dataclass(frozen=True)
class BOTrialProposal:
    """One BO trial: the force to run next, plus everything worth logging."""
    force_n: float
    phase: str
    trial_index: int              # 0-based within the 8-trial BO block
    force_cap_n: float
    n_observations: int
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        return (f'BO trial {self.trial_index + 1}/{N_BO_TRIALS}  {self.phase:<16} '
                f'F = {self.force_n:6.2f} N   (cap {self.force_cap_n:.1f} N, '
                f'{self.n_observations} obs, spacing '
                f'{self.diagnostics.get("spacing_used_n", float("nan")):.1f} N '
                f'[{self.diagnostics.get("spacing_rung")}])')


@dataclass(frozen=True)
class BOGroundTruth:
    """x*_BO and its credible interval — the ground truth blocks 5d and 5e are scored against."""
    optimal_force_n: float
    flag: str
    credible_low_n: float
    credible_high_n: float
    force_cap_n: float
    target_reachable: bool
    n_observations: int
    posterior_grid_n: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    posterior_mean: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    posterior_sd: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    hyperparameters: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (f'x*_BO = {self.optimal_force_n:6.2f} N  '
                f'[{self.credible_low_n:.1f}, {self.credible_high_n:.1f}] '
                f'{int(CREDIBLE_MASS * 100)}% CI   flag={self.flag}   '
                f'target reachable: {self.target_reachable}')


def _split(observations: Iterable[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    pairs = [(float(f), float(y)) for f, y in observations]
    if not pairs:
        raise ValueError('the GP needs at least the 8 seed observations from blocks 5a and 5d')
    return (np.array([p[0] for p in pairs], float), np.array([p[1] for p in pairs], float))


def phase_for(trial_index: int) -> str:
    """Fixed 5 + 3 split, so the protocol is identical for every participant."""
    return PHASE_A if trial_index < N_ACTIVE_LEARNING_TRIALS else PHASE_B


def next_bo_force(observations: Iterable[Sequence[float]], max_force_n: float,
                  n_seed_observations: int = 8, backend: str = 'numpy',
                  phase_b_kind: str = 'straddle') -> BOTrialProposal:
    """THE call: give it every (force, rating) pair so far and it returns the next force to run.

    observations        all (force_n, raw TLX performance) pairs collected so far — the 4 assessment
                        tasks (5a), the 4 method trials (5d), then each BO trial as it completes.
    max_force_n         the participant's max push force; sets F_cap for the candidate set.
    n_seed_observations how many of `observations` are seeds rather than BO trials (8 in the
                        protocol, or 9 if you added the duplicate trial of §6.5). The phase split
                        counts from here.
    backend             'numpy'   — the tested reference implementation (§3), no torch needed
                        'botorch' — the production path (§2); requires torch/gpytorch/botorch
    phase_b_kind        'straddle' or 'qlognei' (see §2.4)
    """
    forces, ratings = _split(observations)
    trial_index = max(0, len(forces) - int(n_seed_observations))
    phase = phase_for(trial_index)
    cap = force_cap(max_force_n)

    if backend == 'botorch':
        value, diag = propose_next_force_botorch(forces, ratings, max_force_n, phase,
                                                 phase_b_kind=phase_b_kind)
        return BOTrialProposal(value, phase, trial_index, cap, len(forces), diag)

    if backend != 'numpy':
        raise ValueError(f"backend must be 'numpy' or 'botorch', got {backend!r}")

    candidates, spacing, rung = candidate_forces(forces, max_force_n, phase)
    gp = ReferenceGP(forces, ratings).fit()

    if phase == PHASE_A:
        scores = gp.alc(candidates, integration_grid(gp, max_force_n))
    elif phase_b_kind == 'straddle':
        scores = gp.straddle(candidates)
    else:
        scores = gp.level_ei(candidates)

    best = int(np.argmax(scores))
    mu_cap, _ = gp.posterior(np.array([cap]))
    return BOTrialProposal(
        force_n=float(candidates[best]), phase=phase, trial_index=trial_index,
        force_cap_n=cap, n_observations=len(forces),
        diagnostics={'acquisition_value': float(scores[best]),
                     'acquisition': ('alc' if phase == PHASE_A else phase_b_kind),
                     'n_candidates': int(candidates.size),
                     'spacing_used_n': spacing, 'spacing_rung': RUNG_NAMES[rung],
                     'integration_max_n': float(integration_grid(gp, max_force_n)[-1]),
                     'lengthscale_n': gp.lengthscale, 'amplitude': gp.amplitude,
                     'target_reachable': target_reachable(cap, float(mu_cap[0])),
                     'backend': 'numpy'})


def run_bo_block(seed_observations: Iterable[Sequence[float]], max_force_n: float,
                 rate_trial: Callable[[float], float], backend: str = 'numpy',
                 phase_b_kind: str = 'straddle') -> Tuple[List[Tuple[float, float]],
                                                          List[BOTrialProposal]]:
    """Drive the whole 8-trial block. `rate_trial(force_n) -> rating` is your ROS 2 side.

    Returns (all observations, the 8 proposals). Store both, plus the final posterior (§6.4).
    """
    observations = [(float(f), float(y)) for f, y in seed_observations]
    n_seed = len(observations)
    proposals: List[BOTrialProposal] = []
    for _ in range(N_BO_TRIALS):
        p = next_bo_force(observations, max_force_n, n_seed_observations=n_seed,
                          backend=backend, phase_b_kind=phase_b_kind)
        proposals.append(p)
        observations.append((p.force_n, float(rate_trial(p.force_n))))
    return observations, proposals


def bo_optimal_force(observations: Iterable[Sequence[float]], max_force_n: float,
                     n_paths: int = POSTERIOR_PATHS, seed: int = 0) -> BOGroundTruth:
    """x*_BO from the posterior, with a credible interval (§6.4).

    Point estimate  = first downward crossing of the posterior MEAN.
    Interval        = the crossing distribution over `n_paths` posterior SAMPLE PATHS. Sampling
                      paths rather than inverting the mean handles non-monotone draws gracefully,
                      which matters because the GP is not constrained to be monotone.
    Censoring       = identical to the rule Methods 1-4 use (§3.2), so the ground truth and the
                      four prescriptions are treated the same way: a path already below the target
                      at 0 N censors at 0 N, one still above it at F_cap censors at F_cap.
    """
    forces, ratings = _split(observations)
    cap = force_cap(max_force_n)
    grid = admissible_grid(max_force_n)
    gp = ReferenceGP(forces, ratings).fit()
    mu, sd = gp.posterior(grid)
    _, cov = gp.posterior(grid, full_cov=True)

    def first_crossing(curve: np.ndarray) -> Tuple[float, str]:
        below = np.flatnonzero(curve <= TARGET_PERFORMANCE_LEVEL)
        if below.size == 0:
            return float(grid[-1]), FLAG_CLIPPED_HIGH        # never reaches the target
        if below[0] == 0:
            return float(grid[0]), FLAG_CLIPPED_LOW          # already below at 0 N
        return float(grid[below[0]]), FLAG_OK

    point, flag = first_crossing(mu)

    jitter = 1e-8 * float(np.trace(cov)) / len(grid)
    chol = np.linalg.cholesky(cov + jitter * np.eye(len(grid)))
    rng = np.random.default_rng(seed)
    draws = mu[None, :] + rng.standard_normal((n_paths, len(grid))) @ chol.T
    crossings = np.array([first_crossing(path)[0] for path in draws])
    tail = 0.5 * (1.0 - CREDIBLE_MASS)

    return BOGroundTruth(
        optimal_force_n=point, flag=flag,
        credible_low_n=float(np.quantile(crossings, tail)),
        credible_high_n=float(np.quantile(crossings, 1.0 - tail)),
        force_cap_n=cap,
        target_reachable=target_reachable(cap, float(mu[-1])),
        n_observations=len(forces),
        posterior_grid_n=grid, posterior_mean=mu, posterior_sd=sd,
        hyperparameters={'amplitude': gp.amplitude, 'lengthscale_n': gp.lengthscale,
                         'noise_sd': gp.noise_sd})


__all__ = [
    'PHASE_A', 'PHASE_B', 'N_ACTIVE_LEARNING_TRIALS', 'N_LEVEL_TARGETING_TRIALS', 'N_BO_TRIALS',
    'PRIOR_MEAN_COEFFS', 'PRIOR_AMPLITUDE', 'OBSERVATION_NOISE_SD', 'PRIOR_LENGTHSCALE',
    'LENGTHSCALE_BOUNDS', 'STRADDLE_BETA',
    'admissible_grid', 'candidate_forces', 'min_spacing_for', 'integration_grid',
    'target_reachable', 'prior_mean',
    'make_prior_mean_module', 'build_covar_module', 'build_botorch_gp', 'build_phase_a_acqf', 'build_phase_b_acqf',
    'propose_next_force_botorch', 'AX_GENERATION_NOTES',
    'ReferenceGP', 'BOTrialProposal', 'BOGroundTruth',
    'next_bo_force', 'run_bo_block', 'bo_optimal_force', 'phase_for',
]


# ============================================================================================
# SELF-TEST / DEMO
# ============================================================================================
if __name__ == '__main__':
    print('=' * 96)
    print('§1  CANDIDATE SET — how F_cap and the minimum spacing are enforced')
    print('=' * 96)
    for fmax in (261.35, 178.0, 96.82, 66.77):
        grid = admissible_grid(fmax)
        print(f'  F_max {fmax:6.1f} N -> F_cap {grid[-1]:6.1f} N, {grid.size:>4} grid points, '
              f'spacing {min_spacing_for(grid[-1]):.1f} N   (300 N is not in the set)')

    print('\n  the fallback ladder, on the tightest participant in n=28 (user 12, F_cap 66.8 N):')
    sampled, cap = [0.0, 40.0, 60.0, 66.77], 66.77
    for step in range(9):
        cands, sp, rung = candidate_forces(sampled, cap)
        pick = float(cands[len(cands) // 2])
        print(f'    step {step + 1}: {cands.size:>3} candidates, spacing {sp:4.1f} N '
              f'[{RUNG_NAMES[rung]:<18}] -> {pick:6.2f} N')
        sampled.append(pick)

    print('\n' + '=' * 96)
    print('§4  full 8-trial block on a synthetic participant  (numpy backend)')
    print('=' * 96)
    TRUE = (-0.0025, -0.12, 72.0)          # a plausible individual curve, crossing ~63 N
    truth_at = lambda F: float(np.polyval(TRUE, F))
    rng = np.random.default_rng(7)
    rate = lambda F: truth_at(F) + rng.normal(0.0, OBSERVATION_NOISE_SD)
    F_MAX = 178.0

    seeds = [(F, rate(F)) for F in (0.0, 40.0, 60.0, 80.0)]          # block 5a
    seeds += [(F, rate(F)) for F in (58.0, 63.0, 55.0, 71.0)]        # block 5d
    print(f'  true crossing of 48.5: {np.max([r.real for r in np.roots([TRUE[0], TRUE[1], TRUE[2] - 48.5]) if abs(r.imag) < 1e-9]):.2f} N')
    print(f'  8 seed observations:   ' +
          '  '.join(f'{f:.0f}N:{y:.0f}' for f, y in seeds))

    obs, proposals = run_bo_block(seeds, F_MAX, rate)
    print()
    for p in proposals:
        print('  ' + str(p))

    truth = bo_optimal_force(obs, F_MAX)
    print(f'\n  {truth}')
    print(f'  hyperparameters: ' + '  '.join(f'{k}={v:.2f}' for k, v in truth.hyperparameters.items()))

    print('\n' + '=' * 96)
    print('§3  acquisition sanity checks')
    print('=' * 96)
    gp = ReferenceGP(*_split(seeds)).fit()
    cands_a, _, _ = candidate_forces([f for f, _ in seeds], F_MAX, PHASE_A)
    cands_b, _, _ = candidate_forces([f for f, _ in seeds], F_MAX, PHASE_B)
    igrid = integration_grid(gp, F_MAX)
    trunc = ('truncated where the rating floors' if igrid[-1] < truth.force_cap_n - 1e-9
             else 'nothing floored yet, so the full range')
    print(f'  integration measure  0-{igrid[-1]:.0f} N of the admissible '
          f'0-{truth.force_cap_n:.0f} N   ({trunc})')
    print(f'  phase A (ALC)        argmax {cands_a[int(np.argmax(gp.alc(cands_a, igrid)))]:6.1f} N'
          f'   -> away from the crowded 55-71 N seed cluster')
    print(f'  phase B (straddle)   argmax {cands_b[np.argmax(gp.straddle(cands_b))]:6.1f} N')
    print(f'  phase B (level EI)   argmax {cands_b[np.argmax(gp.level_ei(cands_b))]:6.1f} N'
          f'   -> the two phase-B criteria should broadly agree')
    mu, sd = gp.posterior(admissible_grid(F_MAX))
    print(f'  posterior at F_cap:  mu {mu[-1]:.1f} +- {sd[-1]:.1f}   '
          f'target reachable: {target_reachable(truth.force_cap_n, float(mu[-1]))}')

    print('\n' + '=' * 96)
    print('§1b  the fallback ladder, forced: a participant who can barely push at all')
    print('=' * 96)
    TINY = 20.0
    print(f'  F_max {TINY:.0f} N -> F_cap {force_cap(TINY):.0f} N, '
          f'phase-A spacing {min_spacing_for(force_cap(TINY)):.1f} N')
    sampled = [0.0, 20.0]                       # 40/60/80 N all clip onto the cap
    for step in range(8):
        cands, sp, rung = candidate_forces(sampled, TINY, PHASE_A)
        pick = float(cands[len(cands) // 2])
        note = '' if rung == 0 else '   <-- ladder engaged'
        print(f'    trial {step + 1}: {cands.size:>3} candidates, spacing {sp:4.1f} N '
              f'[{RUNG_NAMES[rung]:<18}] -> {pick:5.1f} N{note}')
        sampled.append(pick)

    print('\n' + '=' * 96)
    print('COHORT CHECK — the full block for all 26 n=28 participants, real F_cap and real seeds')
    print('=' * 96)
    try:
        import pandas as pd
        ref = pd.read_csv('user_study_optimal_forces.csv')
    except Exception as exc:                                  # pragma: no cover
        print(f'  skipped ({exc}); run evaluate_full_circle.py first')
        raise SystemExit(0)

    rows, rungs_hit, over_cap = [], {}, 0
    for user, g in ref.groupby('user'):
        cap = float(g['force_cap_N'].iloc[0])
        qf = g[g['method'] == 'Quadratic-MaxForce'].iloc[0]
        coeffs = (float(qf['coef_a']), float(qf['coef_b']), float(qf['coef_c']))
        x_true = float(qf['raw_crossing_N'])
        if not np.isfinite(x_true) or not (0.0 <= x_true <= cap):
            continue                                          # censored: no interior truth to hit
        r = np.random.default_rng(int(user))
        rate = lambda F: float(np.polyval(coeffs, F)) + r.normal(0.0, OBSERVATION_NOISE_SD)
        seed_forces = sorted(set([min(f, cap) for f in (0.0, 40.0, 60.0, 80.0)]) |
                             {float(v) for v in g['optimal_force_N']})
        obs, props = run_bo_block([(F, rate(F)) for F in seed_forces], cap, rate)
        for p in props:
            rungs_hit[p.diagnostics['spacing_rung']] = \
                rungs_hit.get(p.diagnostics['spacing_rung'], 0) + 1
            over_cap += int(p.force_n > cap + 1e-9 or p.force_n < 0.0)
        t_bo = bo_optimal_force(obs, cap)
        rows.append((user, cap, x_true, t_bo.optimal_force_n,
                     t_bo.credible_low_n <= x_true <= t_bo.credible_high_n,
                     t_bo.credible_high_n - t_bo.credible_low_n))

    err = np.array([r[3] - r[2] for r in rows])
    print(f'  {len(rows)} participants with an interior ground truth')
    print(f'  proposals outside [0, F_cap]:            {over_cap}   <-- must be 0')
    print(f'  spacing rungs used:                      {rungs_hit}')
    print(f'  x*_BO recovery:  bias {err.mean():+.2f} N   MAE {np.abs(err).mean():.2f} N   '
          f'RMSE {np.sqrt((err ** 2).mean()):.2f} N')
    print(f'  90% CI covers the truth for             {sum(r[4] for r in rows)}/{len(rows)} '
          f'participants  (nominal 90%)')
    print(f'  mean CI width:                           {np.mean([r[5] for r in rows]):.1f} N')
