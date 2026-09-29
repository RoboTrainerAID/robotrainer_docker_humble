from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from copy import deepcopy
from functools import partial
from typing import Any, TypeVar

import torch
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials
from ax.modelbridge.registry import Generators
from botorch import settings
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch.acquisition.cached_cholesky import CachedCholeskyMCSamplerMixin
from botorch.acquisition.input_constructors import acqf_input_constructor
from botorch.acquisition.logei import (
    TAU_MAX,
    TAU_RELU,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.monte_carlo import (
    SampleReducingMCAcquisitionFunction,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.acquisition.risk_measures import RiskMeasureMCObjective
from botorch.acquisition.utils import (
    compute_best_feasible_objective,
    prune_inferior_points,
)
from botorch.acquisition.input_constructors import construct_inputs_qNEI
from botorch.exceptions.errors import BotorchError
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.containers import BotorchContainer
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.safe_math import (
    fatmax,
    log_fatplus,
    log_softplus,
    logmeanexp,
    smooth_amax,
)
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from torch import Tensor

T = TypeVar("T")
MaybeDict = T | dict[Hashable, T]
TOptimizeObjectiveKwargs = (
    None
    | MCAcquisitionObjective
    | PosteriorTransform
    | tuple[Tensor, Tensor]
    | dict[int, float]
    | bool
    | int
    | dict[str, Any]
    | Callable[[Tensor], Tensor]
    | Tensor
)


class LogImprovementMCAcquisitionFunction(SampleReducingMCAcquisitionFunction):
    r"""
    Abstract base class for Monte-Carlo-based batch LogEI acquisition functions.
    """

    _log: bool = True

    def __init__(
        self,
        model: Model,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        eta: Tensor | float = 1e-3,
        fat: bool = True,
        tau_max: float = TAU_MAX,
    ) -> None:
        r"""Constructor of the base class for LogEI acquisition functions.

        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples. If not given,
                a sampler is generated using ``get_sampler``.
                NOTE: For posteriors that do not support base samples,
                a sampler compatible with intended use case must be provided.
                See ``ForkedRNGSampler`` and ``StochasticSampler`` as examples.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to ``IdentityMCObjective()``.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A ``batch_shape, m x d``-dim Tensor of ``m`` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated.
            constraints: A list of constraint callables which map a Tensor
                of posterior samples of dimension
                ``sample_shape x batch-shape x q x m``-dim to a
                ``sample_shape x batch-shape x q``-dim Tensor. The
                associated constraints
                are satisfied if ``constraint(samples) < 0``.
            eta: Temperature parameter(s) governing the smoothness of
                the sigmoid approximation to the constraint indicators.
                See the docs of ``compute_(log_)constraint_indicator``
                for more details on this parameter.
            fat: Toggles the logarithmic / linear asymptotic behavior of the smooth
                approximation to the ReLU.
            tau_max: Temperature parameter controlling the sharpness of the
                approximation to the ``max`` operator over the ``q`` candidate points.
        """
        if isinstance(objective, ConstrainedMCObjective):
            raise BotorchError(
                "Log-Improvement should not be used with `ConstrainedMCObjective`."
                "Please pass the `constraints` directly to the constructor of the "
                "acquisition function."
            )
        q_reduction = partial(fatmax if fat else smooth_amax, tau=tau_max)
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            sample_reduction=logmeanexp,
            q_reduction=q_reduction,
            constraints=constraints,
            eta=eta,
            fat=fat,
        )
        self.tau_max = tau_max


class qLogNoisyExpectedImprovement(
    LogImprovementMCAcquisitionFunction, CachedCholeskyMCSamplerMixin
):
    r"""MC-based batch Log Noisy Expected Improvement.

    This function does not assume a ``best_f`` is known (which would require
    noiseless observations). Instead, it uses samples from the joint posterior
    over the ``q`` test points and previously observed points. A smooth approximation
    to the canonical improvement over previously observed points is computed
    for each sample and the logarithm of the average is returned.

    See [Ament2023logei]_ for details. Formally,

    ``qLogNEI(X) ~ log(qNEI(X)) = Log E(max(max Y - max Y_baseline, 0))``,

    where ``(Y, Y_baseline) ~ f((X, X_baseline)), X = (x_1,...,x_q)``.

    For optimizing a batch of ``q > 1`` points using sequential greedy optimization,
    the incremental improvement from the latest point is computed and returned by
    default. I.e., the pending points are treated as ``X_baseline``.
    Often, the incremental EI is easier to optimize.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qLogNEI = qLogNoisyExpectedImprovement(model, train_X, sampler)
        >>> acqval = qLogNEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        eta: Tensor | float = 1e-3,
        fat: bool = True,
        prune_baseline: bool = True,
        cache_root: bool | None = None,
        tau_max: float = TAU_MAX,
        tau_relu: float = TAU_RELU,
        marginalize_dim: int | None = None,
        incremental: bool = True,
    ) -> None:
        r"""q-Noisy Expected Improvement.

        Args:
            model: A fitted model.
            X_baseline: A ``batch_shape x r x d``-dim Tensor of
                ``r`` design points that have already been observed.
                These points are considered as the potential best design
                point.
            sampler: The sampler used to draw base samples. See
                ``MCAcquisitionFunction`` more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to ``IdentityMCObjective()``.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A ``batch_shape x m x d``-dim Tensor of ``m`` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated. Concatenated into ``X`` upon
                forward call. Copied and set to have no gradient.
            constraints: A list of constraint callables which map a Tensor
                of posterior samples of dimension
                ``sample_shape x batch-shape x q x m``-dim to a
                ``sample_shape x batch-shape x q``-dim Tensor. The
                associated constraints are satisfied if
                ``constraint(samples) < 0``.
            eta: Temperature parameter(s) governing the smoothness of
                the sigmoid approximation to the constraint indicators.
                See the docs of
                ``compute_(log_)smoothed_constraint_indicator`` for details.
            fat: Toggles the logarithmic / linear asymptotic behavior of the smooth
                approximation to the ReLU.
            prune_baseline: If True, remove points in ``X_baseline`` that are
                highly unlikely to be the best point. This can significantly
                improve performance and is generally recommended. In order to
                customize pruning parameters, instead manually call
                ``botorch.acquisition.utils.prune_inferior_points`` on ``X_baseline``
                before instantiating the acquisition function.
            cache_root: A boolean indicating whether to cache the root
                decomposition over ``X_baseline`` and use low-rank updates.
            tau_max: Temperature parameter controlling the sharpness of the smooth
                approximations to max.
            tau_relu: Temperature parameter controlling the sharpness of the smooth
                approximations to ReLU.
            marginalize_dim: The dimension to marginalize over.
            incremental: Whether to compute incremental EI over the pending points
                or compute EI of the joint batch improvement (including pending
                points).

        TODO: similar to qNEHVI, when we are using sequential greedy candidate
        selection, we could incorporate pending points X_baseline and compute
        the incremental q(Log)NEI from the new point. This would greatly increase
        efficiency for large batches.
        """
        # TODO: separate out baseline variables initialization and other functions
        # in qNEI to avoid duplication of both code and work at runtime.
        self.incremental = incremental

        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            # we set X_pending in init_baseline for incremental NEI
            X_pending=X_pending if not incremental else None,
            constraints=constraints,
            eta=eta,
            fat=fat,
            tau_max=tau_max,
        )
        self.tau_relu = tau_relu
        self.prune_baseline = prune_baseline
        self.marginalize_dim = marginalize_dim
        if incremental:
            self.X_pending = None  # required to initialize attribute for optimize_acqf
        self._init_baseline(
            model=model,
            X_baseline=X_baseline,
            # This is ignored in incremental=False
            X_pending=X_pending,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            cache_root=cache_root,
        )

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qLogNoisyExpectedImprovement per sample on the candidate set ``X``.

        Args:
            obj: ``mc_shape x batch_shape x q``-dim Tensor of MC objective values.

        Returns:
            A ``sample_shape x batch_shape x q``-dim Tensor of log noisy
            expected smoothed improvement values.
        """
        return _log_improvement(
            Y=obj,
            best_f=self.compute_best_f(obj),
            tau=self.tau_relu,
            fat=self._fat,
        )

    def _init_baseline(
        self,
        model: Model,
        X_baseline: Tensor,
        X_pending: Tensor | None = None,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        cache_root: bool | None = None,
    ) -> None:
        CachedCholeskyMCSamplerMixin.__init__(
            self, model=model, cache_root=cache_root, sampler=sampler
        )
        if self.prune_baseline:
            X_baseline = prune_inferior_points(
                model=model,
                X=X_baseline,
                objective=objective,
                posterior_transform=posterior_transform,
                marginalize_dim=self.marginalize_dim,
                constraints=self._constraints,
            )
        self.register_buffer("_X_baseline", X_baseline)
        # full_X_baseline is the set of points that should be considered as the
        # incumbent. For incremental EI, this contains the previously evaluated
        # points (X_baseline) and pending points (X_pending). For non-incremental
        # EI, this contains the  previously evaluated points (X_baseline).
        if X_pending is not None and self.incremental:
            full_X_baseline = torch.cat([X_baseline, X_pending], dim=-2)
        else:
            full_X_baseline = X_baseline
        self.register_buffer("_full_X_baseline", full_X_baseline)
        # registering buffers for _get_samples_and_objectives in the next ``if`` block
        self.register_buffer("baseline_samples", None)
        self.register_buffer("baseline_obj", None)
        if self._cache_root:
            self.q_in = -1
            # set baseline samples
            with torch.no_grad():  # this is _get_samples_and_objectives(X_baseline)
                posterior = self.model.posterior(
                    self.X_baseline, posterior_transform=self.posterior_transform
                )
                # Note: The root decomposition is cached in two different places. It
                # may be confusing to have two different caches, but this is not
                # trivial to change since each is needed for a different reason:
                # - LinearOperator caching to ``posterior.mvn`` allows for reuse within
                #   this function, which may be helpful if the same root decomposition
                #   is produced by the calls to ``self.base_sampler`` and
                #   ``self._cache_root_decomposition``.
                # - self._baseline_L allows a root decomposition to be persisted outside
                #   this method.
                self.baseline_samples = self.get_posterior_samples(posterior)
                self.baseline_obj = self.objective(
                    self.baseline_samples, X=self.X_baseline
                )

            # We make a copy here because we will write an attribute ``base_samples``
            # to ``self.base_sampler.base_samples``, and we don't want to mutate
            # ``self.sampler``.
            self.base_sampler = deepcopy(self.sampler)
            self.register_buffer(
                "_baseline_best_f",
                self._compute_best_feasible_objective(
                    samples=self.baseline_samples, obj=self.baseline_obj
                ),
            )
            self._baseline_L = self._compute_root_decomposition(posterior=posterior)

    @property
    def X_baseline(self) -> Tensor:
        """Returns the set of points that should be considered as the incumbent.

        For incremental EI, this contains the previously evaluated points
        (X_baseline) and pending points (X_pending). For non-incremental
        EI, this contains the  previously evaluated points (X_baseline).
        """
        return self._full_X_baseline

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        r"""Informs the acquisition function about pending design points.

        Here pending points are concatenated with X_baseline and incremental
        NEI is computed.

        Args:
            X_pending: ``n x d`` Tensor with ``n`` ``d``-dim design points that have
                been submitted for evaluation but have not yet been evaluated.
        """
        if not self.incremental:
            return super().set_X_pending(X_pending=X_pending)
        if X_pending is None:
            if not hasattr(self, "_full_X_baseline") or (
                self._full_X_baseline.shape[-2] == self._X_baseline.shape[-2]
            ):
                return
            else:
                # reset pending points
                X_pending = None
        self._init_baseline(
            model=self.model,
            X_baseline=self._X_baseline,
            X_pending=X_pending,
            sampler=self.sampler,
            objective=self.objective,
            posterior_transform=self.posterior_transform,
            cache_root=self._cache_root,
        )

    def compute_best_f(self, obj: Tensor) -> Tensor:
        """Computes the best (feasible) noisy objective value.

        Args:
            obj: ``sample_shape x batch_shape x q``-dim Tensor of objectives in forward.

        Returns:
            A ``sample_shape x batch_shape``-dim Tensor of best feasible objectives.
        """
        if self._cache_root:
            val = self._baseline_best_f
        else:
            val = self._compute_best_feasible_objective(
                samples=self.baseline_samples, obj=self.baseline_obj
            )
        # ensuring shape, dtype, device compatibility with obj
        n_sample_dims = len(self.sample_shape)
        view_shape = torch.Size(
            [
                *val.shape[:n_sample_dims],  # sample dimensions
                *(1,) * (obj.ndim - val.ndim - 1),  # pad to match obj without ``q``-dim
                *val.shape[n_sample_dims:],  # the rest
            ]
        )
        return val.view(view_shape).to(obj)  # obj.shape[:-1], i.e. without ``q``-dim`

    def _get_samples_and_objectives(self, X: Tensor) -> tuple[Tensor, Tensor]:
        r"""Compute samples at new points, using the cached root decomposition.

        Args:
            X: A ``batch_shape x q x d``-dim tensor of inputs.

        Returns:
            A two-tuple ``(samples, obj)``, where ``samples`` is a tensor of posterior
            samples with shape ``sample_shape x batch_shape x q x m``, and ``obj`` is a
            tensor of MC objective values with shape ``sample_shape x batch_shape x q``.
        """
        n_baseline, q = self.X_baseline.shape[-2], X.shape[-2]
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        # TODO: Implement more efficient way to compute posterior over both training and
        # test points in GPyTorch (https://github.com/cornellius-gp/gpytorch/issues/567)
        posterior = self.model.posterior(
            X_full, posterior_transform=self.posterior_transform
        )
        if not self._cache_root:
            samples_full = super().get_posterior_samples(posterior)
            obj_full = self.objective(samples_full, X=X_full)
            # Calculate the positive index for splitting the samples & objective values.
            split_dim = len(obj_full.shape) - 1
            # assigning baseline buffers so ``best_f`` can be computed in
            # _sample_forward
            self.baseline_samples, samples = samples_full.split(
                [n_baseline, q], dim=split_dim
            )
            self.baseline_obj, obj = obj_full.split([n_baseline, q], dim=split_dim)
            return samples, obj

        # handle one-to-many input transforms
        n_plus_q = X_full.shape[-2]
        n_w = posterior._extended_shape()[-2] // n_plus_q
        q_in = q * n_w
        self._set_sampler(q_in=q_in, posterior=posterior)
        samples = self._get_f_X_samples(posterior=posterior, q_in=q_in)
        obj = self.objective(samples, X=X_full[..., -q:, :])
        return samples, obj

    def _compute_best_feasible_objective(self, samples: Tensor, obj: Tensor) -> Tensor:
        r"""Computes best feasible objective value from samples.

        Args:
            samples: ``sample_shape x batch_shape x q x m``-dim posterior samples.
            obj: A ``sample_shape x batch_shape x q``-dim Tensor of MC objective values.

        Returns:
            A ``sample_shape x batch_shape``-dim Tensor of best feasible objectives.
        """
        return compute_best_feasible_objective(
            samples=samples,
            obj=obj,
            constraints=self._constraints,
            model=self.model,
            objective=self.objective,
            posterior_transform=self.posterior_transform,
            X_baseline=self.X_baseline,
        )


class qNegIntegratedPosteriorVariance(AcquisitionFunction):
    r"""Batch Integrated Negative Posterior Variance for Active Learning.

    This acquisition function quantifies the (negative) integrated posterior variance
    (excluding observation noise, computed using MC integration) of the model.
    In that, it is a proxy for global model uncertainty, and thus purely focused on
    "exploration", rather the "exploitation" of many of the classic Bayesian
    Optimization acquisition functions.

    See [Seo2014activedata]_, [Chen2014seqexpdesign]_, and [Binois2017repexp]_.
    """

    def __init__(
        self,
        model: Model,
        mc_points: Tensor,
        sampler: MCSampler | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
    ) -> None:
        r"""q-Integrated Negative Posterior Variance.

        Args:
            model: A fitted model.
            mc_points: A `batch_shape x N x d` tensor of points to use for
                MC-integrating the posterior variance. Usually, these are qMC
                samples on the whole design space, but biased sampling directly
                allows weighted integration of the posterior variance.
            sampler: The sampler used for drawing fantasy samples. In the basic setting
                of a standard GP (default) this is a dummy, since the variance of the
                model after conditioning does not actually depend on the sampled values.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            X_pending: A `n' x d`-dim Tensor of `n'` design points that have
                points that have been submitted for function evaluation but
                have not yet been evaluated.
        """
        super().__init__(model=model)
        self.posterior_transform = posterior_transform
        if sampler is None:
            # If no sampler is provided, we use the following dummy sampler for the
            # fantasize() method in forward. IMPORTANT: This assumes that the posterior
            # variance does not depend on the samples y (only on x), which is true for
            # standard GP models, but not in general (e.g. for other likelihoods or
            # heteroskedastic GPs using a separate noise model fit on data).
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1]))
        self.sampler = sampler
        self.X_pending = X_pending
        self.register_buffer("mc_points", mc_points)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        # Construct the fantasy model (we actually do not use the full model,
        # this is just a convenient way of computing fast posterior covariances
        fantasy_model = self.model.fantasize(
            X=X,
            sampler=self.sampler,
        )

        bdims = tuple(1 for _ in X.shape[:-2])
        print(f"X shape: {X.shape}, mc_points shape: {self.mc_points.shape}")
        print(f"bdims: {bdims}")
        if self.model.num_outputs > 1:
            # We use q=1 here b/c ScalarizedObjective currently does not fully exploit
            # LinearOperator operations and thus may be slow / overly memory-hungry.
            # TODO (T52818288): Properly use LinearOperators in scalarize_posterior
            mc_points = self.mc_points.view(-1, *bdims, 1, X.size(-1))
        else:
            # While we only need marginal variances, we can evaluate for q>1
            # b/c for GPyTorch models lazy evaluation can make this quite a bit
            # faster than evaluating in t-batch mode with q-batch size of 1
            mc_points = self.mc_points.view(*bdims, -1, X.size(-1))

        # evaluate the posterior at the grid points
        with settings.propagate_grads(True):
            posterior = fantasy_model.posterior(
                mc_points, posterior_transform=self.posterior_transform
            )

        neg_variance = posterior.variance.mul(-1.0)

        if self.posterior_transform is None:
            # if single-output, shape is 1 x batch_shape x num_grid_points x 1
            return neg_variance.mean(dim=-2).squeeze(-1).squeeze(0)
        else:
            # if multi-output + obj, shape is num_grid_points x batch_shape x 1 x 1
            return neg_variance.mean(dim=0).squeeze(-1).squeeze(-1)


def _log_improvement(
    Y: Tensor,
    best_f: Tensor,
    tau: float | Tensor,
    fat: bool,
) -> Tensor:
    """Computes the logarithm of the softplus-smoothed improvement, i.e.
    ``log_softplus(Y - best_f, beta=(1 / tau))``.
    Note that softplus is an approximation to the regular ReLU objective
    whose maximum pointwise approximation error is linear with respect to
    tau as tau goes to zero.

    Args:
        obj: ``mc_samples x batch_shape x q``-dim Tensor of output samples.
        best_f: Best previously observed objective value(s), broadcastable with
            ``mc_samples x batch_shape``-dim Tensor, i.e. ``obj``'s dims without ``q``.
        tau: Temperature parameter for smooth approximation of ReLU.
            as ``tau -> 0``, maximum pointwise approximation error is linear
            w.r.t. ``tau``.
        fat: Toggles the logarithmic / linear asymptotic behavior of the
            smooth approximation to ReLU.

    Returns:
        A ``mc_samples x batch_shape x q``-dim Tensor of improvement values.
    """
    log_soft_clamp = log_fatplus if fat else log_softplus
    Z = Y - best_f.unsqueeze(-1).to(Y)
    return log_soft_clamp(Z, tau=tau)  # ~ ((Y - best_f) / Y_std).clamp(0)


def _get_dataset_field(
    dataset: MaybeDict[SupervisedDataset],
    fieldname: str,
    transform: Callable[[BotorchContainer], Any] | None = None,
    join_rule: Callable[[Sequence[Any]], Any] | None = None,
    first_only: bool = False,
    assert_shared: bool = False,
) -> Any:
    r"""Convenience method for extracting a given field from one or more datasets."""
    if isinstance(dataset, dict):
        if assert_shared and not _field_is_shared(dataset, fieldname):
            raise ValueError(f"Field `{fieldname}` must be shared.")

        if not first_only:
            fields = (
                _get_dataset_field(d, fieldname, transform) for d in dataset.values()
            )
            return join_rule(tuple(fields)) if join_rule else tuple(fields)

        dataset = next(iter(dataset.values()))

    field = getattr(dataset, fieldname)
    return transform(field) if transform else field


def _field_is_shared(
    datasets: Iterable[SupervisedDataset] | dict[Hashable, SupervisedDataset],
    fieldname: str,
) -> bool:
    r"""Determines whether or not a given field is shared by all datasets."""
    if isinstance(datasets, dict):
        datasets = datasets.values()

    base = None
    for dataset in datasets:
        if not hasattr(dataset, fieldname):
            raise AttributeError(f"{type(dataset)} object has no field `{fieldname}`.")

        obj = getattr(dataset, fieldname)
        if base is None:
            base = obj
        elif isinstance(base, Tensor):
            if not torch.equal(base, obj):
                return False
        elif base != obj:  # pragma: no cover
            return False

    return True


@acqf_input_constructor(qLogNoisyExpectedImprovement)
def construct_inputs_qLogNEI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective: MCAcquisitionObjective | None = None,
    posterior_transform: PosteriorTransform | None = None,
    X_pending: Tensor | None = None,
    sampler: MCSampler | None = None,
    X_baseline: Tensor | None = None,
    prune_baseline: bool | None = True,
    cache_root: bool | None = None,
    constraints: list[Callable[[Tensor], Tensor]] | None = None,
    eta: Tensor | float = 1e-3,
    fat: bool = True,
    tau_max: float = TAU_MAX,
    tau_relu: float = TAU_RELU,
    incremental: bool = True,
):
    r"""Construct kwargs for the ``qLogNoisyExpectedImprovement`` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        X_baseline: A ``batch_shape x r x d``-dim Tensor of ``r`` design points
            that have already been observed. These points are considered as
            the potential best design point. If omitted, checks that all
            training_data have the same input features and take the first ``X``.
        prune_baseline: If True, remove points in ``X_baseline`` that are
            highly unlikely to be the best point. This can significantly
            improve performance and is generally recommended.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension ``sample_shape x batch-shape x q x m``-dim to a
            ``sample_shape x batch-shape x q``-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        eta: Temperature parameter(s) governing the smoothness of the sigmoid
            approximation to the constraint indicators. For more details, on this
            parameter, see the docs of ``compute_smoothed_feasibility_indicator``.
        fat: Toggles the use of the fat-tailed non-linearities to smoothly approximate
            the constraints indicator function.
        tau_max: Temperature parameter controlling the sharpness of the smooth
            approximations to max.
        tau_relu: Temperature parameter controlling the sharpness of the smooth
            approximations to ReLU.
        incremental: Whether to compute incremental EI over the pending points
            or compute EI of the joint batch improvement (including pending
            points).

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    return {
        **construct_inputs_qNEI(
            model=model,
            training_data=training_data,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            sampler=sampler,
            X_baseline=X_baseline,
            prune_baseline=prune_baseline,
            cache_root=cache_root,
            constraints=constraints,
            eta=eta,
        ),
        "fat": fat,
        "tau_max": tau_max,
        "tau_relu": tau_relu,
        "incremental": incremental,
    }

@acqf_input_constructor(qNegIntegratedPosteriorVariance)
def construct_inputs_NIPV(
    model: Model,
    bounds: list[tuple[float, float]],
    num_mc_points: int = 1,
    X_pending: Tensor | None = None,
    posterior_transform: PosteriorTransform | None = None,
) -> dict[str, Any]:
    """Construct inputs for qNegIntegratedPosteriorVariance."""
    bounds = torch.as_tensor(bounds).to(model.train_targets).T
    mc_points = draw_sobol_samples(bounds=bounds, n=num_mc_points, q=1).squeeze(-2)
    inputs = {
        "model": model,
        "mc_points": mc_points,
        "X_pending": X_pending,
        "posterior_transform": posterior_transform,
    }
    return inputs


def construct_generation_strategy(
    node_name: str,
    num_exploration: int
) -> GenerationStrategy:
    """Constructs a Center + Modular BoTorch `GenerationStrategy`
    using the provided `generator_spec` for the Modular BoTorch node.
    """
    exploration_node = GenerationNode(
        node_name=node_name,
        model_specs=GeneratorSpec(
            model_enum=Generators.BOTORCH_MODULAR,
            model_kwargs={
                "botorch_acqf_class": qNegIntegratedPosteriorVariance
            },
        ),
        transition_criteria=[
            # Transition to BoTorch node once there are 5 trials on the experiment.
            MinTrials(
                threshold=num_exploration,
                transition_to=exploration_node.node_name,
                use_all_trials_in_exp=True,
            )
        ],
    )

    exploitation_node = GenerationNode(
        node_name=node_name,
        model_specs=GeneratorSpec(
            model_enum=Generators.BOTORCH_MODULAR,
            model_kwargs={
                "botorch_acqf_class": qLogNoisyExpectedImprovement
            },
        )
    )

    return GenerationStrategy(
        name=f"qNegIntegratedPosteriorVariance+qLogNoisyExpectedImprovement",
        nodes=[exploration_node, exploitation_node],
    )
