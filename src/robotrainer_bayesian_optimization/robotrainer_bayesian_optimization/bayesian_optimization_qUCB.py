import math
from typing import Any, Callable

from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.model_spec import GeneratorSpec
from botorch.acquisition.input_constructors import acqf_input_constructor
from botorch.acquisition.monte_carlo import (
    SampleReducingMCAcquisitionFunction,
)
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.acquisition.utils import get_infeasible_cost
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from torch import Tensor


class qUpperConfidenceBound(SampleReducingMCAcquisitionFunction):
    r"""MC-based batch Upper Confidence Bound.

    Uses a reparameterization to extend UCB to qUCB for q > 1 (See Appendix A
    of [Wilson2017reparam].)

    `qUCB = E(max(mu + |Y_tilde - mu|))`, where `Y_tilde ~ N(mu, beta pi/2 Sigma)`
    and `f(X)` has distribution `N(mu, Sigma)`.

    Constraints should be provided as a `ConstrainedMCObjective`.
    Passing `constraints` as an argument is not supported. This is because
    `SampleReducingMCAcquisitionFunction` computes the acquisition values on the sample
    level and then weights the sample-level acquisition values by a soft feasibility
    indicator. Hence, it expects non-log acquisition function values to be
    non-negative. `qUpperConfidenceBound` acquisition values can be negative, so we
    instead use a `ConstrainedMCObjective` which applies constraints to the objectives
    (e.g. before computing the acquisition function) and shifts negative objective
    values using an infeasible cost to ensure non-negativity (before applying
    constraints and shifting them back).

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qUCB = qUpperConfidenceBound(model, 0.1, sampler)
        >>> qucb = qUCB(test_X)
    """

    def __init__(
        self,
        model: Model,
        beta: float,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
    ) -> None:
        r"""q-Upper Confidence Bound.

        Args:
            model: A fitted model.
            beta: Controls tradeoff between mean and standard deviation in UCB.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation but have not yet
                been evaluated. Concatenated into X upon forward call. Copied and set to
                have no gradient.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.beta_prime = self._get_beta_prime(beta=beta)

    def _get_beta_prime(self, beta: float) -> float:
        return math.sqrt(beta * math.pi / 2)

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qUpperConfidenceBound per sample on the candidate set `X`.

        Args:
            obj: A `sample_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of acquisition values.
        """
        mean = obj.mean(dim=0)
        return mean + self.beta_prime * (obj - mean).abs()


@acqf_input_constructor(qUpperConfidenceBound)
def construct_inputs_qUCB(
    model: Model,
    objective: MCAcquisitionObjective | None = None,
    posterior_transform: PosteriorTransform | None = None,
    X_pending: Tensor | None = None,
    sampler: MCSampler | None = None,
    X_baseline: Tensor | None = None,
    constraints: list[Callable[[Tensor], Tensor]] | None = None,
    beta: float = 200000000,
) -> dict[str, Any]:
    r"""Construct kwargs for the `qUpperConfidenceBound` constructor.

    Args:
        model: The model to be used in the acquisition function.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        X_baseline: A `batch_shape x r x d`-dim Tensor of `r` design points
            that have already been observed. These points are used to
            compute with infeasible cost when there are constraints.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension `sample_shape x batch-shape x q x m`-dim to a
            `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        beta: Controls tradeoff between mean and standard deviation in UCB.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    if constraints is not None:
        if X_baseline is None:
            raise ValueError("Constraints require an X_baseline.")
        if objective is None:
            objective = IdentityMCObjective()
        objective = ConstrainedMCObjective(
            objective=objective,
            constraints=constraints,
            infeasible_cost=get_infeasible_cost(
                X=X_baseline, model=model, objective=objective
            ),
        )
    return {
        "model": model,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
        "sampler": sampler,
        "beta": beta,
    }


def construct_generation_strategy(
    generator_spec: GeneratorSpec,
    node_name: str,
) -> GenerationStrategy:
    """Constructs a Center + Sobol + Modular BoTorch `GenerationStrategy`
    using the provided `generator_spec` for the Modular BoTorch node.
    """
    botorch_node = GenerationNode(
        node_name=node_name,
        model_specs=[generator_spec],
    )
    # sobol_node = GenerationNode(
    #     node_name="Sobol",
    #     model_specs=[
    #         GeneratorSpec(
    #             model_enum=Generators.SOBOL,
    #             # Let's use model_kwargs to set the random seed.
    #             model_kwargs={"seed": 42},
    #         ),
    #     ],
    #     transition_criteria=[
    #         # Transition to BoTorch node once there are 5 trials on the experiment.
    #         MinTrials(
    #             threshold=4,
    #             transition_to=botorch_node.node_name,
    #             use_all_trials_in_exp=True,
    #         )
    #     ],
    # )
    # Center node is a customized node that uses a simplified logic and has a
    # built-in transition criteria that transitions after generating once.
    center_node = CenterGenerationNode(next_node_name=botorch_node.node_name)
    return GenerationStrategy(
        name=f"Center+qUpperConfidenceBound", nodes=[center_node, botorch_node]
    )
    # return GenerationStrategy(name=f"{node_name}", nodes=[botorch_node])
