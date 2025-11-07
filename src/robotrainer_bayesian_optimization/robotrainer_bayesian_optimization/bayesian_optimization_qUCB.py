import math
import os
import re
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials
from ax.modelbridge.registry import Generators
from ax.storage.botorch_modular_registry import register_acquisition_function
from botorch.acquisition.input_constructors import acqf_input_constructor
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
        name=f"Center+qUpperConfidenceBound",
        nodes=[center_node, botorch_node]
    )
    # return GenerationStrategy(name=f"{node_name}", nodes=[botorch_node])

def extract_trial_number(filename: str):
    """
    Extracts the trial number from filenames like:
    'KATE_BO_U010_bayesian_optimisation_trial_10_force_0_2.json'
    Returns an integer trial number (e.g., 10).
    """
    match = re.search(r"trial[_\-]?(\d+)", filename)
    return int(match.group(1)) if match else None

def natural_sort_key(filename):
    """
    Extract numeric portions from filenames for natural sorting.
    E.g. 'trial_2' < 'trial_10'.
    """
    # Split into text and numeric chunks
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

def compute_model_metrics(df, mean_col="mean", std_col="std_dev", real_col="real_result"):
    """
    Compute regression and probabilistic performance metrics.

    Args:
        df (pd.DataFrame): DataFrame with columns for predicted mean, std deviation, and real results.
        mean_col (str): Column name for predicted mean values.
        std_col (str): Column name for predicted standard deviations.
        real_col (str): Column name for real (measured) values.

    Returns:
        dict: Dictionary of computed metrics.
    """
    y_true = df[real_col].values
    y_pred = df[mean_col].values
    std_pred = np.clip(df[std_col].values, 1e-6, None)  # Avoid zero stds

    # Basic metrics
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Probabilistic metrics
    # Negative Log-Likelihood (NLL)
    nll = 0.5 * np.mean(np.log(2 * np.pi * std_pred**2) + ((y_true - y_pred)**2) / (std_pred**2))

    # Log Marginal Likelihood (LML)
    # Approximation: the mean of the log-likelihood across points (just negated NLL)
    lml = -nll

    return {
        "R2": r2,
        "PearsonCorr": corr,
        "RMSE": rmse,
        "MAE": mae,
        "NLL": nll,
        "LML": lml
    }

def plot_model_vs_real(
    df,
    x_col="virtual_force",
    mean_col="mean",
    std_col="std_dev",
    real_col="real_result",
    x_label="Virtual Force",
    y_label="Outcome",
    title_prefix="Model Evaluation",
    show_metrics=True,
    plots_dir="plots"
):
    """
    Plot model predictions vs real results with uncertainty visualization.

    Args:
        df (pd.DataFrame): DataFrame containing model predictions and real results.
        x_col (str): Column name for input variable.
        mean_col (str): Column name for predicted mean.
        std_col (str): Column name for predicted std deviation.
        real_col (str): Column name for real result.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
        title_prefix (str): Title prefix for plots.
        show_metrics (bool): If True, computes and shows metrics on the scatter plot.
    """

    # Compute metrics (if enabled)
    metrics = compute_model_metrics(df, mean_col, std_col, real_col) if show_metrics else None

    plt.figure(figsize=(14, 5))

    # 1️⃣ Predicted vs Real across X
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df, x=x_col, y=mean_col, label="Predicted Mean", color="C0")
    plt.fill_between(
        df[x_col],
        df[mean_col] - df[std_col],
        df[mean_col] + df[std_col],
        color="C0",
        alpha=0.2,
        label="±1 Std Dev",
    )
    sns.scatterplot(data=df, x=x_col, y=real_col, color="C1", label="Real Result")
    plt.title(f"{title_prefix}: Predicted vs Real Across {x_label}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)

    # 2️⃣ Predicted vs Real Scatter
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x=real_col, y=mean_col, color="C2", s=60)
    lims = [
        min(df[real_col].min(), df[mean_col].min()),
        max(df[real_col].max(), df[mean_col].max()),
    ]
    plt.plot(lims, lims, "r--", label="Perfect Prediction (y=x)")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Real Result")
    plt.ylabel("Predicted Mean")
    plt.title(f"{title_prefix}: Model Predictions vs Real Results")
    plt.legend()
    plt.grid(True)

    # Display metrics on plot
    if show_metrics and metrics is not None:
        text_str = "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        plt.text(
            0.05, 0.95, text_str,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{title_prefix.replace(' ', '_')}_evaluation.png"), dpi=300, bbox_inches="tight")
    plt.show()

    return metrics  # Optional: return metrics for programmatic use

def evaluate_predictions(predictions, ground_truth_results, test_grid, trial_number, model_name, plots_dir, logger):
    """
    Evaluate model predictions against ground truth and log metrics.

    Args:
        predictions (pd.DataFrame): DataFrame with model predictions (mean, std_dev).
        ground_truth (pd.DataFrame): DataFrame with real results.
        logger: Logger object for logging metrics.
    """
    
    metric_name = list(predictions[0].keys())[0]
    
    # Extract means and stds
    mean_vals = [pred[metric_name][0] for pred in predictions]
    std_devs = [pred[metric_name][1] for pred in predictions]
    variances = [sd ** 2 for sd in std_devs]

    # Build DataFrame for metrics
    logger.info("Preparing DataFrame for metrics computation...")
    df = pd.DataFrame(test_grid)
    df["mean"] = mean_vals
    df["std_dev"] = std_devs
    df["variance"] = variances
    df["real_result"] = ground_truth_results

    
    # Visualize results
    logger.info("Plotting predicted vs real results...")
    plot_model_vs_real(
        df,
        x_col="virtual_force",
        mean_col="mean",
        std_col="std_dev",
        real_col="real_result",
        x_label="Virtual Force",
        y_label="Outcome",
        title_prefix=f"Model Evaluation: Trial {trial_number}",
        show_metrics=False,
        plots_dir=plots_dir
    )
    # Compute metrics
    logger.info("Computing model performance metrics...")
    metrics = compute_model_metrics(df)
    metrics["model_name"] = model_name

    return metrics

def plot_model_error_evolution(metrics_df, title_prefix="Model Error Evolution", plots_dir="plots"):
    """
    Plot how model performance metrics change across trials.
    Each metric is shown in a separate subplot, and x-axis shows trial number.
    """

    # Sort naturally by filename
    metrics_df = metrics_df.sort_values(
        "model_name", key=lambda col: col.map(natural_sort_key)
    ).reset_index(drop=True)

    # Extract trial number from model_name
    metrics_df["trial_num"] = metrics_df["model_name"].apply(extract_trial_number)

    # Keep only rows where we successfully parsed the trial number
    metrics_df = metrics_df.dropna(subset=["trial_num"])
    metrics_df["trial_num"] = metrics_df["trial_num"].astype(int)

    # Metrics to plot
    metric_cols = [col for col in ["R2", "RMSE", "MAE", "NLL", "LML"] if col in metrics_df.columns]
    n_metrics = len(metric_cols)
    n_cols = 2
    n_rows = (n_metrics + 1) // n_cols

    plt.figure(figsize=(12, 4 * n_rows))

    for i, col in enumerate(metric_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.lineplot(
            data=metrics_df,
            x="trial_num",
            y=col,
            marker="o",
            linewidth=2,
        )
        plt.title(f"{title_prefix}: {col}", fontsize=13)
        plt.xlabel("Trial Number")
        plt.ylabel(col)
        plt.grid(True)

        # Fill region indicating improvement direction
        if col in ["R2", "LML"]:  # higher is better
            plt.fill_between(metrics_df["trial_num"], metrics_df[col], alpha=0.1, color="green")
        elif col in ["RMSE", "MAE", "NLL"]:  # lower is better
            plt.fill_between(metrics_df["trial_num"], metrics_df[col], alpha=0.1, color="red")
            # Save each subplot as a separate PNG image

        plt.figure(figsize=(6, 4))  # Create a new figure for each subplot
        sns.lineplot(
            data=metrics_df,
            x="trial_num",
            y=col,
            marker="o",
            linewidth=2,
        )
        plt.title(f"{title_prefix}: {col}", fontsize=13)
        plt.xlabel("Trial Number")
        plt.ylabel(col)
        plt.grid(True)

        # Fill region indicating improvement direction
        if col in ["R2", "LML"]:  # higher is better
            plt.fill_between(metrics_df["trial_num"], metrics_df[col], alpha=0.1, color="green")
        elif col in ["RMSE", "MAE", "NLL"]:  # lower is better
            plt.fill_between(metrics_df["trial_num"], metrics_df[col], alpha=0.1, color="red")

        subplot_filename = os.path.join(plots_dir, f"{title_prefix.replace(' ', '_')}_{col}.png")
        plt.savefig(subplot_filename, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure to avoid overlapping plots
        
    plt.tight_layout()
    plt.show()