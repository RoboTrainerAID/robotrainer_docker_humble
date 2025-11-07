import math
import os
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.model_spec import GeneratorSpec
from botorch import settings
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import acqf_input_constructor
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import Tensor


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
    generator_spec: GeneratorSpec,
    node_name: str,
) -> GenerationStrategy:
    """Constructs a Center + Modular BoTorch `GenerationStrategy`
    using the provided `generator_spec` for the Modular BoTorch node.
    """
    botorch_node = GenerationNode(
        node_name=node_name,
        model_specs=[generator_spec],
    )
    # Center node is a customized node that uses a simplified logic and has a
    # built-in transition criteria that transitions after generating once.
    center_node = CenterGenerationNode(next_node_name=botorch_node.node_name)
    return GenerationStrategy(
        name=f"Center+qNegIntegratedPosteriorVariance",
        nodes=[center_node, botorch_node]
    )

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
