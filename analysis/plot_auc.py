from constants import ft_pii_tasks

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc
import numpy as np


def plot_combined_faithfulness_with_auc(
    task_list,
    model_list,
    display_name_fn,
    model_display_name_dict,
    figure_name,
    losstype="EAP-IG",
    use_normalized=True,
):
    """
    Plots normalized faithfulness for multiple tasks and adds AUC metrics to the dataframe.

    Args:
        use_normalized: If True, uses normalized loss for calculations. If False, uses raw loss.
    """
    n_tasks = len(task_list)
    # Calculate subplot layout - prefer wider layouts
    if n_tasks <= 3:
        nrows, ncols = 1, n_tasks
        figsize = (5 * n_tasks, 3)
    else:
        nrows = 2
        ncols = (n_tasks + 1) // 2
        figsize = (5 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_tasks == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    # Plot each task in its own subplot
    for i, task_name in enumerate(task_list):
        ax = axes[i]

        for model in model_list:
            df = pd.read_csv(f"results/pareto/{model}/csv/{task_name}.csv")

            # Calculate normalized loss
            df[f"norm_loss_{losstype}"] = (
                df[f"loss_{losstype}"] - df["corrupted_baseline"]
            ) / (df["baseline"] - df["corrupted_baseline"])

            # Choose which metric to use for AUC calculations
            if use_normalized:
                y_values = df[f"norm_loss_{losstype}"]
                metric_name = "norm_loss"
            else:
                y_values = df[f"loss_{losstype}"]
                metric_name = "raw_loss"

            # Sort by edges for proper AUC calculation
            df_sorted = df.sort_values(f"edges_{losstype}")
            x_raw = df_sorted[f"edges_{losstype}"].values
            y = (
                df_sorted[f"norm_loss_{losstype}"].values
                if use_normalized
                else df_sorted[f"loss_{losstype}"].values
            )

            # Ensure x is normalized to [0,1] - proportion of edges should already be in this range
            # But let's normalize just to be safe
            x = x_raw / x_raw.max() if x_raw.max() > 1 else x_raw

            # For CMD calculation, we need to work with normalized faithfulness values
            # The paper's CMD is designed for faithfulness in [0,1] range
            if not use_normalized:
                # If using raw loss, normalize y for CMD calculation
                baseline_val = df["baseline"].iloc[0]
                corrupted_val = df["corrupted_baseline"].iloc[0]
                y_for_cmd = (y - corrupted_val) / (baseline_val - corrupted_val)
                # Clip to reasonable range
                y_for_cmd = np.clip(y_for_cmd, 0, 2)  # Allow some overshoot
            else:
                y_for_cmd = np.clip(y, 0, 2)  # Clip normalized values too

            # CPR: ∫₀¹ f(Cₖ) dk - Area under the faithfulness curve
            # For CPR, use the actual y values (raw or normalized as specified)
            cpr = auc(x, np.clip(y, 0, None))  # Ensure non-negative for CPR

            # CMD: ∫₀¹ |1 - f(Cₖ)| dk - Area of absolute difference from perfect faithfulness
            # Use normalized faithfulness values for CMD
            abs_diff_from_1 = np.abs(1 - y_for_cmd)
            cmd = auc(x, abs_diff_from_1)

            # Alternative CMD calculation if faithfulness is always ≤ 1:
            # cmd = auc(x, np.ones_like(x)) - cpr

            # Note: AUROC isn't part of the paper's metrics, removing it
            # auroc = auc(x, y) / auc(x, np.ones_like(x))  # Normalized AUC

            # Add metrics to dataframe
            df[f"cpr_{metric_name}_{losstype}"] = cpr
            df[f"cmd_{metric_name}_{losstype}"] = cmd

            max_loss_row = df.loc[y_values.idxmax()]
            print(f"Model: {model}, Task: {task_name}")
            print(f"CPR (∫ f(Cₖ) dk): {cpr:.3f}")
            print(f"CMD (∫ |1-f(Cₖ)| dk): {cmd:.3f}")
            print(
                f"Loss: {max_loss_row[f'loss_{losstype}']:.2f}, Edges: {max_loss_row[f'edges_{losstype}']:.2f}, Baseline: {max_loss_row['baseline']:.2f}"
            )

            # Plot the curve (still using normalized for visualization)
            ax.plot(
                df[f"edges_{losstype}"],
                df[f"norm_loss_{losstype}"],
                label=model_display_name_dict(model),
                marker="o",
                markersize=3,
                linewidth=1,
            )

            # Save updated dataframe
            df.to_csv(
                f"results/pareto/{model}/csv/{task_name}_with_auc.csv", index=False
            )

        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(f"{display_name_fn(task_name)}", fontsize="large")
        ax.set_xlabel("% Edges Included", fontsize="medium")
        if i == 0 or i == 3:
            ax.set_ylabel("Faithfulness", fontsize="medium")
        ax.grid(True)

    # Hide any unused subplots
    for i in range(n_tasks, len(axes)):
        axes[i].set_visible(False)

    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(len(labels), 5),
        fontsize="medium",
    )
    plt.tight_layout()
    fig.savefig(
        f"./plots/faithfulness/{figure_name}.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=400,
    )
    plt.close(fig)


if __name__ == "__main__":
    model_list = [
        "gpt2-small-baseline",
        "gpt2-medium-baseline",
        # "gpt2-small-dp2",
        # "gpt2-small-dp4",
        # "gpt2-small-dp8",
        "pythia-160m-baseline",
        "pythia-410m-baseline",
        # "pythia-160m-dp2",
        # "pythia-160m-dp4",
        # "pythia-160m-dp8"
    ]

    def display_name_fn(task_name):
        return task_name.replace("_", " ").title()

    def model_display_name_dict(model_name):
        return model_name.replace("_", " ").title()

    # Use normalized loss for AUC calculations (default)
    plot_combined_faithfulness_with_auc(
        task_list=ft_pii_tasks,
        model_list=model_list,
        display_name_fn=display_name_fn,
        model_display_name_dict=model_display_name_dict,
        figure_name="faithfulness_with_auc_normalized",
        losstype="EAP-IG",
        use_normalized=True,
    )

    # Or use raw loss for AUC calculations
    plot_combined_faithfulness_with_auc(
        task_list=ft_pii_tasks,
        model_list=model_list,
        display_name_fn=display_name_fn,
        model_display_name_dict=model_display_name_dict,
        figure_name="faithfulness_with_auc_raw",
        losstype="EAP-IG",
        use_normalized=False,
    )
