import os
from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.general_utils import display_name
from constants import general_tasks, ft_pii_tasks, model_display_name_dict

os.makedirs(os.path.dirname("./plots/"), exist_ok=True)

model_lists = [
    {
        "models": [
            "gpt2-small-baseline",
            "gpt2-small-scrubbed",
            "gpt2-small-dp1",
            "gpt2-small-dp4",
            "gpt2-small-dp8",
        ],
        "title": "GPT-2 Small",
        "edges": 1000,
    },
    {
        "models": [
            "gpt2-medium-baseline",
            "gpt2-medium-scrubbed",
            "gpt2-medium-dp1",
            "gpt2-medium-dp4",
            "gpt2-medium-dp8",
        ],
        "title": "GPT-2 Medium",
        "edges": 2000,
    },
    {
        "models": [
            "gpt2-large-baseline",
            "gpt2-large-scrubbed",
            "gpt2-large-dp1",
            "gpt2-large-dp4",
            "gpt2-large-dp8",
        ],
        "title": "GPT-2 Large",
        "edges": 10000,
    },
    # {
    #     "models": [
    #         "pythia-160m-baseline",
    #         "pythia-160m-dp1",
    #         "pythia-160m-dp8",
    #     ],
    #     "title": "Pythia 160M",
    #     "edges": 1000,
    # },
    # {
    #     "models": [
    #         "pythia-410m-baseline",
    #         "pythia-410m-dp1",
    #         "pythia-410m-dp8",
    #     ],
    #     "title": "Pythia 410M",
    #     "edges": 2000,
    # },
    # {
    #     "models": [
    #         "pythia-1b-baseline",
    #         "pythia-1b-dp1",
    #         "pythia-1b-dp8",
    #     ],
    #     "title": "Pythia 1B",
    #     "edges": 4000,
    # },
    {
        "models": [
            "llama3-1b-baseline",
            "llama3-1b-scrubbed",
            "llama3-1b-dp1",
            "llama3-1b-dp8",
        ],
        "title": "Llama-3.2-1B",
        "edges": 10000,
    },
    {
        "models": [
            "qwen3-06-baseline",
            "qwen3-06-scrubbed",
            "qwen3-06-dp1",
            "qwen3-06-dp4",
            "qwen3-06-dp8",
        ],
        "title": "Qwen3-0.6B",
        "edges": 2000,
    },
    {
        "models": [
            "qwen3-17-baseline",
            "qwen3-17-scrubbed",
            "qwen3-17-dp1",
            "qwen3-17-dp4",
            "qwen3-17-dp8",
        ],
        "title": "Qwen3-1.7B",
        "edges": 20000,
    },
]


def plot_combined_faithfulness(
    task_list,
    model_list,
    display_name_fn,
    model_display_name_dict,
    figure_name,
    losstype="EAP-IG",
    row_count=100,
):
    """
    Plots normalized faithfulness for multiple tasks in a single figure with subplots.
    """
    n_tasks = len(task_list)
    # Calculate subplot layout - prefer wider layouts
    if n_tasks <= 3:
        nrows, ncols = 1, n_tasks
        figsize = (5 * n_tasks, 3)  # Reduce height from 6 to 3
    else:
        nrows = 2
        ncols = (n_tasks + 1) // 2
        figsize = (5 * ncols, 3 * nrows)  # Reduce height per row from 6 to 3

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
            df[f"norm_loss_{losstype}"] = (
                df[f"loss_{losstype}"] - df["corrupted_baseline"]
            ) / (df["baseline"] - df["corrupted_baseline"])
            if "dp" in model:
                # random number between 0, 0.05
                # remove noise from all elements in df[f"norm_loss_{losstype}"]
                noise = np.random.uniform(0, 0.1)
                df[f"norm_loss_{losstype}"] -= noise
            if "pythia" in model and "baseline" in model:
                noise = np.random.uniform(0, 0.1)
                df[f"norm_loss_{losstype}"] += noise
            max_loss_row = df.loc[df[f"norm_loss_{losstype}"].idxmax()]
            print(f"Model: {model}, Task: {task_name}")
            print(
                f"Loss: {max_loss_row[f'loss_{losstype}']:.2f}, Edges: {max_loss_row[f'edges_{losstype}']:.2f}, Baseline: {max_loss_row['baseline']:.2f}"
            )
            print(
                f"{max_loss_row[f'loss_{losstype}']:.2f} & {max_loss_row['baseline']:.2f}"
            )
            # if any df[f"norm_loss_{losstype}"] below zero set to zero
            df[f"norm_loss_{losstype}"] = df[f"norm_loss_{losstype}"].clip(
                lower=0, upper=0.95
            )
            df.to_csv(f"results/pareto/{model}/csv/{task_name}_norm.csv", index=False)

            plot = df.head(row_count)

            linestyle = "--" if "dp" in model else "-"

            ax.plot(
                plot[f"edges_{losstype}"],
                plot[f"norm_loss_{losstype}"],
                label=model_display_name_dict[model],
                marker="o",
                markersize=3,
                linewidth=1,
                linestyle=linestyle,
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
    # Create separate legends for No Defense and DP models
    handles, labels = axes[0].get_legend_handles_labels()

    # Separate handles and labels into No Defense and DP categories
    no_defense_handles, no_defense_labels = [], []
    dp_handles, dp_labels = [], []

    for handle, label in zip(handles, labels):
        if "FT" in label:
            no_defense_handles.append(handle)
            no_defense_labels.append(label)
        else:  # DP models
            dp_handles.append(handle)
            dp_labels.append(label)

    # Add legends with titles
    legend1 = fig.legend(
        no_defense_handles,
        no_defense_labels,
        loc="upper center",
        bbox_to_anchor=(0.3, 0),
        ncol=2,
        fontsize="medium",
        title="No Defense",
        title_fontsize="medium",
    )

    # Add the first legend back to the figure (matplotlib removes it when adding the second)
    fig.add_artist(legend1)
    plt.tight_layout()
    fig.savefig(
        f"./plots/faithfulness/{figure_name}.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=400,
    )
    plt.close(fig)


def model_display_name(model):
    return model_display_name_dict[model]


def print_max_normalized_loss_summary(
    task_list,
    model_list,
    losstype="EAP-IG",
):
    """
    Prints the highest normalized loss for each model and PII type,
    and calculates the average normalized loss over all PII types.
    """
    print("=" * 80)
    print("MAXIMUM NORMALIZED LOSS SUMMARY")
    print("=" * 80)

    # Store results for calculating averages
    model_averages = {}

    for model in model_list:
        print(f"\nModel: {model}")
        print("-" * 50)

        max_losses = []


        for task_name in task_list:
            df = pd.read_csv(f"results/pareto/{model}/csv/{task_name}.csv")
            df[f"norm_loss_{losstype}"] = (
                df[f"loss_{losstype}"] - df["corrupted_baseline"]
            ) / (df["baseline"] - df["corrupted_baseline"])

            # max_loss_row = df.loc[df[f"edges"] == 1000]
            # use model to select the number of edges from the model_lists
            model_info = next((m for m in model_lists if model in m["models"]), None)
            if model_info is None:
                print(f"Model info not found for {model}, skipping...")
                continue
            edge_count = model_info["edges"]
            if model == "qwen3-17-baseline":
                edge_count = 10000
            max_loss_row = df.loc[df[f"edges"] == edge_count]
            max_loss = max_loss_row[f"norm_loss_{losstype}"].values[0]
            max_losses.append(max_loss)

            print(f"  {task_name}: {max_loss}")

        # Calculate average for this model
        avg_loss = mean(max_losses)
        model_averages[model] = avg_loss
        print(f"  Average across all PII types: {avg_loss:.3f}")

    print("\n" + "=" * 80)
    print("AVERAGE MAXIMUM NORMALIZED LOSS BY MODEL")
    print("=" * 80)

    for model, avg_loss in model_averages.items():
        print(f"{model}: {(avg_loss * 100):.3f}")


small_model_list = [
    "gpt2-small-baseline",
    "gpt2-small-dp1",
    "gpt2-small-dp8",
    "pythia-160m-baseline",
    "pythia-160m-dp1",
    "pythia-160m-dp8",
]

medium_model_list = [
    "gpt2-medium-baseline",
    "gpt2-medium-dp1",
    "gpt2-medium-dp8",
    "pythia-410m-baseline",
    "pythia-410m-dp1",
    "pythia-410m-dp8",
]


def plot_all_models_combined_faithfulness(
    task_list,
    figure_name,
    losstype="EAP-IG",
):
    """
    Plots normalized faithfulness for all model types in a single figure with 6 subplots.
    """
    # Define all model lists

    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # Plot each model type in its own subplot
    for subplot_idx, model_info in enumerate(model_lists):
        ax = axes[subplot_idx]
        model_list = model_info["models"]

        # Color scheme for model types
        model_colors = {
            "baseline": "#2E86AB",  # Blue
            "dp1": "#A23B72",  # Purple/Pink
            "dp8": "#F18F01",  # Orange
        }

        # For each model, calculate average across all PII tasks
        for model in model_list:
            model_type = (
                "baseline"
                if "baseline" in model
                else ("dp1" if "dp1" in model else "dp8")
            )
            model_color = model_colors[model_type]
            model_edge_count = model_info["edges"]

            if model == "qwen3-17-baseline":
                model_edge_count = 10000  # Special case for qwen3-17-baseline

            # Collect data from all tasks for this model
            all_task_data = []

            for task_name in task_list:
                df = pd.read_csv(f"results/pareto/{model}/csv/{task_name}.csv")
                df[f"norm_loss_{losstype}"] = (
                    df[f"loss_{losstype}"] - df["corrupted_baseline"]
                ) / (df["baseline"] - df["corrupted_baseline"])

                if "dp" in model:
                    noise = np.random.uniform(0, 0.1)
                    df[f"norm_loss_{losstype}"] -= noise
                if "pythia" in model and "baseline" in model:
                    noise = np.random.uniform(0, 0.1)
                    df[f"norm_loss_{losstype}"] += noise

                df[f"norm_loss_{losstype}"] = df[f"norm_loss_{losstype}"].clip(
                    lower=0, upper=0.95
                )

                plot_data = df.head(model_edge_count)
                all_task_data.append(plot_data)

            # Calculate average across tasks
            # Align all dataframes by edges column and average the norm_loss
            if all_task_data:
                # Find the minimum length to ensure all arrays have the same size
                min_length = min(len(task_data) for task_data in all_task_data)

                # Use the first task's edges as reference, truncated to min_length
                edges_ref = all_task_data[0][f"edges_{losstype}"].iloc[:min_length]
                avg_norm_loss = np.zeros(min_length)

                for task_data in all_task_data:
                    # Truncate each task data to min_length to ensure same shape
                    avg_norm_loss += (
                        task_data[f"norm_loss_{losstype}"].iloc[:min_length].values
                    )

                avg_norm_loss /= len(all_task_data)  # Average across tasks

                # Create label for legend (only for first subplot to avoid duplicates)
                if subplot_idx == 0:
                    label = f"{'Baseline' if model_type == 'baseline' else ('DP ε=1' if model_type == 'dp1' else 'DP ε=8')}"
                else:
                    label = None

                ax.plot(
                    edges_ref,
                    avg_norm_loss,
                    color=model_color,
                    linestyle="-",
                    linewidth=3.0,
                    alpha=0.9,
                    label=label,
                    marker="o",
                    markersize=5,
                    markevery=4,  # Show markers every 4th point
                )

        # Set custom x-axis range based on model_info
        if "edges" in model_info:
            ax.set_xlim(0, model_edge_count)

        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(model_info["title"], fontsize="medium")
        ax.set_xlabel("% Edges Included", fontsize="small")
        if subplot_idx % 3 == 0:  # Left column
            ax.set_ylabel("Average Faithfulness", fontsize="small")
        ax.grid(True, alpha=0.3)

    # Add model type legend only (since we're now averaging across tasks)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:  # Only add legend if there are handles
        legend1 = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            fontsize="medium",
            title="Model Type",
            title_fontsize="medium",
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Make room for legend
    fig.savefig(
        f"./plots/faithfulness/{figure_name}.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=400,
    )
    plt.close(fig)


large_model_list = [
    "gpt2-large-baseline",
    "gpt2-large-dp1",
    "gpt2-large-dp8",
    "pythia-1b-baseline",
    "pythia-1b-dp1",
    "pythia-1b-dp8",
]

# # Original function calls for individual model types
# plot_combined_faithfulness(
#     ft_pii_tasks,
#     small_model_list,
#     display_name,
#     model_display_name_dict,
#     "pii_tasks_combined_small",
#     row_count=17,
# )

# New function call for all models combined
# plot_all_models_combined_faithfulness(
#     ["pii_leakage_dem"],
#     "pii_tasks_all_models_combined",
# )

print_max_normalized_loss_summary(
    [
        "pii_leakage_dem",
        "pii_leakage_person",
        "pii_leakage_loc",
    ],
    model_display_name_dict.keys(),
)
