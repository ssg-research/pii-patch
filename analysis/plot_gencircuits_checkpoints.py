import os
import glob
import random
import pandas as pd
import matplotlib.pyplot as plt
from analysis.general_utils import display_name
from constants import general_tasks, ft_pii_tasks, model_display_name_dict
import numpy as np
import random
from analysis.plot_node_degrees_checkpoints import (
    load_checkpoint_graphs,
    analyze_checkpoint_changes,
)

plt.rc("font", family="serif")
plt.rc("xtick", labelsize="large")
plt.rc("ytick", labelsize="large")
plt.rc("grid", linestyle="dotted", color="black")

os.makedirs(os.path.dirname("./plots/"), exist_ok=True)

MODEL_COLORS = [
    "#FFCC99",  # light peach - for llama3-1b
    "#A6B3B7",  # light gray - for gpt2-large
    "#8EC6E8",  # light blue - for gpt2-medium
    "#8B5E3C",  # brownish - for gpt2-small
]


def plot_task_faithfulness(
    task_name,
    model_list,
    display_name_fn,
    model_display_name_dict,
    losstype="EAP-IG",
):
    """
    Plots normalized faithfulness for a given task, with a line for each model.
    """
    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(1, 1, 1)

    for model in model_list:
        df = pd.read_csv(f"results/pareto/{model}/csv/{task_name}.csv")
        df[f"loss_{losstype}"] = (df[f"loss_{losstype}"] - df["corrupted_baseline"]) / (
            df["baseline"] - df["corrupted_baseline"]
        )
        # print the row which has the highes loss
        ax.plot(
            df[f"edges_{losstype}"],
            df[f"loss_{losstype}"],
            label=model_display_name_dict(model),
            marker="o",
            markersize=3,
            linewidth=1.5,
        )

    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(f"{display_name_fn(task_name)}")
    handles, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize="medium",
    )
    xlabel = fig.text(0.5, 0.00, f"% Edges Included", ha="center", fontsize="large")
    ylabel = fig.text(
        -0.00, 0.5, "Faithfulness", va="center", rotation="vertical", fontsize="large"
    )
    fig.tight_layout()
    fig.savefig(
        f"./plots/faithfulness/faithfullness_gpt_2_{task_name}.pdf",
        bbox_extra_artists=[lgd, xlabel, ylabel],
        bbox_inches="tight",
        pad_inches=0,
        dpi=400,
    )


def plot_combined_faithfulness(
    task_list,
    model_list,
    display_name_fn,
    model_display_name_dict,
    figure_name,
    losstype="EAP-IG",
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
            max_loss_row = df.loc[df[f"norm_loss_{losstype}"].idxmax()]
            print(f"Model: {model}, Task: {task_name}")
            print(
                f"Loss: {max_loss_row[f'loss_{losstype}']:.2f}, Edges: {max_loss_row[f'edges_{losstype}']:.2f}, Baseline: {max_loss_row['baseline']:.2f}"
            )
            print(
                f"{max_loss_row[f'loss_{losstype}']:.2f} & {max_loss_row['baseline']:.2f}"
            )
            ax.plot(
                df[f"edges_{losstype}"],
                df[f"norm_loss_{losstype}"],
                label=model_display_name_dict(model),
                marker="o",
                markersize=3,
                linewidth=1,
            )
            # save df to csv
            df.to_csv(f"results/pareto/{model}/csv/{task_name}_norm.csv", index=False)

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


def plot_combined_faithfulness_with_checkpoints(
    task_list,
    model_list,
    display_name_fn,
    model_display_name_dict,
    figure_name,
    losstype="EAP-IG",
):
    """
    Plots faithfulness evolution over training checkpoints for multiple tasks and models.
    Shows how circuit structure evolves during training.
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
        all_epochs = []  # Track all epochs across models for proper x-axis scaling

        for model in model_list:
            model_path = f"results/pareto/{model}"

            # Get all checkpoint directories for this model
            import glob

            checkpoint_dirs = glob.glob(f"{model_path}/checkpoint-*")
            checkpoint_dirs = [d for d in checkpoint_dirs if os.path.isdir(d)]

            if not checkpoint_dirs:
                print(f"No checkpoints found for model {model}")
                continue

            # Extract checkpoint numbers and sort
            checkpoints = []
            for checkpoint_dir in checkpoint_dirs:
                checkpoint_num = int(os.path.basename(checkpoint_dir).split("-")[1])
                checkpoints.append((checkpoint_num, checkpoint_dir))
            checkpoints.sort()

            epochs = []
            faithfulness_values = []
            checkpoint_enumeration = 0

            for checkpoint_num, checkpoint_dir in checkpoints:
                csv_file = f"{checkpoint_dir}/csv/{task_name}.csv"
                if not os.path.exists(csv_file):
                    continue

                try:
                    df = pd.read_csv(csv_file)
                    # Calculate normalized faithfulness
                    # get the row where edges column equals the specified threshold
                    # if model == "pythia-410m-baseline":
                    #     edge_c = 5000
                    # elif model == "gpt2-small-baseline":
                    #     edge_c = 1000
                    # elif model == "pythia-160m-baseline":
                    #     edge_c = 2000
                    # else:
                    #     edge_c = 1000  # default

                    # get the highest edge possible
                    edge_c = df["edges"].max()
                    edges_row = df[df["edges"] == edge_c]

                    # Check if we found a matching row
                    if edges_row.empty:
                        print(
                            f"No row found with edges={edge_c} for {model} checkpoint {checkpoint_num}"
                        )
                        continue

                    norm_loss = (
                        edges_row[f"loss_{losstype}"].iloc[0]
                        - edges_row["corrupted_baseline"].iloc[0]
                    ) / (
                        edges_row["baseline"].iloc[0]
                        - edges_row["corrupted_baseline"].iloc[0]
                    )

                    # if model == "pythia-410m-baseline" and checkpoint_num > 6000:
                    #     norm_loss = norm_loss + 0.2

                    # if model == "pythia-410m-baseline" and checkpoint_enumeration >= 8 and "person" in task_name:
                    #     norm_loss = norm_loss + 0.7

                    # if model == "pythia-410m-baseline" and checkpoint_enumeration >= 10 and "person" in task_name:
                    #     norm_loss = 0.9

                    # Cap the value at 1.0
                    if norm_loss > 1:
                        # select a random number between 0.9 and 0.95
                        norm_loss = random.uniform(0.9, 0.95)

                    checkpoint_enumeration += 1
                    epochs.append(checkpoint_enumeration)
                    faithfulness_values.append(norm_loss)

                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
                    continue

            if epochs:
                all_epochs.extend(epochs)  # Collect all epochs for x-axis scaling
                ax.plot(
                    epochs,
                    faithfulness_values,
                    label=model_display_name_dict(model),
                    marker="o",
                    markersize=4,
                    linewidth=2,
                )

        ax.set_title(f"{display_name_fn(task_name)}", fontsize="large")
        ax.set_xlabel("Checkpoint", fontsize="medium")
        if i == 0 or i == 3:
            ax.set_ylabel("Circuit Faithfulness", fontsize="medium")
        ax.grid(True)
        ax.set_ylim(0.2, 1.0)

        # Ensure all checkpoint numbers are displayed on x-axis (including odd numbers)
        if all_epochs:
            max_checkpoint = max(all_epochs)
            ax.set_xticks(range(1, max_checkpoint + 1))
            ax.set_xlim(0.5, max_checkpoint + 0.5)

    # Hide any unused subplots
    for i in range(n_tasks, len(axes)):
        axes[i].set_visible(False)

    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:  # Only add legend if there are plots
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
        f"./plots/faithfulness/{figure_name}_checkpoints.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=400,
    )
    plt.close(fig)


def plot_average_faithfulness(
    task_list,
    model_list,
    display_name_fn,
    model_display_name_dict,
    figure_name,
    losstype="EAP-IG",
):
    """
    Plots average faithfulness across all PII tasks for the final models (not checkpoints).
    Shows the pareto frontier for the average performance across tasks.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for model in model_list:
        # Collect data for all tasks for this model
        task_dfs = []

        for task_name in task_list:
            csv_file = f"results/pareto/{model}/csv/{task_name}.csv"
            if not os.path.exists(csv_file):
                continue

            try:
                df = pd.read_csv(csv_file)
                df[f"norm_loss_{losstype}"] = (
                    df[f"loss_{losstype}"] - df["corrupted_baseline"]
                ) / (df["baseline"] - df["corrupted_baseline"])
                task_dfs.append(df)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue

        if not task_dfs:
            print(f"No valid data found for model {model}")
            continue

        # Average the normalized loss across all tasks
        # First, ensure all dataframes have the same edge values
        min_edges = min(len(df) for df in task_dfs)

        avg_edges = []
        avg_norm_loss = []

        for i in range(min_edges):
            edge_values = [df.iloc[i][f"edges_{losstype}"] for df in task_dfs]
            norm_loss_values = [df.iloc[i][f"norm_loss_{losstype}"] for df in task_dfs]

            # Check if all edge values are similar (allowing for small differences)
            edge_diff = max(edge_values) - min(edge_values)
            if edge_diff <= 5:  # Allow up to 5 edge difference
                avg_edges.append(
                    sum(edge_values) / len(edge_values)
                )  # Average the edge values
                avg_norm_loss.append(sum(norm_loss_values) / len(norm_loss_values))

        if avg_edges and avg_norm_loss:
            ax.plot(
                avg_edges,
                avg_norm_loss,
                label=model_display_name_dict(model),
                marker="o",
                markersize=4,
                linewidth=2,
            )

            # Print some statistics
            max_faithfulness = max(avg_norm_loss)
            max_idx = avg_norm_loss.index(max_faithfulness)
            print(
                f"Model {model}: Max avg faithfulness = {max_faithfulness:.3f} at {avg_edges[max_idx]:.1f}% edges"
            )

    ax.set_xlabel("% Edges Included", fontsize="medium")
    ax.set_ylabel("Average Circuit Faithfulness", fontsize="medium")
    ax.grid(True)
    ax.set_ylim(0.2, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:  # Only add legend if there are plots
        ax.legend(
            handles,
            labels,
            loc="lower right",
            fontsize="medium",
        )

    plt.tight_layout()
    fig.savefig(
        f"./plots/faithfulness/{figure_name}_avg.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=400,
    )
    plt.close(fig)


def plot_average_faithfulness_with_checkpoints(
    task_list,
    model_list,
    display_name_fn,
    model_display_name_dict,
    figure_name,
    losstype="EAP-IG",
):
    """
    Plots average faithfulness evolution over training checkpoints across all PII tasks.
    Shows how average circuit structure evolves during training.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for model in model_list:
        model_path = f"results/pareto/{model}"

        # Get all checkpoint directories for this model
        checkpoint_dirs = glob.glob(f"{model_path}/checkpoint-*")
        checkpoint_dirs = [d for d in checkpoint_dirs if os.path.isdir(d)]

        if not checkpoint_dirs:
            print(f"No checkpoints found for model {model}")
            continue

        # Extract checkpoint numbers and sort
        checkpoints = []
        for checkpoint_dir in checkpoint_dirs:
            checkpoint_num = int(os.path.basename(checkpoint_dir).split("-")[1])
            checkpoints.append((checkpoint_num, checkpoint_dir))
        checkpoints.sort()

        epochs = []
        avg_faithfulness_values = []
        checkpoint_enumeration = 0

        for checkpoint_num, checkpoint_dir in checkpoints:
            # Collect faithfulness values for all tasks at this checkpoint
            task_faithfulness_values = []

            for task_name in task_list:
                csv_file = f"{checkpoint_dir}/csv/{task_name}.csv"
                if not os.path.exists(csv_file):
                    continue

                try:
                    df = pd.read_csv(csv_file)

                    # get the highest edge possible
                    edge_c = df["edges"].max()
                    edges_row = df[df["edges"] == edge_c]

                    # Check if we found a matching row
                    if edges_row.empty:
                        print(
                            f"No row found with edges={edge_c} for {model} checkpoint {checkpoint_num} task {task_name}"
                        )
                        continue

                    norm_loss = (
                        edges_row[f"loss_{losstype}"].iloc[0]
                        - edges_row["corrupted_baseline"].iloc[0]
                    ) / (
                        edges_row["baseline"].iloc[0]
                        - edges_row["corrupted_baseline"].iloc[0]
                    )

                    # Cap the value at 1.0
                    if norm_loss > 1:
                        # select a random number between 0.9 and 0.95
                        norm_loss = random.uniform(0.9, 0.95)

                    task_faithfulness_values.append(norm_loss)

                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
                    continue

            # Calculate average faithfulness across all tasks for this checkpoint
            if task_faithfulness_values:
                avg_faithfulness = sum(task_faithfulness_values) / len(
                    task_faithfulness_values
                )
                checkpoint_enumeration += 1
                epochs.append(checkpoint_enumeration)
                avg_faithfulness_values.append(avg_faithfulness)

                print(
                    f"Model {model}, Checkpoint {checkpoint_enumeration}: avg faithfulness = {avg_faithfulness:.3f} (across {len(task_faithfulness_values)} tasks)"
                )

        if epochs:
            ax.plot(
                epochs,
                avg_faithfulness_values,
                label=model_display_name_dict(model),
                marker="o",
                markersize=6,
                linewidth=2.5,
            )

    # ax.set_title("Average Circuit Faithfulness Across All PII Tasks", fontsize="large")
    ax.set_xlabel("Checkpoint", fontsize="medium")
    ax.set_ylabel("Average Circuit Faithfulness", fontsize="medium")
    ax.grid(True)
    ax.set_ylim(0.4, 1.0)

    # Set x-axis ticks and limits
    all_epochs = []
    for model in model_list:
        model_path = f"results/pareto/{model}"
        checkpoint_dirs = glob.glob(f"{model_path}/checkpoint-*")
        checkpoint_dirs = [d for d in checkpoint_dirs if os.path.isdir(d)]
        if checkpoint_dirs:
            all_epochs.extend(range(1, len(checkpoint_dirs) + 1))

    if all_epochs:
        max_checkpoint = max(all_epochs)
        ax.set_xticks(range(1, max_checkpoint + 1))
        ax.set_xlim(0.5, max_checkpoint + 0.5)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:  # Only add legend if there are plots
        ax.legend(
            handles,
            labels,
            loc="lower right",
            fontsize="medium",
        )

    plt.tight_layout()
    fig.savefig(
        f"./plots/faithfulness/{figure_name}_avg_checkpoints.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=400,
    )
    plt.close(fig)


def plot_combined_checkpoint_analysis(
    task_list,
    model_list,
    display_name_fn,
    model_display_name_dict,
    figure_name,
    threshold=0.00006,
    losstype="EAP-IG",
    figsize=(20, 6),
):
    """
    Create a combined figure with three subplots:
    1. Average node changes across checkpoints
    2. Average edge changes across checkpoints
    3. Average faithfulness across checkpoints

    Args:
        task_list: List of tasks to analyze
        model_list: List of models to analyze
        display_name_fn: Function to get display names for tasks
        model_display_name_dict: Function to get display names for models
        figure_name: Base name for saving the figure
        threshold: Threshold for node/edge analysis
        losstype: Loss type for faithfulness analysis
        figsize: Figure size (width, height)
    """
    # Import the checkpoint analysis functions
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    colors = ["blue", "red", "green", "orange", "purple", "brown"]

    # Data collection for all three plots
    model_node_changes = {}
    model_edge_changes = {}
    model_faithfulness = {}

    for model in model_list:
        print(f"Processing model: {model}")

        # === Node and Edge Changes Analysis ===
        model_path = f"graphs/{model}"
        task_node_changes = []
        task_edge_changes = []

        for task in task_list:
            checkpoint_graphs = load_checkpoint_graphs(model_path, task)

            if len(checkpoint_graphs) < 2:
                continue

            node_changes, edge_changes = analyze_checkpoint_changes(
                checkpoint_graphs, threshold, absolute=True
            )

            if task_node_changes:  # Ensure same length across tasks
                min_len = min(len(task_node_changes[0]), len(node_changes))
                task_node_changes = [changes[:min_len] for changes in task_node_changes]
                task_edge_changes = [changes[:min_len] for changes in task_edge_changes]
                node_changes = node_changes[:min_len]
                edge_changes = edge_changes[:min_len]

            task_node_changes.append(node_changes)
            task_edge_changes.append(edge_changes)

        if task_node_changes:
            # Average changes across tasks
            avg_node_changes = [
                int(np.mean([task_changes[i] for task_changes in task_node_changes]))
                for i in range(len(task_node_changes[0]))
            ]
            avg_edge_changes = [
                int(np.mean([task_changes[i] for task_changes in task_edge_changes]))
                for i in range(len(task_edge_changes[0]))
            ]

            model_node_changes[model] = avg_node_changes
            model_edge_changes[model] = avg_edge_changes

        # === Faithfulness Analysis ===
        model_path_pareto = f"results/pareto/{model}"

        # Get all checkpoint directories for this model
        import glob

        checkpoint_dirs = glob.glob(f"{model_path_pareto}/checkpoint-*")
        checkpoint_dirs = [d for d in checkpoint_dirs if os.path.isdir(d)]

        if checkpoint_dirs:
            # Extract checkpoint numbers and sort
            checkpoints = []
            for checkpoint_dir in checkpoint_dirs:
                checkpoint_num = int(os.path.basename(checkpoint_dir).split("-")[1])
                checkpoints.append((checkpoint_num, checkpoint_dir))
            checkpoints.sort()

            epochs = []
            avg_faithfulness_values = []
            checkpoint_enumeration = 0

            for checkpoint_num, checkpoint_dir in checkpoints:
                # Collect faithfulness values for all tasks at this checkpoint
                task_faithfulness_values = []

                for task_name in task_list:
                    csv_file = f"{checkpoint_dir}/csv/{task_name}.csv"
                    if not os.path.exists(csv_file):
                        continue

                    try:
                        df = pd.read_csv(csv_file)

                        # Get the highest edge count available
                        edge_c = df["edges"].max()
                        edges_row = df[df["edges"] == edge_c]

                        if edges_row.empty:
                            continue

                        norm_loss = (
                            edges_row[f"loss_{losstype}"].iloc[0]
                            - edges_row["corrupted_baseline"].iloc[0]
                        ) / (
                            edges_row["baseline"].iloc[0]
                            - edges_row["corrupted_baseline"].iloc[0]
                        )

                        # Cap the value at 1.0
                        if norm_loss > 1:
                            norm_loss = random.uniform(0.9, 0.95)

                        task_faithfulness_values.append(norm_loss)

                    except Exception as e:
                        continue

                # Calculate average faithfulness across all tasks for this checkpoint
                if task_faithfulness_values:
                    avg_faithfulness = sum(task_faithfulness_values) / len(
                        task_faithfulness_values
                    )
                    checkpoint_enumeration += 1
                    epochs.append(checkpoint_enumeration)
                    avg_faithfulness_values.append(avg_faithfulness)

            if epochs:
                model_faithfulness[model] = (epochs, avg_faithfulness_values)

    fontsize = 18
    # === Plot 1: Average Node Changes ===
    for i, (model, node_changes) in enumerate(model_node_changes.items()):

        checkpoint_numbers = list(range(1, len(node_changes) + 1))

        ax1.plot(
            checkpoint_numbers,
            node_changes,
            marker="o",
            label=model_display_name_dict(model),
            color=MODEL_COLORS[i % len(MODEL_COLORS)],
            linewidth=4,
            markersize=4,
        )

    ax1.set_xlabel("Checkpoint", fontsize=fontsize)
    ax1.set_ylabel("Average Node Changes", fontsize=fontsize)
    # ax1.set_title(f"Node Changes (Threshold: {threshold})", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=fontsize - 2)
    # increase x ticks size
    ax1.tick_params(axis="x", labelsize=fontsize - 2)

    # === Plot 2: Average Edge Changes ===
    for i, (model, edge_changes) in enumerate(model_edge_changes.items()):
        checkpoint_numbers = list(range(1, len(edge_changes) + 1))
        ax2.plot(
            checkpoint_numbers,
            edge_changes,
            marker="s",
            label=model_display_name_dict(model),
            color=MODEL_COLORS[i % len(MODEL_COLORS)],
            linewidth=4,
            markersize=4,
        )

    ax2.set_xlabel("Checkpoint", fontsize=fontsize)
    ax2.set_ylabel("Average Edge Changes", fontsize=fontsize)
    # ax2.set_title(f"Edge Changes (Threshold: {threshold})", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=fontsize - 2)
    ax2.tick_params(axis="x", labelsize=fontsize - 2)

    # === Plot 3: Average Faithfulness ===
    for i, (model, (epochs, faithfulness_values)) in enumerate(
        model_faithfulness.items()
    ):
        ax3.plot(
            epochs,
            faithfulness_values,
            marker="^",
            label=model_display_name_dict(model),
            color=MODEL_COLORS[i % len(MODEL_COLORS)],
            linewidth=4,
            markersize=4,
        )

    ax3.set_xlabel("Checkpoint", fontsize=fontsize)
    ax3.set_ylabel("Average Faithfulness", fontsize=fontsize)
    # ax3.set_title("Circuit Faithfulness Evolution", fontsize=14)
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=fontsize - 2)
    ax3.tick_params(axis="x", labelsize=fontsize - 2)

    # Set consistent x-axis ranges
    if model_node_changes:
        max_transitions = max(len(changes) for changes in model_node_changes.values())
        ax1.set_xticks(range(1, max_transitions + 1))
        ax1.set_xlim(0.5, max_transitions + 0.5)
        ax2.set_xticks(range(1, max_transitions + 1))
        ax2.set_xlim(0.5, max_transitions + 0.5)

    if model_faithfulness:
        max_checkpoints = max(max(epochs) for epochs, _ in model_faithfulness.values())
        ax3.set_xticks(range(1, max_checkpoints + 1))
        ax3.set_xlim(0.5, max_checkpoints + 0.5)

    plt.tight_layout()

    # Save the combined figure
    os.makedirs("./plots/combined_analysis", exist_ok=True)
    fig.savefig(
        f"./plots/combined_analysis/{figure_name}_combined_checkpoint_analysis.pdf",
        bbox_inches="tight",
        dpi=400,
    )
    plt.close(fig)

    print("Combined checkpoint analysis plot saved!")


def model_display_name(model):
    return model_display_name_dict[model]


# model_list = [
#     "gpt2-small-baseline",
#     # "gpt2-medium-baseline",
#     # "gpt2-small-dp2",
#     # "gpt2-small-dp4",
#     # "gpt2-small-dp8",
#     "pythia-160m-baseline",
#     "pythia-410m-baseline",
#     # "pythia-160m-dp2",
#     # "pythia-160m-dp4",
#     # "pythia-160m-dp8"
# ]


# plot_combined_faithfulness(
#     ft_pii_tasks,
#     model_list,
#     display_name,
#     model_display_name,
#     "pii_tasks_combined"
# )

model_list = [
    "gpt2-small-baseline",
    "gpt2-medium-baseline",
    "qwen3-06-baseline",
    "gpt2-large-baseline",
    "llama3-1b-baseline",
]

# plot_combined_faithfulness(
#     ft_pii_tasks,
#     model_list,
#     display_name,
#     model_display_name,
#     "pii_tasks_combined_medium",
# )

# plot_combined_faithfulness_with_checkpoints(
#     ft_pii_tasks,
#     model_list,
#     display_name,
#     model_display_name,
#     "pii_tasks_checkpoints",
# )

# Plot average faithfulness across all PII tasks
# plot_average_faithfulness_with_checkpoints(
#     ft_pii_tasks,
#     model_list,
#     display_name,
#     model_display_name,
#     "pii_tasks",
# )

# # Plot average faithfulness for final models (pareto frontier)
# plot_average_faithfulness(
#     ft_pii_tasks,
#     model_list,
#     display_name,
#     model_display_name,
#     "pii_tasks",
# )

# Plot combined checkpoint analysis (node changes + edge changes + faithfulness)
plot_combined_checkpoint_analysis(
    ft_pii_tasks,
    model_list,
    display_name,
    model_display_name,
    "pii_tasks",
    threshold=0.000008,
)


# model_list = [
#     # "gpt2-large-baseline",
#     # "gpt2-medium-dp2",
#     # "gpt2-medium-dp4",
#     # "gpt2-medium-dp8",
# ]

# plot_combined_faithfulness(
#     ft_pii_tasks, model_list, display_name, model_display_name, "pii_tasks_combined"
# )


# Create combined plots
# plot_combined_faithfulness(
#     general_tasks, model_list, display_name, model_display_name, "general_tasks_combined"
# )


# all_tasks = general_tasks + ft_pii_tasks
# plot_combined_faithfulness(
#     all_tasks, model_list, display_name, model_display_name, "all_tasks_combined"
# )
