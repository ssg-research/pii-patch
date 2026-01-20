import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import pandas as pd
from analysis.node_degrees_plots import (
    plot_attention_block_heatmap
)
from analysis.nodes_analysis import (
    analyze_node_degrees,
    compare_node_degrees_across_models,
)
from constants import ft_pii_tasks
from gencircuits.eap.graph import Graph


def plot_degree_distributions(
    degree_results: Dict[str, Dict],
    degree_type: str = "total_degree",
    figsize: Tuple[int, int] = (15, 10),
    task: str = "dem",  # Default task for file naming
):
    """
    Plot degree distributions across different models.

    Args:
        degree_results: Results from analyze_node_degrees
        degree_type: Type of degree to plot ('in_degree', 'out_degree', 'total_degree')
        figsize: Figure size for the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    model_names = list(degree_results.keys())

    for i, model_name in enumerate(model_names):
        if i >= len(axes):
            break

        ax = axes[i]
        model_results = degree_results[model_name]

        # Extract degrees and node types
        degrees = [info[degree_type] for info in model_results.values()]
        node_types = [info["node_type"] for info in model_results.values()]
        layers = [info["layer"] for info in model_results.values()]

        # Create scatter plot colored by node type
        type_colors = {
            "Input Node": "red",
            "Attention Node": "blue",
            "MLP Node": "green",
        }

        for node_type in type_colors:
            mask = [nt == node_type for nt in node_types]
            if any(mask):
                filtered_degrees = [d for d, m in zip(degrees, mask) if m]
                filtered_layers = [l for l, m in zip(layers, mask) if m]
                ax.scatter(
                    filtered_layers,
                    filtered_degrees,
                    c=type_colors[node_type],
                    label=node_type,
                    alpha=0.7,
                )

        ax.set_xlabel("Layer")
        ax.set_ylabel(f'{degree_type.replace("_", " ").title()}')
        ax.set_yticks(np.arange(0, 90, step=5))
        ax.set_title(f"{model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(model_names), len(axes)):
        axes[i].set_visible(False)

    plt.title(
        f'Node {degree_type.replace("_", " ").title()} Distributions Across Models ({task})'
    )
    plt.tight_layout()
    plt.savefig(
        f"./plots/node_degrees/degree_distributions_{task}_{degree_type}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        dpi=400,
    )


def analyze_degree_differences(
    comparison_df: pd.DataFrame,
    baseline_model: str = "GPT2-Small_Orig",
    degree_type: str = "total_degree",
) -> pd.DataFrame:
    """
    Analyze differences in node degrees compared to a baseline model.

    Args:
        comparison_df: DataFrame from compare_node_degrees_across_models
        baseline_model: Name of the baseline model for comparison
        degree_type: Type of degree to analyze

    Returns:
        DataFrame with degree differences
    """
    print(comparison_df.head())
    baseline_col = f"{baseline_model}_{degree_type}"
    if baseline_col not in comparison_df.columns:
        raise ValueError(f"Baseline model column {baseline_col} not found")

    diff_data = []

    for _, row in comparison_df.iterrows():
        baseline_degree = row[baseline_col]
        row_data = {
            "node": row["node"],
            "layer": row["layer"],
            "node_type": row["node_type"],
            "baseline_degree": baseline_degree,
        }

        # Calculate differences for each model
        for col in comparison_df.columns:
            if col.endswith(f"_{degree_type}") and col != baseline_col:
                model_name = col.replace(f"_{degree_type}", "")
                difference = row[col] - baseline_degree
                row_data[f"{model_name}_diff"] = difference

        diff_data.append(row_data)

    return pd.DataFrame(diff_data)


def average_degree_results_across_tasks(
    all_task_results: Dict[str, Dict],
) -> Dict[str, Dict]:
    """
    Average node degree results across multiple tasks.

    Args:
        all_task_results: Dictionary mapping task names to degree results

    Returns:
        Averaged degree results with same structure as individual task results
    """
    # Get all model names from the first task
    first_task = list(all_task_results.keys())[0]
    model_names = list(all_task_results[first_task].keys())

    averaged_results = {}

    for model_name in model_names:
        # Get all nodes across all tasks for this model
        all_nodes = set()
        for task_results in all_task_results.values():
            if model_name in task_results:
                all_nodes.update(task_results[model_name].keys())

        model_averaged = {}

        for node_name in all_nodes:
            # Collect degrees for this node across all tasks
            in_degrees = []
            out_degrees = []
            total_degrees = []
            layer = None
            node_type = None

            for task_results in all_task_results.values():
                if model_name in task_results and node_name in task_results[model_name]:
                    node_info = task_results[model_name][node_name]
                    in_degrees.append(node_info["in_degree"])
                    out_degrees.append(node_info["out_degree"])
                    total_degrees.append(node_info["total_degree"])
                    layer = node_info["layer"]  # Should be same across tasks
                    node_type = node_info["node_type"]  # Should be same across tasks
                else:
                    # If node doesn't exist in this task, use 0 degrees
                    in_degrees.append(0)
                    out_degrees.append(0)
                    total_degrees.append(0)

            # Calculate averages and round to integers
            model_averaged[node_name] = {
                "in_degree": round(np.mean(in_degrees)),
                "out_degree": round(np.mean(out_degrees)),
                "total_degree": round(np.mean(total_degrees)),
                "layer": layer,
                "node_type": node_type,
            }

        averaged_results[model_name] = model_averaged

    return averaged_results


if __name__ == "__main__":
    # Load graphs for all tasks first
    all_task_degree_results = {}

    model_configs_by_size = {
        "small": {
            "gpt2-small-baseline": "graphs/gpt2-small-baseline",
            "pythia-160m-baseline": "graphs/pythia-160m-baseline",
        },
        "medium": {
            "gpt2-medium-baseline": "graphs/gpt2-medium-baseline",
            "pythia-410m-baseline": "graphs/pythia-410m-baseline",
        },
        "large": {
            "gpt2-large-baseline": "graphs/gpt2-large-baseline",
            "pythia-1b-baseline": "graphs/pythia-1b-baseline",
        },
    }

    for model_config_size in model_configs_by_size.keys():
        model_configs = model_configs_by_size[model_config_size]
        # Analyze node degrees for each threshold
        for threshold in [0.005, 0.01]:
            print(f"Processing threshold: {threshold}")

            # Collect degree results for each task
            task_degree_results = {}

            for task in ft_pii_tasks:
                print(f"Processing task: {task}")
                target_file = f"{task}_kl.json"

                graphs = {}
                for model_name, model_path in model_configs.items():
                    try:
                        graphs[model_name] = Graph.from_json(
                            f"{model_path}/{target_file}"
                        )
                    except FileNotFoundError:
                        print(
                            f"Warning: Graph file not found for {model_name}/{target_file}"
                        )
                        continue

                if graphs:  # Only process if we have at least one graph
                    degree_results = analyze_node_degrees(
                        graphs, threshold=threshold, absolute=False
                    )
                    task_degree_results[task] = degree_results

            if not task_degree_results:
                print(f"No valid graphs found for threshold {threshold}, skipping...")
                continue

            # Average the results across all tasks
            print("Computing averages across tasks...")
            averaged_degree_results = average_degree_results_across_tasks(
                task_degree_results
            )
            comparison_df = compare_node_degrees_across_models(averaged_degree_results)

            # Save results
            root = f"./results/degree_diff_results/{threshold}"
            os.makedirs(f"{root}", exist_ok=True)
            comparison_df.to_csv(
                f"{root}/degree_comparison_avg_ft_pii_tasks.csv", index=False
            )

            # Generate plots with averaged data
            task_name = "avg_ft_pii_tasks"
            print("Generating plots...")

            plot_attention_block_heatmap(
                comparison_df,
                degree_type="total_degree",
                aggregation="sum",
                threshold=threshold,
                task=task_name,
                model_config_size=model_config_size,
                figsize=(10, 4),
            )

            matplotlib.pyplot.close()

            print(f"Completed processing for threshold {threshold}")

    print("All processing completed!")
