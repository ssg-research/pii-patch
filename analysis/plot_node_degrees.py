import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import pandas as pd
from analysis.node_degrees_plots import (
    plot_block_heatmap,
    plot_layer_aggregated_heatmap,
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


if __name__ == "__main__":
    # all_tasks = ft_pii_tasks + general_tasks
    for task in ft_pii_tasks:
        target_file = f"{task}_kl.json"
        graphs = {
            # "GPT2-Small_Orig": Graph.from_json(
            #     f"graphs/gpt2-small-original/{target_file}"
            # ),
            "GPT2-Small_NoDef": Graph.from_json(
                f"graphs/gpt2-small-baseline/{target_file}"
            ),
            "Pythia-160_NoDef": Graph.from_json(
                f"graphs/pythia-160m-baseline/{target_file}"
            ),
            # "GPT2-Small_DP2": Graph.from_json(f"graphs/gpt2-small-dp2/{target_file}"),
            # "GPT2-Small_DP4": Graph.from_json(f"graphs/gpt2-small-dp4/{target_file}"),
            # "GPT2-Small_DP8": Graph.from_json(f"graphs/gpt2-small-dp8/{target_file}"),
        }

        # Analyze node degrees - we identify nodes with EAP-IG scores above certain thresholds
        for threshold in [0.0001, 0.001, 0.01]:
            degree_results = analyze_node_degrees(graphs, threshold=threshold, absolute=False)
            comparison_df = compare_node_degrees_across_models(degree_results)    
            root = f"./results/degree_diff_results/{threshold}"        
            os.makedirs(f"{root}", exist_ok=True)
            comparison_df.to_csv(f"{root}/degree_comparison_{task}.csv", index=False)

            # Plots!
            # plot_degree_distributions(degree_results, degree_type="total_degree", task=task)
            plot_layer_aggregated_heatmap(
                comparison_df,
                degree_type="total_degree",
                aggregation="sum",
                threshold=threshold,
                task=task,
            )
            # plot included atention edges (EAP-IG above a threshold)
            plot_block_heatmap(
                comparison_df,
                degree_type="total_degree",
                aggregation="sum",
                threshold=threshold,
                task=task,
                node_type="AttentionNode",
            )
            # plot included MLP edges (EAP-IG above a threshold)
            plot_block_heatmap(
                comparison_df,
                degree_type="total_degree",
                aggregation="sum",
                threshold=threshold,
                task=task,
                node_type="MLPNode",
            )
            # plot_degree_comparison_heatmap(
            #     comparison_df,
            #     degree_type="total_degree",
            #     threshold=threshold,
            #     task=task,
            # )

            matplotlib.pyplot.close()

            # Analyze differences from the no defense baseline
            baseline_target = "GPT2-Small_NoDef"
            diff_analysis = analyze_degree_differences(
                comparison_df,
                baseline_model=baseline_target,
                degree_type="total_degree",
            )
            diff_analysis.to_csv(
                f"{root}/degree_differences_{task}_{baseline_target}.csv", index=False
            )

            # print("Nodes with largest degree differences:")
            # baseline_model = "GPT2-Small_Orig"
            # baseline_diff = f"{baseline_model}_diff"
            # n = 10
            # top_diff_nodes = diff_analysis.nlargest(n, baseline_diff)[
            #     ["node", "layer", "node_type", baseline_diff]
            # ]
            # top_diff_nodes.to_csv(
            #     f"{root}/top_degree_differences_n{n}_{task}_{baseline_target}_vs_{baseline_diff}.csv",
            #     index=False,
            # )
