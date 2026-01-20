import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
from constants import model_display_name_dict
from analysis.general_utils import display_name


def plot_layer_aggregated_heatmap(
    comparison_df: pd.DataFrame,
    degree_type: str = "total_degree",
    figsize: Tuple[int, int] = (12, 8),
    task: str = "dem",
    aggregation: str = "mean",
    threshold: Optional[float] = None,
    model_config_size: str = "small"
):
    """
    Create a heatmap with nodes aggregated by layer.

    Args:
        comparison_df: DataFrame from compare_node_degrees_across_models
        degree_type: Type of degree to plot
        figsize: Figure size for the plot
        task: Task name for saving
        aggregation: How to aggregate ('mean', 'sum', 'max', 'median')
    """
    # Extract columns for the specified degree type
    degree_columns = [
        col for col in comparison_df.columns if col.endswith(f"_{degree_type}")
    ]

    # Create aggregated data by layer
    df_with_layers = comparison_df.copy()
    df_with_layers["layer_group"] = df_with_layers["layer"].astype(str)

    # Group by layer and aggregate
    if aggregation == "mean":
        agg_data = df_with_layers.groupby("layer_group")[degree_columns].mean()
    elif aggregation == "sum":
        agg_data = df_with_layers.groupby("layer_group")[degree_columns].sum()
    elif aggregation == "max":
        agg_data = df_with_layers.groupby("layer_group")[degree_columns].max()
    elif aggregation == "median":
        agg_data = df_with_layers.groupby("layer_group")[degree_columns].median()

    # Sort layers in ascending order
    sorted_layers = sorted(agg_data.index, key=lambda x: int(x))
    agg_data = agg_data.reindex(sorted_layers)

    # Create labels
    layer_labels = [f"Layer {layer}" for layer in agg_data.index]
    model_labels = [col.replace(f"_{degree_type}", "") for col in degree_columns]

    plt.figure(figsize=figsize)
    sns.heatmap(
        agg_data.values.T,
        xticklabels=layer_labels,
        yticklabels=model_labels,
        annot=True,
        fmt=".1f" if aggregation == "mean" else "d",
        cmap="viridis",
        cbar_kws={
            "label": f'{degree_type.replace("_", " ").title()} ({aggregation.title()})'
        },
    )

    plt.title(
        f'{display_name(task)} Layer-wise {degree_type.replace("_", " ").title()} Comparison ({aggregation.title()})'
    )
    plt.xlabel("Layers")
    plt.ylabel("Models")
    plt.tight_layout()
    plt.savefig(
        f"./plots/heatmaps/layers/heatmap_layer_{aggregation}_{task}_{threshold}_{model_config_size}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        dpi=400,
    )


def plot_block_heatmap(
    comparison_df: pd.DataFrame,
    degree_type: str = "total_degree",
    figsize: Tuple[int, int] = (10, 6),
    node_type: str = "AttentionNode",
    task: str = "dem",
    aggregation: str = "mean",
    threshold: Optional[float] = None,
    model_config_size: str = ""
):
    """
    Create a heatmap with nodes aggregated by block (e.g., a0, a1 for AttentionNode, m0, m1 for MLPNode).

    Args:
        comparison_df: DataFrame from compare_node_degrees_across_models
        degree_type: Type of degree to plot
        node_type: Node type to aggregate ("AttentionNode" or "MLPNode")
        figsize: Figure size for the plot
        task: Task name for saving
        aggregation: How to aggregate ('mean', 'sum', 'max', 'median')
    """
    # Extract columns for the specified degree type
    degree_columns = [
        col for col in comparison_df.columns if col.endswith(f"_{degree_type}")
    ]

    # Filter only specified node type and create block groups
    block_df = comparison_df[comparison_df["node_type"] == node_type].copy()
    if node_type == "AttentionNode":
        block_df["block"] = block_df["node"].str.extract(r"(a\d+)")[0]
        block_prefix = "Attention"
        save_dir = "attention"
    elif node_type == "MLPNode":
        block_df["block"] = block_df["node"].str.extract(r"(m\d+)")[0]
        block_prefix = "MLP"
        save_dir = "mlp"
    else:
        raise ValueError("node_type must be 'AttentionNode' or 'MLPNode'")

    # Group by block and aggregate
    if aggregation == "mean":
        agg_data = block_df.groupby("block")[degree_columns].mean()
    elif aggregation == "sum":
        agg_data = block_df.groupby("block")[degree_columns].sum()
    elif aggregation == "max":
        agg_data = block_df.groupby("block")[degree_columns].max()
    elif aggregation == "median":
        agg_data = block_df.groupby("block")[degree_columns].median()

    # Sort by block number
    agg_data = agg_data.reindex(sorted(agg_data.index, key=lambda x: int(x[1:])))

    # Create labels
    attention_labels = [f"Attention {block.replace('a', '')}" for block in agg_data.index]
    model_labels = [model_display_name_dict[col.replace(f"_{degree_type}", "")] for col in degree_columns]

    plt.figure(figsize=figsize)
    sns.heatmap(
        agg_data.values.T,
        xticklabels=block_labels,
        yticklabels=model_labels,
        annot=True,
        fmt=".1f" if aggregation == "mean" else "d",
        cmap="viridis",
        cbar_kws={
            "label": f'Influential Component Count'
        },
    )

    plt.title(
        f'Influential Attention Blocks in Small Fine-tuned Language Models'
    )
    plt.xlabel("Attention Blocks")
    plt.subplots_adjust(top=0.9)
    plt.ylabel("Models")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(
        f"./plots/heatmaps/attention/heatmap_attention_{aggregation}_{task}_{threshold}_{model_config_size}.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=400,
    )
    plt.show()


def plot_hierarchical_heatmap(
    comparison_df: pd.DataFrame,
    degree_type: str = "total_degree",
    figsize: Tuple[int, int] = (16, 10),
    task: str = "dem",
    top_k: int = 30,
    model_config_size: str = ''
):
    """
    Create a hierarchical heatmap showing top-k most connected nodes,
    grouped by layer and node type.

    Args:
        comparison_df: DataFrame from compare_node_degrees_across_models
        degree_type: Type of degree to plot
        figsize: Figure size for the plot
        task: Task name for saving
        top_k: Number of top nodes to show
    """
    # Extract columns for the specified degree type
    degree_columns = [
        col for col in comparison_df.columns if col.endswith(f"_{degree_type}")
    ]

    # Calculate average degree across all models for sorting
    comparison_df["avg_degree"] = comparison_df[degree_columns].mean(axis=1)

    # Get top-k nodes by average degree
    top_nodes = comparison_df.nlargest(top_k, "avg_degree")

    # Sort by layer then by node type, then by average degree
    top_nodes = top_nodes.sort_values(
        ["layer", "node_type", "avg_degree"], ascending=[True, True, False]
    )

    # Create the heatmap data
    heatmap_data = top_nodes[degree_columns].values

    # Create enhanced node labels with layer and type info
    node_labels = []
    for _, row in top_nodes.iterrows():
        if row["node_type"] == "AttentionNode":
            label = f"{row['node']} (L{row['layer']})"
        else:
            label = f"{row['node']} (L{row['layer']})"
        node_labels.append(label)

    model_labels = [col.replace(f"_{degree_type}", "") for col in degree_columns]

    plt.figure(figsize=figsize)

    # Create the heatmap
    ax = sns.heatmap(
        heatmap_data.T,
        xticklabels=node_labels,
        yticklabels=model_labels,
        annot=True,
        fmt="d",
        cmap="viridis",
        cbar_kws={"label": f'{degree_type.replace("_", " ").title()}'},
    )

    # Add vertical lines to separate layers
    layer_changes = []
    current_layer = top_nodes.iloc[0]["layer"]
    for i, (_, row) in enumerate(top_nodes.iterrows()):
        if row["layer"] != current_layer:
            layer_changes.append(i)
            current_layer = row["layer"]

    for change_point in layer_changes:
        ax.axvline(x=change_point, color="red", linewidth=2, alpha=0.7)

    plt.title(
        f'Top {top_k} Nodes by {degree_type.replace("_", " ").title()} (Grouped by Layer)'
    )
    plt.xlabel("Nodes (Layer)")
    plt.ylabel("Models")
    plt.xticks(rotation=45, ha="right")    
    plt.tight_layout()
    plt.savefig(
        f"./plots/heatmaps/hierarchical/heatmap_hierarchical_top{top_k}_{task}_{model_config_size}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        dpi=400,
    )


def plot_degree_comparison_heatmap(
    comparison_df: pd.DataFrame,
    degree_type: str = "total_degree",
    figsize: Tuple[int, int] = (12, 8),
    task: str = "dem",
    threshold: float = 0.0001,
    model_config_size: str = ""
):
    """
    Create a heatmap comparing degrees across models.

    Args:
        comparison_df: DataFrame from compare_node_degrees_across_models
        degree_type: Type of degree to plot
        figsize: Figure size for the plot
    """
    # Extract columns for the specified degree type
    degree_columns = [
        col for col in comparison_df.columns if col.endswith(f"_{degree_type}")
    ]

    # Create a matrix for the heatmap
    heatmap_data = comparison_df[degree_columns].values
    node_labels = comparison_df["node"].values
    model_labels = [col.replace(f"_{degree_type}", "") for col in degree_columns]

    plt.figure(figsize=figsize)
    sns.heatmap(
        heatmap_data.T,
        xticklabels=node_labels,
        yticklabels=model_labels,
        annot=True,
        fmt="d",
        cmap="viridis",
        cbar_kws={"label": f'{degree_type.replace("_", " ").title()}'},
    )

    plt.title(f'Node {degree_type.replace("_", " ").title()} Comparison Across Models')
    plt.xlabel("Nodes")
    plt.ylabel("Models")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        f"./plots/original/heatmap_{task}_{threshold}_{model_config_size}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        dpi=400,
    )