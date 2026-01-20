import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import pandas as pd
import re
from constants import ft_pii_tasks
from gencircuits.eap.graph import Graph
import torch
from constants import model_display_name_dict, display_name_dict


def get_nodes_above_threshold(
    graph: Graph, threshold: float, absolute: bool = True
) -> set:
    """
    Get set of nodes that have at least one edge above threshold.

    Args:
        graph: Graph object to analyze
        threshold: Minimum edge score to consider
        absolute: Whether to use absolute values of scores

    Returns:
        Set of node names that have edges above threshold
    """
    edge_scores = graph.scores.clone()
    if absolute:
        edge_scores = torch.abs(edge_scores)

    # Create mask for edges above threshold
    above_threshold = (edge_scores >= threshold) & graph.real_edge_mask

    nodes_above_threshold = set()

    for node_name, node in graph.nodes.items():
        if node_name == "logits":
            continue

        forward_idx = graph.forward_index(node, attn_slice=False)

        # Check if node has any outgoing edges above threshold
        has_out_edges = above_threshold[forward_idx, :].sum().item() > 0

        # Check if node has any incoming edges above threshold
        has_in_edges = False
        if not hasattr(node, "__class__") or node.__class__.__name__ != "InputNode":
            if hasattr(node, "qkv_inputs") and node.qkv_inputs:
                for qkv in ["q", "k", "v"]:
                    backward_idx = graph.backward_index(node, qkv=qkv, attn_slice=False)
                    if above_threshold[:, backward_idx].sum().item() > 0:
                        has_in_edges = True
                        break
            else:
                backward_idx = graph.backward_index(node, attn_slice=False)
                has_in_edges = above_threshold[:, backward_idx].sum().item() > 0

        if has_out_edges or has_in_edges:
            nodes_above_threshold.add(node_name)

    return nodes_above_threshold


def get_edges_above_threshold(
    graph: Graph, threshold: float, absolute: bool = True
) -> set:
    """
    Get set of edges that are above threshold.

    Args:
        graph: Graph object to analyze
        threshold: Minimum edge score to consider
        absolute: Whether to use absolute values of scores

    Returns:
        Set of tuples (from_node, to_node) representing edges above threshold
    """
    edge_scores = graph.scores.clone()
    if absolute:
        edge_scores = torch.abs(edge_scores)

    # Create mask for edges above threshold
    above_threshold = (edge_scores >= threshold) & graph.real_edge_mask

    # Count total number of edges above threshold
    edges_count = above_threshold.sum().item()

    return edges_count


def analyze_checkpoint_changes(
    checkpoint_graphs: Dict[str, Graph], threshold: float, absolute: bool = True
) -> Tuple[List[int], List[int]]:
    """
    Analyze changes in nodes and edges across checkpoints.

    Args:
        checkpoint_graphs: Dictionary mapping checkpoint names to Graph objects
        threshold: Minimum edge score to consider
        absolute: Whether to use absolute values of scores

    Returns:
        Tuple of (node_changes, edge_changes) lists where each list contains
        the count of changes for each checkpoint transition
    """
    # Sort checkpoints by number
    checkpoint_names = sorted(
        checkpoint_graphs.keys(), key=lambda x: int(x.split("-")[1]) if "-" in x else 0
    )

    node_changes = []
    edge_changes = []

    prev_nodes = None
    prev_edges_count = None

    for checkpoint_name in checkpoint_names:
        graph = checkpoint_graphs[checkpoint_name]

        current_nodes = get_nodes_above_threshold(graph, threshold, absolute)
        current_edges_count = get_edges_above_threshold(graph, threshold, absolute)

        if prev_nodes is not None:
            # Count nodes that changed status (crossed threshold in either direction)
            node_changes_count = len(current_nodes.symmetric_difference(prev_nodes))
            node_changes.append(node_changes_count)

            # Count change in edge count
            edge_changes_count = abs(current_edges_count - prev_edges_count)
            edge_changes.append(edge_changes_count)

        prev_nodes = current_nodes
        prev_edges_count = current_edges_count

    return node_changes, edge_changes


def plot_checkpoint_changes(
    model_changes: Dict[str, Tuple[List[int], List[int]]],
    threshold: float,
    task: str = "avg_ft_pii_tasks",
    figsize: Tuple[int, int] = (15, 10),
    model_config_size: str = "medium",
):
    """
    Plot node and edge changes across checkpoints for multiple models.

    Args:
        model_changes: Dictionary mapping model names to (node_changes, edge_changes) tuples
        threshold: Threshold value used for analysis
        task: Task name for file naming
        figsize: Figure size for the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    colors = ["blue", "red", "green", "orange", "purple", "brown"]

    for i, (model_name, (node_changes, edge_changes)) in enumerate(
        model_changes.items()
    ):
        color = colors[i % len(colors)]
        checkpoint_numbers = list(range(1, len(node_changes) + 1))

        # Plot node changes
        ax1.plot(
            checkpoint_numbers,
            node_changes,
            marker="o",
            label=model_display_name_dict[model_name],
            color=color,
            linewidth=2,
            markersize=6,
        )

        # Plot edge changes
        ax2.plot(
            checkpoint_numbers,
            edge_changes,
            marker="s",
            label=model_display_name_dict[model_name],
            color=color,
            linewidth=2,
            markersize=6,
        )

    # Configure node changes plot
    ax1.set_xlabel("Checkpoint")
    ax1.set_ylabel("Number of Node Changes")
    ax1.set_title(f"Node Changes Across Checkpoints (Threshold: {threshold})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if model_changes:
        max_checkpoints = max(len(changes[0]) for changes in model_changes.values())
        ax1.set_xticks(range(1, max_checkpoints + 1))

    # Configure edge changes plot
    ax2.set_xlabel("Checkpoint")
    ax2.set_ylabel("Number of Edge Count Changes")
    ax2.set_title(f"Edge Count Changes Across Checkpoints (Threshold: {threshold})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if model_changes:
        max_checkpoints = max(len(changes[1]) for changes in model_changes.values())
        ax2.set_xticks(range(1, max_checkpoints + 1))

    plt.tight_layout()

    # Save plots
    os.makedirs("./plots/checkpoint_changes", exist_ok=True)
    plt.savefig(
        f"./plots/checkpoint_changes/checkpoint_changes_{task}_threshold_{threshold}_{model_config_size}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        dpi=400,
    )


def plot_individual_task_changes(
    task_changes_data: Dict[str, Dict[str, Tuple[List[int], List[int]]]],
    threshold: float,
    figsize: Tuple[int, int] = (15, 12),
    model_config_size: str = "medium",
):
    """
    Plot node and edge changes for individual PII tasks across checkpoints.

    Args:
        task_changes_data: Dictionary mapping model names to task data
        threshold: Threshold value used for analysis
        figsize: Figure size for the plot
    """
    # Create separate plots for nodes and edges
    fig_nodes, ax_nodes = plt.subplots(1, 1, figsize=figsize)
    fig_edges, ax_edges = plt.subplots(1, 1, figsize=figsize)

    colors = ["blue", "red", "green", "orange", "purple", "brown"]
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    plot_index = 0

    for model_name, model_tasks in task_changes_data.items():
        for task_name, (node_changes, edge_changes) in model_tasks.items():
            color = colors[plot_index % len(colors)]
            line_style = line_styles[plot_index % len(line_styles)]
            marker = markers[plot_index % len(markers)]

            checkpoint_numbers = list(range(1, len(node_changes) + 1))
            label = f"{model_display_name_dict[model_name]} - {display_name_dict[task_name]}"

            # Plot node changes
            ax_nodes.plot(
                checkpoint_numbers,
                node_changes,
                color=color,
                linestyle=line_style,
                marker=marker,
                label=label,
                linewidth=2,
                markersize=6,
            )

            # Plot edge changes
            ax_edges.plot(
                checkpoint_numbers,
                edge_changes,
                color=color,
                linestyle=line_style,
                marker=marker,
                label=label,
                linewidth=2,
                markersize=6,
            )

            plot_index += 1

    # Configure node changes plot
    ax_nodes.set_xlabel("Checkpoint Transition")
    ax_nodes.set_ylabel("Number of Node Changes")
    ax_nodes.set_title(
        f"Node Changes Across Checkpoints by Task (Threshold: {threshold})"
    )
    ax_nodes.legend(loc="upper right")
    ax_nodes.grid(True, alpha=0.3)

    # Configure edge changes plot
    ax_edges.set_xlabel("Checkpoint Transition")
    ax_edges.set_ylabel("Number of Edge Count Changes")
    ax_edges.set_title(
        f"Edge Count Changes Across Checkpoints by Task (Threshold: {threshold})"
    )
    ax_edges.legend(loc="upper right")
    ax_edges.grid(True, alpha=0.3)

    # Set x-axis ticks
    if task_changes_data:
        max_checkpoints = 0
        for model_tasks in task_changes_data.values():
            for node_changes, edge_changes in model_tasks.values():
                max_checkpoints = max(max_checkpoints, len(node_changes))

        ax_nodes.set_xticks(range(1, max_checkpoints + 1))
        ax_edges.set_xticks(range(1, max_checkpoints + 1))

    # Save plots
    os.makedirs("./plots/checkpoint_changes", exist_ok=True)

    fig_nodes.tight_layout()
    fig_nodes.savefig(
        f"./plots/checkpoint_changes/individual_task_node_changes_threshold_{threshold}_{model_config_size}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        dpi=400,
    )

    fig_edges.tight_layout()
    fig_edges.savefig(
        f"./plots/checkpoint_changes/individual_task_edge_changes_threshold_{threshold}_{model_config_size}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        dpi=400,
    )

    plt.close(fig_nodes)
    plt.close(fig_edges)


def get_attention_layer_scores(
    graph: Graph, target_layers: List[int] = [0, 3, 10]
) -> Dict[str, float]:
    """
    Extract attention layer scores from the graph for specific layers.

    Args:
        graph: Graph object to analyze
        target_layers: List of attention layer indices to extract scores for

    Returns:
        Dictionary mapping edge names to their scores for the target attention layers
    """
    attention_scores = {}

    for edge_name, edge in graph.edges.items():
        # Check if this edge involves one of our target attention layers
        for layer in target_layers:
            # Pattern for attention-to-attention edges: a{layer}.h{head}->a{layer}.h{head}<{qkv}>
            attn_to_attn_pattern = rf"a{layer}\.h\d+->a\d+\.h\d+<[kqv]>"
            # Pattern for attention-to-MLP edges: a{layer}.h{head}->m{layer}
            attn_to_mlp_pattern = rf"a{layer}\.h\d+->m\d+"
            # Pattern for input/other-to-attention edges: {source}->a{layer}.h{head}<{qkv}>
            to_attn_pattern = rf".*->a{layer}\.h\d+<[kqv]>"
            # Pattern for logits
            to_logits_pattern = rf"a{layer}\.h\d+->logits"

            if (
                re.match(attn_to_attn_pattern, edge_name)
                or re.match(attn_to_mlp_pattern, edge_name)
                or re.match(to_attn_pattern, edge_name)
                or re.match(to_logits_pattern, edge_name)
            ):
                attention_scores[edge_name] = (
                    edge.score.item()
                    if hasattr(edge.score, "item")
                    else float(edge.score)
                )

    return attention_scores


def analyze_attention_scores_across_checkpoints(
    checkpoint_graphs: Dict[str, Graph], target_layers: List[int] = [0, 3, 10]
) -> Dict[str, List[float]]:
    """
    Analyze attention layer scores across checkpoints.

    Args:
        checkpoint_graphs: Dictionary mapping checkpoint names to Graph objects
        target_layers: List of attention layer indices to analyze

    Returns:
        Dictionary mapping edge names to lists of scores across checkpoints
    """
    # Sort checkpoints by number
    checkpoint_names = sorted(
        checkpoint_graphs.keys(), key=lambda x: int(x.split("-")[1]) if "-" in x else 0
    )

    # Collect all unique edge names across all checkpoints
    all_edge_names = set()
    for graph in checkpoint_graphs.values():
        attention_scores = get_attention_layer_scores(graph, target_layers)
        all_edge_names.update(attention_scores.keys())

    # Initialize results dictionary
    edge_score_trajectories = {edge_name: [] for edge_name in all_edge_names}

    # Collect scores for each checkpoint
    for checkpoint_name in checkpoint_names:
        graph = checkpoint_graphs[checkpoint_name]
        attention_scores = get_attention_layer_scores(graph, target_layers)

        for edge_name in all_edge_names:
            score = attention_scores.get(
                edge_name, 0.0
            )  # Default to 0 if edge doesn't exist
            edge_score_trajectories[edge_name].append(score)

    return edge_score_trajectories


def plot_attention_scores_across_checkpoints(
    model_attention_data: Dict[str, Dict[str, Dict[str, List[float]]]],
    target_layers: List[int] = [0, 3, 10],
    figsize: Tuple[int, int] = (20, 15),
    model_config_size: str = "medium",
):
    """
    Plot attention layer scores across checkpoints for multiple models and tasks.

    Args:
        model_attention_data: Nested dict: {model_name: {task_name: {edge_name: [scores]}}}
        target_layers: List of attention layer indices being analyzed
        figsize: Figure size for the plot
    """
    # First pass: find the global maximum across all layers and data
    global_max = 0.0

    for layer in target_layers:
        for model_name, model_tasks in model_attention_data.items():
            for task_name, edge_trajectories in model_tasks.items():
                # Filter edges for this specific layer
                layer_edges = {
                    edge: scores
                    for edge, scores in edge_trajectories.items()
                    if f"a{layer}." in edge
                }

                if layer_edges:
                    max_len = max(len(scores) for scores in layer_edges.values())

                    for checkpoint_idx in range(max_len):
                        checkpoint_scores = []
                        for edge_scores in layer_edges.values():
                            if checkpoint_idx < len(edge_scores):
                                checkpoint_scores.append(
                                    abs(edge_scores[checkpoint_idx])
                                )

                        if checkpoint_scores:
                            avg_score = np.mean(checkpoint_scores)
                            global_max = max(global_max, avg_score)

    # Create subplots for each target layer
    fig, axes = plt.subplots(len(target_layers), 1, figsize=figsize)
    if len(target_layers) == 1:
        axes = [axes]

    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    line_styles = ["-", "--", "-.", ":"]

    for layer_idx, layer in enumerate(target_layers):
        ax = axes[layer_idx]
        plot_index = 0

        # Plot data for each model and task
        for model_name, model_tasks in model_attention_data.items():
            for task_name, edge_trajectories in model_tasks.items():
                # Filter edges for this specific layer
                layer_edges = {
                    edge: scores
                    for edge, scores in edge_trajectories.items()
                    if f"a{layer}." in edge
                }

                if not layer_edges:
                    continue

                # Calculate average score across all edges for this layer
                if layer_edges:
                    max_len = max(len(scores) for scores in layer_edges.values())
                    avg_scores = []

                    for checkpoint_idx in range(max_len):
                        checkpoint_scores = []
                        for edge_scores in layer_edges.values():
                            if checkpoint_idx < len(edge_scores):
                                checkpoint_scores.append(
                                    abs(edge_scores[checkpoint_idx])
                                )  # Use absolute value

                        if checkpoint_scores:
                            avg_scores.append(np.mean(checkpoint_scores))

                    if avg_scores:
                        color = colors[plot_index % len(colors)]
                        line_style = line_styles[plot_index % len(line_styles)]
                        checkpoint_numbers = list(range(1, len(avg_scores) + 1))

                        label = f"{model_display_name_dict[model_name]} - {display_name_dict[task_name]}"
                        ax.plot(
                            checkpoint_numbers,
                            avg_scores,
                            color=color,
                            linestyle=line_style,
                            marker="o",
                            label=label,
                            linewidth=2,
                            markersize=4,
                        )

                        plot_index += 1

        # Configure subplot with consistent y-axis
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Average Absolute Score")
        ax.set_title(f"Attention Layer {layer} - Average Scores Across Checkpoints")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Set consistent y-axis range for all subplots
        ax.set_ylim(0, global_max * 1.05)  # Add 5% padding at the top

        # Set x-axis ticks
        if model_attention_data:
            max_checkpoints = 0
            for model_tasks in model_attention_data.values():
                for edge_trajectories in model_tasks.values():
                    for scores in edge_trajectories.values():
                        max_checkpoints = max(max_checkpoints, len(scores))

            if max_checkpoints > 0:
                ax.set_xticks(range(1, max_checkpoints + 1))

    plt.tight_layout()

    # Save plots
    os.makedirs("./plots/checkpoint_changes", exist_ok=True)

    layer_str = "_".join(map(str, target_layers))
    fig.savefig(
        f"./plots/checkpoint_changes/attention_scores_layers_{layer_str}_{model_config_size}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        dpi=400,
    )

    plt.close(fig)


def load_checkpoint_graphs(model_path: str, task: str) -> Dict[str, Graph]:
    """
    Load graphs from all available checkpoints for a given model and task.

    Args:
        model_path: Path to the model directory
        task: Task name (e.g., 'pii_leakage_dem')

    Returns:
        Dictionary mapping checkpoint names to Graph objects
    """
    checkpoint_graphs = {}
    target_file = f"{task}_kl.json"

    # List all checkpoint directories
    if not os.path.exists(model_path):
        print(f"Warning: Model path {model_path} does not exist")
        return checkpoint_graphs

    for item in os.listdir(model_path):
        if item.startswith("checkpoint-") and os.path.isdir(
            os.path.join(model_path, item)
        ):
            checkpoint_file = os.path.join(model_path, item, target_file)
            if os.path.exists(checkpoint_file):
                try:
                    graph = Graph.from_json(checkpoint_file)
                    checkpoint_graphs[item] = graph
                except Exception as e:
                    print(f"Warning: Failed to load {checkpoint_file}: {e}")

    return checkpoint_graphs


if __name__ == "__main__":
    # Model configurations by size
    model_configs_by_size = {
        "all": {
            "gpt2-small-baseline": "graphs/gpt2-small-baseline",
            "pythia-160m-baseline": "graphs/pythia-160m-baseline",
            "gpt2-medium-baseline": "graphs/gpt2-medium-baseline",
            "pythia-410m-baseline": "graphs/pythia-410m-baseline",
        }
    }

    # Analyze checkpoint changes for different thresholds
    for threshold in [0.00006]:
        print(f"Processing threshold: {threshold}")

        for model_config_size in model_configs_by_size.keys():
            model_configs = model_configs_by_size[model_config_size]

            # Collect changes across all tasks for each model
            all_models_changes = {}
            # Also collect individual task data for separate plots
            task_changes_data = {}
            # Collect attention score data
            attention_data = {}

            for model_name, model_path in model_configs.items():
                print(f"Processing model: {model_name}")

                # Collect changes for each task
                task_node_changes = []
                task_edge_changes = []
                # Store individual task results
                model_task_data = {}
                # Store attention score results
                model_attention_data = {}

                for task in ft_pii_tasks:
                    print(f"  Processing task: {task}")

                    # Load checkpoint graphs for this task
                    checkpoint_graphs = load_checkpoint_graphs(model_path, task)

                    if len(checkpoint_graphs) < 2:
                        print(
                            f"Warning: Need at least 2 checkpoints for {model_name}/{task}, found {len(checkpoint_graphs)}"
                        )
                        continue

                    # Analyze changes across checkpoints
                    node_changes, edge_changes = analyze_checkpoint_changes(
                        checkpoint_graphs, threshold, absolute=True
                    )

                    # Analyze attention scores across checkpoints
                    target_layers = (
                        [0, 3, 4, 9, 10]
                        if model_config_size == "small"
                        else [1, 14, 15, 16, 17]
                    )
                    attention_trajectories = (
                        analyze_attention_scores_across_checkpoints(
                            checkpoint_graphs, target_layers=target_layers
                        )
                    )

                    # Store individual task data
                    model_task_data[task] = (node_changes, edge_changes)
                    model_attention_data[task] = attention_trajectories

                    if (
                        task_node_changes
                    ):  # If we have previous data, ensure same length
                        min_len = min(len(task_node_changes[0]), len(node_changes))
                        task_node_changes = [
                            changes[:min_len] for changes in task_node_changes
                        ]
                        task_edge_changes = [
                            changes[:min_len] for changes in task_edge_changes
                        ]
                        node_changes = node_changes[:min_len]
                        edge_changes = edge_changes[:min_len]

                    task_node_changes.append(node_changes)
                    task_edge_changes.append(edge_changes)

                if task_node_changes:  # If we have data for this model
                    # Store individual task data
                    task_changes_data[model_name] = model_task_data
                    attention_data[model_name] = model_attention_data

                    # Average changes across tasks
                    avg_node_changes = [
                        int(
                            np.mean(
                                [task_changes[i] for task_changes in task_node_changes]
                            )
                        )
                        for i in range(len(task_node_changes[0]))
                    ]
                    avg_edge_changes = [
                        int(
                            np.mean(
                                [task_changes[i] for task_changes in task_edge_changes]
                            )
                        )
                        for i in range(len(task_edge_changes[0]))
                    ]
                    sum_node_changes = int(np.sum(avg_node_changes))
                    sum_edge_changes = int(np.sum(avg_edge_changes))

                    all_models_changes[model_name] = (
                        avg_node_changes,
                        avg_edge_changes,
                    )

                    print(f"  Average node changes per checkpoint: {avg_node_changes}")
                    print(f"  Average edge changes per checkpoint: {avg_edge_changes}")
                    print(f"  Sum node changes: {sum_node_changes}")
                    print(f"  Sum edge changes: {sum_edge_changes}")

            if all_models_changes:  # If we have data for any models
                # Create averaged plots
                task_name = "avg_ft_pii_tasks"
                print(f"Generating averaged plots for threshold {threshold}...")

                # plot_checkpoint_changes(
                #     all_models_changes, threshold, task=task_name, figsize=(12, 10), model_config_size=model_config_size
                # )

                # Create individual task plots
                print(f"Generating individual task plots for threshold {threshold}...")
                plot_individual_task_changes(
                    task_changes_data,
                    threshold,
                    figsize=(15, 10),
                    model_config_size=model_config_size,
                )

                # Create attention score plots
                print(f"Generating attention score plots...")
                plot_attention_scores_across_checkpoints(
                    attention_data,
                    target_layers=target_layers,
                    figsize=(15, 12),
                    model_config_size=model_config_size,
                )

                # Save summary statistics
                results_dir = f"./results/checkpoint_changes/{threshold}"
                os.makedirs(results_dir, exist_ok=True)

                # Create summary DataFrame for averaged data
                summary_data = []
                for model_name, (
                    node_changes,
                    edge_changes,
                ) in all_models_changes.items():
                    for i, (nc, ec) in enumerate(zip(node_changes, edge_changes)):
                        summary_data.append(
                            {
                                "model": model_name,
                                "checkpoint_transition": i + 1,
                                "node_changes": nc,
                                "edge_changes": ec,
                                "threshold": threshold,
                                "data_type": "averaged",
                            }
                        )

                # Create summary DataFrame for individual task data
                for model_name, model_tasks in task_changes_data.items():
                    for task_name, (node_changes, edge_changes) in model_tasks.items():
                        for i, (nc, ec) in enumerate(zip(node_changes, edge_changes)):
                            summary_data.append(
                                {
                                    "model": model_name,
                                    "task": task_name,
                                    "checkpoint_transition": i + 1,
                                    "node_changes": nc,
                                    "edge_changes": ec,
                                    "threshold": threshold,
                                    "data_type": "individual_task",
                                }
                            )

                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(
                    f"{results_dir}/checkpoint_changes_summary_all_data.csv",
                    index=False,
                )

                matplotlib.pyplot.close()
                print(f"Completed processing for threshold {threshold}")
            else:
                print(f"No valid data found for threshold {threshold}")

    print("All processing completed!")
