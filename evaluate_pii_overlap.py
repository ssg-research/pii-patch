import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import hashlib
from scipy.stats import hypergeom
from matplotlib.colors import LinearSegmentedColormap
from gencircuits.eap.graph import Graph
from constants import (
    ft_pii_tasks,
    display_name_dict,
    model_display_name_dict,
    MODEL_COLORS,
)

# Create colormaps from the base colors
MODEL_COLORMAPS = []
for color in MODEL_COLORS:
    # Create a colormap from white to the specified color
    colors_list = ["white", color]
    MODEL_COLORMAPS.append(LinearSegmentedColormap.from_list("", colors_list))


def get_model_colormap(model_name):
    """Get the appropriate colormap for a model based on its base name"""
    base_name = (
        model_name.replace("-baseline", "")
        .replace("-dp1", "")
        .replace("-dp2", "")
        .replace("-dp4", "")
        .replace("-dp8", "")
    )

    # Define the expected order matching analyze_leak_results.py
    model_order = [
        "gpt2-small",
        "gpt2-medium",
        "qwen3-06",
        "gpt2-large",
        "llama3-1b",
        "qwen3-17",
    ]

    if base_name in model_order:
        idx = model_order.index(base_name)
        return MODEL_COLORMAPS[idx]
    else:
        # Default to first colormap if model not found
        return MODEL_COLORMAPS[0]


def get_cache_key(model_name, threshold):
    """Generate a cache key based on model name and threshold"""
    key_string = f"{model_name}_{threshold}_{len(ft_pii_tasks)}"
    return hashlib.md5(key_string.encode()).hexdigest()[:12]


def save_models_data(models_data, cache_dir="cache/models_data"):
    """Save models_data to cache files"""
    os.makedirs(cache_dir, exist_ok=True)

    for model_name, data in models_data.items():
        cache_key = get_cache_key(model_name, data["threshold"])
        cache_file = f"{cache_dir}/{model_name}_{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            print(f"Cached data for {model_name} saved to {cache_file}")
        except Exception as e:
            print(f"Warning: Could not save cache for {model_name}: {e}")


def load_models_data(model_names, thresholds, cache_dir="cache/models_data"):
    """Load models_data from cache files if available"""
    models_data = {}

    for model_name, threshold in zip(model_names, thresholds):
        cache_key = get_cache_key(model_name, threshold)
        cache_file = f"{cache_dir}/{model_name}_{cache_key}.pkl"

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                models_data[model_name] = data
                print(f"Loaded cached data for {model_name} from {cache_file}")
            except Exception as e:
                print(f"Warning: Could not load cache for {model_name}: {e}")
        else:
            print(f"No cache found for {model_name} (looking for {cache_file})")

    return models_data


def clear_cache(cache_dir="cache/models_data"):
    """Clear all cached model data"""
    if os.path.exists(cache_dir):
        import shutil

        shutil.rmtree(cache_dir)
        print(f"Cleared cache directory: {cache_dir}")
    else:
        print(f"Cache directory does not exist: {cache_dir}")


def safe_write_image(fig, filename):
    """Safely write image with error handling"""
    try:
        fig.savefig(filename, dpi=400, bbox_inches="tight")
        print(f"Successfully saved: {filename}")
        plt.close(fig)  # Close the figure to free memory
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        print("Trying alternative formats...")
        try:
            # Try PNG as fallback
            png_filename = filename.replace(".pdf", ".png")
            fig.savefig(png_filename, dpi=400, bbox_inches="tight")
            print(f"Saved as PNG instead: {png_filename}")
            plt.close(fig)
        except Exception as e2:
            print(f"Could not save in any format: {e2}")
            plt.close(fig)


def make_graph_hm(z, hovertext, task_names, title, model):
    # Handle the case where we have a full matrix (not the original slicing)
    if z.shape[0] == len(task_names) and z.shape[1] == len(task_names):
        # Use full matrix
        heat_z = z
        x_labels = task_names
        y_labels = task_names
    else:
        # Use original slicing
        heat_z = z[1:, :-1]
        x_labels = task_names[:-1]
        y_labels = task_names[1:]

    # Create figure and axis
    plt.figure(figsize=(10, 8))

    # Get the appropriate colormap for this model
    model_cmap = get_model_colormap(model)

    # Create seaborn heatmap
    ax = sns.heatmap(
        heat_z,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=model_cmap,
        annot_kws={"fontsize": 15},
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "IoU"},
        square=True,
    )
    model_display_name = model_display_name_dict.get(model, model)
    plt.title(f"{model_display_name} - {title}", fontsize=15)
    plt.xlabel("PII Task", fontsize=15)
    plt.xticks(rotation=0, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    return plt.gcf()


def graph_analysis(g1, g2):
    edges_g1 = {edge.name for edge in g1.edges.values() if edge.in_graph}
    edges_g2 = {edge.name for edge in g2.edges.values() if edge.in_graph}
    edge_intersection = edges_g1 & edges_g2
    edge_union = edges_g1 | edges_g2

    x = len(edge_intersection)
    M = len(g1.edges)
    n = len(edges_g1)
    N = len(edges_g2)
    iou_edge = len(edge_intersection) / len(edge_union) if len(edge_union) > 0 else 0
    p_edge = 1 - hypergeom.cdf(x, M, n, N) if N > 0 and n > 0 else 1

    nodes_g1 = {node.name for node in g1.nodes.values() if node.in_graph} - {
        "inputs",
        "logits",
    }
    nodes_g2 = {node.name for node in g2.nodes.values() if node.in_graph} - {
        "inputs",
        "logits",
    }
    node_intersection = nodes_g1 & nodes_g2
    node_union = nodes_g1 | nodes_g2

    x = len(node_intersection)
    M = len(g1.nodes) - 2
    n = len(nodes_g1)
    N = len(nodes_g2)
    p_node = 1 - hypergeom.cdf(x, M, n, N) if N > 0 and n > 0 else 1
    iou_node = len(node_intersection) / len(node_union) if len(node_union) > 0 else 0

    # directional measures:
    edge_overlap = len(edge_intersection) / len(edges_g1) if len(edges_g1) > 0 else 0
    node_overlap = len(node_intersection) / len(nodes_g1) if len(nodes_g1) > 0 else 0

    return p_edge, iou_edge, p_node, iou_node, edge_overlap, node_overlap


def make_task_comparison_heatmap(model_name, edge_threshold, title):
    """
    Compare PII tasks within a single model after applying the same edge threshold to all

    Args:
        model_name: Name of the model to analyze
        edge_threshold: Single threshold to apply to all task graphs
        title: Title for output files
    """

    def task_to_path(model_name: str, task: str):
        return f"./graphs/{model_name}/{task}_kl.json"

    # Load graphs for all PII tasks for this model
    graphs = [Graph.from_json(task_to_path(model_name, task)) for task in ft_pii_tasks]

    # Apply the same threshold to all graphs
    for graph in graphs:
        graph.apply_greedy(edge_threshold, absolute=True)

    n_tasks = len(ft_pii_tasks)
    pes = np.zeros((n_tasks, n_tasks))
    ies = np.zeros((n_tasks, n_tasks))
    pns = np.zeros((n_tasks, n_tasks))
    ins = np.zeros((n_tasks, n_tasks))
    eos = np.zeros((n_tasks, n_tasks))
    nos = np.zeros((n_tasks, n_tasks))

    # Create display names for tasks
    task_display_names = [display_name_dict.get(task, task) for task in ft_pii_tasks]

    for i, g1 in enumerate(graphs):
        for j, g2 in enumerate(graphs):
            p_edge, iou_edge, p_node, iou_node, edge_overlap, node_overlap = (
                graph_analysis(g1, g2)
            )
            pes[i, j] = p_edge
            ies[i, j] = iou_edge
            pns[i, j] = p_node
            ins[i, j] = iou_node
            eos[i, j] = edge_overlap
            nos[i, j] = node_overlap

    results_root = "results/task_overlap/csv"
    os.makedirs(results_root, exist_ok=True)

    plots_root = "plots/task_overlap"
    os.makedirs(plots_root, exist_ok=True)

    # Save edge results
    edge_iou_df = pd.DataFrame(ies, columns=ft_pii_tasks, index=ft_pii_tasks)
    edge_iou_p_df = pd.DataFrame(pes, columns=ft_pii_tasks, index=ft_pii_tasks)
    edge_iou_df.to_csv(f"{results_root}/{model_name}_edge_ious.csv")
    edge_iou_p_df.to_csv(f"{results_root}/{model_name}_edge_iou_ps.csv")

    # Save node results
    node_iou_df = pd.DataFrame(ins, columns=ft_pii_tasks, index=ft_pii_tasks)
    node_iou_p_df = pd.DataFrame(pns, columns=ft_pii_tasks, index=ft_pii_tasks)
    node_iou_df.to_csv(f"{results_root}/{model_name}_node_ious.csv")
    node_iou_p_df.to_csv(f"{results_root}/{model_name}_node_iou_ps.csv")

    # Save overlap results
    edge_overlap_df = pd.DataFrame(eos, columns=ft_pii_tasks, index=ft_pii_tasks)
    node_overlap_df = pd.DataFrame(nos, columns=ft_pii_tasks, index=ft_pii_tasks)
    edge_overlap_df.to_csv(f"{results_root}/{model_name}_edge_overlap.csv")
    node_overlap_df.to_csv(f"{results_root}/{model_name}_node_overlap.csv")

    # Create heatmaps (mask upper triangle for symmetry)
    ius = np.triu_indices(n_tasks, k=1)  # k=1 to keep diagonal

    # Edge IoU heatmap
    ies_plot = ies.copy()
    ies_plot[ius] = np.nan
    fig = make_graph_hm(
        ies_plot,
        pes,
        task_display_names,
        f"PII Task Edge IoU ({edge_threshold} edges)",
        model_name,
    )
    safe_write_image(fig, f"{plots_root}/{model_name}_{title}_edges.pdf")

    # Node IoU heatmap
    ins_plot = ins.copy()
    ins_plot[ius] = np.nan
    fig = make_graph_hm(
        ins_plot,
        pns,
        task_display_names,
        f"PII Task Node IoU ({edge_threshold} edges)",
        model_name,
    )
    safe_write_image(fig, f"{plots_root}/{model_name}_{title}_nodes.pdf")

    # Overlap heatmaps (these are directional, so don't mask)
    fig = make_graph_hm(
        eos,
        pes,
        task_display_names,
        f"PII Task Edge Overlap ({edge_threshold} edges)",
        model_name,
    )
    safe_write_image(fig, f"{plots_root}/{model_name}_{title}_edge_overlap.pdf")

    fig = make_graph_hm(
        nos,
        ins,
        task_display_names,
        f"PII Task Node Overlap ({edge_threshold} edges)",
        model_name,
    )
    safe_write_image(fig, f"{plots_root}/{model_name}_{title}_node_overlap.pdf")

    return {
        "edge_iou": edge_iou_df,
        "node_iou": node_iou_df,
        "edge_overlap": edge_overlap_df,
        "node_overlap": node_overlap_df,
    }


def create_mega_heatmap(model_name, edge_threshold, title):
    """
    Create a 2x2 subplot showing all four overlap metrics for a single model

    Args:
        model_name: Name of the model to analyze
        edge_threshold: Single threshold to apply to all task graphs
        title: Title for output files
    """

    # Try to load cached data first
    models_data = load_models_data([model_name], [edge_threshold])

    if model_name in models_data:
        print(f"Using cached data for {model_name}")
        data = models_data[model_name]
        # Remove NaN mask to return full matrices
        n_tasks = len(ft_pii_tasks)
        ies = data["edge_iou"].copy()
        ins = data["node_iou"].copy()

        # Fill NaN values in upper triangle with symmetric values
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                if np.isnan(ies[i, j]):
                    ies[i, j] = ies[j, i]
                if np.isnan(ins[i, j]):
                    ins[i, j] = ins[j, i]

        task_display_names = [
            display_name_dict.get(task, task) for task in ft_pii_tasks
        ]
        return ies, ins, task_display_names

    def task_to_path(model_name: str, task: str):
        return f"./graphs/{model_name}/{task}_kl.json"

    print(f"Processing {model_name} with threshold {edge_threshold}...")

    # Load graphs for all PII tasks for this model
    graphs = [Graph.from_json(task_to_path(model_name, task)) for task in ft_pii_tasks]

    # Apply the same threshold to all graphs
    for graph in graphs:
        graph.apply_greedy(edge_threshold, absolute=True)

    n_tasks = len(ft_pii_tasks)
    pes = np.zeros((n_tasks, n_tasks))
    ies = np.zeros((n_tasks, n_tasks))
    pns = np.zeros((n_tasks, n_tasks))
    ins = np.zeros((n_tasks, n_tasks))
    eos = np.zeros((n_tasks, n_tasks))
    nos = np.zeros((n_tasks, n_tasks))

    # Create display names for tasks
    task_display_names = [display_name_dict.get(task, task) for task in ft_pii_tasks]

    for i, g1 in enumerate(graphs):
        for j, g2 in enumerate(graphs):
            p_edge, iou_edge, p_node, iou_node, edge_overlap, node_overlap = (
                graph_analysis(g1, g2)
            )
            pes[i, j] = p_edge
            ies[i, j] = iou_edge
            pns[i, j] = p_node
            ins[i, j] = iou_node
            eos[i, j] = edge_overlap
            nos[i, j] = node_overlap

    # Cache the results
    ius = np.triu_indices(n_tasks, k=1)
    ies_plot = ies.copy()
    ins_plot = ins.copy()
    ies_plot[ius] = np.nan
    ins_plot[ius] = np.nan

    cache_data = {
        model_name: {
            "edge_iou": ies_plot,
            "node_iou": ins_plot,
            "threshold": edge_threshold,
        }
    }
    save_models_data(cache_data)

    return ies, ins, task_display_names


def create_model_comparison_plots(model_names, edge_thresholds, title):
    """
    Create two mega plots: one for Edge IoU and one for Node IoU
    Each plot shows all models as subplots for comparison

    Args:
        model_names: List of model names to analyze
        edge_threshold: Single threshold to apply to all task graphs
        title: Title for output files
    """

    # Collect data for all models
    all_edge_ious = {}
    all_node_ious = {}
    task_display_names = [display_name_dict.get(task, task) for task in ft_pii_tasks]

    for i, model_name in enumerate(model_names):
        try:
            edge_iou, node_iou, task_names = create_mega_heatmap(
                model_name, edge_thresholds[i], title
            )
            all_edge_ious[model_name] = edge_iou
            all_node_ious[model_name] = node_iou
            if task_display_names is None:
                task_display_names = task_names
        except FileNotFoundError:
            print(f"Skipping {model_name}: Graph files not found")
            continue

    n_models = len(all_edge_ious)
    if n_models == 0:
        print("No valid models found for comparison")
        return

    # Create Edge IoU comparison plot

    fig1, axes1 = plt.subplots(1, 6, figsize=(20, 5))
    axes1 = axes1 if hasattr(axes1, "__len__") else [axes1]

    for idx, (model_name, edge_iou) in enumerate(all_edge_ious.items()):
        # Mask upper triangle for symmetry
        ius = np.triu_indices(len(ft_pii_tasks), k=1)
        edge_iou_plot = edge_iou.copy()
        edge_iou_plot[ius] = np.nan

        # Get the appropriate colormap for this model
        model_cmap = get_model_colormap(model_name)

        ax = axes1[idx] if n_models > 1 else axes1[0]
        sns.heatmap(
            edge_iou_plot,
            xticklabels=task_display_names,
            yticklabels=task_display_names,
            cmap=model_cmap,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 16},
            square=True,
            ax=ax,
            linewidths=0.2,
            cbar=False,  # Disable individual colorbars
        )
        ax.set_xlabel("")
        ax.set_xticklabels([])
        if idx != 0:
            ax.set_ylabel("")  # Remove y-labels for non-left plots
            ax.set_yticklabels([])  # Remove y-tick labels for non-left plots
        else:
            ax.tick_params(
                axis="y", rotation=0, labelsize=16
            )  # Only show y-ticks on first plot
        model_display_name = model_display_name_dict.get(model_name, model_name)
        ax.set_title(model_display_name, fontsize=16)
        ax.tick_params(axis="x", rotation=0, labelsize=15)

    # Add shared colorbar for Edge IoU plot
    fig1.subplots_adjust(right=0.9)
    # cbar_ax1 = fig1.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # cbar1 = fig1.colorbar(axes1[0].collections[0], cax=cbar_ax1)
    # cbar1.set_label("IoU Score", rotation=270, labelpad=15)

    # Create Node IoU comparison plot
    fig2, axes2 = plt.subplots(1, 6, figsize=(20, 5))
    axes2 = axes2 if hasattr(axes2, "__len__") else [axes2]

    for idx, (model_name, node_iou) in enumerate(all_node_ious.items()):
        # Mask upper triangle for symmetry
        ius = np.triu_indices(len(ft_pii_tasks), k=1)
        node_iou_plot = node_iou.copy()
        node_iou_plot[ius] = np.nan

        # Get the appropriate colormap for this model
        model_cmap = get_model_colormap(model_name)

        ax = axes2[idx] if n_models > 1 else axes2[0]
        sns.heatmap(
            node_iou_plot,
            xticklabels=task_display_names,
            yticklabels=task_display_names,
            cmap=model_cmap,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 16},
            square=True,
            ax=ax,
            linewidths=0.2,
            cbar=False,  # Disable individual colorbars
        )
        if idx != 0:
            ax.set_ylabel("")  # Remove y-labels for non-left plots
            ax.set_yticklabels([])  # Remove y-tick labels for non-left plots
        else:
            ax.tick_params(
                axis="y", rotation=0, labelsize=16
            )  # Only show y-ticks on first plot
        model_display_name = model_display_name_dict.get(model_name, model_name)
        ax.set_title(model_display_name, fontsize=16)
        ax.tick_params(axis="x", rotation=0, labelsize=16)

    # Add shared colorbar for Node IoU plot
    fig2.subplots_adjust(right=0.9)
    # cbar_ax2 = fig2.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # cbar2 = fig2.colorbar(axes2[0].collections[0], cax=cbar_ax2)
    # cbar2.set_label("IoU Score", rotation=270, labelpad=15)

    # Save the plots
    plots_root = "plots/model_comparison"
    os.makedirs(plots_root, exist_ok=True)

    safe_write_image(fig1, f"{plots_root}/{title}_edge_iou_comparison.pdf")
    safe_write_image(fig2, f"{plots_root}/{title}_node_iou_comparison.pdf")

    # Create a combined figure with both Edge IoU and Node IoU plots
    combined_fig = plt.figure(figsize=(20, 6))
    fontsize = 16
    # Create Edge IoU plots in the top row
    for idx, (model_name, edge_iou) in enumerate(all_edge_ious.items()):
        ax = combined_fig.add_subplot(2, 6, idx + 1)  # Top row (positions 1-4)

        # Mask upper triangle for symmetry
        ius = np.triu_indices(len(ft_pii_tasks), k=1)
        edge_iou_plot = edge_iou.copy()
        edge_iou_plot[ius] = np.nan

        # Get the appropriate colormap for this model
        model_cmap = get_model_colormap(model_name)

        sns.heatmap(
            edge_iou_plot,
            xticklabels=task_display_names,
            yticklabels=task_display_names,
            cmap=model_cmap,
            annot=True,
            fmt=".0%",
            annot_kws={"fontsize": fontsize},
            square=True,
            ax=ax,
            linewidths=0.2,
            cbar=False,
        )
        model_display_name = model_display_name_dict.get(model_name, model_name)
        if idx != 0:
            # ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.set_title(f"{model_display_name.replace(' Baseline', '')}", fontsize=fontsize, weight="heavy")
        else:
            ax.set_ylabel(f"Edges IoU", fontsize=14, weight="bold")
            ax.tick_params(axis="y", rotation=0, labelsize=fontsize)
            ax.set_title(
                f"{model_display_name.replace(' Baseline', '')}",
                fontsize=fontsize,
                # loc="right",
                weight="heavy",
            )
        # Remove x-ticks and labels to eliminate whitespace
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", length=0, pad=0)
        ax.tick_params(axis="y", which="both", length=0, pad=0)

    # Create Node IoU plots in the bottom row
    for idx, (model_name, node_iou) in enumerate(all_node_ious.items()):
        ax = combined_fig.add_subplot(2, 6, idx + 7)  # Bottom row (positions 5-8)

        # Mask upper triangle for symmetry
        ius = np.triu_indices(len(ft_pii_tasks), k=1)
        node_iou_plot = node_iou.copy()
        node_iou_plot[ius] = np.nan

        # Get the appropriate colormap for this model
        model_cmap = get_model_colormap(model_name)

        sns.heatmap(
            node_iou_plot,
            xticklabels=task_display_names,
            yticklabels=task_display_names,
            cmap=model_cmap,
            annot=True,
            fmt=".0%",
            annot_kws={"fontsize": fontsize},
            square=True,
            ax=ax,
            linewidths=0.2,
            cbar=False,
        )

        model_display_name = model_display_name_dict.get(model_name, model_name)
        if idx != 0:
            ax.set_yticklabels([])
            ax.set_title(f"{model_display_name.replace(' Baseline', '')}", fontsize=fontsize, weight="heavy")
        else:
            ax.tick_params(axis="y", rotation=0, labelsize=fontsize)
            ax.set_title(
                f"{model_display_name.replace(' Baseline', '')}",
                fontsize=fontsize,
                # loc="right",
                weight="heavy",
            )
            ax.set_ylabel(f"Nodes IoU", fontsize=14, weight="bold")
        ax.tick_params(axis="x", rotation=25, labelsize=15)
        ax.tick_params(axis="y", which="both", length=0, pad=0)

    safe_write_image(combined_fig, f"{plots_root}/{title}_combined_iou_comparison.pdf")

    return fig1, fig2


if __name__ == "__main__":
    # Define your model names based on available graphs
    model_names = [
        # "gpt2-small-dp1",
        # "gpt2-small-dp2",
        # "gpt2-small-dp4",
        # "gpt2-small-dp8",
        # "gpt2-medium-dp2",
        # "gpt2-medium-dp4",
        # "gpt2-medium-dp8",
        "gpt2-small-baseline",
        "gpt2-medium-baseline",
        "qwen3-06-baseline",
        "gpt2-large-baseline",
        "llama3-1b-baseline",
        "qwen3-17-baseline",
        # "gpt2-large-dp2",
        # "gpt2-large-dp4",
        # "gpt2-large-dp8",
        # "pythia-160m-baseline",
        # "pythia-160m-dp2",
        # "pythia-160m-dp4",
        # "pythia-160m-dp8",
        # "pythia-410m-baseline",
        # "pythia-410m-dp2",
        # "pythia-410m-dp4",
        # "pythia-410m-dp8",
        # "pythia-1b-baseline",
        # "pythia-1b-dp1",
        # "pythia-1b-dp2",
        # "pythia-1b-dp4",
        # "pythia-1b-dp8"
        # "llama3-1b-dp1",
        # "llama3-1b-dp2",
        # "llama3-1b-dp4",
        # "llama3-1b-dp8"
    ]

    edge_threshold = [1000, 4000, 2000, 8000, 8000, 10000]

    # # # Run analysis for each model
    # for i, model_name in enumerate(model_names):  # Just do the first model for testing
    #     print(f"Analyzing PII task overlap for {model_name}...")

    #     try:
    #         # Run the analysis for this model
    #         results = make_task_comparison_heatmap(
    #             model_name, edge_threshold[i], "pii_task_comparison"
    #         )

    #         print(
    #             f"Analysis complete for {model_name}! Results saved with {edge_threshold} edges."
    #         )
    #         print("Edge IoU matrix:")
    #         print(results["edge_iou"])
    #         print("\nNode IoU matrix:")
    #         print(results["node_iou"])
    #         print("-" * 50)

    #     except FileNotFoundError as e:
    #         print(f"Skipping {model_name}: Graph files not found ({e})")
    #         continue
    #     except Exception as e:
    #         print(f"Error processing {model_name}: {e}")
    #         continue

    # Create the mega comparison plots
    print("\nCreating model comparison plots...")
    edge_fig, node_fig = create_model_comparison_plots(
        model_names, edge_threshold, "pii_task_comparison"
    )
    print("Model comparison plots created successfully!")
    # Create two-model side-by-side comparison
    # print("\nCreating two-model side-by-side comparison...")
    # try:
    #     # Example: Compare GPT2-small-baseline and GPT2-large-baseline
    #     model1 = "gpt2-small-baseline"
    #     model2 = "gpt2-large-baseline"
    #     threshold1 = 1000  # threshold for model1
    #     threshold2 = 10000  # threshold for model2

    #     side_by_side_fig = create_two_model_side_by_side_plot(
    #         model1, model2, threshold1, threshold2, "pii_task_comparison"
    #     )
    #     if side_by_side_fig:
    #         print(f"Two-model side-by-side comparison created for {model1} vs {model2}!")
    #     else:
    #         print("Failed to create two-model side-by-side comparison")
    # except Exception as e:
    #     print(f"Error creating two-model side-by-side comparison: {e}")
