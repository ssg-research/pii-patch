import os
from xml.parsers.expat import model
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import hypergeom
from gencircuits.eap.graph import Graph
from constants import (
    ft_pii_tasks,
    display_name_dict,
    model_display_name_dict,
    gpt2_small_models,
    gpt2_medium_models,
    qwen3_06_models,
    gpt2_large_models,
    llama3_1b_models,
    qwen3_17_models,
    MODEL_COLORS,
    MODEL_ORDER,
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

    if base_name in MODEL_ORDER:
        idx = MODEL_ORDER.index(base_name)
        return MODEL_COLORMAPS[idx]
    else:
        # Default to first colormap if model not found
        return MODEL_COLORMAPS[0]


def safe_write_image(fig, filename):
    """Safely write image with error handling"""
    try:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Successfully saved: {filename}")
        plt.close(fig)  # Close the figure to free memory
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        print("Trying alternative formats...")
        try:
            # Try PNG as fallback
            png_filename = filename.replace(".pdf", ".png")
            fig.savefig(png_filename, dpi=300, bbox_inches="tight")
            print(f"Saved as PNG instead: {png_filename}")
            plt.close(fig)
        except Exception as e2:
            print(f"Could not save in any format: {e2}")
            plt.close(fig)


def make_graph_hm(z, hovertext, graphs_names, title, task, d_axis=False):
    # Handle the case where we have a full matrix (not the original slicing)
    if z.shape[0] == len(graphs_names) and z.shape[1] == len(graphs_names):
        # Use full matrix
        heat_z = z
        x_labels = graphs_names
        y_labels = graphs_names
    else:
        # Use original slicing
        heat_z = z[1:, :-1]
        x_labels = graphs_names[:-1]
        y_labels = graphs_names[1:]

    # Create figure and axis
    plt.figure(figsize=(10, 8))

    # Create seaborn heatmap
    ax = sns.heatmap(
        heat_z,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=get_model_colormap(graphs_names[0]),
        annot=True,
        annot_kws={"fontsize": 14},
        fmt=".3f",
        cbar_kws={"label": "IoU"},
        square=True,
    )
    display_name = display_name_dict[task]
    plt.title(f"{display_name} - {title}", fontsize=14, pad=20)
    if d_axis == True:
        plt.xlabel("Model", fontsize=14)
    plt.xticks(rotation=15, ha="left")
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


def load_cached_results(graphs_names, edge_threshold, title, task, model_arch=None):
    """
    Load previously computed IoU matrices from CSV files if they exist

    Args:
        graphs_names: List of model names
        edge_threshold: Edge threshold used
        title: Title for the analysis
        task: Task name
        model_arch: Model architecture (e.g., 'gpt2-small', 'gpt2-medium')

    Returns:
        Dictionary with cached results or None if cache doesn't exist
    """
    results_root = "results/jaccard/csv"

    # Create architecture-specific subdirectory if specified
    if model_arch:
        results_root = os.path.join(results_root, model_arch)

    # Define the expected file paths
    files_to_check = {
        "edge_iou": f"{results_root}/{task}_edge_ious.csv",
        "edge_iou_p": f"{results_root}/{task}_edge_iou_ps.csv",
        "node_iou": f"{results_root}/{task}_node_ious.csv",
        "node_iou_p": f"{results_root}/{task}_node_iou_ps.csv",
        "edge_overlap": f"{results_root}/{task}_edge_overlap.csv",
        "node_overlap": f"{results_root}/{task}_node_overlap.csv",
    }

    # Check if all files exist
    if not all(os.path.exists(filepath) for filepath in files_to_check.values()):
        missing_files = [
            filepath
            for filepath in files_to_check.values()
            if not os.path.exists(filepath)
        ]
        print(f"  Cache files not found for {task} (arch: {model_arch})")
        print(f"  Missing files: {missing_files}")
        print(f"  Will compute from scratch...")
        return None

    try:
        # Load all matrices
        cached_results = {}
        for key, filepath in files_to_check.items():
            df = pd.read_csv(filepath, index_col=0)

            # Verify the matrix has the correct models
            # if not (
            #     set(df.columns) == set(graphs_names)
            #     and set(df.index) == set(graphs_names)
            # ):
            #     print(
            #         f"  Cached results for {task} have different models, will recompute..."
            #     )
            #     return None

            # Reorder to match current graphs_names order
            df = df.reindex(index=graphs_names, columns=graphs_names)
            cached_results[key] = df

        print(f"  Successfully loaded cached results for {task} (arch: {model_arch})")

        # Return in the expected format
        return {
            "edge_iou": cached_results["edge_iou"],
            "node_iou": cached_results["node_iou"],
            "edge_overlap": cached_results["edge_overlap"],
            "node_overlap": cached_results["node_overlap"],
        }

    except Exception as e:
        print(f"  Error loading cached results for {task}: {e}")
        print(f"  Will compute from scratch...")
        return None


def make_comparison_heatmap(graphs_names, edge_threshold, title, task, model_arch=None):
    """
    Compare graphs after applying the same edge threshold to all

    Args:
        graphs: List of Graph objects
        graphs_names: List of model names corresponding to graphs
        edge_threshold: Single threshold to apply to all graphs
        title: Title for output files
        task: Task name
        model_arch: Model architecture for organized file saving
    """

    # First, try to load cached results
    cached_results = load_cached_results(
        graphs_names, edge_threshold, title, task, model_arch
    )
    if cached_results is not None:
        print(f"  Using cached results for {task}")
        return cached_results

    print(f"  Computing IoU matrices for {task} from scratch...")

    graphs = []
    print(f"Loading graphs for {task}...with threshold {edge_threshold}")
    for model in graphs_names:
        try:
            graph = Graph.from_json(task_to_path(model, task))
            graphs.append(graph)
            print(f"  Successfully loaded {model}")
        except Exception as e:
            print(f"  Error loading graph for {model}: {e}: {task}")
            print(f"  Skipping {model}")
            continue

    if len(graphs) == 0:
        print(f"  No graphs loaded successfully for {task}, skipping...")
        return
    elif len(graphs) < len(graphs_names):
        print(
            f"  Warning: Only {len(graphs)} out of {len(graphs_names)} graphs loaded successfully"
        )
        # Update model_names to match successfully loaded graphs
        model_names = [
            model
            for model in model_names
            if any(task_to_path(model, task) in str(g) for g in graphs)
        ]

    # Apply the same threshold to all graphs
    for graph in graphs:
        graph.apply_greedy(edge_threshold, absolute=True)

    n_graphs = len(graphs)
    pes = np.zeros((n_graphs, n_graphs))
    ies = np.zeros((n_graphs, n_graphs))
    pns = np.zeros((n_graphs, n_graphs))
    ins = np.zeros((n_graphs, n_graphs))
    eos = np.zeros((n_graphs, n_graphs))
    nos = np.zeros((n_graphs, n_graphs))

    # Create display names for plots
    display_names = [model_display_name_dict.get(name, name) for name in graphs_names]

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

    results_root = "results/jaccard/csv"
    # Create architecture-specific subdirectory if specified
    if model_arch:
        results_root = os.path.join(results_root, model_arch)
    os.makedirs(results_root, exist_ok=True)

    plots_root = "plots/jaccard"
    os.makedirs(plots_root, exist_ok=True)

    # Save edge results
    edge_iou_df = pd.DataFrame(ies, columns=graphs_names, index=graphs_names)
    edge_iou_p_df = pd.DataFrame(pes, columns=graphs_names, index=graphs_names)
    edge_iou_df.to_csv(f"{results_root}/{task}_edge_ious.csv")
    edge_iou_p_df.to_csv(f"{results_root}/{task}_edge_iou_ps.csv")

    # Save node results
    node_iou_df = pd.DataFrame(ins, columns=graphs_names, index=graphs_names)
    node_iou_p_df = pd.DataFrame(pns, columns=graphs_names, index=graphs_names)
    node_iou_df.to_csv(f"{results_root}/{task}_node_ious.csv")
    node_iou_p_df.to_csv(f"{results_root}/{task}_node_iou_ps.csv")

    # Save overlap results
    edge_overlap_df = pd.DataFrame(eos, columns=graphs_names, index=graphs_names)
    node_overlap_df = pd.DataFrame(nos, columns=graphs_names, index=graphs_names)
    edge_overlap_df.to_csv(f"{results_root}/{task}_edge_overlap.csv")
    node_overlap_df.to_csv(f"{results_root}/{task}_node_overlap.csv")

    print(f"  Saved IoU matrices for {task} (arch: {model_arch}) to {results_root}")

    # Create heatmaps (mask upper triangle for symmetry)
    ius = np.triu_indices(n_graphs, k=1)  # k=1 to keep diagonal

    # Edge IoU heatmap
    ies_plot = ies.copy()
    ies_plot[ius] = np.nan
    fig = make_graph_hm(
        ies_plot, pes, display_names, f"Edge IoU ({edge_threshold} edges)", task
    )
    # safe_write_image(fig, f"{plots_root}/{task}_{title}_edges.pdf")

    # # Edge p-value heatmap
    # pes_plot = pes.copy()
    # pes_plot[ius] = np.nan
    # fig = make_graph_hm(
    #     pes_plot, ies, display_names, f"Edge p-value ({edge_threshold} edges)"
    # )
    # safe_write_image(fig, f"{plots_root}/{task}_{title}_edges_p.pdf", task)

    # Node IoU heatmap
    ins_plot = ins.copy()
    ins_plot[ius] = np.nan
    fig = make_graph_hm(
        ins_plot, pns, display_names, f"Node IoU ({edge_threshold} edges)", task, True
    )
    # safe_write_image(fig, f"{plots_root}/{task}_{title}_nodes.pdf")

    # # Node p-value heatmap
    # pns_plot = pns.copy()
    # pns_plot[ius] = np.nan
    # fig = make_graph_hm(
    #     pns_plot, ins, display_names, f"Node p-value ({edge_threshold} edges)"
    # )
    # safe_write_image(fig, f"{plots_root}/{task}_{title}_nodes_p.pdf")

    # Overlap heatmaps (these are directional, so don't mask)
    fig = make_graph_hm(
        eos, pes, display_names, f"Edge Overlap ({edge_threshold} edges)", task
    )
    # safe_write_image(fig, f"{plots_root}/{task}_{title}_edge_overlap.pdf")

    fig = make_graph_hm(
        nos, ins, display_names, f"Node Overlap ({edge_threshold} edges)", task
    )
    # safe_write_image(fig, f"{plots_root}/{task}_{title}_node_overlap.pdf")

    return {
        "edge_iou": edge_iou_df,
        "node_iou": node_iou_df,
        "edge_overlap": edge_overlap_df,
        "node_overlap": node_overlap_df,
    }


def create_combined_pii_plots(all_results, model_names, edge_thresholds, title):
    """
    Create combined plots showing all PII tasks side by side

    Args:
        all_results: Dictionary with task names as keys and results as values
        model_names: List of model names
        edge_thresholds: List of edge thresholds used for each task
        title: Title for output files
    """
    plots_root = "plots/jaccard"
    os.makedirs(plots_root, exist_ok=True)

    display_names = [model_display_name_dict.get(name, name) for name in model_names]
    tasks = list(all_results.keys())
    n_tasks = len(tasks)
    n_models = len(model_names)

    # Create 4 combined plots: edge_iou, node_iou, edge_overlap, node_overlap
    metric_types = ["edge_iou", "node_iou", "edge_overlap", "node_overlap"]
    metric_titles = ["Edge IoU", "Node IoU", "Edge Overlap", "Node Overlap"]

    for metric_type, metric_title in zip(metric_types, metric_titles):
        fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 5))
        if n_tasks == 1:
            axes = [axes]

        for i, task in enumerate(tasks):
            data = all_results[task][metric_type].values
            if model == "qwen3-17-baseline":
                edge_threshold = 10000
            else:
                edge_threshold = edge_thresholds[i]

            # For IoU metrics, mask upper triangle for symmetry
            if "iou" in metric_type:
                data_plot = data.copy()
                ius = np.triu_indices(n_models, k=1)
                data_plot[ius] = np.nan
            else:
                data_plot = data

            # Create heatmap - only show colorbar on the last subplot
            show_cbar = i == n_tasks - 1
            sns.heatmap(
                data_plot,
                xticklabels=display_names,
                yticklabels=display_names,
                cmap="Blues",
                annot=True,
                fmt=".3f",
                annot_kws={"fontsize": 8},  # Smaller annotation font size
                cbar=show_cbar,
                cbar_kws={"label": metric_title} if show_cbar else {},
                square=True,
                ax=axes[i],
            )

            task_display_name = display_name_dict[task]
            axes[i].set_title(
                f"{task_display_name}\n({edge_threshold} edges)", fontsize=12, pad=10
            )
            axes[i].set_xlabel("Model", fontsize=10)
            axes[i].set_ylabel("Model", fontsize=10)
            axes[i].tick_params(
                axis="x", rotation=30, labelsize=8
            )  # Flatter rotation (30 degrees instead of 45)
            axes[i].tick_params(axis="y", rotation=0, labelsize=8)

        plt.suptitle(f"PII Circuit Comparison - {metric_title}", fontsize=16, y=1.02)
        plt.tight_layout()

        # Save the combined plot
        filename = f"{plots_root}/combined_pii_{metric_type}_{title}.pdf"
        safe_write_image(fig, filename)


def create_averaged_pii_plots(all_results, model_names, title, model_arch=None):
    """
    Create averaged plots across all PII tasks

    Args:
        all_results: Dictionary with task names as keys and results as values
        model_names: List of model names
        title: Title for output files
        model_arch: Model architecture for organized file saving
    """
    plots_root = "plots/jaccard"
    os.makedirs(plots_root, exist_ok=True)

    display_names = [model_display_name_dict.get(name, name) for name in model_names]
    tasks = list(all_results.keys())
    n_models = len(model_names)

    # Create averaged data for each metric type
    metric_types = ["edge_iou", "node_iou", "edge_overlap", "node_overlap"]
    metric_titles = ["Edge IoU", "Node IoU", "Edge Overlap", "Node Overlap"]

    for metric_type, metric_title in zip(metric_types, metric_titles):
        # Calculate average across all tasks
        averaged_data = np.zeros((n_models, n_models))

        for task in tasks:
            data = all_results[task][metric_type].values
            averaged_data += data

        averaged_data /= len(tasks)  # Average across tasks

        # For IoU metrics, mask upper triangle for symmetry
        if "iou" in metric_type:
            data_plot = averaged_data.copy()
            ius = np.triu_indices(n_models, k=1)
            data_plot[ius] = np.nan
        else:
            data_plot = averaged_data

        # Create figure
        plt.figure(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            data_plot,
            xticklabels=display_names,
            yticklabels=display_names,
            cmap="Blues",
            annot=True,
            fmt=".3f",
            annot_kws={"fontsize": 10},
            cbar_kws={"label": metric_title},
            square=True,
        )

        plt.title(
            f"Average PII Circuit Comparison - {metric_title}", fontsize=14, pad=20
        )
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Model", fontsize=14)
        plt.xticks(rotation=15, ha="left")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save the averaged plot
        filename = f"{plots_root}/averaged_pii_{metric_type}_{title}.pdf"
        safe_write_image(plt.gcf(), filename)

        # Also save the averaged data as CSV with architecture-specific directory
        results_root = "results/jaccard/csv"
        if model_arch:
            results_root = os.path.join(results_root, model_arch)
        os.makedirs(results_root, exist_ok=True)
        averaged_df = pd.DataFrame(
            averaged_data, columns=model_names, index=model_names
        )
        averaged_df.to_csv(f"{results_root}/averaged_pii_{metric_type}.csv")
        print(
            f"  Saved averaged {metric_type} data for {model_arch} to {results_root}/averaged_pii_{metric_type}.csv"
        )


def create_combined_architecture_plots(all_arch_results, title_suffix=""):
    """
    Create combined plots showing all model architectures together

    Args:
        all_arch_results: Dictionary with architecture names as keys and averaged results as values
        title_suffix: Optional suffix for the title and filename
    """
    plots_root = "plots/jaccard"
    os.makedirs(plots_root, exist_ok=True)

    architectures = list(all_arch_results.keys())
    n_archs = len(architectures)

    # Create combined plots for both metrics in a single figure
    metric_types = [
        "edge_iou",
        "node_iou",
    ]
    metric_titles = [
        "Edges",
        "Nodes",
    ]

    # Calculate grid dimensions: 2 rows (one for each metric), n_archs columns
    n_cols = n_archs
    n_rows = 2  # One row for edges, one for nodes

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    # Handle single architecture case
    if n_archs == 1:
        axes = axes.reshape(-1, 1)

    ds_model_names = {
        "gpt2": "GPT2-Small",
        "gpt2-medium": "GPT2-Medium",
        "qwen3-06": "Qwen3-0.6B",
        "gpt2-large": "GPT2-Large",
        "llama3-1b": "Llama-3.2-1B",
        "qwen3-17": "Qwen3-1.7B",
    }

    # Iterate through metrics (rows) and architectures (columns)
    for metric_idx, (metric_type, metric_title) in enumerate(
        zip(metric_types, metric_titles)
    ):
        for arch_idx, arch in enumerate(architectures):
            data = all_arch_results[arch][metric_type]
            model_names = data.columns.tolist()
            x_tick_names = [
                model_display_name_dict.get(name, name)
                .split(" ")[-1]
                .replace("(", "")
                .replace(")", "")
                .replace("llama3-1b-dp4", "ε=4")
                .replace("Baseline", "Base")
                .replace("Scrubbed", "Scrub")
                .replace("APNEAP", "AP")
                for name in model_names
            ]
            y_tick_names = [
                model_display_name_dict.get(name, name)
                .split(" ")[-1]
                .replace("(", "")
                .replace(")", "")
                .replace("llama3-1b-dp4", "ε=4")
                for name in model_names
            ]


            # For IoU metrics, mask upper triangle for symmetry
            if "iou" in metric_type:
                data_plot = data.values.copy()
                n_models = len(model_names)
                ius = np.triu_indices(n_models, k=1)
                data_plot[ius] = np.nan
            else:
                # For overlap metrics, we don't mask (they are directional)
                data_plot = data.values

            # Create heatmap without colorbar
            sns.heatmap(
                data_plot,
                xticklabels=x_tick_names,
                yticklabels=y_tick_names,
                cmap=get_model_colormap(model_names[0]),
                annot=True,
                fmt=".2f",
                annot_kws={"fontsize": 12},
                square=True,
                ax=axes[metric_idx, arch_idx],
                cbar=False,
            )

            # Only show y-axis labels on leftmost plots
            if arch_idx > 0:
                axes[metric_idx, arch_idx].set_ylabel("")
                axes[metric_idx, arch_idx].set_yticklabels([])

            # Only show x-axis labels on bottom row
            if metric_idx == 0:
                axes[metric_idx, arch_idx].set_xlabel("")
                axes[metric_idx, arch_idx].set_xticklabels([])

            axes[metric_idx, arch_idx].tick_params(axis="x", rotation=0, labelsize=12)
            axes[metric_idx, arch_idx].tick_params(axis="y", rotation=0, labelsize=12)

            # Set title for top row (edges), include metric type for bottom row
            if metric_idx == 0:
                axes[metric_idx, arch_idx].set_title(
                    f"{ds_model_names[arch]}",
                    fontsize=14,
                    weight="bold",
                )

            # Add row labels on the left side
            if arch_idx == 0:
                axes[metric_idx, arch_idx].set_ylabel(
                    f"{metric_title} IoU",
                    fontsize=14,
                    weight="bold",
                )

    # Even spacing between plots
    plt.tight_layout()

    # Save the combined architecture plot
    filename = f"{plots_root}/all_architectures_averaged_pii_combined.pdf"
    safe_write_image(fig, filename)
    print(f"Created combined architecture plot: {filename}")


if __name__ == "__main__":

    def task_to_path(model_name: str, task: str):
        return f"./graphs/{model_name}/{task}_kl.json"

    # Store results for all architectures
    all_architecture_results = {}

    for model_types in [
        gpt2_small_models,
        gpt2_medium_models,
        qwen3_06_models,
        gpt2_large_models,
        llama3_1b_models,
        qwen3_17_models,
    ]:
        model_arch = list(model_types.keys())[0]
        model_names = list(model_types[model_arch].keys())
        edge_thresholds = [1000, 4000, 2000, 10000, 10000, 20000]

        # Store all results for combined plotting
        all_results = {}

        for task, edge_threshold in zip(ft_pii_tasks, edge_thresholds):
            # Load graphs for all models
            # Run the analysis
            results = make_comparison_heatmap(
                model_names, edge_threshold, f"pii_model_comparison", task, model_arch
            )

            # Store results for combined plotting
            all_results[task] = results

            print(
                f"Analysis complete! Results saved for {len(model_names)} models with {edge_threshold} edges."
            )
            print("Edge IoU matrix:")
            print(results["edge_iou"])
            print("\nNode IoU matrix:")
            print(results["node_iou"])

        # # Create combined plots showing all PII tasks side by side
        # print("\nCreating combined PII plots...")
        # create_combined_pii_plots(all_results, model_names, edge_thresholds, f"{model_arch}_pii_model_comparison")
        # print("Combined plots created successfully!")

        # Store the averaged results for this architecture
        tasks = list(all_results.keys())
        n_models = len(model_names)

        # Calculate averaged data for each metric type
        averaged_arch_results = {}
        metric_types = ["edge_iou", "node_iou"]

        for metric_type in metric_types:
            averaged_data = np.zeros((n_models, n_models))
            for task in tasks:
                data = all_results[task][metric_type].values
                averaged_data += data
            averaged_data /= len(tasks)
            averaged_arch_results[metric_type] = pd.DataFrame(
                averaged_data, columns=model_names, index=model_names
            )

        all_architecture_results[model_arch] = averaged_arch_results

    # Create combined plots for all architectures
    print("\nCreating combined architecture plots...")
    create_combined_architecture_plots(all_architecture_results)
    print("Combined architecture plots created successfully!")
