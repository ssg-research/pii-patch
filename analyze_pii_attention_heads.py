import os

from matplotlib.colors import LinearSegmentedColormap

from analysis.graph_attention_analyzer import GraphAttentionAnalyzer
from gencircuits.eap.graph import Graph
from constants import ft_pii_tasks, model_display_name_dict, MODEL_COLORS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import PowerNorm

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
    # model_order = ["llama3-1b", "qwen3-17", "gpt2-large", "gpt2-medium", "gpt2-small"]
    model_order = ["gpt2-medium", "qwen3-06", "llama3-1b"]

    if base_name in model_order:
        idx = model_order.index(base_name)
        return MODEL_COLORMAPS[idx]
    else:
        # Default to first colormap if model not found
        return MODEL_COLORMAPS[0]


def collect_edge_scores_per_model(models: list):
    """
    Collect edge scores for each model individually to compute model-specific dynamic thresholds.

    Args:
        models: List of model paths to analyze

    Returns:
        dict: Dictionary mapping model paths to their edge scores arrays
    """
    print("Collecting edge scores per model for model-specific dynamic threshold calculation...")
    model_edge_scores = {}

    for model_path in models:
        print(f"Collecting scores from {model_path}...")
        model_scores = []

        for task in ft_pii_tasks:
            graph_file = f"graphs/{model_path}/{task}_kl.json"

            if not os.path.exists(graph_file):
                print(f"  Warning: {graph_file} not found, skipping...")
                continue

            try:
                # Load graph
                graph = Graph.from_json(graph_file)

                # Extract all edge scores without any threshold filtering
                if hasattr(graph, "scores") and graph.scores is not None:
                    edge_scores = graph.scores.flatten()
                    # Only consider real edges
                    if hasattr(graph, "real_edge_mask"):
                        real_mask = graph.real_edge_mask.flatten()
                        edge_scores = edge_scores[real_mask]

                    # Take absolute values and filter out zeros
                    edge_scores = np.abs(edge_scores.cpu().numpy())
                    edge_scores = edge_scores[edge_scores > 0]

                    model_scores.extend(edge_scores.tolist())
                    print(f"    {task}: collected {len(edge_scores)} edge scores")

            except Exception as e:
                print(f"    Error processing {task}: {e}")

        if model_scores:
            model_edge_scores[model_path] = np.array(model_scores)
            print(f"  Total for {model_path}: {len(model_scores)} edge scores")
            print(
                f"  Score range: [{np.min(model_scores):.6f}, {np.max(model_scores):.6f}]"
            )
        else:
            print(f"  No edge scores collected for {model_path}!")
            model_edge_scores[model_path] = np.array([])

    return model_edge_scores


def analyze_high_scoring_attention_heads(models: list, csvs_only: bool = False, dynamic_thresh_percentile: int = 95):
    """
    Main analysis function to identify high-scoring attention heads
    across all PII tasks and models using model-specific dynamic thresholds.
    """
    print("Analyzing High-Scoring Attention Heads in PII Tasks")
    print("=" * 60)

    # Create output directory
    os.makedirs("./plots/attention_analysis", exist_ok=True)
    os.makedirs("./results/circuit_analysis", exist_ok=True)

    # Check if CSV files already exist for all models
    all_csvs_exist = True
    for model_path in models:
        # Check for the frequency CSV file pattern (we'll determine threshold later)
        frequency_files = [f for f in os.listdir("./results/circuit_analysis/") 
                          if f.startswith(f"attention_frequency_{model_path}_dynamic_t") and f.endswith(".csv")]
        analysis_files = [f for f in os.listdir("./results/circuit_analysis/") 
                         if f.startswith(f"attention_analysis_{model_path}_dynamic_t") and f.endswith(".csv")]
        
        if not frequency_files or not analysis_files:
            all_csvs_exist = False
            break

    if all_csvs_exist:
        print("All CSV files already exist. Skipping analysis.")
        print("Found existing files for all models:")
        for model_path in models:
            frequency_files = [f for f in os.listdir("./results/circuit_analysis/") 
                              if f.startswith(f"attention_frequency_{model_path}_dynamic_t")]
            if frequency_files:
                print(f"  - {model_path}: {frequency_files[0]}")
        
        # Load existing thresholds from filenames and reconstruct average scores
        model_thresholds = {}
        results = {"model_specific_thresholds": {}}
        
        for model_path in models:
            frequency_files = [f for f in os.listdir("./results/circuit_analysis/") 
                              if f.startswith(f"attention_frequency_{model_path}_dynamic_t")]
            if frequency_files:
                # Extract threshold from filename: attention_frequency_{model}_dynamic_t{threshold}.csv
                filename = frequency_files[0]
                threshold_part = filename.split("_dynamic_t")[1].replace(".csv", "")
                try:
                    threshold = float(threshold_part)
                    model_thresholds[model_path] = threshold
                    print(f"  - Loaded threshold for {model_path}: {threshold:.6f}")
                    
                    # Reconstruct average scores matrix from existing graphs
                    print(f"  - Reconstructing average scores for {model_path}...")
                    model_results = {}
                    task_scores = []
                    
                    for task in ft_pii_tasks:
                        graph_file = f"graphs/{model_path}/{task}_kl.json"
                        if os.path.exists(graph_file):
                            try:
                                graph = Graph.from_json(graph_file)
                                analyzer = GraphAttentionAnalyzer(graph)
                                scores_matrix = analyzer.extract_attention_scores(
                                    score_type="edge_scores", absolute=True, threshold=threshold
                                )
                                task_scores.append(scores_matrix)
                                model_results[task] = scores_matrix
                            except Exception as e:
                                print(f"    Error reconstructing {task}: {e}")
                    
                    if task_scores:
                        # Compute average scores across tasks
                        avg_scores = np.mean(task_scores, axis=0)
                        model_results["average"] = avg_scores
                        print(f"    Reconstructed average: {np.count_nonzero(avg_scores)} active heads")
                    
                    results["model_specific_thresholds"][model_path] = model_results
                    
                except ValueError:
                    print(f"  - Warning: Could not parse threshold from {filename}")
        
        # return results, model_thresholds

    if all_csvs_exist is False:
        print("Some CSV files are missing. Proceeding with full analysis...")
        # Step 1: Collect edge scores per model to compute model-specific dynamic thresholds
        model_edge_scores = collect_edge_scores_per_model(models)

        if not model_edge_scores or all(len(scores) == 0 for scores in model_edge_scores.values()):
            print("No edge scores found. Exiting analysis.")
            return

        # Step 2: Calculate model-specific dynamic thresholds as 95th percentile
        model_thresholds = {}
        
        for model_path, edge_scores in model_edge_scores.items():
            if len(edge_scores) > 0:
                model_threshold = np.percentile(edge_scores, dynamic_thresh_percentile)
                model_thresholds[model_path] = model_threshold
                print(f"Model {model_path} dynamic threshold ({dynamic_thresh_percentile}th percentile): {model_threshold:.6f}")
            else:
                print(f"Warning: No edge scores for model {model_path}, skipping")
                continue

    
    results = {}
    threshold_results = {}

    for model_path in models:
        if model_path not in model_thresholds:
            continue
            
        threshold = model_thresholds[model_path]
        model_name = model_display_name_dict[model_path]
        print(f"\nAnalyzing model: {model_name} with threshold {threshold:.6f}")
        model_results = {}

        # Collect scores for each task
        task_scores = []
        task_top_heads = []

        for task in ft_pii_tasks:
            graph_file = f"graphs/{model_path}/{task}_kl.json"

            if not os.path.exists(graph_file):
                print(f"  Warning: {graph_file} not found, skipping...")
                continue

            try:
                # Load graph and analyze
                graph = Graph.from_json(graph_file)
                analyzer = GraphAttentionAnalyzer(graph)

                # Extract edge scores (since node scores aren't available)
                scores_matrix = analyzer.extract_attention_scores(
                    score_type="edge_scores", absolute=True, threshold=threshold
                )

                # Analyze top heads for this task
                top_heads = analyzer.analyze_top_scoring_heads(
                    scores_matrix=scores_matrix, top_k=20
                )

                if not top_heads.empty:
                    top_heads["task"] = task
                    top_heads["model"] = model_path
                    task_top_heads.append(top_heads)

                task_scores.append(scores_matrix)
                model_results[task] = scores_matrix

                print(
                    f"    {task}: {np.count_nonzero(scores_matrix)} active heads, max={np.max(scores_matrix):.4f}"
                )

            except Exception as e:
                print(f"    Error processing {task}: {e}")

        if task_scores:
            # Compute average scores across tasks for this model
            avg_scores = np.mean(task_scores, axis=0)
            model_results["average"] = avg_scores

            print(
                f"  Average across tasks: {np.count_nonzero(avg_scores)} active heads, max={np.max(avg_scores):.4f}"
            )

            # Generate heatmap for this model's average
            analyzer = GraphAttentionAnalyzer(
                graph
            )  # Use last loaded graph for structure
            fig = analyzer.plot_attention_heatmap(
                scores_matrix=avg_scores,
                title=f"{model_name} - Average Attention Scores (threshold≥{threshold:.6f})",
                save_path=f"./plots/attention_analysis/{model_path}_avg_dynamic_t{threshold:.6f}.pdf",
            )
            plt.close(fig)

            # Generate layer aggregation plot
            # fig = analyzer.plot_layer_aggregated_scores(
            #     scores_matrix=avg_scores,
            #     aggregation="sum",
            #     save_path=f"./plots/attention_analysis/{model_path}_layers_dynamic_t{threshold:.6f}.pdf",
            # )
            plt.close(fig)

            threshold_results[model_path] = model_results

            # Compile top heads across all tasks for this model
            if task_top_heads:
                all_top_heads = pd.concat(task_top_heads, ignore_index=True)

                # Find heads that appear frequently in top rankings
                head_frequency = (
                    all_top_heads.groupby(["layer", "head"])
                    .agg({"score": ["count", "mean", "std"], "rank": "mean"})
                    .round(6)
                )
                head_frequency.columns = [
                    "frequency",
                    "mean_score",
                    "std_score",
                    "mean_rank",
                ]
                head_frequency = head_frequency.reset_index()
                head_frequency = head_frequency.sort_values(
                    "frequency", ascending=False
                )

                print(f"\n  Most frequently high-scoring heads for {model_name}:")
                print(head_frequency.head(10).to_string(index=False))

                # Save detailed results
                all_top_heads.to_csv(
                    f"./results/circuit_analysis/attention_analysis_{model_path}_dynamic_t{threshold:.6f}.csv",
                    index=False,
                )

                # remove any rows where the layer is zero
                head_frequency = head_frequency[head_frequency["layer"] > 0]

                head_frequency.to_csv(
                    f"./results/circuit_analysis/attention_frequency_{model_path}_dynamic_t{threshold:.6f}.csv",
                    index=False,
                )

        results[f"model_specific_thresholds"] = threshold_results

    # Cross-model comparison
    # if csvs_only is False:
    #     print(f"\n--- Cross-Model Comparison (model-specific thresholds) ---")

    #     # Compare average scores across models
    #     model_averages = {}
    #     for model_path, model_results in threshold_results.items():
    #         if "average" in model_results:
    #             model_averages[model_path] = model_results["average"]

    #     if len(model_averages) > 1:
    #         # Create comparison plot with 2x2 grid layout
    #         n_models = len(model_averages)
    #         n_rows = (
    #             n_models + 1
    #         ) // 2  # Ceiling division to get number of rows
    #         n_cols = 2

    #         fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))

    #         # Flatten axes for easier indexing
    #         axes_flat = axes.flatten() if n_models > 1 else axes

    #         for idx, (model_path, avg_scores) in enumerate(
    #             model_averages.items()
    #         ):
    #             ax = axes_flat[idx]
    #             model_threshold = model_thresholds[model_path]

    #             base_cmap = get_model_colormap(model_path)                
    #             sns.heatmap(
    #                 avg_scores,
    #                 annot=False,
    #                 cmap=base_cmap,
    #                 ax=ax,
    #                 xticklabels=[f"H{i}" for i in range(avg_scores.shape[1])],
    #                 yticklabels=[f"A{i}" for i in range(avg_scores.shape[0])],
    #                 cbar=True,
    #                 cbar_kws={"shrink": 0.9},
    #                 norm=PowerNorm(gamma=0.75)
    #             )

    #             ax.set_title(f"{model_display_name_dict[model_path]}", fontsize=12)
    #             if idx >= n_models - n_cols:  # Only set xlabel for bottom row
    #                 ax.set_xlabel("Attention Head", fontsize=16)
    #             if idx % n_cols == 0:  # Only set ylabel for the first column
    #                 ax.set_ylabel("Attention Layer", fontsize=16)

    #         # Hide any unused subplots
    #         plt.tight_layout()
    #         plt.savefig(
    #             f"./plots/attention_analysis/model_comparison_model_specific_thresholds.pdf",
    #             dpi=400,
    #             bbox_inches="tight",
    #         )

    #         # Statistical comparison
    #         print(f"Model comparison statistics:")
    #         for model_path, avg_scores in model_averages.items():
    #             active_heads = np.count_nonzero(avg_scores)
    #             max_score = np.max(avg_scores)
    #             mean_score = (
    #                 np.mean(avg_scores[avg_scores > 0])
    #                 if active_heads > 0
    #                 else 0
    #             )
    #             model_threshold = model_thresholds[model_path]

    #             print(
    #                 f"  {model_display_name_dict[model_path]} (t={model_threshold:.6f}): {active_heads} active heads, max={max_score:.4f}, mean_active={mean_score:.4f}"
    #             )

    return results, model_thresholds


def create_subset_model_comparison(
    models: list, layers: list, results: dict, model_thresholds: dict
):
    """
    Create a model comparison plot for a specific subset of layers.

    Args:
        models: List of model paths to compare
        layers: List of layer indices (e.g., [3, 4, 9, 10])
        results: Results dictionary from the main analysis
        model_thresholds: Dictionary mapping model paths to their thresholds
    """
    print(f"\n--- Subset Model Comparison for Layers {layers} ---")

    # Get model averages from results
    model_averages = {}
    threshold_results = results["model_specific_thresholds"]
    for model_path in models:
        if (
            model_path in threshold_results
            and "average" in threshold_results[model_path]
        ):
            model_averages[model_path] = threshold_results[model_path]["average"]

    if len(model_averages) < 2:
        print("Need at least 2 models for comparison")
        return

    # Extract subset of layers from each model
    subset_data = {}
    for model_path, avg_scores in model_averages.items():
        # Create subset matrix with only specified layers
        n_heads = avg_scores.shape[1]
        subset_scores = np.zeros((len(layers), n_heads))

        for i, layer_idx in enumerate(layers):
            if layer_idx < avg_scores.shape[0]:  # Check if layer exists in this model
                subset_scores[i, :] = avg_scores[layer_idx, :]  # Taking row layer_idx

        subset_data[model_path] = subset_scores

    # Create comparison plot with 2x2 grid layout
    n_models = len(subset_data)
    n_rows = 3
    n_cols = 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3, 3))

    # Handle different subplot configurations
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    axes_flat = axes.flatten() if n_models > 1 else axes

    for idx, (model_path, subset_scores) in enumerate(subset_data.items()):
        model_cmap = get_model_colormap(model_path)
        ax = axes_flat[idx]

        # For llama3 models, show only every 2nd tick to save space
        if "llama3" in model_path:
            x_tick_indices = list(range(0, subset_scores.shape[1], 2))  # 0, 2, 4, 6, etc.
            x_tick_labels = [f"{i}" if i in x_tick_indices else "" for i in range(subset_scores.shape[1])]
        else:
            x_tick_labels = [f"{i}" for i in range(subset_scores.shape[1])]

        sns.heatmap(
            subset_scores,
            annot=False,
            cmap=model_cmap,
            ax=ax,
            xticklabels=x_tick_labels,
            yticklabels=[f"{layer_idx}" for layer_idx in layers],            
            # cbar=True if idx == n_models - 1 else False,  # Show colorbar only on last subplot
        )
        # make cbar font smaller
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=6)
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'.lstrip('0') if x != 0 else '0'))
        ax.set_title(model_display_name_dict[model_path], fontsize=6, weight="bold")
        if idx % n_cols == 0:
            ax.set_ylabel("Layers", fontsize=6, weight="bold")
        if idx >= n_models - n_cols:
            ax.set_xlabel("Attention Heads", fontsize=6, weight="bold")
        # rotate the y labels for better visibility
        plt.setp(ax.get_yticklabels(), rotation=0, ha="center", fontsize=6)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)

    # Hide unused subplots
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()

    # Save the plot
    layers_str = "_".join(map(str, layers))
    save_path = f"./plots/attention_analysis/subset_comparison_layers_{layers_str}_model_specific_thresholds.pdf"
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"Subset comparison plot saved to: {save_path}")

    # Print statistics for the subset
    print(f"Statistics for layers {layers}:")
    for model_path, subset_scores in subset_data.items():
        active_heads = np.count_nonzero(subset_scores)
        max_score = np.max(subset_scores)
        mean_score = (
            np.mean(subset_scores[subset_scores > 0]) if active_heads > 0 else 0
        )
        model_threshold = model_thresholds.get(model_path, "N/A")

        print(
            f"  {model_display_name_dict[model_path]} (t={model_threshold:.6f}): {active_heads} active positions, max={max_score:.4f}, mean_active={mean_score:.4f}"
        )


def identify_consistent_high_scoring_heads(models):
    """
    Identify attention heads that consistently score high across different tasks and models.
    """
    print("\n" + "=" * 60)
    print("Identifying Consistently High-Scoring Heads")
    print("=" * 60)

    # Read the saved results files (updated for model-specific threshold naming)
    results_files = []
    for file in os.listdir("./results/circuit_analysis/"):
        if (
            file.startswith("attention_frequency_")
            and "dynamic_t" in file
            and file.endswith(".csv")
        ):
            results_files.append(file)

    if not results_files:
        print("No frequency analysis files found. Run the main analysis first.")
        return

    # Combine results across models and thresholds
    all_frequencies = []
    for file in results_files:
        df = pd.read_csv(f"./results/circuit_analysis/{file}")
        # Extract model and threshold from filename (updated for model-specific naming)
        parts = file.replace(".csv", "").split("_")
        # Find the index of "dynamic" to properly parse the filename
        if "dynamic" in parts:
            dynamic_idx = parts.index("dynamic")
            model = "_".join(parts[2:dynamic_idx])  # Get model name
            threshold = parts[dynamic_idx + 1][1:]  # Get threshold (remove 't' prefix)
        else:
            # Fallback for older naming conventions or direct threshold values
            # Expected format: attention_frequency_{model}_dynamic_t{threshold}.csv
            threshold_part = [p for p in parts if p.startswith("t") and p[1:].replace(".", "").replace("-", "").isdigit()]
            if threshold_part:
                threshold = threshold_part[0][1:]  # Remove 't' prefix
                # Get model name as everything between "frequency" and the threshold part
                freq_idx = parts.index("frequency")
                threshold_idx = parts.index(threshold_part[0])
                model = "_".join(parts[freq_idx+1:threshold_idx-1])  # Exclude "dynamic"
            else:
                print(f"Warning: Could not parse filename {file}, skipping")
                continue

        df["model"] = model
        df["threshold"] = threshold
        all_frequencies.append(df)

    if all_frequencies:
        combined_df = pd.concat(all_frequencies, ignore_index=True)

        # Find heads that appear frequently across different conditions
        cross_condition_freq = (
            combined_df.groupby(["layer", "head"])
            .agg(
                {
                    "frequency": "sum",
                    "mean_score": "mean",
                    "model": "nunique",
                    "threshold": "nunique",
                }
            )
            .round(4)
        )

        cross_condition_freq.columns = [
            "total_frequency",
            "avg_score",
            "num_models",
            "num_thresholds",
        ]
        cross_condition_freq = cross_condition_freq.reset_index()
        cross_condition_freq = cross_condition_freq.sort_values(
            "total_frequency", ascending=False
        )

        print("\nMost consistently high-scoring heads across all conditions:")
        print(cross_condition_freq.head(15).to_string(index=False))

        # Save the consolidated results
        combined_df.to_csv(
            f"./results/circuit_analysis/attention_analysis_combined_{','.join(models)}.csv", index=False
        )
        cross_condition_freq.to_csv(
            f"./results/circuit_analysis/attention_heads_consistency_{','.join(models)}.csv", index=False
        )

        return cross_condition_freq

    # cross_condition_freq where we select a unique layer and head combination
    return cross_condition_freq


if __name__ == "__main__":
    # Create results directory
    os.makedirs("./results/circuit_analysis/", exist_ok=True)
    models = [        
        # "gpt2-small-baseline",
        # "gpt2-small-dp8",
        # "gpt2-small-dp4",
        # "gpt2-small-dp1",
        "gpt2-medium-baseline",
        # "gpt2-medium-dp8",
        # "gpt2-medium-dp4",
        # "gpt2-medium-dp1",
        # "gpt2-large-baseline",
        # "gpt2-large-dp8",
        # # "gpt2-large-dp4",
        # "gpt2-large-dp1",
        "qwen3-06-baseline",
        "llama3-1b-baseline",
        # "llama3-1b-dp8",
        # "llama3-1b-dp4",
        # "llama3-1b-dp1",
        # "qwen3-06-dp8",
        # "qwen3-06-dp4",
        # "qwen3-06-dp1",
        # "qwen3-17-baseline",
        # "qwen3-17-dp8",
        # "qwen3-17-dp4",
        # "qwen3-17-dp1",
    ]


    # for model in models:
    print("\n" + "=" * 60)
    # print(f"Starting analysis for model: {model}")
    print("=" * 60)
    # Run the main analysis
    dynamic_thresh_percentile = 90
    results, model_thresholds = analyze_high_scoring_attention_heads(models, False, dynamic_thresh_percentile)
    # print(f"Completed analysis for model: {model}")

    if results:
        subset_layers = [9, 11, 14, 16]  # You can modify this list
        create_subset_model_comparison(models, subset_layers, results, model_thresholds)

    # Identify consistent patterns
    # arched_models = [
    #     [
    #         "gpt2-small-baseline",
    #         "gpt2-small-dp8",
    #         "gpt2-small-dp4",
    #         "gpt2-small-dp1",
    #     ],
    #     [            
    #         "gpt2-medium-baseline",
    #         "gpt2-medium-dp8",
    #         "gpt2-medium-dp4",
    #         "gpt2-medium-dp1",
    #     ],
    #     [
    #         "gpt2-large-baseline",
    #         "gpt2-large-dp8",
    #         "gpt2-large-dp4",
    #         "gpt2-large-dp1",
    #     ],
    #     [
    #         "llama3-1b-baseline",
    #         "llama3-1b-dp8",
    #         "llama3-1b-dp4",
    #         "llama3-1b-dp1",
    #     ]
    # ]
    # for a_models in arched_models:
    #     identify_consistent_high_scoring_heads(a_models)

    print("\n" + "=" * 60)
    print("Analysis completed! Check the following outputs:")
    print("- ./plots/attention_analysis/ for visualizations")
    print("- ./results/ for detailed CSV files")
    print("=" * 60)
