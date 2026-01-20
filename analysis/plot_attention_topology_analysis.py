import json
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import stats


def parse_connection_data(json_data, source_layer="a0"):
    """
    Parse JSON data to extract EAP-IG scores for different connection types
    from a specific source attention layer.

    Args:
        json_data: Dictionary containing connection scores
        source_layer: The attention layer to analyze (e.g., "a0", "a10")

    Returns:
        Dictionary with categorized scores
    """
    # Initialize containers for different connection types
    attention_to_attention = []
    attention_to_mlp = []

    # Pattern to match different connection types
    # Attention-to-attention: a10.h10->a11.h11<k>
    attn_to_attn_pattern = rf"{source_layer}\.h\d+->a\d+\.h\d+<[kqv]>"
    # Attention-to-MLP: a0.h10->m0
    attn_to_mlp_pattern = rf"{source_layer}\.h\d+->m\d+"

    all_edges = json_data['edges'].items()
    for connection_name, connection_data in all_edges:
        score = connection_data.get("score", 0)

        if abs(score) < 1e-10:
            continue

        # Use absolute value for analysis
        if re.match(attn_to_attn_pattern, connection_name):
            attention_to_attention.append(abs(score))  
        elif re.match(attn_to_mlp_pattern, connection_name):
            attention_to_mlp.append(abs(score))

    return {
        "attention_to_attention": attention_to_attention,
        "attention_to_mlp": attention_to_mlp,
        "source_layer": source_layer,
    }


def analyze_score_distributions(scores_dict):
    """
    Perform statistical analysis on the score distributions.
    """
    results = {}

    for conn_type, scores in scores_dict.items():
        if conn_type == "source_layer":
            continue

        if len(scores) > 0:
            results[conn_type] = {
                "count": len(scores),
                "mean": np.mean(scores),
                "median": np.median(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "q25": np.percentile(scores, 25),
                "q75": np.percentile(scores, 75),
            }
        else:
            results[conn_type] = {"count": 0}

    return results


def create_distribution_plots(scores_dict):
    """
    Create comprehensive visualization of score distributions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f'EAP-IG Score Distribution Analysis for {scores_dict["source_layer"]}',
        fontsize=16,
        fontweight="bold",
    )

    # Prepare data for plotting
    attn_scores = scores_dict["attention_to_attention"]
    mlp_scores = scores_dict["attention_to_mlp"]

    # 1. Histogram comparison (top-left)
    ax1 = axes[0, 0]
    if len(attn_scores) > 0:
        ax1.hist(
            attn_scores,
            bins=50,
            alpha=0.7,
            label="Attention→Attention",
            color="skyblue",
            density=True,
        )
    if len(mlp_scores) > 0:
        ax1.hist(
            mlp_scores,
            bins=50,
            alpha=0.7,
            label="Attention→MLP",
            color="lightcoral",
            density=True,
        )

    ax1.set_xlabel("EAP-IG Score (absolute value)")
    ax1.set_ylabel("Density")
    ax1.set_title("Score Distribution Comparison")
    ax1.legend()
    ax1.set_yscale("log")  # Log scale often helpful for EAP-IG scores

    # 2. Box plot comparison (top-right)
    ax2 = axes[0, 1]
    plot_data = []
    labels = []

    if len(attn_scores) > 0:
        plot_data.append(attn_scores)
        labels.append("Attention→Attention")
        
    if len(mlp_scores) > 0:
        plot_data.append(mlp_scores)
        labels.append("Attention→MLP")

    if plot_data:
        bp = ax2.boxplot(plot_data, labels=labels, patch_artist=True)
        colors = ["skyblue", "lightcoral"]
        for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
            patch.set_facecolor(color)

    ax2.set_ylabel("EAP-IG Score (absolute value)")
    ax2.set_title("Score Distribution Summary")
    ax2.set_yscale("log")

    # 3. Cumulative distribution (bottom-left)
    ax3 = axes[1, 0]
    if len(attn_scores) > 0:
        sorted_attn = np.sort(attn_scores)
        p_attn = np.arange(1, len(sorted_attn) + 1) / len(sorted_attn)
        ax3.plot(
            sorted_attn, p_attn, label="Attention→Attention", color="blue", linewidth=2
        )

    if len(mlp_scores) > 0:
        sorted_mlp = np.sort(mlp_scores)
        p_mlp = np.arange(1, len(sorted_mlp) + 1) / len(sorted_mlp)
        ax3.plot(sorted_mlp, p_mlp, label="Attention→MLP", color="red", linewidth=2)

    ax3.set_xlabel("EAP-IG Score (absolute value)")
    ax3.set_ylabel("Cumulative Probability")
    ax3.set_title("Cumulative Distribution Function")
    ax3.legend()
    ax3.set_xscale("log")
    ax3.grid(True, alpha=0.3)

    # 4. Statistical comparison table (bottom-right)
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Create summary statistics table
    stats_data = analyze_score_distributions(scores_dict)

    table_data = []
    headers = ["Metric", "Attention→Attention", "Attention→MLP"]

    if "attention_to_attention" in stats_data and "attention_to_mlp" in stats_data:
        attn_stats = stats_data["attention_to_attention"]
        mlp_stats = stats_data["attention_to_mlp"]

        metrics = ["Count", "Mean", "Median", "Std Dev", "Min", "Max"]
        attn_values = [
            attn_stats.get("count", 0),
            f"{attn_stats.get('mean', 0):.2e}",
            f"{attn_stats.get('median', 0):.2e}",
            f"{attn_stats.get('std', 0):.2e}",
            f"{attn_stats.get('min', 0):.2e}",
            f"{attn_stats.get('max', 0):.2e}",
        ]
        mlp_values = [
            mlp_stats.get("count", 0),
            f"{mlp_stats.get('mean', 0):.2e}",
            f"{mlp_stats.get('median', 0):.2e}",
            f"{mlp_stats.get('std', 0):.2e}",
            f"{mlp_stats.get('min', 0):.2e}",
            f"{mlp_stats.get('max', 0):.2e}",
        ]

        for i, metric in enumerate(metrics):
            table_data.append([metric, attn_values[i], mlp_values[i]])

    if table_data:
        table = ax4.table(
            cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

    ax4.set_title("Statistical Summary", pad=20)

    plt.tight_layout()
    return fig


def perform_statistical_tests(scores_dict):
    """
    Perform statistical tests to compare the two distributions.
    """
    attn_scores = scores_dict["attention_to_attention"]
    mlp_scores = scores_dict["attention_to_mlp"]

    if len(attn_scores) == 0 or len(mlp_scores) == 0:
        return "Cannot perform statistical tests: one or both groups have no data."

    # Mann-Whitney U test (non-parametric test for different distributions)
    statistic, p_value = stats.mannwhitneyu(
        attn_scores, mlp_scores, alternative="two-sided"
    )

    # Kolmogorov-Smirnov test (tests if distributions are different)
    ks_statistic, ks_p_value = stats.ks_2samp(attn_scores, mlp_scores)

    results = {
        "mann_whitney": {
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < 0.05,
        },
        "kolmogorov_smirnov": {
            "statistic": ks_statistic,
            "p_value": ks_p_value,
            "significant": ks_p_value < 0.05,
        },
    }

    return results


# Example usage:
def main():
    # Replace with your actual JSON data
    model = "gpt2-small-baseline"
    pii_task = "pii_leakage_person"
    with open(f"graphs/{model}/{pii_task}_kl.json", "r") as f:
        data = json.load(f)

    source_layer = "a10"
    scores = parse_connection_data(data, source_layer)

    # Create visualizations
    fig = create_distribution_plots(scores)
    os.makedirs("./plots/attention-analysis", exist_ok=True)
    fig.savefig(
        f"./plots/attention-analysis/{model}_{pii_task}_{source_layer}.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=400,
    )

    # Perform statistical analysis
    stats_results = analyze_score_distributions(scores)
    print(f"\nStatistical Summary for {source_layer}:")
    for conn_type, stats in stats_results.items():
        print(f"\n{conn_type.replace('_', ' ').title()}:")
        for metric, value in stats.items():
            print(f"  {metric}: {value}")

    # Perform statistical tests
    test_results = perform_statistical_tests(scores)
    if isinstance(test_results, dict):
        print(f"\nStatistical Tests:")
        print(
            f"Mann-Whitney U test p-value: {test_results['mann_whitney']['p_value']:.6f}"
        )
        print(
            f"  Significant difference: {test_results['mann_whitney']['significant']}"
        )
        print(
            f"Kolmogorov-Smirnov test p-value: {test_results['kolmogorov_smirnov']['p_value']:.6f}"
        )
        print(
            f"  Significant difference: {test_results['kolmogorov_smirnov']['significant']}"
        )
    else:
        print(f"\nStatistical Tests: {test_results}")


if __name__ == "__main__":
    main()
