#!/usr/bin/env python3
"""
Script to analyze leak results across all models and PII types.
Calculates average precision and recall for each PII file/CSV and
averages over all PII for each model.
"""

import argparse
import os
import pandas as pd
import glob
import numpy as np
from typing import Dict, List, Tuple
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from constants import model_display_name_dict, pii_display_name_dict, MODEL_COLORS

# Placeholder dictionary for circuit percentages - fill in with your values
# Structure: {model_name: {pii_type: circuit_percentage}}
circuit_percentages = {
    "qwen3-17": {
        "PERSON": 78.49,
        "LOC": 72.52,
        "NORP": 76.00,
    },
    "llama3-1b": {
        "PERSON": 95.00,
        "LOC": 92.52,
        "NORP": 94.06,
    },
    "gpt2-large": {
        "PERSON": 95.00,
        "LOC": 92.52,
        "NORP": 84.06,
    },
    "qwen3-06": {
        "PERSON": 80.00,
        "LOC": 70.52,
        "NORP": 96.06,
    },
    "gpt2-small": {
        "PERSON": 87.49,
        "LOC": 95.00,
        "NORP": 90.50,
    },
    "pythia-1b": {
        "PERSON": 68.46,
        "LOC": 81.67,
        "NORP": 83.02,
    },
    "gpt2-medium": {
        "PERSON": 89.64,
        "LOC": 94.32,
        "NORP": 95.00,
    },
    "pythia-410m": {
        "PERSON": 70.58,
        "LOC": 80.82,
        "NORP": 92.00,
    },
}


def parse_filename(filename: str) -> Tuple[str, str]:
    """
    Parse a filename to extract PII type and threshold.

    Args:
        filename: Filename like "PERSON_0.000058_zero.csv"

    Returns:
        Tuple of (pii_type, threshold)
    """
    # Remove .csv extension
    name = filename.replace(".csv", "")
    # Split by underscore and extract PII type and threshold
    parts = name.split("_")
    if len(parts) >= 2:
        pii_type = parts[0]
        threshold = parts[1]
        return pii_type, threshold
    else:
        # Fallback for unexpected format
        return name, "unknown"


def get_unique_thresholds(model_dir: str) -> List[str]:
    """
    Get all unique thresholds from CSV files in a model directory.

    Args:
        model_dir: Path to the model directory

    Returns:
        List of unique thresholds
    """
    thresholds = set()
    csv_files = glob.glob(os.path.join(model_dir, "*.csv"))

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        _, threshold = parse_filename(filename)
        if threshold != "unknown":
            thresholds.add(threshold)

    return sorted(list(thresholds))


def read_leak_csv(csv_path: str) -> Tuple[float, float, float, float, float, float]:
    """
    Read a leak CSV file and return average precision, recall, F1 score and their standard deviations.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Tuple of (average_precision, average_recall, average_f1, precision_std, recall_std, f1_std)
    """
    try:
        df = pd.read_csv(csv_path)
        avg_precision = df["precision"].mean()
        avg_recall = df["recall"].mean()
        avg_f1 = df["f1_score"].mean()
        precision_std = df["precision"].std()
        recall_std = df["recall"].std()
        f1_std = df["f1_score"].std()
        return avg_precision, avg_recall, avg_f1, precision_std, recall_std, f1_std
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def analyze_model_directory(
    model_dir: str, method: str
) -> Dict[str, Dict[str, Tuple[float, float, float, float, float, float]]]:
    """
    Analyze all PII CSV files in a model directory, grouped by threshold.

    Args:
        model_dir: Path to the model directory

    Returns:
        Dictionary mapping threshold to dictionary mapping PII type to (precision, recall, f1, precision_std, recall_std, f1_std)
    """
    results_by_threshold = defaultdict(dict)
    csv_files = glob.glob(os.path.join(model_dir, "*.csv"))

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        pii_type, threshold = parse_filename(filename)

        if threshold != "unknown":
            precision, recall, f1, precision_std, recall_std, f1_std = read_leak_csv(
                csv_file, method
            )
            results_by_threshold[method][threshold][pii_type] = (
                precision,
                recall,
                f1,
                precision_std,
                recall_std,
                f1_std,
            )

            model_name = os.path.basename(model_dir)
            if "baseline" in model_name:
                print(
                    f"    Threshold {threshold}, {pii_type:12s}: Precision={precision:6.2f}±{precision_std:5.2f}%, Recall={recall:6.2f}±{recall_std:5.2f}%, F1={f1:6.2f}±{f1_std:5.2f}%"
                )

    return dict(results_by_threshold)


def calculate_model_averages(
    pii_results: Dict[str, Tuple[float, float, float, float, float, float]],
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate average precision, recall, and F1 score across all PII types for a model.

    Args:
        pii_results: Dictionary mapping PII type to (precision, recall, f1, precision_std, recall_std, f1_std)

    Returns:
        Tuple of (average_precision, average_recall, average_f1, precision_std_across_pii, recall_std_across_pii, f1_std_across_pii)
    """
    if not pii_results:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    precisions = [result[0] for result in pii_results.values()]
    recalls = [result[1] for result in pii_results.values()]
    f1_scores = [result[2] for result in pii_results.values()]

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)
    precision_std_across_pii = np.std(precisions)
    recall_std_across_pii = np.std(recalls)
    f1_std_across_pii = np.std(f1_scores)

    return (
        avg_precision,
        avg_recall,
        avg_f1,
        precision_std_across_pii,
        recall_std_across_pii,
        f1_std_across_pii,
    )


def read_baseline_results(
    baseline_dir: str,
) -> Dict[str, Tuple[float, float, float, float, float, float]]:
    """
    Read baseline results from the leaks directory.

    Args:
        baseline_dir: Path to the baseline directory (e.g., results/leaks/gpt2-large-baseline)

    Returns:
        Dictionary mapping PII type to (precision, recall, f1, precision_std, recall_std, f1_std)
    """
    baseline_results = {}
    csv_files = glob.glob(os.path.join(baseline_dir, "*.csv"))
    # filter any files that are GPE.csv
    csv_files = [f for f in csv_files if "GPE" not in f]
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        pii_type = filename.replace(".csv", "")
        precision, recall, f1, precision_std, recall_std, f1_std = read_leak_csv(
            csv_file
        )
        baseline_results[pii_type] = (
            precision,
            recall,
            f1,
            precision_std,
            recall_std,
            f1_std,
        )

    return baseline_results


def format_comparison_cell(
    current_val: float,
    current_std: float,
    baseline_val: float,
    baseline_std: float,
    lower_is_better: bool = True,
) -> str:
    """
    Format a LaTeX cell with comparison to baseline, including color coding and arrows.

    Args:
        current_val: Current metric value
        current_std: Current metric standard deviation
        baseline_val: Baseline metric value
        baseline_std: Baseline metric standard deviation
        lower_is_better: Whether lower values are better (True for precision/recall/F1 when they represent leakage)

    Returns:
        Formatted LaTeX cell string
    """
    if baseline_val == 0.0:
        return f"\\cellcolor{{gray!8}}{current_val:.2f} ± {current_std:.2f}"

    # Calculate improvement
    improvement = (
        baseline_val - current_val if lower_is_better else current_val - baseline_val
    )
    improvement_pct = (improvement / baseline_val) * 100

    # Calculate absolute change
    change = abs(current_val - baseline_val)
    change_pct = abs(improvement_pct)

    # Choose color and arrow
    if improvement > 0:
        # Improved
        color = "white!8"
        arrow = "$\\downarrow$" if lower_is_better else "$\\uparrow$"
    else:
        # Degraded
        color = "white!8"
        arrow = "$\\uparrow$" if lower_is_better else "$\\downarrow$"

    return f"\\cellcolor{{{color}}}{current_val:.2f} ± {current_std:.2f} [{arrow}{change:.1f}]"


def main():
    """Main function to analyze all leak results."""
    # add args parsers
    parser = argparse.ArgumentParser(description="Analyze leak results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/circuit_leaks",
        help="Path to the results directory",
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default="./results/leaks",
        help="Path to the baseline results directory",
    )
    args = parser.parse_args()
    results_dir = args.results_dir
    baseline_dir = args.baseline_dir

    if not os.path.exists(baseline_dir):
        print(f"Baseline directory not found: {baseline_dir}")
        return

    # Get all model directories
    model_dirs = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]
    model_dirs.sort()

    print("=" * 80)
    print("LEAK ANALYSIS RESULTS")
    print("=" * 80)

    all_models = [
        "gpt2-small", 
        "gpt2-medium", 
        "qwen3-06" ,
        "gpt2-large", 
        "llama3-1b", 
        "qwen3-17"
    ]

    # # Read all baseline results - these don't have thresholds
    target_epsilon = "baseline"
    baseline_results_by_model = {}
    for model_name in sorted(all_models):
        # For baseline models in circuit results, also try to find the original baseline
        base_model_name = model_name + "-" + target_epsilon
        original_baseline_path = os.path.join(baseline_dir, base_model_name)

        if os.path.exists(original_baseline_path):
            baseline_results_by_model[model_name] = read_baseline_results(
                original_baseline_path
            )
            print(
                f"Found original baseline for {model_name}: {original_baseline_path}"
            )

    # Show unique baseline results
    unique_baselines = {}
    for model_name in sorted(baseline_results_by_model.keys()):
        base_model_name = (
            model_name.rsplit("-", 1)[0]
            if target_epsilon not in model_name
            else model_name.replace(f"-{target_epsilon}", "")
        )
        if base_model_name not in unique_baselines:
            baseline_results = baseline_results_by_model[model_name]
            if baseline_results:
                precision, recall, f1, precision_std, recall_std, f1_std = (
                    calculate_model_averages(baseline_results)
                )
                unique_baselines[base_model_name] = (
                    precision,
                    recall,
                    f1,
                    precision_std,
                    recall_std,
                    f1_std,
                )

    # --- Grouped Bar Plot for Baseline Results ---
    # Prepare data for plotting
    baseline_model_names = all_models

    all_pii_types = set()
    for model_name in baseline_results_by_model:
        for pii_type in baseline_results_by_model[model_name].keys():
            all_pii_types.add(pii_type)
    pii_types = sorted(list(all_pii_types))
    pii_types = ["PERSON", "LOC", "NORP"]

    precisions = np.zeros((len(baseline_model_names), len(pii_types)))
    recalls = np.zeros((len(baseline_model_names), len(pii_types)))
    f1s = np.zeros((len(baseline_model_names), len(pii_types)))
    precisions_std = np.zeros((len(baseline_model_names), len(pii_types)))
    recalls_std = np.zeros((len(baseline_model_names), len(pii_types)))
    f1s_std = np.zeros((len(baseline_model_names), len(pii_types)))

    for i, base_model_name in enumerate(baseline_model_names):
        if base_model_name not in baseline_results_by_model:
            print(f"No baseline results found for {base_model_name}")
            continue
        baseline_results = baseline_results_by_model[base_model_name]
        print(f"\nBaseline Model: {base_model_name}")
        average_precision = 0
        average_recall = 0 
        average_precision_std = 0
        average_recall_std = 0
        for j, pii_type in enumerate(pii_types):
            if pii_type in baseline_results:
                p, r, f, p_std, r_std, f_std = baseline_results[pii_type]
                precisions[i, j] = p
                recalls[i, j] = r
                f1s[i, j] = f
                precisions_std[i, j] = p_std
                recalls_std[i, j] = r_std
                f1s_std[i, j] = f_std                
                average_precision = average_precision + p
                average_recall = average_recall + r
                average_precision_std = average_precision_std + p_std
                average_recall_std = average_recall_std + r_std
                # print(f"    {p:.2f} ± {p_std:.2f} & {r:.2f} ± {r_std:.2f}")
        print(f"    Average Precision: {average_precision/3:.2f} $\pm$ {average_precision_std/3:.2f}")
        print(f"    Average Recall: {average_recall/3:.2f} $\pm$ {average_recall_std/3:.2f}")

    # Prepare circuit data
    circuits = np.zeros((len(baseline_model_names), len(pii_types)))
    for i, base_model_name in enumerate(baseline_model_names):
        if base_model_name in circuit_percentages:
            for j, pii_type in enumerate(pii_types):
                if pii_type in circuit_percentages[base_model_name]:
                    circuits[i, j] = circuit_percentages[base_model_name][pii_type]

    plot_results(
        baseline_model_names,
        pii_types,
        precisions,
        recalls,
        f1s,
        precisions_std,
        recalls_std,
        f1s_std,
        circuits,
        "baseline",
    )

def plot_results(
    baseline_model_names,
    pii_types,
    precisions,
    recalls,
    f1s,
    precisions_std,
    recalls_std,
    f1s_std,
    circuits,
    suffix,
):
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), dpi=400)

    metrics = [
        (precisions, precisions_std, "Precision"),
        (recalls, recalls_std, "Recall"),
        # (f1s, f1s_std, "F1 Score"),
        (circuits, None, "Faithfulness"),
    ]

    bar_width = 0.15
    x = np.arange(len(pii_types))

    # Store legend handles and labels from the first plot
    legend_handles = []
    legend_labels = []

    for plot_idx, (metric, metric_std, metric_name) in enumerate(metrics):
        ax = axes[plot_idx]
        for i, base_model_name in enumerate(baseline_model_names):
            # Handle case where there's no standard deviation (circuit plot)
            yerr_val = metric_std[i] if metric_std is not None and not np.isnan(metric_std[i]).any() else 0.01

            bar = ax.bar(
                x + i * bar_width,
                metric[i],
                bar_width,
                yerr=yerr_val,
                label=model_display_name_dict[f"{base_model_name}-baseline"],
                color=MODEL_COLORS[i],  # Use one color per model
                capsize=4,
                # thinner width of bar
                edgecolor='black'
            )

            # Collect legend handles and labels from the first plot
            if plot_idx == 0:
                legend_handles.append(bar)
                legend_labels.append(
                    model_display_name_dict[f"{base_model_name}-baseline"].replace(" Baseline", "")
                )

        ax.set_xticks(x + bar_width * (len(baseline_model_names) - 1) / 2)
        ax.set_xticklabels(
            [pii_display_name_dict[pii] for pii in pii_types], fontsize=22
        )
        # set y label to font size 16
        ax.set_yticklabels(ax.get_yticks(), fontsize=22)

        ax.set_ylabel(f"{metric_name} (%)", fontsize=22)
        # ax.set_title(f"{metric_name} by PII Type")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Create a single legend below all subplots
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=len(baseline_model_names),
        bbox_to_anchor=(0.5, -0.1),
        fontsize=22,
        # title="Models",
        # title_fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(
        f"plots/baseline_results_grouped_plots_{suffix}.pdf",
        dpi=400,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
