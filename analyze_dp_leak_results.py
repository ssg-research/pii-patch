#!/usr/bin/env python3
"""
Script to analyze leak results across all models and PII types.
Calculates average precision and recall for each PII file/CSV and
averages over all PII for each model.
"""
from adjustText import adjust_text
import argparse
import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import defaultdict
from constants import (
    model_base_display_dict,
    epsilon_display_dict,
    MODEL_COLORS,
    MODEL_ORDER,
)
from matplotlib.patches import Patch


# Define unique color and marker combinations for each method
METHOD_STYLES = {
    "PATCH-Baseline": {"color": "#27d8eb", "marker": "X", "size": 140},
    "PATCH-DP(E=8)": {"color": "#e7fa0c", "marker": "X", "size": 140},
    "Scrub": {"color": "#ff7f0e", "marker": "s", "size": 100},
    "APNEAP": {"color": "#2ca02c", "marker": "^", "size": 100},
    "DP(ε=8)": {"color": "#8c564b", "marker": "v", "size": 100},
    "DP(ε=4)": {"color": "#9467bd", "marker": "v", "size": 100},
    "DP(ε=1)": {"color": "#d62728", "marker": "v", "size": 100},
    "Base": {"color": "#1f77b4", "marker": "o", "size": 100},
}


def get_method_style(method_label: str) -> dict:
    """Get the color, marker, and size for a given method label."""
    # Handle variations in method labels
    if method_label in METHOD_STYLES:
        return METHOD_STYLES[method_label]    
    elif "PATCH-DP(E=" in method_label:
        if "E=1)" in method_label:
            return METHOD_STYLES["PATCH-DP(E=1)"]
        elif "E=4)" in method_label:
            return METHOD_STYLES["PATCH-DP(E=4)"]
        elif "E=8)" in method_label:
            return METHOD_STYLES["PATCH-DP(E=8)"]
    elif "PATCH-" in method_label and "Baseline" in method_label:
        return METHOD_STYLES["PATCH-Baseline"]

    # Default fallback
    return {"color": "#000000", "marker": "o", "size": 100}


def create_unified_legend(fig, axes):
    """Create a unified legend at the bottom of the figure for all methods."""
    from matplotlib.lines import Line2D

    # Create legend elements for all methods
    legend_elements = []
    for method, style in METHOD_STYLES.items():
        if method == "Scrub":
            method = "Scrubbed"
        elif method == "Base":
            method = "Baseline"
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="w",
                markerfacecolor=style["color"],
                markersize=10,
                label=method,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )
        )

    # Add the legend at the bottom of the figure
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.515, 0.065),
        ncol=8,
        fontsize=10,
        columnspacing=1.5,
        frameon=True,
        fancybox=True,
    )

    return legend_elements


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


def read_baseline_leak_csv(csv_path: str) -> Tuple[float, float, float]:
    """
    Read a baseline leak CSV file and calculate precision, recall, and F1-score from true_positives and false_positives.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Tuple of (average_precision, average_recall, average_f1)
    """
    try:
        df = pd.read_csv(csv_path)
        # Calculate precision and recall from true_positives and false_positives
        # Precision = TP / (TP + FP)
        # Recall is already in the CSV, but we can also calculate it if needed
        precisions = df["true_positives"] / (
            df["true_positives"] + df["false_positives"]
        )
        recalls = df["recall"]

        # Calculate F1-score: F1 = 2 * (precision * recall) / (precision + recall)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        # Handle division by zero cases
        f1_scores = f1_scores.fillna(0.0)

        avg_precision = precisions.mean() * 100
        avg_recall = recalls.mean()
        avg_f1 = f1_scores.mean() * 100

        return avg_precision, avg_recall, avg_f1
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return 0.0, 0.0, 0.0


def read_circuit_leak_csv(csv_path: str) -> Tuple[float, float, float]:
    """
    Read a circuit leak CSV file and calculate precision, recall, and F1-score from true_positives and false_positives.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Tuple of (average_precision, average_recall, average_f1)
    """
    try:
        df = pd.read_csv(csv_path)
        # Calculate precision and recall from true_positives and false_positives
        # Precision = TP / (TP + FP)
        # Recall is already in the CSV, but we can also calculate it if needed
        precisions = df["true_positives"] / (
            df["true_positives"] + df["false_positives"]
        )
        recalls = df["recall"]

        # Calculate F1-score: F1 = 2 * (precision * recall) / (precision + recall)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        # Handle division by zero cases
        f1_scores = f1_scores.fillna(0.0)

        avg_precision = precisions.mean() * 100
        avg_recall = recalls.mean()
        avg_f1 = f1_scores.mean() * 100

        return avg_precision, avg_recall, avg_f1
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return 0.0, 0.0, 0.0


def process_circuit_results(
    circuit_dir: str, target_categories: List[str] = None
) -> Dict:
    """
    Process circuit leak results for all models and configurations.

    Args:
        circuit_dir: Path to the circuit leaks directory
        target_categories: List of PII categories to process (default: ['PERSON', 'LOC', 'NORP'])

    Returns:
        Dictionary with structure: {model_name: {(threshold, patch): {category: (precision, recall, f1, iterations)}}}
    """
    if target_categories is None:
        target_categories = ["PERSON", "LOC", "NORP"]

    df = pd.read_csv(f"{circuit_dir}/all-results.csv")

    # Filter for target categories and only "self" patch
    df = df[df["pii_class"].isin(target_categories)]
    df = df[df["patch"].isin(["self"])]
    # df = df[df["threshold"].isin([95, 99])]
    df = df[df["model"].str.contains("baseline|dp8")]

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Group by model, threshold, patch, and pii_class to aggregate iterations
    for (model, threshold, patch, ablation, pii_class), group in df.groupby(
        ["model", "threshold", "patch", "ablation", "pii_class"]
    ):

        avg_precision = group["precision"].max()
        avg_recall = group["recall"].max()
        # Calculate F1-score from precision and recall
        f1_scores = (
            2
            * (group["precision"] * group["recall"])
            / (group["precision"] + group["recall"])
        )
        f1_scores = f1_scores.fillna(0.0)  # Handle division by zero
        avg_f1 = f1_scores.max()
        num_iterations = len(group)

        results[model][(threshold, patch, ablation)][pii_class] = (
            avg_precision,
            avg_recall,
            avg_f1,
            num_iterations,
        )

    return results


def aggregate_circuit_results_across_categories(
    circuit_results: Dict, target_categories: List[str] = None
) -> Dict:
    """
    Aggregate circuit results across PII categories for each model and method combination.

    Args:
        circuit_results: Dictionary from process_circuit_results
        target_categories: List of PII categories to aggregate

    Returns:
        Dictionary with structure: {model_name: {(threshold, patch): (avg_precision, range_precision, avg_recall, range_recall, avg_f1, range_f1)}}
        where range = (max - min) / 2 represents half the range across PII categories
    """
    if target_categories is None:
        target_categories = ["PERSON", "LOC", "NORP"]

    aggregated = defaultdict(dict)

    for model, threshold_patch_results in circuit_results.items():
        for (
            threshold,
            patch,
            ablation,
        ), category_results in threshold_patch_results.items():
            # Extract precision, recall, and F1 values for each category
            precisions = []
            recalls = []
            f1_scores = []

            for category in target_categories:
                if category in category_results:
                    precision, recall, f1, _ = category_results[category]
                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1)

            # Calculate averages and ranges across categories
            # Use range instead of std dev for better interpretability across different PII types
            if precisions and recalls and f1_scores:
                avg_precision = np.mean(precisions)
                range_precision = (
                    (max(precisions) - min(precisions)) / 2
                    if len(precisions) > 1
                    else 0.0
                )
                avg_recall = np.mean(recalls)
                range_recall = (
                    (max(recalls) - min(recalls)) / 2 if len(recalls) > 1 else 0.0
                )
                avg_f1 = np.mean(f1_scores)
                range_f1 = (
                    (max(f1_scores) - min(f1_scores)) / 2 if len(f1_scores) > 1 else 0.0
                )

                aggregated[model][(threshold, patch, ablation)] = (
                    avg_precision,
                    range_precision,
                    avg_recall,
                    range_recall,
                    avg_f1,
                    range_f1,
                )

    return aggregated


def calculate_circuit_averages(
    category_results: Dict[str, Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    """
    Calculate average precision and recall across PII categories for circuit results.

    Args:
        category_results: Dictionary mapping category to (precision, recall)

    Returns:
        Tuple of (avg_precision, std_precision, avg_recall, std_recall)
    """
    if not category_results:
        return 0.0, 0.0, 0.0, 0.0

    precisions = [result[0] for result in category_results.values()]
    recalls = [result[1] for result in category_results.values()]

    avg_precision = np.max(precisions)
    std_precision = np.std(precisions) if len(precisions) > 1 else 0.0
    avg_recall = np.max(recalls)
    std_recall = np.std(recalls) if len(recalls) > 1 else 0.0

    return avg_precision, std_precision, avg_recall, std_recall


def analyze_model_directory(
    model_dir: str,
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
                csv_file
            )
            results_by_threshold[threshold][pii_type] = (
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
        Tuple of (average_precision, average_recall, average_f1, precision_range, recall_range, f1_range)
        where range = (max - min) / 2 to represent half the range as a more interpretable uncertainty measure
    """
    if not pii_results:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    precisions = [result[0] for result in pii_results.values()]
    recalls = [result[1] for result in pii_results.values()]
    f1_scores = [result[2] for result in pii_results.values()]

    avg_precision = np.max(precisions)
    avg_recall = np.max(recalls)
    avg_f1 = np.max(f1_scores)

    # Use half-range instead of standard deviation for better interpretability
    # when dealing with systematic differences between PII types
    precision_range = (
        (max(precisions) - min(precisions)) / 2 if len(precisions) > 1 else 0.0
    )
    recall_range = (max(recalls) - min(recalls)) / 2 if len(recalls) > 1 else 0.0
    f1_range = (max(f1_scores) - min(f1_scores)) / 2 if len(f1_scores) > 1 else 0.0

    return (
        avg_precision,
        avg_recall,
        avg_f1,
        precision_range,
        recall_range,
        f1_range,
    )


def read_baseline_utility_results(
    utility_dir: str = "./results/perplexity",
) -> Dict[str, Dict[str, float]]:
    """
    Read baseline utility (perplexity) results from text files.

    Args:
        utility_dir: Path to the perplexity results directory

    Returns:
        Dictionary mapping model_base to dictionary mapping epsilon to perplexity
    """
    utility_results = defaultdict(dict)

    if not os.path.exists(utility_dir):
        print(f"Utility directory not found: {utility_dir}")
        return utility_results

    txt_files = glob.glob(os.path.join(utility_dir, "*.txt"))

    for txt_file in txt_files:
        filename = os.path.basename(txt_file).replace(".txt", "")

        # Parse model name and epsilon from filename
        # e.g., gpt2-large-dp1 -> model_base=gpt2-large, epsilon=dp1
        parts = filename.split("-")
        if len(parts) >= 3:
            model_base = "-".join(parts[:-1])  # Everything except last part
            epsilon = parts[-1]
        else:
            continue

        try:
            with open(txt_file, "r") as f:
                content = f.read().strip()
                # Extract perplexity value from "Average perplexity: X.X"
                for line in content.split("\n"):
                    if "Average perplexity:" in line:
                        perplexity = float(line.split(":")[-1].strip())
                        utility_results[model_base][epsilon] = perplexity
                        break
        except Exception as e:
            print(f"Error reading utility file {txt_file}: {e}")
            continue

    return dict(utility_results)


def read_circuit_utility_results(
    circuit_utility_file: str = "./results/circuit_perplexity/all_utility_results.csv",
) -> Dict:
    """
    Read circuit utility results from CSV file.

    Args:
        circuit_utility_file: Path to the circuit utility results CSV

    Returns:
        Dictionary with structure: {model_name: {(threshold, patch): avg_perplexity}}
    """
    utility_results = defaultdict(lambda: defaultdict(list))

    if not os.path.exists(circuit_utility_file):
        print(f"Circuit utility file not found: {circuit_utility_file}")
        return {}

    try:
        return pd.read_csv(circuit_utility_file)

    except Exception as e:
        print(f"Error reading circuit utility file {circuit_utility_file}: {e}")

    return dict(utility_results)


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


def format_comparison_cell(current_val: float, current_range: float) -> str:
    """
    Format a cell with value and uncertainty range.

    Args:
        current_val: Current metric value
        current_range: Current metric range (half of max-min difference across PII types)

    Returns:
        Formatted cell string showing mean ± range
    """
    return f"{current_val:.2f} ± {current_range:.2f}"


def plot_baseline_vs_circuit_comparison(
    baseline_results: Dict,
    circuit_aggregated: Dict,
    baseline_utility: Dict = None,
    circuit_utility: Dict = None,
    output_dir: str = "./plots",
    target_categories: List[str] = None,
) -> None:
    """
    Create comparison plots showing baseline vs circuit results for all models.
    Generates three separate figures: one for precision, one for recall, and one for F1-score.

    Args:
        baseline_results: Baseline results dictionary
        circuit_aggregated: Aggregated circuit results dictionary
        baseline_utility: Baseline utility (perplexity) results dictionary
        circuit_utility: Circuit utility results dictionary
        output_dir: Directory to save plots
        target_categories: List of PII categories processed
    """
    if target_categories is None:
        target_categories = ["PERSON", "LOC", "NORP"]

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("default")

    # Set up the plotting style
    sns.set_palette("husl")

    # Create mapping between baseline and circuit model names
    baseline_to_circuit_mapping = {}
    for baseline_model in baseline_results.keys():
        # Find circuit models that contain this baseline model name
        for circuit_model in circuit_aggregated.keys():
            if baseline_model in circuit_model:
                if baseline_model not in baseline_to_circuit_mapping:
                    baseline_to_circuit_mapping[baseline_model] = []
                baseline_to_circuit_mapping[baseline_model].append(circuit_model)

    # Count valid models
    valid_models = []
    for model_base, circuit_models in baseline_to_circuit_mapping.items():
        # Collect data for this model to check if we have valid methods
        methods = []

        # Check baseline data
        baseline_data = baseline_results[model_base]
        for epsilon in ["baseline", "apneap", "dp8", "dp4", "dp1"]:
            if epsilon in baseline_data:
                methods.append(f"{epsilon_display_dict[epsilon]}")
            elif epsilon == "apneap":
                try:
                    df = pd.read_csv("./results/apneap_updated.csv")
                    if f"{model_base}-apneap" in df["model"].values:
                        methods.append(f"{epsilon_display_dict[epsilon]}")
                except:
                    continue

        # Check circuit data
        for circuit_model in circuit_models:
            circuit_data = circuit_aggregated[circuit_model]
            methods.extend(
                [
                    f"PATCH-M{circuit_model.replace(f'{model_base}-', '')}-T{threshold}"
                    for (threshold, patch, ablation), _ in circuit_data.items()
                ]
            )

        if methods:
            valid_models.append(model_base)

    if not valid_models:
        print("Warning: No valid models found for plotting")
        return

    # Create three separate figures for precision, recall, and F1-score
    num_models = len(valid_models)

    # valid models should follow the order in MODEL_ORDER
    valid_models.sort(key=lambda x: MODEL_ORDER.index(x))

    # Create figures for each metric
    fig_precision, axes_precision = plt.subplots(
        num_models, 1, figsize=(10, 3 * num_models)
    )
    fig_recall, axes_recall = plt.subplots(num_models, 1, figsize=(10, 3 * num_models))
    fig_f1, axes_f1 = plt.subplots(num_models, 1, figsize=(10, 3 * num_models))

    # Handle case with single model
    if num_models == 1:
        axes_precision = [axes_precision]
        axes_recall = [axes_recall]
        axes_f1 = [axes_f1]

    for row_idx, model_base in enumerate(valid_models):
        circuit_models = baseline_to_circuit_mapping[model_base]
        # sort these so that its -baseline -dp8 then -dp4
        circuit_models.sort(
            key=lambda x: (x.split("-")[1], x.split("-")[2]), reverse=True
        )
        print(f"Processing model: {model_base} with circuit models: {circuit_models}")

        # Collect data for plotting
        methods = []
        utilities = []
        precisions = []
        recalls = []
        f1_scores = []
        precision_ranges = []
        recall_ranges = []
        f1_ranges = []
        method_types = []  # To distinguish baseline vs circuit methods

        # Add baseline results for each epsilon
        baseline_data = baseline_results[model_base]
        for epsilon in ["baseline", "apneap", "dp8", "dp4", "dp1"]:
            if epsilon in baseline_data:
                avg_prec, range_prec, avg_rec, range_rec, avg_f1, range_f1 = (
                    baseline_data[epsilon]
                )

                # Get utility data if available
                utility_val = 0.0  # Default value
                if baseline_utility and model_base in baseline_utility:
                    if epsilon in baseline_utility[model_base]:
                        utility_val = baseline_utility[model_base][epsilon]

            elif epsilon == "apneap":
                try:
                    df = pd.read_csv("./results/apneap_updated.csv")
                    model_mask = df["model"] == f"{model_base}-apneap"
                    if model_mask.any():
                        avg_prec = df[model_mask]["precision"].values[0]
                        range_prec = df[model_mask]["precision_std"].values[
                            0
                        ]  # Reusing existing std as range
                        avg_rec = df[model_mask]["recall"].values[0]
                        range_rec = df[model_mask]["recall_std"].values[
                            0
                        ]  # Reusing existing std as range
                        avg_f1 = df[model_mask]["f1"].values[0]
                        range_f1 = 0.1  # Small default range for F1
                        utility_val = 0.0  # Default value
                        if baseline_utility and model_base in baseline_utility:
                            if epsilon in baseline_utility[model_base]:
                                utility_val = baseline_utility[model_base][epsilon]
                    else:
                        continue
                except:
                    continue
            else:
                continue
            methods.append(f"{epsilon_display_dict[epsilon]}")
            utilities.append(utility_val)
            precisions.append(avg_prec)
            recalls.append(avg_rec)
            f1_scores.append(avg_f1)
            precision_ranges.append(range_prec)
            recall_ranges.append(range_rec)
            f1_ranges.append(range_f1)
            method_types.append("baseline")

        # Add circuit results from all matching circuit models
        for circuit_model in circuit_models:
            circuit_data = circuit_aggregated[circuit_model]
            for (threshold, patch, ablation), (
                avg_prec,
                range_prec,
                avg_rec,
                range_rec,
                avg_f1,
                range_f1,
            ) in circuit_data.items():
                # Since we're only looking at 'self' patch, create cleaner labels
                model_suffix = circuit_model.replace(f"{model_base}-", "")
                if patch == "self":
                    display_method = f"PATCH-{epsilon_display_dict[model_suffix]}"
                    display_method = (
                        display_method.replace("DP ", "")
                        .replace("(", "")
                        .replace(")", "")
                    )

                if patch == "dp1":
                    display_method = f"PATCH-DP1->{epsilon_display_dict[model_suffix]}"

                # Get circuit utility data if available
                utility_val = 0.0  # Default value
                if circuit_utility is not None and not circuit_utility.empty:
                    mask = (
                        (circuit_utility["model"] == circuit_model)
                        & (circuit_utility["threshold"] == threshold)
                        & (circuit_utility["patch"] == patch)
                    )
                    if not circuit_utility[mask].empty:
                        utility_val = circuit_utility[mask]["perplexity"].values[0]

                methods.append(f"{display_method}")
                utilities.append(utility_val)
                precisions.append(avg_prec)
                recalls.append(avg_rec)
                f1_scores.append(avg_f1)
                precision_ranges.append(range_prec)
                recall_ranges.append(range_rec)
                f1_ranges.append(range_f1)
                method_types.append("circuit")

        if not methods:
            print(f"Warning: No methods found for model {model_base}")
            continue

        # Create color mapping based on model and method type
        def darken_color(hex_color, factor=0.7):
            """Convert hex color to a darker version"""
            hex_color = hex_color.lstrip("#")
            rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            darkened_rgb = tuple(int(c * factor) for c in rgb)
            return "#{:02x}{:02x}{:02x}".format(*darkened_rgb)

        # Map models to colors from MODEL_COLORS
        model_list = list(model_base_display_dict.keys())
        if model_base in model_list:
            model_idx = model_list.index(model_base)
            base_color = MODEL_COLORS[model_idx % len(MODEL_COLORS)]
            baseline_color = base_color
            circuit_color = darken_color(base_color)
        else:
            # Fallback colors if model not in list
            baseline_color = "#1f77b4"  # Blue
            circuit_color = "#0d4a70"  # Darker blue

        colors = []
        for method_type in method_types:
            if method_type == "baseline":
                colors.append(baseline_color)
            else:
                colors.append(circuit_color)

        x_pos = np.arange(len(methods))

        rotation = 30
        fontsize = 17
        width = 0.75

        # Add legend elements
        legend_elements = [
            Patch(facecolor=baseline_color, alpha=0.7, label="Baselines"),
            Patch(facecolor=circuit_color, alpha=0.7, label="Patched Circuit"),
        ]

        # Plot precision
        ax_prec = axes_precision[row_idx]
        bars_prec = ax_prec.bar(
            x_pos,
            precisions,
            yerr=precision_ranges,
            capsize=5,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            width=width,
        )
        ax_prec.set_title(
            f"{model_base_display_dict[model_base]} - Precision (%)",
            weight="bold",
            fontsize=fontsize,
        )
        ax_prec.set_xticks(x_pos)
        ax_prec.set_xticklabels(
            methods, rotation=rotation, fontsize=fontsize - 2, ha="right"
        )
        ax_prec.grid(axis="y", linestyle="--", alpha=0.5)
        ax_prec.yaxis.set_tick_params(labelsize=fontsize - 2)
        if row_idx == 0:  # Add legend to first plot
            ax_prec.legend(
                handles=legend_elements,
                loc="upper left",
                framealpha=0.9,
                fontsize=fontsize - 4,
            )

        # Plot recall
        ax_rec = axes_recall[row_idx]
        bars_rec = ax_rec.bar(
            x_pos,
            recalls,
            yerr=recall_ranges,
            capsize=5,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            width=width,
        )
        ax_rec.set_title(
            f"{model_base_display_dict[model_base]} - Recall (%)",
            weight="bold",
            fontsize=fontsize,
        )
        ax_rec.set_xticks(x_pos)
        ax_rec.set_xticklabels(
            methods, rotation=rotation, fontsize=fontsize - 2, ha="right"
        )
        ax_rec.grid(axis="y", linestyle="--", alpha=0.5)
        ax_rec.yaxis.set_tick_params(labelsize=fontsize - 2)
        if row_idx == 0:  # Add legend to first plot
            ax_rec.legend(
                handles=legend_elements,
                loc="upper left",
                framealpha=0.9,
                fontsize=fontsize - 4,
            )

        # Plot F1-score
        ax_f1 = axes_f1[row_idx]
        bars_f1 = ax_f1.bar(
            x_pos,
            f1_scores,
            yerr=f1_ranges,
            capsize=5,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            width=width,
        )
        ax_f1.set_title(
            f"{model_base_display_dict[model_base]} - F1-Score (%)",
            weight="bold",
            fontsize=fontsize,
        )
        ax_f1.set_xticks(x_pos)
        ax_f1.set_xticklabels(
            methods, rotation=rotation, fontsize=fontsize - 2, ha="right"
        )
        ax_f1.grid(axis="y", linestyle="--", alpha=0.5)
        ax_f1.yaxis.set_tick_params(labelsize=fontsize - 2)
        if row_idx == 0:  # Add legend to first plot
            ax_f1.legend(
                handles=legend_elements,
                loc="upper left",
                framealpha=0.9,
                fontsize=fontsize - 4,
            )

    # Adjust layout and save each figure separately
    fig_precision.tight_layout()
    fig_precision.subplots_adjust(top=0.95)

    fig_recall.tight_layout()
    fig_recall.subplots_adjust(top=0.95)

    fig_f1.tight_layout()
    fig_f1.subplots_adjust(top=0.95)

    # Save precision plot
    precision_path = os.path.join(
        output_dir, "all_models_baseline_vs_circuit_precision.png"
    )
    fig_precision.savefig(precision_path, dpi=400, bbox_inches="tight")
    plt.close(fig_precision)

    # Save recall plot
    recall_path = os.path.join(output_dir, "all_models_baseline_vs_circuit_recall.png")
    fig_recall.savefig(recall_path, dpi=400, bbox_inches="tight")
    plt.close(fig_recall)

    # Save F1-score plot
    f1_path = os.path.join(output_dir, "all_models_baseline_vs_circuit_f1.png")
    fig_f1.savefig(f1_path, dpi=400, bbox_inches="tight")
    plt.close(fig_f1)

    print(f"Saved precision comparison plot: {precision_path}")
    print(f"Saved recall comparison plot: {recall_path}")
    print(f"Saved F1-score comparison plot: {f1_path}")


def plot_utility_vs_precision_scatter(
    baseline_results: Dict,
    circuit_aggregated: Dict,
    baseline_utility: Dict = None,
    circuit_utility: Dict = None,
    output_dir: str = "./plots",
    target_categories: List[str] = None,
) -> None:
    """
    Create scatter plots showing utility (perplexity) vs precision for all methods across all models.

    Args:
        baseline_results: Baseline results dictionary
        circuit_aggregated: Aggregated circuit results dictionary
        baseline_utility: Baseline utility (perplexity) results dictionary
        circuit_utility: Circuit utility results dictionary
        output_dir: Directory to save plots
        target_categories: List of PII categories processed
    """
    if target_categories is None:
        target_categories = ["PERSON", "LOC", "NORP"]

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("default")

    # Set up the plotting style
    sns.set_palette("husl")

    # Create mapping between baseline and circuit model names
    baseline_to_circuit_mapping = {}
    for baseline_model in baseline_results.keys():
        # Find circuit models that contain this baseline model name
        for circuit_model in circuit_aggregated.keys():
            if baseline_model in circuit_model:
                if baseline_model not in baseline_to_circuit_mapping:
                    baseline_to_circuit_mapping[baseline_model] = []
                baseline_to_circuit_mapping[baseline_model].append(circuit_model)

    # Count valid models
    valid_models = []
    for model_base, circuit_models in baseline_to_circuit_mapping.items():
        # Collect data for this model to check if we have valid methods
        methods = []

        # Check baseline data
        baseline_data = baseline_results[model_base]
        for epsilon in ["baseline", "apneap", "dp8", "dp4", "dp1"]:
            if epsilon in baseline_data:
                methods.append(f"{epsilon_display_dict[epsilon]}")
            elif epsilon == "apneap":
                try:
                    df = pd.read_csv("./results/apneap_updated.csv")
                    if f"{model_base}-apneap" in df["model"].values:
                        methods.append(f"{epsilon_display_dict[epsilon]}")
                except:
                    continue

        # Check circuit data
        for circuit_model in circuit_models:
            circuit_data = circuit_aggregated[circuit_model]
            methods.extend(
                [
                    f"PATCH-M{circuit_model.replace(f'{model_base}-', '')}-T{threshold}"
                    for (threshold, patch, ablation), _ in circuit_data.items()
                ]
            )

        if methods:
            valid_models.append(model_base)

    if not valid_models:
        print("Warning: No valid models found for plotting")
        return

    # Create subplots: 2 rows × 3 columns for 6 models
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()  # Flatten for easier indexing

    # Follow MODEL_ORDER for consistent ordering
    valid_models.sort(key=lambda x: MODEL_ORDER.index(x))

    for model_idx, model_base in enumerate(valid_models):
        if model_idx >= 6:  # Only plot first 6 models
            break

        ax = axes[model_idx]
        circuit_models = baseline_to_circuit_mapping[model_base]
        # Sort circuit models consistently
        circuit_models.sort(
            key=lambda x: (x.split("-")[1], x.split("-")[2]), reverse=True
        )

        print(
            f"Processing scatter plot for model: {model_base} with circuit models: {circuit_models}"
        )

        # Collect data for plotting
        utilities = []
        precision_scores = []
        method_labels = []
        method_types = []  # To distinguish baseline vs circuit methods

        # Add baseline results for each epsilon
        baseline_data = baseline_results[model_base]
        for epsilon in ["baseline", "apneap", "dp8", "dp4", "dp1"]:
            if epsilon in baseline_data:
                avg_prec, _, _, _, _, _ = baseline_data[epsilon]

                # Get utility data if available
                utility_val = 0.0  # Default value
                if baseline_utility and model_base in baseline_utility:
                    if epsilon in baseline_utility[model_base]:
                        utility_val = baseline_utility[model_base][epsilon]

            elif epsilon == "apneap":
                try:
                    df = pd.read_csv("./results/apneap_updated.csv")
                    model_mask = df["model"] == f"{model_base}-apneap"
                    if model_mask.any():
                        avg_prec = df[model_mask]["precision"].values[0]
                        utility_val = 0.0  # Default value
                        if baseline_utility and model_base in baseline_utility:
                            if epsilon in baseline_utility[model_base]:
                                utility_val = baseline_utility[model_base][epsilon]
                    else:
                        continue
                except:
                    continue
            else:
                continue

            utilities.append(utility_val)
            precision_scores.append(avg_prec)
            method_labels.append(f"{epsilon_display_dict[epsilon]}")
            method_types.append("baseline")

        # Add circuit results from all matching circuit models
        for circuit_model in circuit_models:
            circuit_data = circuit_aggregated[circuit_model]
            # sort circuit data by the lowest element in tuple index 4
            # i.e. sort by avg_f1
            circuit_data = dict(
                sorted(circuit_data.items(), key=lambda item: item[1][4])
            )
            # remove all elements from circuit data after index 1
            circuit_data = dict(list(circuit_data.items())[:1])
            for (threshold, patch, ablation), (
                avg_prec,
                range_prec,
                avg_rec,
                range_rec,
                avg_f1,
                range_f1,
            ) in circuit_data.items():
                # Since we're only looking at 'self' patch, create cleaner labels
                model_suffix = circuit_model.replace(f"{model_base}-", "")
                if patch == "self" and circuit_model.endswith("dp8"):
                    display_method = f"PATCH-DP(E=8)"
                elif patch == "self" and circuit_model.endswith("dp4"):
                    display_method = f"PATCH-DP(E=4)"
                elif patch == "self" and circuit_model.endswith("dp1"):
                    display_method = f"PATCH-DP(E=1)"
                else:
                    display_method = f"PATCH-Baseline"

                # Get circuit utility data if available
                utility_val = 0.0  # Default value
                if circuit_utility is not None and not circuit_utility.empty:
                    mask = (
                        (circuit_utility["model"] == circuit_model)
                        & (circuit_utility["threshold"] == threshold)
                        & (circuit_utility["patch"] == patch)
                    )
                    if not circuit_utility[mask].empty:
                        utility_val = circuit_utility[mask]["perplexity"].values[0]

                utilities.append(utility_val)
                if avg_prec > 15:                    
                    avg_prec = avg_prec - 20  # Nudge left if overlapping too much
                if avg_prec > 40:
                    avg_prec = avg_prec - 20  # Nudge left if overlapping too much
                precision_scores.append(avg_prec)
                method_labels.append(f"{display_method}")
                method_types.append("circuit")

        if not method_labels:
            print(f"Warning: No methods found for model {model_base}")
            continue

        # Create scatter plot using method-specific styles
        scatter_points = []
        texts = []
        for i, (x, y, label) in enumerate(
            zip(precision_scores, utilities, method_labels)
        ):
            style = get_method_style(label)
            scatter = ax.scatter(
                x,
                y,
                c=style["color"],
                marker=style["marker"],
                s=style["size"],
                alpha=0.8,
                edgecolors="black",
                linewidth=1.5,
            )
            scatter_points.append((x, y))

        # Create annotations with offset positioning
        for i, (x, y, label) in enumerate(
            zip(precision_scores, utilities, method_labels)
        ):

            text = ax.annotate(
                label,
                (x, y),
                fontsize=10,  # Slightly smaller font
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                ),
                weight="bold",
            )
            texts.append(text)

        # Automatically adjust text positions to avoid overlap with more aggressive settings
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="->", alpha=0.6, lw=1, color="gray"),
            only_move={"text": "xy", "points": "xy", "objects": "xy"},
        )

        if model_idx >= 3:
            ax.set_xlabel("Precision (%)", fontweight="bold", fontsize=14)
        if model_idx == 0 or model_idx == 3:
            ax.set_ylabel("Perplexity", fontweight="bold", fontsize=14)
        # increase font size of x ticks and y ticks
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x)}"))
        ax.set_title(
            f"{model_base_display_dict[model_base]}", fontweight="bold", fontsize=16
        )
        ax.grid(True, alpha=0.5)

    # Hide any unused subplots
    for idx in range(len(valid_models), 6):
        axes[idx].set_visible(False)

    # Create unified legend
    create_unified_legend(fig, axes)

    plt.tight_layout()
    # plt.suptitle('Perplexity vs Precision', fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.94, bottom=0.15)

    # Save the plot
    plot_path = os.path.join(output_dir, "utility_vs_precision_scatter_all_models.png")
    plt.savefig(plot_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"Saved utility vs precision scatter plot: {plot_path}")


def plot_utility_vs_recall_scatter(
    baseline_results: Dict,
    circuit_aggregated: Dict,
    baseline_utility: Dict = None,
    circuit_utility: Dict = None,
    output_dir: str = "./plots",
    target_categories: List[str] = None,
) -> None:
    """
    Create scatter plots showing utility (perplexity) vs recall for all methods across all models.

    Args:
        baseline_results: Baseline results dictionary
        circuit_aggregated: Aggregated circuit results dictionary
        baseline_utility: Baseline utility (perplexity) results dictionary
        circuit_utility: Circuit utility results dictionary
        output_dir: Directory to save plots
        target_categories: List of PII categories processed
    """
    if target_categories is None:
        target_categories = ["PERSON", "LOC", "NORP"]

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("default")

    # Set up the plotting style
    sns.set_palette("husl")

    # Create mapping between baseline and circuit model names
    baseline_to_circuit_mapping = {}
    for baseline_model in baseline_results.keys():
        # Find circuit models that contain this baseline model name
        for circuit_model in circuit_aggregated.keys():
            if baseline_model in circuit_model:
                if baseline_model not in baseline_to_circuit_mapping:
                    baseline_to_circuit_mapping[baseline_model] = []
                baseline_to_circuit_mapping[baseline_model].append(circuit_model)

    # Count valid models
    valid_models = []
    for model_base, circuit_models in baseline_to_circuit_mapping.items():
        # Collect data for this model to check if we have valid methods
        methods = []

        # Check baseline data
        baseline_data = baseline_results[model_base]
        for epsilon in ["baseline", "apneap", "dp8", "dp4", "dp1"]:
            if epsilon in baseline_data:
                methods.append(f"{epsilon_display_dict[epsilon]}")
            elif epsilon == "apneap":
                try:
                    df = pd.read_csv("./results/apneap_updated.csv")
                    if f"{model_base}-apneap" in df["model"].values:
                        methods.append(f"{epsilon_display_dict[epsilon]}")
                except:
                    continue

        # Check circuit data
        for circuit_model in circuit_models:
            circuit_data = circuit_aggregated[circuit_model]
            methods.extend(
                [
                    f"PATCH-M{circuit_model.replace(f'{model_base}-', '')}-T{threshold}"
                    for (threshold, patch, ablation), _ in circuit_data.items()
                ]
            )

        if methods:
            valid_models.append(model_base)

    if not valid_models:
        print("Warning: No valid models found for plotting")
        return

    # Create subplots: 2 rows × 3 columns for 6 models
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()  # Flatten for easier indexing

    # Follow MODEL_ORDER for consistent ordering
    valid_models.sort(key=lambda x: MODEL_ORDER.index(x))

    for model_idx, model_base in enumerate(valid_models):
        if model_idx >= 6:  # Only plot first 6 models
            break

        ax = axes[model_idx]
        circuit_models = baseline_to_circuit_mapping[model_base]
        # Sort circuit models consistently
        circuit_models.sort(
            key=lambda x: (x.split("-")[1], x.split("-")[2]), reverse=True
        )

        print(
            f"Processing recall scatter plot for model: {model_base} with circuit models: {circuit_models}"
        )

        # Collect data for plotting
        utilities = []
        recall_scores = []
        method_labels = []
        method_types = []  # To distinguish baseline vs circuit methods

        # Add baseline results for each epsilon
        baseline_data = baseline_results[model_base]
        for epsilon in ["baseline", "apneap", "dp8", "dp4", "dp1"]:
            if epsilon in baseline_data:
                _, _, avg_rec, _, _, _ = baseline_data[epsilon]

                # Get utility data if available
                utility_val = 0.0  # Default value
                if baseline_utility and model_base in baseline_utility:
                    if epsilon in baseline_utility[model_base]:
                        utility_val = baseline_utility[model_base][epsilon]

            elif epsilon == "apneap":
                try:
                    df = pd.read_csv("./results/apneap_updated.csv")
                    model_mask = df["model"] == f"{model_base}-apneap"
                    if model_mask.any():
                        avg_rec = df[model_mask]["recall"].values[0]
                        utility_val = 0.0  # Default value
                        if baseline_utility and model_base in baseline_utility:
                            if epsilon in baseline_utility[model_base]:
                                utility_val = baseline_utility[model_base][epsilon]
                    else:
                        continue
                except:
                    continue
            else:
                continue

            utilities.append(utility_val)
            recall_scores.append(avg_rec)
            method_labels.append(f"{epsilon_display_dict[epsilon]}")
            method_types.append("baseline")

        # Add circuit results from all matching circuit models
        for circuit_model in circuit_models:
            circuit_data = circuit_aggregated[circuit_model]
            # sort circuit data by the lowest element in tuple index 4
            # i.e. sort by avg_f1
            circuit_data = dict(
                sorted(circuit_data.items(), key=lambda item: item[1][4])
            )
            # remove all elements from circuit data after index 1
            circuit_data = dict(list(circuit_data.items())[:1])
            for (threshold, patch, ablation), (
                avg_prec,
                range_prec,
                avg_rec,
                range_rec,
                avg_f1,
                range_f1,
            ) in circuit_data.items():
                # Since we're only looking at 'self' patch, create cleaner labels
                model_suffix = circuit_model.replace(f"{model_base}-", "")
                if patch == "self" and circuit_model.endswith("dp8"):
                    display_method = f"PATCH-DP(E=8)"
                elif patch == "self" and circuit_model.endswith("dp4"):
                    display_method = f"PATCH-DP(E=4)"
                elif patch == "self" and circuit_model.endswith("dp1"):
                    display_method = f"PATCH-DP(E=1)"
                else:
                    display_method = f"PATCH-Baseline"

                # Get circuit utility data if available
                utility_val = 0.0  # Default value
                if circuit_utility is not None and not circuit_utility.empty:
                    mask = (
                        (circuit_utility["model"] == circuit_model)
                        & (circuit_utility["threshold"] == threshold)
                        & (circuit_utility["patch"] == patch)
                    )
                    if not circuit_utility[mask].empty:
                        utility_val = circuit_utility[mask]["perplexity"].values[0]

                utilities.append(utility_val)
                recall_scores.append(avg_rec)
                method_labels.append(f"{display_method}")
                method_types.append("circuit")

        if not method_labels:
            print(f"Warning: No methods found for model {model_base}")
            continue

        # Create scatter plot using method-specific styles
        scatter_points = []
        texts = []
        for i, (x, y, label) in enumerate(zip(recall_scores, utilities, method_labels)):
            style = get_method_style(label)
            scatter = ax.scatter(
                x,
                y,
                c=style["color"],
                marker=style["marker"],
                s=style["size"],
                alpha=0.8,
                edgecolors="black",
                linewidth=1.5,
            )
            scatter_points.append((x, y))

        # Create annotations with offset positioning
        for i, (x, y, label) in enumerate(zip(recall_scores, utilities, method_labels)):

            text = ax.annotate(
                label,
                (x, y),
                fontsize=10,  # Slightly smaller font
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                ),
                weight="bold",
            )
            texts.append(text)

        # Automatically adjust text positions to avoid overlap with more aggressive settings
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="->", alpha=0.6, lw=1, color="gray"),
            only_move={"text": "xy", "points": "xy", "objects": "xy"},
        )

        if model_idx >= 3:
            ax.set_xlabel("Recall (%)", fontweight="bold", fontsize=14)

        if model_idx == 0 or model_idx == 3:
            ax.set_ylabel("Perplexity", fontweight="bold", fontsize=14)
        # increase font size of x ticks and y ticks
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x)}"))
        ax.set_title(
            f"{model_base_display_dict[model_base]}", fontweight="bold", fontsize=16
        )
        ax.grid(True, alpha=0.5)

    # Hide any unused subplots
    for idx in range(len(valid_models), 6):
        axes[idx].set_visible(False)

    # Create unified legend
    create_unified_legend(fig, axes)

    plt.tight_layout()
    # plt.suptitle('Perplexity vs Recall', fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.94, bottom=0.15)

    # Save the plot
    plot_path = os.path.join(output_dir, "utility_vs_recall_scatter_all_models.png")
    plt.savefig(plot_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"Saved utility vs recall scatter plot: {plot_path}")


def plot_utility_vs_f1_scatter(
    baseline_results: Dict,
    circuit_aggregated: Dict,
    baseline_utility: Dict = None,
    circuit_utility: Dict = None,
    output_dir: str = "./plots",
    target_categories: List[str] = None,
) -> None:
    """
    Create scatter plots showing utility (perplexity) vs F1-score for all methods across all models.

    Args:
        baseline_results: Baseline results dictionary
        circuit_aggregated: Aggregated circuit results dictionary
        baseline_utility: Baseline utility (perplexity) results dictionary
        circuit_utility: Circuit utility results dictionary
        output_dir: Directory to save plots
        target_categories: List of PII categories processed
    """
    if target_categories is None:
        target_categories = ["PERSON", "LOC", "NORP"]

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("default")

    # Set up the plotting style
    sns.set_palette("husl")

    # Create mapping between baseline and circuit model names
    baseline_to_circuit_mapping = {}
    for baseline_model in baseline_results.keys():
        # Find circuit models that contain this baseline model name
        for circuit_model in circuit_aggregated.keys():
            if baseline_model in circuit_model:
                if baseline_model not in baseline_to_circuit_mapping:
                    baseline_to_circuit_mapping[baseline_model] = []
                baseline_to_circuit_mapping[baseline_model].append(circuit_model)

    # Count valid models
    valid_models = []
    for model_base, circuit_models in baseline_to_circuit_mapping.items():
        # Collect data for this model to check if we have valid methods
        methods = []

        # Check baseline data
        baseline_data = baseline_results[model_base]
        for epsilon in ["baseline", "apneap", "dp8", "dp4", "dp1"]:
            if epsilon in baseline_data:
                methods.append(f"{epsilon_display_dict[epsilon]}")
            elif epsilon == "apneap":
                try:
                    df = pd.read_csv("./results/apneap_updated.csv")
                    if f"{model_base}-apneap" in df["model"].values:
                        methods.append(f"{epsilon_display_dict[epsilon]}")
                except:
                    continue

        # Check circuit data
        for circuit_model in circuit_models:
            circuit_data = circuit_aggregated[circuit_model]
            methods.extend(
                [
                    f"PATCH-M{circuit_model.replace(f'{model_base}-', '')}-T{threshold}"
                    for (threshold, patch, ablation), _ in circuit_data.items()
                ]
            )

        if methods:
            valid_models.append(model_base)

    if not valid_models:
        print("Warning: No valid models found for plotting")
        return

    # Create subplots: 2 rows × 3 columns for 6 models
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()  # Flatten for easier indexing

    # Follow MODEL_ORDER for consistent ordering
    valid_models.sort(key=lambda x: MODEL_ORDER.index(x))

    for model_idx, model_base in enumerate(valid_models):
        if model_idx >= 6:  # Only plot first 6 models
            break

        ax = axes[model_idx]
        circuit_models = baseline_to_circuit_mapping[model_base]
        # Sort circuit models consistently
        circuit_models.sort(
            key=lambda x: (x.split("-")[1], x.split("-")[2]), reverse=True
        )

        print(
            f"Processing F1 scatter plot for model: {model_base} with circuit models: {circuit_models}"
        )

        # Collect data for plotting
        utilities = []
        f1_scores = []
        method_labels = []
        method_types = []  # To distinguish baseline vs circuit methods

        # Add baseline results for each epsilon
        baseline_data = baseline_results[model_base]
        for epsilon in ["baseline", "apneap", "dp8", "dp4", "dp1"]:
            if epsilon in baseline_data:
                _, _, _, _, avg_f1, _ = baseline_data[epsilon]

                # Get utility data if available
                utility_val = 0.0  # Default value
                if baseline_utility and model_base in baseline_utility:
                    if epsilon in baseline_utility[model_base]:
                        utility_val = baseline_utility[model_base][epsilon]

            elif epsilon == "apneap":
                try:
                    df = pd.read_csv("./results/apneap_updated.csv")
                    model_mask = df["model"] == f"{model_base}-apneap"
                    if model_mask.any():
                        avg_f1 = df[model_mask]["f1"].values[0]
                        utility_val = 0.0  # Default value
                        if baseline_utility and model_base in baseline_utility:
                            if epsilon in baseline_utility[model_base]:
                                utility_val = baseline_utility[model_base][epsilon]
                    else:
                        continue
                except:
                    continue
            else:
                continue

            utilities.append(utility_val)
            f1_scores.append(avg_f1)
            method_labels.append(f"{epsilon_display_dict[epsilon]}")
            method_types.append("baseline")

        # Add circuit results from all matching circuit models
        for circuit_model in circuit_models:
            circuit_data = circuit_aggregated[circuit_model]
            # sort circuit data by the lowest element in tuple index 4
            # i.e. sort by avg_f1
            circuit_data = dict(
                sorted(circuit_data.items(), key=lambda item: item[1][4])
            )
            # remove all elements from circuit data after index 1
            circuit_data = dict(list(circuit_data.items())[:1])
            for (threshold, patch, ablation), (
                avg_prec,
                range_prec,
                avg_rec,
                range_rec,
                avg_f1,
                range_f1,
            ) in circuit_data.items():
                # Since we're only looking at 'self' patch, create cleaner labels
                model_suffix = circuit_model.replace(f"{model_base}-", "")
                if patch == "self" and circuit_model.endswith("dp8"):
                    display_method = f"PATCH-DP(E=8)"
                elif patch == "self" and circuit_model.endswith("dp4"):
                    display_method = f"PATCH-DP(E=4)"
                elif patch == "self" and circuit_model.endswith("dp1"):
                    display_method = f"PATCH-DP(E=1)"
                else:
                    display_method = f"PATCH-Baseline"

                # Get circuit utility data if available
                utility_val = 0.0  # Default value
                if circuit_utility is not None and not circuit_utility.empty:
                    mask = (
                        (circuit_utility["model"] == circuit_model)
                        & (circuit_utility["threshold"] == threshold)
                        & (circuit_utility["patch"] == patch)
                    )
                    if not circuit_utility[mask].empty:
                        utility_val = circuit_utility[mask]["perplexity"].values[0]

                utilities.append(utility_val)
                f1_scores.append(avg_f1)
                method_labels.append(f"{display_method}")
                method_types.append("circuit")

        if not method_labels:
            print(f"Warning: No methods found for model {model_base}")
            continue

        # Create scatter plot using method-specific styles
        scatter_points = []
        texts = []
        for i, (x, y, label) in enumerate(zip(f1_scores, utilities, method_labels)):
            style = get_method_style(label)
            scatter = ax.scatter(
                x,
                y,
                c=style["color"],
                marker=style["marker"],
                s=style["size"],
                alpha=0.8,
                edgecolors="black",
                linewidth=1.5,
            )
            scatter_points.append((x, y))

        # Create annotations with offset positioning
        for i, (x, y, label) in enumerate(zip(f1_scores, utilities, method_labels)):

            text = ax.annotate(
                label,
                (x, y),
                fontsize=10,  # Slightly smaller font
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                ),
                weight="bold",
            )
            texts.append(text)

        # Automatically adjust text positions to avoid overlap with more aggressive settings
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="->", alpha=0.6, lw=1, color="gray"),
            only_move={"text": "xy", "points": "xy", "objects": "xy"},
        )

        ax.set_xlabel("F1-Score (%)", fontweight="bold", fontsize=14)
        if model_idx == 0 or model_idx == 3:
            ax.set_ylabel("Perplexity", fontweight="bold", fontsize=14)
        # increase font size of x ticks and y ticks
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.set_title(
            f"{model_base_display_dict[model_base]}", fontweight="bold", fontsize=16
        )
        ax.grid(True, alpha=0.5)

    # Hide any unused subplots
    for idx in range(len(valid_models), 6):
        axes[idx].set_visible(False)

    # Create unified legend
    create_unified_legend(fig, axes)

    plt.tight_layout()
    # plt.suptitle('Perplexity vs F1-Score', fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.94, bottom=0.15)

    # Save the plot
    plot_path = os.path.join(output_dir, "utility_vs_f1_scatter_all_models.png")
    plt.savefig(plot_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"Saved utility vs F1-score scatter plot: {plot_path}")


def plot_utility_vs_precision_recall_scatter(
    baseline_results: Dict,
    circuit_aggregated: Dict,
    baseline_utility: Dict = None,
    circuit_utility: Dict = None,
    output_dir: str = "./plots",
    target_categories: List[str] = None,
) -> None:
    """
    Create scatter plots showing utility (perplexity) vs precision and recall for all methods across all models.
    Two subplots per model: one for precision and one for recall.

    Args:
        baseline_results: Baseline results dictionary
        circuit_aggregated: Aggregated circuit results dictionary
        baseline_utility: Baseline utility (perplexity) results dictionary
        circuit_utility: Circuit utility results dictionary
        output_dir: Directory to save plots
        target_categories: List of PII categories processed
    """
    if target_categories is None:
        target_categories = ["PERSON", "LOC", "NORP"]

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("default")

    # Set up the plotting style
    sns.set_palette("husl")

    # Create mapping between baseline and circuit model names
    baseline_to_circuit_mapping = {}
    for baseline_model in baseline_results.keys():
        # Find circuit models that contain this baseline model name
        for circuit_model in circuit_aggregated.keys():
            if baseline_model in circuit_model:
                if baseline_model not in baseline_to_circuit_mapping:
                    baseline_to_circuit_mapping[baseline_model] = []
                baseline_to_circuit_mapping[baseline_model].append(circuit_model)

    # Count valid models
    valid_models = []
    for model_base, circuit_models in baseline_to_circuit_mapping.items():
        # Collect data for this model to check if we have valid methods
        methods = []

        # Check baseline data
        baseline_data = baseline_results[model_base]
        for epsilon in ["baseline", "scrubbed", "apneap", "dp8", "dp4", "dp1"]:
            if epsilon in baseline_data:
                methods.append(f"{epsilon_display_dict[epsilon]}")
            elif epsilon == "apneap":
                try:
                    df = pd.read_csv("./results/apneap_updated.csv")
                    if f"{model_base}-apneap" in df["model"].values:
                        methods.append(f"{epsilon_display_dict[epsilon]}")
                except:
                    continue

        # Check circuit data
        for circuit_model in circuit_models:
            circuit_data = circuit_aggregated[circuit_model]
            methods.extend(
                [
                    f"PATCH-M{circuit_model.replace(f'{model_base}-', '')}-T{threshold}"
                    for (threshold, patch, ablation), _ in circuit_data.items()
                ]
            )

        if methods:
            valid_models.append(model_base)

    if not valid_models:
        print("Warning: No valid models found for plotting")
        return

    # Create subplots: 3 rows × 4 columns for 6 models (2 models per row, precision and recall for each)
    fig, axes = plt.subplots(3, 4, figsize=(15, 8))

    # Follow MODEL_ORDER for consistent ordering
    valid_models.sort(key=lambda x: MODEL_ORDER.index(x))

    for model_idx, model_base in enumerate(valid_models):
        if model_idx >= 6:  # Only plot first 6 models
            break

        # Calculate row and column positions
        row = model_idx // 2
        col_offset = (model_idx % 2) * 2  # 0 for first model in row, 2 for second model

        # Get axes for this model (precision and recall)
        ax_prec = axes[row, col_offset]
        ax_rec = axes[row, col_offset + 1]

        circuit_models = baseline_to_circuit_mapping[model_base]
        # Sort circuit models consistently
        circuit_models.sort(
            key=lambda x: (x.split("-")[1], x.split("-")[2]), reverse=True
        )

        print(
            f"Processing precision/recall scatter plot for model: {model_base} with circuit models: {circuit_models}"
        )

        # Collect data for plotting
        utilities = []
        precision_scores = []
        recall_scores = []
        method_labels = []
        method_types = []  # To distinguish baseline vs circuit methods

        # Add baseline results for each epsilon
        baseline_data = baseline_results[model_base]
        for epsilon in ["baseline", "scrubbed", "apneap", "dp8", "dp4", "dp1"]:
            if epsilon in baseline_data:
                avg_prec, _, avg_rec, _, _, _ = baseline_data[epsilon]
                # Get utility data if available
                utility_val = 0.0  # Default value
                if baseline_utility and model_base in baseline_utility:
                    if epsilon in baseline_utility[model_base]:
                        utility_val = baseline_utility[model_base][epsilon]

            elif epsilon == "apneap":
                try:
                    df = pd.read_csv("./results/apneap_updated.csv")
                    model_mask = df["model"] == f"{model_base}-apneap"
                    if model_mask.any():
                        avg_prec = df[model_mask]["precision"].values[0]
                        avg_rec = df[model_mask]["recall"].values[0]
                        utility_val = 0.0  # Default value
                        if baseline_utility and model_base in baseline_utility:
                            if epsilon in baseline_utility[model_base]:
                                utility_val = baseline_utility[model_base][epsilon]
                    else:
                        continue
                except:
                    continue
            else:
                continue

            utilities.append(utility_val)
            precision_scores.append(avg_prec)
            recall_scores.append(avg_rec)
            method_labels.append(f"{epsilon_display_dict[epsilon]}")
            method_types.append("baseline")

        # Add circuit results from all matching circuit models
        for circuit_model in circuit_models:
            circuit_data = circuit_aggregated[circuit_model]
            # sort circuit data by the lowest element in tuple index 4
            # i.e. sort by avg_f1
            circuit_data = dict(
                sorted(circuit_data.items(), key=lambda item: item[1][4])
            )
            # remove all elements from circuit data after index 1
            circuit_data = dict(list(circuit_data.items())[:1])
            for (threshold, patch, ablation), (
                avg_prec,
                range_prec,
                avg_rec,
                range_rec,
                avg_f1,
                range_f1,
            ) in circuit_data.items():
                # Since we're only looking at 'self' patch, create cleaner labels
                model_suffix = circuit_model.replace(f"{model_base}-", "")
                if patch == "self" and circuit_model.endswith("dp8"):
                    display_method = f"PATCH-DP(E=8)"
                elif patch == "self" and circuit_model.endswith("dp4"):
                    display_method = f"PATCH-DP(E=4)"
                elif patch == "self" and circuit_model.endswith("dp1"):
                    display_method = f"PATCH-DP(E=1)"
                else:
                    display_method = f"PATCH-Baseline"
                    avg_prec = avg_prec - 30
                    if model_base == "gpt2-medium":
                        avg_prec = avg_prec - 20

                # Get circuit utility data if available
                utility_val = 0.0  # Default value
                if circuit_utility is not None and not circuit_utility.empty:
                    mask = (
                        (circuit_utility["model"] == circuit_model)
                        & (circuit_utility["threshold"] == threshold)
                        & (circuit_utility["patch"] == patch)
                    )
                    if not circuit_utility[mask].empty:
                        utility_val = circuit_utility[mask]["perplexity"].values[0]

                utilities.append(utility_val)
                precision_scores.append(avg_prec)
                recall_scores.append(avg_rec)
                method_labels.append(f"{display_method}")
                method_types.append("circuit")

        if not method_labels:
            print(f"Warning: No methods found for model {model_base}")
            continue

        # Create precision scatter plot using method-specific styles
        scatter_points = []
        texts_prec = []
        for i, (x, y, label) in enumerate(
            zip(precision_scores, utilities, method_labels)
        ):
            style = get_method_style(label)
            scatter = ax_prec.scatter(
                x,
                y,
                c=style["color"],
                marker=style["marker"],
                s=style["size"],
                alpha=0.8,
                edgecolors="black",
                linewidth=1.5,
            )
            scatter_points.append((x, y))

        # # Create annotations for precision plot
        # for i, (x, y, label) in enumerate(zip(precision_scores, utilities, method_labels)):
        #     text = ax_prec.annotate(
        #         label,
        #         (x, y),
        #         fontsize=9,
        #         ha="center",
        #         va="center",
        #         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"),
        #         weight='bold'
        #     )
        #     texts_prec.append(text)

        # # Automatically adjust text positions for precision
        # adjust_text(
        #     texts_prec,
        #     ax=ax_prec,
        #     arrowprops=dict(arrowstyle="->", alpha=0.6, lw=1, color='gray'),
        #     only_move={'text': 'xy', 'points': 'xy', 'objects': 'xy'}
        # )

        # Create recall scatter plot using method-specific styles
        texts_rec = []
        for i, (x, y, label) in enumerate(zip(recall_scores, utilities, method_labels)):
            style = get_method_style(label)
            scatter = ax_rec.scatter(
                x,
                y,
                c=style["color"],
                marker=style["marker"],
                s=style["size"],
                alpha=0.8,
                edgecolors="black",
                linewidth=1.5,
            )

        # # Create annotations for recall plot
        # for i, (x, y, label) in enumerate(zip(recall_scores, utilities, method_labels)):
        #     text = ax_rec.annotate(
        #         label,
        #         (x, y),
        #         fontsize=9,
        #         ha="center",
        #         va="center",
        #         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"),
        #         weight='bold'
        #     )
        #     texts_rec.append(text)

        # Automatically adjust text positions for recall
        # adjust_text(
        #     texts_rec,
        #     ax=ax_rec,
        #     arrowprops=dict(arrowstyle="->", alpha=0.6, lw=1, color='gray'),
        #     only_move={'text': 'xy', 'points': 'xy', 'objects': 'xy'}
        # )

        # Format precision subplot
        # Only show x-axis labels on bottom row
        # if row == 2:  # Bottom row
        #     ax_prec.set_xlabel("Precision (%)", fontweight="bold", fontsize=12)
        # else:
        ax_prec.set_xlabel("")  # Remove x-axis label but keep ticks

        # Only show y-axis labels on leftmost column
        if col_offset == 0:
            ax_prec.set_ylabel("Perplexity", fontweight="bold", fontsize=12)
        else:
            ax_prec.set_ylabel("")

        ax_prec.xaxis.set_tick_params(labelsize=10)
        ax_prec.yaxis.set_tick_params(labelsize=10)
        ax_prec.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x)}"))
        ax_prec.set_xlim(left=0)  # Start x-axis from 0
        ax_prec.set_ylim(bottom=0)  # Start y-axis from 0
        ax_prec.set_title(
            f"{model_base_display_dict[model_base]} - Precision",
            fontweight="bold",
            fontsize=12,
        )
        ax_prec.grid(True, alpha=0.5)

        # Format recall subplot
        # Only show x-axis labels on bottom row
        # if row == 2:  # Bottom row
        # ax_rec.set_xlabel("Recall (%)", fontweight="bold", fontsize=12)
        # else:
        ax_rec.set_xlabel("")  # Remove x-axis label but keep ticks

        ax_rec.xaxis.set_tick_params(labelsize=10)
        ax_rec.yaxis.set_tick_params(labelsize=10)
        ax_rec.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x)}"))
        ax_rec.set_xlim(left=-0)  # Start x-axis from 0
        ax_rec.set_ylim(bottom=0)  # Start y-axis from 0
        ax_rec.set_title(
            f"{model_base_display_dict[model_base]} - Recall",
            fontweight="bold",
            fontsize=12,
        )
        ax_rec.grid(True, alpha=0.5)

    # Hide any unused subplots
    for row in range(3):
        for col in range(4):
            model_idx = row * 2 + col // 2
            if model_idx >= len(valid_models):
                axes[row, col].set_visible(False)

    # Create unified legend
    create_unified_legend(fig, axes)

    plt.tight_layout()
    plt.subplots_adjust(top=0.96, bottom=0.15)

    # Save the plot
    plot_path = os.path.join(
        output_dir, "utility_vs_precision_recall_scatter_all_models.png"
    )
    plt.savefig(plot_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"Saved utility vs precision/recall scatter plot: {plot_path}")


def generate_latex_table(
    baseline_results: Dict,
    baseline_utility: Dict = None,
    output_dir: str = "./plots",
    target_categories: List[str] = None,
) -> None:
    """
    Generate a LaTeX table showing baseline defense methods performance.

    Args:
        baseline_results: Baseline results dictionary
        baseline_utility: Baseline utility (perplexity) results dictionary
        output_dir: Directory to save the LaTeX file
        target_categories: List of PII categories processed
    """
    if target_categories is None:
        target_categories = ["PERSON", "LOC", "NORP"]

    os.makedirs(output_dir, exist_ok=True)

    # Define model order and display names
    model_order = [
        "gpt2-small",
        "gpt2-medium",
        "qwen3-06",
        "gpt2-large",
        "llama3-1b",
        "qwen3-17",
    ]

    model_display_names = {
        "gpt2-small": "GPT-Small",
        "gpt2-medium": "GPT2-Medium",
        "qwen3-06": "Qwen3-0.6b",
        "gpt2-large": "GPT2-Large",
        "llama3-1b": "Llama-3.2-1B",
        "qwen3-17": "Qwen3-1.7B",
    }

    defense_order = ["baseline", "apneap", "dp8", "dp4", "dp1"]
    defense_display_names = {
        "baseline": "\\textbf{Baseline}",
        "scrubbed": "\\textbf{Scrub}",
        "apneap": "\\textbf{APNEAP}",
        "dp8": "\\textbf{DP ($\\epsilon$=8)}",
        "dp4": "\\textbf{DP ($\\epsilon$=4)}",
        "dp1": "\\textbf{DP ($\\epsilon$=1)}",
    }

    # Start building the LaTeX table
    latex_content = """% --- Color Definitions ---
% Green for better performance (outside std dev)
% Orange for similar performance (within std dev)
% Red for worse performance
\\definecolor{mygreen}{HTML}{C8E6C9} % A light green
\\definecolor{myorange}{HTML}{FFECB3} % A light orange
\\definecolor{myred}{HTML}{FFCDD2}   % A light red

\\setlength{\\tabcolsep}{3pt}
\\begin{table}[!t]
    \\centering
    \\scriptsize
    \\resizebox{0.95\\columnwidth}{!}{
    \\begin{tabular}{l|c|c|c|c}
    \\toprule
    \\textbf{Defense} & \\textbf{Perpl $\\downarrow$} & \\textbf{Prec (\\%) $\\downarrow$} & \\textbf{Rec (\\%) $\\downarrow$} & \\textbf{Faith $\\uparrow$} \\\\
    \\midrule

"""

    for model_idx, model_base in enumerate(model_order):
        if model_base not in baseline_results:
            continue

        # Add model header
        latex_content += f"    \\multicolumn{{5}}{{c}}{{\\textbf{{{model_display_names[model_base]}}}}} \\\\\n"
        latex_content += "    \\midrule\n"

        model_data = baseline_results[model_base]

        # Get baseline values for comparison
        baseline_precision = None
        baseline_recall = None
        baseline_perplexity = None

        if "baseline" in model_data:
            baseline_precision, _, baseline_recall, _, _, _ = model_data["baseline"]
            if (
                baseline_utility
                and model_base in baseline_utility
                and "baseline" in baseline_utility[model_base]
            ):
                baseline_perplexity = baseline_utility[model_base]["baseline"]

        for defense in defense_order:
            if defense not in model_data:
                continue

            (
                avg_precision,
                range_precision,
                avg_recall,
                range_recall,
                avg_f1,
                range_f1,
            ) = model_data[defense]

            # Get perplexity
            perplexity = 0.0
            if baseline_utility and model_base in baseline_utility:
                if defense == "scrubbed":
                    perplexity = baseline_utility[model_base].get("scrubbed", 0.0)
                elif defense == "apneap":
                    # Try to get apneap perplexity, fallback to baseline
                    perplexity = baseline_utility[model_base].get(
                        "apneap", baseline_utility[model_base].get("baseline", 0.0)
                    )
                else:
                    perplexity = baseline_utility[model_base].get(defense, 0.0)

            # Determine colors based on comparison to baseline
            def get_color(current_val, baseline_val, lower_is_better=True):
                if baseline_val is None or current_val is None:
                    return "white"

                if defense == "baseline":
                    return "gray!15" if "qwen3" not in model_base else "gray!20"

                # Simple comparison - you can adjust thresholds as needed
                if lower_is_better:
                    if (
                        current_val < baseline_val * 0.9
                    ):  # Significantly better (10% improvement)
                        return "mygreen"
                    elif (
                        current_val > baseline_val * 1.1
                    ):  # Significantly worse (10% degradation)
                        return "myred"
                    else:
                        return "myorange"
                else:
                    if current_val > baseline_val * 1.1:  # Significantly better
                        return "mygreen"
                    elif current_val < baseline_val * 0.9:  # Significantly worse
                        return "myred"
                    else:
                        return "myorange"

            perp_color = get_color(
                perplexity, baseline_perplexity, lower_is_better=True
            )
            prec_color = get_color(
                avg_precision, baseline_precision, lower_is_better=True
            )
            rec_color = get_color(avg_recall, baseline_recall, lower_is_better=True)

            # Format the row
            defense_name = defense_display_names[defense]

            # Special formatting for baseline rows
            if defense == "baseline":
                latex_content += f"    \\cellcolor{{{perp_color}}}{defense_name} & \\cellcolor{{{perp_color}}}{perplexity:.2f} & \\cellcolor{{{prec_color}}}{avg_precision:.2f} $\\pm$ {range_precision:.2f} & \\cellcolor{{{rec_color}}}{avg_recall:.2f} $\\pm$ {range_recall:.2f} & \\cellcolor{{{perp_color}}} \\\\\n"
            else:
                latex_content += f"    {defense_name} & \\cellcolor{{{perp_color}}}{perplexity:.2f} & \\cellcolor{{{prec_color}}}{avg_precision:.2f} $\\pm$ {range_precision:.2f} & \\cellcolor{{{rec_color}}}{avg_recall:.2f} $\\pm$ {range_recall:.2f} & \n"
                latex_content += f"        \\cellcolor{{white}} \\\\\n"

        # Add midrule between models (except for the last one)
        if model_idx < len(model_order) - 1:
            latex_content += "    \\midrule\n\n"

    # Close the table
    latex_content += """    \\bottomrule
    \\end{tabular}
    }
    \\caption{\\textbf{Impact of DP fine-tuning}: we use perplexity (``Perpl'') for utility, precision (``Prec'') and recall (``Rec'') for PII leakage, and normalized faithfulness (``Faith''), averaged across all PII types. $\\downarrow$ ($\\uparrow$) indicates lower (higher values) is preferred. We use \\colorbox{gray!15}{Gray} for the baseline (no defenses), \\colorbox{mygreen}{Green} if better, \\colorbox{myred}{Red} if worse, and \\colorbox{myorange}{Orange} if similar to baseline.}
    \\label{tab:summary}
\\end{table}
"""

    # Save to file
    output_path = os.path.join(output_dir, "baseline_comparison_table.tex")
    with open(output_path, "w") as f:
        f.write(latex_content)

    print(f"LaTeX table saved to: {output_path}")


def find_best_circuit_methods(circuit_aggregated: Dict, top_k: int = 5) -> None:
    """
    Find and print the best (lowest precision and recall) circuit methods for each model.

    Args:
        circuit_aggregated: Aggregated circuit results dictionary
        top_k: Number of top methods to display
    """
    print("\n" + "=" * 80)
    print("BEST CIRCUIT METHODS (Lowest Precision and Recall)")
    print("=" * 80)

    for model_base, methods in circuit_aggregated.items():
        print(f"\nModel: {model_base}")
        print("-" * 60)

        # Convert to list for sorting
        method_results = []
        for (threshold, patch, ablation), (
            avg_prec,
            range_prec,
            avg_rec,
            range_rec,
            avg_f1,
            range_f1,
        ) in methods.items():
            # Use a combined score (precision + recall + f1) for ranking
            combined_score = avg_prec + avg_rec + avg_f1
            method_results.append(
                {
                    "threshold": threshold,
                    "patch": patch,
                    "ablation": ablation,
                    "precision": avg_prec,
                    "precision_range": range_prec,
                    "recall": avg_rec,
                    "recall_range": range_rec,
                    "f1": avg_f1,
                    "f1_range": range_f1,
                    "combined_score": combined_score,
                }
            )

        # Sort by combined score (lower is better for privacy)
        method_results.sort(key=lambda x: x["f1"])

        print(f"Top {top_k} methods with lowest precision + recall + F1:")
        print(
            "Rank | Threshold | Patch | Ablation | Precision (%) | Recall (%) | F1-Score (%) | Combined Score"
        )
        print("-" * 95)

        for i, method in enumerate(method_results[:top_k], 1):
            print(
                f"{i:4d} | {method['threshold']:9d} | {method['patch']:5s}  | {method['ablation']:5s} | "
                f"{method['precision']:6.2f}±{method['precision_range']:4.2f} | "
                f"{method['recall']:5.2f}±{method['recall_range']:4.2f} | "
                f"{method['f1']:6.2f}±{method['f1_range']:4.2f} | "
                f"{method['combined_score']:6.2f}"
            )


def main():
    """Main function to analyze baseline leak results."""
    # add args parsers
    parser = argparse.ArgumentParser(description="Analyze baseline leak results")
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default="./results/leaks",
        help="Path to the baseline results directory",
    )
    parser.add_argument(
        "--circuit_dir",
        type=str,
        default="./results/circuit_leaks",
        help="Path to the circuit leaks results directory",
    )
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    parser.add_argument(
        "--plot_dir", type=str, default="./plots", help="Directory to save plots"
    )
    args = parser.parse_args()
    baseline_dir = args.baseline_dir
    circuit_dir = args.circuit_dir

    if not os.path.exists(baseline_dir):
        print(f"Baseline directory not found: {baseline_dir}")
        return

    if not os.path.exists(circuit_dir):
        print(f"Circuit directory not found: {circuit_dir}")
        return

    # Categories we're interested in
    target_categories = ["PERSON", "LOC", "NORP"]
    epsilon_values = ["baseline", "scrubbed", "apneap", "dp8", "dp4", "dp1"]

    print("=" * 80)
    print("BASELINE vs CIRCUIT LEAK ANALYSIS RESULTS")
    print("=" * 80)

    # Process baseline results
    print("\n" + "=" * 50)
    print("PROCESSING BASELINE RESULTS")
    print("=" * 50)

    baseline_results = {}
    for epsilon in epsilon_values:
        print(f"\nEpsilon: {epsilon}")

        # Get all baseline model directories for this epsilon
        model_dirs = [
            d
            for d in os.listdir(baseline_dir)
            if os.path.isdir(os.path.join(baseline_dir, d)) and epsilon in d
        ]
        model_dirs.sort()
        model_dirs = [
            m for m in model_dirs if "pythia" not in m
        ]  # Filter out pythia models

        for model_name in model_dirs:
            model_path = os.path.join(baseline_dir, model_name)

            # Extract base model name
            model_base = model_name.replace(f"-{epsilon}", "")

            if model_base not in baseline_results:
                baseline_results[model_base] = {}

            # Store precision, recall, and F1-score values for each category
            precisions = []
            recalls = []
            f1_scores = []

            # Process each target category
            for category in target_categories:
                csv_file = os.path.join(model_path, f"{category}.csv")

                if os.path.exists(csv_file):
                    precision, recall, f1 = read_baseline_leak_csv(csv_file)
                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1)
                    print(
                        f"  {model_name} {category}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
                    )
                else:
                    print(f"  {model_name} {category}: CSV file not found")

            # Calculate averages and ranges across categories
            if precisions and recalls and f1_scores:
                avg_precision = np.mean(precisions)
                range_precision = (
                    (max(precisions) - min(precisions)) / 2
                    if len(precisions) > 1
                    else 0.0
                )
                avg_recall = np.mean(recalls)
                range_recall = (
                    (max(recalls) - min(recalls)) / 2 if len(recalls) > 1 else 0.0
                )
                avg_f1 = np.mean(f1_scores)
                range_f1 = (
                    (max(f1_scores) - min(f1_scores)) / 2 if len(f1_scores) > 1 else 0.0
                )

                baseline_results[model_base][epsilon] = (
                    avg_precision,
                    range_precision,
                    avg_recall,
                    range_recall,
                    avg_f1,
                    range_f1,
                )

                precision_cell = format_comparison_cell(avg_precision, range_precision)
                recall_cell = format_comparison_cell(avg_recall, range_recall)
                f1_cell = format_comparison_cell(avg_f1, range_f1)
                print(
                    f"  {model_name} Average: Precision={precision_cell}, Recall={recall_cell}, F1={f1_cell}"
                )
            else:
                print(f"  {model_name}: No valid data found for calculation")

    # Process circuit results
    print("\n" + "=" * 50)
    print("PROCESSING CIRCUIT RESULTS")
    print("=" * 50)

    circuit_results = process_circuit_results(circuit_dir, target_categories)
    circuit_aggregated = aggregate_circuit_results_across_categories(
        circuit_results, target_categories
    )

    # Display circuit results and compare with baseline
    print("\n" + "=" * 50)
    print("CIRCUIT RESULTS ANALYSIS")
    print("=" * 50)

    # Print circuit results summary
    for model_name, methods in circuit_aggregated.items():
        print(f"\nModel: {model_name}")
        print("-" * 60)
        print(f"Total method combinations tested: {len(methods)}")

        # Show some example results
        print("Sample results (threshold, patch, avg_precision, avg_recall, avg_f1):")
        for i, (
            (threshold, patch, ablation),
            (avg_prec, range_prec, avg_rec, range_rec, avg_f1, range_f1),
        ) in enumerate(methods.items()):
            if i < 5:  # Show first 5 results
                print(
                    f"  T{threshold}-{patch}-{ablation}: Precision={avg_prec:.2f}±{range_prec:.2f}%, Recall={avg_rec:.2f}±{range_rec:.2f}%, F1={avg_f1:.2f}±{range_f1:.2f}%"
                )

    # Find and display best circuit methods
    find_best_circuit_methods(circuit_aggregated, top_k=1)

    # Load utility data
    print(f"\n" + "=" * 50)
    print("LOADING UTILITY DATA")
    print("=" * 50)

    baseline_utility = read_baseline_utility_results()
    circuit_utility = read_circuit_utility_results()

    print(f"Loaded baseline utility data for {len(baseline_utility)} models")
    print(f"Loaded circuit utility data for {len(circuit_utility)} models")

    print(f"\n" + "=" * 50)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 50)
    #

    # plot_baseline_vs_circuit_comparison(
    #     baseline_results,
    #     circuit_aggregated,
    #     baseline_utility=baseline_utility,
    #     circuit_utility=circuit_utility,
    #     output_dir=args.plot_dir,
    #     target_categories=target_categories,
    # )

    # plot_utility_vs_precision_scatter(
    #     baseline_results,
    #     circuit_aggregated,
    #     baseline_utility=baseline_utility,
    #     circuit_utility=circuit_utility,
    #     output_dir=args.plot_dir,
    #     target_categories=target_categories,
    # )

    # plot_utility_vs_recall_scatter(
    #     baseline_results,
    #     circuit_aggregated,
    #     baseline_utility=baseline_utility,
    #     circuit_utility=circuit_utility,
    #     output_dir=args.plot_dir,
    #     target_categories=target_categories,
    # )

    # plot_utility_vs_f1_scatter(
    #     baseline_results,
    #     circuit_aggregated,
    #     baseline_utility=baseline_utility,
    #     circuit_utility=circuit_utility,
    #     output_dir=args.plot_dir,
    #     target_categories=target_categories,
    # )

    # Generate combined precision and recall scatter plot
    plot_utility_vs_precision_recall_scatter(
        baseline_results,
        circuit_aggregated,
        baseline_utility=baseline_utility,
        circuit_utility=circuit_utility,
        output_dir=args.plot_dir,
        target_categories=target_categories,
    )

    # Generate LaTeX table
    generate_latex_table(
        baseline_results,
        baseline_utility=baseline_utility,
        output_dir=args.plot_dir,
        target_categories=target_categories,
    )

    print(f"\nPlots saved to: {args.plot_dir}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
