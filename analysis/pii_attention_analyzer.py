def analyze_privacy_models(data):
    """
    Comprehensive analysis comparing fine-tuned vs differentially private models
    based on attention pattern differences when processing PII.
    """
    fine_tuned_data = data[data["model"] == "gpt2-small-baseline"]
    dp_data = data[data["model"] == "gpt2-small-dp8"]

    print("=== PRIVACY MODEL COMPARISON ANALYSIS ===\n")

    # 1. Overall Sensitivity Analysis
    print("1. OVERALL SENSITIVITY TO PII CHANGES:")
    print("-" * 50)

    ft_stats = {
        "mean_sensitivity": fine_tuned_data["mean_abs_diff"].mean(),
        "max_sensitivity": fine_tuned_data["mean_abs_diff"].max(),
        "std_sensitivity": fine_tuned_data["mean_abs_diff"].std(),
        "extreme_responses": (fine_tuned_data["max_abs_diff"] > 0.4).sum(),
    }

    dp_stats = {
        "mean_sensitivity": dp_data["mean_abs_diff"].mean(),
        "max_sensitivity": dp_data["mean_abs_diff"].max(),
        "std_sensitivity": dp_data["mean_abs_diff"].std(),
        "extreme_responses": (dp_data["max_abs_diff"] > 0.4).sum(),
    }

    print(f"Fine-tuned Model:")
    print(f"  Mean sensitivity: {ft_stats['mean_sensitivity']:.6f}")
    print(f"  Max sensitivity: {ft_stats['max_sensitivity']:.6f}")
    print(f"  Std deviation: {ft_stats['std_sensitivity']:.6f}")
    print(f"  Extreme responses (>0.4): {ft_stats['extreme_responses']}")

    print(f"\nDifferentially Private Model:")
    print(f"  Mean sensitivity: {dp_stats['mean_sensitivity']:.6f}")
    print(f"  Max sensitivity: {dp_stats['max_sensitivity']:.6f}")
    print(f"  Std deviation: {dp_stats['std_sensitivity']:.6f}")
    print(f"  Extreme responses (>0.4): {dp_stats['extreme_responses']}")

    sensitivity_ratio = ft_stats["mean_sensitivity"] / dp_stats["mean_sensitivity"]
    print(f"\nSensitivity Ratio (FT/DP): {sensitivity_ratio:.2f}x")

    # 2. PII-Specific Attention Analysis
    print(f"\n2. PII-SPECIFIC ATTENTION PATTERNS:")
    print("-" * 50)

    ft_pii_pos = (fine_tuned_data["pii_attention_change"] > 0).sum()
    ft_pii_neg = (fine_tuned_data["pii_attention_change"] < 0).sum()
    dp_pii_pos = (dp_data["pii_attention_change"] > 0).sum()
    dp_pii_neg = (dp_data["pii_attention_change"] < 0).sum()

    print(f"Fine-tuned Model - PII Attention Changes:")
    print(f"  Increased attention: {ft_pii_pos} heads")
    print(f"  Decreased attention: {ft_pii_neg} heads")
    print(f"  Mean PII change: {fine_tuned_data['pii_attention_change'].mean():.6f}")
    print(
        f"  Extreme PII focus: {(abs(fine_tuned_data['pii_attention_change']) > 0.05).sum()} heads"
    )

    print(f"\nDifferentially Private Model - PII Attention Changes:")
    print(f"  Increased attention: {dp_pii_pos} heads")
    print(f"  Decreased attention: {dp_pii_neg} heads")
    print(f"  Mean PII change: {dp_data['pii_attention_change'].mean():.6f}")
    print(
        f"  Extreme PII focus: {(abs(dp_data['pii_attention_change']) > 0.05).sum()} heads"
    )

    # 3. Layer Distribution Analysis
    print(f"\n3. LAYER DISTRIBUTION ANALYSIS:")
    print("-" * 50)

    ft_layer_dist = fine_tuned_data["layer"].value_counts().sort_index()
    dp_layer_dist = dp_data["layer"].value_counts().sort_index()

    print("Fine-tuned Model - Most sensitive layers:")
    for layer, count in ft_layer_dist.head(11).items():
        avg_sensitivity = fine_tuned_data[fine_tuned_data["layer"] == layer][
            "mean_abs_diff"
        ].mean()
        print(f"  Layer {layer}: {count} heads, avg sensitivity {avg_sensitivity:.6f}")

    print("\nDifferentially Private Model - Most sensitive layers:")
    for layer, count in dp_layer_dist.head(11).items():
        avg_sensitivity = dp_data[dp_data["layer"] == layer]["mean_abs_diff"].mean()
        print(f"  Layer {layer}: {count} heads, avg sensitivity {avg_sensitivity:.6f}")

    # 4. Privacy Implications
    print(f"\n4. PRIVACY IMPLICATIONS:")
    print("-" * 50)

    # Calculate privacy risk indicators
    ft_privacy_risk = calculate_privacy_risk(fine_tuned_data)
    dp_privacy_risk = calculate_privacy_risk(dp_data)

    print(f"Privacy Risk Score (0-1, lower is better):")
    print(f"  Fine-tuned Model: {ft_privacy_risk:.4f}")
    print(f"  Differentially Private Model: {dp_privacy_risk:.4f}")
    print(
        f"  Risk Reduction: {((ft_privacy_risk - dp_privacy_risk) / ft_privacy_risk * 100):.1f}%"
    )


def calculate_privacy_risk(df):
    """
    Calculate a composite privacy risk score based on:
    - Overall sensitivity to PII changes
    - Extreme attention patterns
    - Consistency of PII-focused responses
    """
    sensitivity_component = df["mean_abs_diff"].mean() * 10  # Scale to 0-1
    extreme_component = (df["max_abs_diff"] > 0.3).mean()
    pii_focus_component = (abs(df["pii_attention_change"]) > 0.03).mean()

    # Weighted combination
    risk_score = (
        0.4 * sensitivity_component
        + 0.3 * extreme_component
        + 0.3 * pii_focus_component
    )

    return min(risk_score, 1.0)  # Cap at 1.0


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv("results/attention/attention_analysis_stats.csv")
    analyze_privacy_models(data)
