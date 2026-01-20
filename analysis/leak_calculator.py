import os
from datetime import datetime
import pandas as pd


def gen_classification_metrics(real_pii_set, leaked_pii):
    # Overall metrics
    precision = (
        len(real_pii_set.intersection(leaked_pii)) / len(leaked_pii)
        if len(leaked_pii) > 0
        else 0
    )
    recall = (
        len(real_pii_set.intersection(leaked_pii)) / len(real_pii_set)
        if len(real_pii_set) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return precision, recall, f1_score


def print_classification_metrics(real_pii_set, leaked_pii, precision, recall, f1_score):
    print(f"True Positives: {len(real_pii_set.intersection(leaked_pii))}")
    print(f"False Positives: {len(leaked_pii.difference(real_pii_set))}")
    print(f"Overall Precision: {100 * precision:.2f}%")
    print(f"Overall Recall:    {100 * recall:.2f}%")
    print(f"Overall F1 Score:  {100 * f1_score:.2f}%")


def save_classification_metrics(
    model_ckpt,
    pii_class,
    real_pii_set,
    leaked_pii,
    precision,
    recall,
    f1_score,
    i,
    is_circuits=False,
    threshold=None,
    patch=None,
    ablation=None
):
    root = "./results/circuit_leaks" if is_circuits else "./results/leaks"
    threshold = str(threshold)

    if is_circuits:
        results_file = f"{root}/all-results.csv"
    else:
        os.makedirs(f"{root}/{model_ckpt}", exist_ok=True)
        results_file = f"{root}/{model_ckpt}/{pii_class}.csv"

    # Prepare data row
    data = {        
        "timestamp": datetime.now().isoformat(),
        "model": model_ckpt,
        "threshold": threshold,
        "ablation": ablation,
        "patch": patch,
        "pii_class": pii_class,
        "iteration": i,
        "true_positives": len(real_pii_set.intersection(leaked_pii)),
        "false_positives": len(leaked_pii.difference(real_pii_set)),
        "precision": f"{100 * precision:.2f}",
        "recall": f"{100 * recall:.2f}",
        "f1_score": f"{100 * f1_score:.2f}",
        "total_real_pii": len(real_pii_set),
    }

    # Create DataFrame and append to CSV
    df = pd.DataFrame([data])
    if os.path.exists(results_file):
        df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        df.to_csv(results_file, mode="w", header=True, index=False)
