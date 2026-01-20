import torch
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple
import pandas as pd


def plot_heatmap_summary(stats_df: pd.DataFrame, metric: str = "mean_abs_diff"):
    pivot_data = stats_df.pivot(index="layer", columns="head", values=metric)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0,
        vmax=0.02
    )
    plt.title(f'{metric.replace("_", " ").title()} Across All Layers and Heads')
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.tight_layout()
    return plt.gcf()


def plot_layer_overview(
    attn_data,
    n_heads: int,
    layer_idx: int,
    figsize: Tuple[int, int] = (20, 15),
):
    fig, axes = plt.subplots(3, n_heads, figsize=figsize)
    fig.suptitle(f"Layer {layer_idx} Attention Patterns - All Heads", fontsize=16)

    for head in range(n_heads):
        clean_data = attn_data["clean"][head].cpu().detach().tolist()
        sns.heatmap(
            clean_data,
            ax=axes[0, head],
            cmap="Blues",
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        axes[0, head].set_title(f"Head {head}\nClean")

        corrupted_data = attn_data["corrupted"][head].cpu().detach().tolist()
        sns.heatmap(
            corrupted_data,
            ax=axes[1, head],
            cmap="Reds",
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        axes[1, head].set_title("Corrupted")

        diff_data = attn_data["normalized_difference"][head].cpu().detach().tolist()
        max_abs = np.max(np.abs(diff_data))
        sns.heatmap(
            diff_data,
            ax=axes[2, head],
            cmap="RdBu_r",
            center=0,
            vmin=-max_abs,
            vmax=max_abs,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        axes[2, head].set_title("Norm. Diff.")

    plt.tight_layout()
    return fig


def plot_specific_head(
    attn_data,
    layer_idx: int,
    head_idx: int,
    pii_mask_clean: torch.Tensor,
    pii_mask_corrupted: torch.Tensor,
    show_tokens: bool = True,
    figsize: Tuple[int, int] = (15, 12),
    clean_str_tokens: List[str] = None,
    corrupted_str_tokens: List[str] = None,
):
    """Detailed analysis of a specific attention head"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"Layer {layer_idx}, Head {head_idx} - Detailed Analysis", fontsize=14)

    # Token labels for axes
    clean_labels = clean_str_tokens if show_tokens else False
    corrupted_labels = corrupted_str_tokens if show_tokens else False

    # Clean attention
    sns.heatmap(
        attn_data["clean_normalized"][head_idx].cpu().detach().tolist(),
        ax=axes[0, 0],
        cmap="Blues",
        xticklabels=clean_labels,
        yticklabels=clean_labels,
    )
    axes[0, 0].set_title("Clean (Normalized)")

    # Corrupted attention
    sns.heatmap(
        attn_data["corrupted_normalized"][head_idx].cpu().detach().tolist(),
        ax=axes[0, 1],
        cmap="Reds",
        xticklabels=corrupted_labels,
        yticklabels=corrupted_labels,
    )
    axes[0, 1].set_title("Corrupted (Normalized)")

    # Difference
    diff_data = attn_data["normalized_difference"][head_idx].cpu().detach().tolist()
    max_abs = np.max(np.abs(diff_data))
    sns.heatmap(
        diff_data,
        ax=axes[1, 0],
        cmap="RdBu_r",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        xticklabels=corrupted_labels,
        yticklabels=corrupted_labels,
    )
    axes[1, 0].set_title("Normalized Difference")

    # PII attention analysis
    plot_pii_attention_summary(
        layer_idx, head_idx, pii_mask_clean, pii_mask_corrupted, axes[1, 1]
    )

    plt.tight_layout()
    return fig


def plot_pii_attention_summary(
    self,
    layer_idx: int,
    head_idx: int,
    pii_mask_clean: torch.Tensor,
    pii_mask_corrupted: torch.Tensor,
    ax,
):
    """Summarize attention to/from PII tokens"""
    attn_data = self.compute_attention_differences(layer_idx)

    clean_attn = attn_data["clean_normalized"][head_idx]
    corrupted_attn = attn_data["corrupted_normalized"][head_idx]

    # Attention TO PII tokens (averaged across PII positions)
    clean_to_pii = (
        clean_attn[:, pii_mask_clean].mean(dim=1)
        if pii_mask_clean.any()
        else torch.zeros(clean_attn.shape[0])
    )
    corrupted_to_pii = (
        corrupted_attn[:, pii_mask_corrupted].mean(dim=1)
        if pii_mask_corrupted.any()
        else torch.zeros(corrupted_attn.shape[0])
    )

    # Attention FROM PII tokens (averaged across PII positions)
    clean_from_pii = (
        clean_attn[pii_mask_clean, :].mean(dim=0)
        if pii_mask_clean.any()
        else torch.zeros(clean_attn.shape[1])
    )
    corrupted_from_pii = (
        corrupted_attn[pii_mask_corrupted, :].mean(dim=0)
        if pii_mask_corrupted.any()
        else torch.zeros(corrupted_attn.shape[1])
    )

    # Plot
    positions = np.arange(len(clean_to_pii))
    width = 0.35

    ax.bar(
        positions - width / 2,
        clean_to_pii.cpu().numpy(),
        width,
        label="Clean → PII",
        alpha=0.7,
        color="blue",
    )
    ax.bar(
        positions + width / 2,
        corrupted_to_pii.cpu().numpy(),
        width,
        label="Corrupted → PII",
        alpha=0.7,
        color="red",
    )

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Avg Attention to PII")
    ax.set_title("Attention Summary")
    ax.legend()
    ax.grid(True, alpha=0.3)
