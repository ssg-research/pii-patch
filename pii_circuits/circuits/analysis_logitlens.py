import os
import torch
import numpy as np
import einops
from transformer_lens import ActivationCache
from fancy_einsum import einsum
import plotly.express as px

def residual_stack_to_logit_diff(model, answer_tokens, prompts, residual_stack, cache):
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)

    answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
    # print("Answer residual directions shape:", answer_residual_directions.shape)
    logit_diff_directions = (answer_residual_directions[:, 0] - answer_residual_directions[:, 1])

    return einsum("... batch d_model, batch d_model -> ...",scaled_residual_stack,logit_diff_directions) / len(prompts)


def visualize_attention_patterns(
    heads,
    local_cache,
    local_tokens,
    model,
    filename="attention_pattern",
    save_dir="results/logitlens",
    title: str = "",
):
    # If a single head is given, convert to a list
    if isinstance(heads, int):
        heads = [heads]

    batch_index = 0
    str_tokens = model.to_str_tokens(local_tokens)
    for idx, head in enumerate(heads):
        layer = head // model.cfg.n_heads
        head_index = head % model.cfg.n_heads
        label = f"L{layer}H{head_index}"
        pattern = local_cache["attn", layer][batch_index, head_index].detach().cpu().numpy()

        fig = px.imshow(
            pattern,
            labels={"x": "Key Position", "y": "Query Position"},
            x=str_tokens,
            y=str_tokens,
            color_continuous_scale="Viridis",
            title=f"{title} {label}" if title else label,
        )
    if save_dir and not os.path.isabs(filename) and not filename.startswith(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, filename)
    fig.write_image(filename)


def run_logit_diff_attribution_and_visualization(
    model, cache, answer_tokens, prompts, tokens, plot_utils, analysis, top_k=3
):
    # Residual stack to logit difference
    accumulated_residual, labels = cache.accumulated_resid(
        layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
    )
    logit_lens_logit_diffs = residual_stack_to_logit_diff(
        model, answer_tokens, prompts, accumulated_residual, cache
    )
    plot_utils.line(
        logit_lens_logit_diffs,
        x=np.arange(model.cfg.n_layers * 2 + 1) / 2,
        hover_name=labels,
        title="Logit Difference From Accumulate Residual Stream",
        filename="logit_lens_logit_diffs.png",save_dir="results/logitlens"
    )

    # Layer attribution
    per_layer_residual, labels = cache.decompose_resid(
        layer=-1, pos_slice=-1, return_labels=True
    )
    per_layer_logit_diffs = residual_stack_to_logit_diff(
        model, answer_tokens, prompts, per_layer_residual, cache
    )
    plot_utils.line(
        per_layer_logit_diffs,
        hover_name=labels,
        title="Logit Difference From Each Layer",
        filename="per_layer_logit_diffs.png",save_dir="results/logitlens"
    )

    # Per-head attribution
    per_head_residual, labels = cache.stack_head_results(
        layer=-1, pos_slice=-1, return_labels=True
    )
    per_head_logit_diffs = residual_stack_to_logit_diff(
        model, answer_tokens, prompts, per_head_residual, cache
    )
    per_head_logit_diffs = einops.rearrange(
        per_head_logit_diffs,
        "(layer head_index) -> layer head_index",
        layer=model.cfg.n_layers,
        head_index=model.cfg.n_heads,
    )
    plot_utils.imshow(
        per_head_logit_diffs,
        labels={"x": "Head", "y": "Layer"},
        title="Logit Difference From Each Head",
        filename="per_head_logit_diffs.png",save_dir="results/logitlens"
    )

    # Top positive heads
    top_positive_logit_attr_heads = torch.topk(
        per_head_logit_diffs.flatten(), k=top_k
    ).indices
    visualize_attention_patterns(
        top_positive_logit_attr_heads,
        cache,
        tokens[0],
        model,
        filename="top_positive_attention.png",save_dir="results/logitlens",
        title=f"Top {top_k} Positive Logit Attribution Heads",
    )

    # Top negative heads
    top_negative_logit_attr_heads = torch.topk(
        -per_head_logit_diffs.flatten(), k=top_k
    ).indices
    visualize_attention_patterns(
        top_negative_logit_attr_heads,
        cache,
        tokens[0],
        model,
        filename="top_negative_attention.png",save_dir="results/logitlens",
        title=f"Top {top_k} Negative Logit Attribution Heads",
    )