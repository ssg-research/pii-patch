from transformer_lens import HookedTransformer
import torch
from typing import List, Tuple
import pandas as pd
from transformers import AutoModelForCausalLM


class PIIAttentionAnalyzer:
    def __init__(self, model_name: str = "gpt2"):
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = HookedTransformer.from_pretrained(
            "gpt2",
            hf_model=hf_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.n_layers = self.model.cfg.n_layers
        self.n_heads = self.model.cfg.n_heads

    def run_comparison(self, clean_prompt: str, corrupted_prompt: str):
        """Run both prompts and cache activations"""
        self.clean_tokens = self.model.to_tokens(clean_prompt)
        self.corrupted_tokens = self.model.to_tokens(corrupted_prompt)

        self.clean_logits, self.clean_cache = self.model.run_with_cache(clean_prompt)
        self.corrupted_logits, self.corrupted_cache = self.model.run_with_cache(
            corrupted_prompt
        )

        # Store token strings for visualization
        self.clean_str_tokens = self.model.to_str_tokens(clean_prompt)
        self.corrupted_str_tokens = self.model.to_str_tokens(corrupted_prompt)

        return self

    def identify_pii_tokens(
        self, pii_spans: List[Tuple[int, int]], prompt_type: str = "clean"
    ):
        """
        Mark PII token positions
        pii_spans: List of (start_idx, end_idx) tuples for PII tokens
        prompt_type: "clean" or "corrupted" to select the appropriate token list

        Returns:
            torch.Tensor: Boolean mask where True indicates PII tokens
        """
        if prompt_type == "clean":
            tokens = self.clean_str_tokens
        else:
            tokens = self.corrupted_str_tokens

        pii_mask = torch.zeros(len(tokens), dtype=torch.bool)

        for start, end in pii_spans:
            if start < 0 or end >= len(tokens) or start > end:
                print(
                    f"Warning: Invalid span ({start}, {end}) for token length {len(tokens)}"
                )
                continue
            pii_mask[start : end + 1] = True

        return pii_mask

    def compute_attention_differences(self, layer_idx: int):
        """Compute attention pattern differences for a specific layer"""
        # Use current TransformerLens syntax

        # Removed the batch dim..
        clean_attn = self.clean_cache["pattern", layer_idx][0]
        corrupted_attn = self.corrupted_cache["pattern", layer_idx][0]

        # Ensure tensors are properly shaped
        if clean_attn.dim() == 4:  # If still has batch dimension
            clean_attn = clean_attn.squeeze(0)
        if corrupted_attn.dim() == 4:
            corrupted_attn = corrupted_attn.squeeze(0)

        # Compute difference (corrupted - clean)
        attn_diff = corrupted_attn - clean_attn

        # Compute normalized versions
        clean_attn_norm = clean_attn / (clean_attn.sum(dim=-1, keepdim=True) + 1e-8)
        corrupted_attn_norm = corrupted_attn / (
            corrupted_attn.sum(dim=-1, keepdim=True) + 1e-8
        )
        norm_diff = corrupted_attn_norm - clean_attn_norm

        return {
            "clean": clean_attn,
            "corrupted": corrupted_attn,
            "difference": attn_diff,
            "clean_normalized": clean_attn_norm,
            "corrupted_normalized": corrupted_attn_norm,
            "normalized_difference": norm_diff,
        }

    def compute_layer_statistics(
        self, pii_mask_clean: torch.Tensor, pii_mask_corrupted: torch.Tensor
    ):
        """Compute summary statistics across all layers and heads"""
        stats = []

        for layer in range(self.n_layers):
            attn_data = self.compute_attention_differences(layer)

            for head in range(self.n_heads):
                clean_attn = attn_data["clean_normalized"][head]
                corrupted_attn = attn_data["corrupted_normalized"][head]
                diff = attn_data["normalized_difference"][head]

                # Overall statistics
                mean_abs_diff = torch.abs(diff).mean().item()
                max_abs_diff = torch.abs(diff).max().item()

                # PII-specific statistics
                if pii_mask_clean.any() and pii_mask_corrupted.any():
                    pii_to_pii_clean = (
                        clean_attn[pii_mask_clean][:, pii_mask_clean].mean().item()
                    )
                    pii_to_pii_corrupted = (
                        corrupted_attn[pii_mask_corrupted][:, pii_mask_corrupted]
                        .mean()
                        .item()
                    )
                    pii_attention_change = pii_to_pii_corrupted - pii_to_pii_clean
                else:
                    pii_attention_change = 0

                stats.append(
                    {
                        "layer": layer,
                        "head": head,
                        "mean_abs_diff": mean_abs_diff,
                        "max_abs_diff": max_abs_diff,
                        "pii_attention_change": pii_attention_change,
                    }
                )

        return pd.DataFrame(stats)
