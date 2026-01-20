import os
import pandas as pd
from matplotlib import pyplot as plt
from analysis.attention_analyzer import PIIAttentionAnalyzer
from analysis.attention_plots import plot_heatmap_summary
from analysis.general_utils import find_pii_token_spans
from transformers import AutoTokenizer

if __name__ == "__main__":
    clean_pii = "Sikora"
    clean_prompt = "was represented by Mr S. Sikora"
    corrupted_pii = "Duarte"
    corrupted_prompt = "was represented by Mr S. Duarte"

    final_stats = pd.DataFrame(
        columns=["model", "layer", "head", "mean_abs_diff", "mean_diff"]
    )

    for local_model_name in ['gpt2-small-baseline', 'gpt2-medium-baseline', 'pythia-160m-baseline', 'pythia-410m-baseline']:
        target_model = f"./models/{local_model_name}"
        analyzer = PIIAttentionAnalyzer(target_model)
        analyzer.run_comparison(clean_prompt, corrupted_prompt)

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        pii_spans_clean = find_pii_token_spans(
            tokenizer=tokenizer, prompt=clean_prompt, pii_text=clean_pii
        )
        pii_spans_corrupted = find_pii_token_spans(
            tokenizer=tokenizer, prompt=corrupted_prompt, pii_text=corrupted_pii
        )

        pii_mask_clean = analyzer.identify_pii_tokens(
            pii_spans=pii_spans_clean, prompt_type="clean"
        )
        pii_mask_corrupted = analyzer.identify_pii_tokens(
            pii_spans=pii_spans_corrupted, prompt_type="corrupted"
        )

        stats = analyzer.compute_layer_statistics(pii_mask_clean, pii_mask_corrupted)
        stats["model"] = local_model_name
        fig3 = plot_heatmap_summary(stats, "mean_abs_diff")
        os.makedirs(f"plots/attention/{local_model_name}", exist_ok=True)
        plt.savefig(
            f"plots/attention/{local_model_name}/attention_analysis_overview.png"
        )
        final_stats = pd.concat([final_stats, stats], ignore_index=True)

    final_stats.to_csv(f"results/attention/attention_analysis_stats.csv", index=False)
