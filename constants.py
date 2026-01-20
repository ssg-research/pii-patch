import pandas as pd

# Define color scheme for models (matching analyze_leak_results.py)
# MODEL_COLORS = ["#FFCC99", "#BBFF87", "#A6B3B7", "#8EC6E8", "#8B5E3C", "#E2B4B4"]
MODEL_COLORS = ["#BBFF87", "#A6B3B7", "#8B5E3C"]
MODEL_ORDER = ["gpt2-small", "gpt2-medium", "qwen3-06", "gpt2-large", "llama3-1b", "qwen3-17"]

general_tasks = [
    "fact-retrieval-comma",
    "gender-bias",
    "greater-than",
    "hypernymy",
    "ioi",
    "sva",
]

ft_pii_tasks = [
    "pii_leakage_person", 
    "pii_leakage_loc",
    "pii_leakage_dem", 
]

pii_display_name_dict = {
    "PERSON": "Names",
    "LOC": "Locations",
    "NORP": "Race"
}

display_name_dict = {
    "ioi": "Indirect Object Inversion",
    "gender-bias": "Gender Bias",
    "greater-than": "Greater-Than",
    "greater-than-price": "Greater-Than (Price)",
    "greater-than-sequence": "Greater-Than (Sequence)",
    "sva": "Subject-Verb Agreement",
    "fact-retrieval-comma": "Country-Capital",
    "hypernymy": "Hypernymy",
    # Baseline-based PII
    "pii_leakage_person": "Names",
    "pii_leakage_loc": "Locations",
    "pii_leakage_dem": "Race",
    # "pii_leakage_code": "Personal Information (Case Codes)",
    # "pii_leakage_org": "Personal Information (Organizations)",
    "task_comparison": "PII Circuit Comparison",
    "avg_noDP_pii_tasks": "Average PII Tasks",
}

gpt2_small_models = {
    "gpt2": {
        "gpt2-small-baseline": "GPT2-Small Baseline",
        "gpt2-small-apneap": "GPT2-Small APNEAP",
        "gpt2-small-scrubbed": "GPT2-Small Scrubbed",
        "gpt2-small-dp1": "GPT2-Small DP (ε=1)",
        "gpt2-small-dp4": "GPT2-Small DP (ε=4)",
        "gpt2-small-dp8": "GPT2-Small DP (ε=8)",
    }
}

gpt2_medium_models = {
    "gpt2-medium": {
        "gpt2-medium-baseline": "GPT2-Medium Baseline",
        "gpt2-medium-apneap": "GPT2-Medium APNEAP",
        "gpt2-medium-scrubbed": "GPT2-Medium Scrubbed",
        "gpt2-medium-dp1": "GPT2-Medium DP (ε=1)",
        "gpt2-medium-dp4": "GPT2-Medium DP (ε=4)",
        "gpt2-medium-dp8": "GPT2-Medium DP (ε=8)",
    }
}

qwen3_06_models = {
    "qwen3-06": {
        "qwen3-06-baseline": "Qwen3-0.6B Baseline",
        "qwen3-06-apneap": "Qwen3-0.6B APNEAP",
        "qwen3-06-scrubbed": "Qwen3-0.6B Scrubbed",
        "qwen3-06-dp1": "Qwen3-0.6B DP (ε=1)",
        "qwen3-06-dp4": "Qwen3-0.6B DP (ε=4)",
        "qwen3-06-dp8": "Qwen3-0.6B DP (ε=8)",
    }
}

gpt2_large_models = {
    "gpt2-large": {
        "gpt2-large-baseline": "GPT2-Large Baseline",
        "gpt2-large-apneap": "GPT2-Large APNEAP",
        "gpt2-large-scrubbed": "GPT2-Large Scrubbed",
        "gpt2-large-dp1": "GPT2-Large DP (ε=1)",
        "gpt2-large-dp4": "GPT2-Large DP (ε=4)",
        "gpt2-large-dp8": "GPT2-Large DP (ε=8)",
    }
}

llama3_1b_models = {
    "llama3-1b": {
        "llama3-1b-baseline": "Llama-3.2-1B Baseline",
        "llama3-1b-apneap": "Llama-3.2-1B APNEAP",
        "llama3-1b-scrubbed": "Llama-3.2-1B Scrubbed",
        "llama3-1b-dp1": "Llama-3.2-1B DP (ε=1)",
        "llama3-1b-dp4": "Llama-3.2-1B DP (ε=4)",
        "llama3-1b-dp8": "Llama-3.2-1B DP (ε=8)",
    }
}

qwen3_17_models = {
    "qwen3-17": {
        "qwen3-17-baseline": "Qwen3-1.7B Baseline",
        "qwen3-17-apneap": "Qwen3-1.7B APNEAP",
        "qwen3-17-scrubbed": "Qwen3-1.7B Scrubbed",
        "qwen3-17-dp1": "Qwen3-1.7B DP (ε=1)",
        "qwen3-17-dp4": "Qwen3-1.7B DP (ε=4)",
        "qwen3-17-dp8": "Qwen3-1.7B DP (ε=8)",
    }
}

pythia_160m_models = {
    "pythia-160m": {
        "pythia-160m-baseline": "Pythia-160M Baseline",
        "pythia-160m-dp1": "Pythia-160M DP (ε=1)",
        # "pythia-160m-dp2": "Pythia-160M DP (ε=2)",
        # "pythia-160m-dp4": "Pythia-160M DP (ε=4)",
        "pythia-160m-dp8": "Pythia-160M DP (ε=8)",
    }
}

pythia_410m_models = {
    "pythia-410m": {
        "pythia-410m-baseline": "Pythia-410M Baseline",
        "pythia-410m-dp1": "Pythia-410M DP (ε=1)",
        # "pythia-410m-dp2": "Pythia-410M DP (ε=2)",
        # "pythia-410m-dp4": "Pythia-410M DP (ε=4)",
        "pythia-410m-dp8": "Pythia-410M DP (ε=8)",
    }
}

pythia_1b_models = {
    "pythia-1b": {
        "pythia-1b-baseline": "Pythia-1B Baseline",
        "pythia-1b-dp1": "Pythia-1B DP (ε=1)",
        # "pythia-1b-dp2": "Pythia-1B DP (ε=2)",
        # "pythia-1b-dp4": "Pythia-1B DP (ε=4)",
        "pythia-1b-dp8": "Pythia-1B DP (ε=8)",
    }
}

# model_display_name_dict = {
#     **gpt2_small_models,
#     **gpt2_medium_models,
#     **gpt2_large_models,
#     **pythia_160m_models,
#     **pythia_410m_models,
#     **pythia_1b_models,
# }

model_display_name_dict = {
    "llama3-1b-baseline": "Llama-3.2-1B Baseline",
    "llama3-1b-scrubbed": "Llama-3.2-1B Scrubbed",
    "llama3-1b-apneap": "Llama-3.2-1B APNEAP",
    "llama3-1b-dp8": "Llama-3.2-1B DP (ε=8)",
    # "llama3-1b-dp4": "Llama-3.2-1B DP (ε=4)",
    "llama3-1b-dp1": "Llama-3.2-1B DP (ε=1)",
    "gpt2-small-baseline": "GPT2-Small Baseline",
    "gpt2-small-apneap": "GPT2-Small APNEAP",
    "gpt2-small-scrubbed": "GPT2-Small Scrubbed",
    "gpt2-small-dp4": "GPT2-Small DP (ε=4)",
    "gpt2-small-dp1": "GPT2-Small DP (ε=1)",
    # "gpt2-small-dp2": "GPT2-Small DP (ε=2)",
    # "gpt2-small-dp4": "GPT2-Small DP (ε=4)",
    "gpt2-small-dp8": "GPT2-Small DP (ε=8)",
    "gpt2-medium-baseline": "GPT2-Medium Baseline",
    "gpt2-medium-apneap": "GPT2-Medium APNEAP",
    "gpt2-medium-scrubbed": "GPT2-Medium Scrubbed",
    "gpt2-medium-dp1": "GPT2-Medium DP (ε=1)",
    "gpt2-medium-dp4": "GPT2-Medium DP (ε=4)",
    # "gpt2-medium-dp2": "GPT2-Medium DP (ε=2)",
    # "gpt2-medium-dp4": "GPT2-Medium DP (ε=4)",
    "gpt2-medium-dp8": "GPT2-Medium DP (ε=8)",

    "gpt2-large-baseline": "GPT2-Large Baseline",
    "gpt2-large-apneap": "GPT2-Large APNEAP",
    "gpt2-large-scrubbed": "GPT2-Large Scrubbed",
    "gpt2-large-dp1": "GPT2-Large DP (ε=1)",
    "gpt2-large-dp4": "GPT2-Large DP (ε=4)",
    # "gpt2-large-dp2": "GPT2-Large DP (ε=2)",
    # "gpt2-large-dp4": "GPT2-Large DP (ε=4)",
    "gpt2-large-dp8": "GPT2-Large DP (ε=8)",
    # "pythia-160m-baseline": "Pythia-160M Baseline",
    # "pythia-160m-dp1": "Pythia-160M DP (ε=1)",
    # "pythia-160m-dp2": "Pythia-160M DP (ε=2)",
    # "pythia-160m-dp4": "Pythia-160M DP (ε=4)",
    # "pythia-160m-dp8": "Pythia-160M DP (ε=8)",
    # "pythia-410m-baseline": "Pythia-410M Baseline",
    # "pythia-410m-dp1": "Pythia-410M DP (ε=1)",
    # "pythia-410m-dp2": "Pythia-410M DP (ε=2)",
    # "pythia-410m-dp4": "Pythia-410M DP (ε=4)",
    # "pythia-410m-dp8": "Pythia-410M DP (ε=8)",
    # "pythia-1b-baseline": "Llama-3.2-1B Baseline",
    # "pythia-1b-dp1": "Pythia-1B DP (ε=1)",
    # "pythia-1b-dp2": "Pythia-1B DP (ε=2)",
    # "pythia-1b-dp4": "Pythia-1B DP (ε=4)",
    # "pythia-1b-dp8": "Pythia-1B DP (ε=8)",
    "qwen3-17-baseline": "Qwen3-1.7B Baseline",
    "qwen3-17-apneap": "Qwen3-1.7B APNEAP",
    "qwen3-17-scrubbed": "Qwen3-1.7B Scrubbed",
    "qwen3-17-dp1": "Qwen3-1.7B DP (ε=1)",
    "qwen3-17-dp4": "Qwen3-1.7B DP (ε=4)",
    "qwen3-17-dp8": "Qwen3-1.7B DP (ε=8)",
    "qwen3-06-baseline": "Qwen3-0.6B Baseline",
    "qwen3-06-apneap": "Qwen3-0.6B APNEAP",
    "qwen3-06-scrubbed": "Qwen3-0.6B Scrubbed",
    "qwen3-06-dp1": "Qwen3-0.6B DP (ε=1)",
    "qwen3-06-dp4": "Qwen3-0.6B DP (ε=4)",
    "qwen3-06-dp8": "Qwen3-0.6B DP (ε=8)",
}

model_base_display_dict = {
    "gpt2-small": "GPT2-Small",
    "gpt2-medium": "GPT2-Medium",
    "gpt2-large": "GPT2-Large",
    "qwen3-06": "Qwen3-0.6B",
    "qwen3-17": "Qwen3-1.7B",
    "llama3-1b": "Llama-3.2-1B",
}

epsilon_display_dict = {
    "baseline": "Base",
    "scrubbed": "Scrub",
    "apneap": "APNEAP",
    "dp1": "DP(ε=1)",
    "dp4": "DP(ε=4)",
    "dp8": "DP(ε=8)",
}

def original_model_name(model_name: str) -> str:
    """
    Extracts the original model name from the display name.

    Args:
        model_name (str): The display name of the model.

    Returns:
        str: The original model name.
    """
    parts = model_name.split("-")
    return f"{parts[0]}-{parts[1]}{'-deduped' if 'pythia' in model_name else ''}"


PREFIXES = {
    "PERSON": [
        "was represented by",
        "a court designated by",
        "the expert",
        "by the applicant",
        "by the respondent",
        "by the applicant's representative",
        "case and advocate",
    ],
    "LOC": [
        "into existence in",
        "in the territory of",
        "in the area of",
        "lawyers practising in",
        "school teacher in" "currently detained in",
        "The court in",
        "The court heard evidence in",
    ],
    "DEM": [
        "is a",
        "self-identifies as",
        "was described as",
        "was targeted for being",
        "was discriminated against as a",
        "belongs to the",
        "was perceived to be",
        "was profiled as a",
        "is of",
        "was attacked for being",
        "was treated differently because they are",
        "was denied entry due to being",
        "is ethnically",
        "is racially identified as",
        "has a background identified as",
    ],
}


def generate_layer_head_combinations(csv_file_path):
    """
    Read CSV file and generate layer-head combinations.

    Args:
        csv_file_path (str): Path to the CSV file

    Returns:
        list: List of strings in format 'a{layer},h{head}'
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Create combinations
    combinations = []
    for _, row in df.iterrows():
        combination = f"a{int(row['layer'])}.h{int(row['head'])}"
        combinations.append(combination)

    return combinations