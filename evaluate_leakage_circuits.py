import glob
import json
import os

import pandas as pd
from constants import generate_layer_head_combinations
from src.pii_leakage.arguments.circuit_args import CircuitArgs
from src.pii_leakage.attacks.attack_factory import AttackFactory
from src.pii_leakage.attacks.extraction.circuit_based_naive_extraction import (
    CircuitBasedNaiveExtractionAttack,
)
from src.pii_leakage.models.model_factory import ModelFactory
from src.pii_leakage.utils.output import print_dict_highlighted
from src.pii_leakage.dataset.dataset_factory import DatasetFactory
from src.pii_leakage.ner.pii_results import ListPII
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM
import transformers

from pii_catcher import PIICircuitPatcher
from src.pii_leakage.arguments.attack_args import AttackArgs
from src.pii_leakage.arguments.config_args import ConfigArgs
from src.pii_leakage.arguments.env_args import EnvArgs
from src.pii_leakage.arguments.model_args import ModelArgs
from src.pii_leakage.arguments.ner_args import NERArgs
from analysis.leak_calculator import (
    gen_classification_metrics,
    save_classification_metrics,
)


def parse_args():
    parser = transformers.HfArgumentParser(
        (ModelArgs, NERArgs, AttackArgs, EnvArgs, ConfigArgs, CircuitArgs)
    )
    return parser.parse_args_into_dataclasses()


def extract_pii_with_patching(
    model_args: ModelArgs,
    ner_args: NERArgs,
    attack_args: AttackArgs,
    env_args: EnvArgs,
    config_args: ConfigArgs,
    circuit_args: CircuitArgs,
):
    print("Evaluating baseline leakage with patching !!!")
    """
    Modified version of your extract_pii function with string-based circuit patching.

    Args:
        pii_nodes: List of node strings like ['a2.h1', 'm6', 'a5.h3']
        positions: Specific token positions to patch (e.g., [2, 3, 4] for name positions)
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        dataset_args = config_args.get_dataset_args()
        ner_args = config_args.get_ner_args()
        env_args = config_args.get_env_args()
        attack_args = config_args.get_attack_args()
        circuit_args = config_args.get_circuit_args()

    # before running the job make sure a result doesn;t already exist
    root = "./results/circuit_leaks"
    results_file = f"{root}/all-results.csv"
    if os.path.exists(results_file):
        df = (
            pd.read_csv(results_file)
            if os.path.exists(results_file)
            else pd.DataFrame()
        )
        existing_entry = df[
            (df["model"] == model_args.model_ckpt.split("/")[-1])
            & (df["threshold"] == circuit_args.threshold)
            & (df["ablation"] == circuit_args.ablation)
            & (df["patch"] == circuit_args.patch)
        ]
        if not existing_entry.empty:
            print("Entry already exists in results. Skipping evaluation.")
            return

    print_dict_highlighted(vars(attack_args))
    print_dict_highlighted(vars(circuit_args))
    print_dict_highlighted(vars(model_args))

    model_name = model_args.model_ckpt.split("/")[-1]
    patch_method = circuit_args.patch
    ablation_method = circuit_args.ablation
    scale_factor = circuit_args.scale_factor
    threshold = f"{circuit_args.threshold}"

    # make sure the correct set of targets are loaded based on the ablation strategy
    # got any of the following ablation strategies: self, dp8-patch-zero, dp1-patch-zero, dp4-patch-zero
    if patch_method == "self":
        circuit_name = model_name
    elif patch_method == "dp8":
        circuit_name = model_name.replace("-baseline", "-dp8")
    elif patch_method == "dp1":
        circuit_name = model_name.replace("-baseline", "-dp1")
    elif patch_method == "dp4":
        circuit_name = model_name.replace("-baseline", "-dp4")

    # find a file thats matched the following regex pattern threshold results/circuit_analysis/99/attention_frequency_gpt2-large-dp8_dynamic_t0.000033.csv
    # we load the analysis we want to patch into the baseline model
    pattern = f"results/circuit_analysis/{threshold}/attention_frequency_{circuit_name}_dynamic_t*.csv"
    matching_files = glob.glob(pattern)

    if not matching_files:
        raise FileNotFoundError(f"No matching files found for pattern: {pattern}")

    target_file = matching_files[0]
    print(f"Using target file: {target_file} for patching")
    pii_nodes = generate_layer_head_combinations(target_file)

    # We first load the model we want to attack - i.e our non-patched model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_ckpt,
        trust_remote_code=True,
        token="",
    )
    hooked_model = HookedTransformer.from_pretrained(
        model_args.architecture,
        hf_model=model,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device="cuda",
    )
    hooked_model.cfg.use_attn_result = True

    # Apply circuit patches using string identifiers
    # Initialize patcher
    patcher = PIICircuitPatcher(hooked_model)

    if ablation_method == "mean":
        with open("gencircuits/data/no_pii_corpus.txt", "r") as f:
            clean_dataset = [line.strip() for line in f if line.strip()]
        patcher.compute_mean_cache(clean_dataset, max_samples=1000)

    print(f"Applying {patch_method} patches to PII nodes: {pii_nodes}")
    patcher.apply_circuit_patches_from_strings(
        pii_nodes, method=ablation_method, scale_factor=scale_factor, positions=None
    )
    hooked_attack: CircuitBasedNaiveExtractionAttack = AttackFactory.from_attack_args(
        attack_args, ner_args=ner_args, env_args=env_args, hooked_model=hooked_model
    )

    # get the private PII
    train_dataset = DatasetFactory.from_dataset_args(
        dataset_args=dataset_args.set_split("train"), ner_args=ner_args
    )
    real_pii: ListPII = train_dataset.load_pii().flatten(attack_args.pii_class)

    # get the public PII
    if "gpt2" in model_args.architecture:
        pii_public_baseline = "gpt2"
    else:
        pii_public_baseline = "EleutherAI-pythia-160m-deduped"

    baseline_pii = json.load(
        open(f"pii/baseline_pii_{pii_public_baseline}_{attack_args.pii_class}.list")
    )
    print(f"Baseline PII count: {len(baseline_pii)}")
    print(f"Baseline PII sample: {baseline_pii[:20]}")

    dud_args = ModelArgs(**vars(model_args))
    dud_args.model_ckpt = None

    if "pythia" in model_args.architecture:
        dud_args.architecture = "EleutherAI/pythia-160m-deduped"
    elif "Qwen3-0.6" in model_args.architecture:
        dud_args.architecture = "Qwen/Qwen3-0.6B"
    elif "Qwen3-1.7" in model_args.architecture:
        dud_args.architecture = "Qwen/Qwen3-1.7B"
    elif "gpt2" in model_args.architecture:
        dud_args.architecture = "gpt2"
    elif "llama" in model_args.architecture:
        dud_args.architecture = "gpt2"

    dud_lm = ModelFactory.from_model_args(dud_args, env_args=env_args).load()

    for i in range(0, 3):
        print(f"\nEvaluation run {i}")
        generated_pii = set(
            hooked_attack.attack(lm=dud_lm, hooked_model=hooked_model).keys()
        )
        real_pii_set = set(real_pii.unique().mentions())
        leaked_pii = generated_pii.difference(baseline_pii)

        print(f"Generated: {len(generated_pii)}")

        # Overall metrics
        precision, recall, f1_score = gen_classification_metrics(
            real_pii_set, leaked_pii
        )

        model_name = model_args.model_ckpt.split("/")[-1]
        save_classification_metrics(
            model_name,
            attack_args.pii_class,
            real_pii_set,
            leaked_pii,
            precision,
            recall,
            f1_score,
            i,
            is_circuits=True,
            threshold=threshold,
            ablation=ablation_method,
            patch=patch_method,
        )


if __name__ == "__main__":
    extract_pii_with_patching(*parse_args())
