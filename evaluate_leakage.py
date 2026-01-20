# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from analysis.leak_calculator import (
    gen_classification_metrics,
    print_classification_metrics,
    save_classification_metrics,
)
import transformers

from src.pii_leakage.arguments.attack_args import AttackArgs
from src.pii_leakage.arguments.config_args import ConfigArgs
from src.pii_leakage.arguments.dataset_args import DatasetArgs
from src.pii_leakage.arguments.env_args import EnvArgs
from src.pii_leakage.arguments.evaluation_args import EvaluationArgs
from src.pii_leakage.arguments.model_args import ModelArgs
from src.pii_leakage.arguments.ner_args import NERArgs
from src.pii_leakage.attacks.attack_factory import AttackFactory
from src.pii_leakage.attacks.extraction.naive_extraction import (
    NaiveExtractionAttack,
)
from src.pii_leakage.dataset.dataset_factory import DatasetFactory
from src.pii_leakage.models.language_model import LanguageModel
from src.pii_leakage.models.model_factory import ModelFactory
from src.pii_leakage.ner.pii_results import ListPII
from src.pii_leakage.utils.output import print_dict_highlighted


def parse_args():
    parser = transformers.HfArgumentParser(
        (
            ModelArgs,
            NERArgs,
            DatasetArgs,
            AttackArgs,
            EvaluationArgs,
            EnvArgs,
            ConfigArgs,
        )
    )
    return parser.parse_args_into_dataclasses()


def evaluate(
    model_args: ModelArgs,
    ner_args: NERArgs,
    dataset_args: DatasetArgs,
    attack_args: AttackArgs,
    eval_args: EvaluationArgs,
    env_args: EnvArgs,
    config_args: ConfigArgs,
):
    print("Evaluating baseline leakage - no patching...")
    if config_args.exists():
        model_args = config_args.get_model_args()
        dataset_args = config_args.get_dataset_args()
        attack_args = config_args.get_attack_args()
        ner_args = config_args.get_ner_args()
    env_args = config_args.get_env_args()

    print_dict_highlighted(vars(attack_args))

    # Load the target model (trained on private data)
    lm: LanguageModel = ModelFactory.from_model_args(
        model_args, env_args=env_args
    ).load(verbose=True)

    train_dataset = DatasetFactory.from_dataset_args(
        dataset_args=dataset_args.set_split("train"), ner_args=ner_args
    )
    real_pii: ListPII = train_dataset.load_pii().flatten(attack_args.pii_class)
    print(
        f"Sample 20 real PII out of {len(real_pii.unique().mentions())}: {real_pii.unique().mentions()[:20]}"
    )
    
    if "gpt2" in model_args.architecture:
        model_arch = "gpt2"
    elif "Qwen" in model_args.architecture:
        model_arch = "qwen3-06"
    else:
        model_arch = "EleutherAI-pythia-160m-deduped"

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

    """Evaluate a model and attack pair."""
    attack: NaiveExtractionAttack = AttackFactory.from_attack_args(
        attack_args, ner_args=ner_args, env_args=env_args
    )
    for i in range(0, 3):
        print(f"\nEvaluation run {i}")
        generated_pii = set(attack.attack(lm).keys())
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
            is_circuits=False,
            threshold=0,
            ablation=None,
            patch=None,
        )


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate(*parse_args())
# ----------------------------------------------------------------------------
