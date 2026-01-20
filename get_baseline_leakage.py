import json
import os

from tqdm import tqdm
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
from src.pii_leakage.models.language_model import LanguageModel
from src.pii_leakage.models.model_factory import ModelFactory
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
    if config_args.exists():
        model_args = config_args.get_model_args()
        dataset_args = config_args.get_dataset_args()
        attack_args = config_args.get_attack_args()
        ner_args = config_args.get_ner_args()
    env_args = config_args.get_env_args()

    print_dict_highlighted(vars(attack_args))

    # Load the baseline model (publicly pre-trained).
    baseline_args = ModelArgs(**vars(model_args))
    baseline_args.model_ckpt = None
    baseline_lm: LanguageModel = ModelFactory.from_model_args(
        baseline_args, env_args=env_args
    ).load(verbose=True)

    attack: NaiveExtractionAttack = AttackFactory.from_attack_args(
        attack_args, ner_args=ner_args, env_args=env_args
    )

    os.makedirs("pii", exist_ok=True)
    model_name = model_args.architecture.replace("/", "-")
    for atype in ["PERSON", "LOC"]:
        print(f"=== Evaluating baseline model for PII type: {atype} ===")
        all_baseline_pii_raw = []
        attack_args.pii_class = atype
        for i in tqdm(range(0, 100), desc="Baseline PII extraction runs"):
            print(f"=== Baseline PII extraction run {i} ===")
            baseline_pii_raw = attack.attack(baseline_lm)
            baseline_pii = set(baseline_pii_raw.keys())
            print(f"Extracted {len(list(baseline_pii))} unique PII from baseline model")
            all_baseline_pii_raw.extend(pii for pii in baseline_pii if pii not in all_baseline_pii_raw)
            print("Total PII extracted so far:", len(all_baseline_pii_raw))
            if (i + 1) % 25 == 0:
                with open(
                    f"pii/baseline_pii_{model_name}_{attack_args.pii_class}_temp.list", "w"
                ) as f:
                    json.dump(list(all_baseline_pii_raw), f)

        
        with open(
            f"pii/baseline_pii_{model_name}_{attack_args.pii_class}.list", "w"
        ) as f:
            json.dump(list(all_baseline_pii_raw), f)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate(*parse_args())
# ----------------------------------------------------------------------------
