import os
from numpy import mean
import transformers

from src.pii_leakage.arguments.config_args import ConfigArgs
from src.pii_leakage.arguments.dataset_args import DatasetArgs
from src.pii_leakage.arguments.env_args import EnvArgs
from src.pii_leakage.arguments.model_args import ModelArgs
from src.pii_leakage.arguments.ner_args import NERArgs
from src.pii_leakage.dataset.dataset_factory import DatasetFactory
from src.pii_leakage.models.language_model import LanguageModel
from src.pii_leakage.models.model_factory import ModelFactory
from tqdm import tqdm


def parse_args():
    parser = transformers.HfArgumentParser(
        (
            ModelArgs,
            NERArgs,
            DatasetArgs,
            EnvArgs,
            ConfigArgs,
        )
    )
    return parser.parse_args_into_dataclasses()


def evaluate(
    model_args: ModelArgs,
    ner_args: NERArgs,
    dataset_args: DatasetArgs,
    env_args: EnvArgs,
    config_args: ConfigArgs,
):
    """Evaluate a model and attack pair."""
    if config_args.exists():
        model_args = config_args.get_model_args()
        dataset_args = config_args.get_dataset_args()
        ner_args = config_args.get_ner_args()
        env_args = config_args.get_env_args()

    # Load the target model (trained on private data)
    lm: LanguageModel = ModelFactory.from_model_args(
        model_args, env_args=env_args
    ).load(verbose=True)

    test_dataset = DatasetFactory.from_dataset_args(
        dataset_args=dataset_args.set_split("test"), ner_args=ner_args
    )
    print(f"Evaluating on {len(test_dataset)} test samples...")

    all_perplexities = []
    for sample in tqdm(
        test_dataset,
        desc="Evaluating samples",
    ):
        perplexity = lm.perplexity(data=sample["text"], verbose=False)
        all_perplexities.append(perplexity)
    print(f"Average perplexity for {model_args.architecture}: {mean(all_perplexities)}")
    # Save the perplexity results
    clean_model_name = model_args.model_ckpt.split("/")[-1]  
    os.makedirs('./results/perplexity', exist_ok=True)
    with(open(f'./results/perplexity/{clean_model_name}.txt', "w")) as f:
        f.write(f"Average perplexity: {mean(all_perplexities)}\n")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate(*parse_args())
# ----------------------------------------------------------------------------
