# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ..arguments.attack_args import AttackArgs
from ..arguments.env_args import EnvArgs
from ..arguments.ner_args import NERArgs
from .extraction.naive_extraction import NaiveExtractionAttack
from .extraction.circuit_based_naive_extraction import CircuitBasedNaiveExtractionAttack
from .inference.perpexity_inference import PerplexityInferenceAttack
from .privacy_attack import PrivacyAttack
from .reconstruction.perplexity_reconstruction import PerplexityReconstructionAttack
from transformer_lens import HookedTransformer


class AttackFactory:
    @staticmethod
    def from_attack_args(
        attack_args: AttackArgs,
        ner_args: NERArgs = None,
        env_args: EnvArgs = None,
        hooked_model: HookedTransformer = None,
    ) -> PrivacyAttack:
        if attack_args.attack_name == "circuit_based_naive_extraction":
            print(f"> Instantiating the circuit-based naive extraction attack.")
            return CircuitBasedNaiveExtractionAttack(
                attack_args=attack_args,
                ner_args=ner_args,
                env_args=env_args,
                hooked_model=hooked_model,
            )
        elif attack_args.attack_name == "naive_extraction":
            print(f"> Instantiating the naive extraction attack.")
            return NaiveExtractionAttack(
                attack_args=attack_args, ner_args=ner_args, env_args=env_args
            )
        elif attack_args.attack_name == "perplexity_inference":
            print(f"> Instantiating the perplexity inference attack.")
            return PerplexityInferenceAttack(
                attack_args=attack_args, ner_args=ner_args, env_args=env_args
            )
        elif attack_args.attack_name == "perplexity_reconstruction":
            print(f"> Instantiating the perplexity reconstruction attack.")
            return PerplexityReconstructionAttack(
                attack_args=attack_args, ner_args=ner_args, env_args=env_args
            )
        else:
            raise ValueError(attack_args.attack_name)
