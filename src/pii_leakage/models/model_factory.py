# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ..arguments.env_args import EnvArgs
from ..arguments.model_args import ModelArgs
from .gpt2 import GPT2
from .llama2 import Llama2
from .language_model import LanguageModel
from .pythia import Pythia

class ModelFactory:
    @staticmethod
    def from_model_args(model_args: ModelArgs, env_args: EnvArgs = None) -> LanguageModel:
        if "opt" in model_args.architecture:
            raise NotImplementedError
        elif "pythia" in model_args.architecture:
            return Pythia(model_args=model_args, env_args=env_args)    
        elif "gpt" in model_args.architecture:
            return GPT2(model_args=model_args, env_args=env_args)
        elif "Llama-2" in model_args.architecture:
            return Llama2(model_args=model_args, env_args=env_args)
        elif "Llama-3" in model_args.architecture:
            return LanguageModel(model_args=model_args, env_args=env_args)
        elif "Qwen" in model_args.architecture:
            return LanguageModel(model_args=model_args, env_args=env_args)
        else:
            raise ValueError(model_args.architecture)
