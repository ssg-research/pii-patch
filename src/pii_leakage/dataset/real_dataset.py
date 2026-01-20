# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from copy import deepcopy

from datasets import load_dataset

from ..arguments.dataset_args import DatasetArgs
from ..arguments.env_args import EnvArgs
from ..arguments.ner_args import NERArgs
from .dataset import Dataset
from ..global_configs import system_configs


class RealDataset(Dataset):
    """ A lightweight wrapper around a huggingface text dataset
    that allows caching and indexing PII
    """

    def __init__(self, dataset_args: DatasetArgs, ner_args: NERArgs = None, env_args: EnvArgs = None):
        """ A real dataset is a dataset loaded from real training data.
        """
        self.dataset_args = dataset_args
        super().__init__(ner_args, env_args)
        # self._tokenized_base_dataset = self._load_base_dataset()  # this is the reference to the huggingface dataset

    def _load_base_dataset(self, force_undefended=False):
        """ Loads the huggingface dataset. """
        # Check if this is a local dataset script (custom builder)
        if self.dataset_args.dataset_path.startswith("../src/extern/"):
            # Handle custom dataset builders directly
            import os
            
            if "echr" in self.dataset_args.dataset_path:
                from ..extern.echr.echr import CustomECHR, CustomECHRBuilder
                
                # Create builder instance using config_name
                builder = CustomECHR(
                    cache_dir=self.dataset_args.cache_dir(), 
                    config_name=self.dataset_args.dataset_mode if not force_undefended else "undefended"
                )
                
                # Download and prepare the dataset
                builder.download_and_prepare()
                
                # Return the dataset split
                return builder.as_dataset(split=self.dataset_args.split)
            elif "enron" in self.dataset_args.dataset_path:
                from ..extern.enron.enron import CustomEnron, CustomEnronBuilder
                
                # Create builder instance using config_name
                builder = CustomEnron(
                    cache_dir=self.dataset_args.cache_dir(), 
                    config_name=self.dataset_args.dataset_mode if not force_undefended else "undefended"
                )
                
                # Download and prepare the dataset
                builder.download_and_prepare()
                
                # Return the dataset split
                return builder.as_dataset(split=self.dataset_args.split)
            else:
                # For other custom datasets, might need similar treatment
                raise NotImplementedError(f"Custom dataset builder for {self.dataset_args.dataset_path} not implemented yet")
        else:
            # Handle regular HuggingFace datasets
            return load_dataset(self.dataset_args.dataset_path, 
                                cache_dir=self.dataset_args.cache_dir(),
                                name=self.dataset_args.dataset_mode if not force_undefended else "undefended")[self.dataset_args.split]

    @property
    def _pii_cache(self):
        """ Returns the filepath for the file that contains all pii and their location. """
        return os.path.join(os.path.abspath(system_configs.CACHE_DIR), self.dataset_args.hash(suffix="pii"))

    def shuffle(self):
        self._base_dataset.shuffle()
        return self

    def copy(self):
        return deepcopy(self)

    def select(self, indices):
        clone = self.copy()
        # clone._tokenized_base_dataset = clone._tokenized_base_dataset.select(indices)
        clone._base_dataset = clone._base_dataset.select(indices)
        return clone

    def __iter__(self):
        return self._base_dataset.__iter__()

    def __getitem__(self, item):
        return self._base_dataset.__getitem__(item)

    def get_hf_dataset(self):
        return self._base_dataset

    def first(self, column_name="text"):
        return self[column_name][0]
