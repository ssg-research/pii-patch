# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Tuple

import datasets
import pandas as pd
from datasets import load_dataset, concatenate_datasets

from src.pii_leakage.arguments.ner_args import NERArgs
from src.pii_leakage.ner.pii_results import ListPII
from src.pii_leakage.ner.tagger import Tagger
from src.pii_leakage.ner.tagger_factory import TaggerFactory
from src.pii_leakage.utils.output import print_highlighted, print_dict_highlighted
from src.pii_leakage.utils.random import rnd_idx

from dataclasses import dataclass

@dataclass
class CustomEnronBuilder(datasets.BuilderConfig):
    name: str = None
    sample_duplication_rate: int = 1    # number of times a sample is repeated
    shuffle_emails_seed: int = 42
    pseudonymize: bool = True


class CustomEnron(datasets.GeneratorBasedBuilder):
    """ A wrapper around the Enron email dataset that uses anonymization.  """

    VERSION = datasets.Version("1.0.0")
    _DESCRIPTION = "A custom wrapper for the Enron email dataset."
    _TEXT = "text"

    _URLS = {
        "url": "LLM-PBE/enron-email"
    }

    BUILDER_CONFIGS = [
        CustomEnronBuilder(name="undefended", sample_duplication_rate=1, version=VERSION,
                          description="undefended, private data"),
        CustomEnronBuilder(name="scrubbed", sample_duplication_rate=1, version=VERSION,
                          description="PII replaced with anon token")
    ]
    DEFAULT_CONFIG_NAME = "undefended"

    def __init__(self, *args, **kwargs):
        self.df: pd.DataFrame = pd.DataFrame()
        ner_args = NERArgs(ner='flair',
                           ner_model="flair/ner-english-ontonotes-large",
                           anon_token="<MASK>",
                           anonymize=kwargs.setdefault("config_name", None) == "scrubbed")
        self._tagger: Tagger = TaggerFactory.from_ner_args(ner_args)
        print_dict_highlighted(ner_args.__dict__)
        super().__init__(*args, **kwargs)

    def _info(self):
        fea_dict = {self._TEXT: datasets.Value("string"),}
        if self.config.pseudonymize:
            fea_dict.update({entity_class: datasets.Value("string") 
               for entity_class in self._tagger.get_entity_classes()})
        features = datasets.Features(fea_dict)
        return datasets.DatasetInfo(
            description=self._DESCRIPTION,
            features=features
        )

    def _split_generators(self, dl_manager):
        # Load the Enron email dataset
        self.df = load_dataset(self._URLS["url"])
        
        # The Enron dataset has a train split with email texts in 'text' column
        emails_data = self.df["train"]
        
        # Extract email text content
        self.data = emails_data['text']
        
        # Filter out empty or very short emails
        self.data = [email for email in self.data if email and len(email.strip()) > 50]
        
        # Shuffle emails if seed is provided
        if self.config.shuffle_emails_seed > 0:
            self.data = [self.data[i] for i in rnd_idx(N=len(self.data), seed=self.config.shuffle_emails_seed)]

        return [
            datasets.SplitGenerator(  # use ~45% of samples for the target model
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "start": 0.0,
                    "end": 0.45  # default: 0.45
                },
            ),
            datasets.SplitGenerator(  # use 10% of the training samples for test
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "start": 0.45,
                    "end": 0.55  # default: 0.55
                },
            ),
            datasets.SplitGenerator(  # Use remaining ~45% samples for shadow models
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "validation",
                    "start": 0.55,
                    "end": 1.0  # default: 1.0
                },
            ),
        ]

    def _generate_examples(self, split: str, start: float, end: float):
        """ Given a start and stop location, tag all PII and generate the dataset.
        We use multi_gpu generation for improved speed.
        """
        start_pos, end_pos = int(len(self.data) * start), int(len(self.data) * end)
        print_highlighted(
            f"Length of email data: {len(self.data)}. Processing emails from {start_pos} to {end_pos} (Total={end_pos - start_pos}).")

        unique_identifier = start_pos
        for i, email_text in enumerate(self.data[start_pos:end_pos]):
            if self.config.pseudonymize:
                pseudonymized_text, piis = self._tagger.pseudonymize(email_text)
                
                if i == 0:
                    print_highlighted(f"Sample pseudonymized email:\n{pseudonymized_text[:500]}...")

                pii_annotations = {k: ListPII() for k in self._tagger.get_entity_classes()}
                if hasattr(piis, 'group_by_class'):
                    pii_annotations.update({k: v.dumps() for k, v in piis.group_by_class().items()})
                else:
                    # Handle case where piis is a simple dict (for testing)
                    for entity_class in self._tagger.get_entity_classes():
                        if entity_class in piis:
                            pii_annotations[entity_class] = piis[entity_class]
            else:
                pseudonymized_text = email_text
                pii_annotations = {}
            
            for _ in range(self.config.sample_duplication_rate):
                yield f"{unique_identifier}", {
                    self._TEXT: pseudonymized_text,
                    **pii_annotations
                }
                unique_identifier += 1
