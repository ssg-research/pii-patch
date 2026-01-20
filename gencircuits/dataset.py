# %%
from functools import partial
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import constants


def collate_EAP(xs, task):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    if "hypernymy" not in task:
        labels = torch.tensor(labels, dtype=torch.long)
    return clean, corrupted, labels


class EAPDataset(Dataset):
    def __init__(self, task: str, model_name: str, filename: Optional[str] = None):
        if "gpt2" in model_name:
            self.df = pd.read_csv(f"gencircuits/data/{task}/gpt2-small.csv")
        elif "pythia" in model_name:
            self.df = pd.read_csv(f"gencircuits/data/{task}/pythia-1b-deduped.csv")
        elif "llama3" in model_name:
            self.df = pd.read_csv(f"gencircuits/data/{task}/llama3.csv")
        elif "qwen" in model_name:
            self.df = pd.read_csv(f"gencircuits/data/{task}/qwen3.csv")
        else:
            self.df = pd.read_csv(f"gencircuits/data/{task}/{filename}")
        self.df = self.df.head(1000)
        self.task = task

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = None
        all_idx_tasks = constants.ft_pii_tasks + constants.general_tasks
        if self.task in all_idx_tasks:
            label = [row["correct_idx"], row["incorrect_idx"]]
        elif "greater-than" in self.task:
            label = row["correct_idx"]
        elif "hypernymy" in self.task:
            answer = torch.tensor(eval(row["answers_idx"]))
            corrupted_answer = torch.tensor(eval(row["corrupted_answers_idx"]))
            label = [answer, corrupted_answer]
        elif self.task == "sva":
            label = row["plural"]
        else:
            raise ValueError(f"Got invalid task: {self.task}")
        return row["clean"], row["corrupted"], label

    def to_dataloader(self, batch_size: int):
        return DataLoader(
            self, batch_size=batch_size, collate_fn=partial(collate_EAP, task=self.task)
        )
