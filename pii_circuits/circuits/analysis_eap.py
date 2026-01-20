from functools import partial

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute 


gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    label_indices = []
    for correct, incorrect in labels:
        correct_id = gpt2_tokenizer.encode(correct, add_special_tokens=False)[-1]
        incorrect_id = gpt2_tokenizer.encode(incorrect, add_special_tokens=False)[-1]
        label_indices.append([correct_id, incorrect_id])
    labels = torch.tensor(label_indices, dtype=torch.long)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row['clean'], row['corrupted'], [row['correct_name'], row['incorrect_name']]

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def logit_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    logits = get_logit_positions(logits, input_length)
    good_bad = torch.gather(logits, -1, labels.to(logits.device))
    results = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results


def run_eap_ioi_pipeline(
    model,
    dataset_path='ioi.csv',
    batch_size=10,
    ig_steps=5,
    topn=20000,
    graph_pt_path='ioi_graph.pt',
    graph_png_path=None
):
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    ds = EAPDataset(dataset_path)
    dataloader = ds.to_dataloader(batch_size)

    g = Graph.from_model(model)

    attribute(
        model, g, dataloader,
        partial(logit_diff, loss=True, mean=True),
        method='EAP-IG-inputs', ig_steps=ig_steps
    )

    g.apply_topn(topn, True)
    g.to_pt(graph_pt_path)

    if graph_png_path is not None:
        try:
            import pygraphviz
            g.to_image(graph_png_path)
        except ImportError:
            print("No pygraphviz installed; skipping graph image export.")

    baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
    results = evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
    print(f"Original performance was {baseline}; the circuit's performance is {results}")

if __name__ == "__main__":
    model_name = 'gpt2'
    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device='cuda',
        dtype=torch.float16
    )
    run_eap_ioi_pipeline(
        model,
        dataset_path='./data/ioi.csv',
        batch_size=10,
        ig_steps=5,
        topn=20000,
        graph_pt_path='./results/eap/ioi_graph.pt',
        graph_png_path=None
    )