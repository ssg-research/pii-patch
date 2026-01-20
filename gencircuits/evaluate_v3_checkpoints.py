from argparse import ArgumentParser
from functools import partial
import os
from pathlib import Path

import pandas as pd
import numpy as np
from transformer_lens import HookedTransformer
from tqdm import tqdm

from gencircuits.eap.graph import Graph
from gencircuits.eap.attribute import attribute
from gencircuits.eap.evaluate import evaluate_graph, evaluate_baseline

from dataset import EAPDataset
from metrics import get_metric
from transformers import AutoModelForCausalLM

parser = ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--metric", type=str, required=True)
parser.add_argument("--end", type=int, default=2001)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--checkpoint", type=str, required=True)

args = parser.parse_args()

model_name = args.model
checkpoint = args.checkpoint

print(f"Loading model: {model_name} with checkpoint: {checkpoint}")

model = AutoModelForCausalLM.from_pretrained("/mnt/d1/acp23ajh/" + model_name + "/" + checkpoint)
model_name_noslash = model_name.split("/")[-1]

# Determine the base model type from the model name
if "small" in model_name:
    base_model_name = "gpt2"
elif "medium" in model_name:
    base_model_name = "gpt2-medium"
elif "large" in model_name:
    base_model_name = "gpt2-large"
elif "xl" in model_name:
    base_model_name = "gpt2-xl"
elif "pythia-160m" in model_name:
    base_model_name = "pythia-160m-deduped"
elif "pythia-410m" in model_name:
    base_model_name = "pythia-410m-deduped"
elif "pythia-1b" in model_name:
    base_model_name = "pythia-1b-deduped"
elif "llama3" in model_name:
    base_model_name = "meta-llama/Llama-3.2-1B"
elif "qwen3-06" in model_name:
    base_model_name = "Qwen/Qwen3-0.6B"
elif "qwen3-17" in model_name:
    base_model_name = "Qwen/Qwen3-1.7B"
else:
    # Default fallback - you might want to adjust this based on your models
    base_model_name = "gpt2"

hooked_model = HookedTransformer.from_pretrained(
    base_model_name,
    hf_model=model,
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device="cuda",
)
hooked_model.cfg.use_split_qkv_input = True
hooked_model.cfg.use_attn_result = True
hooked_model.cfg.use_hook_mlp_in = True
hooked_model.cfg.ungroup_grouped_query_attention = True
# %%
task = args.task
task_metric_name = args.metric
ds = EAPDataset(task, model_name)
ds.head(3000)

batch_size = args.batch_size
dataloader = ds.to_dataloader(batch_size)
task_metric = get_metric(task_metric_name, task, model=hooked_model)
kl_div = get_metric("kl_divergence", task, model=hooked_model)

# %%
baseline = (
    evaluate_baseline(
        hooked_model, dataloader, partial(task_metric, mean=False, loss=False)
    )
    .mean()
    .item()
)

corrupted_baseline = (
    evaluate_baseline(
        hooked_model,
        dataloader,
        partial(task_metric, mean=False, loss=False),
        run_corrupted=True,
    )
    .mean()
    .item()
)

# %%
# Instantiate a graph with a model
graph = Graph.from_model(hooked_model)
attribute(
    hooked_model,
    graph,
    dataloader,
    partial(kl_div, mean=True, loss=True),
    quiet=True,
    method="EAP-IG-inputs",
    ig_steps=5,
)
os.makedirs(f"graphs/{model_name_noslash}/{checkpoint}", exist_ok=True)
graph.to_json(f"graphs/{model_name_noslash}/{checkpoint}/{task}_kl.json")

# %%
gs = [graph]
n_edges = []
results = []

s = 100
if "gpt2-small" in model_name_noslash or "pythia-160m" in model_name_noslash or "qwen3-06" in model_name_noslash:
    print("Using gpt2-small or pythia-160m settings")
    e = 8001
else:
    print("Using larger circuit end")
    e = 25001

step = 100
first_steps = list(range(30, 100, 10))
later_steps = list(range(s, e, step))
steps = first_steps + later_steps
with tqdm(total=len(gs) * len(steps)) as pbar:
    for i in steps:
        n_edge = []
        result = []
        for graph in gs:
            graph.apply_greedy(i, absolute=True)
            n = graph.count_included_edges()
            r = evaluate_graph(
                hooked_model,
                graph,
                dataloader,
                partial(task_metric, mean=False, loss=False),
                quiet=True,
            )
            n_edge.append(n)
            result.append(r.mean().item())
            pbar.update(1)
        n_edges.append(n_edge)
        results.append(result)

n_edges = np.array(n_edges)
results = np.array(results)
# %%
d = {
    "baseline": [baseline] * len(steps),
    "corrupted_baseline": [corrupted_baseline] * len(steps),
    "edges": steps,
}

d[f"edges_EAP-IG"] = n_edges[:, 0].tolist()
d[f"loss_EAP-IG"] = results[:, 0].tolist()

df = pd.DataFrame.from_dict(d)
Path(f"results/pareto/{model_name_noslash}/{checkpoint}/csv").mkdir(exist_ok=True, parents=True)
df.to_csv(f"results/pareto/{model_name_noslash}/{checkpoint}/csv/{task}.csv", index=False)
