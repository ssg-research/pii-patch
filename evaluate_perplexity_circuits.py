import csv
from datetime import datetime
import glob
import gc
import os
from numpy import mean
import transformers
import torch
from tqdm import tqdm
from typing import Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# Memory optimization settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from constants import generate_layer_head_combinations
from src.pii_leakage.arguments import dataset_args
from src.pii_leakage.arguments.circuit_args import CircuitArgs
from src.pii_leakage.arguments.config_args import ConfigArgs
from src.pii_leakage.arguments.dataset_args import DatasetArgs
from src.pii_leakage.arguments.env_args import EnvArgs
from src.pii_leakage.arguments.model_args import ModelArgs
from src.pii_leakage.arguments.ner_args import NERArgs
from src.pii_leakage.dataset.dataset_factory import DatasetFactory
from pii_catcher import PIICircuitPatcher
import os
import pandas as pd


def parse_args():
    # Parse HF args from remaining arguments
    parser = transformers.HfArgumentParser(
        (ModelArgs, NERArgs, DatasetArgs, EnvArgs, ConfigArgs, CircuitArgs)
    )
    hf_args = parser.parse_args_into_dataclasses()
    return hf_args


def save_perplexity(
    model_ckpt,
    threshold=None,
    patch=None,
    ablation=None,
    perplexity=0,
):
    root = "./results/circuit_perplexity"
    results_file = f"{root}/all_utility_results.csv"
    model_ckpt = model_ckpt.split("/")[-1]

    # Prepare data row
    data = {
        "timestamp": datetime.now().isoformat(),
        "model": model_ckpt,
        "threshold": threshold,
        "ablation": ablation,
        "patch": patch,
        "perplexity": perplexity,
    }

    # Create DataFrame and append to CSV
    df = pd.DataFrame([data])
    if os.path.exists(results_file):
        df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        df.to_csv(results_file, mode="w", header=True, index=False)


def calculate_hooked_perplexity(
    hooked_model: HookedTransformer,
    data: Union[list, str],
    tokenizer,
    max_length: int = 1024,
    offset: int = 0,
    apply_exp: bool = True,
    verbose: bool = True,
    return_as_list: bool = False,
    device: str = "cuda",
) -> float:
    """Calculate perplexity using a HookedTransformer model."""
    original_mode = hooked_model.training
    hooked_model.eval()

    if isinstance(data, str):
        data = [data]

    nlls = []  # negative log likelihoods
    ctr = 0  # Number of tokens viewed

    for txt in tqdm(data, desc="Compute PPL", disable=not verbose):
        # Limit sequence length more aggressively for large models
        if max_length > 512:
            max_length = 512  # Cap at 512 tokens for memory efficiency
            
        input_ids = (
            torch.tensor(tokenizer.encode(txt, truncation=True, max_length=max_length))
            .unsqueeze(0)
            .to(device)
        )

        target_ids = input_ids.clone()

        if offset > 0:  # ignore everything up to the offset
            target_ids[:, :offset] = -100

        tgt_len = target_ids.size(1) - offset

        with torch.no_grad():
            # Get logits from HookedTransformer
            logits = hooked_model(input_ids)

            # Calculate loss manually (similar to what AutoModelForCausalLM does)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            # Move to CPU immediately and detach to free GPU memory
            loss_value = loss.cpu().detach().float().item()

        if return_as_list:
            nlls.append(loss_value)
        else:
            nlls.append(loss_value)
            ctr += tgt_len
        
        # Clean up intermediate tensors - more aggressive cleanup
        del input_ids, target_ids, logits, shift_logits, shift_labels, loss
        # Force immediate cleanup for large models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    hooked_model.training = original_mode

    if return_as_list:
        if apply_exp:
            return [float(torch.exp(torch.tensor(nll)).item()) for nll in nlls]
        return nlls

    if apply_exp:
        return float(torch.exp(torch.tensor(nlls).mean()).item())
    return float(torch.tensor(nlls).mean().item())


def evaluate(
    model_args: ModelArgs,
    ner_args: NERArgs,
    dataset_args: DatasetArgs,
    env_args: EnvArgs,
    config_args: ConfigArgs,
    circuit_args: CircuitArgs,
):
    """Evaluate a model's perplexity, optionally with circuit patching."""
    if config_args.exists():
        model_args = config_args.get_model_args()
        dataset_args = config_args.get_dataset_args()
        ner_args = config_args.get_ner_args()
        env_args = config_args.get_env_args()
        circuit_args = config_args.get_circuit_args()

    # before running the job make sure a result doesn;t already exist
    root = "./results/circuit_perplexity"
    results_file = f"{root}/all_utility_results.csv"
    if os.path.exists(results_file):
        df = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()
        existing_entry = df[
            (df["model"] == model_args.model_ckpt.split("/")[-1])
            & (df["threshold"] == circuit_args.threshold)
            & (df["ablation"] == circuit_args.ablation)
            & (df["patch"] == circuit_args.patch)
        ]
        if not existing_entry.empty:
            print("Entry already exists in results. Skipping evaluation.")
            return


    model_name = model_args.model_ckpt.split("/")[-1]
    patch_method = circuit_args.patch
    ablation_method = circuit_args.ablation
    scale_factor = circuit_args.scale_factor
    threshold = f"{circuit_args.threshold}"

    # make sure the correct set of targets are loaded based on the ablation strategy
    if patch_method == "self":
        circuit_name = model_name
    elif patch_method == "dp8":
        circuit_name = model_name.replace("-baseline", "-dp8")
    elif patch_method == "dp1":
        circuit_name = model_name.replace("-baseline", "-dp1")
    elif patch_method == "dp4":
        circuit_name = model_name.replace("-baseline", "-dp4")

    # find a file thats matched the following regex pattern threshold results/circuit_analysis/99/attention_frequency_gpt2-large-dp8_dynamic_t0.000033.csv
    pattern = f"results/circuit_analysis/{threshold}/attention_frequency_{circuit_name}_dynamic_t*.csv"
    matching_files = glob.glob(pattern)

    if not matching_files:
        raise FileNotFoundError(f"No matching files found for pattern: {pattern}")

    target_file = matching_files[0]
    print(f"Using target file: {target_file} for patching")
    pii_nodes = generate_layer_head_combinations(target_file)

    # Load model with memory optimizations
    print(f"Loading model {model_args.model_ckpt}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_ckpt,
        torch_dtype=torch.float16 if "large" in model_name.lower() or "llama" in model_name.lower() or "qwen3-17" in model_name.lower() else torch.float32,
        device_map="cuda",
        low_cpu_mem_usage=True
    )
    
    hooked_model = HookedTransformer.from_pretrained(
        model_args.architecture,
        hf_model=model,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device="cuda",
        dtype=torch.float16 if "large" in model_name.lower() or "llama" in model_name.lower() or "qwen3-17" in model_name.lower() else torch.float32,
    )
    hooked_model.cfg.use_attn_result = True

    # Initialize patcher
    patcher = PIICircuitPatcher(hooked_model)

    if ablation_method == "mean":
        with open('gencircuits/data/no_pii_corpus.txt', 'r') as f:
            clean_dataset = [line.strip() for line in f if line.strip()]
        
        # Use fewer samples for large models
        max_samples = 50 if ("large" in model_name.lower() or "llama" in model_name.lower()) else 100
        print(f"Computing mean cache with {max_samples} samples for {model_name}")
        patcher.compute_mean_cache(clean_dataset, max_samples=max_samples)

    # Apply circuit patches using string identifiers
    print(f"Applying {patch_method} patches to PII nodes: {pii_nodes}")
    patcher.apply_circuit_patches_from_strings(
        pii_nodes, method=ablation_method, scale_factor=scale_factor, positions=None
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.architecture)
    tokenizer.pad_token = tokenizer.eos_token

    # Calculate perplexity using patched model
    test_dataset = DatasetFactory.from_dataset_args(
        dataset_args=dataset_args.set_split("test"), ner_args=ner_args
    )
    
    # Determine optimal sizes based on model size
    if "large" in model_name.lower() or "llama" in model_name.lower() or "qwen" in model_name.lower():
        subset_size = 1000  # Much smaller for large models
        batch_size = 1      # Process one sample at a time for large models
    else:
        subset_size = 5000  # Smaller for medium models
        batch_size = 5      # Smaller batches for medium models
    
    # Select a subset of the test dataset
    if len(test_dataset) > subset_size:
        test_dataset = test_dataset.select(range(subset_size))
        print(f"Selected {subset_size} samples from test dataset for evaluation")
    else:
        print(f"Test dataset has {len(test_dataset)} samples (using all)")

    print(f"Evaluating on {len(test_dataset)} test samples with batch_size={batch_size}...")

    all_perplexities = []
    
    with torch.no_grad():  # Critical: prevent gradient computation
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating batches with patched model"):
            batch_end = min(i + batch_size, len(test_dataset))
            batch_samples = test_dataset[i:batch_end]
            
            # batch_samples is a dict with "text" key containing list of texts
            batch_texts = batch_samples["text"] if isinstance(batch_samples, dict) else [sample["text"] for sample in batch_samples]
            
            batch_perplexities = []
            for j, text in enumerate(batch_texts):
                try:
                    perplexity = calculate_hooked_perplexity(
                        hooked_model=hooked_model,
                        data=text,
                        tokenizer=tokenizer,
                        max_length=min(512, model_args.tokenizer_max_length),  # Reduced max length for large models
                        verbose=False,
                        device=env_args.device,
                    )
                    batch_perplexities.append(perplexity)
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM at sample {i+j}, clearing cache and retrying with shorter sequence...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    try:
                        # Retry with much shorter sequence
                        perplexity = calculate_hooked_perplexity(
                            hooked_model=hooked_model,
                            data=text[:200],  # Truncate to 200 chars
                            tokenizer=tokenizer,
                            max_length=256,   # Much shorter
                            verbose=False,
                            device=env_args.device,
                        )
                        batch_perplexities.append(perplexity)
                    except torch.cuda.OutOfMemoryError:
                        print(f"Still OOM, skipping sample {i+j}")
                        continue
            
            all_perplexities.extend(batch_perplexities)
            
            # More frequent cleanup for large models
            cleanup_freq = 1 if "large" in model_name.lower() or "llama" in model_name.lower() or "qwen" in model_name.lower() else 10
            if i % cleanup_freq == 0:
                torch.cuda.empty_cache()
                gc.collect()

    save_perplexity(
        model_ckpt=model_args.model_ckpt,
        threshold=threshold,
        ablation=ablation_method,
        patch=patch_method,
        perplexity=mean(all_perplexities),
    )
    
    # Clean up memory
    patcher.clear_hooks()
    del hooked_model, model, patcher
    torch.cuda.empty_cache()
    gc.collect()


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate(*parse_args())
# ----------------------------------------------------------------------------
