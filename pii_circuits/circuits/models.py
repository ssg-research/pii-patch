import os
import torch
import transformer_lens.utils as utils
from transformer_lens import ActivationCache, HookedTransformer

from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM

from . import data
from . import plot_utils

torch.set_grad_enabled(False)

def load_model(model_name="gpt2-small"):

    model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device='cuda'
    )
    return model