import torch
from functools import partial
import einops
from jaxtyping import Float
import transformer_lens.utils as utils


from . import plot_utils

device: torch.device = utils.get_device()

def logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False):
    # Only the final logits are relevant for the answer
    final_logits = logits[:, -1, :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
    if per_prompt:
        return answer_logit_diff
    else:
        return answer_logit_diff.mean()
    

#patching heads
def patch_head_vector(corrupted_head_vector,hook,head_index,clean_cache):
    corrupted_head_vector[:, :, head_index, :] = clean_cache[hook.name][:, :, head_index, :]
    return corrupted_head_vector


def patch_residual_component(corrupted_residual_component,hook,pos,clean_cache):
    corrupted_residual_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_residual_component


#patching heads
def patch_head_vector(corrupted_head_vector,hook,head_index,clean_cache):
    corrupted_head_vector[:, :, head_index, :] = clean_cache[hook.name][:, :, head_index, :]
    return corrupted_head_vector

def run_activation_patching_analysis(
    model, 
    cache, 
    tokens, 
    prompts, 
    answer_tokens, 
    original_logits, 
    plot_utils
):
    # Compute clean and corrupted logit diffs
    original_average_logit_diff = logits_to_ave_logit_diff(original_logits, answer_tokens)

    # Generate corrupted prompts by swapping pairs
    corrupted_prompts = []
    for i in range(0, len(prompts), 2):
        corrupted_prompts.append(prompts[i + 1])
        corrupted_prompts.append(prompts[i])
    corrupted_tokens = model.to_tokens(corrupted_prompts, prepend_bos=True)
    corrupted_logits, corrupted_cache = model.run_with_cache(
        corrupted_tokens, return_type="logits"
    )
    corrupted_average_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)
    print("Corrupted Average Logit Diff", round(corrupted_average_logit_diff.item(), 2))
    print("Clean Average Logit Diff", round(original_average_logit_diff.item(), 2))

    # Helper for normalization
    def normalize_patched_logit_diff(patched_logit_diff):
        return (patched_logit_diff - corrupted_average_logit_diff) / (original_average_logit_diff - corrupted_average_logit_diff)

    # Patch residual stream
    patched_residual_stream_diff = torch.zeros(model.cfg.n_layers, tokens.shape[1], device=device, dtype=torch.float32)
    for layer in range(model.cfg.n_layers):
        for position in range(tokens.shape[1]):
            hook_fn = partial(patch_residual_component, pos=position, clean_cache=cache)
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(utils.get_act_name("resid_pre", layer), hook_fn)],
                return_type="logits",
            )
            patched_logit_diff = logits_to_ave_logit_diff(patched_logits, answer_tokens)
            patched_residual_stream_diff[layer, position] = normalize_patched_logit_diff(
                patched_logit_diff
            )

    prompt_position_labels = [f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(tokens[0]))]
    plot_utils.imshow(
        patched_residual_stream_diff,
        x=prompt_position_labels,
        title="Logit Difference From Patched Residual Stream",
        labels={"x": "Position", "y": "Layer"},
        filename="patched_residual_stream_diff.png",save_dir="results/actpatching"
    )

    # Patch attention and MLP layers
    patched_attn_diff = torch.zeros(model.cfg.n_layers, tokens.shape[1], device=device, dtype=torch.float32)
    patched_mlp_diff = torch.zeros(model.cfg.n_layers, tokens.shape[1], device=device, dtype=torch.float32)
    for layer in range(model.cfg.n_layers):
        for position in range(tokens.shape[1]):
            hook_fn = partial(patch_residual_component, pos=position, clean_cache=cache)
            patched_attn_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(utils.get_act_name("attn_out", layer), hook_fn)],
                return_type="logits",
            )
            patched_attn_logit_diff = logits_to_ave_logit_diff(
                patched_attn_logits, answer_tokens
            )
            patched_mlp_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(utils.get_act_name("mlp_out", layer), hook_fn)],
                return_type="logits",
            )
            patched_mlp_logit_diff = logits_to_ave_logit_diff(
                patched_mlp_logits, answer_tokens
            )

            patched_attn_diff[layer, position] = normalize_patched_logit_diff(
                patched_attn_logit_diff
            )
            patched_mlp_diff[layer, position] = normalize_patched_logit_diff(
                patched_mlp_logit_diff
            )
    plot_utils.imshow(
        patched_attn_diff,
        x=prompt_position_labels,
        title="Logit Difference From Patched Attention Layer",
        labels={"x": "Position", "y": "Layer"},
        filename="patched_attn_diff.png",save_dir="results/actpatching"
    )
    plot_utils.imshow(
        patched_mlp_diff,
        x=prompt_position_labels,
        title="Logit Difference From Patched MLP Layer",
        labels={"x": "Position", "y": "Layer"},
        filename="patched_mlp_diff.png",save_dir="results/actpatching"
    )

    # Patch heads
    patched_head_z_diff = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)
    for layer in range(model.cfg.n_layers):
        for head_index in range(model.cfg.n_heads):
            hook_fn = partial(patch_head_vector, head_index=head_index, clean_cache=cache)
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(utils.get_act_name("z", layer, "attn"), hook_fn)],
                return_type="logits",
            )
            patched_logit_diff = logits_to_ave_logit_diff(patched_logits, answer_tokens)
            patched_head_z_diff[layer, head_index] = normalize_patched_logit_diff(
                patched_logit_diff
            )

    plot_utils.imshow(
        patched_head_z_diff,
        title="Logit Difference From Patched Head Output",
        labels={"x": "Head", "y": "Layer"},
        filename="patched_head_z_diff.png",save_dir="results/actpatching"
    )
