import torch
from typing import List, Dict, Tuple, Optional
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
import numpy as np


class PIICircuitPatcher:
    """
    Patches specific nodes/edges in a hooked transformer to reduce PII leakage
    based on circuit analysis findings.
    """

    def __init__(self, hooked_model: HookedTransformer):
        self.hooked_model = hooked_model
        self.active_hooks = []
        self.patch_cache = {}

    def zero_ablation_hook(
        self,
        activation: torch.Tensor,
        hook: HookPoint,
        positions: Optional[List[int]] = None,
        heads: Optional[List[int]] = None,
        neurons: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Hook function that zeros out specified components.

        Args:
            activation: The activation tensor to modify
            hook: The hook point object
            positions: Token positions to ablate (None = all positions)
            heads: Attention heads to ablate (for attention layers)
            neurons: MLP neurons to ablate (for MLP layers)
        """
        if positions is not None:
            if heads is not None:
                # Attention head ablation at specific positions
                activation[:, positions, heads, :] = 0
            elif neurons is not None:
                # MLP neuron ablation at specific positions
                activation[:, positions, neurons] = 0
            else:
                # Full position ablation
                activation[:, positions, :] = 0
        else:
            if heads is not None:
                # Full attention head ablation
                activation[:, :, heads, :] = 0
            elif neurons is not None:
                # Full MLP neuron ablation
                activation[:, :, neurons] = 0
            else:
                # Full activation ablation
                activation[:] = 0

        return activation

    def mean_ablation_hook(
        self,
        activation: torch.Tensor,
        hook: HookPoint,
        mean_cache: torch.Tensor,
        positions: Optional[List[int]] = None,
        heads: Optional[List[int]] = None,
        neurons: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Hook function that replaces activations with cached mean values.
        """
        # Handle sequence length mismatch by adjusting mean_cache to match activation
        current_seq_len = activation.shape[1]
        cache_seq_len = mean_cache.shape[1]
        
        # Cache the adjusted mean to avoid recomputing for same sequence length
        cache_key = f"{hook.name}_{current_seq_len}"
        if cache_key not in self.patch_cache:
            if cache_seq_len != current_seq_len:
                if cache_seq_len > current_seq_len:
                    # Truncate cache to match current sequence length
                    adjusted_cache = mean_cache[:, :current_seq_len, ...]
                else:
                    # Use the last position of cache for all additional positions (more natural than repeating)
                    last_pos_cache = mean_cache[:, -1:, ...].expand(-1, current_seq_len - cache_seq_len, *(-1,) * (len(mean_cache.shape) - 2))
                    adjusted_cache = torch.cat([mean_cache, last_pos_cache], dim=1)
            else:
                adjusted_cache = mean_cache
            
            # Cache the adjusted version for reuse
            self.patch_cache[cache_key] = adjusted_cache
        else:
            adjusted_cache = self.patch_cache[cache_key]
        
        # Ensure adjusted_cache is on the same device as activation
        if adjusted_cache.device != activation.device:
            adjusted_cache = adjusted_cache.to(activation.device)
            # Update cache with device-corrected version
            self.patch_cache[cache_key] = adjusted_cache
        
        if positions is not None:
            if heads is not None:
                activation[:, positions, heads, :] = adjusted_cache[:, positions, heads, :]
            elif neurons is not None:
                activation[:, positions, neurons] = adjusted_cache[:, positions, neurons]
            else:
                activation[:, positions, :] = adjusted_cache[:, positions, :]
        else:
            if heads is not None:
                activation[:, :, heads, :] = adjusted_cache[:, :, heads, :]
            elif neurons is not None:
                activation[:, :, neurons] = adjusted_cache[:, :, neurons]
            else:
                activation[:] = adjusted_cache

        return activation

    def scaling_hook(
        self,
        activation: torch.Tensor,
        hook: HookPoint,
        scale_factor: float,
        positions: Optional[List[int]] = None,
        heads: Optional[List[int]] = None,
        neurons: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Hook function that scales down specified components.
        """
        if positions is not None:
            if heads is not None:
                activation[:, positions, heads, :] *= scale_factor
            elif neurons is not None:
                activation[:, positions, neurons] *= scale_factor
            else:
                activation[:, positions, :] *= scale_factor
        else:
            if heads is not None:
                activation[:, :, heads, :] *= scale_factor
            elif neurons is not None:
                activation[:, :, neurons] *= scale_factor
            else:
                activation *= scale_factor

        return activation

    def patch_attention_heads(
        self,
        layer: int,
        heads: List[int],
        method: str = "zero",
        scale_factor: float = 0.1,
        positions: Optional[List[int]] = None,
    ):
        """
        Patch specific attention heads in a layer.

        Args:
            layer: Layer number to patch
            heads: List of head indices to patch
            method: "zero", "mean", or "scale"
            scale_factor: Factor to scale by (for "scale" method)
            positions: Specific token positions to patch (None = all)
        """
        hook_name = f"blocks.{layer}.attn.hook_result"

        if method == "zero" or method == "dp-zero":
            hook_fn = lambda act, hook: self.zero_ablation_hook(
                act, hook, positions=positions, heads=heads
            )
        elif method == "scale":
            hook_fn = lambda act, hook: self.scaling_hook(
                act, hook, scale_factor=scale_factor, positions=positions, heads=heads
            )
        elif method == "mean":
            # You'd need to compute mean cache first
            if hook_name not in self.patch_cache:
                raise ValueError(
                    f"No mean cache found for {hook_name}. Run compute_mean_cache first."
                )
            mean_cache = self.patch_cache[hook_name]
            hook_fn = lambda act, hook: self.mean_ablation_hook(
                act, hook, mean_cache, positions=positions, heads=heads
            )
        else:
            # Default to zero ablation for any unrecognized method
            hook_fn = lambda act, hook: self.zero_ablation_hook(
                act, hook, positions=positions, heads=heads
            )

        hook_handle = self.hooked_model.add_hook(hook_name, hook_fn)
        self.active_hooks.append(hook_handle)

    def patch_mlp_neurons(
        self,
        layer: int,
        neurons: List[int],
        method: str = "zero",
        scale_factor: float = 0.1,
        positions: Optional[List[int]] = None,
    ):
        """
        Patch specific MLP neurons in a layer.
        """
        hook_name = f"blocks.{layer}.mlp.hook_post"

        if method == "zero":
            hook_fn = lambda act, hook: self.zero_ablation_hook(
                act, hook, positions=positions, neurons=neurons
            )
        elif method == "scale":
            hook_fn = lambda act, hook: self.scaling_hook(
                act,
                hook,
                scale_factor=scale_factor,
                positions=positions,
                neurons=neurons,
            )
        elif method == "mean":
            if hook_name not in self.patch_cache:
                raise ValueError(
                    f"No mean cache found for {hook_name}. Run compute_mean_cache first."
                )
            mean_cache = self.patch_cache[hook_name]
            hook_fn = lambda act, hook: self.mean_ablation_hook(
                act, hook, mean_cache, positions=positions, neurons=neurons
            )
        else:
            # Default to zero ablation for any unrecognized method
            hook_fn = lambda act, hook: self.zero_ablation_hook(
                act, hook, positions=positions, neurons=neurons
            )

        hook_handle = self.hooked_model.add_hook(hook_name, hook_fn)
        self.active_hooks.append(hook_handle)

    def patch_residual_stream(
        self,
        layer: int,
        method: str = "zero",
        scale_factor: float = 0.1,
        positions: Optional[List[int]] = None,
    ):
        """
        Patch the residual stream at a specific layer.
        """
        hook_name = f"blocks.{layer}.hook_resid_post"

        if method == "zero":
            hook_fn = lambda act, hook: self.zero_ablation_hook(
                act, hook, positions=positions
            )
        elif method == "scale":
            hook_fn = lambda act, hook: self.scaling_hook(
                act, hook, scale_factor=scale_factor, positions=positions
            )
        elif method == "mean":
            if hook_name not in self.patch_cache:
                raise ValueError(
                    f"No mean cache found for {hook_name}. Run compute_mean_cache first."
                )
            mean_cache = self.patch_cache[hook_name]
            hook_fn = lambda act, hook: self.mean_ablation_hook(
                act, hook, mean_cache, positions=positions
            )
        else:
            # Default to zero ablation for any unrecognized method
            hook_fn = lambda act, hook: self.zero_ablation_hook(
                act, hook, positions=positions
            )

        hook_handle = self.hooked_model.add_hook(hook_name, hook_fn)
        self.active_hooks.append(hook_handle)

    def compute_mean_cache(self, clean_dataset: List[str], max_samples: int = 100):  # Reduced from 1000 to 100
        """
        Compute mean activations over a clean dataset for mean ablation.
        Memory-efficient version that processes in smaller batches.

        Args:
            clean_dataset: List of clean text samples (no PII)
            max_samples: Maximum number of samples to use
        """
        self.hooked_model.eval()

        # Sample subset if dataset is large
        if len(clean_dataset) > max_samples:
            clean_dataset = np.random.choice(clean_dataset, max_samples, replace=False)

        print(f"Computing mean cache with {len(clean_dataset)} samples...")

        # Process in very small batches to avoid memory issues - especially for large models
        # Detect model size and adjust batch size accordingly
        model_params = sum(p.numel() for p in self.hooked_model.parameters())
        if model_params > 1e9:  # Large model (>1B params)
            batch_size = 2  # Very small batches for large models
        elif model_params > 5e8:  # Medium-large model
            batch_size = 5
        else:
            batch_size = 10
            
        print(f"Using batch_size={batch_size} for model with {model_params/1e6:.1f}M parameters")
        
        hook_means = {}  # To accumulate running means
        hook_counts = {}  # To track number of samples processed

        for batch_start in range(0, len(clean_dataset), batch_size):
            batch_end = min(batch_start + batch_size, len(clean_dataset))
            batch_texts = clean_dataset[batch_start:batch_end]
            
            # Collect activations for this batch only
            activation_cache = {}

            def cache_hook(activation, hook):
                if hook.name not in activation_cache:
                    activation_cache[hook.name] = []
                # Move to CPU immediately to save GPU memory
                activation_cache[hook.name].append(activation.detach().cpu())
                return activation

            # Add temporary hooks to collect activations
            temp_hooks = []
            for layer in range(self.hooked_model.cfg.n_layers):
                for hook_type in ["hook_result", "hook_post", "hook_resid_post"]:
                    if hook_type == "hook_result":
                        hook_name = f"blocks.{layer}.attn.hook_result"
                    elif hook_type == "hook_post":
                        hook_name = f"blocks.{layer}.mlp.hook_post"
                    else:
                        hook_name = f"blocks.{layer}.hook_resid_post"

                    hook_handle = self.hooked_model.add_hook(hook_name, cache_hook)
                    if hook_handle is not None:
                        temp_hooks.append(hook_handle)

            # Run batch through model
            with torch.no_grad():
                for text in batch_texts:
                    tokens = self.hooked_model.to_tokens(text)
                    # Limit token length for memory efficiency
                    if tokens.shape[1] > 512:
                        tokens = tokens[:, :512]
                    self.hooked_model(tokens)
                    # Clean up tokens immediately
                    del tokens

            # Remove temporary hooks
            for hook in temp_hooks:
                if hook is not None and hasattr(hook, 'remove'):
                    hook.remove()
            
            if len(temp_hooks) == 0 or not hasattr(temp_hooks[0], 'remove'):
                self.hooked_model.reset_hooks()

            # Compute batch means and update running means
            for hook_name, activations in activation_cache.items():
                if not activations:
                    continue
                
                # Find the maximum sequence length in this batch
                max_seq_len = max(act.shape[1] for act in activations)
                
                # Pad activations in batch
                padded_batch = []
                for act in activations:
                    if act.shape[1] < max_seq_len:
                        pad_size = max_seq_len - act.shape[1]
                        if len(act.shape) == 4:  # Attention: [batch, seq, heads, dim]
                            padding = torch.zeros(act.shape[0], pad_size, act.shape[2], act.shape[3], device=act.device, dtype=act.dtype)
                        elif len(act.shape) == 3:  # MLP/Residual: [batch, seq, dim]
                            padding = torch.zeros(act.shape[0], pad_size, act.shape[2], device=act.device, dtype=act.dtype)
                        else:
                            padding = torch.zeros(act.shape[0], pad_size, *act.shape[2:], device=act.device, dtype=act.dtype)
                        
                        padded_act = torch.cat([act, padding], dim=1)
                    else:
                        padded_act = act
                    
                    padded_batch.append(padded_act)
                
                # Compute batch mean
                if padded_batch:
                    batch_mean = torch.stack(padded_batch, dim=0).mean(dim=0)
                    
                    # Update running mean
                    if hook_name not in hook_means:
                        hook_means[hook_name] = batch_mean
                        hook_counts[hook_name] = len(padded_batch)
                    else:
                        # Update running mean: new_mean = (old_mean * old_count + batch_mean * batch_count) / (old_count + batch_count)
                        old_count = hook_counts[hook_name]
                        batch_count = len(padded_batch)
                        total_count = old_count + batch_count
                        
                        # Ensure same sequence length (pad if needed)
                        if hook_means[hook_name].shape[1] != batch_mean.shape[1]:
                            max_len = max(hook_means[hook_name].shape[1], batch_mean.shape[1])
                            
                            # Pad existing mean if needed
                            if hook_means[hook_name].shape[1] < max_len:
                                pad_size = max_len - hook_means[hook_name].shape[1]
                                if len(hook_means[hook_name].shape) == 4:
                                    padding = torch.zeros(hook_means[hook_name].shape[0], pad_size, hook_means[hook_name].shape[2], hook_means[hook_name].shape[3], device=hook_means[hook_name].device, dtype=hook_means[hook_name].dtype)
                                else:
                                    padding = torch.zeros(hook_means[hook_name].shape[0], pad_size, hook_means[hook_name].shape[2], device=hook_means[hook_name].device, dtype=hook_means[hook_name].dtype)
                                hook_means[hook_name] = torch.cat([hook_means[hook_name], padding], dim=1)
                            
                            # Pad batch mean if needed
                            if batch_mean.shape[1] < max_len:
                                pad_size = max_len - batch_mean.shape[1]
                                if len(batch_mean.shape) == 4:
                                    padding = torch.zeros(batch_mean.shape[0], pad_size, batch_mean.shape[2], batch_mean.shape[3], device=batch_mean.device, dtype=batch_mean.dtype)
                                else:
                                    padding = torch.zeros(batch_mean.shape[0], pad_size, batch_mean.shape[2], device=batch_mean.device, dtype=batch_mean.dtype)
                                batch_mean = torch.cat([batch_mean, padding], dim=1)
                        
                        hook_means[hook_name] = (hook_means[hook_name] * old_count + batch_mean * batch_count) / total_count
                        hook_counts[hook_name] = total_count
                
                # Clean up batch data
                del padded_batch, activations
            
            # Clear activation cache for this batch
            del activation_cache
            torch.cuda.empty_cache()
            
            if batch_start % (batch_size * 5) == 0:  # Print progress every 5 batches
                print(f"Processed {min(batch_end, len(clean_dataset))}/{len(clean_dataset)} samples")

        # Move final means to GPU and store in cache
        for hook_name, mean_activation in hook_means.items():
            self.patch_cache[hook_name] = mean_activation.to(self.hooked_model.cfg.device)
        
        print(f"Completed computing mean cache for {len(hook_means)} hooks")

    def clear_hooks(self):
        """Remove all active hooks."""
        for hook in self.active_hooks:
            if hook is not None and hasattr(hook, 'remove'):
                hook.remove()
        self.active_hooks = []
        
        # Also clear patch cache to free memory
        self.patch_cache.clear()
        #     hook.remove()
        self.active_hooks = []

    def parse_node_string(self, node_str: str) -> Tuple[str, int, Optional[int]]:
        """
        Parse node string format like 'a2.h1' or 'm6'.

        Args:
            node_str: Node identifier (e.g., 'a2.h1', 'm6')

        Returns:
            Tuple of (node_type, layer, head_or_neuron)
            - For attention: ('attention', 2, 1) from 'a2.h1'
            - For MLP: ('mlp', 6, None) from 'm6'
        """
        if node_str.startswith("a") and ".h" in node_str:
            # Attention head: a2.h1
            parts = node_str[1:].split(".h")  # Remove 'a' and split on '.h'
            layer = int(parts[0])
            head = int(parts[1])
            return ("attention", layer, head)
        elif node_str.startswith("m"):
            # MLP: m6
            layer = int(node_str[1:])  # Remove 'm'
            return ("mlp", layer, None)
        else:
            raise ValueError(f"Unrecognized node format: {node_str}")

    def apply_circuit_patches_from_strings(
        self,
        node_strings: List[str],
        method: str = "zero",
        scale_factor: float = 0.1,
        positions: Optional[List[int]] = None,
    ):
        """
        Apply patches based on node string identifiers.

        Args:
            node_strings: List of node identifiers like ['a2.h1', 'm6', 'a5.h3']
            method: Patching method ("zero", "mean", "scale")
            scale_factor: Scaling factor for "scale" method
            positions: Specific token positions to patch (None = all)
        """
        # Group nodes by type and layer
        attention_heads = {}  # layer -> [heads]
        mlp_layers = set()

        for node_str in node_strings:
            node_type, layer, head_or_neuron = self.parse_node_string(node_str)

            if node_type == "attention":
                if layer not in attention_heads:
                    attention_heads[layer] = []
                attention_heads[layer].append(head_or_neuron)
            elif node_type == "mlp":
                mlp_layers.add(layer)

        # Apply attention head patches
        for layer, heads in attention_heads.items():
            print(f"Patching attention heads in layer {layer}: {heads}")
            self.patch_attention_heads(layer, heads, method, scale_factor, positions)

        # Apply MLP patches (patch entire MLP layer)
        for layer in mlp_layers:
            print(f"Patching MLP layer {layer}")
            self.patch_mlp_layer_full(layer, method, scale_factor, positions)

    def patch_mlp_layer_full(
        self,
        layer: int,
        method: str = "zero",
        scale_factor: float = 0.1,
        positions: Optional[List[int]] = None,
    ):
        """
        Patch an entire MLP layer (all neurons).
        """
        hook_name = f"blocks.{layer}.mlp.hook_post"

        if method == "zero":
            hook_fn = lambda act, hook: self.zero_ablation_hook(
                act, hook, positions=positions
            )
        elif method == "scale":
            hook_fn = lambda act, hook: self.scaling_hook(
                act, hook, scale_factor=scale_factor, positions=positions
            )
        elif method == "mean":
            if hook_name not in self.patch_cache:
                raise ValueError(
                    f"No mean cache found for {hook_name}. Run compute_mean_cache first."
                )
            mean_cache = self.patch_cache[hook_name]
            hook_fn = lambda act, hook: self.mean_ablation_hook(
                act, hook, mean_cache, positions=positions
            )
        else:
            # Default to zero ablation for any unrecognized method
            hook_fn = lambda act, hook: self.zero_ablation_hook(
                act, hook, positions=positions
            )

        hook_handle = self.hooked_model.add_hook(hook_name, hook_fn)
        self.active_hooks.append(hook_handle)

    def apply_circuit_patches(
        self,
        circuit_components: Dict[str, List],
        method: str = "zero",
        scale_factor: float = 0.1,
    ):
        """
        Apply patches based on circuit analysis results.

        Args:
            circuit_components: Dict with keys like "attention_heads", "mlp_neurons", "positions"
                Format: {
                    "attention_heads": [(layer, head), ...],
                    "mlp_neurons": [(layer, neuron), ...],
                    "positions": [pos1, pos2, ...],  # token positions
                    "layers": [layer1, layer2, ...]  # full layer ablation
                }
            method: Patching method ("zero", "mean", "scale")
            scale_factor: Scaling factor for "scale" method
        """
        positions = circuit_components.get("positions", None)

        # Patch attention heads
        if "attention_heads" in circuit_components:
            layer_heads = {}
            for layer, head in circuit_components["attention_heads"]:
                if layer not in layer_heads:
                    layer_heads[layer] = []
                layer_heads[layer].append(head)

            for layer, heads in layer_heads.items():
                self.patch_attention_heads(
                    layer, heads, method, scale_factor, positions
                )

        # Patch MLP neurons
        if "mlp_neurons" in circuit_components:
            layer_neurons = {}
            for layer, neuron in circuit_components["mlp_neurons"]:
                if layer not in layer_neurons:
                    layer_neurons[layer] = []
                layer_neurons[layer].append(neuron)

            for layer, neurons in layer_neurons.items():
                self.patch_mlp_neurons(layer, neurons, method, scale_factor, positions)

        # Patch full layers
        if "layers" in circuit_components:
            for layer in circuit_components["layers"]:
                self.patch_residual_stream(layer, method, scale_factor, positions)
