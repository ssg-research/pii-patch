import torch
from collections import defaultdict
from typing import Dict
import pandas as pd
from gencircuits.eap.graph import Graph, InputNode


def analyze_node_degrees(
    graphs: Dict[str, "Graph"], threshold: float = 0.0, absolute: bool = True
) -> Dict[str, Dict]:
    """
    Analyze node degree distributions across different graph configurations.

    Args:
        graphs: Dictionary mapping model names to Graph objects
        threshold: Minimum edge score to consider (default: 0.0)
        absolute: Whether to use absolute values of scores (default: True)

    Returns:
        Dictionary containing node degree analysis results
    """
    results = {}

    for model_name, graph in graphs.items():
        node_degrees = defaultdict(
            lambda: {"in_degree": 0, "out_degree": 0, "total_degree": 0}
        )

        # get all edge scores from the graph
        edge_scores = graph.scores.clone()
        # absolute scores tell us how much the edge contributes for both pos+neg values
        if absolute:
            edge_scores = torch.abs(edge_scores)

        # check all edge scores are non-negative
        if (edge_scores < 0).any():
            # force any non zero values to be 0?!
            edge_scores[edge_scores < 0] = 0
            # raise ValueError(
            #     f"Edge scores for model {model_name} contain negative values. "
            #     "Ensure scores are non-negative before applying threshold."
            # )
        
        # Create mask for edges above threshold
        # Only include edges that are real edges part of the computational graph
        above_threshold = (edge_scores >= threshold) & graph.real_edge_mask

        # Calculate degrees for each node
        for node_name, node in graph.nodes.items():
            if node_name == "logits":
                continue

            forward_idx = graph.forward_index(node, attn_slice=False)

            # Out-degree: number of outgoing edges above threshold
            out_degree = above_threshold[forward_idx, :].sum().item()

            # In-degree: number of incoming edges above threshold
            if isinstance(node, InputNode):
                in_degree = 0
            else:
                if hasattr(node, "qkv_inputs") and node.qkv_inputs:
                    # For attention nodes, I count an incoming edge for each qkv input
                    in_degree = 0
                    for qkv in ["q", "k", "v"]:
                        backward_idx = graph.backward_index(
                            node, qkv=qkv, attn_slice=False
                        )
                        in_degree += above_threshold[:, backward_idx].sum().item()
                else:
                    # For MLP nodes
                    backward_idx = graph.backward_index(node, attn_slice=False)
                    in_degree = above_threshold[:, backward_idx].sum().item()

            node_degrees[node_name] = {
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": in_degree + out_degree,
                "layer": node.layer,
                "node_type": type(node).__name__,
            }

        results[model_name] = dict(node_degrees)

    return results


def compare_node_degrees_across_models(
    degree_results: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Create a comparison DataFrame of node degrees across models.

    Args:
        degree_results: Results from analyze_node_degrees
        info_type: Type of information being analyzed (for labeling)

    Returns:
        DataFrame with node degree comparisons
    """
    comparison_data = []

    # Get all unique node names
    all_nodes = set()
    for model_results in degree_results.values():
        all_nodes.update(model_results.keys())

    for node_name in sorted(all_nodes):
        row = {"node": node_name}

        # Add degree information for each model!
        for model_name, model_results in degree_results.items():
            if node_name in model_results:
                node_info = model_results[node_name]
                row[f"{model_name}_in_degree"] = node_info["in_degree"]
                row[f"{model_name}_out_degree"] = node_info["out_degree"]
                row[f"{model_name}_total_degree"] = node_info["total_degree"]
                row["layer"] = node_info["layer"]
                row["node_type"] = node_info["node_type"]
            else:
                row[f"{model_name}_in_degree"] = 0
                row[f"{model_name}_out_degree"] = 0
                row[f"{model_name}_total_degree"] = 0

        comparison_data.append(row)

    return pd.DataFrame(comparison_data)
