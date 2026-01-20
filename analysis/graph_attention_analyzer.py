import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional
from gencircuits.eap.graph import Graph, AttentionNode, MLPNode, InputNode
import os


class GraphAttentionAnalyzer:
    """
    Attention analyzer that extracts attention head scores from a Graph object
    and generates visualizations to understand high-scoring heads across layers.
    """
    
    def __init__(self, graph: Graph):
        """
        Initialize the analyzer with a graph.
        
        Args:
            graph: The Graph object containing node and edge scores
        """
        self.graph = graph
        self.n_layers = graph.cfg['n_layers']
        self.n_heads = graph.cfg['n_heads'] 
        
        # Extract attention nodes for analysis
        self.attention_nodes = self._extract_attention_nodes()
        
    def _extract_attention_nodes(self) -> Dict[Tuple[int, int], AttentionNode]:
        """
        Extract all attention nodes from the graph.
        
        Returns:
            Dictionary mapping (layer, head) tuples to AttentionNode objects
        """
        attention_nodes = {}
        
        for node_name, node in self.graph.nodes.items():
            if isinstance(node, AttentionNode):
                attention_nodes[(node.layer, node.head)] = node
                
        return attention_nodes
    
    def extract_attention_scores(self, 
                                score_type: str = "node_scores",
                                absolute: bool = True,
                                threshold: float = 0.0) -> np.ndarray:
        """
        Extract attention head scores from the graph.
        
        Args:
            score_type: Type of scores to extract ("node_scores" or "edge_scores")
            absolute: Whether to use absolute values of scores
            threshold: Minimum score threshold to consider
            
        Returns:
            2D numpy array of shape (n_layers, n_heads) with attention head scores
        """
        scores_matrix = np.zeros((self.n_layers, self.n_heads))
        
        if score_type == "node_scores":
            scores_matrix = self._extract_node_scores(absolute, threshold)
        elif score_type == "edge_scores":
            scores_matrix = self._extract_edge_scores(absolute, threshold)
        else:
            raise ValueError(f"Unknown score_type: {score_type}. Use 'node_scores' or 'edge_scores'")
            
        return scores_matrix
    
    def _extract_node_scores(self, absolute: bool, threshold: float) -> np.ndarray:
        """Extract node scores for attention heads."""
        scores_matrix = np.zeros((self.n_layers, self.n_heads))
        
        if self.graph.nodes_scores is None:
            print("Warning: Graph has no node scores, returning zeros")
            return scores_matrix
            
        for (layer, head), node in self.attention_nodes.items():
            try:
                # Get the score for this attention node
                score = node.score
                if score is not None:
                    if absolute:
                        score = abs(score)
                    if score >= threshold:
                        scores_matrix[layer, head] = score
            except Exception as e:
                print(f"Warning: Could not get score for node {node.name}: {e}")
                
        return scores_matrix
    
    def _extract_edge_scores(self, absolute: bool, threshold: float) -> np.ndarray:
        """Extract aggregated incoming edge scores for attention heads."""
        scores_matrix = np.zeros((self.n_layers, self.n_heads))
        
        for (layer, head), node in self.attention_nodes.items():
            try:
                # Get all incoming edges to this attention node
                incoming_scores = []
                
                # For attention nodes, we need to consider q, k, v inputs
                if hasattr(node, "qkv_inputs") and node.qkv_inputs:
                    for qkv in ["q", "k", "v"]:
                        backward_idx = self.graph.backward_index(node, qkv=qkv, attn_slice=False)
                        # Get scores for edges coming into this node
                        edge_scores = self.graph.scores[:, backward_idx]
                        
                        if absolute:
                            edge_scores = torch.abs(edge_scores)
                            
                        # Filter by threshold and real edges
                        real_edges = self.graph.real_edge_mask[:, backward_idx]
                        valid_scores = edge_scores[(edge_scores >= threshold) & real_edges]
                        
                        if len(valid_scores) > 0:
                            incoming_scores.extend(valid_scores.tolist())
                
                # Aggregate the incoming scores (using sum, could also use mean)
                if incoming_scores:
                    scores_matrix[layer, head] = sum(incoming_scores)
                    
            except Exception as e:
                print(f"Warning: Could not get edge scores for node {node.name}: {e}")
                
        return scores_matrix
    
    def plot_attention_heatmap(self,
                              scores_matrix: Optional[np.ndarray] = None,
                              score_type: str = "node_scores",
                              absolute: bool = True,
                              threshold: float = 0.0,
                              figsize: Tuple[int, int] = (10, 6),
                              title: str = None,
                              save_path: str = None) -> plt.Figure:
        """
        Plot a heatmap of attention head scores across layers and heads.
        
        Args:
            scores_matrix: Pre-computed scores matrix. If None, will compute from graph
            score_type: Type of scores to extract if scores_matrix is None
            absolute: Whether to use absolute values
            threshold: Score threshold
            figsize: Figure size
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if scores_matrix is None:
            scores_matrix = self.extract_attention_scores(score_type, absolute, threshold)
            
        # Create the heatmap
        plt.figure(figsize=figsize)
        
        # Use a colormap that highlights high values
        cmap = "viridis" if not absolute else "plasma"
        
        sns.heatmap(
            scores_matrix,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            xticklabels=[f"H{i}" for i in range(self.n_heads)],
            yticklabels=[f"L{i}" for i in range(self.n_layers)],
            cbar_kws={"label": "Score"}
        )
        
        if title is None:
            score_desc = "Absolute" if absolute else "Raw"
            title = f"{score_desc} {score_type.replace('_', ' ').title()} - Attention Heads (threshold≥{threshold})"
            
        plt.title(title, fontsize=14)
        plt.xlabel("Attention Head", fontsize=12)
        plt.ylabel("Layer", fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def analyze_top_scoring_heads(self,
                                 scores_matrix: Optional[np.ndarray] = None,
                                 score_type: str = "node_scores",
                                 absolute: bool = True,
                                 threshold: float = 0.0,
                                 top_k: int = 10) -> pd.DataFrame:
        """
        Identify and analyze the top-k scoring attention heads.
        
        Args:
            scores_matrix: Pre-computed scores matrix
            score_type: Type of scores to extract if scores_matrix is None
            absolute: Whether to use absolute values
            threshold: Score threshold
            top_k: Number of top heads to return
            
        Returns:
            DataFrame with top scoring heads and their properties
        """
        if scores_matrix is None:
            scores_matrix = self.extract_attention_scores(score_type, absolute, threshold)
            
        # Find top-k heads
        flat_indices = np.argsort(scores_matrix.flatten())[::-1][:top_k]
        top_heads = []
        
        for flat_idx in flat_indices:
            layer, head = np.unravel_index(flat_idx, scores_matrix.shape)
            score = scores_matrix[layer, head]
            
            if score > 0:  # Only include heads with non-zero scores
                top_heads.append({
                    'layer': layer,
                    'head': head,
                    'score': score,
                    'node_name': f'a{layer}.h{head}',
                    'rank': len(top_heads) + 1
                })
                
        return pd.DataFrame(top_heads)
    
    def plot_layer_aggregated_scores(self,
                                   scores_matrix: Optional[np.ndarray] = None,
                                   score_type: str = "node_scores",
                                   absolute: bool = True,
                                   threshold: float = 0.0,
                                   aggregation: str = "sum",
                                   figsize: Tuple[int, int] = (10, 6),
                                   save_path: str = None) -> plt.Figure:
        """
        Plot layer-aggregated attention scores.
        
        Args:
            scores_matrix: Pre-computed scores matrix
            score_type: Type of scores to extract if scores_matrix is None
            absolute: Whether to use absolute values
            threshold: Score threshold
            aggregation: How to aggregate across heads ("sum", "mean", "max")
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if scores_matrix is None:
            scores_matrix = self.extract_attention_scores(score_type, absolute, threshold)
            
        # Aggregate across heads
        if aggregation == "sum":
            layer_scores = np.sum(scores_matrix, axis=1)
        elif aggregation == "mean":
            layer_scores = np.mean(scores_matrix, axis=1)
        elif aggregation == "max":
            layer_scores = np.max(scores_matrix, axis=1)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
            
        # Create bar plot
        plt.figure(figsize=figsize)
        bars = plt.bar(range(self.n_layers), layer_scores, alpha=0.7)
        
        # Color bars by score magnitude
        norm = plt.Normalize(vmin=min(layer_scores), vmax=max(layer_scores))
        cmap = plt.cm.viridis
        for bar, score in zip(bars, layer_scores):
            bar.set_color(cmap(norm(score)))
            
        plt.xlabel("Layer", fontsize=12)
        plt.ylabel(f"{aggregation.title()} Score", fontsize=12)
        plt.title(f"Layer-wise {aggregation.title()} of Attention Head Scores", fontsize=14)
        plt.xticks(range(self.n_layers), [f"L{i}" for i in range(self.n_layers)])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def compare_multiple_graphs(self,
                               graphs: Dict[str, Graph],
                               score_type: str = "node_scores",
                               absolute: bool = True,
                               threshold: float = 0.0,
                               figsize: Tuple[int, int] = (15, 10),
                               save_path: str = None) -> plt.Figure:
        """
        Compare attention scores across multiple graphs (e.g., different models).
        
        Args:
            graphs: Dictionary mapping graph names to Graph objects
            score_type: Type of scores to extract
            absolute: Whether to use absolute values
            threshold: Score threshold
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        n_graphs = len(graphs)
        fig, axes = plt.subplots(1, n_graphs, figsize=figsize)
        
        if n_graphs == 1:
            axes = [axes]
            
        for idx, (graph_name, graph) in enumerate(graphs.items()):
            analyzer = GraphAttentionAnalyzer(graph)
            scores_matrix = analyzer.extract_attention_scores(score_type, absolute, threshold)
            
            ax = axes[idx]
            
            # Plot heatmap on this subplot
            sns.heatmap(
                scores_matrix,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                ax=ax,
                xticklabels=[f"H{i}" for i in range(self.n_heads)],
                yticklabels=[f"L{i}" for i in range(self.n_layers)] if idx == 0 else False,
                cbar=True
            )
            
            ax.set_title(graph_name, fontsize=12)
            ax.set_xlabel("Head", fontsize=10)
            if idx == 0:
                ax.set_ylabel("Layer", fontsize=10)
                
        plt.suptitle(f"Attention Head Scores Comparison ({score_type})", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


def analyze_attention_across_tasks(graph_paths: Dict[str, str],
                                  score_type: str = "node_scores",
                                  absolute: bool = True,
                                  threshold: float = 0.0,
                                  save_dir: str = "./plots/attention_analysis") -> Dict[str, np.ndarray]:
    """
    Utility function to analyze attention across multiple tasks/models.
    
    Args:
        graph_paths: Dictionary mapping task/model names to graph file paths
        score_type: Type of scores to extract
        absolute: Whether to use absolute values
        threshold: Score threshold
        save_dir: Directory to save plots
        
    Returns:
        Dictionary mapping task names to score matrices
    """
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    
    # Load graphs and extract scores
    graphs = {}
    for name, path in graph_paths.items():
        try:
            graph = Graph.from_json(path)
            graphs[name] = graph
            
            analyzer = GraphAttentionAnalyzer(graph)
            scores_matrix = analyzer.extract_attention_scores(score_type, absolute, threshold)
            results[name] = scores_matrix
            
            # Generate individual heatmap
            fig = analyzer.plot_attention_heatmap(
                scores_matrix=scores_matrix,
                title=f"Attention Head Scores - {name}",
                save_path=os.path.join(save_dir, f"attention_heatmap_{name}.pdf")
            )
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not process {name} from {path}: {e}")
            
    # Generate comparison plot if we have multiple graphs
    if len(graphs) > 1:
        first_analyzer = GraphAttentionAnalyzer(list(graphs.values())[0])
        fig = first_analyzer.compare_multiple_graphs(
            graphs=graphs,
            score_type=score_type,
            absolute=absolute,
            threshold=threshold,
            save_path=os.path.join(save_dir, "attention_comparison.pdf")
        )
        plt.close(fig)
        
    return results
