#!/usr/bin/env python3
"""
ENHANCED True Coarsened GraphSAGE - TWO-LEVEL TRAINING VERSION

MAJOR FIX: Level 1 now has proper trainable parameters for intra-cluster learning
Level 1: Trainable intra-cluster GraphSAGE → super-node features
Level 2: Trainable inter-cluster GraphSAGE → super-node embeddings  
Level 3: Three-way spectral refinement → individualized node embeddings

Training Distribution:
- Level 1: ~30% of epochs (intra-cluster learning)
- Level 2: ~70% of epochs (inter-cluster learning)
"""

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
import time
import sys
import os
import scipy.sparse as sp
from scipy.sparse import diags
from scipy.linalg import eigh

# ============= LEVEL 1: TRAINABLE INTRA-CLUSTER GRAPHSAGE =============

class SimpleGraphSAGELayer(nn.Module):
    """Simple GraphSAGE layer for intra-cluster learning"""
    
    def __init__(self, input_dim, output_dim, aggregator='mean'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregator = aggregator
        
        # Learnable weight matrices
        self.W_self = nn.Linear(input_dim, output_dim, bias=False)
        self.W_neigh = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.xavier_uniform_(self.W_neigh.weight)
    
    def forward(self, node_features, adj_matrix):
        """
        Forward pass for GraphSAGE layer
        
        Args:
            node_features: [num_nodes, input_dim]
            adj_matrix: [num_nodes, num_nodes] adjacency matrix
        """
        # Self transformation
        h_self = self.W_self(node_features)
        
        # Neighbor aggregation
        if self.aggregator == 'mean':
            # Normalize adjacency matrix by degree
            degrees = adj_matrix.sum(dim=1, keepdim=True)
            degrees = torch.clamp(degrees, min=1.0)  # Avoid division by zero
            normalized_adj = adj_matrix / degrees
            h_neigh = torch.matmul(normalized_adj, node_features)
        else:
            # Simple sum aggregation
            h_neigh = torch.matmul(adj_matrix, node_features)
        
        h_neigh = self.W_neigh(h_neigh)
        
        # Combine self and neighbor representations
        output = h_self + h_neigh + self.bias
        
        return F.relu(output)

class IntraClusterGraphSAGE(nn.Module):
    """Trainable GraphSAGE for intra-cluster aggregation"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(SimpleGraphSAGELayer(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SimpleGraphSAGELayer(hidden_dim, hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.layers.append(SimpleGraphSAGELayer(hidden_dim, output_dim))
        else:
            # Single layer case
            self.layers[0] = SimpleGraphSAGELayer(input_dim, output_dim)
    
    def forward(self, node_features, adj_matrix):
        """Forward pass through all layers"""
        h = node_features
        
        for layer in self.layers:
            h = layer(h, adj_matrix)
        
        return h

def create_cluster_adjacency_tensor(cluster_subgraph):
    """Convert NetworkX cluster subgraph to PyTorch adjacency tensor"""
    nodes = sorted(list(cluster_subgraph.nodes()))
    node_map = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    
    adj = torch.zeros((n, n), dtype=torch.float32)
    
    for edge in cluster_subgraph.edges():
        i, j = node_map[edge[0]], node_map[edge[1]]
        adj[i, j] = 1.0
        adj[j, i] = 1.0  # Undirected
    
    return adj

def train_intra_cluster_model(cluster_subgraph, cluster_features, 
                             embed_dim=64, hidden_dim=32, 
                             epochs=50, lr=0.01):
    """
    Train GraphSAGE model for a single cluster
    
    Args:
        cluster_subgraph: NetworkX subgraph for this cluster
        cluster_features: numpy array of features for nodes in cluster
        embed_dim: output embedding dimension
        hidden_dim: hidden layer dimension
        epochs: training epochs
        lr: learning rate
    
    Returns:
        super_feature: aggregated embedding for the super-node
    """
    if len(cluster_features) == 0:
        return np.zeros(embed_dim)
    
    if len(cluster_features) == 1:
        # Single node - just transform dimension
        feature = cluster_features[0]
        result = np.zeros(embed_dim)
        result[:min(len(feature), embed_dim)] = feature[:min(len(feature), embed_dim)]
        return result
    
    # Convert to PyTorch tensors
    node_features = torch.tensor(cluster_features, dtype=torch.float32)
    adj_matrix = create_cluster_adjacency_tensor(cluster_subgraph)
    
    input_dim = cluster_features.shape[1]
    
    # Create model
    model = IntraClusterGraphSAGE(input_dim, hidden_dim, embed_dim, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Self-supervised training via reconstruction
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(node_features, adj_matrix)
        
        # Self-supervised loss: reconstruct node features from embeddings
        # Simple approach: minimize distance to input features (auto-encoder style)
        if embed_dim >= input_dim:
            reconstruction = embeddings[:, :input_dim]
        else:
            # Use a simple linear layer to map back
            reconstruction = torch.matmul(embeddings, embeddings.t()) @ node_features
            reconstruction = reconstruction / (embeddings.shape[0] + 1e-8)
        
        loss = F.mse_loss(reconstruction, node_features)
        
        # Add regularization for stability
        reg_loss = 0.01 * sum(p.pow(2.0).sum() for p in model.parameters())
        total_loss = loss + reg_loss
        
        total_loss.backward()
        optimizer.step()
    
    # Generate final super-node embedding
    model.eval()
    with torch.no_grad():
        final_embeddings = model(node_features, adj_matrix)
        # Pool to create single super-node feature (mean pooling)
        super_feature = torch.mean(final_embeddings, dim=0).numpy()
    
    return super_feature

def create_trainable_super_node_features(original_graph: nx.Graph,
                                       features: np.ndarray,
                                       clusters: List[List[int]],
                                       embed_dim: int = 64,
                                       hidden_dim: int = 32,
                                       level1_epochs: int = 50) -> np.ndarray:
    """
    ENHANCED Level 1: Create super-node features using TRAINABLE intra-cluster GraphSAGE
    
    Args:
        level1_epochs: Training epochs for Level 1 (distributed from total budget)
    """
    print(f"🔥 ENHANCED LEVEL 1: TRAINABLE INTRA-CLUSTER GRAPHSAGE")
    print(f"   Training epochs per cluster: {level1_epochs}")
    print(f"   Hidden dimension: {hidden_dim}")
    print(f"   Output embedding dimension: {embed_dim}")
    print("="*60)
    
    start_time = time.time()
    super_features = []
    
    for cluster_id, cluster_nodes in enumerate(clusters):
        if len(cluster_nodes) == 0:
            super_features.append(np.zeros(embed_dim))
            continue
        
        print(f"\n🎯 Training cluster {cluster_id}: {cluster_nodes}")
        
        # Extract cluster subgraph
        cluster_subgraph = original_graph.subgraph(cluster_nodes).copy()
        
        # Extract cluster features
        try:
            cluster_features = features[cluster_nodes]
        except (IndexError, TypeError):
            cluster_features = np.array([features[i] for i in cluster_nodes if i < len(features)])
            if len(cluster_features) == 0:
                super_features.append(np.zeros(embed_dim))
                continue
        
        # Train intra-cluster GraphSAGE model
        cluster_start = time.time()
        super_feature = train_intra_cluster_model(
            cluster_subgraph, cluster_features, 
            embed_dim=embed_dim, hidden_dim=hidden_dim, 
            epochs=level1_epochs, lr=0.01
        )
        cluster_time = time.time() - cluster_start
        
        super_features.append(super_feature)
        
        print(f"   ✅ Cluster {cluster_id} trained in {cluster_time:.3f}s")
        print(f"   Super-feature stats: mean={np.mean(super_feature):.4f}, std={np.std(super_feature):.4f}")
    
    super_features = np.array(super_features)
    total_time = time.time() - start_time
    
    print(f"\n📊 LEVEL 1 TRAINING SUMMARY:")
    print(f"   🎯 Trained {len(clusters)} intra-cluster models")
    print(f"   ⏱️  Total Level 1 training time: {total_time:.3f}s")
    print(f"   📈 Super-features shape: {super_features.shape}")
    print(f"   📊 Super-features stats: mean={np.mean(super_features):.4f}, std={np.std(super_features):.4f}")
    
    return super_features

# ============= TRAINING EPOCHS DISTRIBUTION =============

def calculate_training_distribution(total_epochs: int, 
                                  original_nodes: int,
                                  coarsened_nodes: int,
                                  strategy: str = "balanced") -> Tuple[int, int]:
    """
    Calculate training epochs distribution between Level 1 and Level 2
    
    Args:
        total_epochs: Total training budget
        original_nodes: Number of original nodes
        coarsened_nodes: Number of super-nodes
        strategy: "balanced", "level1_heavy", "level2_heavy"
    
    Returns:
        (level1_epochs, level2_epochs)
    """
    coarse_ratio = original_nodes / coarsened_nodes
    
    if strategy == "balanced":
        # 30% Level 1, 70% Level 2 (since Level 2 is more complex)
        level1_epochs = int(total_epochs * 0.3)
        level2_epochs = int(total_epochs * 0.7)
    
    elif strategy == "level1_heavy":
        # 50% Level 1, 50% Level 2 (equal focus)
        level1_epochs = int(total_epochs * 0.5)
        level2_epochs = int(total_epochs * 0.5)
    
    elif strategy == "level2_heavy":
        # 20% Level 1, 80% Level 2 (focus on global structure)
        level1_epochs = int(total_epochs * 0.2)
        level2_epochs = int(total_epochs * 0.8)
    
    elif strategy == "adaptive":
        # Adapt based on coarsening ratio
        if coarse_ratio > 3.0:  # Aggressive coarsening
            level1_epochs = int(total_epochs * 0.4)  # More local learning
            level2_epochs = int(total_epochs * 0.6)
        else:  # Mild coarsening
            level1_epochs = int(total_epochs * 0.25)
            level2_epochs = int(total_epochs * 0.75)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"📊 TRAINING EPOCHS DISTRIBUTION ({strategy}):")
    print(f"   🎯 Total budget: {total_epochs} epochs")
    print(f"   🔥 Level 1 (intra-cluster): {level1_epochs} epochs")
    print(f"   🚀 Level 2 (inter-cluster): {level2_epochs} epochs")
    print(f"   📈 Coarsening ratio: {coarse_ratio:.2f}x")
    
    return level1_epochs, level2_epochs

# ============= COMPUTATIONAL STRATEGY OPTIONS =============

def calculate_strategic_epochs(base_epochs: int,
                             original_nodes: int, 
                             coarsened_nodes: int,
                             strategy: str = "fair_comparison") -> int:
    """
    Calculate total training epochs based on computational strategy
    
    Args:
        base_epochs: Base epoch count (e.g., 1000)
        original_nodes: Number of original nodes
        coarsened_nodes: Number of super-nodes  
        strategy: "speed_advantage", "quality_advantage", "fair_comparison"
    
    Returns:
        total_epochs: Total epochs to use
    """
    coarse_ratio = original_nodes / coarsened_nodes
    
    if strategy == "speed_advantage":
        # Target 3-5x speedup
        efficiency_factor = 4.0
        total_epochs = max(50, int(base_epochs / (coarse_ratio * efficiency_factor)))
        print(f"🚀 SPEED ADVANTAGE: {total_epochs} epochs (targeting 3-5x speedup)")
    
    elif strategy == "quality_advantage":
        # Use computational savings to train longer
        complexity_factor = 1.5
        total_epochs = int(base_epochs / coarse_ratio * complexity_factor)
        print(f"🎯 QUALITY ADVANTAGE: {total_epochs} epochs (using saved computation)")
    
    elif strategy == "fair_comparison":
        # Match regular GraphZoom baseline
        total_epochs = int(base_epochs / coarse_ratio)
        print(f"⚖️ FAIR COMPARISON: {total_epochs} epochs (matching GraphZoom baseline)")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"   📊 Coarsening ratio: {coarse_ratio:.2f}x")
    print(f"   📈 Computational factor: {total_epochs / base_epochs:.2f}x vs base")
    
    return total_epochs

# ============= ENHANCED MAIN FUNCTION =============

def enhanced_true_coarsened_graphsage(original_graph: nx.Graph,
                                    features: np.ndarray,
                                    clusters: List[List[int]],
                                    coarsened_graph: nx.Graph,
                                    projections: List,
                                    laplacians: List,
                                    super_embed_dim: int = 64,
                                    final_embed_dim: int = 64,
                                    hidden_dim: int = 32,
                                    base_epochs: int = 1000,
                                    computational_strategy: str = "fair_comparison",
                                    training_distribution: str = "balanced",
                                    refinement_alpha: float = 0.76,
                                    refinement_beta: float = 0.12,
                                    refinement_gamma: float = 0.12) -> np.ndarray:
    """
    ENHANCED True Coarsened GraphSAGE with TWO-LEVEL TRAINING
    
    Level 1: TRAINABLE intra-cluster GraphSAGE → super-node features
    Level 2: TRAINABLE inter-cluster GraphSAGE → super-node embeddings  
    Level 3: Three-way spectral refinement → individualized node embeddings
    
    Args:
        computational_strategy: "speed_advantage", "quality_advantage", "fair_comparison"
        training_distribution: "balanced", "level1_heavy", "level2_heavy", "adaptive"
    """
    
    print("🚀" * 20)
    print("ENHANCED TRUE COARSENED GRAPHSAGE - TWO-LEVEL TRAINING")
    print("🚀" * 20)
    
    total_start = time.time()
    
    # Calculate strategic training epochs
    total_epochs = calculate_strategic_epochs(
        base_epochs, original_graph.number_of_nodes(), 
        coarsened_graph.number_of_nodes(), computational_strategy
    )
    
    # Distribute epochs between levels
    level1_epochs, level2_epochs = calculate_training_distribution(
        total_epochs, original_graph.number_of_nodes(),
        coarsened_graph.number_of_nodes(), training_distribution
    )
    
    # LEVEL 1: TRAINABLE Intra-cluster GraphSAGE
    print(f"\n🔥 LEVEL 1: TRAINABLE INTRA-CLUSTER GRAPHSAGE")
    super_features = create_trainable_super_node_features(
        original_graph, features, clusters, 
        embed_dim=super_embed_dim, hidden_dim=hidden_dim,
        level1_epochs=level1_epochs
    )
    
    # LEVEL 2: TRAINABLE Inter-cluster GraphSAGE (FIXED EPOCHS)
    print(f"\n🚀 LEVEL 2: TRAINABLE INTER-CLUSTER GRAPHSAGE")
    print(f"   Training epochs: {level2_epochs}")
    
    # Import and use existing GraphSAGE with CORRECTED epochs
    from embed_methods.graphsage.graphsage import graphsage
    
    # Set required attributes
    nx.set_node_attributes(coarsened_graph, False, "test")
    nx.set_node_attributes(coarsened_graph, False, "val")
    
    level2_start = time.time()
    super_embeddings = graphsage(
        coarsened_graph,
        super_features,
        "mean",  # aggregator
        True,    # weighted
        level2_epochs  # FIXED: Use calculated epochs, not broken formula
    )
    level2_time = time.time() - level2_start
    
    print(f"   ✅ Level 2 completed in {level2_time:.3f}s")
    print(f"   📊 Super-embeddings shape: {super_embeddings.shape}")
    
    # LEVEL 3: Three-way spectral refinement (unchanged)
    print(f"\n✨ LEVEL 3: THREE-WAY SPECTRAL REFINEMENT")
    
    # Create spectral contexts (reuse existing function)
    super_embeddings, spectral_contexts = create_spectral_enhanced_super_embeddings(
        clusters, original_graph, super_embeddings
    )
    
    # Apply three-way refinement
    final_embeddings = three_way_spectral_refinement(
        super_embeddings=super_embeddings,
        spectral_contexts=spectral_contexts,
        projections=projections,
        laplacians=laplacians,
        original_features=features,
        clusters=clusters,
        alpha=refinement_alpha,
        beta=refinement_beta,
        gamma=refinement_gamma,
        lda=0.1,
        power=False
    )
    
    total_time = time.time() - total_start
    
    # Performance summary
    print(f"\n📊 ENHANCED TRUE COARSENED GRAPHSAGE SUMMARY:")
    print(f"   🎯 Computational strategy: {computational_strategy}")
    print(f"   📈 Training distribution: {training_distribution}")
    print(f"   🔥 Level 1 epochs: {level1_epochs}")
    print(f"   🚀 Level 2 epochs: {level2_epochs}")
    print(f"   ⏱️  Total time: {total_time:.3f}s")
    print(f"   📊 Final embeddings: {final_embeddings.shape}")
    print(f"   🎯 Coarsening: {original_graph.number_of_nodes()} → {coarsened_graph.number_of_nodes()} nodes")
    
    # Calculate theoretical speedup
    regular_epochs = int(base_epochs / (original_graph.number_of_nodes() / coarsened_graph.number_of_nodes()))
    speedup = regular_epochs / total_epochs if total_epochs > 0 else 1
    print(f"   ⚡ Computational efficiency: {speedup:.2f}x vs regular GraphZoom")
    
    return final_embeddings

# Import existing functions (spectral refinement etc.)
# These are unchanged from your existing implementation
def create_spectral_enhanced_super_embeddings(clusters, original_graph, super_embeddings):
    """Reuse existing implementation - unchanged"""
    # This should import from your existing true_coarsened_graphsage.py
    from true_coarsened_graphsage import create_spectral_enhanced_super_embeddings as orig_func
    return orig_func(clusters, original_graph, super_embeddings)

def three_way_spectral_refinement(super_embeddings, spectral_contexts, projections, 
                                 laplacians, original_features, clusters, alpha, beta, gamma, lda, power):
    """Reuse existing implementation - unchanged"""
    from true_coarsened_graphsage import three_way_spectral_refinement as orig_func
    return orig_func(super_embeddings, spectral_contexts, projections, laplacians, 
                    original_features, clusters, alpha, beta, gamma, lda, power)

if __name__ == "__main__":
    print("Enhanced True Coarsened GraphSAGE with Two-Level Training")
    print("This module provides trainable Level 1 intra-cluster learning")
    print("Use enhanced_true_coarsened_graphsage() as main entry point")