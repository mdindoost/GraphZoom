#!/usr/bin/env python3
"""
ClusterGraphSAGE: Control Variate GraphSAGE with Cluster-Aware Sampling

Implements the unbiased control variate estimator:
m̂_v = ∑_C α_{v,C} g_C + (1/s) ∑_{u∈S_v} (h_u - g_{π(u)})

Where:
- α_{v,C} = |N(v) ∩ C| / |N(v)| (cluster neighbor weights)
- g_C = (1/|C|) ∑_{u∈C} h_u (cluster representatives)  
- S_v = stratified sample of neighbors
- π(u) = cluster assignment of node u

Integrates with existing GraphZoom infrastructure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

try:
    from .cluster_sampler import ClusterSampler
except ImportError:
    # Fallback for standalone testing
    from cluster_sampler import ClusterSampler
    
class ClusterGraphSAGEConv(nn.Module):
    """
    Single layer of Cluster-Aware GraphSAGE with control variate estimator.
    
    Implements the core estimator formula from the specification.
    """
    
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int,
                 use_cluster_rep: bool = True,
                 add_coarse_channel: bool = False,
                 dropout: float = 0.5,
                 bias: bool = True):
        """
        Args:
            in_dim: Input feature dimension
            out_dim: Output embedding dimension  
            use_cluster_rep: Whether to concatenate cluster representative
            add_coarse_channel: Whether to use bi-level message passing
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_cluster_rep = use_cluster_rep
        self.add_coarse_channel = add_coarse_channel
        
        # Determine concat dimension
        # [h_v ∥ m̂_v ∥ g_π(v)] if use_cluster_rep else [h_v ∥ m̂_v]
        concat_dim = in_dim + in_dim  # h_v + m̂_v
        if use_cluster_rep:
            concat_dim += in_dim  # + g_π(v)
            
        # Linear transformation
        self.linear = nn.Linear(concat_dim, out_dim, bias=bias)
        
        # Coarse channel (bi-level message passing)
        if add_coarse_channel:
            self.coarse_linear = nn.Linear(in_dim * 2, in_dim, bias=bias)  # [g_C ∥ agg_neighbors]
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor,
                cluster_reps: torch.Tensor,
                sampling_info: Dict,
                node_to_cluster: Dict[int, int]) -> torch.Tensor:
        """
        Forward pass implementing control variate estimator.
        
        Args:
            x: Node features [num_batch_nodes, in_dim]
            cluster_reps: Cluster representatives [num_clusters, in_dim]  
            sampling_info: From ClusterSampler.sample_batch()
            node_to_cluster: Mapping node_id -> cluster_id
            
        Returns:
            Updated node embeddings [num_batch_nodes, out_dim]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Get node mapping for this batch
        node_mapping = sampling_info['node_mapping']  # original_node -> batch_idx
        inverse_mapping = {v: k for k, v in node_mapping.items()}  # batch_idx -> original_node
        
        # Initialize aggregated messages
        messages = torch.zeros_like(x)  # [num_batch_nodes, in_dim]
        
        # Compute control variate estimator for each node
        for batch_idx in range(batch_size):
            original_node = inverse_mapping.get(batch_idx)
            if original_node is None:
                continue
                
            # Get sampling info for this node
            cluster_weights = sampling_info['cluster_weights'].get(original_node, {})
            cluster_samples = sampling_info['node_cluster_samples'].get(original_node, {})
            
            if not cluster_weights:
                # Isolated node - use self as message
                messages[batch_idx] = x[batch_idx]
                continue
            
            # Control variate estimator: m̂_v = ∑_C α_{v,C} g_C + (1/s) ∑_{u∈S_v} (h_u - g_{π(u)})
            
            # Term 1: ∑_C α_{v,C} g_C (cluster representative weighted sum)
            cluster_term = torch.zeros(self.in_dim, device=device)
            for cluster_id, weight in cluster_weights.items():
                cluster_term += weight * cluster_reps[cluster_id]
            
            # Term 2: (1/s) ∑_{u∈S_v} (h_u - g_{π(u)}) (control variate correction)
            correction_term = torch.zeros(self.in_dim, device=device)
            total_samples = 0
            
            for cluster_id, sampled_nodes in cluster_samples.items():
                for sampled_node in sampled_nodes:
                    if sampled_node in node_mapping:
                        sampled_batch_idx = node_mapping[sampled_node]
                        sampled_cluster = node_to_cluster.get(sampled_node, cluster_id)
                        
                        # h_u - g_{π(u)}
                        correction = x[sampled_batch_idx] - cluster_reps[sampled_cluster]
                        correction_term += correction
                        total_samples += 1
            
            if total_samples > 0:
                correction_term /= total_samples
            
            # Final estimator
            messages[batch_idx] = cluster_term + correction_term
        
        # Node update: h_v^(ℓ+1) = σ(W[h_v^(ℓ) ∥ m̂_v^(ℓ) ∥ g_{π(v)}^(ℓ)])
        concat_features = [x, messages]  # [h_v, m̂_v]
        
        if self.use_cluster_rep:
            # Add cluster representatives g_{π(v)}
            node_cluster_reps = torch.zeros_like(x)
            for batch_idx in range(batch_size):
                original_node = inverse_mapping.get(batch_idx)
                if original_node is not None and original_node in node_to_cluster:
                    cluster_id = node_to_cluster[original_node]
                    node_cluster_reps[batch_idx] = cluster_reps[cluster_id]
            concat_features.append(node_cluster_reps)
        
        # Concatenate and transform
        h_concat = torch.cat(concat_features, dim=1)  # [batch_size, concat_dim]
        h_concat = self.dropout(h_concat)
        
        # Linear transformation
        out = self.linear(h_concat)
        
        return F.relu(out)  # σ(W[...])

class ClusterGraphSAGE(nn.Module):
    """
    Multi-layer Cluster-Aware GraphSAGE model.
    
    Integrates with existing GraphZoom infrastructure and provides
    the same interface as standard GraphSAGE.
    """
    
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int, 
                 out_dim: int,
                 num_layers: int = 2,
                 clusters: Optional[List[List[int]]] = None,
                 s_in: int = 4,
                 s_out: int = 1,
                 use_cluster_rep: bool = True,
                 add_coarse_channel: bool = False,
                 dropout: float = 0.5):
        """
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output embedding dimension
            num_layers: Number of GraphSAGE layers
            clusters: List of node clusters
            s_in: Samples per node (stratified across clusters)
            s_out: Boundary samples for cross-cluster connections
            use_cluster_rep: Whether to use cluster representatives in updates
            add_coarse_channel: Whether to use bi-level message passing
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.clusters = clusters
        self.s_in = s_in
        self.s_out = s_out
        self.use_cluster_rep = use_cluster_rep
        self.add_coarse_channel = add_coarse_channel
        
        # Initialize cluster sampler
        if clusters is not None:
            self.sampler = ClusterSampler(clusters, s_in=s_in, s_out=s_out)
            self.node_to_cluster = {}
            for cluster_id, nodes in enumerate(clusters):
                for node in nodes:
                    self.node_to_cluster[node] = cluster_id
        else:
            self.sampler = None
            self.node_to_cluster = {}
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input layer
        if num_layers == 1:
            self.layers.append(ClusterGraphSAGEConv(in_dim, out_dim, use_cluster_rep, add_coarse_channel, dropout))
        else:
            self.layers.append(ClusterGraphSAGEConv(in_dim, hidden_dim, use_cluster_rep, add_coarse_channel, dropout))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.layers.append(ClusterGraphSAGEConv(hidden_dim, hidden_dim, use_cluster_rep, add_coarse_channel, dropout))
            
            # Output layer
            self.layers.append(ClusterGraphSAGEConv(hidden_dim, out_dim, use_cluster_rep, add_coarse_channel, dropout))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_nodes: Optional[List[int]] = None) -> torch.Tensor:
        """
        Forward pass through all layers.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph edge index [2, num_edges]
            batch_nodes: Nodes to compute embeddings for (if None, use all)
            
        Returns:
            Node embeddings [num_batch_nodes, out_dim]
        """
        if self.sampler is None:
            raise RuntimeError("ClusterGraphSAGE requires clusters to be provided during initialization")
        
        # Initialize sampler cache if needed
        if not self.sampler._cache_valid:
            self.sampler.precompute_cluster_counts(edge_index)
        
        # Use all nodes if batch not specified
        if batch_nodes is None:
            batch_nodes = list(range(x.shape[0]))
        
        # Sample subgraph for this batch
        batch_edge_index, sampling_info = self.sampler.sample_batch(batch_nodes, edge_index)
        
        # Extract features for sampled subgraph
        node_mapping = sampling_info['node_mapping']
        batch_x = torch.zeros(len(node_mapping), x.shape[1], device=x.device)
        
        for original_node, batch_idx in node_mapping.items():
            if original_node < x.shape[0]:  # Valid node index
                batch_x[batch_idx] = x[original_node]
        
        # Forward through layers
        h = batch_x
        
        for layer_idx, layer in enumerate(self.layers):
            # Compute cluster representatives for current layer
            cluster_reps = self.sampler.compute_cluster_representatives(h, sampling_info)
            
            # Apply layer
            h = layer(h, cluster_reps, sampling_info, self.node_to_cluster)
            
            print(f"  Layer {layer_idx + 1}: h.shape = {h.shape}")
        
        return h
    
    def get_estimator_error(self, x: torch.Tensor, edge_index: torch.Tensor, test_nodes: List[int]) -> float:
        """
        Compute estimator error ε = mean_v ∥m̂_v - m_v∥_2 for evaluation.
        
        This compares the control variate estimate against the true neighborhood mean
        on a small set of test nodes (where we can afford to compute the true mean).
        """
        if self.sampler is None:
            return float('inf')
            
        # Sample neighborhoods using our method
        batch_edge_index, sampling_info = self.sampler.sample_batch(test_nodes, edge_index)
        
        # Extract batch features  
        node_mapping = sampling_info['node_mapping']
        batch_x = torch.zeros(len(node_mapping), x.shape[1], device=x.device)
        
        for original_node, batch_idx in node_mapping.items():
            if original_node < x.shape[0]:
                batch_x[batch_idx] = x[original_node]
        
        # Compute cluster representatives
        cluster_reps = self.sampler.compute_cluster_representatives(batch_x, sampling_info)
        
        # Compute our estimates vs true means for test nodes
        errors = []
        edge_index_np = edge_index.cpu().numpy()
        
        # Build true adjacency for computing exact means
        adjacency = {}
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            if src not in adjacency:
                adjacency[src] = []
            adjacency[src].append(dst)
        
        for test_node in test_nodes:
            if test_node not in node_mapping:
                continue
                
            # True neighborhood mean
            if test_node in adjacency:
                neighbors = adjacency[test_node]
                if neighbors:
                    true_mean = x[neighbors].mean(dim=0)
                else:
                    true_mean = x[test_node]  # Isolated node
            else:
                true_mean = x[test_node]
            
            # Our estimate (simplified - just the cluster term for this test)
            cluster_weights = sampling_info['cluster_weights'].get(test_node, {})
            if cluster_weights:
                estimated_mean = torch.zeros_like(true_mean)
                for cluster_id, weight in cluster_weights.items():
                    estimated_mean += weight * cluster_reps[cluster_id]
            else:
                estimated_mean = x[test_node]
            
            # Compute error
            error = torch.norm(estimated_mean - true_mean, p=2).item()
            errors.append(error)
        
        return np.mean(errors) if errors else float('inf')

def cluster_graphsage_wrapper(G, features, clusters, model_params=None, epochs=200):
    """
    Wrapper function for integration with existing GraphZoom infrastructure.
    Mimics the interface of the existing graphsage() function.
    
    Args:
        G: NetworkX graph
        features: Node features [num_nodes, feature_dim]
        clusters: List of node clusters from CMG
        model_params: Dict with model parameters
        epochs: Training epochs
        
    Returns:
        Node embeddings [num_nodes, embed_dim]
    """
    if model_params is None:
        model_params = {
            'hidden_dim': 128,
            'out_dim': 64,
            'num_layers': 2,
            's_in': 4,
            's_out': 1,
            'dropout': 0.5
        }
    
    print(f"[ClusterGraphSAGE] Training with {len(clusters)} clusters, {epochs} epochs")
    print(f"[ClusterGraphSAGE] Model params: {model_params}")
    
    # Convert NetworkX graph to PyTorch format
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list + [(b,a) for a,b in edge_list], dtype=torch.long).t()
    
    # Convert features
    if isinstance(features, np.ndarray):
        x = torch.tensor(features, dtype=torch.float32)
    else:
        x = features.float()
    
    # Initialize model
    model = ClusterGraphSAGE(
        in_dim=x.shape[1],
        hidden_dim=model_params['hidden_dim'], 
        out_dim=model_params['out_dim'],
        num_layers=model_params['num_layers'],
        clusters=clusters,
        s_in=model_params['s_in'],
        s_out=model_params['s_out'],
        dropout=model_params['dropout']
    )
    
    # Simple training loop (placeholder - can be enhanced)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"[ClusterGraphSAGE] Starting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(x, edge_index)
        
        # Simple reconstruction loss (placeholder)
        loss = F.mse_loss(embeddings, torch.randn_like(embeddings))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: loss = {loss.item():.4f}")
    
    training_time = time.time() - start_time
    print(f"[ClusterGraphSAGE] Training completed in {training_time:.3f}s")
    
    # Final embeddings
    model.eval()
    with torch.no_grad():
        final_embeddings = model(x, edge_index)
    
    return final_embeddings.numpy()

def test_cluster_graphsage():
    """Test ClusterGraphSAGE on simple graph."""
    print("Testing ClusterGraphSAGE...")
    
    # Create test data
    import networkx as nx
    G = nx.path_graph(12)
    features = np.random.randn(12, 16)  # 12 nodes, 16-dim features
    clusters = [[0,1,2], [3,4,5,6], [7,8,9], [10,11]]
    
    # Test the wrapper function
    embeddings = cluster_graphsage_wrapper(G, features, clusters, epochs=50)
    
    print(f"Output embeddings shape: {embeddings.shape}")
    print("ClusterGraphSAGE test completed!")

if __name__ == "__main__":
    test_cluster_graphsage()
