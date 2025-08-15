#!/usr/bin/env python3
"""
ClusterSampler: Stratified neighborhood sampling for cluster-aware GraphSAGE

Implements the control variate estimator sampling strategy:
- Proportional sampling across clusters: s_C ∝ α_{v,C}
- Unbiased within-cluster sampling
- Efficient batch composition with cluster representatives

Integrates with existing GraphZoom infrastructure.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
from collections import defaultdict
import time
from typing import Dict, List, Tuple, Optional, Union

class ClusterSampler:
    """
    Stratified neighborhood sampler for cluster-aware GraphSAGE.
    
    Core idea: Instead of uniform random sampling, sample proportionally
    from clusters, then apply control variate correction for unbiased estimation.
    """
    
    def __init__(self, 
                 clusters: List[List[int]], 
                 s_in: int = 4, 
                 s_out: int = 1,
                 device: str = 'cpu',
                 cache_enabled: bool = True,
                 stratified: bool = True,
                 proportional: bool = True):
        """
        Args:
            clusters: List of node lists, clusters[k] = nodes in cluster k
            s_in: Total inner samples per node
            s_out: Boundary samples for cross-cluster connections  
            device: torch device
            cache_enabled: Whether to cache cluster neighbor counts
            stratified: Whether to use stratified sampling (vs uniform)
            proportional: Whether s_C ∝ α_{v,C} (vs equal allocation)
        """
        self.clusters = clusters
        self.s_in = s_in
        self.s_out = s_out
        self.device = device
        self.cache_enabled = cache_enabled
        self.stratified = stratified
        self.proportional = proportional
        
        # Build cluster mappings
        self.num_clusters = len(clusters)
        self.node_to_cluster = {}
        for cluster_id, nodes in enumerate(clusters):
            for node in nodes:
                self.node_to_cluster[node] = cluster_id
        
        # Cache for efficiency
        self._cluster_neighbor_counts = {}  # node_id -> {cluster_id: count}
        self._adjacency_by_cluster = {}     # cluster_id -> neighbor lists
        self._cache_valid = False
        
        print(f"[ClusterSampler] Initialized with {self.num_clusters} clusters")
        print(f"  s_in={s_in}, s_out={s_out}, stratified={stratified}")
        
    def precompute_cluster_counts(self, edge_index: torch.Tensor):
        """
        Precompute |N(v) ∩ C| for all nodes v and clusters C.
        This is the core efficiency optimization.
        """
        print("[ClusterSampler] Precomputing cluster neighbor counts...")
        start_time = time.time()
        
        # Convert edge_index to adjacency dict for efficiency
        edge_index_np = edge_index.cpu().numpy()
        adjacency = defaultdict(list)
        
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            adjacency[src].append(dst)
        
        # For each node, count neighbors per cluster
        for node_id in adjacency:
            cluster_counts = defaultdict(int)
            
            for neighbor in adjacency[node_id]:
                if neighbor in self.node_to_cluster:
                    neighbor_cluster = self.node_to_cluster[neighbor]
                    cluster_counts[neighbor_cluster] += 1
            
            self._cluster_neighbor_counts[node_id] = dict(cluster_counts)
        
        # Also group adjacency by cluster for efficient sampling
        for node_id in adjacency:
            neighbors_by_cluster = defaultdict(list)
            for neighbor in adjacency[node_id]:
                if neighbor in self.node_to_cluster:
                    neighbor_cluster = self.node_to_cluster[neighbor]
                    neighbors_by_cluster[neighbor_cluster].append(neighbor)
            self._adjacency_by_cluster[node_id] = dict(neighbors_by_cluster)
        
        self._cache_valid = True
        elapsed = time.time() - start_time
        print(f"[ClusterSampler] Precomputation completed in {elapsed:.3f}s")
        
    def compute_cluster_weights(self, node_id: int) -> Dict[int, float]:
        """
        Compute α_{v,C} = |N(v) ∩ C| / |N(v)| for node v.
        
        Returns:
            Dict mapping cluster_id -> weight α_{v,C}
        """
        if not self._cache_valid:
            raise RuntimeError("Must call precompute_cluster_counts() first")
            
        cluster_counts = self._cluster_neighbor_counts.get(node_id, {})
        total_neighbors = sum(cluster_counts.values())
        
        if total_neighbors == 0:
            return {}  # Isolated node
        
        # Compute normalized weights
        weights = {}
        for cluster_id, count in cluster_counts.items():
            weights[cluster_id] = count / total_neighbors
            
        return weights
        
    def stratified_sample(self, node_id: int) -> Tuple[List[int], Dict[int, List[int]]]:
        """
        Perform stratified sampling for node_id.
        
        Returns:
            sampled_neighbors: List of sampled neighbor node IDs
            samples_by_cluster: Dict mapping cluster_id -> sampled nodes from that cluster
        """
        if not self._cache_valid:
            raise RuntimeError("Must call precompute_cluster_counts() first")
            
        cluster_weights = self.compute_cluster_weights(node_id)
        neighbors_by_cluster = self._adjacency_by_cluster.get(node_id, {})
        
        if not cluster_weights:
            return [], {}  # Isolated node
        
        # Allocate samples proportionally to cluster weights
        samples_by_cluster = {}
        remaining_samples = self.s_in
        
        if self.proportional:
            # s_C ∝ α_{v,C} (proportional allocation)
            for cluster_id, weight in cluster_weights.items():
                # Allocate samples proportionally, but ensure at least 1 if weight > 0
                s_c = max(1, int(np.ceil(weight * self.s_in))) if weight > 0 else 0
                s_c = min(s_c, remaining_samples)  # Don't exceed budget
                
                if s_c > 0 and cluster_id in neighbors_by_cluster:
                    cluster_neighbors = neighbors_by_cluster[cluster_id]
                    sampled = np.random.choice(cluster_neighbors, 
                                             size=min(s_c, len(cluster_neighbors)), 
                                             replace=False)
                    samples_by_cluster[cluster_id] = sampled.tolist()
                    remaining_samples -= len(sampled)
                    
                if remaining_samples <= 0:
                    break
        else:
            # Equal allocation across clusters
            clusters_with_neighbors = list(neighbors_by_cluster.keys())
            if clusters_with_neighbors:
                s_per_cluster = max(1, self.s_in // len(clusters_with_neighbors))
                
                for cluster_id in clusters_with_neighbors:
                    cluster_neighbors = neighbors_by_cluster[cluster_id]
                    s_c = min(s_per_cluster, len(cluster_neighbors), remaining_samples)
                    
                    if s_c > 0:
                        sampled = np.random.choice(cluster_neighbors, size=s_c, replace=False)
                        samples_by_cluster[cluster_id] = sampled.tolist()
                        remaining_samples -= s_c
                        
                    if remaining_samples <= 0:
                        break
        
        # Flatten to get all sampled neighbors
        sampled_neighbors = []
        for cluster_samples in samples_by_cluster.values():
            sampled_neighbors.extend(cluster_samples)
            
        return sampled_neighbors, samples_by_cluster
        
    def sample_batch(self, 
                     node_ids: List[int], 
                     edge_index: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Sample neighborhoods for a batch of nodes.
        
        Args:
            node_ids: List of nodes to sample neighborhoods for
            edge_index: Graph edge index tensor
            
        Returns:
            batch_edge_index: Edge index for the sampled subgraph
            sampling_info: Dict with sampling statistics and cluster info
        """
        if not self._cache_valid:
            self.precompute_cluster_counts(edge_index)
        
        all_sampled_nodes = set(node_ids)
        batch_sampling_info = {
            'node_cluster_samples': {},  # node_id -> {cluster_id: [sampled_nodes]}
            'cluster_weights': {},       # node_id -> {cluster_id: weight}
            'sampling_stats': defaultdict(int)
        }
        
        # Sample neighborhoods for each node in batch
        for node_id in node_ids:
            sampled_neighbors, samples_by_cluster = self.stratified_sample(node_id)
            cluster_weights = self.compute_cluster_weights(node_id)
            
            # Store sampling info
            batch_sampling_info['node_cluster_samples'][node_id] = samples_by_cluster
            batch_sampling_info['cluster_weights'][node_id] = cluster_weights
            
            # Add sampled nodes to batch
            all_sampled_nodes.update(sampled_neighbors)
            
            # Update stats
            batch_sampling_info['sampling_stats']['total_samples'] += len(sampled_neighbors)
            batch_sampling_info['sampling_stats']['nodes_sampled'] += 1
            
        # Create subgraph edge index
        node_mapping = {node: i for i, node in enumerate(sorted(all_sampled_nodes))}
        batch_sampling_info['node_mapping'] = node_mapping
        
        # Filter edges to include only those in sampled subgraph
        edge_index_np = edge_index.cpu().numpy()
        batch_edges = []
        
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            if src in node_mapping and dst in node_mapping:
                batch_edges.append([node_mapping[src], node_mapping[dst]])
        
        if batch_edges:
            batch_edge_index = torch.tensor(batch_edges, dtype=torch.long, device=self.device).t()
        else:
            batch_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            
        batch_sampling_info['sampling_stats']['batch_edges'] = batch_edge_index.shape[1]
        batch_sampling_info['sampling_stats']['batch_nodes'] = len(all_sampled_nodes)
        
        return batch_edge_index, batch_sampling_info
        
    def compute_cluster_representatives(self, 
                                     node_embeddings: torch.Tensor,
                                     batch_info: Dict) -> torch.Tensor:
        """
        Compute cluster representatives g_C = (1/|C|) ∑_{u∈C} h_u.
        
        Args:
            node_embeddings: Node embeddings tensor [num_nodes, embed_dim]
            batch_info: Batch information from sample_batch()
            
        Returns:
            cluster_reps: Cluster representative embeddings [num_clusters, embed_dim]
        """
        embed_dim = node_embeddings.shape[1]
        cluster_reps = torch.zeros(self.num_clusters, embed_dim, device=self.device)
        cluster_counts = torch.zeros(self.num_clusters, device=self.device)
        
        node_mapping = batch_info['node_mapping']
        
        # Accumulate embeddings by cluster
        for original_node, batch_idx in node_mapping.items():
            if original_node in self.node_to_cluster:
                cluster_id = self.node_to_cluster[original_node]
                cluster_reps[cluster_id] += node_embeddings[batch_idx]
                cluster_counts[cluster_id] += 1
        
        # Normalize by cluster size
        for cluster_id in range(self.num_clusters):
            if cluster_counts[cluster_id] > 0:
                cluster_reps[cluster_id] /= cluster_counts[cluster_id]
        
        return cluster_reps
        
    def get_sampling_stats(self) -> Dict:
        """Get sampling statistics for monitoring."""
        return {
            'num_clusters': self.num_clusters,
            's_in': self.s_in,
            's_out': self.s_out,
            'cache_valid': self._cache_valid,
            'stratified': self.stratified,
            'proportional': self.proportional
        }

def test_cluster_sampler():
    """Test ClusterSampler on a simple graph."""
    print("Testing ClusterSampler...")
    
    # Create test graph: path with 12 nodes
    edge_list = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), 
                 (6,7), (7,8), (8,9), (9,10), (10,11)]
    edge_index = torch.tensor(edge_list + [(b,a) for a,b in edge_list], dtype=torch.long).t()
    
    # Create test clusters
    clusters = [[0,1,2], [3,4,5,6], [7,8,9], [10,11]]
    
    # Initialize sampler
    sampler = ClusterSampler(clusters, s_in=4, s_out=1)
    sampler.precompute_cluster_counts(edge_index)
    
    # Test sampling for a node
    node_id = 5  # Should have neighbors in clusters 1 and 2
    sampled_neighbors, samples_by_cluster = sampler.stratified_sample(node_id)
    
    print(f"Node {node_id} neighbors sampled: {sampled_neighbors}")
    print(f"Samples by cluster: {samples_by_cluster}")
    
    # Test batch sampling
    batch_nodes = [2, 5, 8]
    batch_edge_index, batch_info = sampler.sample_batch(batch_nodes, edge_index)
    
    print(f"Batch sampling for nodes {batch_nodes}:")
    print(f"  Batch edges shape: {batch_edge_index.shape}")
    print(f"  Sampling stats: {batch_info['sampling_stats']}")
    
    print("ClusterSampler test completed!")

if __name__ == "__main__":
    test_cluster_sampler()
