#!/usr/bin/env python3
"""
Simple Prototype: Message-Passing-Aware CMG Clustering

This is a minimal test to see if MP-based similarity improves CMG clustering
compared to spectral filtering. We replace ONLY the filtering step.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import cosine_similarity
import time

def simple_gnn_propagation(edge_index, features, num_layers=2, hidden_dim=64):
    """
    Simple GNN propagation to get message-passing-based features.
    
    This is the core of the prototype - replace spectral filtering
    with actual GNN message passing.
    """
    print(f"[MP-PROTOTYPE] Running {num_layers}-layer GNN propagation")
    start_time = time.time()
    
    n_nodes, n_features = features.shape
    
    # Convert to PyTorch
    device = torch.device('cpu')  # Keep simple for prototype
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
    x = torch.tensor(features, dtype=torch.float).to(device)
    
    # Simple 2-layer GCN
    layers = []
    input_dim = n_features
    
    for i in range(num_layers):
        output_dim = hidden_dim if i < num_layers - 1 else hidden_dim
        layers.append(GCNConv(input_dim, output_dim))
        input_dim = output_dim
    
    # Forward pass (no training - just propagation)
    h = x
    with torch.no_grad():
        for i, layer in enumerate(layers):
            h = layer(h, edge_index)
            if i < len(layers) - 1:  # No activation on last layer
                h = F.relu(h)
    
    # Convert back to numpy
    mp_features = h.cpu().numpy()
    
    elapsed = time.time() - start_time
    print(f"[MP-PROTOTYPE] GNN propagation completed in {elapsed:.3f}s")
    print(f"[MP-PROTOTYPE] Output shape: {mp_features.shape}")
    
    return mp_features

def mp_aware_graph_reweighting(edge_index, features, threshold=0.1, num_layers=2):
    """
    Create MP-aware adjacency matrix instead of spectral filtering.
    
    This replaces the spectral filtering step in your current pipeline.
    """
    print(f"[MP-PROTOTYPE] Creating MP-aware graph reweighting")
    
    # Step 1: Get MP-propagated features
    mp_features = simple_gnn_propagation(edge_index, features, num_layers)
    
    # Step 2: Compute cosine similarity of MP features
    print(f"[MP-PROTOTYPE] Computing cosine similarity...")
    similarity_matrix = cosine_similarity(mp_features)
    
    # Step 3: Apply threshold (same as current pipeline)
    print(f"[MP-PROTOTYPE] Applying threshold {threshold}")
    
    # Create edge weights based on MP similarity
    if edge_index.ndim == 1:
        edge_index = edge_index.reshape(2, -1)
    
    rows, cols = edge_index[0], edge_index[1]
    similarities = similarity_matrix[rows, cols]
    weights = np.where(similarities > threshold, similarities, 0.0)
    
    print(f"[MP-PROTOTYPE] Created {np.sum(weights > 0)} weighted edges")
    
    return weights, similarity_matrix

def prototype_mp_aware_cmg(data, features, threshold=0.1, num_layers=2):
    """
    Prototype: CMG with MP-aware similarity instead of spectral filtering.
    
    This is a drop-in replacement for your current cmg_filtered_clustering
    that uses MP similarity instead of spectral similarity.
    """
    print(f"\n🧪 MP-AWARE CMG PROTOTYPE")
    print("="*50)
    print(f"Input: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    print(f"Features: {features.shape}")
    print(f"MP layers: {num_layers}, threshold: {threshold}")
    
    total_start = time.time()
    
    # Convert PyG data to edge_index array
    edge_index = data.edge_index.cpu().numpy()
    
    # Get MP-aware weights
    mp_weights, similarity_matrix = mp_aware_graph_reweighting(
        edge_index, features, threshold, num_layers
    )
    
    # Build adjacency matrix from MP weights
    n = data.num_nodes
    from scipy.sparse import coo_matrix
    
    if edge_index.ndim == 1:
        edge_index = edge_index.reshape(2, -1)
    
    rows, cols = edge_index[0], edge_index[1]
    A_mp = coo_matrix((mp_weights, (rows, cols)), shape=(n, n))
    A_mp = A_mp.maximum(A_mp.T).tocsr()  # Make symmetric
    
    print(f"[MP-PROTOTYPE] MP-reweighted adjacency: {A_mp.nnz} nonzeros")
    
    # Build Laplacian for CMG
    degrees = np.array(A_mp.sum(axis=1)).flatten()
    from scipy.sparse import diags
    L_mp = diags(degrees) - A_mp
    
    # Run CMG on MP-aware Laplacian
    print(f"[MP-PROTOTYPE] Running CMG on MP-aware Laplacian...")
    try:
        from cmgx.core import cmgCluster
        cI_raw, nc = cmgCluster(L_mp.tocsc())
        cI = cI_raw - 1  # Convert to 0-indexed
        
        print(f"[MP-PROTOTYPE] ✅ CMG found {nc} clusters")
        
    except Exception as e:
        print(f"[MP-PROTOTYPE] ❌ CMG failed: {e}")
        # Fallback to simple clustering
        cI = np.arange(n) % 10  # Simple fallback
        nc = 10
        print(f"[MP-PROTOTYPE] Using fallback: {nc} clusters")
    
    total_time = time.time() - total_start
    print(f"[MP-PROTOTYPE] Total time: {total_time:.3f}s")
    
    # Quick cluster analysis
    cluster_sizes = [np.sum(cI == i) for i in range(nc)]
    print(f"[MP-PROTOTYPE] Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.1f}")
    
    return cI, nc, similarity_matrix

def compare_spectral_vs_mp(data, features):
    """
    Direct comparison: Run both spectral and MP-aware clustering
    """
    print(f"\n🔬 COMPARISON: SPECTRAL vs MP-AWARE")
    print("="*60)
    
    # Run original spectral version
    print(f"\n1️⃣ RUNNING ORIGINAL SPECTRAL CLUSTERING...")
    try:
        from filtered_timed import cmg_filtered_clustering
        spectral_clusters, spectral_nc, spectral_stats, _ = cmg_filtered_clustering(
            data, k=10, d=20, threshold=0.1
        )
        print(f"✅ Spectral: {spectral_nc} clusters")
    except Exception as e:
        print(f"❌ Spectral failed: {e}")
        spectral_clusters, spectral_nc = None, 0
    
    # Run MP-aware version  
    print(f"\n2️⃣ RUNNING MP-AWARE CLUSTERING...")
    mp_clusters, mp_nc, mp_similarity = prototype_mp_aware_cmg(
        data, features, threshold=0.1, num_layers=2
    )
    print(f"✅ MP-aware: {mp_nc} clusters")
    
    # Compare results
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"Spectral clusters: {spectral_nc}")
    print(f"MP-aware clusters: {mp_nc}")
    
    if spectral_clusters is not None and mp_clusters is not None:
        # Cluster size distribution comparison
        spectral_sizes = [np.sum(spectral_clusters == i) for i in range(spectral_nc)]
        mp_sizes = [np.sum(mp_clusters == i) for i in range(mp_nc)]
        
        print(f"\nCluster size stats:")
        print(f"Spectral - mean: {np.mean(spectral_sizes):.1f}, std: {np.std(spectral_sizes):.1f}")
        print(f"MP-aware - mean: {np.mean(mp_sizes):.1f}, std: {np.std(mp_sizes):.1f}")
    
    return {
        'spectral_clusters': spectral_clusters,
        'spectral_nc': spectral_nc,
        'mp_clusters': mp_clusters,
        'mp_nc': mp_nc,
        'mp_similarity': mp_similarity
    }

def test_on_cora():
    """
    Test the prototype on Cora dataset
    """
    print(f"🧪 TESTING MP-AWARE CMG PROTOTYPE ON CORA")
    print("="*60)
    
    # Load Cora data (assuming you have this)
    try:
        # You'll need to adapt this to your data loading
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0]
        
        print(f"Loaded Cora: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
        
        # Use node features
        features = data.x.cpu().numpy()
        print(f"Features shape: {features.shape}")
        
    except Exception as e:
        print(f"❌ Could not load Cora: {e}")
        print("🔧 Create synthetic data for testing...")
        
        # Create synthetic test data
        n_nodes = 100
        n_features = 20
        
        # Random features
        features = np.random.randn(n_nodes, n_features)
        
        # Create simple graph structure
        from torch_geometric.data import Data
        import torch
        
        # Ring graph for testing
        edges = []
        for i in range(n_nodes):
            edges.append([i, (i+1) % n_nodes])
            if i % 10 == 0:  # Add some cross-connections
                edges.append([i, (i+5) % n_nodes])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data = Data(x=torch.tensor(features), edge_index=edge_index, num_nodes=n_nodes)
        
        print(f"Created synthetic data: {n_nodes} nodes, {len(edges)} edges")
    
    # Run comparison
    results = compare_spectral_vs_mp(data, features)
    
    print(f"\n🎯 PROTOTYPE TEST COMPLETE!")
    print(f"Next step: If MP-aware gives better/different clusters,")
    print(f"integrate into your full True Coarsened GraphSAGE pipeline")
    
    return results

if __name__ == "__main__":
    results = test_on_cora()
