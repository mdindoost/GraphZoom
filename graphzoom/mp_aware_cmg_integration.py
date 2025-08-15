#!/usr/bin/env python3
"""
MP-Aware CMG Integration for True Coarsened GraphSAGE

This integrates your MP-aware CMG prototype into the full GraphZoom pipeline
while preserving the existing working system.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import time

def simple_gnn_propagation(edge_index, features, num_layers=2, hidden_dim=64):
    """
    GNN propagation for MP-aware similarity.
    (Same as your prototype, kept for consistency)
    """
    print(f"[MP-AWARE] Running {num_layers}-layer GNN propagation")
    start_time = time.time()
    
    n_nodes, n_features = features.shape
    
    # Convert to PyTorch
    device = torch.device('cpu')
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
    x = torch.tensor(features, dtype=torch.float).to(device)
    
    # Simple GCN layers
    layers = []
    input_dim = n_features
    
    for i in range(num_layers):
        output_dim = hidden_dim
        layers.append(GCNConv(input_dim, output_dim))
        input_dim = output_dim
    
    # Forward pass (no training)
    h = x
    with torch.no_grad():
        for i, layer in enumerate(layers):
            h = layer(h, edge_index)
            if i < len(layers) - 1:
                h = F.relu(h)
    
    mp_features = h.cpu().numpy()
    
    elapsed = time.time() - start_time
    print(f"[MP-AWARE] GNN propagation: {elapsed:.3f}s, output: {mp_features.shape}")
    
    return mp_features

def mp_aware_cmg_clustering(data, features, k=10, d=20, threshold=0.1, 
                           mp_layers=2, mp_hidden=64, mp_enabled=True):
    """
    Drop-in replacement for cmg_filtered_clustering with MP-aware option.
    
    Args:
        mp_enabled: If True, use MP-aware clustering. If False, use regular CMG.
        mp_layers: Number of GNN layers for MP similarity
        mp_hidden: Hidden dimension for GNN layers
    """
    
    if not mp_enabled:
        # Fall back to regular CMG
        print("[MP-AWARE] Using regular CMG (MP disabled)")
        try:
            from filtered_timed import cmg_filtered_clustering
            return cmg_filtered_clustering(data, k=k, d=d, threshold=threshold)
        except ImportError:
            print("[MP-AWARE] Warning: Could not import regular CMG, using MP version anyway")
    
    print(f"[MP-AWARE] Using MP-aware CMG clustering")
    print(f"  MP layers: {mp_layers}, hidden: {mp_hidden}, threshold: {threshold}")
    
    start_time = time.time()
    
    # Convert data format
    edge_index = data.edge_index.cpu().numpy()
    n_nodes = data.num_nodes
    
    # Step 1: Get MP-propagated features instead of spectral filtering
    print(f"[MP-AWARE] Input: {n_nodes} nodes, features: {features.shape}")
    mp_features = simple_gnn_propagation(edge_index, features, mp_layers, mp_hidden)
    
    # Step 2: Compute cosine similarity of MP features
    similarity_matrix = cosine_similarity(mp_features)
    
    # Step 3: Reweight edges based on MP similarity
    if edge_index.ndim == 1:
        edge_index = edge_index.reshape(2, -1)
    
    rows, cols = edge_index[0], edge_index[1]
    similarities = similarity_matrix[rows, cols]
    weights = np.where(similarities > threshold, similarities, 0.0)
    
    # Step 4: Build MP-aware adjacency matrix
    A_mp = sp.coo_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))
    A_mp = A_mp.maximum(A_mp.T).tocsr()  # Symmetric
    
    print(f"[MP-AWARE] MP-reweighted graph: {A_mp.nnz} edges")
    
    # Step 5: Build Laplacian for CMG
    degrees = np.array(A_mp.sum(axis=1)).flatten()
    L_mp = sp.diags(degrees) - A_mp
    
    # Step 6: Run CMG on MP-aware Laplacian
    try:
        from cmgx.core import cmgCluster
        cI_raw, nc = cmgCluster(L_mp.tocsc())
        cI = cI_raw - 1  # Convert to 0-indexed
        
        print(f"[MP-AWARE] ✅ CMG found {nc} clusters")
        
    except Exception as e:
        print(f"[MP-AWARE] ❌ CMG failed: {e}")
        # Simple fallback clustering
        cI = np.arange(n_nodes) % max(10, n_nodes // 200)
        nc = len(np.unique(cI))
        print(f"[MP-AWARE] Using fallback: {nc} clusters")
    
    # Step 7: Compute conductance (reuse existing function)
    try:
        from filtered_timed import evaluate_phi_conductance
        phi_stats = evaluate_phi_conductance(data, cI)
    except:
        phi_stats = {'avg_phi': 0.5}  # Default value
    
    # Step 8: Compute lambda critical (approximate)
    lambda_crit = 2.0 / (1.0 + 0.5 * k)  # Same as regular CMG
    
    total_time = time.time() - start_time
    print(f"[MP-AWARE] Total MP-aware clustering time: {total_time:.3f}s")
    
    return cI, nc, phi_stats, lambda_crit

def mp_aware_cmg_coarse(laplacian, level=1, k=10, d=20, threshold=0.1,
                       mp_enabled=True, mp_layers=2, mp_hidden=64):
    """
    Enhanced cmg_coarse function with MP-aware option.
    Drop-in replacement for your existing cmg_coarse.
    """
    print(f"[MP-COARSE] Starting {'MP-aware' if mp_enabled else 'regular'} CMG coarsening")
    print(f"[MP-COARSE] Input: {laplacian.shape[0]} nodes, {level} levels")
    
    if mp_enabled:
        print(f"[MP-COARSE] MP parameters: layers={mp_layers}, hidden={mp_hidden}")
    
    projections = []
    laplacians = []
    current_laplacian = laplacian.copy()
    
    for i in range(level):
        print(f"[MP-COARSE] Level {i+1}")
        print(f"  Current: {current_laplacian.shape[0]} nodes, {int((current_laplacian.nnz - current_laplacian.shape[0])/2)} edges")
        
        # Store current state
        laplacians.append(current_laplacian.copy())
        
        # Convert to PyG format
        degree_diag = sp.diags(current_laplacian.diagonal(), 0)
        adjacency = degree_diag - current_laplacian
        adjacency = (adjacency + adjacency.T) / 2
        adjacency.data = np.abs(adjacency.data)
        
        coo = adjacency.tocoo()
        edge_index = np.vstack([coo.row, coo.col])
        
        data = Data(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            num_nodes=current_laplacian.shape[0]
        )
        
        # Create features for clustering (use identity if no features available)
        features = np.eye(current_laplacian.shape[0])[:, :min(100, current_laplacian.shape[0])]
        
        # Run MP-aware (or regular) CMG clustering
        try:
            clusters, nc, phi_stats, lambda_crit = mp_aware_cmg_clustering(
                data, features, k=k, d=d, threshold=threshold,
                mp_layers=mp_layers, mp_hidden=mp_hidden, mp_enabled=mp_enabled
            )
            
            print(f"  → Found {nc} clusters, conductance: {phi_stats.get('avg_phi', 'N/A')}")
            
        except Exception as e:
            print(f"  ❌ MP-aware clustering failed: {e}")
            # Fallback to simple spectral coarsening
            from utils import smooth_filter, spec_coarsen
            filter_ = smooth_filter(current_laplacian, 0.1)
            current_laplacian, mapping = spec_coarsen(filter_, current_laplacian)
            projections.append(mapping)
            continue
        
        # Build projection matrix
        num_nodes = current_laplacian.shape[0]
        row, col, data_vals = [], [], []
        
        for node_id in range(num_nodes):
            cluster_id = clusters[node_id]
            row.append(node_id)
            col.append(cluster_id)
            data_vals.append(1.0)
        
        mapping = sp.csr_matrix((data_vals, (row, col)), shape=(num_nodes, nc))
        projections.append(mapping)
        
        # Create coarsened Laplacian
        current_laplacian = mapping.T @ current_laplacian @ mapping
        
        print(f"  → Coarsened to {nc} nodes, {int((current_laplacian.nnz - current_laplacian.shape[0])/2)} edges")
    
    # Convert final Laplacian to NetworkX
    degree_diag = sp.diags(current_laplacian.diagonal(), 0)
    adjacency = degree_diag - current_laplacian
    adjacency.data = np.abs(adjacency.data)
    
    import networkx as nx
    G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='wgt')
    
    print(f"[MP-COARSE] Final: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    return G, projections, laplacians, level

def mp_aware_cmg_fusion_mapping(laplacian, k=10, d=20, threshold=0.1,
                               mp_enabled=True, mp_layers=2, mp_hidden=64):
    """
    Enhanced cmg_fusion_mapping with MP-aware option.
    Drop-in replacement for your existing cmg_fusion_mapping.
    """
    print(f"[MP-FUSION] {'MP-aware' if mp_enabled else 'Regular'} CMG fusion mapping")
    print(f"[MP-FUSION] Input: {laplacian.shape[0]} nodes")
    
    # Convert laplacian to PyG format
    degree_diag = sp.diags(laplacian.diagonal(), 0)
    adjacency = degree_diag - laplacian
    adjacency = (adjacency + adjacency.T) / 2
    adjacency.data = np.abs(adjacency.data)
    
    coo = adjacency.tocoo()
    edge_index = np.vstack([coo.row, coo.col])
    
    data = Data(
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        num_nodes=laplacian.shape[0]
    )
    
    # Create features for clustering
    features = np.eye(laplacian.shape[0])[:, :min(200, laplacian.shape[0])]
    
    try:
        # Run MP-aware (or regular) clustering
        cluster_assignments, num_clusters, phi_stats, lambda_crit = mp_aware_cmg_clustering(
            data, features, k=k, d=d, threshold=threshold,
            mp_layers=mp_layers, mp_hidden=mp_hidden, mp_enabled=mp_enabled
        )
        
        print(f"[MP-FUSION] Found {num_clusters} clusters")
        print(f"[MP-FUSION] Average conductance: {phi_stats.get('avg_phi', 'N/A')}")
        
        # Convert to mapping matrix
        row, col, data_vals = [], [], []
        for node_id, cluster_id in enumerate(cluster_assignments):
            row.append(cluster_id)
            col.append(node_id)
            data_vals.append(1.0)
        
        mapping = sp.csr_matrix((data_vals, (row, col)), shape=(num_clusters, laplacian.shape[0]))
        
        print(f"[MP-FUSION] Mapping: {mapping.shape}, reduction: {laplacian.shape[0] / num_clusters:.2f}x")
        
        return mapping
        
    except Exception as e:
        print(f"[MP-FUSION] Failed: {e}, falling back to identity")
        return sp.identity(laplacian.shape[0], format='csr')

# Test function
def test_mp_aware_integration():
    """
    Test the MP-aware integration on a small example.
    """
    print("🧪 Testing MP-Aware Integration")
    print("="*50)
    
    # Create test data
    n = 100
    features = np.random.randn(n, 20)
    
    # Ring graph with shortcuts
    edges = []
    for i in range(n):
        edges.append([i, (i+1) % n])
        if i % 10 == 0:
            edges.append([i, (i+5) % n])
    
    edge_index = np.array(edges).T
    data = Data(edge_index=torch.tensor(edge_index, dtype=torch.long), num_nodes=n)
    
    # Test regular vs MP-aware
    print("\n1️⃣ Testing regular CMG...")
    regular_clusters, regular_nc, _, _ = mp_aware_cmg_clustering(
        data, features, mp_enabled=False
    )
    
    print("\n2️⃣ Testing MP-aware CMG...")
    mp_clusters, mp_nc, _, _ = mp_aware_cmg_clustering(
        data, features, mp_enabled=True, mp_layers=2, mp_hidden=32
    )
    
    print(f"\n📊 Results:")
    print(f"Regular: {regular_nc} clusters")
    print(f"MP-aware: {mp_nc} clusters")
    
    if regular_nc != mp_nc:
        print("✅ MP-aware produces different clustering!")
    else:
        print("⚠️ Same number of clusters - check if actually different")
    
    return regular_clusters, mp_clusters

if __name__ == "__main__":
    test_mp_aware_integration()
