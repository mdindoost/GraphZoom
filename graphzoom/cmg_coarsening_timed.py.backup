import time
import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix, diags
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import torch

# Import your CMG functions (adjust paths as needed)
from filtered_timed import cmg_filtered_clustering, save_timing_data

def scipy_to_pyg_data(laplacian_matrix):
    """Convert scipy sparse Laplacian to PyTorch Geometric Data object"""
    # Convert Laplacian to adjacency matrix
    degree_diag = diags(laplacian_matrix.diagonal(), 0)
    adjacency = degree_diag - laplacian_matrix
    
    # Make sure adjacency is symmetric and non-negative
    adjacency = (adjacency + adjacency.T) / 2
    adjacency.data = np.abs(adjacency.data)
    
    # Convert to PyG format
    edge_index, edge_weight = from_scipy_sparse_matrix(adjacency)
    
    # Create PyG Data object
    num_nodes = laplacian_matrix.shape[0]
    data = Data(edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes)
    
    return data

def cmg_coarse(laplacian, level=1, k=10, d=20, threshold=0.1):
    """
    CMG coarsening function that matches GraphZoom's sim_coarse interface
    
    Args:
        laplacian: scipy sparse Laplacian matrix
        level: number of coarsening levels (for compatibility)
        k: CMG filter order
        d: CMG embedding dimension  
        threshold: CMG cosine similarity threshold
        
    Returns:
        G: NetworkX graph of coarsened graph
        projections: list of projection matrices
        laplacians: list of Laplacian matrices at each level
        level: number of levels
    """
    print(f"[CMG] Starting CMG coarsening with k={k}, d={d}, threshold={threshold}")
    total_start_time = time.time()
    
    projections = []
    laplacians = []
    current_laplacian = laplacian.copy()
    
    for i in range(level):
        print(f"[CMG] Coarsening Level: {i+1}")
        print(f"[CMG] Current nodes: {current_laplacian.shape[0]}, edges: {int((current_laplacian.nnz - current_laplacian.shape[0])/2)}")
        
        # Store current Laplacian
        laplacians.append(current_laplacian.copy())
        
        # Convert to PyTorch Geometric format for CMG
        data = scipy_to_pyg_data(current_laplacian)
        
        # Run CMG clustering
        try:
            clusters, nc, phi_stats, lambda_crit = cmg_filtered_clustering(
                data, k=k, d=d, threshold=threshold
            )
            print(f"[CMG] Found {nc} clusters, λ_critical ≈ {lambda_crit:.4f}")
            
        except Exception as e:
            print(f"[CMG] Error in CMG clustering: {e}")
            # Fallback to simple clustering if CMG fails
            print("[CMG] Falling back to simple spectral clustering")
            from utils import smooth_filter, spec_coarsen
            filter_ = smooth_filter(current_laplacian, 0.1)
            current_laplacian, mapping = spec_coarsen(filter_, current_laplacian)
            projections.append(mapping)
            continue
        
        # Build projection matrix from CMG clusters
        num_nodes = current_laplacian.shape[0]
        row = []
        col = []
        data_vals = []
        
        for node_id in range(num_nodes):
            cluster_id = clusters[node_id]
            row.append(node_id)
            col.append(cluster_id)
            data_vals.append(1.0)
        
        # Create projection matrix: nodes -> clusters
        mapping = csr_matrix((data_vals, (row, col)), shape=(num_nodes, nc))
        projections.append(mapping)
        
        # Create coarsened Laplacian
        current_laplacian = mapping.T @ current_laplacian @ mapping
        
        print(f"[CMG] Coarsened to {nc} nodes, {int((current_laplacian.nnz - current_laplacian.shape[0])/2)} edges")
    
    # Convert final Laplacian to NetworkX graph
    degree_diag = diags(current_laplacian.diagonal(), 0)
    adjacency = degree_diag - current_laplacian
    
    # Ensure non-negative weights
    adjacency.data = np.abs(adjacency.data)
    
    # Create NetworkX graph
    G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='wgt')
    
    print(f"[CMG] Final graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    total_cmg_time = time.time() - total_start_time
    print(f"[CMG] Total coarsening time: {total_cmg_time:.3f}s")
    
    # Save detailed timing data
    import os
    timing_file = f"results/timing_results/cmg_detailed_{level}level.json"
    os.makedirs("results/timing_results", exist_ok=True)
    save_timing_data(timing_file)
    
    return G, projections, laplacians, level

def cmg_coarse_fusion(laplacian, k=10, d=20, threshold=0.1):
    """
    CMG version of fusion coarsening (for graph fusion step)
    
    Returns a mapping matrix like sim_coarse_fusion
    """
    print("[CMG] Starting CMG fusion coarsening")
    
    # Convert to PyG format
    data = scipy_to_pyg_data(laplacian)
    
    # Run CMG clustering  
    try:
        clusters, nc, _, _ = cmg_filtered_clustering(
            data, k=k, d=d, threshold=threshold
        )
        print(f"[CMG] Fusion found {nc} clusters")
        
    except Exception as e:
        print(f"[CMG] Fusion error: {e}, falling back to simple")
        from utils import sim_coarse_fusion
        return sim_coarse_fusion(laplacian)
    
    # Build mapping matrix
    num_nodes = laplacian.shape[0]
    row = []
    col = []
    data_vals = []
    
    for cluster_id in range(nc):
        cluster_nodes = np.where(clusters == cluster_id)[0]
        for node_id in cluster_nodes:
            row.append(cluster_id)
            col.append(node_id)
            data_vals.append(1.0)
    
    # Mapping: clusters -> nodes (transpose of projection)
    mapping = csr_matrix((data_vals, (row, col)), shape=(nc, num_nodes))
    
    return mapping
