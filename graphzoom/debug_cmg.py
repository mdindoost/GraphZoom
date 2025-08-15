#!/usr/bin/env python3
"""
Debug CMG coarsening issues on larger graphs
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix, diags
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import torch

# Import your existing modules
from filtered_timed import cmg_filtered_clustering
from cmg_coarsening import scipy_to_pyg_data

def create_test_graph(n_nodes=100):
    """Create the same type of graph as the failing test"""
    print(f"Creating test graph with {n_nodes} nodes...")
    
    # Use Barabási-Albert model for scale-free properties
    G = nx.barabasi_albert_graph(n_nodes, 3, seed=42)
    
    # Add some small-world properties
    edges_to_rewire = int(0.1 * G.number_of_edges())
    edges = list(G.edges())
    
    for _ in range(edges_to_rewire):
        if edges:
            old_edge = edges.pop(np.random.randint(len(edges)))
            G.remove_edge(*old_edge)
            
            # Add new random edge
            u = np.random.randint(n_nodes)
            v = np.random.randint(n_nodes)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
    
    print(f"Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def debug_scipy_to_pyg_conversion(L):
    """Debug the Laplacian to PyG conversion"""
    print(f"\n{'='*50}")
    print("DEBUGGING SCIPY TO PyG CONVERSION")
    print(f"{'='*50}")
    
    print(f"Input Laplacian shape: {L.shape}")
    print(f"Input Laplacian nnz: {L.nnz}")
    print(f"Input Laplacian diagonal (first 10): {L.diagonal()[:10]}")
    
    try:
        data = scipy_to_pyg_data(L)
        print(f"PyG Data created successfully:")
        print(f"  num_nodes: {data.num_nodes}")
        print(f"  edge_index shape: {data.edge_index.shape}")
        print(f"  edge_attr shape: {data.edge_attr.shape if data.edge_attr is not None else 'None'}")
        print(f"  First few edges: {data.edge_index[:, :5]}")
        
        return data, True
        
    except Exception as e:
        print(f"ERROR in scipy_to_pyg_data: {e}")
        
        # Try manual conversion
        print("Attempting manual conversion...")
        degree_diag = diags(L.diagonal(), 0)
        adjacency = degree_diag - L
        adjacency = (adjacency + adjacency.T) / 2
        adjacency.data = np.abs(adjacency.data)
        
        edge_index, edge_weight = from_scipy_sparse_matrix(adjacency)
        data = Data(edge_index=edge_index, edge_attr=edge_weight, num_nodes=L.shape[0])
        
        print(f"Manual conversion successful:")
        print(f"  num_nodes: {data.num_nodes}")
        print(f"  edge_index shape: {data.edge_index.shape}")
        
        return data, False

def debug_cmg_parameters(data, test_params):
    """Test different CMG parameters to see what works"""
    print(f"\n{'='*50}")
    print("TESTING DIFFERENT CMG PARAMETERS")
    print(f"{'='*50}")
    
    results = {}
    
    for params in test_params:
        k, d, threshold = params
        print(f"\nTesting k={k}, d={d}, threshold={threshold}")
        print("-" * 40)
        
        try:
            clusters, nc, phi_stats, lambda_crit = cmg_filtered_clustering(
                data, k=k, d=d, threshold=threshold
            )
            
            print(f"✓ SUCCESS: {nc} clusters, λ_critical={lambda_crit:.4f}")
            print(f"  Cluster sizes: {np.bincount(clusters)}")
            print(f"  Conductance: {phi_stats['avg_phi']:.4f}")
            
            results[f"k{k}_d{d}_t{threshold}"] = {
                'success': True,
                'clusters': clusters,
                'nc': nc,
                'lambda_crit': lambda_crit,
                'conductance': phi_stats['avg_phi']
            }
            
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results[f"k{k}_d{d}_t{threshold}"] = {
                'success': False,
                'error': str(e)
            }
    
    return results

def compare_clustering_methods(A, target_clusters=10):
    """Compare CMG vs simple clustering on the same graph"""
    print(f"\n{'='*50}")
    print("COMPARING CLUSTERING METHODS")
    print(f"{'='*50}")
    
    # Method 1: Simple spectral clustering
    print("1. Simple Spectral Clustering:")
    degrees = np.array(A.sum(axis=1)).flatten()
    L = diags(degrees) - A
    
    try:
        eigenvals, eigenvecs = sp.linalg.eigsh(L, k=min(target_clusters+1, L.shape[0]-1), which='SM')
        
        # Use multiple eigenvectors for clustering
        features = eigenvecs[:, 1:min(4, eigenvecs.shape[1])]  # Skip first eigenvector
        
        # K-means style clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=target_clusters, random_state=42, n_init=10)
        simple_clusters = kmeans.fit_predict(features)
        simple_nc = len(np.unique(simple_clusters))
        
        print(f"  ✓ Simple clustering: {simple_nc} clusters")
        print(f"  Cluster sizes: {np.bincount(simple_clusters)}")
        
    except Exception as e:
        print(f"  ✗ Simple clustering failed: {e}")
        simple_clusters = np.arange(A.shape[0]) % target_clusters
        simple_nc = target_clusters
    
    # Method 2: CMG clustering
    print("\n2. CMG Clustering:")
    data = scipy_to_pyg_data(L)
    
    try:
        cmg_clusters, cmg_nc, phi_stats, lambda_crit = cmg_filtered_clustering(
            data, k=5, d=16, threshold=0.05  # Try different parameters
        )
        
        print(f"  ✓ CMG clustering: {cmg_nc} clusters")
        print(f"  Cluster sizes: {np.bincount(cmg_clusters)}")
        print(f"  Conductance: {phi_stats['avg_phi']:.4f}")
        
    except Exception as e:
        print(f"  ✗ CMG clustering failed: {e}")
        cmg_clusters = simple_clusters.copy()
        cmg_nc = simple_nc
    
    # Compare clustering quality
    print(f"\n3. Clustering Comparison:")
    
    # Adjusted Rand Index
    try:
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(simple_clusters, cmg_clusters)
        print(f"  Adjusted Rand Index: {ari:.3f} (1.0 = identical, 0.0 = random)")
    except:
        print("  Could not compute ARI")
    
    # Modularity comparison
    try:
        G = nx.from_scipy_sparse_matrix(A)
        
        # Convert clusters to partition format
        simple_partition = {}
        cmg_partition = {}
        
        for i, (sc, cc) in enumerate(zip(simple_clusters, cmg_clusters)):
            simple_partition[i] = sc
            cmg_partition[i] = cc
        
        simple_mod = nx.algorithms.community.modularity(G, [set([n for n, c in simple_partition.items() if c == cid]) for cid in range(simple_nc)])
        cmg_mod = nx.algorithms.community.modularity(G, [set([n for n, c in cmg_partition.items() if c == cid]) for cid in range(cmg_nc)])
        
        print(f"  Simple modularity: {simple_mod:.3f}")
        print(f"  CMG modularity: {cmg_mod:.3f}")
        
    except Exception as e:
        print(f"  Could not compute modularity: {e}")
    
    return simple_clusters, cmg_clusters

def main():
    """Debug CMG issues comprehensively"""
    
    print("CMG DEBUGGING SESSION")
    print("="*80)
    
    # Create test graph
    G = create_test_graph(n_nodes=100)
    A = nx.adjacency_matrix(G).tocsr()
    
    # Build Laplacian
    degrees = np.array(A.sum(axis=1)).flatten()
    L = diags(degrees) - A
    
    print(f"Graph properties:")
    print(f"  Nodes: {A.shape[0]}")
    print(f"  Edges: {A.nnz // 2}")
    print(f"  Density: {A.nnz / (A.shape[0] * (A.shape[0] - 1)):.4f}")
    print(f"  Average degree: {np.mean(degrees):.2f}")
    
    # Step 1: Debug PyG conversion
    data, conversion_success = debug_scipy_to_pyg_conversion(L)
    
    if not conversion_success:
        print("WARNING: PyG conversion had issues!")
    
    # Step 2: Test different CMG parameters
    test_params = [
        (5, 16, 0.05),   # Smaller k, d, lower threshold
        (10, 20, 0.1),   # Original parameters
        (15, 32, 0.2),   # Larger parameters
        (3, 8, 0.01),    # Very conservative
    ]
    
    param_results = debug_cmg_parameters(data, test_params)
    
    # Step 3: Compare clustering methods
    simple_clusters, cmg_clusters = compare_clustering_methods(A, target_clusters=10)
    
    # Step 4: Summary and recommendations
    print(f"\n{'='*80}")
    print("DEBUGGING SUMMARY & RECOMMENDATIONS")
    print(f"{'='*80}")
    
    successful_params = [k for k, v in param_results.items() if v['success']]
    
    if successful_params:
        print("✓ CMG is working, but may need parameter tuning:")
        for param_name in successful_params:
            result = param_results[param_name]
            print(f"  {param_name}: {result['nc']} clusters, conductance={result['conductance']:.4f}")
        
        print(f"\nRecommendations:")
        print(f"1. Use smaller k values (3-5) for larger graphs")
        print(f"2. Adjust d based on graph size (8-16 for 100 nodes)")  
        print(f"3. Lower threshold (0.01-0.05) for sparser graphs")
        
    else:
        print("✗ CMG is failing completely. Possible issues:")
        print("1. PyG conversion problems")
        print("2. Graph structure incompatible with CMG")
        print("3. Parameter ranges too restrictive")
        
        for param_name, result in param_results.items():
            if not result['success']:
                print(f"  {param_name}: {result['error']}")
    
    print(f"\nNext steps:")
    print(f"1. Fix parameter selection for larger graphs")
    print(f"2. Add fallback mechanisms when CMG fails")
    print(f"3. Test on different graph types (dense vs sparse)")

if __name__ == "__main__":
    main()
