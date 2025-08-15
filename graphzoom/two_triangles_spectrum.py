# focused_spectral_CMG_test.py
# ---------------------------------------------------------------
# Test CMG clustering on larger structured graphs:
# 1. 100-node path graph
# 2. 15x15 grid graph (225 nodes)
# ---------------------------------------------------------------

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import sys
import os

# Add path for CMG imports (adjust as needed)
sys.path.append('.')
sys.path.append('..')

# ---------- Core utilities ----------

def laplacian_from_graph(G):
    """Return (nodes, D, L) with nodes in sorted order."""
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Adjacency (for constructing L)
    A = np.zeros((n, n), dtype=float)
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        A[i, j] = 1.0
        A[j, i] = 1.0

    # Degree vector and matrices
    d = A.sum(axis=1)
    D = np.diag(d)
    L = D - A
    return nodes, D, L

def eigh_sorted(M):
    """Eigen-decomposition with ascending eigenvalues; columns aligned."""
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]

def networkx_to_pyg_data(G):
    """Convert NetworkX graph to PyTorch Geometric Data for CMG."""
    # Get edge list
    edge_list = list(G.edges())
    if len(edge_list) == 0:
        # Handle isolated nodes
        num_nodes = G.number_of_nodes()
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        # Convert to tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create PyG Data object
    data = Data(edge_index=edge_index, num_nodes=G.number_of_nodes())
    return data

def run_cmg_clustering(G, k=10, d=20, threshold=0.1):
    """Run CMG clustering on the graph and return cluster assignments."""
    try:
        # Import CMG function
        from filtered_timed import cmg_filtered_clustering
        
        # Convert to PyG format
        data = networkx_to_pyg_data(G)
        
        # Run CMG clustering
        cluster_assignments, num_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
            data, k=k, d=d, threshold=threshold
        )
        
        return cluster_assignments, num_clusters, phi_stats, lambda_crit, None
        
    except ImportError as e:
        return None, 0, {}, 0, f"CMG not available: {e}"
    except Exception as e:
        return None, 0, {}, 0, f"CMG failed: {e}"

def print_cmg_results(G, nodes, round_decimals=6):
    """Print CMG clustering results."""
    print("\n--- CMG Clustering Results ---")
    
    # Test with standard parameters
    k, d, threshold = 10, 30, 0.1
    
    clusters, num_clusters, phi_stats, lambda_crit, error = run_cmg_clustering(
        G, k=k, d=d, threshold=threshold
    )
    
    if error:
        print(f"CMG (k={k}, d={d}, threshold={threshold}): {error}")
        return
    
    print(f"CMG (k={k}, d={d}, threshold={threshold}):")
    print(f"  Found {num_clusters} clusters")
    print(f"  λ_critical ≈ {lambda_crit:.{round_decimals}f}")
    print(f"  Average conductance: {phi_stats.get('avg_phi', 'N/A'):.{round_decimals}f}")
    
    # Group nodes by cluster
    if clusters is not None:
        cluster_groups = {}
        for node_idx, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            if node_idx < len(nodes):
                cluster_groups[cluster_id].append(nodes[node_idx])
        
        # Print cluster summary
        print("  Cluster sizes:")
        for cluster_id in sorted(cluster_groups.keys()):
            cluster_nodes = sorted(cluster_groups[cluster_id])
            print(f"    Cluster {cluster_id}: {len(cluster_nodes)} nodes")
        
        # For smaller graphs, show actual nodes
        if G.number_of_nodes() <= 30:
            print("  Cluster assignments:")
            for cluster_id in sorted(cluster_groups.keys()):
                cluster_nodes = sorted(cluster_groups[cluster_id])
                print(f"    Cluster {cluster_id}: {cluster_nodes}")
        else:
            # For larger graphs, show first few nodes of each cluster
            print("  Cluster assignments (first 10 nodes of each):")
            for cluster_id in sorted(cluster_groups.keys()):
                cluster_nodes = sorted(cluster_groups[cluster_id])
                if len(cluster_nodes) <= 10:
                    print(f"    Cluster {cluster_id}: {cluster_nodes}")
                else:
                    print(f"    Cluster {cluster_id}: {cluster_nodes[:10]}... ({len(cluster_nodes)} total)")

def print_L_summary(title, nodes, D, L, G, round_decimals=6):
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"Graph info: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # For large graphs, don't print the full matrices
    if len(nodes) <= 10:
        print(f"Nodes (ordered): {nodes}")
        
        # Print degree diagonal compactly
        d = np.diag(D)
        np.set_printoptions(precision=3, suppress=True)
        print("\nDegree vector (diag(D)):\n", np.round(d, 3))
        print("\nUnnormalized Laplacian L = D - A:\n", np.round(L, 3))
    else:
        print(f"Nodes: 0 to {len(nodes)-1} (too large to display)")
        d = np.diag(D)
        print(f"Degree statistics: min={np.min(d):.3f}, max={np.max(d):.3f}, mean={np.mean(d):.3f}")

    # Eigen-decomposition of L (only show summary for large graphs)
    vals, vecs = eigh_sorted(L)
    print("\n--- Eigen-decomposition: L ---")
    
    if len(vals) <= 10:
        print("Eigenvalues(L):", np.round(vals, round_decimals))
    else:
        print(f"Eigenvalues(L): {len(vals)} total")
        print(f"  Smallest 5: {np.round(vals[:5], round_decimals)}")
        print(f"  Largest 5:  {np.round(vals[-5:], round_decimals)}")

    # Connectivity check via zero eigenvalues
    zero_count = np.sum(np.isclose(vals, 0.0, atol=1e-10))
    print("\nConnectivity check (L):")
    print(f"  Number of ~zero eigenvalues: {int(zero_count)}  --> #connected components")
    if zero_count == 1:
        print("  Graph appears connected (exactly one zero eigenvalue).")
    else:
        print("  Graph appears disconnected (multiple zero eigenvalues).")

    # Fiedler value/vector (2nd smallest eigenvalue of L), if applicable
    if len(vals) >= 2:
        print(f"\nFiedler value (λ2 of L): {float(vals[1]):.{round_decimals}f}")
        
        if len(nodes) <= 20:
            # Show full Fiedler vector for small graphs
            print("Fiedler vector (L) components by node:")
            for node, val in zip(nodes, vecs[:, 1]):
                print(f"  node {node:>2}: {val:+.6f}")
            print("  (Global sign can flip; relative signs indicate a soft partition)")
            
            # Show Fiedler-based partitioning
            print("\nFiedler-based partition (sign-based):")
            positive_nodes = [nodes[i] for i, val in enumerate(vecs[:, 1]) if val > 0]
            negative_nodes = [nodes[i] for i, val in enumerate(vecs[:, 1]) if val <= 0]
            if positive_nodes:
                print(f"  Positive side ({len(positive_nodes)} nodes): {positive_nodes}")
            if negative_nodes:
                print(f"  Negative side ({len(negative_nodes)} nodes): {negative_nodes}")
        else:
            # For large graphs, just show statistics
            fiedler_vec = vecs[:, 1]
            positive_count = np.sum(fiedler_vec > 0)
            negative_count = np.sum(fiedler_vec <= 0)
            print(f"Fiedler-based partition (sign-based):")
            print(f"  Positive side: {positive_count} nodes")
            print(f"  Negative side: {negative_count} nodes")
            print(f"  Fiedler vector stats: min={np.min(fiedler_vec):.6f}, max={np.max(fiedler_vec):.6f}")

    # ADD CMG CLUSTERING RESULTS HERE
    print_cmg_results(G, nodes, round_decimals)

# ---------- Graph builders ----------

def path_100():
    """100-node path graph."""
    return nx.path_graph(100)

def grid_15x15():
    """15x15 grid graph with regular node numbering (not (x,y) coordinates)."""
    # Create 15x15 grid but relabel nodes to 0, 1, 2, ..., 224
    G = nx.grid_2d_graph(50, 50)
    
    # Relabel nodes from (x,y) to sequential integers
    mapping = {}
    node_id = 0
    for i in range(50):
        for j in range(50):
            mapping[(i, j)] = node_id
            node_id += 1
    
    G = nx.relabel_nodes(G, mapping)
    return G

# ---------- Main ----------

def main():
    print("Testing CMG clustering on structured graphs...")
    print("This will help understand CMG behavior on larger, well-defined structures.\n")
    
    cases = [
        ("100-node path graph", path_100()),
        ("15x15 grid graph (225 nodes)", grid_15x15()),
    ]

    for title, G in cases:
        nodes, D, L = laplacian_from_graph(G)
        print_L_summary(title, nodes, D, L, G)
        print("\n" + "="*80 + "\n")

    print("Analysis complete!")
    print("\nKey questions to consider:")
    print("1. How does CMG partition the path graph? (Should be roughly balanced)")
    print("2. How does CMG handle the 2D grid structure?") 
    print("3. How do CMG clusters compare to Fiedler partitioning?")
    print("4. What are the conductance values telling us?")

if __name__ == "__main__":
    main()