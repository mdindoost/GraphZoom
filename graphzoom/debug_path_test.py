#!/usr/bin/env python3
"""
Debug test on simple path graph to investigate CMG issues
Path graph: 0-1-2-3-4-5-6-7-8-9 (10 nodes)
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix, diags
import time
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Import your existing modules
from filtered_timed import cmg_filtered_clustering
from cmg_coarsening import scipy_to_pyg_data

def create_path_graph(n_nodes=10):
    """Create simple path graph: 0-1-2-3-...-9"""
    print(f"Creating path graph with {n_nodes} nodes")
    
    G = nx.path_graph(n_nodes)
    
    print(f"Path graph created:")
    print(f"  Nodes: {list(G.nodes())}")
    print(f"  Edges: {list(G.edges())}")
    print(f"  Total: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G

def create_path_labels(n_nodes=10):
    """Create simple labels for path graph"""
    labels = np.zeros(n_nodes, dtype=int)
    
    # Simple pattern: first third=0, middle third=1, last third=2
    third = n_nodes // 3
    
    for i in range(n_nodes):
        if i < third:
            labels[i] = 0
        elif i < 2 * third:
            labels[i] = 1
        else:
            labels[i] = 2
    
    print(f"Labels created: {labels}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return labels

def debug_simple_coarsening(adjacency, target_nodes=5):
    """Simple spectral coarsening with detailed debugging"""
    print(f"\n{'='*50}")
    print("DEBUG: SIMPLE COARSENING")
    print(f"{'='*50}")
    
    print(f"Input adjacency shape: {adjacency.shape}")
    print(f"Input adjacency nnz: {adjacency.nnz}")
    
    # Build Laplacian
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    print(f"Degrees: {degrees}")
    
    L = diags(degrees) - adjacency
    print(f"Laplacian shape: {L.shape}, nnz: {L.nnz}")
    
    # Simple spectral clustering using Fiedler vector
    print("Computing eigendecomposition...")
    eigenvals, eigenvecs = sp.linalg.eigsh(L, k=min(3, L.shape[0]-1), which='SM')
    print(f"Eigenvalues: {eigenvals}")
    print(f"Eigenvector shape: {eigenvecs.shape}")
    
    fiedler = eigenvecs[:, 1]  # Second smallest eigenvector
    print(f"Fiedler vector: {fiedler}")
    
    # Threshold-based clustering
    median_val = np.median(fiedler)
    clusters = (fiedler > median_val).astype(int)
    print(f"Initial clusters (threshold={median_val:.3f}): {clusters}")
    
    # Refine to get more clusters if needed
    unique_clusters = len(np.unique(clusters))
    print(f"Initial cluster count: {unique_clusters}")
    
    if unique_clusters < target_nodes and unique_clusters > 1:
        print("Refining clusters...")
        for cluster_id in range(unique_clusters):
            mask = clusters == cluster_id
            cluster_size = np.sum(mask)
            print(f"  Cluster {cluster_id}: {cluster_size} nodes {np.where(mask)[0]}")
            
            if cluster_size > 2:
                sub_indices = np.where(mask)[0]
                mid_point = len(sub_indices) // 2
                new_cluster_id = len(np.unique(clusters))
                clusters[sub_indices[mid_point:]] = new_cluster_id
                print(f"    Split into cluster {new_cluster_id}: nodes {sub_indices[mid_point:]}")
    
    nc = len(np.unique(clusters))
    print(f"Final clusters: {clusters}")
    print(f"Final cluster count: {nc}")
    
    # Build projection matrix Q^+ (nodes → clusters)
    n_nodes = adjacency.shape[0]
    Q_plus_data, Q_plus_row, Q_plus_col = [], [], []
    
    for node_id in range(n_nodes):
        cluster_id = clusters[node_id]
        Q_plus_data.append(1.0)
        Q_plus_row.append(node_id)
        Q_plus_col.append(cluster_id)
    
    Q_plus = csr_matrix((Q_plus_data, (Q_plus_row, Q_plus_col)), shape=(n_nodes, nc))
    print(f"Q^+ matrix shape: {Q_plus.shape}")
    print(f"Q^+ matrix:\n{Q_plus.toarray()}")
    
    # Build coarsening matrix Q (clusters → nodes)
    cluster_sizes = np.array(Q_plus.sum(axis=0)).flatten()
    print(f"Cluster sizes: {cluster_sizes}")
    
    Q_data, Q_row, Q_col = [], [], []
    
    for cluster_id in range(nc):
        cluster_size = cluster_sizes[cluster_id]
        if cluster_size > 0:
            nodes_in_cluster = Q_plus[:, cluster_id].nonzero()[0]
            print(f"Cluster {cluster_id}: nodes {nodes_in_cluster} (size {cluster_size})")
            for node_id in nodes_in_cluster:
                Q_data.append(1.0 / cluster_size)
                Q_row.append(cluster_id)
                Q_col.append(node_id)
    
    Q = csr_matrix((Q_data, (Q_row, Q_col)), shape=(nc, n_nodes))
    print(f"Q matrix shape: {Q.shape}")
    print(f"Q matrix:\n{Q.toarray()}")
    
    return Q, Q_plus

def debug_cmg_coarsening(adjacency):
    """CMG coarsening with detailed debugging"""
    print(f"\n{'='*50}")
    print("DEBUG: CMG COARSENING")
    print(f"{'='*50}")
    
    print(f"Input adjacency shape: {adjacency.shape}")
    print(f"Input adjacency nnz: {adjacency.nnz}")
    
    # Convert to PyG format for CMG
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    print(f"Degrees: {degrees}")
    
    L = diags(degrees) - adjacency
    print(f"Laplacian shape: {L.shape}, nnz: {L.nnz}")
    
    try:
        data = scipy_to_pyg_data(L)
        print(f"PyG data created:")
        print(f"  num_nodes: {data.num_nodes}")
        print(f"  edge_index shape: {data.edge_index.shape}")
        print(f"  edge_index: {data.edge_index}")
        
        # Run CMG clustering with debugging
        print("\nCalling CMG with parameters: k=10, d=20, threshold=0.1")
        clusters, nc, phi_stats, lambda_crit = cmg_filtered_clustering(
            data, k=10, d=20, threshold=0.1
        )
        
        print(f"\nCMG Results:")
        print(f"  clusters: {clusters}")
        print(f"  nc: {nc}")
        print(f"  lambda_crit: {lambda_crit}")
        print(f"  phi_stats: {phi_stats}")
        
        if nc <= 1:
            print("WARNING: CMG produced only 1 cluster! This will cause issues.")
            return None, None
        
        # Build Q and Q^+ from CMG results
        n_nodes = adjacency.shape[0]
        Q_plus_data, Q_plus_row, Q_plus_col = [], [], []
        
        for node_id in range(n_nodes):
            cluster_id = clusters[node_id]
            Q_plus_data.append(1.0)
            Q_plus_row.append(node_id)
            Q_plus_col.append(cluster_id)
        
        Q_plus = csr_matrix((Q_plus_data, (Q_plus_row, Q_plus_col)), shape=(n_nodes, nc))
        print(f"Q^+ matrix shape: {Q_plus.shape}")
        print(f"Q^+ matrix:\n{Q_plus.toarray()}")
        
        # Build Q matrix
        cluster_sizes = np.array(Q_plus.sum(axis=0)).flatten()
        print(f"CMG Cluster sizes: {cluster_sizes}")
        
        Q_data, Q_row, Q_col = [], [], []
        
        for cluster_id in range(nc):
            cluster_size = cluster_sizes[cluster_id]
            if cluster_size > 0:
                nodes_in_cluster = Q_plus[:, cluster_id].nonzero()[0]
                print(f"CMG Cluster {cluster_id}: nodes {nodes_in_cluster} (size {cluster_size})")
                for node_id in nodes_in_cluster:
                    Q_data.append(1.0 / cluster_size)
                    Q_row.append(cluster_id)
                    Q_col.append(node_id)
        
        Q = csr_matrix((Q_data, (Q_row, Q_col)), shape=(nc, n_nodes))
        print(f"Q matrix shape: {Q.shape}")
        print(f"Q matrix:\n{Q.toarray()}")
        
        return Q, Q_plus
        
    except Exception as e:
        print(f"ERROR in CMG: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def compute_propagation_matrix(adjacency, gnn_type='gcn'):
    """Compute GNN propagation matrix with debugging"""
    print(f"\nComputing {gnn_type.upper()} propagation matrix...")
    print(f"Input adjacency shape: {adjacency.shape}, nnz: {adjacency.nnz}")
    
    if gnn_type == 'gcn':
        # GCN: S = D^(-1/2) (A + I) D^(-1/2)
        A_self = adjacency + sp.identity(adjacency.shape[0])
        degrees = np.array(A_self.sum(axis=1)).flatten()
        print(f"Degrees with self-loops: {degrees}")
        
        degrees[degrees == 0] = 1
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
        S = D_inv_sqrt @ A_self @ D_inv_sqrt
        
    elif gnn_type == 'graphsage':
        # GraphSAGE: S = D^(-1) A
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        print(f"Degrees: {degrees}")
        
        degrees[degrees == 0] = 1
        D_inv = sp.diags(1.0 / degrees)
        S = D_inv @ adjacency
        
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")
    
    print(f"Propagation matrix shape: {S.shape}, nnz: {S.nnz}")
    print(f"Propagation matrix:\n{S.toarray()}")
    
    return S

def debug_propagation_methods(Q, Q_plus, adjacency, gnn_type='gcn'):
    """Debug both naive and MP-aware propagation"""
    print(f"\n{'='*50}")
    print(f"DEBUG: PROPAGATION METHODS - {gnn_type.upper()}")
    print(f"{'='*50}")
    
    if Q is None or Q_plus is None:
        print("ERROR: Q or Q^+ is None, skipping propagation test")
        return None, None
    
    # Method 1: Naive
    print("\n--- NAIVE PROPAGATION ---")
    A_coarsened = Q_plus.T @ adjacency @ Q_plus
    print(f"Coarsened adjacency shape: {A_coarsened.shape}, nnz: {A_coarsened.nnz}")
    print(f"Coarsened adjacency:\n{A_coarsened.toarray()}")
    
    S_naive = compute_propagation_matrix(A_coarsened, gnn_type)
    
    # Method 2: MP-aware
    print("\n--- MP-AWARE PROPAGATION ---")
    S_original = compute_propagation_matrix(adjacency, gnn_type)
    S_c_MP = Q @ S_original @ Q_plus
    
    print(f"MP-aware propagation shape: {S_c_MP.shape}, nnz: {S_c_MP.nnz}")
    print(f"MP-aware propagation:\n{S_c_MP.toarray()}")
    
    return S_naive, S_c_MP

def debug_embedding_creation(S_matrix, method='spectral', dim=4):
    """Debug embedding creation"""
    print(f"\nCreating {method} embeddings (dim={dim})...")
    print(f"Input matrix shape: {S_matrix.shape}")
    
    if method == 'spectral':
        try:
            eigenvals, eigenvecs = sp.linalg.eigsh(S_matrix, k=min(dim, S_matrix.shape[0]-1), which='LM')
            print(f"Eigenvalues: {eigenvals}")
            embeddings = eigenvecs[:, :dim]
            print(f"Embeddings shape: {embeddings.shape}")
            print(f"Embeddings:\n{embeddings}")
        except Exception as e:
            print(f"Spectral embedding failed: {e}")
            embeddings = np.random.randn(S_matrix.shape[0], dim)
            print(f"Using random embeddings: {embeddings.shape}")
    else:
        embeddings = np.random.randn(S_matrix.shape[0], dim)
        print(f"Random embeddings: {embeddings.shape}")
    
    return embeddings

def debug_full_pipeline(method_name, Q, Q_plus, adjacency, labels, gnn_type='gcn'):
    """Debug complete pipeline for one method"""
    print(f"\n{'#'*80}")
    print(f"DEBUG FULL PIPELINE: {method_name.upper()}")
    print(f"{'#'*80}")
    
    if Q is None or Q_plus is None:
        print(f"ERROR: {method_name} coarsening failed, skipping pipeline")
        return {'method': method_name, 'accuracy': 0.0, 'error': 'coarsening_failed'}
    
    # Step 1: Propagation
    if 'naive' in method_name:
        A_coarsened = Q_plus.T @ adjacency @ Q_plus
        S_coarsened = compute_propagation_matrix(A_coarsened, gnn_type)
    else:  # mp_aware
        S_original = compute_propagation_matrix(adjacency, gnn_type)
        S_coarsened = Q @ S_original @ Q_plus
    
    print(f"Coarsened propagation matrix shape: {S_coarsened.shape}")
    
    # Step 2: Embeddings
    coarse_embeddings = debug_embedding_creation(S_coarsened, dim=4)
    
    # Step 3: Refinement
    print(f"\nRefinement: {coarse_embeddings.shape} -> {Q_plus.shape}")
    refined_embeddings = Q_plus @ coarse_embeddings
    print(f"Refined embeddings shape: {refined_embeddings.shape}")
    print(f"Refined embeddings:\n{refined_embeddings}")
    
    # Step 4: Classification
    print(f"\nClassification test...")
    print(f"Labels: {labels}")
    
    try:
        if len(refined_embeddings) < 4:
            accuracy = 0.5
            print(f"Too few samples for train/test split, using dummy accuracy: {accuracy}")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                refined_embeddings, labels, test_size=0.3, random_state=42
            )
            
            print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
            
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"True labels: {y_test}")
            print(f"Predictions: {y_pred}")
            print(f"Accuracy: {accuracy:.3f}")
    except Exception as e:
        print(f"Classification failed: {e}")
        accuracy = 0.0
    
    return {
        'method': method_name,
        'accuracy': accuracy,
        'coarsened_nodes': Q.shape[0],
        'compression_ratio': adjacency.shape[0] / Q.shape[0]
    }

def main():
    """Run comprehensive debug test on path graph"""
    print("COMPREHENSIVE DEBUG TEST - PATH GRAPH")
    print("="*80)
    
    # Create path graph
    G = create_path_graph(n_nodes=10)
    labels = create_path_labels(n_nodes=10)
    A = nx.adjacency_matrix(G).tocsr()
    
    print(f"Adjacency matrix:\n{A.toarray()}")
    
    # Test both coarsening methods
    print(f"\n{'*'*80}")
    print("TESTING COARSENING METHODS")
    print(f"{'*'*80}")
    
    # Simple coarsening
    Q_simple, Q_plus_simple = debug_simple_coarsening(A, target_nodes=5)
    
    # CMG coarsening
    Q_cmg, Q_plus_cmg = debug_cmg_coarsening(A)
    
    # Test propagation methods
    gnn_type = 'gcn'
    
    methods_to_test = [
        ('simple_naive', Q_simple, Q_plus_simple),
        ('simple_mp_aware', Q_simple, Q_plus_simple),
        ('cmg_naive', Q_cmg, Q_plus_cmg),
        ('cmg_mp_aware', Q_cmg, Q_plus_cmg),
    ]
    
    results = []
    
    for method_name, Q, Q_plus in methods_to_test:
        result = debug_full_pipeline(method_name, Q, Q_plus, A, labels, gnn_type)
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}")
    
    for result in results:
        method = result['method']
        accuracy = result.get('accuracy', 0.0)
        error = result.get('error', 'none')
        
        print(f"{method:<20}: Accuracy = {accuracy:.3f}, Error = {error}")

if __name__ == "__main__":
    main()
