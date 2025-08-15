#!/usr/bin/env python3
"""
Comprehensive test of Message-Passing Aware Coarsening in GraphZoom Pipeline
Tests: CMG + MP-aware vs Simple + MP-aware vs Naive methods
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

def create_test_graph():
    """Create the 12-node test graph"""
    edges = [
        (0, 1), (1, 2), (1, 4), (2, 3), (3, 4), (4, 9),
        (4, 5), (5, 6), (5, 7), (6, 8), (8, 9),
        (9, 10), (9, 11), (10, 11)
    ]
    
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # Ensure all nodes 0-11 are present
    for i in range(12):
        if i not in G.nodes():
            G.add_node(i)
    
    return G

def simple_coarsening(adjacency, target_nodes=6):
    """Simple spectral coarsening (baseline)"""
    print(f"[Simple] Coarsening {adjacency.shape[0]} nodes to ~{target_nodes}")
    
    # Build Laplacian
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    L = diags(degrees) - adjacency
    
    # Simple spectral clustering using Fiedler vector
    eigenvals, eigenvecs = sp.linalg.eigsh(L, k=min(3, L.shape[0]-1), which='SM')
    fiedler = eigenvecs[:, 1]  # Second smallest eigenvector
    
    # Threshold-based clustering
    clusters = (fiedler > np.median(fiedler)).astype(int)
    
    # Refine to get desired number of clusters
    if len(np.unique(clusters)) < target_nodes and len(np.unique(clusters)) > 1:
        # Add more clusters by further subdividing
        for cluster_id in range(len(np.unique(clusters))):
            mask = clusters == cluster_id
            if np.sum(mask) > 2:  # Only subdivide large clusters
                sub_indices = np.where(mask)[0]
                if len(sub_indices) > 2:
                    mid_point = len(sub_indices) // 2
                    clusters[sub_indices[mid_point:]] = len(np.unique(clusters))
    
    nc = len(np.unique(clusters))
    print(f"[Simple] Found {nc} clusters")
    
    # Build projection matrix Q^+ (nodes → clusters)
    n_nodes = adjacency.shape[0]
    Q_plus_data, Q_plus_row, Q_plus_col = [], [], []
    
    for node_id in range(n_nodes):
        cluster_id = clusters[node_id]
        Q_plus_data.append(1.0)
        Q_plus_row.append(node_id)
        Q_plus_col.append(cluster_id)
    
    Q_plus = csr_matrix((Q_plus_data, (Q_plus_row, Q_plus_col)), shape=(n_nodes, nc))
    
    # Build coarsening matrix Q (clusters → nodes)
    cluster_sizes = np.array(Q_plus.sum(axis=0)).flatten()
    Q_data, Q_row, Q_col = [], [], []
    
    for cluster_id in range(nc):
        cluster_size = cluster_sizes[cluster_id]
        if cluster_size > 0:
            nodes_in_cluster = Q_plus[:, cluster_id].nonzero()[0]
            for node_id in nodes_in_cluster:
                Q_data.append(1.0 / cluster_size)
                Q_row.append(cluster_id)
                Q_col.append(node_id)
    
    Q = csr_matrix((Q_data, (Q_row, Q_col)), shape=(nc, n_nodes))
    
    return Q, Q_plus

def compute_propagation_matrix(adjacency, gnn_type='gcn'):
    """Compute GNN propagation matrix"""
    if gnn_type == 'gcn':
        # GCN: S = D^(-1/2) (A + I) D^(-1/2)
        A_self = adjacency + sp.identity(adjacency.shape[0])
        degrees = np.array(A_self.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
        S = D_inv_sqrt @ A_self @ D_inv_sqrt
        
    elif gnn_type == 'graphsage':
        # GraphSAGE: S = D^(-1) A
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1
        D_inv = sp.diags(1.0 / degrees)
        S = D_inv @ adjacency
        
    elif gnn_type == 'deepwalk':
        # DeepWalk/Node2Vec: normalized adjacency
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1
        D_inv = sp.diags(1.0 / degrees)
        S = D_inv @ adjacency
        
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")
    
    return S

def compute_naive_propagation(Q_plus, adjacency, gnn_type='gcv'):
    """Naive approach: coarsen adjacency then compute propagation"""
    # Coarsen adjacency
    A_coarsened = Q_plus.T @ adjacency @ Q_plus
    
    # Compute propagation on coarsened graph
    S_coarsened = compute_propagation_matrix(A_coarsened, gnn_type)
    
    return S_coarsened

def compute_mp_aware_propagation(Q, Q_plus, adjacency, gnn_type='gcn'):
    """MP-aware approach: S_c^MP = Q @ S @ Q^+"""
    # Compute original propagation matrix
    S_original = compute_propagation_matrix(adjacency, gnn_type)
    
    # Apply MP-aware formula
    S_c_MP = Q @ S_original @ Q_plus
    
    return S_c_MP

def simple_embedding(S_matrix, dim=8, method='spectral'):
    """Create embeddings from propagation matrix"""
    if method == 'spectral':
        # Use eigenvectors of propagation matrix
        try:
            eigenvals, eigenvecs = sp.linalg.eigsh(S_matrix, k=min(dim, S_matrix.shape[0]-1), which='LM')
            embeddings = eigenvecs[:, :dim]
        except:
            # Fallback to random
            embeddings = np.random.randn(S_matrix.shape[0], dim)
            
    elif method == 'random_walk':
        # Simple random walk embedding
        # Power iteration approximation
        embeddings = np.random.randn(S_matrix.shape[0], dim)
        for _ in range(5):  # 5 iterations
            embeddings = S_matrix @ embeddings
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=0, keepdims=True) + 1e-8)
            
    return embeddings

def create_synthetic_node_features(n_nodes, dim=16):
    """Create synthetic node features for testing"""
    np.random.seed(42)
    
    # Create features with some structure
    features = np.random.randn(n_nodes, dim)
    
    # Add some structure - nodes in similar positions get similar features
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if abs(i - j) <= 2:  # Nearby nodes get similar features
                features[j] += 0.3 * features[i]
    
    return features

def create_synthetic_labels(G, n_classes=3):
    """Create synthetic labels based on graph structure"""
    np.random.seed(42)
    
    # Use node degrees and positions to create labels
    degrees = dict(G.degree())
    nodes = sorted(G.nodes())
    
    labels = np.zeros(len(nodes), dtype=int)
    
    for i, node in enumerate(nodes):
        if degrees[node] <= 2:
            labels[i] = 0  # Low degree
        elif degrees[node] <= 4:
            labels[i] = 1  # Medium degree  
        else:
            labels[i] = 2  # High degree
    
    return labels

def test_downstream_accuracy(embeddings, labels, test_size=0.3):
    """Test node classification accuracy"""
    if len(embeddings) < 4:  # Too few nodes for train/test split
        return 0.5  # Random accuracy
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=42
        )
        
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    except:
        return 0.5  # Return random accuracy if training fails

def run_pipeline_test(coarsening_method, propagation_method, gnn_type, G, labels):
    """Run complete pipeline test"""
    print(f"\n{'='*60}")
    print(f"Testing: {coarsening_method.upper()} + {propagation_method.upper()} + {gnn_type.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Get adjacency matrix
    A = nx.adjacency_matrix(G).tocsr()
    
    # Step 1: Coarsening
    if coarsening_method == 'cmg':
        # Convert to PyG format for CMG
        degrees = np.array(A.sum(axis=1)).flatten()
        L = diags(degrees) - A
        data = scipy_to_pyg_data(L)
        
        # Run CMG clustering
        clusters, nc, _, _ = cmg_filtered_clustering(data, k=10, d=20, threshold=0.1)
        
        # Build Q and Q^+ from CMG results
        n_nodes = A.shape[0]
        Q_plus_data, Q_plus_row, Q_plus_col = [], [], []
        
        for node_id in range(n_nodes):
            cluster_id = clusters[node_id]
            Q_plus_data.append(1.0)
            Q_plus_row.append(node_id)
            Q_plus_col.append(cluster_id)
        
        Q_plus = csr_matrix((Q_plus_data, (Q_plus_row, Q_plus_col)), shape=(n_nodes, nc))
        
        # Build Q matrix
        cluster_sizes = np.array(Q_plus.sum(axis=0)).flatten()
        Q_data, Q_row, Q_col = [], [], []
        
        for cluster_id in range(nc):
            cluster_size = cluster_sizes[cluster_id]
            if cluster_size > 0:
                nodes_in_cluster = Q_plus[:, cluster_id].nonzero()[0]
                for node_id in nodes_in_cluster:
                    Q_data.append(1.0 / cluster_size)
                    Q_row.append(cluster_id)
                    Q_col.append(node_id)
        
        Q = csr_matrix((Q_data, (Q_row, Q_col)), shape=(nc, n_nodes))
        
    elif coarsening_method == 'simple':
        Q, Q_plus = simple_coarsening(A, target_nodes=6)
        
    else:
        raise ValueError(f"Unknown coarsening method: {coarsening_method}")
    
    coarsening_time = time.time() - start_time
    
    # Step 2: Compute propagation matrix
    prop_start = time.time()
    
    if propagation_method == 'naive':
        S_coarsened = compute_naive_propagation(Q_plus, A, gnn_type)
    elif propagation_method == 'mp_aware':
        S_coarsened = compute_mp_aware_propagation(Q, Q_plus, A, gnn_type)
    else:
        raise ValueError(f"Unknown propagation method: {propagation_method}")
    
    propagation_time = time.time() - prop_start
    
    # Step 3: Create embeddings
    embed_start = time.time()
    coarse_embeddings = simple_embedding(S_coarsened, dim=8, method='spectral')
    embedding_time = time.time() - embed_start
    
    # Step 4: Refine embeddings back to original size
    refine_start = time.time()
    refined_embeddings = Q_plus @ coarse_embeddings
    refinement_time = time.time() - refine_start
    
    # Step 5: Test accuracy
    accuracy_start = time.time()
    accuracy = test_downstream_accuracy(refined_embeddings, labels)
    accuracy_time = time.time() - accuracy_start
    
    total_time = time.time() - start_time
    
    # Results summary
    results = {
        'method': f"{coarsening_method}+{propagation_method}+{gnn_type}",
        'accuracy': accuracy,
        'coarsening_time': coarsening_time,
        'propagation_time': propagation_time,
        'embedding_time': embedding_time,
        'refinement_time': refinement_time,
        'accuracy_time': accuracy_time,
        'total_time': total_time,
        'coarsened_nodes': Q.shape[0],
        'compression_ratio': A.shape[0] / Q.shape[0]
    }
    
    print(f"Results:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Coarsened nodes: {A.shape[0]} → {Q.shape[0]} ({results['compression_ratio']:.1f}x compression)")
    print(f"  Total time: {total_time:.3f}s")
    print(f"    Coarsening: {coarsening_time:.3f}s")
    print(f"    Propagation: {propagation_time:.3f}s")
    print(f"    Embedding: {embedding_time:.3f}s")
    print(f"    Refinement: {refinement_time:.3f}s")
    
    return results

def main():
    """Run comprehensive MP-aware GraphZoom test"""
    print("COMPREHENSIVE MP-AWARE GRAPHZOOM TEST")
    print("="*80)
    
    # Create test graph
    G = create_test_graph()
    labels = create_synthetic_labels(G)
    
    print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Labels: {np.bincount(labels)}")
    
    # Test configurations
    coarsening_methods = ['simple', 'cmg']
    propagation_methods = ['naive', 'mp_aware']
    gnn_types = ['gcn', 'graphsage']
    
    all_results = []
    
    # Baseline: Full graph (no coarsening)
    print(f"\n{'#'*80}")
    print("BASELINE: FULL GRAPH (NO COARSENING)")
    print(f"{'#'*80}")
    
    A_full = nx.adjacency_matrix(G).tocsr()
    for gnn_type in gnn_types:
        S_full = compute_propagation_matrix(A_full, gnn_type)
        full_embeddings = simple_embedding(S_full, dim=8)
        full_accuracy = test_downstream_accuracy(full_embeddings, labels)
        
        baseline_result = {
            'method': f'full_graph+{gnn_type}',
            'accuracy': full_accuracy,
            'total_time': 0.0,  # Assume negligible for small graph
            'coarsened_nodes': G.number_of_nodes(),
            'compression_ratio': 1.0
        }
        all_results.append(baseline_result)
        
        print(f"Full graph + {gnn_type.upper()}: Accuracy = {full_accuracy:.3f}")
    
    # Test all combinations
    print(f"\n{'#'*80}")
    print("COARSENING METHOD COMPARISON")
    print(f"{'#'*80}")
    
    for coarsening_method in coarsening_methods:
        for propagation_method in propagation_methods:
            for gnn_type in gnn_types:
                try:
                    result = run_pipeline_test(
                        coarsening_method, propagation_method, gnn_type, G, labels
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"Error in {coarsening_method}+{propagation_method}+{gnn_type}: {e}")
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Sort by accuracy
    all_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"{'Method':<25} {'Accuracy':<10} {'Time':<8} {'Compression':<12} {'Efficiency'}")
    print("-" * 70)
    
    baseline_accuracy = max([r['accuracy'] for r in all_results if 'full_graph' in r['method']])
    
    for result in all_results:
        method = result['method']
        accuracy = result['accuracy']
        time_taken = result['total_time']
        compression = result['compression_ratio']
        
        # Efficiency score: (accuracy_ratio) / (time_ratio) * compression
        accuracy_ratio = accuracy / baseline_accuracy if baseline_accuracy > 0 else 0
        time_ratio = max(time_taken, 0.001)  # Avoid division by zero
        efficiency = accuracy_ratio * compression / time_ratio
        
        print(f"{method:<25} {accuracy:<10.3f} {time_taken:<8.3f} {compression:<12.1f}x {efficiency:<10.1f}")
    
    # Analysis
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    
    # Find best MP-aware vs naive for each coarsening method
    for coarsening in ['simple', 'cmg']:
        for gnn_type in ['gcn', 'graphsage']:
            naive_results = [r for r in all_results if coarsening in r['method'] and 'naive' in r['method'] and gnn_type in r['method']]
            mp_results = [r for r in all_results if coarsening in r['method'] and 'mp_aware' in r['method'] and gnn_type in r['method']]
            
            if naive_results and mp_results:
                naive_acc = naive_results[0]['accuracy']
                mp_acc = mp_results[0]['accuracy']
                improvement = mp_acc / naive_acc if naive_acc > 0 else 1.0
                
                print(f"{coarsening.upper()} + {gnn_type.upper()}: MP-aware = {mp_acc:.3f}, Naive = {naive_acc:.3f}, Improvement = {improvement:.2f}x")
    
    return all_results

if __name__ == "__main__":
    main()
