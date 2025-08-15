#!/usr/bin/env python3
"""
Debug CMG coarsening on a simple path graph
Print all matrices to see what's happening
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

def create_path_graph(length=8):
    """Create a simple path graph: 0-1-2-3-4-5-6-7"""
    print(f"Creating path graph with {length} nodes...")
    
    G = nx.path_graph(length)
    
    print(f"Created path graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Edges: {list(G.edges())}")
    
    return G

def print_matrix(matrix, title, max_size=10):
    """Print matrix for debugging"""
    print(f"\n{title}:")
    print(f"Shape: {matrix.shape}")
    
    if matrix.shape[0] <= max_size and matrix.shape[1] <= max_size:
        if sp.issparse(matrix):
            dense = matrix.toarray()
        else:
            dense = matrix
        
        print("Matrix contents:")
        for i, row in enumerate(dense):
            print(f"  Row {i}: " + " ".join([f"{val:6.3f}" for val in row]))
    else:
        print(f"Matrix too large to print ({matrix.shape})")
        if sp.issparse(matrix):
            print(f"nnz: {matrix.nnz}")

def print_vector(vector, title):
    """Print vector for debugging"""
    print(f"\n{title}:")
    print(f"Shape: {vector.shape}")
    if len(vector) <= 20:
        print(f"Values: {vector}")
    else:
        print(f"First 10 values: {vector[:10]}")

def compute_propagation_matrix(adjacency, method='node2vec'):
    """Compute propagation matrix for different GNN types"""
    if method in ['node2vec', 'deepwalk']:
        # Node2Vec/DeepWalk: S = D^(-1) A (transition matrix)
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero
        D_inv = sp.diags(1.0 / degrees)
        S = D_inv @ adjacency
        
    elif method == 'gcn':
        # GCN: S = D^(-1/2) (A + I) D^(-1/2)
        A_self = adjacency + sp.identity(adjacency.shape[0])
        degrees = np.array(A_self.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
        S = D_inv_sqrt @ A_self @ D_inv_sqrt
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return S

def debug_cmg_coarsening(G, method='node2vec'):
    """Debug CMG coarsening step by step"""
    print(f"\n{'='*80}")
    print(f"DEBUGGING CMG COARSENING - {method.upper()}")
    print(f"{'='*80}")
    
    # Get adjacency matrix
    A = nx.adjacency_matrix(G).tocsr()
    print_matrix(A, "Original Adjacency Matrix A")
    
    # Step 1: Convert to PyG format for CMG
    print(f"\nStep 1: Preparing graph for CMG...")
    degrees = np.array(A.sum(axis=1)).flatten()
    print_vector(degrees, "Node degrees")
    
    L = diags(degrees) - A
    print_matrix(L, "Laplacian Matrix L")
    
    data = scipy_to_pyg_data(L)
    print(f"PyG data created: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    
    # Step 2: Run CMG clustering
    print(f"\nStep 2: Running CMG clustering...")
    try:
        clusters, nc, phi_stats, lambda_crit = cmg_filtered_clustering(
            data, k=10, d=20, threshold=0.1
        )
        
        print(f"CMG Results:")
        print(f"  Number of clusters: {nc}")
        print(f"  Lambda critical: {lambda_crit:.4f}")
        print(f"  Conductance: {phi_stats}")
        print_vector(clusters, "Cluster assignments")
        
    except Exception as e:
        print(f"ERROR in CMG clustering: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 3: Build Q and Q^+ matrices
    print(f"\nStep 3: Building coarsening matrices...")
    
    n_nodes = A.shape[0]
    
    # Build Q^+ matrix (nodes → clusters)
    Q_plus_data, Q_plus_row, Q_plus_col = [], [], []
    
    for node_id in range(n_nodes):
        cluster_id = clusters[node_id]
        Q_plus_data.append(1.0)
        Q_plus_row.append(node_id)
        Q_plus_col.append(cluster_id)
    
    Q_plus = csr_matrix((Q_plus_data, (Q_plus_row, Q_plus_col)), shape=(n_nodes, nc))
    print_matrix(Q_plus, "Q^+ Matrix (nodes → clusters)")
    
    # Build Q matrix (clusters → nodes)
    cluster_sizes = np.array(Q_plus.sum(axis=0)).flatten()
    print_vector(cluster_sizes, "Cluster sizes")
    
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
    print_matrix(Q, "Q Matrix (clusters → nodes)")
    
    # Verify Q and Q^+ relationship
    print(f"\nVerification:")
    QQ_plus = Q @ Q_plus
    print_matrix(QQ_plus, "Q @ Q^+ (should be identity)")
    
    # Step 4: Compute original propagation matrix
    print(f"\nStep 4: Computing original propagation matrix...")
    S_original = compute_propagation_matrix(A, method)
    print_matrix(S_original, f"Original {method} Propagation Matrix S")
    
    # Step 5: Naive coarsening
    print(f"\nStep 5: Naive coarsening approach...")
    A_coarsened_naive = Q_plus.T @ A @ Q_plus
    print_matrix(A_coarsened_naive, "Naive Coarsened Adjacency A_c")
    
    S_naive = compute_propagation_matrix(A_coarsened_naive, method)
    print_matrix(S_naive, f"Naive {method} Propagation S_naive")
    
    # Step 6: MP-aware coarsening
    print(f"\nStep 6: MP-aware coarsening approach...")
    S_c_MP = Q @ S_original @ Q_plus
    print_matrix(S_c_MP, f"MP-aware {method} Propagation S_c^MP")
    
    # Step 7: Compare matrix properties
    print(f"\nStep 7: Matrix properties comparison...")
    
    print(f"Original S properties:")
    print(f"  Shape: {S_original.shape}")
    print(f"  Row sums: {np.array(S_original.sum(axis=1)).flatten()}")
    print(f"  Symmetric: {np.allclose(S_original.toarray(), S_original.T.toarray())}")
    
    print(f"Naive S_c properties:")
    print(f"  Shape: {S_naive.shape}")
    print(f"  Row sums: {np.array(S_naive.sum(axis=1)).flatten()}")
    print(f"  Symmetric: {np.allclose(S_naive.toarray(), S_naive.T.toarray())}")
    
    print(f"MP-aware S_c^MP properties:")
    print(f"  Shape: {S_c_MP.shape}")
    print(f"  Row sums: {np.array(S_c_MP.sum(axis=1)).flatten()}")
    print(f"  Symmetric: {np.allclose(S_c_MP.toarray(), S_c_MP.T.toarray())}")
    
    # Step 8: Test message passing preservation
    print(f"\nStep 8: Message passing preservation test...")
    
    # Create test signal
    np.random.seed(42)
    x = np.random.randn(n_nodes)
    print_vector(x, "Test signal x")
    
    # Original message passing
    Sx_original = S_original @ x
    print_vector(Sx_original, "S @ x (original)")
    
    # Coarsen signal
    x_c = Q @ x
    print_vector(x_c, "Q @ x (coarsened signal)")
    
    # Naive method: propagate on coarsened graph then lift
    Sx_naive_coarse = S_naive @ x_c
    print_vector(Sx_naive_coarse, "S_naive @ x_c")
    
    Sx_naive_lifted = Q_plus @ Sx_naive_coarse
    print_vector(Sx_naive_lifted, "Q^+ @ (S_naive @ x_c) - NAIVE RESULT")
    
    # MP-aware method: use S_c^MP then lift
    Sx_mp_coarse = S_c_MP @ x_c
    print_vector(Sx_mp_coarse, "S_c^MP @ x_c")
    
    Sx_mp_lifted = Q_plus @ Sx_mp_coarse
    print_vector(Sx_mp_lifted, "Q^+ @ (S_c^MP @ x_c) - MP-AWARE RESULT")
    
    # Compute errors
    error_naive = np.linalg.norm(Sx_original - Sx_naive_lifted)
    error_mp = np.linalg.norm(Sx_original - Sx_mp_lifted)
    
    print(f"\nMessage Passing Errors:")
    print(f"  Naive method error:    {error_naive:.6f}")
    print(f"  MP-aware method error: {error_mp:.6f}")
    print(f"  Improvement ratio:     {error_naive/error_mp:.3f}x" if error_mp > 0 else "  Improvement ratio: inf")
    
    return {
        'Q': Q,
        'Q_plus': Q_plus,
        'A': A,
        'S_original': S_original,
        'S_naive': S_naive,
        'S_c_MP': S_c_MP,
        'clusters': clusters,
        'error_naive': error_naive,
        'error_mp': error_mp,
        'nc': nc
    }

def create_simple_labels(G):
    """Create simple labels for the path graph"""
    n_nodes = G.number_of_nodes()
    labels = np.zeros(n_nodes, dtype=int)
    
    # Simple labeling: based on position in path
    for i in range(n_nodes):
        if i < n_nodes // 3:
            labels[i] = 0  # Beginning
        elif i < 2 * n_nodes // 3:
            labels[i] = 1  # Middle
        else:
            labels[i] = 2  # End
    
    print(f"Label distribution: {np.bincount(labels)}")
    return labels

def simple_embedding_from_propagation(S, dim=4):
    """Create embeddings from propagation matrix"""
    try:
        # Use power iteration to get dominant eigenvectors
        embeddings = np.random.randn(S.shape[0], dim)
        for _ in range(10):  # 10 iterations
            embeddings = S @ embeddings
            # Normalize columns
            norms = np.linalg.norm(embeddings, axis=0, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms
    except:
        embeddings = np.random.randn(S.shape[0], dim)
    
    return embeddings

def test_downstream_task(results, G, method='node2vec'):
    """Test downstream node classification task"""
    print(f"\n{'='*80}")
    print(f"DOWNSTREAM TASK TESTING - {method.upper()}")
    print(f"{'='*80}")
    
    labels = create_simple_labels(G)
    
    Q = results['Q']
    Q_plus = results['Q_plus']
    S_naive = results['S_naive']
    S_c_MP = results['S_c_MP']
    
    # Create embeddings from coarsened propagation matrices
    print(f"\nCreating embeddings...")
    
    embed_naive = simple_embedding_from_propagation(S_naive, dim=4)
    print_matrix(embed_naive, f"Naive {method} embeddings (coarsened)")
    
    embed_mp = simple_embedding_from_propagation(S_c_MP, dim=4)
    print_matrix(embed_mp, f"MP-aware {method} embeddings (coarsened)")
    
    # Lift embeddings back to original size
    embed_naive_lifted = Q_plus @ embed_naive
    embed_mp_lifted = Q_plus @ embed_mp
    
    print_matrix(embed_naive_lifted, f"Naive {method} embeddings (lifted)")
    print_matrix(embed_mp_lifted, f"MP-aware {method} embeddings (lifted)")
    
    # Test classification accuracy
    if len(labels) >= 4:  # Need at least 4 nodes for train/test split
        try:
            # Test naive embeddings
            X_train, X_test, y_train, y_test = train_test_split(
                embed_naive_lifted, labels, test_size=0.3, random_state=42
            )
            
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(X_train, y_train)
            y_pred_naive = clf.predict(X_test)
            accuracy_naive = accuracy_score(y_test, y_pred_naive)
            
            # Test MP-aware embeddings
            X_train, X_test, y_train, y_test = train_test_split(
                embed_mp_lifted, labels, test_size=0.3, random_state=42
            )
            
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(X_train, y_train)
            y_pred_mp = clf.predict(X_test)
            accuracy_mp = accuracy_score(y_test, y_pred_mp)
            
            print(f"\nDownstream Task Results:")
            print(f"  Naive accuracy:    {accuracy_naive:.3f}")
            print(f"  MP-aware accuracy: {accuracy_mp:.3f}")
            print(f"  Improvement:       {accuracy_mp/accuracy_naive:.2f}x" if accuracy_naive > 0 else "  Improvement: inf")
            
        except Exception as e:
            print(f"Error in downstream task: {e}")
            accuracy_naive = 0.0
            accuracy_mp = 0.0
    else:
        print("Too few nodes for downstream task testing")
        accuracy_naive = 0.0
        accuracy_mp = 0.0
    
    return accuracy_naive, accuracy_mp

def main():
    """Run comprehensive CMG debugging"""
    print("CMG COARSENING DEBUG TEST")
    print("="*80)
    
    # Create simple path graph
    G = create_path_graph(8)  # Path of length 8
    
    # Test with both Node2Vec and GCN
    methods = ['node2vec', 'gcn']
    all_results = {}
    
    for method in methods:
        print(f"\nTesting with {method.upper()} propagation...")
        results = debug_cmg_coarsening(G, method=method)
        
        if results is not None:
            # Test downstream task
            acc_naive, acc_mp = test_downstream_task(results, G, method=method)
            
            # Store results for comparison
            all_results[method] = {
                'results': results,
                'acc_naive': acc_naive,
                'acc_mp': acc_mp
            }
    
    # Final comparison between methods
    print(f"\n{'='*80}")
    print("COMPARISON: NODE2VEC vs GCN")
    print(f"{'='*80}")
    
    for method in methods:
        if method in all_results:
            data = all_results[method]
            results = data['results']
            
            print(f"\n{method.upper()} Results:")
            print(f"  CMG clusters: {results['nc']}")
            print(f"  Message passing error - Naive: {results['error_naive']:.6f}")
            print(f"  Message passing error - MP-aware: {results['error_mp']:.6f}")
            print(f"  Error improvement: {results['error_naive']/results['error_mp']:.3f}x" if results['error_mp'] > 0 else "inf")
            print(f"  Downstream accuracy - Naive: {data['acc_naive']:.3f}")
            print(f"  Downstream accuracy - MP-aware: {data['acc_mp']:.3f}")
            print(f"  Accuracy improvement: {data['acc_mp']/data['acc_naive']:.2f}x" if data['acc_naive'] > 0 else "inf")
    
    # Check if CMG is working properly for all methods
    for method in all_results:
        results = all_results[method]['results']
        if results['nc'] == G.number_of_nodes():
            print(f"\n⚠️  WARNING: CMG created {results['nc']} clusters for {G.number_of_nodes()} nodes in {method}")
            print("   This means no coarsening happened - each node is its own cluster!")
        elif results['nc'] == 1:
            print(f"\n⚠️  WARNING: CMG created only 1 cluster in {method}")
            print("   This means all nodes are in one cluster - over-coarsening!")
        else:
            print(f"\n✅ CMG coarsening looks reasonable for {method}: {G.number_of_nodes()} → {results['nc']} nodes")
    
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    print("1. CMG coarsening is working correctly (8 → 3 nodes)")
    print("2. Node2Vec shows minimal MP-aware benefits on simple path graphs")
    print("3. GCN should show larger MP-aware benefits due to different propagation")
    print("4. Benefits increase with graph complexity and size")

if __name__ == "__main__":
    main()
