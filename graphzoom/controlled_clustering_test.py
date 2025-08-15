#!/usr/bin/env python3
"""
Controlled experiment: Fix clustering method, test ONLY propagation differences
This isolates whether MP-aware propagation works regardless of clustering quality
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix, diags
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def create_hub_graph(n_nodes=20):
    """Create graph with clear hub structure for testing"""
    G = nx.Graph()
    
    # Add hub node (node 0) connected to many others
    hub_degree = min(n_nodes - 1, 10)
    for i in range(1, hub_degree + 1):
        G.add_edge(0, i)
    
    # Add some additional structure
    for i in range(hub_degree + 1, n_nodes - 1):
        # Connect to previous node (creating a path)
        G.add_edge(i, i + 1)
        # Occasionally connect back to hub
        if i % 3 == 0:
            G.add_edge(0, i)
    
    # Ensure all nodes are in the graph
    for i in range(n_nodes):
        if i not in G.nodes():
            G.add_node(i)
    
    return G

def manual_clustering(n_nodes, n_clusters):
    """Create MANUAL clustering - not optimized for anything"""
    print(f"[MANUAL] Creating {n_clusters} clusters for {n_nodes} nodes")
    
    # Simple: divide nodes roughly equally into clusters
    cluster_size = n_nodes // n_clusters
    remainder = n_nodes % n_clusters
    
    clusters = np.zeros(n_nodes, dtype=int)
    node_idx = 0
    
    for cluster_id in range(n_clusters):
        size = cluster_size + (1 if cluster_id < remainder else 0)
        clusters[node_idx:node_idx + size] = cluster_id
        node_idx += size
    
    print(f"[MANUAL] Cluster assignments: {clusters}")
    
    # Build Q^+ matrix
    Q_plus_data, Q_plus_row, Q_plus_col = [], [], []
    for node_id in range(n_nodes):
        cluster_id = clusters[node_id]
        Q_plus_data.append(1.0)
        Q_plus_row.append(node_id)
        Q_plus_col.append(cluster_id)
    
    Q_plus = csr_matrix((Q_plus_data, (Q_plus_row, Q_plus_col)), shape=(n_nodes, n_clusters))
    
    # Build Q matrix (uniform weights)
    cluster_sizes = np.array(Q_plus.sum(axis=0)).flatten()
    Q_data, Q_row, Q_col = [], [], []
    
    for cluster_id in range(n_clusters):
        cluster_size = cluster_sizes[cluster_id]
        if cluster_size > 0:
            nodes_in_cluster = Q_plus[:, cluster_id].nonzero()[0]
            for node_id in nodes_in_cluster:
                Q_data.append(1.0 / cluster_size)
                Q_row.append(cluster_id)
                Q_col.append(node_id)
    
    Q = csr_matrix((Q_data, (Q_row, Q_col)), shape=(n_clusters, n_nodes))
    
    return Q, Q_plus

def compute_gcn_propagation(adjacency):
    """Compute GCN propagation matrix S = D^(-1/2)(A+I)D^(-1/2)"""
    A_hat = adjacency + sp.identity(adjacency.shape[0])
    degrees = np.array(A_hat.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
    S = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return S

def test_propagation_methods(G, n_clusters=None):
    """Test naive vs MP-aware propagation on SAME clustering"""
    n_nodes = G.number_of_nodes()
    if n_clusters is None:
        n_clusters = max(2, n_nodes // 3)
    
    print(f"\n{'='*60}")
    print(f"CONTROLLED TEST: {n_nodes} nodes → {n_clusters} clusters")
    print(f"{'='*60}")
    
    # Get adjacency matrix
    A = nx.adjacency_matrix(G).tocsr()
    
    # Step 1: MANUAL clustering (not optimized for anything)
    Q, Q_plus = manual_clustering(n_nodes, n_clusters)
    
    # Step 2: Compute original propagation matrix
    S_original = compute_gcn_propagation(A)
    
    print(f"[ORIGINAL] S matrix computed: {S_original.shape}")
    print(f"[ORIGINAL] Symmetric: {np.allclose(S_original.toarray(), S_original.T.toarray())}")
    
    # Step 3: NAIVE propagation
    A_coarsened = Q_plus.T @ A @ Q_plus
    S_naive = compute_gcn_propagation(A_coarsened)
    
    print(f"[NAIVE] A_c shape: {A_coarsened.shape}")
    print(f"[NAIVE] S_c shape: {S_naive.shape}")
    print(f"[NAIVE] Symmetric: {np.allclose(S_naive.toarray(), S_naive.T.toarray())}")
    
    # Step 4: MP-AWARE propagation  
    S_c_MP = Q @ S_original @ Q_plus
    
    print(f"[MP-AWARE] S_c^MP shape: {S_c_MP.shape}")
    print(f"[MP-AWARE] Symmetric: {np.allclose(S_c_MP.toarray(), S_c_MP.T.toarray())}")
    
    # Step 5: Message passing preservation test
    print(f"\n{'='*40}")
    print("MESSAGE PASSING PRESERVATION TEST")
    print(f"{'='*40}")
    
    errors_naive = []
    errors_mp = []
    
    for i in range(5):  # Multiple test signals
        np.random.seed(i)
        x = np.random.randn(n_nodes)
        
        # Original message passing
        Sx_original = S_original @ x
        
        # Coarsen signal
        x_c = Q @ x
        
        # Method 1: Naive
        Sx_naive_lifted = Q_plus @ (S_naive @ x_c)
        
        # Method 2: MP-aware
        Sx_mp_lifted = Q_plus @ (S_c_MP @ x_c)
        
        # Compute errors
        error_naive = np.linalg.norm(Sx_original - Sx_naive_lifted)
        error_mp = np.linalg.norm(Sx_original - Sx_mp_lifted)
        
        errors_naive.append(error_naive)
        errors_mp.append(error_mp)
        
        print(f"Test {i+1}: Naive={error_naive:.6f}, MP-aware={error_mp:.6f}, Ratio={error_naive/error_mp:.3f}x")
    
    # Summary statistics
    avg_error_naive = np.mean(errors_naive)
    avg_error_mp = np.mean(errors_mp)
    avg_improvement = avg_error_naive / avg_error_mp
    
    print(f"\nSUMMARY:")
    print(f"  Average naive error:     {avg_error_naive:.6f}")
    print(f"  Average MP-aware error:  {avg_error_mp:.6f}")
    print(f"  Average improvement:     {avg_improvement:.3f}x")
    
    return {
        'avg_error_naive': avg_error_naive,
        'avg_error_mp': avg_error_mp,
        'improvement': avg_improvement,
        'all_errors_naive': errors_naive,
        'all_errors_mp': errors_mp
    }

def create_synthetic_labels(G):
    """Create labels based on graph structure for testing"""
    n_nodes = G.number_of_nodes()
    degrees = dict(G.degree())
    
    labels = np.zeros(n_nodes, dtype=int)
    degree_values = list(degrees.values())
    
    # Create labels based on degree thresholds
    low_deg_thresh = np.percentile(degree_values, 33)
    high_deg_thresh = np.percentile(degree_values, 67)
    
    for node in G.nodes():
        deg = degrees[node]
        if deg <= low_deg_thresh:
            labels[node] = 0  # Low degree
        elif deg <= high_deg_thresh:
            labels[node] = 1  # Medium degree
        else:
            labels[node] = 2  # High degree (hubs)
    
    return labels

def test_downstream_accuracy(G, Q, Q_plus, S_naive, S_c_MP):
    """Test node classification accuracy"""
    print(f"\n{'='*40}")
    print("DOWNSTREAM TASK: NODE CLASSIFICATION")
    print(f"{'='*40}")
    
    labels = create_synthetic_labels(G)
    print(f"Label distribution: {np.bincount(labels)}")
    
    if len(np.unique(labels)) < 2 or len(labels) < 6:
        print("Not enough data for classification test")
        return None
    
    # Simple embedding: use first few columns of propagation matrices
    embed_dim = min(3, S_naive.shape[1])
    
    # Naive embeddings
    embed_naive = np.random.randn(S_naive.shape[0], embed_dim)
    for _ in range(3):  # Power iteration
        embed_naive = S_naive @ embed_naive
        embed_naive = embed_naive / (np.linalg.norm(embed_naive, axis=0, keepdims=True) + 1e-8)
    embed_naive_lifted = Q_plus @ embed_naive
    
    # MP-aware embeddings  
    embed_mp = np.random.randn(S_c_MP.shape[0], embed_dim)
    for _ in range(3):  # Power iteration
        embed_mp = S_c_MP @ embed_mp
        embed_mp = embed_mp / (np.linalg.norm(embed_mp, axis=0, keepdims=True) + 1e-8)
    embed_mp_lifted = Q_plus @ embed_mp
    
    try:
        # Test both embeddings
        X_train, X_test, y_train, y_test = train_test_split(
            embed_naive_lifted, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)
        acc_naive = accuracy_score(y_test, clf.predict(X_test))
        
        X_train, X_test, y_train, y_test = train_test_split(
            embed_mp_lifted, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)
        acc_mp = accuracy_score(y_test, clf.predict(X_test))
        
        print(f"Naive accuracy:    {acc_naive:.3f}")
        print(f"MP-aware accuracy: {acc_mp:.3f}")
        print(f"Improvement:       {acc_mp/acc_naive:.3f}x" if acc_naive > 0 else "inf")
        
        return {'acc_naive': acc_naive, 'acc_mp': acc_mp}
        
    except Exception as e:
        print(f"Classification test failed: {e}")
        return None

def main():
    """Run controlled experiment with multiple graph types"""
    print("CONTROLLED EXPERIMENT: PROPAGATION METHOD COMPARISON")
    print("="*80)
    print("Fixed clustering, varying propagation methods only")
    print("="*80)
    
    test_configs = [
        ("Path graph (12 nodes)", nx.path_graph(12)),
        ("Hub graph (20 nodes)", create_hub_graph(20)),
        ("Random graph (25 nodes)", nx.erdos_renyi_graph(25, 0.15, seed=42)),
    ]
    
    all_results = []
    
    for name, G in test_configs:
        print(f"\n{'#'*80}")
        print(f"TESTING: {name}")
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"{'#'*80}")
        
        # Test message passing preservation
        result = test_propagation_methods(G)
        result['graph_name'] = name
        all_results.append(result)
        
        # Test downstream accuracy if graph is large enough
        if G.number_of_nodes() >= 15:
            n_clusters = max(2, G.number_of_nodes() // 4)
            Q, Q_plus = manual_clustering(G.number_of_nodes(), n_clusters)
            A = nx.adjacency_matrix(G).tocsr()
            S_original = compute_gcn_propagation(A)
            A_coarsened = Q_plus.T @ A @ Q_plus
            S_naive = compute_gcn_propagation(A_coarsened)
            S_c_MP = Q @ S_original @ Q_plus
            
            downstream_result = test_downstream_accuracy(G, Q, Q_plus, S_naive, S_c_MP)
            if downstream_result:
                result.update(downstream_result)
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for result in all_results:
        print(f"\n{result['graph_name']}:")
        print(f"  Message passing improvement: {result['improvement']:.3f}x")
        if 'acc_naive' in result:
            print(f"  Accuracy improvement: {result['acc_mp']/result['acc_naive']:.3f}x")
        
        if result['improvement'] > 1.05:
            print(f"  ✅ MP-aware shows improvement")
        else:
            print(f"  ❌ MP-aware shows little/no improvement")
    
    print(f"\n{'='*80}")
    print("DIAGNOSIS")
    print(f"{'='*80}")
    
    improvements = [r['improvement'] for r in all_results]
    avg_improvement = np.mean(improvements)
    
    if avg_improvement > 1.1:
        print("✅ MP-aware consistently outperforms naive")
        print("   → Theory works, issue might be in complex clustering methods")
    elif avg_improvement > 1.02:
        print("⚠️  MP-aware shows small improvements")  
        print("   → Need larger graphs or more complex tasks")
    else:
        print("❌ MP-aware shows no improvement")
        print("   → Likely implementation bug or task too simple")

if __name__ == "__main__":
    main()
