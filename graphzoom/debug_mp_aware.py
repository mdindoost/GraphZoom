#!/usr/bin/env python3
"""
Debug script for MP-Aware pipeline failures
Add detailed logging to identify where the pipeline breaks
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix, diags, identity

def debug_mp_aware_coarsening(graph, k=10, d=20, threshold=0.1, mp_type='gcn'):
    """
    Debug version of MP-aware coarsening with detailed logging
    """
    print("\n" + "="*50)
    print("DEBUGGING MP-AWARE COARSENING")
    print("="*50)
    
    try:
        print(f"Input graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Step 1: Get CMG clusters
        print("\nStep 1: Getting CMG clusters...")
        from cmg_coarsening_timed import cmg_coarse
        
        L = nx.laplacian_matrix(graph, nodelist=sorted(graph.nodes()))
        print(f"Laplacian shape: {L.shape}")
        
        G_coarse, projections, laplacians, levels = cmg_coarse(L, level=1, k=k, d=d, threshold=threshold)
        print(f"CMG coarsening successful: {L.shape[0]} → {G_coarse.number_of_nodes()} nodes")
        
        # Step 2: Extract cluster assignments
        print("\nStep 2: Extracting cluster assignments...")
        projection_matrix = projections[0]
        print(f"Projection matrix shape: {projection_matrix.shape}")
        print(f"Projection matrix type: {type(projection_matrix)}")
        print(f"Projection matrix nnz: {projection_matrix.nnz}")
        
        # Convert to dense to examine
        proj_dense = projection_matrix.toarray()
        print(f"Projection matrix range: [{proj_dense.min():.3f}, {proj_dense.max():.3f}]")
        
        cluster_assignments = np.argmax(proj_dense, axis=1)
        print(f"Cluster assignments: {cluster_assignments}")
        print(f"Number of clusters: {len(set(cluster_assignments))}")
        print(f"Cluster sizes: {[np.sum(cluster_assignments == i) for i in set(cluster_assignments)]}")
        
        # Step 3: Create Q matrix
        print("\nStep 3: Creating Q matrix...")
        Q_matrix = create_coarsening_matrix_Q_debug(cluster_assignments, L.shape[0])
        print(f"Q matrix shape: {Q_matrix.shape}")
        print(f"Q matrix nnz: {Q_matrix.nnz}")
        
        Q_dense = Q_matrix.toarray()
        print(f"Q matrix range: [{Q_dense.min():.3f}, {Q_dense.max():.3f}]")
        print(f"Q row sums: {Q_dense.sum(axis=1)}")  # Should be 1
        print(f"Q col sums: {Q_dense.sum(axis=0)}")
        
        # Step 4: Create message passing matrix
        print("\nStep 4: Creating message passing matrix...")
        S = create_message_passing_matrix_debug(graph, mp_type)
        print(f"S matrix shape: {S.shape}")
        print(f"S matrix nnz: {S.nnz}")
        
        S_dense = S.toarray()
        print(f"S matrix range: [{S_dense.min():.3f}, {S_dense.max():.3f}]")
        
        # Step 5: Create S_c^MP
        print("\nStep 5: Creating S_c^MP...")
        print(f"Computing Q @ S @ Q^T...")
        print(f"Q shape: {Q_matrix.shape}, S shape: {S.shape}")
        
        QS = Q_matrix @ S
        print(f"Q @ S shape: {QS.shape}")
        
        S_c_mp = QS @ Q_matrix.T
        print(f"S_c^MP shape: {S_c_mp.shape}")
        print(f"S_c^MP nnz: {S_c_mp.nnz}")
        
        S_c_mp_dense = S_c_mp.toarray()
        print(f"S_c^MP range: [{S_c_mp_dense.min():.3f}, {S_c_mp_dense.max():.3f}]")
        print(f"S_c^MP matrix:\n{S_c_mp_dense}")
        
        # Step 6: Create coarsened graph
        print("\nStep 6: Creating coarsened graph...")
        A_c_mp = np.abs(S_c_mp_dense)
        A_c_mp[A_c_mp < 1e-6] = 0  # Remove tiny values
        
        print(f"Coarsened adjacency range: [{A_c_mp.min():.3f}, {A_c_mp.max():.3f}]")
        print(f"Coarsened adjacency nnz: {np.count_nonzero(A_c_mp)}")
        print(f"Coarsened adjacency matrix:\n{A_c_mp}")
        
        G_c = nx.from_numpy_array(A_c_mp)
        print(f"Coarsened graph: {G_c.number_of_nodes()} nodes, {G_c.number_of_edges()} edges")
        
        if G_c.number_of_edges() == 0:
            print("⚠️  WARNING: Coarsened graph has NO EDGES!")
        
        return G_c, Q_matrix, S_c_mp, projection_matrix, True
        
    except Exception as e:
        print(f"❌ MP-aware coarsening failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, False

def create_coarsening_matrix_Q_debug(cluster_assignments, n_nodes):
    """
    Debug version of Q matrix creation
    """
    print(f"  Creating Q matrix from {len(cluster_assignments)} assignments for {n_nodes} nodes")
    
    unique_clusters = set(cluster_assignments)
    n_clusters = len(unique_clusters)
    print(f"  Found {n_clusters} unique clusters: {sorted(unique_clusters)}")
    
    Q = np.zeros((n_clusters, n_nodes))
    
    for cluster_id in range(n_clusters):
        nodes_in_cluster = [i for i, c in enumerate(cluster_assignments) if c == cluster_id]
        cluster_size = len(nodes_in_cluster)
        
        print(f"  Cluster {cluster_id}: {cluster_size} nodes {nodes_in_cluster}")
        
        if cluster_size > 0:
            for node_id in nodes_in_cluster:
                Q[cluster_id, node_id] = 1.0 / cluster_size
    
    return csr_matrix(Q)

def create_message_passing_matrix_debug(graph, mp_type='gcn'):
    """
    Debug version of message passing matrix creation
    """
    print(f"  Creating {mp_type} message passing matrix")
    
    A = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes()))
    print(f"  Adjacency matrix shape: {A.shape}, nnz: {A.nnz}")
    
    if mp_type == 'gcn':
        # GCN: D^(-1/2) (A + I) D^(-1/2)
        A_tilde = A + identity(A.shape[0])
        degrees = np.array(A_tilde.sum(axis=1)).flatten()
        print(f"  Degrees (with self-loops): {degrees}")
        
        D_inv_sqrt = diags(1.0 / np.sqrt(degrees + 1e-6))
        S = D_inv_sqrt @ A_tilde @ D_inv_sqrt
        
    elif mp_type == 'sage':
        # SAGE: D^(-1) A
        degrees = np.array(A.sum(axis=1)).flatten()
        print(f"  Degrees: {degrees}")
        
        D_inv = diags(1.0 / (degrees + 1e-6))
        S = D_inv @ A
        
    else:
        raise ValueError(f"Unknown MP type: {mp_type}")
    
    print(f"  Message passing matrix nnz: {S.nnz}")
    return S

def debug_embedding_generation(graph, embedding_type='spectral'):
    """
    Debug embedding generation
    """
    print(f"\n--- Debugging {embedding_type} embedding generation ---")
    print(f"Input graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    if graph.number_of_nodes() == 0:
        print("❌ EMPTY GRAPH - cannot generate embeddings!")
        return None
    
    if graph.number_of_edges() == 0:
        print("⚠️  WARNING: Graph has no edges - embeddings will be poor")
    
    try:
        if embedding_type == 'spectral':
            L = nx.normalized_laplacian_matrix(graph, nodelist=sorted(graph.nodes()))
            print(f"Laplacian shape: {L.shape}, nnz: {L.nnz}")
            
            k = min(8, L.shape[0] - 1)
            if k <= 0:
                print("❌ Cannot compute eigenvectors - graph too small")
                return np.random.randn(graph.number_of_nodes(), 8) * 0.1
                
            eigenvalues, eigenvectors = sp.linalg.eigsh(L, k=k, which='SM')
            print(f"Eigenvalues: {eigenvalues}")
            
            embeddings = eigenvectors[:, 1:min(8, eigenvectors.shape[1])]
            if embeddings.shape[1] < 8:
                padding = np.random.randn(embeddings.shape[0], 8 - embeddings.shape[1]) * 0.1
                embeddings = np.hstack([embeddings, padding])
            
            print(f"✅ Generated embeddings: {embeddings.shape}")
            return embeddings
            
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return np.random.randn(graph.number_of_nodes(), 8) * 0.1

def debug_full_mp_aware_pipeline():
    """
    Debug the full MP-aware pipeline step by step
    """
    print("🔍 DEBUGGING FULL MP-AWARE PIPELINE")
    print("="*60)
    
    # Create simple test graph
    edges = [(0,1), (1,2), (2,3), (3,0), (1,3)]  # Simple 4-node cycle with diagonal
    G = nx.Graph()
    G.add_edges_from(edges)
    print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create synthetic features and labels
    features = np.random.randn(4, 8)
    labels = np.array([0, 0, 1, 1])
    print(f"Features shape: {features.shape}")
    print(f"Labels: {labels}")
    
    # Step 1: Debug MP-aware coarsening
    G_c, Q, S_c_mp, P, success = debug_mp_aware_coarsening(G)
    
    if not success:
        print("❌ Coarsening failed - stopping debug")
        return
    
    # Step 2: Debug feature coarsening
    print(f"\nStep 2: Coarsening features...")
    print(f"Q shape: {Q.shape}, features shape: {features.shape}")
    
    try:
        coarse_features = Q @ features
        print(f"✅ Coarsened features shape: {coarse_features.shape}")
    except Exception as e:
        print(f"❌ Feature coarsening failed: {e}")
        return
    
    # Step 3: Debug embedding generation
    embeddings = debug_embedding_generation(G_c, 'spectral')
    if embeddings is None:
        print("❌ Embedding generation failed - stopping debug")
        return
    
    # Step 4: Debug lifting back
    print(f"\nStep 4: Lifting embeddings back...")
    print(f"Q^T shape: {Q.T.shape}, embeddings shape: {embeddings.shape}")
    
    try:
        lifted = Q.T @ embeddings
        print(f"✅ Lifted embeddings shape: {lifted.shape}")
    except Exception as e:
        print(f"❌ Lifting failed: {e}")
        return
    
    # Step 5: Debug final refinement
    print(f"\nStep 5: Final refinement...")
    try:
        from mp_aware_accuracy_pipeline import graphzoom_refinement
        refined = graphzoom_refinement(lifted, P, G)
        print(f"✅ Refined embeddings shape: {refined.shape}")
    except Exception as e:
        print(f"❌ Refinement failed: {e}")
        return
    
    # Step 6: Debug evaluation
    print(f"\nStep 6: Evaluation...")
    try:
        from mp_aware_accuracy_pipeline import node_classification_accuracy
        accuracy = node_classification_accuracy(refined, labels)
        print(f"✅ Final accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return
    
    print("\n🎉 MP-aware pipeline debug completed successfully!")

if __name__ == "__main__":
    debug_full_mp_aware_pipeline()
