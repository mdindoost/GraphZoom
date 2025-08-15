#!/usr/bin/env python3
"""
Complete MP-Aware Pipeline Fix - All components in one file
Fixes the dimension mismatch and import issues
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix, diags, identity
from scipy.io import mmwrite, mmread
import json
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# Import GraphZoom functions
from utils import smooth_filter
from scoring import lr as graphzoom_evaluate

# =================== DATASET LOADING ===================

def load_dataset(dataset_name='test_12'):
    """Load dataset for testing"""
    if dataset_name == 'test_12':
        # Use our 12-node test graph
        edges = [(0,1), (1,2), (1,4), (2,3), (3,4), (4,9), (4,5), 
                 (5,6), (5,7), (6,8), (8,9), (9,10), (9,11), (10,11)]
        
        G = nx.Graph()
        G.add_edges_from(edges)
        for i in range(12):
            if i not in G.nodes():
                G.add_node(i)
        
        # Create synthetic features and labels
        features = np.random.randn(12, 8)
        labels = np.array([0,0,0,0,1,1,1,1,2,2,2,2])  # 3 communities
        
        return G, features, labels
    
    else:
        # Load real dataset (Cora, CiteSeer, etc.)
        try:
            # Load graph
            G_data = json.load(open(f"dataset/{dataset_name}/{dataset_name}-G.json"))
            G = nx.node_link_graph(G_data)
            
            # Load features  
            features = np.load(f"dataset/{dataset_name}/{dataset_name}-feats.npy")
            
            # Load labels (assuming they exist)
            try:
                labels_data = json.load(open(f"dataset/{dataset_name}/{dataset_name}-class_map.json"))
                labels = np.array([labels_data[str(i)] for i in range(len(G.nodes()))])
            except:
                print(f"[WARNING] No labels found for {dataset_name}, creating synthetic labels")
                labels = np.random.randint(0, 3, size=len(G.nodes()))
            
            print(f"[DATASET] Loaded {dataset_name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G, features, labels
            
        except Exception as e:
            print(f"[ERROR] Failed to load {dataset_name}: {e}")
            print("[INFO] Falling back to test graph")
            return load_dataset('test_12')

# =================== FIXED MP-AWARE COARSENING ===================

def create_coarsening_matrix_Q_fixed(cluster_assignments, n_nodes):
    """
    Fixed version of Q matrix creation with better error handling
    """
    print(f"[Q_MATRIX] Creating Q from {len(cluster_assignments)} assignments")
    
    # Handle edge cases
    if len(cluster_assignments) != n_nodes:
        print(f"[Q_MATRIX] WARNING: Assignment length {len(cluster_assignments)} != nodes {n_nodes}")
        cluster_assignments = cluster_assignments[:n_nodes]
    
    # Ensure cluster IDs start from 0 and are contiguous
    unique_clusters = sorted(set(cluster_assignments))
    cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
    mapped_assignments = [cluster_mapping[c] for c in cluster_assignments]
    
    n_clusters = len(unique_clusters)
    print(f"[Q_MATRIX] {n_clusters} clusters: {unique_clusters} → {list(range(n_clusters))}")
    
    # Create Q matrix
    Q = np.zeros((n_clusters, n_nodes))
    
    for new_cluster_id in range(n_clusters):
        nodes_in_cluster = [i for i, c in enumerate(mapped_assignments) if c == new_cluster_id]
        cluster_size = len(nodes_in_cluster)
        
        if cluster_size > 0:
            weight = 1.0 / cluster_size
            for node_id in nodes_in_cluster:
                Q[new_cluster_id, node_id] = weight
        else:
            print(f"[Q_MATRIX] WARNING: Empty cluster {new_cluster_id}")
    
    # Verify and fix Q properties
    row_sums = Q.sum(axis=1)
    col_sums = Q.sum(axis=0)
    
    # Fix nodes with no assignment
    unassigned_nodes = np.where(col_sums == 0)[0]
    if len(unassigned_nodes) > 0:
        print(f"[Q_MATRIX] Fixing {len(unassigned_nodes)} unassigned nodes")
        largest_cluster = np.argmax(row_sums)
        for node_id in unassigned_nodes:
            Q[largest_cluster, node_id] = 1.0
    
    return csr_matrix(Q)

def create_mp_aware_coarsened_graph_fixed(S, Q):
    """
    Fixed version of MP-aware coarsened graph creation
    """
    print(f"[MP_GRAPH] Creating coarsened graph: Q{Q.shape} @ S{S.shape} @ Q^T")
    
    try:
        # Compute S_c^MP = Q S Q^T
        QS = Q @ S
        S_c_mp = QS @ Q.T
        
        print(f"[MP_GRAPH] S_c^MP shape: {S_c_mp.shape}, nnz: {S_c_mp.nnz}")
        
        # Convert to dense for processing
        S_c_mp_dense = S_c_mp.toarray()
        
        # Handle the matrix for graph creation
        A_c_mp = np.abs(S_c_mp_dense)
        
        # Remove very small values to reduce noise
        threshold = 1e-6
        A_c_mp[A_c_mp < threshold] = 0
        
        print(f"[MP_GRAPH] After thresholding: {np.count_nonzero(A_c_mp)} nonzeros")
        
        # If no edges, add minimal connectivity
        if np.count_nonzero(A_c_mp) == 0:
            print(f"[MP_GRAPH] No edges after thresholding, adding minimal connectivity")
            n_clusters = A_c_mp.shape[0]
            for i in range(n_clusters):
                for j in range(i+1, n_clusters):
                    A_c_mp[i, j] = 0.1
                    A_c_mp[j, i] = 0.1
        
        # Create NetworkX graph
        G_c = nx.from_numpy_array(A_c_mp)
        
        print(f"[MP_GRAPH] Created graph: {G_c.number_of_nodes()} nodes, {G_c.number_of_edges()} edges")
        
        return G_c, S_c_mp
        
    except Exception as e:
        print(f"[MP_GRAPH] Failed: {e}")
        
        # Fallback: create simple complete graph
        n_clusters = Q.shape[0]
        G_c = nx.complete_graph(n_clusters)
        S_c_mp = sp.identity(n_clusters)
        
        print(f"[MP_GRAPH] Fallback: complete graph with {n_clusters} nodes")
        return G_c, S_c_mp

def mp_aware_coarsening_fixed(graph, k=10, d=20, threshold=0.1, mp_type='gcn'):
    """
    Fixed version of MP-aware coarsening 
    """
    print("[MP_AWARE_FIXED] Starting MP-aware coarsening...")
    
    try:
        # Step 1: Get CMG clusters
        from cmg_coarsening_timed import cmg_coarse
        
        L = nx.laplacian_matrix(graph, nodelist=sorted(graph.nodes()))
        G_coarse, projections, laplacians, levels = cmg_coarse(L, level=1, k=k, d=d, threshold=threshold)
        
        # Step 2: Extract cluster assignments with fixes
        projection_matrix = projections[0]
        proj_dense = projection_matrix.toarray()
        
        # Ensure each node is assigned to exactly one cluster
        cluster_assignments = np.argmax(proj_dense, axis=1)
        
        # Handle edge case: if projection matrix is all zeros
        if np.all(proj_dense == 0):
            print("[MP_AWARE_FIXED] Projection matrix is all zeros, using simple clustering")
            n_nodes = L.shape[0]
            cluster_assignments = list(range(min(n_nodes, 3)))  # Max 3 clusters
            cluster_assignments = cluster_assignments + [cluster_assignments[-1]] * (n_nodes - len(cluster_assignments))
        
        print(f"[MP_AWARE_FIXED] Cluster assignments: {cluster_assignments}")
        
        # Step 3: Create fixed Q matrix
        Q_matrix = create_coarsening_matrix_Q_fixed(cluster_assignments, L.shape[0])
        
        # Step 4: Create message passing matrix
        A = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes()))
        
        if mp_type == 'gcn':
            A_tilde = A + identity(A.shape[0])
            degrees = np.array(A_tilde.sum(axis=1)).flatten()
            degrees[degrees == 0] = 1  # Avoid division by zero
            D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
            S = D_inv_sqrt @ A_tilde @ D_inv_sqrt
        else:
            degrees = np.array(A.sum(axis=1)).flatten()
            degrees[degrees == 0] = 1
            D_inv = diags(1.0 / degrees)
            S = D_inv @ A
        
        # Step 5: Create fixed MP-aware coarsened graph
        G_coarse_mp, S_c_mp = create_mp_aware_coarsened_graph_fixed(S, Q_matrix)
        
        print(f"[MP_AWARE_FIXED] Success: {L.shape[0]} → {G_coarse_mp.number_of_nodes()} nodes")
        
        return G_coarse_mp, Q_matrix, S_c_mp, projection_matrix
        
    except Exception as e:
        print(f"[MP_AWARE_FIXED] Complete failure: {e}")
        import traceback
        traceback.print_exc()
        
        # Ultimate fallback: simple clustering
        n_nodes = graph.number_of_nodes()
        n_coarse = max(1, n_nodes // 2)
        
        # Simple clustering
        cluster_assignments = [i // 2 for i in range(n_nodes)]
        
        # Create Q matrix
        Q_matrix = create_coarsening_matrix_Q_fixed(cluster_assignments, n_nodes)
        
        # Create simple coarse graph
        G_coarse = nx.path_graph(n_coarse)
        S_c_mp = nx.adjacency_matrix(G_coarse)
        
        # Create projection matrix for compatibility
        projection_matrix = csr_matrix((n_nodes, n_coarse))
        for i, cluster in enumerate(cluster_assignments):
            if cluster < n_coarse:
                projection_matrix[i, cluster] = 1.0
        
        return G_coarse, Q_matrix, S_c_mp, projection_matrix

# =================== TRADITIONAL CMG ===================

def traditional_cmg_coarsening(graph, k=10, d=20, threshold=0.1):
    """
    Traditional CMG coarsening using GraphZoom approach
    """
    print("[TRADITIONAL] Running traditional CMG coarsening...")
    
    try:
        from cmg_coarsening_timed import cmg_coarse
        
        L = nx.laplacian_matrix(graph, nodelist=sorted(graph.nodes()))
        G_coarse, projections, laplacians, levels = cmg_coarse(L, level=1, k=k, d=d, threshold=threshold)
        
        projection_matrix = projections[0]  # Shape: (fine_nodes, coarse_nodes)
        
        print(f"[TRADITIONAL] Coarsened: {L.shape[0]} → {G_coarse.number_of_nodes()} nodes")
        
        return G_coarse, projection_matrix
        
    except Exception as e:
        print(f"[TRADITIONAL] CMG failed: {e}, using fallback")
        
        # Simple fallback
        n_nodes = graph.number_of_nodes()
        n_coarse = max(1, n_nodes // 2)
        
        # Simple clustering
        cluster_assignments = [i // 2 for i in range(n_nodes)]
        
        # Create projection matrix
        projection_matrix = csr_matrix((n_nodes, n_coarse))
        for i, cluster in enumerate(cluster_assignments):
            if cluster < n_coarse:
                projection_matrix[i, cluster] = 1.0
        
        # Create coarse graph
        G_coarse = nx.path_graph(n_coarse)
        
        return G_coarse, projection_matrix

# =================== EMBEDDING GENERATION ===================

def generate_embeddings(graph, features, embedding_type='spectral', dim=32):
    """Generate embeddings for coarsened graph"""
    print(f"[EMBEDDING] Generating {embedding_type} embeddings (dim={dim})...")
    
    if embedding_type == 'spectral':
        return spectral_embedding(graph, dim)
    elif embedding_type == 'random_walk':
        return simple_random_walk_embedding(graph, dim)
    else:
        return spectral_embedding(graph, dim)  # Default fallback

def spectral_embedding(graph, dim=32):
    """Spectral embedding using Laplacian eigenvectors"""
    try:
        L = nx.normalized_laplacian_matrix(graph, nodelist=sorted(graph.nodes()))
        
        k = min(dim + 1, L.shape[0] - 1)
        if k <= 0:
            return np.random.randn(graph.number_of_nodes(), dim) * 0.1
            
        eigenvalues, eigenvectors = sp.linalg.eigsh(L, k=k, which='SM')
        
        # Skip first eigenvector and take next 'dim' eigenvectors
        embeddings = eigenvectors[:, 1:min(dim+1, eigenvectors.shape[1])]
        
        # Pad if needed
        if embeddings.shape[1] < dim:
            remaining = dim - embeddings.shape[1]
            random_vecs = np.random.randn(embeddings.shape[0], remaining) * 0.1
            embeddings = np.hstack([embeddings, random_vecs])
        
        print(f"[SPECTRAL] Embedding shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        print(f"[SPECTRAL] Failed: {e}, using random embeddings")
        return np.random.randn(graph.number_of_nodes(), dim) * 0.1

def simple_random_walk_embedding(graph, dim=32, num_walks=10, walk_length=5):
    """Simple random walk-based embedding"""
    try:
        nodes = list(graph.nodes())
        n_nodes = len(nodes)
        
        if n_nodes == 0:
            return np.random.randn(1, dim) * 0.1
        
        # Generate random walks
        walks = []
        for node in nodes:
            for _ in range(num_walks):
                walk = [node]
                current = node
                for _ in range(walk_length):
                    neighbors = list(graph.neighbors(current))
                    if neighbors:
                        current = np.random.choice(neighbors)
                        walk.append(current)
                    else:
                        break
                walks.append(walk)
        
        # Create co-occurrence matrix
        co_occurrence = np.zeros((n_nodes, n_nodes))
        for walk in walks:
            for i, node1 in enumerate(walk):
                for j, node2 in enumerate(walk):
                    if i != j and node1 < n_nodes and node2 < n_nodes:
                        co_occurrence[node1, node2] += 1.0 / (abs(i - j) + 1)
        
        # SVD for dimensionality reduction
        try:
            U, s, Vt = np.linalg.svd(co_occurrence)
            embeddings = U[:, :dim] * np.sqrt(s[:dim].reshape(1, -1))
        except:
            embeddings = np.random.randn(n_nodes, dim) * 0.1
        
        print(f"[RANDOM_WALK] Embedding shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        print(f"[RANDOM_WALK] Failed: {e}, using random embeddings")
        return np.random.randn(graph.number_of_nodes(), dim) * 0.1

# =================== FIXED REFINEMENT ===================

def mp_aware_refinement_fixed(lifted_embeddings, original_graph, lda=0.1):
    """
    Fixed refinement for MP-aware pipeline - ONLY smoothing, no projection
    """
    print(f"[MP_REFINEMENT] Input embeddings shape: {lifted_embeddings.shape}")
    
    try:
        # Get Laplacian and create smooth filter
        L = nx.laplacian_matrix(original_graph, nodelist=sorted(original_graph.nodes()))
        filter_matrix = smooth_filter(L, lda)
        
        print(f"[MP_REFINEMENT] Filter matrix shape: {filter_matrix.shape}")
        
        # Apply smoothing only (GraphZoom does 2 iterations)
        refined_embeddings = lifted_embeddings.copy()
        for i in range(2):
            refined_embeddings = filter_matrix @ refined_embeddings
        
        print(f"[MP_REFINEMENT] ✅ Smoothing completed successfully")
        return refined_embeddings
        
    except Exception as e:
        print(f"[MP_REFINEMENT] ❌ Smoothing failed: {e}")
        return lifted_embeddings

def graphzoom_refinement_fixed(coarse_embeddings, projection_matrix, original_graph, lda=0.1):
    """
    Fixed GraphZoom refinement for traditional pipeline
    """
    print(f"[GRAPHZOOM_REFINEMENT] Input: {coarse_embeddings.shape}")
    
    try:
        # Step 1: Project back to original size
        refined_embeddings = projection_matrix @ coarse_embeddings
        print(f"[GRAPHZOOM_REFINEMENT] After projection: {refined_embeddings.shape}")
        
        # Step 2: Apply smooth filter
        L = nx.laplacian_matrix(original_graph, nodelist=sorted(original_graph.nodes()))
        filter_matrix = smooth_filter(L, lda)
        
        # Apply smoothing (2 iterations)
        for i in range(2):
            refined_embeddings = filter_matrix @ refined_embeddings
            
        print(f"[GRAPHZOOM_REFINEMENT] ✅ Refinement completed")
        return refined_embeddings
        
    except Exception as e:
        print(f"[GRAPHZOOM_REFINEMENT] ❌ Failed: {e}")
        return projection_matrix @ coarse_embeddings

# =================== EVALUATION ===================

def evaluate_downstream_task(embeddings, labels, task='node_classification', original_graph=None):
    """Evaluate final task accuracy"""
    if task == 'node_classification':
        return node_classification_accuracy(embeddings, labels)
    elif task == 'link_prediction':
        if original_graph is None:
            return 0.0
        return link_prediction_accuracy(embeddings, original_graph)
    else:
        return 0.0

def node_classification_accuracy(embeddings, labels):
    """Node classification using logistic regression"""
    try:
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 1.0 / len(unique_labels)
        
        # Check minimum samples per class
        min_samples = 2
        class_counts = {label: np.sum(labels == label) for label in unique_labels}
        valid_classes = [label for label, count in class_counts.items() if count >= min_samples]
        
        if len(valid_classes) < 2:
            return 1.0 / len(unique_labels)
        
        # Filter to valid classes
        valid_mask = np.isin(labels, valid_classes)
        X_valid = embeddings[valid_mask]
        y_valid = labels[valid_mask]
        
        # Split data
        test_size = min(0.3, 0.5)
        X_train, X_test, y_train, y_test = train_test_split(
            X_valid, y_valid, test_size=test_size, random_state=42, 
            stratify=y_valid if len(y_valid) > 10 else None
        )
        
        # Train classifier
        clf = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        clf.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"[NODE_CLASSIFICATION] Accuracy: {accuracy:.4f}")
        return accuracy
        
    except Exception as e:
        print(f"[NODE_CLASSIFICATION] Failed: {e}")
        return 1.0 / len(np.unique(labels))

def link_prediction_accuracy(embeddings, original_graph):
    """Link prediction using embedding similarity"""
    try:
        edges = list(original_graph.edges())
        all_nodes = list(original_graph.nodes())
        
        # Create non-edges
        non_edges = []
        for i in all_nodes:
            for j in all_nodes:
                if i < j and not original_graph.has_edge(i, j):
                    non_edges.append((i, j))
        
        if len(edges) == 0 or len(non_edges) == 0:
            return 0.5
        
        # Sample edges
        n_samples = min(len(edges), len(non_edges), 100)
        if n_samples < 10:
            return 0.5
            
        pos_edges = np.random.choice(len(edges), n_samples, replace=False)
        neg_edges = np.random.choice(len(non_edges), n_samples, replace=False)
        
        # Compute edge scores
        pos_scores = []
        neg_scores = []
        
        for idx in pos_edges:
            i, j = edges[idx]
            if i < embeddings.shape[0] and j < embeddings.shape[0]:
                score = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8)
                pos_scores.append(score)
        
        for idx in neg_edges:
            i, j = non_edges[idx]
            if i < embeddings.shape[0] and j < embeddings.shape[0]:
                score = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8)
                neg_scores.append(score)
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.5
        
        # Compute AUC
        y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
        y_scores = pos_scores + neg_scores
        
        auc = roc_auc_score(y_true, y_scores)
        print(f"[LINK_PREDICTION] AUC: {auc:.4f}")
        return auc
        
    except Exception as e:
        print(f"[LINK_PREDICTION] Failed: {e}")
        return 0.5

# =================== COMPLETE FIXED PIPELINE ===================

def complete_fixed_pipeline(graph, features, labels, method='mp_aware', 
                           embedding_type='spectral', task='node_classification'):
    """
    Complete fixed pipeline with proper refinement handling
    """
    print(f"\n🔧 COMPLETE FIXED {method.upper()} PIPELINE")
    print(f"Embedding: {embedding_type}, Task: {task}")
    print("="*60)
    
    start_time = time.time()
    
    try:
        if method == 'mp_aware':
            # Step 1: MP-aware coarsening
            coarse_graph, Q_matrix, S_c_mp, projection_matrix = mp_aware_coarsening_fixed(graph)
            
            print(f"[FIXED] Coarsening: {graph.number_of_nodes()} → {coarse_graph.number_of_nodes()} nodes")
            
            # Step 2: Coarsen features
            coarse_features = Q_matrix @ features
            print(f"[FIXED] Features: {features.shape} → {coarse_features.shape}")
            
            # Step 3: Generate embeddings
            coarse_embeddings = generate_embeddings(coarse_graph, coarse_features, embedding_type)
            print(f"[FIXED] Coarse embeddings: {coarse_embeddings.shape}")
            
            # Step 4: Lift back with Q^T
            lifted_embeddings = Q_matrix.T @ coarse_embeddings
            print(f"[FIXED] Lifted embeddings: {lifted_embeddings.shape}")
            
            # Step 5: Apply ONLY smoothing (key fix!)
            refined_embeddings = mp_aware_refinement_fixed(lifted_embeddings, graph)
            
        elif method == 'traditional':
            # Step 1: Traditional coarsening
            coarse_graph, projection_matrix = traditional_cmg_coarsening(graph)
            
            # Step 2: Coarsen features
            coarse_features = projection_matrix.T @ features
            
            # Step 3: Generate embeddings
            coarse_embeddings = generate_embeddings(coarse_graph, coarse_features, embedding_type)
            
            # Step 4: Traditional refinement (projection + smoothing)
            refined_embeddings = graphzoom_refinement_fixed(coarse_embeddings, projection_matrix, graph)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Final evaluation
        accuracy = evaluate_downstream_task(refined_embeddings, labels, task, graph)
        
        total_time = time.time() - start_time
        
        print(f"[FIXED] 🎯 Final accuracy: {accuracy:.4f}")
        print(f"[FIXED] ⏱️  Total time: {total_time:.2f}s")
        
        return {
            'method': method,
            'accuracy': accuracy,
            'time': total_time,
            'embeddings': refined_embeddings
        }
        
    except Exception as e:
        print(f"[FIXED] ❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'method': method,
            'accuracy': 0.0,
            'time': time.time() - start_time,
            'embeddings': None
        }

def run_complete_comparison():
    """
    Run the complete fixed comparison
    """
    print("🚀 RUNNING COMPLETE FIXED MP-AWARE COMPARISON")
    print("="*70)
    
    # Set random seed
    np.random.seed(42)
    
    # Load dataset
    graph, features, labels = load_dataset('test_12')
    # graph, features, labels = load_dataset('cora')

    print(f"Dataset: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Test configurations
    embedding_types = ['spectral', 'random_walk']
    tasks = ['node_classification', 'link_prediction']
    
    results = []
    
    for task in tasks:
        print(f"\n{'='*50}")
        print(f"TASK: {task.upper()}")
        print(f"{'='*50}")
        
        for embedding_type in embedding_types:
            print(f"\n🔹 {embedding_type.upper()} EMBEDDINGS:")
            print("-" * 40)
            
            # Traditional pipeline
            print("\n--- Traditional CMG ---")
            trad_result = complete_fixed_pipeline(graph, features, labels, 'traditional', embedding_type, task)
            
            # MP-aware pipeline  
            print("\n--- MP-Aware CMG ---")
            mp_result = complete_fixed_pipeline(graph, features, labels, 'mp_aware', embedding_type, task)
            
            # Store results
            results.append({
                'task': task,
                'embedding': embedding_type,
                'traditional': trad_result['accuracy'],
                'mp_aware': mp_result['accuracy']
            })
            
            # Print comparison
            improvement = mp_result['accuracy'] - trad_result['accuracy']
            print(f"\n📊 COMPARISON:")
            print(f"Traditional: {trad_result['accuracy']:.4f}")
            print(f"MP-Aware:    {mp_result['accuracy']:.4f}")
            print(f"Improvement: {improvement:+.4f}")
            
            if improvement > 0.01:
                print("🏆 MP-Aware WINS!")
            elif improvement < -0.01:
                print("🔴 Traditional wins")
            else:
                print("🤝 Tie")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL COMPLETE COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    for result in results:
        task = result['task']
        embedding = result['embedding']
        trad = result['traditional']
        mp = result['mp_aware']
        improvement = mp - trad
        
        print(f"\n{task.upper()} - {embedding.capitalize()}:")
        print(f"  Traditional: {trad:.4f}")
        print(f"  MP-Aware:    {mp:.4f}")
        print(f"  Improvement: {improvement:+.4f}")
        
        if improvement > 0.01:
            print(f"  🏆 MP-Aware WINS!")
        elif improvement < -0.01:
            print(f"  🔴 Traditional wins")
        else:
            print(f"  🤝 Tie")
    
    return results

if __name__ == "__main__":
    run_complete_comparison()
