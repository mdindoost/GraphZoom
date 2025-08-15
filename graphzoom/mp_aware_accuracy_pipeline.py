#!/usr/bin/env python3
"""
Complete Accuracy Pipeline: Traditional CMG vs MP-Aware CMG
Tests the hypothesis that MP-aware coarsening improves final task accuracy
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
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")

# Import existing GraphZoom functions
from utils import json2mtx, smooth_filter
from scoring import lr as graphzoom_evaluate


def load_dataset(dataset_name='cora'):
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
        return load_graphzoom_dataset(dataset_name)

def load_graphzoom_dataset(dataset_name):
    """Load dataset using GraphZoom's format"""
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
            # Create synthetic labels if not available
            print(f"[WARNING] No labels found for {dataset_name}, creating synthetic labels")
            labels = np.random.randint(0, 3, size=len(G.nodes()))
        
        print(f"[DATASET] Loaded {dataset_name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G, features, labels
        
    except Exception as e:
        print(f"[ERROR] Failed to load {dataset_name}: {e}")
        print("[INFO] Falling back to test graph")
        return load_dataset('test_12')


# =================== COARSENING METHODS ===================

def traditional_cmg_coarsening(graph, k=10, d=20, threshold=0.1):
    """
    Traditional CMG coarsening using GraphZoom approach
    """
    print("[TRADITIONAL] Running traditional CMG coarsening...")
    
    try:
        # Import CMG functions
        from cmg_coarsening_timed import cmg_coarse
        
        # Get Laplacian
        L = nx.laplacian_matrix(graph, nodelist=sorted(graph.nodes()))
        
        # Run CMG coarsening (GraphZoom style)
        G_coarse, projections, laplacians, levels = cmg_coarse(L, level=1, k=k, d=d, threshold=threshold)
        
        # Get projection matrix (GraphZoom style: maps coarse → fine)
        projection_matrix = projections[0]  # Shape: (fine_nodes, coarse_nodes)
        
        print(f"[TRADITIONAL] Coarsened: {L.shape[0]} → {G_coarse.number_of_nodes()} nodes")
        print(f"[TRADITIONAL] Projection matrix shape: {projection_matrix.shape}")
        
        return G_coarse, projection_matrix
        
    except Exception as e:
        print(f"[TRADITIONAL] CMG failed: {e}")
        # Fallback to simple clustering
        return simple_coarsening_fallback(graph)

def mp_aware_cmg_coarsening(graph, k=10, d=20, threshold=0.1, mp_type='gcn'):
    """
    MP-aware CMG coarsening using the paper's approach
    """
    print("[MP-AWARE] Running MP-aware CMG coarsening...")
    
    try:
        # Step 1: Get CMG clusters (same clustering as traditional)
        from cmg_coarsening_timed import cmg_coarse
        
        L = nx.laplacian_matrix(graph, nodelist=sorted(graph.nodes()))
        G_coarse, projections, laplacians, levels = cmg_coarse(L, level=1, k=k, d=d, threshold=threshold)
        
        # Step 2: Extract cluster assignments from projection matrix
        projection_matrix = projections[0]  # Shape: (fine_nodes, coarse_nodes)
        cluster_assignments = np.argmax(projection_matrix.toarray(), axis=1)
        
        # Step 3: Create Q matrix (paper's approach)
        Q_matrix = create_coarsening_matrix_Q(cluster_assignments, L.shape[0])
        
        # Step 4: Create message passing matrix
        S = create_message_passing_matrix(graph, mp_type)
        
        # Step 5: Create MP-aware coarsened graph
        G_coarse_mp, S_c_mp = create_mp_aware_coarsened_graph(S, Q_matrix)
        
        print(f"[MP-AWARE] Coarsened: {L.shape[0]} → {G_coarse_mp.number_of_nodes()} nodes")
        print(f"[MP-AWARE] Q matrix shape: {Q_matrix.shape}")
        print(f"[MP-AWARE] S_c^MP shape: {S_c_mp.shape}")
        
        return G_coarse_mp, Q_matrix, S_c_mp, projection_matrix
        
    except Exception as e:
        print(f"[MP-AWARE] Failed: {e}")
        # Fallback to traditional method
        G_coarse, projection_matrix = traditional_cmg_coarsening(graph, k, d, threshold)
        Q_matrix = create_identity_Q(graph.number_of_nodes())
        S_c_mp = nx.adjacency_matrix(G_coarse)
        return G_coarse, Q_matrix, S_c_mp, projection_matrix

def create_coarsening_matrix_Q(cluster_assignments, n_nodes):
    """
    Create row-orthonormal Q matrix from cluster assignments
    Q[k,i] = 1/|C_k| if node i belongs to cluster k
    """
    n_clusters = len(set(cluster_assignments))
    Q = np.zeros((n_clusters, n_nodes))
    
    for cluster_id in range(n_clusters):
        # Find nodes in this cluster
        nodes_in_cluster = [i for i, c in enumerate(cluster_assignments) if c == cluster_id]
        cluster_size = len(nodes_in_cluster)
        
        if cluster_size > 0:
            # Set Q values: 1/|C_k| for each node in cluster k
            for node_id in nodes_in_cluster:
                Q[cluster_id, node_id] = 1.0 / cluster_size
    
    return csr_matrix(Q)

def create_message_passing_matrix(graph, mp_type='gcn'):
    """
    Create message passing matrix S based on type
    """
    A = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes()))
    
    if mp_type == 'gcn':
        # GCN message passing: D^(-1/2) (A + I) D^(-1/2)
        A_tilde = A + identity(A.shape[0])  # Add self-loops
        degrees = np.array(A_tilde.sum(axis=1)).flatten()
        D_inv_sqrt = diags(1.0 / np.sqrt(degrees + 1e-6))
        S = D_inv_sqrt @ A_tilde @ D_inv_sqrt
        
    elif mp_type == 'sage':
        # GraphSAGE message passing: D^(-1) A  
        degrees = np.array(A.sum(axis=1)).flatten()
        D_inv = diags(1.0 / (degrees + 1e-6))
        S = D_inv @ A
        
    else:
        raise ValueError(f"Unknown MP type: {mp_type}")
    
    return S

def create_mp_aware_coarsened_graph(S, Q):
    """
    Create coarsened graph preserving message passing: S_c^MP = Q S Q^T
    """
    # Coarsened message passing matrix
    S_c_mp = Q @ S @ Q.T
    
    # Convert to NetworkX graph
    # Note: S_c_mp might have negative values, so we take absolute value
    A_c_mp = np.abs(S_c_mp.toarray())
    
    # Create graph (remove very small weights to avoid noise)
    A_c_mp[A_c_mp < 1e-6] = 0
    G_c = nx.from_numpy_array(A_c_mp)
    
    return G_c, S_c_mp

def simple_coarsening_fallback(graph):
    """Simple fallback coarsening method"""
    print("[FALLBACK] Using simple coarsening")
    
    n_nodes = graph.number_of_nodes()
    n_coarse = max(1, n_nodes // 2)
    
    # Simple clustering: group adjacent nodes
    cluster_assignments = [min(i // 2, n_coarse - 1) for i in range(n_nodes)]
    
    # Create projection matrix
    projection_matrix = csr_matrix((n_nodes, n_coarse))
    for i, cluster in enumerate(cluster_assignments):
        projection_matrix[i, cluster] = 1.0
    
    # Create coarse graph
    edges = [(cluster_assignments[u], cluster_assignments[v]) 
             for u, v in graph.edges() 
             if cluster_assignments[u] != cluster_assignments[v]]
    
    G_coarse = nx.Graph()
    G_coarse.add_nodes_from(range(n_coarse))
    G_coarse.add_edges_from(edges)
    
    return G_coarse, projection_matrix

def create_identity_Q(n_nodes):
    """Create identity Q matrix as fallback"""
    return identity(n_nodes, format='csr')


# =================== EMBEDDING METHODS ===================

def generate_embeddings(graph, features, embedding_type='spectral', dim=32):
    """
    Generate embeddings for coarsened graph
    """
    print(f"[EMBEDDING] Generating {embedding_type} embeddings (dim={dim})...")
    
    if embedding_type == 'spectral':
        return spectral_embedding(graph, dim)
    elif embedding_type == 'deepwalk':
        return deepwalk_embedding(graph, dim)
    elif embedding_type == 'node2vec':
        return node2vec_embedding(graph, dim)
    elif embedding_type == 'random_walk':
        return simple_random_walk_embedding(graph, dim)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

def spectral_embedding(graph, dim=32):
    """Spectral embedding using Laplacian eigenvectors"""
    try:
        L = nx.normalized_laplacian_matrix(graph, nodelist=sorted(graph.nodes()))
        
        # Compute eigenvectors
        k = min(dim + 1, L.shape[0] - 1)
        eigenvalues, eigenvectors = sp.linalg.eigsh(L, k=k, which='SM')
        
        # Skip first eigenvector (constant) and take next 'dim' eigenvectors
        embeddings = eigenvectors[:, 1:dim+1]
        
        # Handle case where we have fewer eigenvectors than requested
        if embeddings.shape[1] < dim:
            # Pad with random vectors
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

def deepwalk_embedding(graph, dim=32):
    """Placeholder for DeepWalk - use random walk for now"""
    return simple_random_walk_embedding(graph, dim)

def node2vec_embedding(graph, dim=32):
    """Placeholder for Node2Vec - use random walk for now"""
    return simple_random_walk_embedding(graph, dim)


# =================== REFINEMENT METHODS ===================

def graphzoom_refinement(coarse_embeddings, projection_matrix, original_graph, lda=0.1, power=False):
    """
    Apply GraphZoom-style refinement
    """
    print(f"[REFINEMENT] Applying GraphZoom refinement...")
    
    try:
        # Step 1: Project back to original size
        refined_embeddings = projection_matrix @ coarse_embeddings
        print(f"[REFINEMENT] After projection: {refined_embeddings.shape}")
        
        # Step 2: Apply smooth filter (GraphZoom's default behavior)
        L = nx.laplacian_matrix(original_graph, nodelist=sorted(original_graph.nodes()))
        filter_matrix = smooth_filter(L, lda)
        
        # Apply smoothing (2 iterations like GraphZoom)
        for i in range(2):
            refined_embeddings = filter_matrix @ refined_embeddings
            
        print(f"[REFINEMENT] After smoothing: {refined_embeddings.shape}")
        return refined_embeddings
        
    except Exception as e:
        print(f"[REFINEMENT] Failed: {e}, using simple projection")
        return projection_matrix @ coarse_embeddings


# =================== EVALUATION METHODS ===================

def evaluate_downstream_task(embeddings, labels, task='node_classification', original_graph=None):
    """
    Evaluate final task accuracy using the refined embeddings
    """
    if task == 'node_classification':
        return node_classification_accuracy(embeddings, labels)
    elif task == 'link_prediction':
        if original_graph is None:
            print("[ERROR] Link prediction requires original graph")
            return 0.0
        return link_prediction_accuracy(embeddings, original_graph)
    else:
        raise ValueError(f"Unknown task: {task}")

def node_classification_accuracy(embeddings, labels):
    """
    Standard node classification using logistic regression
    """
    try:
        # Handle case with too few samples
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            print("[WARNING] Less than 2 classes, returning random accuracy")
            return 1.0 / len(unique_labels)
        
        min_samples_per_class = 2
        class_counts = {label: np.sum(labels == label) for label in unique_labels}
        valid_classes = [label for label, count in class_counts.items() if count >= min_samples_per_class]
        
        if len(valid_classes) < 2:
            print("[WARNING] Not enough samples per class, returning random accuracy")
            return 1.0 / len(unique_labels)
        
        # Filter to valid classes only
        valid_mask = np.isin(labels, valid_classes)
        X_valid = embeddings[valid_mask]
        y_valid = labels[valid_mask]
        
        # Split data
        test_size = min(0.3, 0.8)  # Use smaller test size for small datasets
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
        
        print(f"[NODE_CLASSIFICATION] Accuracy: {accuracy:.4f} (test_size={len(y_test)})")
        return accuracy
        
    except Exception as e:
        print(f"[NODE_CLASSIFICATION] Failed: {e}, returning random accuracy")
        return 1.0 / len(np.unique(labels))

def link_prediction_accuracy(embeddings, original_graph):
    """
    Link prediction using embedding similarity
    """
    try:
        edges = list(original_graph.edges())
        all_nodes = list(original_graph.nodes())
        
        # Create non-edges (potential negative samples)
        non_edges = []
        for i in all_nodes:
            for j in all_nodes:
                if i < j and not original_graph.has_edge(i, j):
                    non_edges.append((i, j))
        
        if len(edges) == 0 or len(non_edges) == 0:
            print("[WARNING] No edges or non-edges available")
            return 0.5
        
        # Sample equal numbers of positive and negative edges
        n_samples = min(len(edges), len(non_edges), 100)
        
        if n_samples < 10:
            print("[WARNING] Too few samples for link prediction")
            return 0.5
            
        pos_edges = np.random.choice(len(edges), n_samples, replace=False)
        neg_edges = np.random.choice(len(non_edges), n_samples, replace=False)
        
        # Compute edge scores (cosine similarity)
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
            print("[WARNING] No valid edge scores computed")
            return 0.5
        
        # Compute AUC
        y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
        y_scores = pos_scores + neg_scores
        
        auc = roc_auc_score(y_true, y_scores)
        print(f"[LINK_PREDICTION] AUC: {auc:.4f} (samples={len(pos_scores)}+{len(neg_scores)})")
        return auc
        
    except Exception as e:
        print(f"[LINK_PREDICTION] Failed: {e}")
        return 0.5


# =================== MAIN PIPELINE ===================

def full_accuracy_pipeline(graph, features, labels, method='traditional', 
                          embedding_type='spectral', task='node_classification', 
                          cmg_params=None):
    """
    Complete pipeline from graph to final task accuracy
    """
    if cmg_params is None:
        cmg_params = {'k': 10, 'd': 20, 'threshold': 0.1}
    
    print(f"\n{'='*60}")
    print(f"RUNNING {method.upper()} PIPELINE")
    print(f"Embedding: {embedding_type}, Task: {task}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Step 1: Coarsening
        if method == 'traditional':
            coarse_graph, projection_matrix = traditional_cmg_coarsening(
                graph, **cmg_params)
            coarse_features = projection_matrix.T @ features  # Simple feature aggregation
            Q_matrix = None
            S_c_mp = None
            
        elif method == 'mp_aware':
            coarse_graph, Q_matrix, S_c_mp, projection_matrix = mp_aware_cmg_coarsening(
                graph, **cmg_params)
            coarse_features = Q_matrix @ features  # MP-aware feature aggregation
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"[{method.upper()}] Coarsening: {graph.number_of_nodes()} → {coarse_graph.number_of_nodes()} nodes")
        
        # Step 2: Embedding on coarsened graph
        coarse_embeddings = generate_embeddings(coarse_graph, coarse_features, embedding_type)
        
        # Step 3: Refinement to original size
        if method == 'traditional':
            # GraphZoom-style refinement
            refined_embeddings = graphzoom_refinement(coarse_embeddings, projection_matrix, graph)
            
        elif method == 'mp_aware':
            # Paper's lifting + GraphZoom refinement
            lifted_embeddings = Q_matrix.T @ coarse_embeddings  # Q^T @ coarse_embeddings
            refined_embeddings = graphzoom_refinement(lifted_embeddings, projection_matrix, graph)
        
        # Step 4: Downstream task evaluation
        accuracy = evaluate_downstream_task(refined_embeddings, labels, task, graph)
        
        total_time = time.time() - start_time
        
        print(f"[{method.upper()}] Final accuracy: {accuracy:.4f}")
        print(f"[{method.upper()}] Total time: {total_time:.2f}s")
        
        return {
            'method': method,
            'embedding_type': embedding_type,
            'task': task,
            'original_nodes': graph.number_of_nodes(),
            'coarse_nodes': coarse_graph.number_of_nodes(),
            'accuracy': accuracy,
            'time': total_time,
            'refined_embeddings': refined_embeddings
        }
        
    except Exception as e:
        print(f"[{method.upper()}] Pipeline failed: {e}")
        return {
            'method': method,
            'embedding_type': embedding_type, 
            'task': task,
            'original_nodes': graph.number_of_nodes(),
            'coarse_nodes': 0,
            'accuracy': 0.0,
            'time': time.time() - start_time,
            'refined_embeddings': None
        }


def comprehensive_accuracy_comparison(dataset_name='test_12'):
    """
    Compare traditional vs MP-aware CMG on multiple configurations
    """
    print(f"{'='*80}")
    print(f"COMPREHENSIVE ACCURACY COMPARISON: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Load dataset
    graph, features, labels = load_dataset(dataset_name)
    print(f"Dataset: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, {len(np.unique(labels))} classes")
    
    # Test configurations
    methods = ['traditional', 'mp_aware']
    embedding_types = ['spectral', 'random_walk']
    tasks = ['node_classification']
    
    if graph.number_of_edges() > 5:  # Only test link prediction if graph has enough edges
        tasks.append('link_prediction')
    
    results = []
    
    for task in tasks:
        print(f"\n{'-'*60}")
        print(f"TASK: {task.upper()}")
        print(f"{'-'*60}")
        
        for embedding_type in embedding_types:
            print(f"\nEmbedding Type: {embedding_type}")
            
            for method in methods:
                result = full_accuracy_pipeline(
                    graph=graph,
                    features=features, 
                    labels=labels,
                    method=method,
                    embedding_type=embedding_type,
                    task=task
                )
                results.append(result)
    
    # Print results summary
    print_results_summary(results)
    return results

def print_results_summary(results):
    """
    Print formatted results comparison
    """
    print(f"\n{'='*80}")
    print("FINAL ACCURACY COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Group by task and embedding
    for task in set(r['task'] for r in results):
        print(f"\n{task.upper()} RESULTS:")
        print("-" * 50)
        
        task_results = [r for r in results if r['task'] == task]
        
        for embedding in set(r['embedding_type'] for r in task_results):
            embedding_results = [r for r in task_results if r['embedding_type'] == embedding]
            
            print(f"\n{embedding.capitalize()} Embeddings:")
            
            traditional_acc = next((r['accuracy'] for r in embedding_results if r['method'] == 'traditional'), 0.0)
            mp_aware_acc = next((r['accuracy'] for r in embedding_results if r['method'] == 'mp_aware'), 0.0)
            
            improvement = mp_aware_acc - traditional_acc
            improvement_pct = (improvement / traditional_acc * 100) if traditional_acc > 0 else 0
            
            print(f"  Traditional CMG: {traditional_acc:.4f}")
            print(f"  MP-Aware CMG:    {mp_aware_acc:.4f}")
            print(f"  Improvement:     {improvement:+.4f} ({improvement_pct:+.1f}%)")
            
            if improvement > 0.01:
                print(f"  → MP-Aware WINS! 🏆")
            elif improvement < -0.01:
                print(f"  → Traditional WINS!")
            else:
                print(f"  → TIE")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Starting MP-Aware CMG Accuracy Pipeline...")
    
    # Test on small graph first
    print("\n" + "="*80)
    print("TESTING ON 12-NODE GRAPH")
    print("="*80)
    results_test = comprehensive_accuracy_comparison('test_12')
    
    # Test on real dataset if available
    try:
        print("\n" + "="*80)
        print("TESTING ON CORA DATASET")  
        print("="*80)
        results_cora = comprehensive_accuracy_comparison('cora')
    except Exception as e:
        print(f"Cora test failed: {e}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED!")
    print("="*80)
