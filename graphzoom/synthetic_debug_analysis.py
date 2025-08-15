#!/usr/bin/env python3
"""
Synthetic Small Graph Debugging for True Coarsened GraphSAGE
=====================================================

This script creates a small synthetic graph and tracks information flow
through each level of True Coarsened GraphSAGE to identify exactly where
and why accuracy is lost.

UPDATED: Now includes test for FIXED refinement version.
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import diags, csr_matrix
import torch
from torch_geometric.data import Data
from collections import Counter
import matplotlib.pyplot as plt
import time
import sys
import os

# Add paths for imports (adjust as needed)
sys.path.append('.')
sys.path.append('..')

def create_synthetic_graph():
    """Create the synthetic test graph with known structure."""
    
    print("🏗️  CREATING SYNTHETIC GRAPH")
    print("="*50)
    
    # Define edges as specified
    edges = [(0,1), (1,2), (0,2), (2,3), (3,4), (3,5), (4,5), (4,6), (6,7), (7,8), (8,9)]
    
    # Create NetworkX graph
    G = nx.Graph()
    G.add_edges_from(edges)
    
    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Edges: {edges}")
    
    # Visualize structure
    print("\nExpected community structure:")
    print("  Triangle 1: [0,1,2] - densely connected")
    print("  Triangle 2: [3,4,5] - densely connected") 
    print("  Path: [6,7,8,9] - linear chain")
    print("  Bridge: node 2 connects to node 3")
    
    return G, edges

def create_synthetic_features():
    """Create discriminative synthetic features aligned with expected clusters."""
    
    print("\n🎨 CREATING SYNTHETIC FEATURES")
    print("="*50)
    
    # 10 nodes, 8 features
    # Design features to be discriminative for expected clusters
    features = np.array([
        # Expected Cluster 1 (nodes 0,1,2): Class A pattern
        [1.0, 0.0, 0.0, 1.0, 0.5, 0.1, 0.0, 0.0],  # node 0
        [0.9, 0.1, 0.0, 1.1, 0.4, 0.0, 0.1, 0.0],  # node 1  
        [1.1, 0.0, 0.1, 0.9, 0.6, 0.1, 0.0, 0.1],  # node 2
        
        # Expected Cluster 2 (nodes 3,4,5): Class B pattern  
        [0.0, 1.0, 0.0, 0.0, 0.1, 1.0, 0.5, 0.0],  # node 3
        [0.1, 1.1, 0.1, 0.0, 0.0, 0.9, 0.4, 0.1],  # node 4
        [0.0, 0.9, 0.0, 0.1, 0.0, 1.1, 0.6, 0.0],  # node 5
        
        # Expected Cluster 3 (nodes 6,7,8,9): Class C pattern
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.1, 1.0],  # node 6
        [0.1, 0.0, 1.1, 0.0, 0.1, 0.0, 0.0, 0.9],  # node 7
        [0.0, 0.1, 0.9, 0.1, 0.0, 0.0, 0.0, 1.1],  # node 8
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.1, 1.0],  # node 9
    ])
    
    # Ground truth labels
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])  # Classes A, B, C
    
    print(f"Features shape: {features.shape}")
    print(f"Labels: {labels}")
    
    # Show feature patterns
    print("\nFeature patterns by expected cluster:")
    for cluster_id, cluster_name in enumerate(['A (0,1,2)', 'B (3,4,5)', 'C (6,7,8,9)']):
        cluster_nodes = np.where(labels == cluster_id)[0]
        cluster_features = features[cluster_nodes]
        print(f"  Cluster {cluster_name}:")
        print(f"    Mean: {np.mean(cluster_features, axis=0)}")
        print(f"    Std:  {np.std(cluster_features, axis=0)}")
    
    return features, labels

def networkx_to_laplacian(G):
    """Convert NetworkX graph to Laplacian matrix for CMG."""
    
    print("\n🔄 CONVERTING TO LAPLACIAN MATRIX")
    print("="*50)
    
    # Get adjacency matrix
    adj = nx.adjacency_matrix(G)
    
    # Build Laplacian
    degrees = np.array(adj.sum(axis=1)).flatten()
    degree_matrix = sp.diags(degrees)
    laplacian = degree_matrix - adj
    
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Laplacian matrix shape: {laplacian.shape}")
    print(f"Laplacian nnz: {laplacian.nnz}")
    
    return laplacian.tocsr()

def run_cmg_clustering(laplacian, k=5, d=10, threshold=0.1):
    """Run CMG clustering on the synthetic graph."""
    
    print("\n🧠 RUNNING CMG CLUSTERING")
    print("="*50)
    print(f"CMG parameters: k={k}, d={d}, threshold={threshold}")
    
    try:
        # Try to import CMG
        from filtered_timed import cmg_filtered_clustering
        
        # Convert to PyG format
        degree_diag = diags(laplacian.diagonal(), 0)
        adjacency = degree_diag - laplacian
        adjacency = (adjacency + adjacency.T) / 2
        adjacency.data = np.abs(adjacency.data)
        
        # Convert to edge_index
        coo = adjacency.tocoo()
        edge_index = np.vstack([coo.row, coo.col])
        
        data = Data(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            num_nodes=laplacian.shape[0]
        )
        
        # Run CMG
        cluster_assignments, num_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
            data, k=k, d=d, threshold=threshold
        )
        
        print(f"✅ CMG completed successfully!")
        print(f"Found {num_clusters} clusters")
        print(f"Lambda critical: {lambda_crit:.4f}")
        print(f"Average conductance: {phi_stats.get('avg_phi', 'N/A')}")
        
        return cluster_assignments, num_clusters
        
    except Exception as e:
        print(f"❌ CMG failed: {e}")
        print("📋 Using fallback clustering...")
        
        # Simple fallback clustering for debugging
        cluster_assignments = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])  # Expected clusters
        num_clusters = 3
        
        print(f"Using fallback clusters: {cluster_assignments}")
        return cluster_assignments, num_clusters

def analyze_cmg_clusters(cluster_assignments, num_clusters, labels):
    """Analyze CMG clustering quality."""
    
    print("\n📊 CMG CLUSTERING ANALYSIS") 
    print("="*50)
    
    # Extract clusters
    clusters = []
    for cluster_id in range(num_clusters):
        cluster = [node for node in range(len(cluster_assignments)) if cluster_assignments[node] == cluster_id]
        clusters.append(cluster)
    
    print(f"CMG found {len(clusters)} clusters:")
    
    total_purity = 0
    total_nodes = 0
    
    for i, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue
            
        cluster_labels = [labels[node] for node in cluster]
        label_counts = Counter(cluster_labels)
        majority_count = max(label_counts.values())
        purity = majority_count / len(cluster_labels)
        
        print(f"\n  Cluster {i}: {cluster}")
        print(f"    Size: {len(cluster)}")
        print(f"    Labels: {cluster_labels}")
        print(f"    Label distribution: {dict(label_counts)}")
        print(f"    Purity: {purity:.3f}")
        
        total_purity += purity * len(cluster)
        total_nodes += len(cluster)
    
    overall_purity = total_purity / total_nodes if total_nodes > 0 else 0
    print(f"\n📈 Overall cluster purity: {overall_purity:.3f}")
    
    if overall_purity < 0.8:
        print("⚠️  WARNING: Low cluster purity detected!")
        print("   This could be a major source of accuracy loss!")
    else:
        print("✅ Good cluster purity - clusters align well with classes")
    
    return clusters

def analyze_level1_intra_cluster(G, features, clusters, labels):
    """Analyze Level 1: Intra-cluster GraphSAGE aggregation."""
    
    print("\n🔥 LEVEL 1: INTRA-CLUSTER ANALYSIS")
    print("="*50)
    
    super_features = []
    
    for cluster_id, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue
            
        print(f"\n--- Cluster {cluster_id}: {cluster} ---")
        
        # Original features in this cluster
        cluster_features = features[cluster]
        cluster_labels = [labels[node] for node in cluster]
        
        print(f"Original features shape: {cluster_features.shape}")
        print(f"Cluster labels: {cluster_labels}")
        print(f"Original features:")
        for i, (node, feature) in enumerate(zip(cluster, cluster_features)):
            print(f"  Node {node}: {feature}")
        
        # Compute feature diversity before aggregation
        if len(cluster) > 1:
            feature_variance = np.var(cluster_features, axis=0)
            feature_std = np.std(cluster_features, axis=0)
            print(f"Feature variance: {feature_variance}")
            print(f"Feature std: {feature_std}")
            print(f"Average std: {np.mean(feature_std):.4f}")
        
        # Simple aggregation for debugging (mean)
        if len(cluster_features) > 0:
            super_feature = np.mean(cluster_features, axis=0)
        else:
            super_feature = np.zeros(features.shape[1])
            
        super_features.append(super_feature)
        
        print(f"Super-node feature: {super_feature}")
        
        # Information loss analysis
        if len(cluster) > 1:
            print("🔍 Information Loss Analysis:")
            
            # Can we still distinguish individual nodes?
            for i, node in enumerate(cluster):
                original = cluster_features[i]
                cosine_sim = np.dot(original, super_feature) / (np.linalg.norm(original) * np.linalg.norm(super_feature))
                print(f"  Node {node} similarity to super-feature: {cosine_sim:.3f}")
            
            # Can we still distinguish classes within cluster?
            unique_labels = list(set(cluster_labels))
            if len(unique_labels) > 1:
                print(f"  ⚠️  MIXED CLASSES IN CLUSTER: {unique_labels}")
                print(f"     This will cause information loss!")
            else:
                print(f"  ✅ Pure cluster: all nodes have label {unique_labels[0]}")
        
        print()
    
    super_features = np.array(super_features)
    print(f"📊 Level 1 Summary:")
    print(f"   Input: {features.shape[0]} nodes × {features.shape[1]} features")
    print(f"   Output: {len(clusters)} super-nodes × {super_features.shape[1]} features")
    print(f"   Compression ratio: {features.shape[0] / len(clusters):.2f}x")
    
    return super_features

def analyze_level2_inter_cluster(clusters, super_features, labels):
    """Analyze Level 2: Inter-cluster GraphSAGE."""
    
    print("\n🚀 LEVEL 2: INTER-CLUSTER ANALYSIS")
    print("="*50)
    
    # Build coarsened graph (simplified)
    num_clusters = len(clusters)
    coarsened_G = nx.Graph()
    coarsened_G.add_nodes_from(range(num_clusters))
    
    # Add edges between clusters (simplified logic)
    # In practice, this comes from CMG coarsening
    for i in range(num_clusters - 1):
        coarsened_G.add_edge(i, i + 1)
    
    print(f"Coarsened graph: {coarsened_G.number_of_nodes()} nodes, {coarsened_G.number_of_edges()} edges")
    print(f"Super-features shape: {super_features.shape}")
    
    # Show super-node class assignments
    super_labels = []
    for cluster in clusters:
        if len(cluster) > 0:
            cluster_labels = [labels[node] for node in cluster]
            majority_label = Counter(cluster_labels).most_common(1)[0][0]
            super_labels.append(majority_label)
        else:
            super_labels.append(-1)
    
    print(f"Super-node labels: {super_labels}")
    
    # Check if super-features are distinguishable
    print("\n🔍 Super-feature Distinguishability:")
    for i in range(len(super_features)):
        for j in range(i + 1, len(super_features)):
            cosine_sim = np.dot(super_features[i], super_features[j]) / (
                np.linalg.norm(super_features[i]) * np.linalg.norm(super_features[j])
            )
            print(f"  Super-node {i} vs {j}: similarity = {cosine_sim:.3f}, labels = {super_labels[i]} vs {super_labels[j]}")
    
    # Simplified GraphSAGE (just return mean of features for debugging)
    super_embeddings = super_features.copy()  # Simplified for debugging
    
    print(f"📊 Level 2 Summary:")
    print(f"   Input: {len(clusters)} super-nodes × {super_features.shape[1]} features")
    print(f"   Output: {len(clusters)} super-nodes × {super_embeddings.shape[1]} embeddings")
    
    return super_embeddings, super_labels, coarsened_G

def analyze_level3_refinement(clusters, super_embeddings, labels):
    """Analyze Level 3: Refinement back to individual nodes."""
    
    print("\n✨ LEVEL 3: REFINEMENT ANALYSIS")
    print("="*50)
    
    # Simple refinement: assign super-embedding to all nodes in cluster
    num_nodes = sum(len(cluster) for cluster in clusters)
    final_embeddings = np.zeros((num_nodes, super_embeddings.shape[1]))
    
    print("Refinement mapping:")
    for cluster_id, cluster in enumerate(clusters):
        for node in cluster:
            final_embeddings[node] = super_embeddings[cluster_id]
            print(f"  Node {node}: gets super-embedding from cluster {cluster_id}")
    
    print(f"\n📊 Level 3 Summary:")
    print(f"   Input: {len(clusters)} super-embeddings")
    print(f"   Output: {num_nodes} node embeddings")
    print(f"   Expansion ratio: {num_nodes / len(clusters):.2f}x")
    
    # Check if nodes in same cluster get identical embeddings
    print("\n🔍 Embedding Uniqueness Check:")
    unique_embeddings = []
    for i, embedding in enumerate(final_embeddings):
        is_unique = True
        for unique_emb in unique_embeddings:
            if np.allclose(embedding, unique_emb):
                is_unique = False
                break
        if is_unique:
            unique_embeddings.append(embedding)
    
    print(f"   Total embeddings: {len(final_embeddings)}")
    print(f"   Unique embeddings: {len(unique_embeddings)}")
    print(f"   Embedding diversity: {len(unique_embeddings) / len(final_embeddings):.3f}")
    
    if len(unique_embeddings) < len(final_embeddings):
        print("⚠️  WARNING: Multiple nodes have identical embeddings!")
        print("   This reduces the model's ability to distinguish individual nodes!")
    
    return final_embeddings

def classify_with_simple_classifier(embeddings, labels):
    """Simple k-NN classification for evaluation."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # k-NN classification
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def compare_with_baseline(features, labels, final_embeddings):
    """Compare with simple baselines."""
    
    print("\n📊 BASELINE COMPARISON")
    print("="*50)
    
    # Baseline 1: Original features
    acc_original = classify_with_simple_classifier(features, labels)
    
    # Baseline 2: Final embeddings
    acc_embeddings = classify_with_simple_classifier(final_embeddings, labels)
    
    print(f"Original features accuracy: {acc_original:.3f}")
    print(f"Final embeddings accuracy: {acc_embeddings:.3f}")
    print(f"Accuracy change: {acc_embeddings - acc_original:.3f}")
    
    if acc_embeddings < acc_original:
        print("❌ True Coarsened GraphSAGE decreased accuracy!")
        print("   This confirms information loss through the pipeline!")
    else:
        print("✅ True Coarsened GraphSAGE maintained/improved accuracy!")
    
    return acc_original, acc_embeddings

def create_projection_matrices(clusters, num_nodes):
    """Create projection matrices for GraphZoom refinement."""
    
    print("\n🔧 CREATING PROJECTION MATRICES")
    print("="*50)
    
    num_clusters = len(clusters)
    
    # Create projection matrix: rows = nodes, cols = clusters
    projection = sp.lil_matrix((num_nodes, num_clusters))
    
    for node in range(num_nodes):
        for cluster_id, cluster in enumerate(clusters):
            if node in cluster:
                projection[node, cluster_id] = 1.0
                break
    
    projections = [projection.tocsr()]
    
    print(f"Created projection matrix: {projection.shape}")
    print(f"Projection nnz: {projection.nnz}")
    
    return projections

def test_fixed_refinement(G, features, labels, clusters, super_embeddings, laplacians):
    """Test the FIXED True Coarsened GraphSAGE with proper refinement."""
    
    print("\n🔧 TESTING FIXED TRUE COARSENED GRAPHSAGE")
    print("="*60)
    
    try:
        # Try to import the fixed version
        from embed_methods.graphsage.true_coarsened_graphsage import true_coarsened_graphsage
        
        print("✅ Successfully imported fixed True Coarsened GraphSAGE")
        
        # Create coarsened graph (simplified for testing)
        num_clusters = len(clusters)
        coarsened_G = nx.Graph()
        coarsened_G.add_nodes_from(range(num_clusters))
        
        # Add edges between adjacent clusters
        for i in range(num_clusters - 1):
            coarsened_G.add_edge(i, i + 1)
        
        print(f"Test coarsened graph: {coarsened_G.number_of_nodes()} nodes, {coarsened_G.number_of_edges()} edges")
        
        # Create projection matrices for GraphZoom refinement
        projections = create_projection_matrices(clusters, G.number_of_nodes())
        
        print(f"Using {len(projections)} projection matrices")
        print(f"Projection 0 shape: {projections[0].shape}")
        
        # Run FIXED True Coarsened GraphSAGE
        print(f"\n🚀 Running FIXED True Coarsened GraphSAGE...")
        fixed_start = time.time()
        
        fixed_embeddings = true_coarsened_graphsage(
            original_graph=G,
            features=features,
            clusters=clusters,
            coarsened_graph=coarsened_G,
            projections=projections,
            laplacians=laplacians,
            super_embed_dim=features.shape[1],  # Match feature dimension
            final_embed_dim=features.shape[1],   # Match feature dimension
            hidden_dim=16,                       # Small for testing
            training_epochs=50                   # Fast for testing
        )
        
        fixed_time = time.time() - fixed_start
        
        print(f"✅ FIXED version completed in {fixed_time:.3f}s")
        print(f"Fixed embeddings shape: {fixed_embeddings.shape}")
        
        # Evaluate accuracy
        fixed_accuracy = classify_with_simple_classifier(fixed_embeddings, labels)
        
        print(f"\n📊 FIXED REFINEMENT RESULTS:")
        print(f"Fixed embeddings accuracy: {fixed_accuracy:.3f}")
        
        # Check embedding uniqueness
        unique_embeddings = []
        for i, embedding in enumerate(fixed_embeddings):
            is_unique = True
            for unique_emb in unique_embeddings:
                if np.allclose(embedding, unique_emb, atol=1e-6):
                    is_unique = False
                    break
            if is_unique:
                unique_embeddings.append(embedding)
        
        print(f"Embedding uniqueness: {len(unique_embeddings)}/{len(fixed_embeddings)} unique")
        print(f"Embedding diversity: {len(unique_embeddings) / len(fixed_embeddings):.3f}")
        
        if len(unique_embeddings) == len(fixed_embeddings):
            print("✅ SUCCESS: All embeddings are unique!")
        else:
            print(f"⚠️  Still have {len(fixed_embeddings) - len(unique_embeddings)} duplicate embeddings")
        
        return fixed_accuracy, fixed_embeddings
        
    except ImportError as e:
        print(f"❌ Could not import fixed True Coarsened GraphSAGE: {e}")
        print("📋 Make sure true_coarsened_graphsage.py is in the current directory")
        return None, None
    
    except Exception as e:
        print(f"❌ FIXED version failed: {e}")
        print("📋 Error details:")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main debugging pipeline."""
    
    print("🔬 SYNTHETIC GRAPH DEBUGGING FOR TRUE COARSENED GRAPHSAGE")
    print("="*70)
    
    # Step 1: Create synthetic graph and features
    G, edges = create_synthetic_graph()
    features, labels = create_synthetic_features()
    
    # Step 2: Convert to format for CMG
    laplacian = networkx_to_laplacian(G)
    laplacians = [laplacian]  # For refinement
    
    # Step 3: Run CMG clustering
    cluster_assignments, num_clusters = run_cmg_clustering(laplacian)
    
    # Step 4: Analyze clustering quality
    clusters = analyze_cmg_clusters(cluster_assignments, num_clusters, labels)
    
    # Step 5: Analyze each level of the pipeline
    super_features = analyze_level1_intra_cluster(G, features, clusters, labels)
    super_embeddings, super_labels, coarsened_G = analyze_level2_inter_cluster(clusters, super_features, labels)
    final_embeddings = analyze_level3_refinement(clusters, super_embeddings, labels)
    
    # Step 6: Compare with baseline
    acc_original, acc_embeddings = compare_with_baseline(features, labels, final_embeddings)
    
    # Step 7: Test FIXED refinement
    fixed_accuracy, fixed_embeddings = test_fixed_refinement(
        G, features, labels, clusters, super_embeddings, laplacians
    )
    
    # Step 8: Final comparison
    print("\n🎯 FINAL DIAGNOSIS & COMPARISON")
    print("="*60)
    
    print("Key findings:")
    print(f"  1. CMG cluster purity: Check output above")
    print(f"  2. Information compression: {features.shape[0]} → {len(clusters)} → {features.shape[0]}")
    print(f"  3. Embedding uniqueness: Check Level 3 output above")
    
    print(f"\n📊 ACCURACY COMPARISON:")
    print(f"  Original features:     {acc_original:.3f}")
    print(f"  Broken refinement:     {acc_embeddings:.3f} (loss: {acc_embeddings - acc_original:.3f})")
    
    if fixed_accuracy is not None:
        print(f"  Fixed refinement:      {fixed_accuracy:.3f} (vs broken: +{fixed_accuracy - acc_embeddings:.3f})")
        
        if fixed_accuracy > acc_embeddings:
            print("✅ FIXED refinement improved accuracy!")
            if fixed_accuracy >= acc_original * 0.9:  # Within 90% of original
                print("🎉 REFINEMENT FIX SUCCESSFUL!")
            else:
                print("⚠️  Still some room for improvement")
        else:
            print("❌ Fixed refinement did not improve - need more investigation")
    else:
        print("  Fixed refinement:      FAILED TO RUN")
    
    print("\nNext steps:")
    if fixed_accuracy is not None and fixed_accuracy > acc_embeddings:
        print("  ✅ The fix works! Now test on real Cora dataset")
        print("  🚀 Run: python graphzoom_timed_mpaware.py --coarse cmg --embed_method true_coarsened_graphsage --dataset cora")
    else:
        print("  - Debug why the fixed refinement didn't work as expected")
        print("  - Check projection matrices and GraphZoom refinement import")
        print("  - Verify the spectral smoothing is working correctly")
    
    return {
        'graph': G,
        'features': features,
        'labels': labels,
        'clusters': clusters,
        'super_features': super_features,
        'super_embeddings': super_embeddings,
        'final_embeddings': final_embeddings,
        'fixed_embeddings': fixed_embeddings,
        'accuracy_original': acc_original,
        'accuracy_embeddings': acc_embeddings,
        'accuracy_fixed': fixed_accuracy
    }

if __name__ == "__main__":
    results = main()