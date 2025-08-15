#!/usr/bin/env python3
"""
True Coarsened GraphSAGE - ENHANCED WITH FEATURE-AWARE REFINEMENT

Key Enhancement: Combines spectral refinement with original features
for better individual node representation.

Expected improvement: +1-3% accuracy (71.2% → 72-74%)
"""

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import time
import sys
import os

def extract_cluster_subgraph(G: nx.Graph, cluster_nodes: List[int]) -> nx.Graph:
    """Extract subgraph for a cluster, preserving internal structure."""
    return G.subgraph(cluster_nodes).copy()

def simple_graphsage_aggregate(graph: nx.Graph, features: np.ndarray, embed_dim: int = 64) -> np.ndarray:
    """
    Simplified but robust GraphSAGE aggregation for small clusters.
    """
    n_nodes = len(graph.nodes())
    
    if n_nodes == 0:
        return np.zeros(embed_dim)
    
    if n_nodes == 1:
        # Single node - just transform dimension
        feature = features[0] if len(features) > 0 else np.zeros(embed_dim)
        result = np.zeros(embed_dim)
        result[:min(len(feature), embed_dim)] = feature[:min(len(feature), embed_dim)]
        return result
    
    # Multi-node cluster: GraphSAGE-style aggregation
    try:
        # Ensure features have consistent dimension
        input_dim = features.shape[1] if len(features.shape) > 1 else len(features[0])
        normalized_features = np.zeros((n_nodes, input_dim))
        
        node_list = list(graph.nodes())
        for i, feature in enumerate(features):
            if i < n_nodes:
                if isinstance(feature, np.ndarray):
                    normalized_features[i][:min(len(feature), input_dim)] = feature[:min(len(feature), input_dim)]
                else:
                    normalized_features[i][0] = feature
        
        # Build adjacency list
        adj_list = {}
        for i, node in enumerate(node_list):
            neighbors = [node_list.index(n) for n in graph.neighbors(node) if n in node_list]
            adj_list[i] = neighbors
        
        # 2-layer GraphSAGE aggregation
        h = normalized_features.copy()
        
        for layer in range(2):
            new_h = np.zeros_like(h)
            
            for node_idx in range(n_nodes):
                neighbors = adj_list.get(node_idx, [])
                
                if len(neighbors) > 0:
                    # Aggregate neighbors (mean)
                    neighbor_features = h[neighbors]
                    aggregated = np.mean(neighbor_features, axis=0)
                    
                    # Combine self + neighbors (mean combination)
                    combined = (h[node_idx] + aggregated) / 2.0
                    new_h[node_idx] = combined
                else:
                    # Isolated node
                    new_h[node_idx] = h[node_idx]
            
            h = new_h
        
        # Final pooling to create single super-node feature
        result_feature = np.mean(h, axis=0)
        
        # Transform to target dimension
        final_result = np.zeros(embed_dim)
        final_result[:min(len(result_feature), embed_dim)] = result_feature[:min(len(result_feature), embed_dim)]
        
        return final_result
        
    except Exception as e:
        print(f"[WARNING] GraphSAGE aggregation failed: {e}, using mean fallback")
        # Robust fallback
        mean_feature = np.mean(features, axis=0) if len(features) > 0 else np.zeros(embed_dim)
        result = np.zeros(embed_dim)
        result[:min(len(mean_feature), embed_dim)] = mean_feature[:min(len(mean_feature), embed_dim)]
        return result

def create_super_node_features(original_graph: nx.Graph,
                             features: np.ndarray,
                             clusters: List[List[int]],
                             embed_dim: int = 64) -> np.ndarray:
    """Create super-node features by applying GraphSAGE within each cluster."""
    print(f"[SUPER-NODE] Creating super-node features for {len(clusters)} clusters")
    start_time = time.time()
    
    super_features = []
    
    for cluster_id, cluster_nodes in enumerate(clusters):
        if len(cluster_nodes) == 0:
            super_features.append(np.zeros(embed_dim))
            continue
            
        # Extract cluster subgraph
        cluster_subgraph = extract_cluster_subgraph(original_graph, cluster_nodes)
        
        # Extract cluster features
        try:
            cluster_features = features[cluster_nodes]
        except (IndexError, TypeError):
            # Handle edge cases
            cluster_features = np.array([features[i] for i in cluster_nodes if i < len(features)])
            if len(cluster_features) == 0:
                super_features.append(np.zeros(embed_dim))
                continue
        
        # Apply GraphSAGE aggregation within cluster
        super_feature = simple_graphsage_aggregate(cluster_subgraph, cluster_features, embed_dim)
        super_features.append(super_feature)
        
        if cluster_id % 10 == 0 and cluster_id > 0:
            print(f"  Processed {cluster_id}/{len(clusters)} clusters")
    
    super_features = np.array(super_features)
    elapsed = time.time() - start_time
    
    print(f"[SUPER-NODE] Created super-node features: {super_features.shape} in {elapsed:.3f}s")
    print(f"[SUPER-NODE] Feature stats: mean={np.mean(super_features):.4f}, std={np.std(super_features):.4f}")
    
    return super_features

def simplified_graphsage_on_coarsened(coarsened_graph: nx.Graph, 
                                    super_features: np.ndarray,
                                    final_embed_dim: int = 64,
                                    num_layers: int = 2) -> np.ndarray:
    """Simplified GraphSAGE for the coarsened graph."""
    n_super_nodes = coarsened_graph.number_of_nodes()
    
    if n_super_nodes == 0:
        return np.zeros((0, final_embed_dim))
    
    if n_super_nodes == 1:
        # Single super-node
        result = np.zeros((1, final_embed_dim))
        if len(super_features) > 0:
            feature = super_features[0]
            result[0][:min(len(feature), final_embed_dim)] = feature[:min(len(feature), final_embed_dim)]
        return result
    
    print(f"[SIMPLIFIED-SAGE] Processing {n_super_nodes} super-nodes")
    
    # Build adjacency list
    node_list = sorted(list(coarsened_graph.nodes()))
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    adj_list = {}
    for i, node in enumerate(node_list):
        neighbors = []
        for neighbor in coarsened_graph.neighbors(node):
            if neighbor in node_to_idx:
                neighbors.append(node_to_idx[neighbor])
        adj_list[i] = neighbors
    
    # Ensure super_features has correct shape
    if len(super_features.shape) == 1:
        super_features = super_features.reshape(1, -1)
    
    input_dim = super_features.shape[1]
    h = super_features.copy()
    
    # Multi-layer aggregation
    for layer in range(num_layers):
        new_h = np.zeros((n_super_nodes, input_dim))
        
        for node_idx in range(n_super_nodes):
            neighbors = adj_list.get(node_idx, [])
            
            if len(neighbors) > 0:
                # Aggregate neighbors
                neighbor_features = h[neighbors]
                aggregated = np.mean(neighbor_features, axis=0)
                
                # GraphSAGE update: combine self + aggregated neighbors
                combined = (h[node_idx] + aggregated) / 2.0
                new_h[node_idx] = combined
            else:
                # Isolated super-node
                new_h[node_idx] = h[node_idx]
        
        h = new_h
    
    # Transform to final embedding dimension
    final_embeddings = np.zeros((n_super_nodes, final_embed_dim))
    for i in range(n_super_nodes):
        final_embeddings[i][:min(len(h[i]), final_embed_dim)] = h[i][:min(len(h[i]), final_embed_dim)]
    
    return final_embeddings

def apply_inter_cluster_graphsage(coarsened_graph: nx.Graph,
                                super_features: np.ndarray,
                                final_embed_dim: int = 64,
                                hidden_dim: int = 128,
                                num_layers: int = 2,
                                training_epochs: int = 200) -> np.ndarray:
    """Apply GraphSAGE on the coarsened graph with super-node features."""
    print(f"[INTER-CLUSTER] Applying GraphSAGE on coarsened graph")
    print(f"  Coarsened graph: {coarsened_graph.number_of_nodes()} nodes, {coarsened_graph.number_of_edges()} edges")
    print(f"  Super-features shape: {super_features.shape}")
    
    start_time = time.time()
    
    # Try to use existing GraphSAGE implementation
    try:
        # Add required paths for GraphSAGE import
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        # Try to import existing GraphSAGE
        from embed_methods.graphsage.graphsage import graphsage
        
        print("[INTER-CLUSTER] Using EXISTING GraphSAGE implementation")
        
        # Set required attributes for GraphSAGE
        nx.set_node_attributes(coarsened_graph, False, "test")
        nx.set_node_attributes(coarsened_graph, False, "val")
        
        # Calculate appropriate training iterations
        coarse_ratio = max(1.0, super_features.shape[0] / max(1, coarsened_graph.number_of_nodes()))
        training_iterations = max(50, int(training_epochs / coarse_ratio))
        
        print(f"[INTER-CLUSTER] Training iterations: {training_iterations}")
        
        # Call existing GraphSAGE
        super_embeddings = graphsage(
            coarsened_graph,
            super_features,
            "mean",  # sage_model
            True,    # sage_weighted
            training_iterations
        )
        
        print("[INTER-CLUSTER] ✅ Existing GraphSAGE completed successfully!")
        
    except Exception as e:
        print(f"[INTER-CLUSTER] ⚠️  Existing GraphSAGE failed: {e}")
        print("[INTER-CLUSTER] Using simplified GraphSAGE implementation")
        
        # Use simplified implementation
        super_embeddings = simplified_graphsage_on_coarsened(
            coarsened_graph, super_features, final_embed_dim, num_layers
        )
    
    elapsed = time.time() - start_time
    print(f"[INTER-CLUSTER] Completed in {elapsed:.3f}s, output shape: {super_embeddings.shape}")
    
    return super_embeddings

def feature_aware_spectral_refinement(super_embeddings: np.ndarray,
                                    projections: List,
                                    laplacians: List,
                                    original_features: np.ndarray,
                                    clusters: List[List[int]],
                                    alpha: float = 0.7,
                                    beta: float = 0.3,
                                    lda: float = 0.1,
                                    power: bool = False) -> np.ndarray:
    """
    ENHANCED: Feature-Aware Spectral Refinement
    
    Combines GraphZoom's spectral refinement with original node features
    for better individual node representation.
    
    Args:
        alpha: Weight for spectral refinement (0.7 = 70% spectral)
        beta: Weight for original features (0.3 = 30% features)
    """
    print("\n🔧 LEVEL 3: FEATURE-AWARE SPECTRAL REFINEMENT (ENHANCED)")
    print("="*70)
    
    print(f"🚨 EXTREME DEBUG: feature_aware_spectral_refinement called!")
    print(f"🚨 α={alpha}, β={beta}")
    print(f"🚨 If you see this with different α,β but same accuracy, there's a bug!")

    try:
        # Try to import GraphZoom utilities
        try:
            from utils import smooth_filter
        except ImportError:
            # Try alternative import paths
            sys.path.append('..')
            sys.path.append('../..')
            from utils import smooth_filter
        
        print(f"[FEATURE-REFINE] Input super-embeddings shape: {super_embeddings.shape}")
        print(f"[FEATURE-REFINE] Original features shape: {original_features.shape}")
        print(f"[FEATURE-REFINE] Combination weights: α={alpha} (spectral), β={beta} (features)")
        print(f"[FEATURE-REFINE] Number of projection levels: {len(projections)}")
        
        # STEP 1: Standard GraphZoom spectral refinement
        embeddings = super_embeddings.copy()
        
        for i in reversed(range(len(projections))):
            print(f"[FEATURE-REFINE] Processing level {i}")
            print(f"  Projection shape: {projections[i].shape}")
            print(f"  Embeddings shape before: {embeddings.shape}")
            
            # Project embeddings to next level
            embeddings = projections[i] @ embeddings
            print(f"  Embeddings shape after projection: {embeddings.shape}")
            
            # Apply spectral smoothing filter
            filter_ = smooth_filter(laplacians[i], lda)
            
            # Apply smoothing (GraphZoom's approach)
            if power or i == 0:
                embeddings = filter_ @ (filter_ @ embeddings)
                print(f"  Applied double smoothing (power={power}, level={i})")
            else:
                print(f"  Skipped smoothing (power={power}, level={i})")
        
        print(f"[FEATURE-REFINE] ✅ Spectral refinement completed: {embeddings.shape}")
        
        # STEP 2: ENHANCEMENT - Combine with original features
        print(f"[FEATURE-REFINE] 🌟 ENHANCING with original features...")
        
        embed_dim = embeddings.shape[1]
        feature_dim = original_features.shape[1]
        
        # Handle dimension mismatch
        if feature_dim != embed_dim:
            print(f"[FEATURE-REFINE] Dimension mismatch: embeddings {embed_dim}D, features {feature_dim}D")
            
            if feature_dim > embed_dim:
                # Truncate features to match embedding dimension
                aligned_features = original_features[:, :embed_dim]
                print(f"[FEATURE-REFINE] Truncated features to {embed_dim}D")
            else:
                # Pad features to match embedding dimension
                aligned_features = np.zeros((original_features.shape[0], embed_dim))
                aligned_features[:, :feature_dim] = original_features
                print(f"[FEATURE-REFINE] Padded features to {embed_dim}D")
        else:
            aligned_features = original_features
            print(f"[FEATURE-REFINE] Features already aligned: {embed_dim}D")
        
        # STEP 3: Feature-aware combination
        print(f"[FEATURE-REFINE] Combining spectral + features with weights α={alpha}, β={beta}")
        
        # Normalize embeddings and features for stable combination
        embedding_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embedding_norm[embedding_norm == 0] = 1  # Avoid division by zero
        normalized_embeddings = embeddings / embedding_norm
        
        feature_norm = np.linalg.norm(aligned_features, axis=1, keepdims=True)
        feature_norm[feature_norm == 0] = 1  # Avoid division by zero
        normalized_features = aligned_features / feature_norm
        
        # Weighted combination
        # After the combination line:
        enhanced_embeddings = alpha * normalized_embeddings + beta * normalized_features

        # Add these debug prints:
        print(f"🔧 DEBUG: Combined embeddings stats:")
        print(f"   Spectral component mean: {np.mean(alpha * normalized_embeddings):.6f}")
        print(f"   Feature component mean: {np.mean(beta * normalized_features):.6f}")
        print(f"   Combined mean: {np.mean(enhanced_embeddings):.6f}")
        print(f"   Should be different for different α,β!")

        # Check if they're actually different
        if alpha == 1 and beta == 0:
            print("🔧 DEBUG: This is PURE SPECTRAL - should be very different from pure features!")
        elif alpha == 0 and beta == 1:
            print("🔧 DEBUG: This is PURE FEATURES - should be very different from pure spectral!")
    
        
        print(f"[FEATURE-REFINE] ✅ Enhanced embeddings shape: {enhanced_embeddings.shape}")
        
        # STEP 4: Optional cluster-aware refinement
        print(f"[FEATURE-REFINE] 🎯 Applying cluster-aware adjustments...")
        
        cluster_map = {}
        for cluster_id, cluster_nodes in enumerate(clusters):
            for node in cluster_nodes:
                cluster_map[node] = cluster_id
        
        # Slight adjustment based on cluster membership
        for node in range(enhanced_embeddings.shape[0]):
            if node in cluster_map:
                cluster_id = cluster_map[node]
                # Small boost to preserve cluster coherence
                cluster_weight = 0.05  # 5% cluster influence
                if cluster_id < super_embeddings.shape[0]:
                    super_emb_normalized = super_embeddings[cluster_id] / (np.linalg.norm(super_embeddings[cluster_id]) + 1e-8)
                    if len(super_emb_normalized) == enhanced_embeddings.shape[1]:
                        enhanced_embeddings[node] = (
                            (1 - cluster_weight) * enhanced_embeddings[node] + 
                            cluster_weight * super_emb_normalized
                        )
        
        print(f"[FEATURE-REFINE] ✅ Feature-aware refinement completed!")
        print(f"[FEATURE-REFINE] Final embeddings shape: {enhanced_embeddings.shape}")
        print(f"[FEATURE-REFINE] Enhancement stats:")
        print(f"  Mean: {np.mean(enhanced_embeddings):.6f}")
        print(f"  Std:  {np.std(enhanced_embeddings):.6f}")
        
        # Just before return:
        print(f"🔧 DEBUG: Returning enhanced_embeddings with shape {enhanced_embeddings.shape}")
        print(f"🔧 DEBUG: Final stats: mean={np.mean(enhanced_embeddings):.6f}, std={np.std(enhanced_embeddings):.6f}")
        return enhanced_embeddings
        
    except Exception as e:
        print(f"[FEATURE-REFINE] ❌ Enhanced refinement failed: {e}")
        print(f"[FEATURE-REFINE] 📋 Error details:")
        import traceback
        traceback.print_exc()
        
        # Fallback to original spectral refinement
        print(f"[FEATURE-REFINE] Using fallback: standard spectral refinement")
        
        # Standard spectral refinement as fallback
        try:
            from utils import smooth_filter
            embeddings = super_embeddings.copy()
            
            for i in reversed(range(len(projections))):
                embeddings = projections[i] @ embeddings
                filter_ = smooth_filter(laplacians[i], lda)
                if power or i == 0:
                    embeddings = filter_ @ (filter_ @ embeddings)
            
            return embeddings
        except:
            # Ultimate fallback
            return super_embeddings

def true_coarsened_graphsage(original_graph: nx.Graph,
                           features: np.ndarray,
                           clusters: List[List[int]],
                           coarsened_graph: nx.Graph,
                           projections: List,
                           laplacians: List,
                           super_embed_dim: int = 64,
                           final_embed_dim: int = 64,
                           hidden_dim: int = 128,
                           training_epochs: int = 200,
                           refinement_alpha: float = 1.0,
                           refinement_beta: float = 0.0) -> np.ndarray:
    """
    ENHANCED VERSION: True 3-level GraphSAGE with Feature-Aware Refinement!
    
    Level 1: Intra-cluster GraphSAGE → super-node features
    Level 2: Inter-cluster GraphSAGE → super-node embeddings
    Level 3: Feature-aware spectral refinement → individualized node embeddings (ENHANCED!)
    
    New Parameters:
        refinement_alpha: Weight for spectral refinement (default: 0.7)
        refinement_beta: Weight for original features (default: 0.3)
    """
    print("="*70)
    print("TRUE COARSENED GRAPHSAGE - FEATURE-AWARE REFINEMENT VERSION")
    print("="*70)
    
    total_start = time.time()
    
    # LEVEL 1: Intra-cluster GraphSAGE aggregation
    print("\n🔥 LEVEL 1: Intra-cluster GraphSAGE aggregation")
    super_features = create_super_node_features(
        original_graph, features, clusters, super_embed_dim
    )
    
    # LEVEL 2: Inter-cluster GraphSAGE (on coarsened graph)
    print("\n🚀 LEVEL 2: Inter-cluster GraphSAGE")
    super_embeddings = apply_inter_cluster_graphsage(
        coarsened_graph, super_features, final_embed_dim, hidden_dim, 
        num_layers=2, training_epochs=training_epochs
    )
    
    # LEVEL 3: Feature-Aware Spectral Refinement (ENHANCED!)
    print("\n✨ LEVEL 3: Feature-Aware Spectral Refinement (ENHANCED)")
    refine_start = time.time()
    
    final_embeddings = feature_aware_spectral_refinement(
        super_embeddings=super_embeddings,
        projections=projections,
        laplacians=laplacians,
        original_features=features,  # NEW: Pass original features
        clusters=clusters,           # NEW: Pass cluster information
        alpha=refinement_alpha,      # NEW: Spectral weight
        beta=refinement_beta,        # NEW: Feature weight
        lda=0.1,
        power=False
    )
    
    refine_time = time.time() - refine_start
    total_time = time.time() - total_start
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"  Original graph: {original_graph.number_of_nodes()} nodes → {coarsened_graph.number_of_nodes()} super-nodes")
    print(f"  Coarsening ratio: {original_graph.number_of_nodes() / max(1, coarsened_graph.number_of_nodes()):.2f}x")
    print(f"  Refinement time: {refine_time:.3f}s")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Final embeddings shape: {final_embeddings.shape}")
    print(f"  Refinement weights: α={refinement_alpha} (spectral), β={refinement_beta} (features)")
    
    # CRITICAL: Validate that we now have unique embeddings per node
    print(f"\n🔍 EMBEDDING UNIQUENESS VALIDATION:")
    unique_embeddings = []
    for i, embedding in enumerate(final_embeddings):
        is_unique = True
        for unique_emb in unique_embeddings:
            if np.allclose(embedding, unique_emb, atol=1e-6):
                is_unique = False
                break
        if is_unique:
            unique_embeddings.append(embedding)
    
    print(f"  Total embeddings: {len(final_embeddings)}")
    print(f"  Unique embeddings: {len(unique_embeddings)}")
    print(f"  Embedding diversity: {len(unique_embeddings) / len(final_embeddings):.3f}")
    
    if len(unique_embeddings) == len(final_embeddings):
        print("  ✅ SUCCESS: All embeddings are unique!")
    else:
        print(f"  ⚠️  Still have {len(final_embeddings) - len(unique_embeddings)} duplicate embeddings")
    
    print(f"\n🌟 EXPECTED IMPROVEMENT:")
    print(f"  Previous accuracy: ~71.2%")
    print(f"  Enhanced refinement target: 72-74%")
    print(f"  Feature-aware combination should preserve more individual node information!")
    
    # Just before the final return:
    print(f"🔧 DEBUG: true_coarsened_graphsage returning embeddings:")
    print(f"   Shape: {final_embeddings.shape}")
    print(f"   Mean: {np.mean(final_embeddings):.6f}")
    print(f"   This should be different for different α,β values!")
    return final_embeddings

def test_enhanced_true_coarsened_graphsage():
    """Test the ENHANCED true coarsened GraphSAGE on a simple example."""
    print("Testing ENHANCED True Coarsened GraphSAGE...")
    
    # Create test graph
    G = nx.path_graph(12)
    features = np.random.randn(12, 16)
    clusters = [[0,1,2], [3,4,5,6], [7,8,9], [10,11]]
    
    # Create simple coarsened graph
    coarsened_G = nx.Graph()
    coarsened_G.add_nodes_from(range(len(clusters)))
    coarsened_G.add_edges_from([(0,1), (1,2), (2,3)])  # Connect adjacent clusters
    
    # Create dummy projections (simplified)
    import scipy.sparse as sp
    projections = []
    num_nodes = 12
    num_clusters = len(clusters)
    
    projection = sp.lil_matrix((num_nodes, num_clusters))
    for node_id in range(num_nodes):
        for cluster_id, cluster in enumerate(clusters):
            if node_id in cluster:
                projection[node_id, cluster_id] = 1.0
    
    projections.append(projection.tocsr())
    laplacians = [nx.laplacian_matrix(G)]
    
    # Run ENHANCED true coarsened GraphSAGE
    embeddings = true_coarsened_graphsage(
        G, features, clusters, coarsened_G, projections, laplacians,
        training_epochs=50,  # Faster for testing
        refinement_alpha=0.0,  # 70% spectral
        refinement_beta=1.0    # 30% features
    )
    
    print(f"✅ Success! Output embeddings shape: {embeddings.shape}")
    print("ENHANCED True Coarsened GraphSAGE test completed!")

if __name__ == "__main__":
    test_enhanced_true_coarsened_graphsage()