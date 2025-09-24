#!/usr/bin/env python3
"""
True Coarsened GraphSAGE - THREE-WAY SPECTRAL-AWARE REFINEMENT

Revolutionary Enhancement: Combines hierarchical structure + individual features + spectral context
using eigenspace-aware spectral tokens per supernode.

Expected improvement: 72.7% → 74-76% (targeting LAMG's 78.2%)

Components:
α: Hierarchical structure (spectral refinement)
β: Individual characteristics (original features)  
γ: Spectral structure (eigenspace context)
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
import scipy.sparse as sp
from scipy.sparse import diags
from scipy.linalg import eigh

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

def compute_cluster_spectral_context(cluster_subgraph: nx.Graph, target_dim: int = 64) -> np.ndarray:
    """
    Build GraFT-style spectral context for a cluster subgraph.
    
    Returns basis-invariant spectral features that capture eigenspace structure.
    """
    try:
        print(f"[SPECTRAL] Computing spectral context for {cluster_subgraph.number_of_nodes()} node cluster")
        
        if cluster_subgraph.number_of_nodes() <= 1:
            # Single node or empty - return zero context
            return np.zeros(target_dim)
        
        # Get Laplacian matrix
        laplacian = nx.laplacian_matrix(cluster_subgraph).astype(float)
        
        if laplacian.shape[0] <= 1:
            return np.zeros(target_dim)
        
        # Eigenvalue decomposition
        try:
            eigenvals, eigenvecs = eigh(laplacian.toarray())
        except:
            # Fallback for numerical issues
            return np.zeros(target_dim)
        
        # Sort eigenvalues/vectors
        idx = np.argsort(eigenvals)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Group eigenvalues (handle near-degeneracies)
        eigenspace_groups = group_eigenvalues(eigenvals, threshold=0.01)
        
        # Build spectral tokens for each eigenspace
        spectral_features = []
        
        for group_start, group_end in eigenspace_groups:
            group_eigenvals = eigenvals[group_start:group_end+1]
            group_eigenvecs = eigenvecs[:, group_start:group_end+1]
            
            # Basic spectral scalars
            spectral_features.extend([
                np.mean(group_eigenvals),      # Mean eigenvalue
                np.std(group_eigenvals),       # Std eigenvalue  
                len(group_eigenvals),          # Multiplicity
            ])
            
            # Projector diagonal statistics (basis-invariant)
            if group_eigenvecs.shape[1] > 0:
                projector_diag = np.sum(group_eigenvecs ** 2, axis=1)  # diag(V V^T)
                spectral_features.extend([
                    np.mean(projector_diag),   # Mean projection strength
                    np.std(projector_diag),    # Std projection strength
                    np.max(projector_diag),    # Max projection strength
                    np.percentile(projector_diag, 25),  # 25th percentile
                    np.percentile(projector_diag, 75),  # 75th percentile
                ])
            else:
                spectral_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Convert to fixed-size vector
        spectral_vector = np.array(spectral_features)
        
        # Resize to target dimension
        if len(spectral_vector) > target_dim:
            # Truncate if too long
            context = spectral_vector[:target_dim]
        else:
            # Pad if too short
            context = np.zeros(target_dim)
            context[:len(spectral_vector)] = spectral_vector
        
        print(f"[SPECTRAL] Generated spectral context: {len(spectral_features)} features → {target_dim}D")
        return context
        
    except Exception as e:
        print(f"[SPECTRAL] Error computing spectral context: {e}")
        return np.zeros(target_dim)

def group_eigenvalues(eigenvals: np.ndarray, threshold: float = 0.01) -> List[Tuple[int, int]]:
    """
    Group eigenvalues that are close together (handles multiplicities).
    
    Returns list of (start_idx, end_idx) for each group.
    """
    if len(eigenvals) <= 1:
        return [(0, len(eigenvals)-1)] if len(eigenvals) > 0 else []
    
    groups = []
    start = 0
    
    for i in range(1, len(eigenvals)):
        if eigenvals[i] - eigenvals[i-1] > threshold:
            # End current group
            groups.append((start, i-1))
            start = i
    
    # Add final group
    groups.append((start, len(eigenvals)-1))
    
    return groups

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

def create_spectral_enhanced_super_embeddings(clusters: List[List[int]], 
                                            original_graph: nx.Graph, 
                                            super_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    NEW: Create spectral contexts for each supernode.
    
    Returns:
        super_embeddings: Original super-embeddings (unchanged)
        spectral_contexts: Spectral context for each supernode
    """
    print(f"\n🌟 CREATING SPECTRAL CONTEXTS FOR SUPERNODES")
    print("="*60)
    
    spectral_contexts = []
    target_dim = super_embeddings.shape[1]  # Match embedding dimension
    
    for cluster_id, cluster in enumerate(clusters):
        if len(cluster) == 0:
            spectral_contexts.append(np.zeros(target_dim))
            continue
        
        print(f"[SPECTRAL] Processing cluster {cluster_id}: {cluster}")
        
        # Extract cluster subgraph
        cluster_subgraph = extract_cluster_subgraph(original_graph, cluster)
        
        # Compute spectral context for this cluster
        spectral_context = compute_cluster_spectral_context(cluster_subgraph, target_dim)
        spectral_contexts.append(spectral_context)
        
        if cluster_id % 5 == 0 and cluster_id > 0:
            print(f"  Processed {cluster_id}/{len(clusters)} spectral contexts")
    
    spectral_contexts = np.array(spectral_contexts)
    
    print(f"[SPECTRAL] ✅ Created spectral contexts: {spectral_contexts.shape}")
    print(f"[SPECTRAL] Context stats: mean={np.mean(spectral_contexts):.4f}, std={np.std(spectral_contexts):.4f}")
    
    return super_embeddings, spectral_contexts

def map_spectral_context_to_nodes(spectral_contexts: np.ndarray, 
                                 clusters: List[List[int]], 
                                 num_nodes: int) -> np.ndarray:
    """
    Map supernode spectral contexts to individual nodes.
    """
    print(f"[SPECTRAL-MAP] Mapping spectral contexts to {num_nodes} nodes")
    
    node_contexts = np.zeros((num_nodes, spectral_contexts.shape[1]))
    
    for cluster_id, cluster in enumerate(clusters):
        cluster_spectral = spectral_contexts[cluster_id]
        
        for node in cluster:
            if node < num_nodes:
                # Each node gets its cluster's spectral context
                node_contexts[node] = cluster_spectral
    
    print(f"[SPECTRAL-MAP] ✅ Mapped contexts: {node_contexts.shape}")
    return node_contexts

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
        print("------------------------this is wrong fix it coarse_ratio=", coarse_ratio)
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

def three_way_spectral_refinement(super_embeddings: np.ndarray,
                                 spectral_contexts: np.ndarray,
                                 projections: List,
                                 laplacians: List,
                                 original_features: np.ndarray,
                                 clusters: List[List[int]],
                                 alpha: float = 0.7,
                                 beta: float = 0.2, 
                                 gamma: float = 0.1,
                                 lda: float = 0.1,
                                 power: bool = False) -> np.ndarray:
    """
    REVOLUTIONARY: Three-Way Spectral-Aware Refinement
    
    Combines:
    α: Hierarchical structure (spectral refinement)
    β: Individual characteristics (original features)  
    γ: Spectral structure (eigenspace context)
    """
    print("\n🚀 LEVEL 3: THREE-WAY SPECTRAL-AWARE REFINEMENT (REVOLUTIONARY)")
    print("="*80)
    
    print(f"🎯 THREE-WAY COMBINATION WEIGHTS:")
    print(f"   α={alpha:.2f} (Hierarchical structure)")
    print(f"   β={beta:.2f} (Individual characteristics)")  
    print(f"   γ={gamma:.2f} (Spectral eigenspace context)")
    print(f"   Total: {alpha + beta + gamma:.2f}")
    
    try:
        # Import GraphZoom utilities
        try:
            from utils import smooth_filter
        except ImportError:
            sys.path.append('..')
            sys.path.append('../..')
            from utils import smooth_filter
        
        print(f"[3-WAY] Input shapes:")
        print(f"  Super-embeddings: {super_embeddings.shape}")
        print(f"  Spectral contexts: {spectral_contexts.shape}")
        print(f"  Original features: {original_features.shape}")
        print(f"  Projection levels: {len(projections)}")
        
        # COMPONENT 1: Hierarchical Structure (α)
        print(f"\n🔥 COMPONENT 1: HIERARCHICAL STRUCTURE (α={alpha})")
        embeddings = super_embeddings.copy()
        
        for i in reversed(range(len(projections))):
            print(f"  Level {i}: projection {projections[i].shape}")
            embeddings = projections[i] @ embeddings
            filter_ = smooth_filter(laplacians[i], lda)
            if power or i == 0:
                embeddings = filter_ @ (filter_ @ embeddings)
        
        hierarchical_component = embeddings
        print(f"  ✅ Hierarchical component: {hierarchical_component.shape}")
        
        # COMPONENT 2: Individual Characteristics (β)
        print(f"\n🎨 COMPONENT 2: INDIVIDUAL CHARACTERISTICS (β={beta})")
        embed_dim = hierarchical_component.shape[1]
        feature_dim = original_features.shape[1]
        
        if feature_dim != embed_dim:
            if feature_dim > embed_dim:
                aligned_features = original_features[:, :embed_dim]
                print(f"  Truncated features: {feature_dim}D → {embed_dim}D")
            else:
                aligned_features = np.zeros((original_features.shape[0], embed_dim))
                aligned_features[:, :feature_dim] = original_features
                print(f"  Padded features: {feature_dim}D → {embed_dim}D")
        else:
            aligned_features = original_features
            print(f"  Features already aligned: {embed_dim}D")
        
        individual_component = aligned_features
        print(f"  ✅ Individual component: {individual_component.shape}")
        
        # COMPONENT 3: Spectral Eigenspace Context (γ)
        print(f"\n🌟 COMPONENT 3: SPECTRAL EIGENSPACE CONTEXT (γ={gamma})")
        node_spectral_contexts = map_spectral_context_to_nodes(
            spectral_contexts, clusters, original_features.shape[0]
        )
        spectral_component = node_spectral_contexts
        print(f"  ✅ Spectral component: {spectral_component.shape}")
        
        # NORMALIZATION (for stable combination)
        print(f"\n⚖️  NORMALIZING COMPONENTS...")
        def safe_normalize(x):
            norm = np.linalg.norm(x, axis=1, keepdims=True)
            norm[norm == 0] = 1
            return x / norm
        
        norm_hierarchical = safe_normalize(hierarchical_component)
        norm_individual = safe_normalize(individual_component) 
        norm_spectral = safe_normalize(spectral_component)
        
        print(f"  Normalized component stats:")
        print(f"    Hierarchical: mean={np.mean(norm_hierarchical):.4f}, std={np.std(norm_hierarchical):.4f}")
        print(f"    Individual:   mean={np.mean(norm_individual):.4f}, std={np.std(norm_individual):.4f}")
        print(f"    Spectral:     mean={np.mean(norm_spectral):.4f}, std={np.std(norm_spectral):.4f}")
        
        # THREE-WAY COMBINATION
        print(f"\n🎪 THREE-WAY COMBINATION...")
        final_embeddings = (
            alpha * norm_hierarchical +
            beta * norm_individual +
            gamma * norm_spectral
        )
        
        print(f"  ✅ Combined embeddings: {final_embeddings.shape}")
        print(f"  Final stats: mean={np.mean(final_embeddings):.6f}, std={np.std(final_embeddings):.6f}")
        
        # VALIDATION
        print(f"\n🔍 VALIDATION:")
        unique_embeddings = []
        for i, embedding in enumerate(final_embeddings):
            is_unique = True
            for unique_emb in unique_embeddings:
                if np.allclose(embedding, unique_emb, atol=1e-6):
                    is_unique = False
                    break
            if is_unique:
                unique_embeddings.append(embedding)
        
        diversity = len(unique_embeddings) / len(final_embeddings)
        print(f"  Embedding diversity: {len(unique_embeddings)}/{len(final_embeddings)} = {diversity:.3f}")
        
        if diversity > 0.98:
            print(f"  ✅ Excellent diversity!")
        elif diversity > 0.95:
            print(f"  ✅ Good diversity!")
        else:
            print(f"  ⚠️  Moderate diversity")
        
        print(f"\n🎯 THREE-WAY REFINEMENT COMPLETED!")
        return final_embeddings
        
    except Exception as e:
        print(f"[3-WAY] ❌ Three-way refinement failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to hierarchical only
        print(f"[3-WAY] Fallback: hierarchical component only")
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
                           refinement_alpha: float = 0.75,
                           refinement_beta: float = 0.125,
                           refinement_gamma: float = 0.125) -> np.ndarray:
    """
    New VERSION: True 3-level GraphSAGE with Three-Way Spectral-Aware Refinement!
    
    Level 1: Intra-cluster GraphSAGE → super-node features
    Level 2: Inter-cluster GraphSAGE → super-node embeddings + spectral contexts
    Level 3: Three-way spectral refinement → individualized node embeddings (REVOLUTIONARY!)
    
    Parameters:
        refinement_alpha: Weight for hierarchical structure (default: 0.7)
        refinement_beta: Weight for individual characteristics (default: 0.2)
        refinement_gamma: Weight for spectral eigenspace context (default: 0.1)
    """
    print("="*80)
    print("TRUE COARSENED GRAPHSAGE - THREE-WAY SPECTRAL-AWARE REFINEMENT")
    print("="*80)
    
    total_start = time.time()
    
    # Validate weights
    total_weight = refinement_alpha + refinement_beta + refinement_gamma
    if abs(total_weight - 1.0) > 0.01:
        print(f"⚠️  WARNING: Weights don't sum to 1.0: α+β+γ = {total_weight:.3f}")
        print(f"  Normalizing weights...")
        refinement_alpha /= total_weight
        refinement_beta /= total_weight  
        refinement_gamma /= total_weight
        print(f"  Normalized: α={refinement_alpha:.3f}, β={refinement_beta:.3f}, γ={refinement_gamma:.3f}")
    
    # LEVEL 1: Intra-cluster GraphSAGE aggregation
    print("\n🔥 LEVEL 1: Intra-cluster GraphSAGE aggregation")
    super_features = create_super_node_features(
        original_graph, features, clusters, super_embed_dim
    )
    
    # LEVEL 2: Inter-cluster GraphSAGE (on coarsened graph) + Spectral Context Creation
    print("\n🚀 LEVEL 2: Inter-cluster GraphSAGE + Spectral Context Creation")
    super_embeddings = apply_inter_cluster_graphsage(
        coarsened_graph, super_features, final_embed_dim, hidden_dim, 
        num_layers=2, training_epochs=training_epochs
    )
    
    # NEW: Create spectral contexts for each supernode
    super_embeddings, spectral_contexts = create_spectral_enhanced_super_embeddings(
        clusters, original_graph, super_embeddings
    )
    
    # LEVEL 3: Three-Way Spectral-Aware Refinement (REVOLUTIONARY!)
    print("\n✨ LEVEL 3: Three-Way Spectral-Aware Refinement (REVOLUTIONARY)")
    refine_start = time.time()
    
    final_embeddings = three_way_spectral_refinement(
        super_embeddings=super_embeddings,
        spectral_contexts=spectral_contexts,    # NEW: Spectral context per supernode
        projections=projections,
        laplacians=laplacians,
        original_features=features,
        clusters=clusters,
        alpha=refinement_alpha,                 # Hierarchical weight
        beta=refinement_beta,                   # Individual weight
        gamma=refinement_gamma,                 # NEW: Spectral weight
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
    print(f"  Three-way weights: α={refinement_alpha:.2f}, β={refinement_beta:.2f}, γ={refinement_gamma:.2f}")
    
    # FINAL VALIDATION
    print(f"\n🔍 FINAL EMBEDDING VALIDATION:")
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
        print("  ✅ PERFECT: All embeddings are unique!")
    else:
        duplicates = len(final_embeddings) - len(unique_embeddings)
        print(f"  ⚠️  {duplicates} duplicate embeddings ({duplicates/len(final_embeddings)*100:.1f}%)")
    
    print(f"\n🎯 EXPECTED PERFORMANCE:")
    print(f"  Previous best: 72.7% (α=0.88, β=0.12)")
    print(f"  Target with spectral context: 74-76%")
    print(f"  Three-way enhancement should add +1-3% accuracy!")
    
    return final_embeddings

def test_three_way_spectral_graphsage():
    """Test the THREE-WAY spectral-aware true coarsened GraphSAGE."""
    print("Testing THREE-WAY Spectral-Aware True Coarsened GraphSAGE...")
    
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
    
    # Run THREE-WAY spectral-aware true coarsened GraphSAGE
    embeddings = true_coarsened_graphsage(
        G, features, clusters, coarsened_G, projections, laplacians,
        training_epochs=50,       # Faster for testing
        refinement_alpha=0.7,     # Hierarchical
        refinement_beta=0.2,      # Individual  
        refinement_gamma=0.1      # NEW: Spectral
    )
    
    print(f"✅ Success! Output embeddings shape: {embeddings.shape}")
    print("THREE-WAY Spectral-Aware True Coarsened GraphSAGE test completed!")

if __name__ == "__main__":
    test_three_way_spectral_graphsage()