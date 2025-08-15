#!/usr/bin/env python3
"""
Complete Pipeline Analysis: Coarsening → Embedding → Refinement → Reconstruction
Tests what information is preserved through the complete GraphZoom pipeline
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix, diags
from scipy.io import mmwrite, mmread
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_experiment_results():
    """Load the results from the previous coarsening experiment"""
    print("Loading previous coarsening experiment results...")
    
    # Load original graph
    L_original = mmread("experiment_results/test_graph.mtx").tocsr()
    
    # Load simple coarsening results
    simple_projection = mmread("experiment_results/simple_projection.mtx").tocsr()
    simple_coarsened = mmread("experiment_results/simple_coarsened.mtx").tocsr()
    
    # Load CMG coarsening results
    cmg_projection = mmread("experiment_results/cmg_projection.mtx").tocsr()
    cmg_coarsened = mmread("experiment_results/cmg_coarsened.mtx").tocsr()
    
    return {
        'original': L_original,
        'simple': {'projection': simple_projection, 'coarsened': simple_coarsened},
        'cmg': {'projection': cmg_projection, 'coarsened': cmg_coarsened}
    }

def laplacian_to_networkx(L):
    """Convert Laplacian matrix to NetworkX graph"""
    # Extract adjacency: A = D - L
    degree_diag = diags(L.diagonal(), 0)
    adjacency = degree_diag - L
    
    # Ensure non-negative weights and convert to NetworkX
    adjacency.data = np.abs(adjacency.data)
    G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='weight')
    
    return G

def simple_spectral_embedding(G, dim=8):
    """Create simple spectral embeddings using Laplacian eigenvectors"""
    print(f"  Computing spectral embedding (dim={dim})...")
    
    # Get Laplacian matrix
    L = nx.normalized_laplacian_matrix(G, nodelist=sorted(G.nodes()))
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = sp.linalg.eigsh(L, k=min(dim+1, L.shape[0]-1), which='SM')
    
    # Skip the first eigenvector (constant) and take next 'dim' eigenvectors
    embeddings = eigenvectors[:, 1:dim+1]
    
    print(f"  Eigenvalues (smallest): {eigenvalues[:dim+1]}")
    print(f"  Embedding shape: {embeddings.shape}")
    
    return embeddings

def node2vec_like_embedding(G, dim=8, num_walks=10, walk_length=5):
    """Simple random walk-based embedding (Node2Vec-like)"""
    print(f"  Computing random walk embedding (dim={dim})...")
    
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    # Simple random walk sampling
    walks = []
    for node in nodes:
        for _ in range(num_walks):
            walk = [node]
            current = node
            for _ in range(walk_length):
                neighbors = list(G.neighbors(current))
                if neighbors:
                    current = np.random.choice(neighbors)
                    walk.append(current)
                else:
                    break
            walks.append(walk)
    
    # Simple co-occurrence matrix
    co_occurrence = np.zeros((n_nodes, n_nodes))
    for walk in walks:
        for i, node1 in enumerate(walk):
            for j, node2 in enumerate(walk):
                if i != j:
                    co_occurrence[node1, node2] += 1.0 / abs(i - j)
    
    # SVD for dimensionality reduction
    U, s, Vt = np.linalg.svd(co_occurrence)
    embeddings = U[:, :dim] * np.sqrt(s[:dim])
    
    print(f"  Embedding shape: {embeddings.shape}")
    return embeddings

def apply_refinement(coarse_embeddings, projection_matrix):
    """Apply GraphZoom refinement to project embeddings back to original size"""
    print(f"  Refining embeddings from {coarse_embeddings.shape} to original size...")
    
    # Project back: original_embeddings = P @ coarse_embeddings
    refined_embeddings = projection_matrix @ coarse_embeddings
    
    print(f"  Refined embedding shape: {refined_embeddings.shape}")
    return refined_embeddings

def smooth_refinement(coarse_embeddings, projection_matrix, L_original, lda=0.1):
    """Apply GraphZoom refinement with smoothing filter"""
    print(f"  Applying smooth refinement...")
    
    # Initial projection
    refined_embeddings = projection_matrix @ coarse_embeddings
    
    # Apply smoothing filter (simplified version of GraphZoom's approach)
    # Create normalized adjacency for smoothing
    degree_diag = diags(L_original.diagonal(), 0)
    adjacency = degree_diag - L_original
    
    # Add self-loops and normalize
    adjacency_smooth = adjacency + lda * sp.identity(L_original.shape[0])
    degree_smooth = np.array(adjacency_smooth.sum(axis=1)).flatten()
    
    # Avoid division by zero
    degree_smooth[degree_smooth == 0] = 1
    degree_inv_sqrt = sp.diags(1.0 / np.sqrt(degree_smooth))
    
    # Normalized adjacency for smoothing
    norm_adj = degree_inv_sqrt @ adjacency_smooth @ degree_inv_sqrt
    
    # Apply smoothing (multiple iterations)
    for _ in range(2):  # GraphZoom typically uses 2 iterations
        refined_embeddings = norm_adj @ refined_embeddings
    
    print(f"  Smooth refined embedding shape: {refined_embeddings.shape}")
    return refined_embeddings

def reconstruct_graph_from_embeddings(embeddings, method='cosine', threshold=0.1):
    """Attempt to reconstruct graph structure from embeddings"""
    print(f"  Reconstructing graph using {method} similarity (threshold={threshold})...")
    
    if method == 'cosine':
        # Cosine similarity
        similarity = cosine_similarity(embeddings)
    elif method == 'euclidean':
        # Euclidean distance → similarity
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(embeddings)
        similarity = 1.0 / (1.0 + distances)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create adjacency matrix by thresholding similarity
    n_nodes = similarity.shape[0]
    reconstructed_adj = np.zeros_like(similarity)
    
    # Keep only similarities above threshold, excluding self-loops
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):  # Upper triangle only
            if similarity[i, j] > threshold:
                reconstructed_adj[i, j] = similarity[i, j]
                reconstructed_adj[j, i] = similarity[i, j]  # Make symmetric
    
    # Convert to NetworkX graph
    G_reconstructed = nx.from_numpy_array(reconstructed_adj)
    
    print(f"  Reconstructed graph: {G_reconstructed.number_of_nodes()} nodes, {G_reconstructed.number_of_edges()} edges")
    return G_reconstructed, reconstructed_adj

def compare_graphs(G_original, G_reconstructed, title=""):
    """Compare original and reconstructed graphs"""
    print(f"\n{title} Graph Comparison:")
    print("-" * (len(title) + 20))
    
    # Basic statistics
    print(f"Original:      {G_original.number_of_nodes()} nodes, {G_original.number_of_edges()} edges")
    print(f"Reconstructed: {G_reconstructed.number_of_nodes()} nodes, {G_reconstructed.number_of_edges()} edges")
    
    # Edge overlap analysis
    original_edges = set(G_original.edges())
    reconstructed_edges = set(G_reconstructed.edges())
    
    # Make edges undirected for comparison
    original_edges = {tuple(sorted(edge)) for edge in original_edges}
    reconstructed_edges = {tuple(sorted(edge)) for edge in reconstructed_edges}
    
    common_edges = original_edges.intersection(reconstructed_edges)
    missing_edges = original_edges - reconstructed_edges
    extra_edges = reconstructed_edges - original_edges
    
    print(f"\nEdge Analysis:")
    print(f"  Common edges:    {len(common_edges)} / {len(original_edges)} = {len(common_edges)/len(original_edges)*100:.1f}% recall")
    print(f"  Missing edges:   {len(missing_edges)}")
    print(f"  Extra edges:     {len(extra_edges)}")
    
    if len(reconstructed_edges) > 0:
        precision = len(common_edges) / len(reconstructed_edges)
        print(f"  Precision:       {precision*100:.1f}%")
    
    # Show specific edges
    if len(missing_edges) > 0:
        print(f"  Missing: {sorted(list(missing_edges))}")
    if len(extra_edges) > 0:
        print(f"  Extra: {sorted(list(extra_edges))}")
    
    return {
        'recall': len(common_edges) / len(original_edges) if len(original_edges) > 0 else 0,
        'precision': len(common_edges) / len(reconstructed_edges) if len(reconstructed_edges) > 0 else 0,
        'common_edges': len(common_edges),
        'missing_edges': len(missing_edges),
        'extra_edges': len(extra_edges)
    }

def embedding_analysis_pipeline(method_name, coarsened_L, projection, L_original, embedding_type='spectral'):
    """Run complete embedding analysis for one coarsening method"""
    print(f"\n{'='*60}")
    print(f"EMBEDDING ANALYSIS: {method_name.upper()} COARSENING")
    print(f"{'='*60}")
    
    # Convert to NetworkX graphs
    G_coarsened = laplacian_to_networkx(coarsened_L)
    G_original = laplacian_to_networkx(L_original)
    
    print(f"Original graph: {G_original.number_of_nodes()} nodes, {G_original.number_of_edges()} edges")
    print(f"Coarsened graph: {G_coarsened.number_of_nodes()} nodes, {G_coarsened.number_of_edges()} edges")
    
    # Step 1: Embed coarsened graph
    print(f"\nStep 1: Embedding coarsened graph using {embedding_type}...")
    if embedding_type == 'spectral':
        coarse_embeddings = simple_spectral_embedding(G_coarsened)
    elif embedding_type == 'random_walk':
        coarse_embeddings = node2vec_like_embedding(G_coarsened)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    # Step 2: Embed original graph (for comparison)
    print(f"\nStep 2: Embedding original graph for comparison...")
    if embedding_type == 'spectral':
        original_embeddings = simple_spectral_embedding(G_original)
    elif embedding_type == 'random_walk':
        original_embeddings = node2vec_like_embedding(G_original)
    
    # Step 3: Apply refinement
    print(f"\nStep 3: Refining coarsened embeddings to original size...")
    refined_embeddings_simple = apply_refinement(coarse_embeddings, projection)
    refined_embeddings_smooth = smooth_refinement(coarse_embeddings, projection, L_original)
    
    # Step 4: Reconstruct graphs from embeddings
    print(f"\nStep 4: Reconstructing graphs from embeddings...")
    
    # Original embedding reconstruction
    G_recon_original, _ = reconstruct_graph_from_embeddings(original_embeddings, threshold=0.1)
    
    # Simple refinement reconstruction
    G_recon_simple, _ = reconstruct_graph_from_embeddings(refined_embeddings_simple, threshold=0.1)
    
    # Smooth refinement reconstruction
    G_recon_smooth, _ = reconstruct_graph_from_embeddings(refined_embeddings_smooth, threshold=0.1)
    
    # Step 5: Compare reconstructions
    print(f"\nStep 5: Comparing reconstructions...")
    
    original_comparison = compare_graphs(G_original, G_recon_original, "Original Embedding")
    simple_comparison = compare_graphs(G_original, G_recon_simple, f"{method_name} + Simple Refinement")
    smooth_comparison = compare_graphs(G_original, G_recon_smooth, f"{method_name} + Smooth Refinement")
    
    # Step 6: Embedding quality analysis
    print(f"\nStep 6: Embedding quality analysis...")
    
    # Compare embedding similarities
    original_sim = cosine_similarity(original_embeddings)
    refined_simple_sim = cosine_similarity(refined_embeddings_simple)
    refined_smooth_sim = cosine_similarity(refined_embeddings_smooth)
    
    # Correlation between similarity matrices
    orig_flat = original_sim[np.triu_indices_from(original_sim, k=1)]
    simple_flat = refined_simple_sim[np.triu_indices_from(refined_simple_sim, k=1)]
    smooth_flat = refined_smooth_sim[np.triu_indices_from(refined_smooth_sim, k=1)]
    
    simple_corr = np.corrcoef(orig_flat, simple_flat)[0, 1]
    smooth_corr = np.corrcoef(orig_flat, smooth_flat)[0, 1]
    
    print(f"  Embedding similarity correlation:")
    print(f"    Simple refinement: {simple_corr:.3f}")
    print(f"    Smooth refinement: {smooth_corr:.3f}")
    
    return {
        'method': method_name,
        'original_comparison': original_comparison,
        'simple_comparison': simple_comparison,
        'smooth_comparison': smooth_comparison,
        'simple_correlation': simple_corr,
        'smooth_correlation': smooth_corr
    }

def main():
    """Run complete embedding and refinement analysis"""
    print("COMPLETE PIPELINE ANALYSIS: COARSENING → EMBEDDING → REFINEMENT")
    print("="*80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load previous experiment results
    results = load_experiment_results()
    
    # Analyze both coarsening methods
    embedding_types = ['spectral', 'random_walk']
    
    all_results = {}
    
    for embedding_type in embedding_types:
        print(f"\n{'#'*80}")
        print(f"TESTING {embedding_type.upper()} EMBEDDINGS")
        print(f"{'#'*80}")
        
        all_results[embedding_type] = {}
        
        # Test simple coarsening
        simple_results = embedding_analysis_pipeline(
            'simple', 
            results['simple']['coarsened'], 
            results['simple']['projection'], 
            results['original'],
            embedding_type
        )
        all_results[embedding_type]['simple'] = simple_results
        
        # Test CMG coarsening
        cmg_results = embedding_analysis_pipeline(
            'cmg', 
            results['cmg']['coarsened'], 
            results['cmg']['projection'], 
            results['original'],
            embedding_type
        )
        all_results[embedding_type]['cmg'] = cmg_results
    
    # Final comparison summary
    print(f"\n{'='*80}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for embedding_type in embedding_types:
        print(f"\n{embedding_type.upper()} EMBEDDING RESULTS:")
        print("-" * 40)
        
        simple_data = all_results[embedding_type]['simple']
        cmg_data = all_results[embedding_type]['cmg']
        
        print(f"Edge Reconstruction Recall (Original Graph):")
        print(f"  Simple + Simple Refinement: {simple_data['simple_comparison']['recall']*100:.1f}%")
        print(f"  Simple + Smooth Refinement: {simple_data['smooth_comparison']['recall']*100:.1f}%")
        print(f"  CMG + Simple Refinement:    {cmg_data['simple_comparison']['recall']*100:.1f}%")
        print(f"  CMG + Smooth Refinement:    {cmg_data['smooth_comparison']['recall']*100:.1f}%")
        
        print(f"\nEmbedding Similarity Correlation:")
        print(f"  Simple coarsening: {simple_data['simple_correlation']:.3f} (simple), {simple_data['smooth_correlation']:.3f} (smooth)")
        print(f"  CMG coarsening:    {cmg_data['simple_correlation']:.3f} (simple), {cmg_data['smooth_correlation']:.3f} (smooth)")
    
    print(f"\n{'='*80}")
    print("CONCLUSIONS")
    print(f"{'='*80}")
    print("1. Check which coarsening method preserves more structural information")
    print("2. Compare simple vs smooth refinement effectiveness")  
    print("3. See which embedding type works better with each coarsening method")
    print("4. Analyze what information is lost vs preserved in the pipeline")

if __name__ == "__main__":
    main()
