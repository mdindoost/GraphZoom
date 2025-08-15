#!/usr/bin/env python3
"""
Test: Message-Passing Only vs Traditional Refinement
Compares three approaches:
1. Original GraphZoom refinement
2. Traditional + Enhanced MP refinement  
3. MP-Only refinement (skip traditional entirely)
"""

import numpy as np
import time
from scipy.sparse import csr_matrix, diags
import sys
sys.path.append('.')

def mp_only_refinement(levels, projections, laplacians, embeddings, lda=0.1):
    """
    Message-passing ONLY refinement - skip traditional GraphZoom steps
    Just project + apply MP correction
    """
    from enhanced_refinement import (
        build_mp_preserving_matrix, 
        extract_clusters_from_projection,
        compute_message_passing_error,
        laplacian_to_propagation,
        build_correction_filter
    )
    
    print(f"[MP-ONLY REFINEMENT] Skipping traditional spectral smoothing")
    
    for i in reversed(range(levels)):
        print(f"  Level {i}: MP-only processing...")
        
        # Step 1: ONLY projection (no spectral smoothing)
        embeddings = projections[i] @ embeddings
        print(f"    After projection: {embeddings.shape}")
        
        # Step 2: ONLY message-passing correction
        laplacian = laplacians[i]
        clusters = extract_clusters_from_projection(projections[i])
        Q = build_mp_preserving_matrix(clusters, laplacian.shape[0])
        S = laplacian_to_propagation(laplacian)
        mp_error, S_approx, S_c = compute_message_passing_error(S, Q)
        
        print(f"    MP error: {mp_error:.6f}")
        
        if mp_error > 0.05:  # Lower threshold for MP-only
            print(f"    Applying MP correction...")
            correction_filter = build_correction_filter(S_approx, lda)
            embeddings = correction_filter @ embeddings
            print(f"    After MP correction: {embeddings.shape}")
        else:
            print(f"    MP error low - skipping correction")
    
    return embeddings

def traditional_refinement_only(levels, projections, laplacians, embeddings, lda=0.1):
    """
    Traditional GraphZoom refinement ONLY (for comparison)
    """
    print(f"[TRADITIONAL REFINEMENT] Original GraphZoom approach")
    
    for i in reversed(range(levels)):
        print(f"  Level {i}: Traditional processing...")
        
        # Traditional GraphZoom steps
        embeddings = projections[i] @ embeddings
        
        # Import smooth_filter
        try:
            from utils import smooth_filter
        except:
            # Fallback implementation
            def smooth_filter(laplacian_matrix, lda):
                dim = laplacian_matrix.shape[0]
                adj_matrix = diags(laplacian_matrix.diagonal(), 0) - laplacian_matrix + lda * sp.identity(dim)
                degree_vec = adj_matrix.sum(axis=1)
                with np.errstate(divide='ignore'):
                    d_inv_sqrt = np.squeeze(np.asarray(np.power(degree_vec, -0.5)))
                d_inv_sqrt[np.isinf(d_inv_sqrt)|np.isnan(d_inv_sqrt)] = 0
                degree_matrix = diags(d_inv_sqrt, 0)
                norm_adj = degree_matrix @ (adj_matrix @ degree_matrix)
                return norm_adj
        
        filter_ = smooth_filter(laplacians[i], lda)
        embeddings = filter_ @ (filter_ @ embeddings)  # Double smoothing
        print(f"    After traditional smoothing: {embeddings.shape}")
    
    return embeddings

def enhanced_refinement_traditional_plus_mp(levels, projections, laplacians, embeddings, lda=0.1):
    """
    Enhanced refinement: Traditional + MP correction (current approach)
    """
    from enhanced_refinement import (
        build_mp_preserving_matrix, 
        extract_clusters_from_projection,
        compute_message_passing_error,
        laplacian_to_propagation,
        build_correction_filter
    )
    
    print(f"[TRADITIONAL + MP REFINEMENT] Combined approach")
    
    for i in reversed(range(levels)):
        print(f"  Level {i}: Traditional + MP processing...")
        
        # Step 1: Traditional projection
        embeddings = projections[i] @ embeddings
        
        # Step 2: Traditional spectral smoothing
        try:
            from utils import smooth_filter
        except:
            def smooth_filter(laplacian_matrix, lda):
                import scipy.sparse as sp
                dim = laplacian_matrix.shape[0]
                adj_matrix = diags(laplacian_matrix.diagonal(), 0) - laplacian_matrix + lda * sp.identity(dim)
                degree_vec = adj_matrix.sum(axis=1)
                with np.errstate(divide='ignore'):
                    d_inv_sqrt = np.squeeze(np.asarray(np.power(degree_vec, -0.5)))
                d_inv_sqrt[np.isinf(d_inv_sqrt)|np.isnan(d_inv_sqrt)] = 0
                degree_matrix = diags(d_inv_sqrt, 0)
                norm_adj = degree_matrix @ (adj_matrix @ degree_matrix)
                return norm_adj
        
        filter_ = smooth_filter(laplacians[i], lda)
        embeddings = filter_ @ (filter_ @ embeddings)
        print(f"    After traditional smoothing: {embeddings.shape}")
        
        # Step 3: THEN apply MP correction
        laplacian = laplacians[i]
        clusters = extract_clusters_from_projection(projections[i])
        Q = build_mp_preserving_matrix(clusters, laplacian.shape[0])
        S = laplacian_to_propagation(laplacian)
        mp_error, S_approx, S_c = compute_message_passing_error(S, Q)
        
        print(f"    MP error: {mp_error:.6f}")
        
        if mp_error > 0.1:
            print(f"    Applying MP correction on top...")
            correction_filter = build_correction_filter(S_approx, lda)
            embeddings = correction_filter @ embeddings
            print(f"    After MP correction: {embeddings.shape}")
    
    return embeddings

def run_refinement_comparison_test():
    """
    Test the three refinement approaches on a small example
    """
    print("REFINEMENT APPROACH COMPARISON TEST")
    print("="*60)
    
    # Load test data
    from fixed_downstream_evaluation import load_cora_dataset, create_train_test_split, evaluate_node_classification, train_real_embeddings
    
    laplacian, features, labels, dataset_name = load_cora_dataset()
    if laplacian is None:
        print("❌ Failed to load dataset")
        return
    
    n_nodes = laplacian.shape[0]
    train_mask, val_mask, test_mask = create_train_test_split(n_nodes)
    
    # Run one coarsening method for testing
    print("\n🧪 Testing with CMG-2 coarsening...")
    
    try:
        from cmg_coarsening_timed import cmg_coarse
        from scipy.sparse import diags
        import networkx as nx
        
        # Convert to NetworkX graph
        degree_diag = diags(laplacian.diagonal(), 0)
        adjacency = degree_diag - laplacian
        G = nx.from_scipy_sparse_matrix(adjacency)
        
        # Run CMG coarsening
        G_coarse, projections, laplacians, levels = cmg_coarse(
            laplacian, level=2, k=10, d=20, threshold=0.1
        )
        
        print(f"Coarsened: {laplacian.shape[0]} → {G_coarse.number_of_nodes()} nodes")
        
        # Generate real embeddings
        coarse_embeddings = train_real_embeddings(G_coarse, method="deepwalk")
        print(f"Generated embeddings: {coarse_embeddings.shape}")
        
        # Test three refinement approaches
        approaches = [
            ("Traditional Only", traditional_refinement_only),
            ("Traditional + MP", enhanced_refinement_traditional_plus_mp),
            ("MP Only", mp_only_refinement),
        ]
        
        results = []
        
        for approach_name, refinement_func in approaches:
            print(f"\n--- Testing {approach_name} ---")
            
            start_time = time.time()
            refined_embeddings = refinement_func(
                levels, projections, laplacians, coarse_embeddings.copy(), lda=0.1
            )
            refinement_time = time.time() - start_time
            
            print(f"Refinement time: {refinement_time:.3f}s")
            print(f"Final embeddings shape: {refined_embeddings.shape}")
            
            # Evaluate performance
            eval_results = evaluate_node_classification(
                refined_embeddings, None, labels, train_mask, val_mask, test_mask, use_features=False
            )
            
            results.append({
                'approach': approach_name,
                'time': refinement_time,
                **eval_results
            })
        
        # Print comparison
        print(f"\n" + "="*80)
        print("REFINEMENT APPROACH COMPARISON RESULTS")
        print("="*80)
        
        print(f"{'Approach':<20} {'Test Acc':<8} {'Val Acc':<8} {'Time(s)':<8} {'Efficiency':<10}")
        print("-" * 70)
        
        baseline_time = results[0]['time']
        
        for result in results:
            acc = result['test_accuracy']
            val_acc = result['val_accuracy']
            time_taken = result['time']
            efficiency = acc / time_taken  # Accuracy per second
            
            print(f"{result['approach']:<20} {acc:<8.4f} {val_acc:<8.4f} {time_taken:<8.3f} {efficiency:<10.3f}")
        
        # Analysis
        print(f"\n📊 ANALYSIS:")
        
        traditional_only = results[0]
        traditional_plus_mp = results[1]
        mp_only = results[2]
        
        # Compare MP-only vs Traditional-only
        mp_vs_trad = mp_only['test_accuracy'] - traditional_only['test_accuracy']
        print(f"   MP-Only vs Traditional-Only: {mp_vs_trad:+.4f}")
        
        # Compare MP-only vs Traditional+MP
        mp_vs_combined = mp_only['test_accuracy'] - traditional_plus_mp['test_accuracy']
        print(f"   MP-Only vs Traditional+MP: {mp_vs_combined:+.4f}")
        
        # Time comparison
        mp_speedup = traditional_plus_mp['time'] / mp_only['time']
        print(f"   MP-Only is {mp_speedup:.1f}x faster than Traditional+MP")
        
        # Efficiency analysis
        mp_efficiency = mp_only['test_accuracy'] / mp_only['time']
        combined_efficiency = traditional_plus_mp['test_accuracy'] / traditional_plus_mp['time']
        print(f"   MP-Only efficiency: {mp_efficiency:.3f} acc/sec")
        print(f"   Traditional+MP efficiency: {combined_efficiency:.3f} acc/sec")
        
        # Conclusion
        if mp_vs_combined > -0.005 and mp_speedup > 2:
            print(f"\n🎯 CONCLUSION: MP-Only appears superior!")
            print(f"   - Similar accuracy ({mp_vs_combined:+.4f})")
            print(f"   - Much faster ({mp_speedup:.1f}x speedup)")
        elif mp_vs_trad > 0.01:
            print(f"\n🎯 CONCLUSION: MP correction provides value!")
            print(f"   - MP-Only beats Traditional-Only by {mp_vs_trad:+.4f}")
        else:
            print(f"\n🤔 CONCLUSION: Traditional refinement still valuable")
            print(f"   - Combined approach may be best overall")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_refinement_comparison_test()
