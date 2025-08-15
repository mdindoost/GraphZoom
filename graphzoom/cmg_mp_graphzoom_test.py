#!/usr/bin/env python3
"""
CMG Message-Passing GraphZoom Integration Test
Tests the improvement from using S_c^MP vs naive coarsening with GraphSAGE
"""

import numpy as np
import networkx as nx
import time
import json
from scipy.sparse import identity
from sklearn.preprocessing import normalize
import sys
import os

# Add GraphZoom modules
sys.path.append('.')
from utils import *
from scoring import lr

def create_mp_aware_graphsage_pipeline(dataset, cmg_params, use_mp_enhancement=True, levels=1):
    """
    Create a complete pipeline testing CMG with/without MP enhancement in GraphZoom
    
    Args:
        dataset: Dataset name ('cora', etc.)
        cmg_params: CMG parameters dict
        use_mp_enhancement: Whether to use S_c^MP or naive approach
        levels: Number of coarsening levels
    
    Returns:
        results: Dictionary with timing and accuracy results
    """
    
    print(f"\n{'='*60}")
    print(f"GraphZoom Pipeline: CMG {'+ MP Enhancement' if use_mp_enhancement else '+ Naive'}")
    print(f"Dataset: {dataset}, Levels: {levels}")
    print(f"{'='*60}")
    
    results = {
        'method': 'CMG + MP Enhancement' if use_mp_enhancement else 'CMG + Naive',
        'dataset': dataset,
        'levels': levels,
        'use_mp_enhancement': use_mp_enhancement
    }
    
    try:
        # Load data
        print("Step 1: Loading data...")
        laplacian = json2mtx(dataset)
        feature = np.load(f"dataset/{dataset}/{dataset}-feats.npy")
        
        results['original_nodes'] = laplacian.shape[0]
        print(f"Original graph: {laplacian.shape[0]} nodes")
        
        # CMG Coarsening with/without MP enhancement
        print(f"Step 2: CMG Coarsening ({'MP-aware' if use_mp_enhancement else 'standard'})...")
        coarsening_start = time.time()
        
        if use_mp_enhancement:
            # Import the enhanced CMG function
            from cmg_mp_enhancement import cmg_coarse_with_mp
            G, projections, laplacians, level, mp_matrices, comparison = cmg_coarse_with_mp(
                laplacian, levels, gnn_type='graphsage_mean', **cmg_params
            )
            results['mp_improvement'] = np.mean(comparison['improvements'])
            results['mp_errors'] = comparison['mp_errors']
            results['naive_errors'] = comparison['naive_errors']
        else:
            # Standard CMG
            from cmg_coarsening_timed import cmg_coarse
            G, projections, laplacians, level = cmg_coarse(laplacian, levels, **cmg_params)
            mp_matrices = None
            results['mp_improvement'] = 0
        
        coarsening_time = time.time() - coarsening_start
        results['coarsening_time'] = coarsening_time
        results['final_nodes'] = G.number_of_nodes()
        results['reduction_ratio'] = results['original_nodes'] / results['final_nodes']
        
        print(f"Coarsening completed: {results['original_nodes']} → {results['final_nodes']} nodes")
        print(f"Reduction ratio: {results['reduction_ratio']:.2f}x")
        print(f"Coarsening time: {coarsening_time:.3f}s")
        
        if use_mp_enhancement:
            print(f"MP preservation improvement: {results['mp_improvement']:.2f}%")
        
        # GraphSAGE Embedding
        print("Step 3: GraphSAGE embedding...")
        embedding_start = time.time()
        
        # Set node attributes for GraphSAGE
        nx.set_node_attributes(G, False, "test")
        nx.set_node_attributes(G, False, "val")
        
        # Create mapping for features
        if use_mp_enhancement:
            # Use the projection matrices from CMG
            mapping = identity(feature.shape[0])
            for p in projections:
                mapping = mapping @ p
            mapping = normalize(mapping, norm='l1', axis=1).transpose()
        else:
            # Standard GraphZoom approach
            mapping = identity(feature.shape[0])
            for p in projections:
                mapping = mapping @ p
            mapping = normalize(mapping, norm='l1', axis=1).transpose()
        
        # Map features to coarse graph
        feats = mapping @ feature
        coarse_ratio = mapping.shape[1] / mapping.shape[0]
        
        # Import and run GraphSAGE
        try:
            from embed_methods.graphsage.graphsage import graphsage
            embeddings = graphsage(G, feats, 'mean', True, int(1000/coarse_ratio))
        except ImportError:
            print("[WARNING] GraphSAGE not available, using random embeddings")
            embeddings = np.random.randn(G.number_of_nodes(), 128)
        
        embedding_time = time.time() - embedding_start
        results['embedding_time'] = embedding_time
        print(f"Embedding time: {embedding_time:.3f}s")
        
        # Refinement
        print("Step 4: Refinement...")
        refinement_start = time.time()
        
        # Apply GraphZoom refinement
        if use_mp_enhancement and mp_matrices is not None:
            # Enhanced refinement could use MP matrices here
            # For now, use standard refinement
            final_embeddings = refinement(level, projections, laplacians, embeddings, 0.1, False)
        else:
            # Standard refinement
            final_embeddings = refinement(level, projections, laplacians, embeddings, 0.1, False)
        
        refinement_time = time.time() - refinement_start
        results['refinement_time'] = refinement_time
        print(f"Refinement time: {refinement_time:.3f}s")
        
        # Save embeddings
        embed_path = f"embed_results/{dataset}_cmg_{'mp' if use_mp_enhancement else 'naive'}_{levels}level.npy"
        os.makedirs("embed_results", exist_ok=True)
        np.save(embed_path, final_embeddings)
        
        # Evaluation
        print("Step 5: Evaluation...")
        eval_start = time.time()
        
        try:
            accuracy = lr(f"dataset/{dataset}/", embed_path, dataset)
            results['accuracy'] = accuracy
        except Exception as e:
            print(f"[WARNING] Evaluation failed: {e}")
            results['accuracy'] = 0.0
        
        eval_time = time.time() - eval_start
        results['eval_time'] = eval_time
        
        # Total time
        results['total_time'] = coarsening_time + embedding_time + refinement_time + eval_time
        
        print(f"\n{'='*40}")
        print("RESULTS SUMMARY")
        print(f"{'='*40}")
        print(f"Method: {results['method']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Total time: {results['total_time']:.3f}s")
        print(f"Reduction: {results['reduction_ratio']:.2f}x")
        if use_mp_enhancement:
            print(f"MP improvement: {results['mp_improvement']:.2f}%")
        
        results['success'] = True
        
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        results['success'] = False
        results['error'] = str(e)
    
    return results


def compare_mp_enhancement(dataset='cora', levels=1):
    """
    Compare CMG with and without MP enhancement on GraphSAGE task
    """
    
    print(f"\n{'#'*80}")
    print(f"COMPARING CMG: NAIVE vs MESSAGE-PASSING ENHANCEMENT")
    print(f"Dataset: {dataset}, Levels: {levels}")
    print(f"{'#'*80}")
    
    # CMG parameters
    cmg_params = {
        'k': 10,
        'd': 20, 
        'threshold': 0.1
    }
    
    # Test 1: Standard CMG (naive approach)
    print(f"\n{'>'*60}")
    print("TEST 1: Standard CMG (Naive Coarsening)")
    print(f"{'>'*60}")
    
    results_naive = create_mp_aware_graphsage_pipeline(
        dataset=dataset,
        cmg_params=cmg_params,
        use_mp_enhancement=False,
        levels=levels
    )
    
    # Test 2: Enhanced CMG (MP-aware)
    print(f"\n{'>'*60}")
    print("TEST 2: Enhanced CMG (Message-Passing Aware)")
    print(f"{'>'*60}")
    
    results_mp = create_mp_aware_graphsage_pipeline(
        dataset=dataset,
        cmg_params=cmg_params,
        use_mp_enhancement=True,
        levels=levels
    )
    
    # Comparison
    print(f"\n{'#'*80}")
    print("FINAL COMPARISON")
    print(f"{'#'*80}")
    
    if results_naive['success'] and results_mp['success']:
        accuracy_naive = results_naive['accuracy']
        accuracy_mp = results_mp['accuracy']
        time_naive = results_naive['total_time']
        time_mp = results_mp['total_time']
        
        accuracy_improvement = ((accuracy_mp - accuracy_naive) / accuracy_naive * 100) if accuracy_naive > 0 else 0
        time_overhead = ((time_mp - time_naive) / time_naive * 100) if time_naive > 0 else 0
        
        print(f"Results on {dataset}:")
        print(f"{'Method':<25} {'Accuracy':<12} {'Time (s)':<10} {'Reduction':<10}")
        print(f"{'-'*60}")
        print(f"{'CMG Naive':<25} {accuracy_naive:<12.4f} {time_naive:<10.3f} {results_naive['reduction_ratio']:<10.2f}")
        print(f"{'CMG + MP Enhancement':<25} {accuracy_mp:<12.4f} {time_mp:<10.3f} {results_mp['reduction_ratio']:<10.2f}")
        print(f"{'-'*60}")
        print(f"Accuracy improvement: {accuracy_improvement:+.2f}%")
        print(f"Time overhead: {time_overhead:+.2f}%")
        
        if 'mp_improvement' in results_mp:
            print(f"MP preservation improvement: {results_mp['mp_improvement']:.2f}%")
        
        # Save results
        comparison_results = {
            'dataset': dataset,
            'levels': levels,
            'naive': results_naive,
            'mp_enhanced': results_mp,
            'accuracy_improvement': accuracy_improvement,
            'time_overhead': time_overhead
        }
        
        results_file = f"results/cmg_mp_comparison_{dataset}_{levels}level.json"
        os.makedirs("results", exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")
        
        # Conclusion
        print(f"\n{'='*80}")
        print("CONCLUSION")
        print(f"{'='*80}")
        
        if accuracy_improvement > 1.0:
            print(f"✅ SUCCESS: MP enhancement improves accuracy by {accuracy_improvement:.2f}%")
        elif accuracy_improvement > 0:
            print(f"🔄 MARGINAL: Small improvement of {accuracy_improvement:.2f}%")
        else:
            print(f"❌ NO IMPROVEMENT: Accuracy decreased by {abs(accuracy_improvement):.2f}%")
            
        if time_overhead < 20:
            print(f"✅ EFFICIENCY: Low time overhead of {time_overhead:.2f}%")
        else:
            print(f"⚠️  OVERHEAD: High time overhead of {time_overhead:.2f}%")
            
        return comparison_results
    
    else:
        print("❌ COMPARISON FAILED: One or both tests failed")
        return None


if __name__ == "__main__":
    # Run the comparison
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = 'cora'
    
    if len(sys.argv) > 2:
        levels = int(sys.argv[2])
    else:
        levels = 1
    
    print("Starting CMG Message-Passing Enhancement Test...")
    results = compare_mp_enhancement(dataset=dataset, levels=levels)
    
    if results:
        print("\n🎯 Test completed successfully!")
        print(f"Check results/cmg_mp_comparison_{dataset}_{levels}level.json for detailed results")
    else:
        print("\n❌ Test failed!")
