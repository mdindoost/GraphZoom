#!/usr/bin/env python3
"""
Fix the MP-aware refinement step
The issue: GraphZoom refinement does projection + smoothing
But MP-aware already did the projection with Q^T
So we only need the smoothing part!
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix, diags, identity

def mp_aware_refinement_fixed(lifted_embeddings, original_graph, lda=0.1):
    """
    Fixed refinement for MP-aware pipeline
    Only applies smoothing since Q^T already did the projection
    """
    print(f"[MP_REFINEMENT] Input embeddings shape: {lifted_embeddings.shape}")
    
    try:
        # Import GraphZoom's smooth filter
        from utils import smooth_filter
        
        # Get Laplacian and create smooth filter
        L = nx.laplacian_matrix(original_graph, nodelist=sorted(original_graph.nodes()))
        filter_matrix = smooth_filter(L, lda)
        
        print(f"[MP_REFINEMENT] Filter matrix shape: {filter_matrix.shape}")
        print(f"[MP_REFINEMENT] Embeddings shape: {lifted_embeddings.shape}")
        
        # Apply smoothing only (GraphZoom does 2 iterations)
        refined_embeddings = lifted_embeddings.copy()
        for i in range(2):
            refined_embeddings = filter_matrix @ refined_embeddings
            print(f"[MP_REFINEMENT] After smoothing iteration {i+1}: {refined_embeddings.shape}")
        
        print(f"[MP_REFINEMENT] ✅ Smoothing completed successfully")
        return refined_embeddings
        
    except Exception as e:
        print(f"[MP_REFINEMENT] ❌ Smoothing failed: {e}")
        print(f"[MP_REFINEMENT] Returning unsmoothed embeddings")
        return lifted_embeddings

def fixed_mp_aware_pipeline(graph, features, labels, method='mp_aware', 
                           embedding_type='spectral', task='node_classification'):
    """
    Fixed MP-aware pipeline with corrected refinement
    """
    print(f"\n🔧 FIXED {method.upper()} PIPELINE")
    print(f"Embedding: {embedding_type}, Task: {task}")
    print("="*50)
    
    import time
    start_time = time.time()
    
    try:
        if method == 'mp_aware':
            # Import the fixed MP-aware coarsening
            from fix_mp_aware import mp_aware_coarsening_fixed
            
            # Step 1: MP-aware coarsening
            coarse_graph, Q_matrix, S_c_mp, projection_matrix = mp_aware_coarsening_fixed(graph)
            
            print(f"[FIXED] Coarsening: {graph.number_of_nodes()} → {coarse_graph.number_of_nodes()} nodes")
            
            # Step 2: Coarsen features
            coarse_features = Q_matrix @ features
            print(f"[FIXED] Features: {features.shape} → {coarse_features.shape}")
            
            # Step 3: Generate embeddings on coarsened graph
            from mp_aware_accuracy_pipeline import generate_embeddings
            coarse_embeddings = generate_embeddings(coarse_graph, coarse_features, embedding_type)
            print(f"[FIXED] Coarse embeddings: {coarse_embeddings.shape}")
            
            # Step 4: Lift back to original size (Q^T @ embeddings)
            lifted_embeddings = Q_matrix.T @ coarse_embeddings
            print(f"[FIXED] Lifted embeddings: {lifted_embeddings.shape}")
            
            # Step 5: Apply ONLY smoothing (not projection)
            refined_embeddings = mp_aware_refinement_fixed(lifted_embeddings, graph)
            print(f"[FIXED] Final embeddings: {refined_embeddings.shape}")
            
        elif method == 'traditional':
            # Traditional pipeline (for comparison)
            from mp_aware_accuracy_pipeline import traditional_cmg_coarsening, graphzoom_refinement
            
            coarse_graph, projection_matrix = traditional_cmg_coarsening(graph)
            coarse_features = projection_matrix.T @ features
            
            from mp_aware_accuracy_pipeline import generate_embeddings
            coarse_embeddings = generate_embeddings(coarse_graph, coarse_features, embedding_type)
            
            # Traditional GraphZoom refinement (does projection + smoothing)
            refined_embeddings = graphzoom_refinement(coarse_embeddings, projection_matrix, graph)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Step 6: Evaluate
        from mp_aware_accuracy_pipeline import evaluate_downstream_task
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

def test_fixed_pipeline():
    """
    Test the fixed pipeline
    """
    print("🧪 TESTING FIXED MP-AWARE PIPELINE")
    print("="*50)
    
    # Load test data
    from mp_aware_accuracy_pipeline import load_dataset
    graph, features, labels = load_dataset('test_12')
    
    print(f"Dataset: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Test traditional pipeline
    print("\n--- Testing Traditional Pipeline ---")
    trad_result = fixed_mp_aware_pipeline(
        graph, features, labels,
        method='traditional',
        embedding_type='spectral',
        task='node_classification'
    )
    
    # Test fixed MP-aware pipeline
    print("\n--- Testing Fixed MP-Aware Pipeline ---")
    mp_result = fixed_mp_aware_pipeline(
        graph, features, labels,
        method='mp_aware',
        embedding_type='spectral', 
        task='node_classification'
    )
    
    # Compare results
    print(f"\n📊 FIXED PIPELINE COMPARISON:")
    print(f"Traditional CMG: {trad_result['accuracy']:.4f}")
    print(f"MP-Aware CMG:    {mp_result['accuracy']:.4f}")
    
    improvement = mp_result['accuracy'] - trad_result['accuracy']
    print(f"Improvement:     {improvement:+.4f}")
    
    if improvement > 0:
        print("🏆 MP-Aware WINS!")
    elif improvement < 0:
        print("🤔 Traditional wins")
    else:
        print("🤝 Tie")
    
    return trad_result, mp_result

def comprehensive_fixed_comparison():
    """
    Run comprehensive comparison with fixed pipeline
    """
    print("🔬 COMPREHENSIVE FIXED COMPARISON")
    print("="*50)
    
    from mp_aware_accuracy_pipeline import load_dataset
    
    # Test configurations
    datasets = ['test_12']
    embedding_types = ['spectral', 'random_walk']
    tasks = ['node_classification']
    
    all_results = []
    
    for dataset in datasets:
        print(f"\n📊 DATASET: {dataset}")
        print("-" * 30)
        
        graph, features, labels = load_dataset(dataset)
        
        # Add link prediction if graph has enough edges
        current_tasks = tasks.copy()
        if graph.number_of_edges() > 5:
            current_tasks.append('link_prediction')
        
        for task in current_tasks:
            print(f"\n🎯 TASK: {task}")
            
            for embedding_type in embedding_types:
                print(f"\n🔹 Embedding: {embedding_type}")
                
                # Traditional
                trad = fixed_mp_aware_pipeline(graph, features, labels, 'traditional', embedding_type, task)
                
                # MP-aware  
                mp = fixed_mp_aware_pipeline(graph, features, labels, 'mp_aware', embedding_type, task)
                
                # Store results
                all_results.extend([
                    {'dataset': dataset, 'task': task, 'embedding': embedding_type, 
                     'method': 'traditional', 'accuracy': trad['accuracy']},
                    {'dataset': dataset, 'task': task, 'embedding': embedding_type,
                     'method': 'mp_aware', 'accuracy': mp['accuracy']}
                ])
                
                # Print comparison
                improvement = mp['accuracy'] - trad['accuracy']
                print(f"  Traditional: {trad['accuracy']:.4f}")
                print(f"  MP-Aware:    {mp['accuracy']:.4f}")
                print(f"  Improvement: {improvement:+.4f}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL FIXED COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    # Group and summarize results
    from collections import defaultdict
    grouped = defaultdict(list)
    
    for result in all_results:
        key = (result['task'], result['embedding'], result['method'])
        grouped[key].append(result['accuracy'])
    
    # Print summary by task and embedding
    tasks_tested = set(r['task'] for r in all_results)
    embeddings_tested = set(r['embedding'] for r in all_results)
    
    for task in tasks_tested:
        print(f"\n{task.upper()} RESULTS:")
        print("-" * 40)
        
        for embedding in embeddings_tested:
            trad_accs = grouped.get((task, embedding, 'traditional'), [0])
            mp_accs = grouped.get((task, embedding, 'mp_aware'), [0])
            
            trad_avg = np.mean(trad_accs)
            mp_avg = np.mean(mp_accs)
            improvement = mp_avg - trad_avg
            
            print(f"{embedding.capitalize()} Embeddings:")
            print(f"  Traditional: {trad_avg:.4f}")
            print(f"  MP-Aware:    {mp_avg:.4f}")
            print(f"  Improvement: {improvement:+.4f}")
            
            if improvement > 0.01:
                print(f"  → MP-Aware WINS! 🏆")
            elif improvement < -0.01:
                print(f"  → Traditional wins")
            else:
                print(f"  → Tie")
    
    return all_results

if __name__ == "__main__":
    # Test the fixed pipeline
    test_fixed_pipeline()
    
    print("\n" + "="*60)
    
    # Run comprehensive comparison
    comprehensive_fixed_comparison()
