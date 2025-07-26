#!/usr/bin/env python3
"""
Comprehensive GraphZoom Testing Script
Tests CMG vs LAMG vs Simple across multiple datasets and configurations
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import subprocess
from contextlib import redirect_stdout, redirect_stderr
import io

# Add the current directory to Python path for imports
sys.path.append('.')

from utils import *
from cmg_coarsening_timed import cmg_coarse, cmg_fusion_mapping
from scoring import lr


def count_connected_components(laplacian_or_graph):
    """Count connected components from laplacian matrix or NetworkX graph"""
    if hasattr(laplacian_or_graph, 'number_of_nodes'):  # NetworkX graph
        return nx.number_connected_components(laplacian_or_graph)
    else:  # Scipy sparse matrix (laplacian)
        # Convert laplacian to adjacency
        degree_diag = diags(laplacian_or_graph.diagonal(), 0)
        adjacency = degree_diag - laplacian_or_graph
        adjacency.data = np.abs(adjacency.data)  # Ensure non-negative
        G = nx.from_scipy_sparse_matrix(adjacency)
        return nx.number_connected_components(G)


def get_graph_stats(laplacian_or_graph, label="Graph"):
    """Extract nodes, edges, components from laplacian or NetworkX graph"""
    if hasattr(laplacian_or_graph, 'number_of_nodes'):  # NetworkX graph
        nodes = laplacian_or_graph.number_of_nodes()
        edges = laplacian_or_graph.number_of_edges()
        components = nx.number_connected_components(laplacian_or_graph)
    else:  # Scipy sparse matrix
        nodes = laplacian_or_graph.shape[0]
        edges = int((laplacian_or_graph.nnz - laplacian_or_graph.shape[0]) / 2)
        components = count_connected_components(laplacian_or_graph)
    
    print(f"[STATS] {label}: {nodes} nodes, {edges} edges, {components} components")
    return nodes, edges, components


def run_graphzoom_experiment(dataset, method, fusion_enabled, level_or_ratio, param_type, cmg_params=None):
    """
    Run single GraphZoom experiment and track graph statistics at each step
    
    Args:
        level_or_ratio: For simple/cmg this is 'level', for lamg this is 'reduce_ratio'
        param_type: 'level' or 'reduce_ratio'
    
    Returns dict with all statistics and results
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {dataset} | {method} | fusion={fusion_enabled} | {param_type}={level_or_ratio}")
    print(f"{'='*80}")
    
    result = {
        'dataset': dataset,
        'method': method,
        'fusion': fusion_enabled,
        'status': 'running'
    }
    
    # Set the appropriate parameter in result
    if param_type == 'level':
        result['level'] = level_or_ratio
        result['reduce_ratio_param'] = None
    else:  # param_type == 'reduce_ratio'
        result['level'] = None
        result['reduce_ratio_param'] = level_or_ratio
    
    # Default CMG parameters
    if cmg_params is None:
        cmg_params = {'k': 10, 'd': 15, 'threshold': 0.1}
    
    try:
        # Setup paths
        feature_path = f"dataset/{dataset}/{dataset}-feats.npy"
        reduce_results = "reduction_results/"
        mapping_path = f"{reduce_results}Mapping.mtx"
        param_str = f"{param_type}{level_or_ratio}"
        embed_path = f"results/{dataset}/embeddings_{method}_fusion{fusion_enabled}_{param_str}.npy"
        
        # Create results directory
        os.makedirs(f"results/{dataset}", exist_ok=True)
        os.makedirs(reduce_results, exist_ok=True)
        
        total_start_time = time.time()
        
        ########## Load Data ##########
        print("=" * 40 + " LOADING DATA " + "=" * 40)
        laplacian = json2mtx(dataset)
        
        # Original graph statistics
        orig_nodes, orig_edges, orig_components = get_graph_stats(laplacian, "Original")
        result.update({
            'original_nodes': orig_nodes,
            'original_edges': orig_edges, 
            'original_components': orig_components
        })
        
        # Load features if needed
        if fusion_enabled:
            feature = np.load(feature_path)
            print(f"[DATA] Loaded features: {feature.shape}")
        
        ########## Graph Fusion ##########
        fusion_time = 0
        if fusion_enabled:
            print("=" * 40 + " GRAPH FUSION " + "=" * 40)
            fusion_start = time.time()
            
            if method == "simple":
                mapping = sim_coarse_fusion(laplacian)
            elif method == "lamg":
                fusion_input_path = f"dataset/{dataset}/{dataset}.mtx"
                os.system(f'./run_coarsening.sh ~/matlab/R2018a {fusion_input_path} 12 f {reduce_results}')
                mapping = mtx2matrix(mapping_path)
            elif method == "cmg":
                mapping = cmg_fusion_mapping(laplacian, cmg_params['k'], cmg_params['d'], cmg_params['threshold'])
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Create feature graph and fuse
            feats_laplacian = feats2graph(feature, 2, mapping)  # num_neighs=2
            laplacian = laplacian + feats_laplacian
            
            fusion_time = time.time() - fusion_start
            
            # Fused graph statistics
            fused_nodes, fused_edges, fused_components = get_graph_stats(laplacian, "Fused")
            result.update({
                'fused_nodes': fused_nodes,
                'fused_edges': fused_edges,
                'fused_components': fused_components,
                'fusion_time': fusion_time
            })
        else:
            result.update({
                'fused_nodes': orig_nodes,  # No fusion, same as original
                'fused_edges': orig_edges,
                'fused_components': orig_components,
                'fusion_time': 0
            })
        
        ########## Graph Reduction ##########
        print("=" * 40 + " GRAPH REDUCTION " + "=" * 40)
        reduce_start = time.time()
        
        if method == "simple":
            level = level_or_ratio  # For simple, this is the level
            G, projections, laplacians, actual_level = sim_coarse(laplacian, level)
            
        elif method == "lamg":
            # For LAMG, level_or_ratio is the reduce_ratio parameter
            reduce_ratio = level_or_ratio
            
            # Determine input path based on fusion
            if fusion_enabled:
                coarsen_input_path = f"dataset/{dataset}/fused_{dataset}.mtx"
                # Save fused graph for LAMG
                file = open(coarsen_input_path, "wb")
                mmwrite(coarsen_input_path, laplacian)
                file.close()
            else:
                coarsen_input_path = f"dataset/{dataset}/{dataset}.mtx"
            
            # Run LAMG coarsening with reduce_ratio
            os.system(f'./run_coarsening.sh ~/matlab/R2018a {coarsen_input_path} {reduce_ratio} n {reduce_results}')
            
            # Read LAMG results
            reduce_time_file = read_time(f"{reduce_results}CPUtime.txt")
            G = mtx2graph(f"{reduce_results}Gs.mtx")
            actual_level = read_levels(f"{reduce_results}NumLevels.txt")
            projections, laplacians = construct_proj_laplacian(laplacian, actual_level, reduce_results)
            
        elif method == "cmg":
            level = level_or_ratio  # For CMG, this is the level
            G, projections, laplacians, actual_level = cmg_coarse(
                laplacian, level, cmg_params['k'], cmg_params['d'], cmg_params['threshold']
            )
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        reduce_time = time.time() - reduce_start
        
        # Final graph statistics
        final_nodes, final_edges, final_components = get_graph_stats(G, "Final")
        reduction_ratio = result['fused_nodes'] / final_nodes if final_nodes > 0 else float('inf')
        
        result.update({
            'final_nodes': final_nodes,
            'final_edges': final_edges,
            'final_components': final_components,
            'actual_level': actual_level,
            'reduction_ratio': reduction_ratio,
            'reduce_time': reduce_time
        })
        
        ########## Graph Embedding ##########
        print("=" * 40 + " GRAPH EMBEDDING " + "=" * 40)
        embed_start = time.time()
        
        # Use DeepWalk for embedding
        from embed_methods.deepwalk.deepwalk import deepwalk
        embeddings = deepwalk(G)
        
        embed_time = time.time() - embed_start
        result['embed_time'] = embed_time
        
        ########## Refinement ##########
        print("=" * 40 + " REFINEMENT " + "=" * 40)
        refine_start = time.time()
        
        # Refine embeddings back to original size
        for i in reversed(range(actual_level)):
            embeddings = projections[i] @ embeddings
            filter_ = smooth_filter(laplacians[i], 0.1)  # lda=0.1
            embeddings = filter_ @ (filter_ @ embeddings)
        
        refine_time = time.time() - refine_start
        result['refine_time'] = refine_time
        
        ########## Save and Evaluate ##########
        print("=" * 40 + " EVALUATION " + "=" * 40)
        
        # Save embeddings
        np.save(embed_path, embeddings)
        
        # Run logistic regression evaluation and capture printed output
        try:
            # Capture stdout to extract accuracy from printed output
            from io import StringIO
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            # Run lr() function (it prints the accuracy)
            lr(f"dataset/{dataset}/", embed_path, dataset)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Extract accuracy from captured output
            output = captured_output.getvalue()
            print(output, end='')  # Print the captured output for visibility
            
            # Parse the accuracy from "Test Accuracy:  0.769"
            accuracy = 0.0
            for line in output.split('\n'):
                if 'Test Accuracy:' in line:
                    try:
                        accuracy = float(line.split('Test Accuracy:')[1].strip())
                        break
                    except (ValueError, IndexError):
                        continue
            
            result['accuracy'] = accuracy
            print(f"[CAPTURED] Extracted accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            result['accuracy'] = 0.0
        
        # Total time
        total_time = time.time() - total_start_time
        result['total_time'] = total_time
        result['status'] = 'completed'
        
        print(f"\n[RESULT] Accuracy: {accuracy:.4f}")
        print(f"[RESULT] Total Time: {total_time:.3f}s")
        print(f"[RESULT] Reduction: {result['original_nodes']} → {result['final_nodes']} ({reduction_ratio:.2f}x)")
        
        return result
        
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {e}")
        result['status'] = 'failed'
        result['error'] = str(e)
        raise e  # Re-raise to stop testing as requested


def main():
    """Run comprehensive GraphZoom testing"""
    
    print("🚀 Starting Comprehensive GraphZoom Testing")
    print("=" * 80)
    
    # Test configurations
    datasets = ['cora', 'citeseer', 'pubmed']
    methods_config = {
        'cmg': {'levels': [1, 2, 3], 'params': {'k': 10, 'd': 15, 'threshold': 0.1}},
        'lamg': {'reduce_ratios': [2, 3, 6], 'params': None},  # These are reduce_ratios, not levels
        'simple': {'levels': [1, 2, 3], 'params': None}
    }
    fusion_options = [True, False]
    
    # Calculate total experiments
    total_experiments = 0
    for dataset in datasets:
        for method, config in methods_config.items():
            for fusion in fusion_options:
                if method == 'lamg':
                    total_experiments += len(config['reduce_ratios'])
                else:
                    total_experiments += len(config['levels'])
    
    print(f"📊 Total experiments planned: {total_experiments}")
    print(f"📁 Results will be saved in: results/")
    print("=" * 80)
    
    # Prepare results storage
    all_results = []
    experiment_count = 0
    
    # Main testing loop
    for dataset in datasets:
        print(f"\n🔬 TESTING DATASET: {dataset.upper()}")
        
        for method, config in methods_config.items():
            print(f"\n📈 METHOD: {method.upper()}")
            
            for fusion in fusion_options:
                fusion_label = "WITH" if fusion else "WITHOUT"
                print(f"\n🔗 FUSION: {fusion_label}")
                
                if method == 'lamg':
                    param_list = config['reduce_ratios']
                    param_name = 'reduce_ratio'
                else:
                    param_list = config['levels'] 
                    param_name = 'level'
                
                for param_value in param_list:
                    experiment_count += 1
                    print(f"\n[{experiment_count}/{total_experiments}] Testing {method} {param_name}={param_value}")
                    
                    try:
                        result = run_graphzoom_experiment(
                            dataset=dataset,
                            method=method, 
                            fusion_enabled=fusion,
                            level_or_ratio=param_value,
                            param_type=param_name,
                            cmg_params=config['params']
                        )
                        all_results.append(result)
                        
                        print(f"✅ Experiment {experiment_count} completed successfully")
                        
                    except Exception as e:
                        print(f"❌ Experiment {experiment_count} FAILED!")
                        print(f"Error: {e}")
                        print("🛑 Stopping tests as requested")
                        
                        # Save partial results before exiting
                        if all_results:
                            df = pd.DataFrame(all_results)
                            df.to_csv('results/partial_results.csv', index=False)
                            print(f"💾 Saved {len(all_results)} partial results to results/partial_results.csv")
                        
                        return 1
    
    # Save final results
    print(f"\n🎉 ALL EXPERIMENTS COMPLETED! ({total_experiments} total)")
    
    df = pd.DataFrame(all_results)
    
    # Save comprehensive results
    results_file = 'results/comprehensive_graphzoom_results.csv'
    df.to_csv(results_file, index=False)
    print(f"💾 Saved all results to: {results_file}")
    
    # Display summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    print("=" * 50)
    
    # Group by method and fusion
    summary = df.groupby(['method', 'fusion']).agg({
        'accuracy': ['mean', 'std'],
        'reduction_ratio': ['mean', 'std'], 
        'total_time': ['mean', 'std']
    }).round(4)
    
    print(summary)
    
    # Best results per dataset
    print(f"\n🏆 BEST ACCURACY PER DATASET:")
    print("=" * 50)
    
    for dataset in datasets:
        dataset_results = df[df['dataset'] == dataset]
        best_idx = dataset_results['accuracy'].idxmax()
        best = dataset_results.loc[best_idx]
        
        print(f"{dataset}: {best['accuracy']:.4f} ({best['method']}, fusion={best['fusion']}, " + 
              (f"level={best['level']}" if best['level'] is not None else f"reduce_ratio={best['reduce_ratio_param']}"))
    
    print(f"\n✨ Testing completed! Check {results_file} for detailed results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())