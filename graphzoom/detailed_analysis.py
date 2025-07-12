#!/usr/bin/env python3
"""
Detailed analysis script to answer specific research questions about CMG-GraphZoom
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

def load_comprehensive_results():
    """Load the comprehensive results with detailed info extraction"""
    possible_files = [
        'results/accuracy_results/all_results.csv',
        'results/all_results.csv',
        'results/quick_results.csv'
    ]
    
    for file_path in possible_files:
        if Path(file_path).exists():
            print(f"📁 Loading results from: {file_path}")
            df = pd.read_csv(file_path)
            print(f"✅ Loaded {len(df)} experiments")
            return df, file_path
    
    print("❌ No results file found!")
    return None, None

def extract_log_details(experiment_name):
    """Extract detailed information from log files"""
    log_path = f"results/logs/{experiment_name}.log"
    
    details = {
        'original_nodes': None,
        'coarsened_nodes': None,
        'original_edges': None,
        'coarsened_edges': None,
        'compression_ratio': None,
        'fusion_nodes': None,
        'conductance': None,
        'lambda_critical': None
    }
    
    if not Path(log_path).exists():
        return details
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Extract CMG clustering information
        cmg_match = re.search(r'CMG found (\d+) clusters', content)
        if cmg_match:
            details['coarsened_nodes'] = int(cmg_match.group(1))
        
        # Extract original graph size from debug output
        cluster_matches = re.findall(r'Cluster \d+: \d+ nodes \[([^\]]+)\]', content)
        if cluster_matches:
            # Find the highest node ID to estimate original graph size
            max_node = 0
            for match in cluster_matches:
                nodes = [int(x.strip()) for x in match.split(',')]
                max_node = max(max_node, max(nodes))
            details['original_nodes'] = max_node + 1
        
        # Extract edges information
        edges_match = re.search(r'Coarsened to (\d+) nodes, (\d+) edges', content)
        if edges_match:
            details['coarsened_nodes'] = int(edges_match.group(1))
            details['coarsened_edges'] = int(edges_match.group(2))
        
        # Extract final graph info
        final_match = re.search(r'Final graph: (\d+) nodes, (\d+) edges', content)
        if final_match:
            details['coarsened_nodes'] = int(final_match.group(1))
            details['coarsened_edges'] = int(final_match.group(2))
        
        # Extract conductance
        conductance_match = re.search(r'Average unweighted conductance - Standard: ([\d.]+)', content)
        if conductance_match:
            details['conductance'] = float(conductance_match.group(1))
        
        # Extract lambda critical
        lambda_match = re.search(r'λ_critical ≈ ([\d.]+)', content)
        if lambda_match:
            details['lambda_critical'] = float(lambda_match.group(1))
        
        # Calculate compression ratio
        if details['original_nodes'] and details['coarsened_nodes']:
            details['compression_ratio'] = details['original_nodes'] / details['coarsened_nodes']
    
    except Exception as e:
        print(f"⚠️ Error reading {log_path}: {e}")
    
    return details

def analyze_graph_compression(df):
    """Analyze graph compression ratios"""
    print("\n" + "=" * 60)
    print("📊 GRAPH COMPRESSION ANALYSIS")
    print("=" * 60)
    
    # Known original graph sizes
    original_sizes = {
        'cora': {'nodes': 2708, 'edges': 5429},
        'citeseer': {'nodes': 3327, 'edges': 4732}, 
        'pubmed': {'nodes': 19717, 'edges': 44338}
    }
    
    compression_results = []
    
    for _, row in df.iterrows():
        if 'cmg' not in row['experiment']:
            continue
            
        details = extract_log_details(row['experiment'])
        dataset = row['dataset']
        
        if dataset in original_sizes:
            orig_nodes = original_sizes[dataset]['nodes']
            orig_edges = original_sizes[dataset]['edges']
            
            # Use extracted or estimated coarsened size
            coarse_nodes = details.get('coarsened_nodes')
            coarse_edges = details.get('coarsened_edges')
            
            if coarse_nodes:
                compression_ratio = orig_nodes / coarse_nodes
                compression_results.append({
                    'experiment': row['experiment'],
                    'dataset': dataset,
                    'original_nodes': orig_nodes,
                    'coarsened_nodes': coarse_nodes,
                    'compression_ratio': compression_ratio,
                    'original_edges': orig_edges,
                    'coarsened_edges': coarse_edges,
                    'conductance': details.get('conductance'),
                    'lambda_critical': details.get('lambda_critical'),
                    'accuracy': row['accuracy']
                })
    
    if compression_results:
        comp_df = pd.DataFrame(compression_results)
        
        print("\n📏 COMPRESSION RATIOS:")
        print(f"{'Dataset':<10} {'Original':<8} {'Coarsened':<10} {'Ratio':<6} {'Accuracy':<8}")
        print("-" * 50)
        
        for dataset in comp_df['dataset'].unique():
            dataset_results = comp_df[comp_df['dataset'] == dataset]
            avg_ratio = dataset_results['compression_ratio'].mean()
            avg_acc = dataset_results['accuracy'].mean()
            orig_nodes = dataset_results['original_nodes'].iloc[0]
            avg_coarse = dataset_results['coarsened_nodes'].mean()
            
            print(f"{dataset:<10} {orig_nodes:<8} {avg_coarse:<10.0f} {avg_ratio:<6.1f}x {avg_acc:<8.3f}")
        
        print(f"\n🏆 BEST COMPRESSION:")
        best_comp = comp_df.loc[comp_df['compression_ratio'].idxmax()]
        print(f"   {best_comp['dataset']}: {best_comp['compression_ratio']:.1f}x compression")
        print(f"   ({best_comp['original_nodes']} → {best_comp['coarsened_nodes']} nodes)")
        print(f"   Accuracy: {best_comp['accuracy']:.3f}")
        
        return comp_df
    else:
        print("❌ No compression data found in logs")
        return None

def analyze_parameters(df):
    """Comprehensive parameter analysis"""
    print("\n" + "=" * 60)
    print("🔬 COMPREHENSIVE PARAMETER ANALYSIS")
    print("=" * 60)
    
    cmg_df = df[df['coarsening'] == 'cmg'].copy()
    
    # Extract parameters from experiment names
    def extract_param(exp_name, param):
        patterns = [
            f'{param}(\\d+(?:\\.\\d+)?)',
            f'cmg_{param}(\\d+(?:\\.\\d+)?)',
            f'--cmg_{param}\\s+(\\d+(?:\\.\\d+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, str(exp_name))
            if match:
                return float(match.group(1))
        return None
    
    # Add parameter columns
    cmg_df['k_param'] = cmg_df['experiment'].apply(lambda x: extract_param(x, 'k'))
    cmg_df['d_param'] = cmg_df['experiment'].apply(lambda x: extract_param(x, 'd'))
    cmg_df['thresh_param'] = cmg_df['experiment'].apply(lambda x: extract_param(x, 'thresh'))
    
    # K Parameter Analysis
    k_data = cmg_df[cmg_df['k_param'].notna()]
    if not k_data.empty:
        print("\n📊 K PARAMETER (Filter Order) ANALYSIS:")
        k_summary = k_data.groupby('k_param').agg({
            'accuracy': ['mean', 'std', 'count', 'max'],
            'total_time': 'mean'
        }).round(4)
        
        print(f"{'K':<4} {'Mean Acc':<8} {'Std':<6} {'Max Acc':<8} {'Count':<6} {'Avg Time':<8}")
        print("-" * 45)
        
        for k in sorted(k_data['k_param'].unique()):
            k_rows = k_data[k_data['k_param'] == k]
            mean_acc = k_rows['accuracy'].mean()
            std_acc = k_rows['accuracy'].std()
            max_acc = k_rows['accuracy'].max()
            count = len(k_rows)
            avg_time = k_rows['total_time'].mean()
            
            print(f"{k:<4.0f} {mean_acc:<8.3f} {std_acc or 0:<6.3f} {max_acc:<8.3f} {count:<6} {avg_time:<8.1f}")
        
        best_k = k_data.loc[k_data['accuracy'].idxmax()]
        print(f"\n🏆 BEST K: {best_k['k_param']:.0f} (Accuracy: {best_k['accuracy']:.3f})")
    
    # D Parameter Analysis  
    d_data = cmg_df[cmg_df['d_param'].notna()]
    if not d_data.empty:
        print("\n📊 D PARAMETER (Embedding Dimension) ANALYSIS:")
        print(f"{'D':<4} {'Mean Acc':<8} {'Std':<6} {'Max Acc':<8} {'Count':<6} {'Avg Time':<8}")
        print("-" * 45)
        
        for d in sorted(d_data['d_param'].unique()):
            d_rows = d_data[d_data['d_param'] == d]
            mean_acc = d_rows['accuracy'].mean()
            std_acc = d_rows['accuracy'].std()
            max_acc = d_rows['accuracy'].max()
            count = len(d_rows)
            avg_time = d_rows['total_time'].mean()
            
            print(f"{d:<4.0f} {mean_acc:<8.3f} {std_acc or 0:<6.3f} {max_acc:<8.3f} {count:<6} {avg_time:<8.1f}")
        
        best_d = d_data.loc[d_data['accuracy'].idxmax()]
        print(f"\n🏆 BEST D: {best_d['d_param']:.0f} (Accuracy: {best_d['accuracy']:.3f})")
    
    # Threshold Analysis
    thresh_data = cmg_df[cmg_df['thresh_param'].notna()]
    if not thresh_data.empty:
        print("\n📊 THRESHOLD PARAMETER ANALYSIS:")
        print(f"{'Thresh':<8} {'Mean Acc':<8} {'Std':<6} {'Max Acc':<8} {'Count':<6}")
        print("-" * 42)
        
        for thresh in sorted(thresh_data['thresh_param'].unique()):
            thresh_rows = thresh_data[thresh_data['thresh_param'] == thresh]
            mean_acc = thresh_rows['accuracy'].mean()
            std_acc = thresh_rows['accuracy'].std()
            max_acc = thresh_rows['accuracy'].max()
            count = len(thresh_rows)
            
            print(f"{thresh:<8.2f} {mean_acc:<8.3f} {std_acc or 0:<6.3f} {max_acc:<8.3f} {count:<6}")
        
        best_thresh = thresh_data.loc[thresh_data['accuracy'].idxmax()]
        print(f"\n🏆 BEST THRESHOLD: {best_thresh['thresh_param']:.2f} (Accuracy: {best_thresh['accuracy']:.3f})")
    
    # Overall best parameters
    if not cmg_df.empty:
        overall_best = cmg_df.loc[cmg_df['accuracy'].idxmax()]
        print(f"\n🎯 OVERALL BEST CMG CONFIGURATION:")
        print(f"   Experiment: {overall_best['experiment']}")
        print(f"   Dataset: {overall_best['dataset']}")
        print(f"   Embedding: {overall_best['embedding']}")
        print(f"   Accuracy: {overall_best['accuracy']:.3f}")
        print(f"   Time: {overall_best['total_time']:.1f}s")
        if overall_best['k_param']:
            print(f"   K: {overall_best['k_param']:.0f}")
        if overall_best['d_param']:
            print(f"   D: {overall_best['d_param']:.0f}")
        if overall_best['thresh_param']:
            print(f"   Threshold: {overall_best['thresh_param']:.2f}")

def comprehensive_accuracy_time_comparison(df):
    """Detailed accuracy and time comparison"""
    print("\n" + "=" * 60)
    print("📈 COMPREHENSIVE ACCURACY & TIME COMPARISON")
    print("=" * 60)
    
    # Group by dataset and embedding
    results = []
    
    for dataset in df['dataset'].unique():
        for embedding in df['embedding'].unique():
            simple_rows = df[
                (df['dataset'] == dataset) & 
                (df['embedding'] == embedding) & 
                (df['coarsening'] == 'simple')
            ]
            
            cmg_rows = df[
                (df['dataset'] == dataset) & 
                (df['embedding'] == embedding) & 
                (df['coarsening'] == 'cmg')
            ]
            
            if not simple_rows.empty and not cmg_rows.empty:
                simple_acc = simple_rows['accuracy'].mean()
                cmg_acc_mean = cmg_rows['accuracy'].mean()
                cmg_acc_max = cmg_rows['accuracy'].max()
                cmg_acc_std = cmg_rows['accuracy'].std()
                
                simple_time = simple_rows['total_time'].mean()
                cmg_time_mean = cmg_rows['total_time'].mean()
                
                improvement_mean = cmg_acc_mean - simple_acc
                improvement_max = cmg_acc_max - simple_acc
                speedup = simple_time / cmg_time_mean if cmg_time_mean > 0 else np.nan
                
                results.append({
                    'dataset': dataset,
                    'embedding': embedding,
                    'simple_acc': simple_acc,
                    'cmg_acc_mean': cmg_acc_mean,
                    'cmg_acc_max': cmg_acc_max,
                    'cmg_acc_std': cmg_acc_std or 0,
                    'improvement_mean': improvement_mean,
                    'improvement_max': improvement_max,
                    'simple_time': simple_time,
                    'cmg_time_mean': cmg_time_mean,
                    'speedup': speedup,
                    'cmg_experiments': len(cmg_rows)
                })
    
    if results:
        results_df = pd.DataFrame(results)
        
        print("\n📊 ACCURACY COMPARISON (Simple vs CMG):")
        print(f"{'Dataset':<10} {'Embedding':<10} {'Simple':<8} {'CMG Mean':<8} {'CMG Max':<8} {'CMG Std':<8} {'Best Δ':<8}")
        print("-" * 70)
        
        for _, row in results_df.iterrows():
            print(f"{row['dataset']:<10} {row['embedding']:<10} {row['simple_acc']:<8.3f} "
                  f"{row['cmg_acc_mean']:<8.3f} {row['cmg_acc_max']:<8.3f} {row['cmg_acc_std']:<8.3f} "
                  f"{row['improvement_max']:<+8.3f}")
        
        print("\n⚡ TIME & SPEEDUP COMPARISON:")
        print(f"{'Dataset':<10} {'Embedding':<10} {'Simple(s)':<10} {'CMG(s)':<10} {'Speedup':<8} {'#Exp':<5}")
        print("-" * 60)
        
        for _, row in results_df.iterrows():
            print(f"{row['dataset']:<10} {row['embedding']:<10} {row['simple_time']:<10.1f} "
                  f"{row['cmg_time_mean']:<10.1f} {row['speedup']:<8.1f}x {row['cmg_experiments']:<5}")
        
        # Summary statistics
        print(f"\n🎯 OVERALL PERFORMANCE SUMMARY:")
        avg_improvement = results_df['improvement_mean'].mean()
        max_improvement = results_df['improvement_max'].max()
        win_rate = (results_df['improvement_mean'] > 0).mean() * 100
        avg_speedup = results_df['speedup'].mean()
        
        print(f"   Average Improvement: {avg_improvement:+.3f}")
        print(f"   Best Improvement: {max_improvement:+.3f}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Average Speedup: {avg_speedup:.1f}x")
        
        return results_df
    
    return None

def main():
    """Main detailed analysis"""
    print("🔍 DETAILED CMG-GRAPHZOOM ANALYSIS")
    print("=" * 60)
    
    df, file_path = load_comprehensive_results()
    if df is None:
        return
    
    # Convert numeric columns
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df['total_time'] = pd.to_numeric(df['total_time'], errors='coerce')
    df = df.dropna(subset=['accuracy'])
    
    print(f"\n📋 EXPERIMENT OVERVIEW:")
    print(f"   Total experiments: {len(df)}")
    print(f"   Datasets: {list(df['dataset'].unique())}")
    print(f"   Methods: {list(df['coarsening'].unique())}")
    print(f"   Embeddings: {list(df['embedding'].unique())}")
    print(f"   Source file: {file_path}")
    
    # Run all analyses
    analyze_graph_compression(df)
    comprehensive_accuracy_time_comparison(df)
    analyze_parameters(df)
    
    print(f"\n✅ DETAILED ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()
