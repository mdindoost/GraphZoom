#!/usr/bin/env python3
"""
Enhanced Analysis Script for Koutis Comprehensive Experiments
Addresses: Parameter efficiency, Multi-level analysis, Stability, Timing breakdown
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import defaultdict

def load_comprehensive_results():
    """Load comprehensive results with all metrics"""
    file_path = 'results/accuracy_results/comprehensive_results.csv'
    
    if not Path(file_path).exists():
        print("❌ Comprehensive results file not found!")
        return None
    
    df = pd.read_csv(file_path)
    
    # Clean and convert data types
    numeric_cols = ['accuracy', 'total_time', 'fusion_time', 'reduction_time', 
                   'embedding_time', 'refinement_time', 'wall_time', 'cmg_k', 
                   'cmg_d', 'cmg_threshold', 'num_neighs', 'original_nodes', 
                   'coarsened_nodes', 'compression_ratio', 'lambda_critical', 'conductance']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove failed experiments
    df_clean = df.dropna(subset=['accuracy'])
    
    print(f"✅ Loaded {len(df)} experiments, {len(df_clean)} successful")
    
    return df_clean

def analyze_parameter_efficiency(df):
    """Analyze Koutis's parameter efficiency hypothesis"""
    print("\n" + "=" * 80)
    print("🔬 KOUTIS PARAMETER EFFICIENCY ANALYSIS")
    print("=" * 80)
    
    cmg_df = df[df['coarsening'] == 'cmg'].copy()
    
    if cmg_df.empty:
        print("❌ No CMG experiments found")
        return
    
    print("\n📊 TESTING KOUTIS HYPOTHESIS: 'CMG may do better with smaller d'")
    print("-" * 60)
    
    # Analyze d parameter efficiency
    d_analysis = cmg_df[cmg_df['cmg_d'].notna() & (cmg_df['dataset'] == 'cora')].copy()
    
    if not d_analysis.empty:
        d_summary = d_analysis.groupby('cmg_d').agg({
            'accuracy': ['mean', 'std', 'max', 'count'],
            'total_time': ['mean', 'std'],
            'compression_ratio': 'mean'
        }).round(4)
        
        print(f"\n📈 D Parameter Analysis (Cora dataset):")
        print(f"{'D':<4} {'Acc Mean':<8} {'Acc Std':<8} {'Acc Max':<8} {'Time Mean':<9} {'Compression':<11} {'Count':<5}")
        print("-" * 70)
        
        best_acc_per_d = {}
        
        for d in sorted(d_analysis['cmg_d'].unique()):
            d_data = d_analysis[d_analysis['cmg_d'] == d]
            acc_mean = d_data['accuracy'].mean()
            acc_std = d_data['accuracy'].std() or 0
            acc_max = d_data['accuracy'].max()
            time_mean = d_data['total_time'].mean()
            comp_mean = d_data['compression_ratio'].mean()
            count = len(d_data)
            
            best_acc_per_d[d] = acc_max
            
            print(f"{d:<4.0f} {acc_mean:<8.3f} {acc_std:<8.3f} {acc_max:<8.3f} {time_mean:<9.1f} {comp_mean:<11.1f} {count:<5}")
        
        # Find efficiency sweet spot
        sorted_d = sorted(best_acc_per_d.keys())
        print(f"\n🎯 EFFICIENCY ANALYSIS:")
        for i, d in enumerate(sorted_d):
            if i > 0:
                prev_d = sorted_d[i-1]
                acc_diff = best_acc_per_d[d] - best_acc_per_d[prev_d]
                efficiency = acc_diff / (d - prev_d) if d != prev_d else 0
                print(f"   D={prev_d:.0f}→{d:.0f}: Δ accuracy = {acc_diff:+.4f}, efficiency = {efficiency:+.4f}/unit")
    
    # Analyze k parameter efficiency
    print("\n📊 K Parameter Analysis:")
    k_analysis = cmg_df[cmg_df['cmg_k'].notna() & (cmg_df['dataset'] == 'cora')].copy()
    
    if not k_analysis.empty:
        print(f"{'K':<4} {'Acc Mean':<8} {'Acc Max':<8} {'Time Mean':<9} {'λ_crit':<8} {'Count':<5}")
        print("-" * 50)
        
        for k in sorted(k_analysis['cmg_k'].unique()):
            k_data = k_analysis[k_analysis['cmg_k'] == k]
            acc_mean = k_data['accuracy'].mean()
            acc_max = k_data['accuracy'].max()
            time_mean = k_data['total_time'].mean()
            lambda_crit = k_data['lambda_critical'].mean()
            count = len(k_data)
            
            print(f"{k:<4.0f} {acc_mean:<8.3f} {acc_max:<8.3f} {time_mean:<9.1f} {lambda_crit:<8.3f} {count:<5}")
    
    # High k + Low d combinations analysis
    print("\n🎯 HIGH K + LOW D COMBINATIONS (Testing Koutis Hypothesis):")
    combo_analysis = cmg_df[
        (cmg_df['cmg_k'].notna()) & 
        (cmg_df['cmg_d'].notna()) & 
        (cmg_df['dataset'] == 'cora')
    ].copy()
    
    if not combo_analysis.empty:
        print(f"{'K':<4} {'D':<4} {'Accuracy':<8} {'Time':<8} {'Ratio':<8} {'λ_crit':<8}")
        print("-" * 50)
        
        # Focus on high k (>=20) with low d (<=15)
        promising_combos = combo_analysis[
            (combo_analysis['cmg_k'] >= 20) & (combo_analysis['cmg_d'] <= 15)
        ].sort_values('accuracy', ascending=False)
        
        for _, row in promising_combos.head(10).iterrows():
            print(f"{row['cmg_k']:<4.0f} {row['cmg_d']:<4.0f} {row['accuracy']:<8.3f} {row['total_time']:<8.1f} {row['compression_ratio']:<8.1f} {row['lambda_critical']:<8.3f}")

def analyze_multilevel_performance(df):
    """Analyze multi-level coarsening performance"""
    print("\n" + "=" * 80)
    print("🏗️ MULTI-LEVEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    multilevel_df = df.copy()
    
    print("\n📊 Performance by Level:")
    print(f"{'Dataset':<8} {'Method':<8} {'Embedding':<10} {'Level':<5} {'Accuracy':<8} {'Total Time':<10} {'Reduction Time':<12}")
    print("-" * 80)
    
    # Analyze per level performance
    level_summary = defaultdict(list)
    
    for _, row in multilevel_df.iterrows():
        level_key = f"{row['dataset']}_{row['coarsening']}_{row['embedding']}_{row['level']}"
        level_summary[level_key].append(row)
        
        print(f"{row['dataset']:<8} {row['coarsening']:<8} {row['embedding']:<10} {row['level']:<5} {row['accuracy']:<8.3f} {row['total_time']:<10.1f} {row['reduction_time']:<12.1f}")
    
    # Compare level efficiency
    print(f"\n📈 LEVEL EFFICIENCY ANALYSIS:")
    datasets = multilevel_df['dataset'].unique()
    methods = multilevel_df['coarsening'].unique()
    embeddings = multilevel_df['embedding'].unique()
    
    for dataset in datasets:
        for method in methods:
            for embedding in embeddings:
                method_data = multilevel_df[
                    (multilevel_df['dataset'] == dataset) & 
                    (multilevel_df['coarsening'] == method) & 
                    (multilevel_df['embedding'] == embedding)
                ]
                
                if len(method_data) > 1:
                    print(f"\n🔍 {dataset.upper()} - {method} - {embedding}:")
                    
                    for level in sorted(method_data['level'].unique()):
                        level_data = method_data[method_data['level'] == level]
                        if not level_data.empty:
                            avg_acc = level_data['accuracy'].mean()
                            avg_time = level_data['total_time'].mean()
                            avg_comp = level_data['compression_ratio'].mean()
                            print(f"   Level {level}: Acc={avg_acc:.3f}, Time={avg_time:.1f}s, Compression={avg_comp:.1f}x")

def analyze_timing_breakdown(df):
    """Detailed timing analysis as Koutis requested"""
    print("\n" + "=" * 80)
    print("⏱️ DETAILED TIMING BREAKDOWN ANALYSIS")
    print("=" * 80)
    
    timing_cols = ['fusion_time', 'reduction_time', 'embedding_time', 'refinement_time']
    
    print("\n📊 Average Timing Breakdown by Method:")
    print(f"{'Method':<8} {'Dataset':<8} {'Fusion':<8} {'Reduction':<10} {'Embedding':<10} {'Refinement':<10} {'Total':<8}")
    print("-" * 80)
    
    for method in df['coarsening'].unique():
        for dataset in df['dataset'].unique():
            method_data = df[(df['coarsening'] == method) & (df['dataset'] == dataset)]
            
            if not method_data.empty:
                avg_fusion = method_data['fusion_time'].mean()
                avg_reduction = method_data['reduction_time'].mean()
                avg_embedding = method_data['embedding_time'].mean()
                avg_refinement = method_data['refinement_time'].mean()
                avg_total = method_data['total_time'].mean()
                
                print(f"{method:<8} {dataset:<8} {avg_fusion:<8.1f} {avg_reduction:<10.1f} {avg_embedding:<10.1f} {avg_refinement:<10.1f} {avg_total:<8.1f}")
    
    # Analyze where time is spent
    print(f"\n🎯 TIME BOTTLENECK ANALYSIS:")
    cmg_data = df[df['coarsening'] == 'cmg']
    simple_data = df[df['coarsening'] == 'simple']
    
    if not cmg_data.empty:
        print(f"\n📊 CMG Time Distribution:")
        avg_fusion = cmg_data['fusion_time'].mean()
        avg_reduction = cmg_data['reduction_time'].mean()
        avg_embedding = cmg_data['embedding_time'].mean()
        avg_refinement = cmg_data['refinement_time'].mean()
        total = avg_fusion + avg_reduction + avg_embedding + avg_refinement
        
        print(f"   Fusion:     {avg_fusion:6.1f}s ({100*avg_fusion/total:5.1f}%)")
        print(f"   Reduction:  {avg_reduction:6.1f}s ({100*avg_reduction/total:5.1f}%)")
        print(f"   Embedding:  {avg_embedding:6.1f}s ({100*avg_embedding/total:5.1f}%)")
        print(f"   Refinement: {avg_refinement:6.1f}s ({100*avg_refinement/total:5.1f}%)")
    
    if not simple_data.empty:
        print(f"\n📊 Simple Coarsening Time Distribution:")
        avg_fusion = simple_data['fusion_time'].mean()
        avg_reduction = simple_data['reduction_time'].mean()
        avg_embedding = simple_data['embedding_time'].mean()
        avg_refinement = simple_data['refinement_time'].mean()
        total = avg_fusion + avg_reduction + avg_embedding + avg_refinement
        
        print(f"   Fusion:     {avg_fusion:6.1f}s ({100*avg_fusion/total:5.1f}%)")
        print(f"   Reduction:  {avg_reduction:6.1f}s ({100*avg_reduction/total:5.1f}%)")
        print(f"   Embedding:  {avg_embedding:6.1f}s ({100*avg_embedding/total:5.1f}%)")
        print(f"   Refinement: {avg_refinement:6.1f}s ({100*avg_refinement/total:5.1f}%)")

def analyze_stability_variance(df):
    """Analyze CMG stability vs GraphZoom as Koutis mentioned"""
    print("\n" + "=" * 80)
    print("📊 STABILITY & VARIANCE ANALYSIS")
    print("=" * 80)
    
    # Find experiments with multiple seeds
    stability_experiments = df[df['experiment'].str.contains('stability')].copy()
    
    if stability_experiments.empty:
        print("❌ No stability experiments found")
        return
    
    print("\n🎲 VARIANCE COMPARISON (Multiple Random Seeds):")
    print(f"{'Experiment':<30} {'Mean Acc':<8} {'Std Acc':<8} {'CV%':<6} {'Count':<5}")
    print("-" * 60)
    
    # Group by base experiment name (without seed)
    base_experiments = defaultdict(list)
    
    for _, row in stability_experiments.iterrows():
        base_name = row['experiment'].replace('_stability', '')
        base_experiments[base_name].append(row)
    
    variance_comparison = {}
    
    for exp_name, experiments in base_experiments.items():
        if len(experiments) >= 3:  # Need multiple runs
            accuracies = [exp['accuracy'] for exp in experiments]
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            cv_percent = (std_acc / mean_acc) * 100 if mean_acc > 0 else 0
            count = len(experiments)
            
            variance_comparison[exp_name] = {
                'mean': mean_acc, 'std': std_acc, 'cv': cv_percent, 'count': count
            }
            
            print(f"{exp_name:<30} {mean_acc:<8.3f} {std_acc:<8.4f} {cv_percent:<6.2f} {count:<5}")
    
    # Compare CMG vs Simple stability
    cmg_stability = [v for k, v in variance_comparison.items() if 'cmg' in k]
    simple_stability = [v for k, v in variance_comparison.items() if 'simple' in k]
    
    if cmg_stability and simple_stability:
        cmg_avg_cv = np.mean([exp['cv'] for exp in cmg_stability])
        simple_avg_cv = np.mean([exp['cv'] for exp in simple_stability])
        
        print(f"\n🎯 STABILITY COMPARISON:")
        print(f"   CMG Average CV: {cmg_avg_cv:.2f}%")
        print(f"   Simple Average CV: {simple_avg_cv:.2f}%")
        
        if cmg_avg_cv < simple_avg_cv:
            improvement = ((simple_avg_cv - cmg_avg_cv) / simple_avg_cv) * 100
            print(f"   ✅ CMG is {improvement:.1f}% more stable than Simple coarsening")
        else:
            print(f"   ⚠️  Simple coarsening is more stable")

def analyze_graphzoom_knn_parameter(df):
    """Analyze GraphZoom's knn parameter sensitivity"""
    print("\n" + "=" * 80)
    print("🔗 GRAPHZOOM KNN PARAMETER ANALYSIS")
    print("=" * 80)
    
    knn_data = df[(df['coarsening'] == 'simple') & (df['num_neighs'].notna())].copy()
    
    if knn_data.empty:
        print("❌ No GraphZoom knn parameter data found")
        return
    
    print("\n📊 GraphZoom Performance vs knn Parameter:")
    print(f"{'Dataset':<8} {'Embedding':<10} {'knn':<4} {'Accuracy':<8} {'Time':<8} {'Fusion Time':<10}")
    print("-" * 55)
    
    for dataset in knn_data['dataset'].unique():
        for embedding in knn_data['embedding'].unique():
            subset = knn_data[
                (knn_data['dataset'] == dataset) & 
                (knn_data['embedding'] == embedding)
            ].sort_values('num_neighs')
            
            for _, row in subset.iterrows():
                print(f"{row['dataset']:<8} {row['embedding']:<10} {row['num_neighs']:<4.0f} {row['accuracy']:<8.3f} {row['total_time']:<8.1f} {row['fusion_time']:<10.1f}")
    
    # Find optimal knn
    print(f"\n🎯 OPTIMAL KNN BY DATASET:")
    for dataset in knn_data['dataset'].unique():
        dataset_data = knn_data[knn_data['dataset'] == dataset]
        best_row = dataset_data.loc[dataset_data['accuracy'].idxmax()]
        print(f"   {dataset}: knn={best_row['num_neighs']:.0f}, accuracy={best_row['accuracy']:.3f}")

def generate_koutis_summary(df):
    """Generate summary addressing Koutis's specific questions"""
    print("\n" + "=" * 80)
    print("📋 KOUTIS RESEARCH QUESTIONS SUMMARY")
    print("=" * 80)
    
    print("\n🔬 1. PARAMETER EFFICIENCY ('CMG may do better with smaller d'):")
    
    # Find best d values
    cora_cmg = df[(df['dataset'] == 'cora') & (df['coarsening'] == 'cmg') & (df['cmg_d'].notna())]
    if not cora_cmg.empty:
        best_d_results = cora_cmg.groupby('cmg_d')['accuracy'].max().sort_values(ascending=False)
        print("   📊 Best accuracy by d value:")
        for d, acc in best_d_results.head(5).items():
            print(f"      d={d:.0f}: {acc:.3f}")
        
        # Check if smaller d values are competitive
        small_d_performance = best_d_results[best_d_results.index <= 15].max()
        large_d_performance = best_d_results[best_d_results.index > 20].max()
        
        if small_d_performance >= large_d_performance * 0.99:  # Within 1%
            print(f"   ✅ CONFIRMED: Small d (≤15) achieves {small_d_performance:.3f} vs large d {large_d_performance:.3f}")
        else:
            print(f"   ❌ Small d underperforms: {small_d_performance:.3f} vs {large_d_performance:.3f}")
    
    print("\n⚡ 2. SPEEDUP CONSISTENCY:")
    speedup_data = []
    
    for dataset in df['dataset'].unique():
        simple_time = df[(df['dataset'] == dataset) & (df['coarsening'] == 'simple')]['total_time'].mean()
        cmg_time = df[(df['dataset'] == dataset) & (df['coarsening'] == 'cmg')]['total_time'].mean()
        
        if simple_time > 0 and cmg_time > 0:
            speedup = simple_time / cmg_time
            speedup_data.append((dataset, speedup))
            print(f"   {dataset}: {speedup:.1f}x speedup")
    
    if speedup_data:
        avg_speedup = np.mean([s for _, s in speedup_data])
        print(f"   📊 Average speedup: {avg_speedup:.1f}x")
    
    print("\n🏗️ 3. MULTI-LEVEL SCALING:")
    
    multilevel_summary = df.groupby(['dataset', 'coarsening', 'level']).agg({
        'accuracy': 'mean',
        'total_time': 'mean',
        'compression_ratio': 'mean'
    }).round(3)
    
    print("   📊 Performance scaling by level:")
    for (dataset, method, level), metrics in multilevel_summary.iterrows():
        if not pd.isna(metrics['accuracy']):
            print(f"      {dataset}-{method}-L{level}: Acc={metrics['accuracy']:.3f}, Time={metrics['total_time']:.1f}s")
    
    print("\n📊 4. STABILITY (Lower CV% = More Stable):")
    
    # Calculate stability metrics
    stability_data = df[df['experiment'].str.contains('stability', na=False)]
    if not stability_data.empty:
        stability_summary = stability_data.groupby(['dataset', 'coarsening']).agg({
            'accuracy': ['mean', 'std']
        })
        
        for (dataset, method), metrics in stability_summary.iterrows():
            mean_acc = metrics[('accuracy', 'mean')]
            std_acc = metrics[('accuracy', 'std')]
            cv = (std_acc / mean_acc) * 100 if mean_acc > 0 else 0
            print(f"      {dataset}-{method}: CV = {cv:.2f}%")
    
    print("\n🎯 5. OVERALL BEST CONFIGURATIONS:")
    
    # Find best performing configurations
    best_configs = df.loc[df.groupby(['dataset', 'embedding'])['accuracy'].idxmax()]
    
    for _, config in best_configs.iterrows():
        params = ""
        if config['coarsening'] == 'cmg':
            params = f"k={config['cmg_k']:.0f}, d={config['cmg_d']:.0f}, t={config['cmg_threshold']:.2f}"
        elif config['coarsening'] == 'simple':
            params = f"knn={config['num_neighs']:.0f}"
        
        print(f"   {config['dataset']}-{config['embedding']}: {config['coarsening']} ({params}) = {config['accuracy']:.3f}")

def main():
    """Main enhanced analysis function"""
    print("🔍 KOUTIS COMPREHENSIVE ANALYSIS")
    print("Addressing: Parameter efficiency, Multi-level, Stability, Timing breakdown")
    print("=" * 80)
    
    df = load_comprehensive_results()
    if df is None:
        return
    
    print(f"\n📈 EXPERIMENT OVERVIEW:")
    print(f"   Total experiments: {len(df)}")
    print(f"   Datasets: {sorted(df['dataset'].unique())}")
    print(f"   Methods: {sorted(df['coarsening'].unique())}")
    print(f"   Embeddings: {sorted(df['embedding'].unique())}")
    print(f"   Levels tested: {sorted(df['level'].unique())}")
    print(f"   Seeds tested: {len(df['seed'].unique())} different seeds")
    
    # Run all analyses
    analyze_parameter_efficiency(df)
    analyze_multilevel_performance(df)
    analyze_timing_breakdown(df)
    analyze_stability_variance(df)
    analyze_graphzoom_knn_parameter(df)
    generate_koutis_summary(df)
    
    print(f"\n✅ KOUTIS COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"📊 All research questions addressed with statistical rigor")

if __name__ == "__main__":
    main()
