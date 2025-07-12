#!/usr/bin/env python3
"""
Analysis script for Koutis experiments
Focuses on: Parameter efficiency, multi-level comparison, stability analysis, timing breakdown
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import re

def load_and_clean_data(csv_path):
    """Load experimental results and clean data"""
    df = pd.read_csv(csv_path)
    
    # Convert numeric columns
    numeric_cols = ['cmg_k', 'cmg_d', 'cmg_threshold', 'knn_neighbors', 'accuracy', 
                   'total_time', 'fusion_time', 'reduction_time', 'embedding_time', 
                   'refinement_time', 'compression_ratio', 'level', 'seed']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove failed/timeout experiments
    df_clean = df[~df['accuracy'].isin(['TIMEOUT', 'FAILED'])].copy()
    df_clean = df_clean.dropna(subset=['accuracy'])
    
    print(f"✅ Loaded {len(df)} total experiments, {len(df_clean)} successful")
    
    return df_clean

def analyze_parameter_efficiency(df):
    """Analyze CMG parameter efficiency (Koutis's main hypothesis)"""
    print("\n" + "=" * 80)
    print("🎯 PARAMETER EFFICIENCY ANALYSIS (Koutis's Hypothesis)")
    print("=" * 80)
    print("Testing: 'CMG may do better than Zoom even for a somewhat smaller d'")
    
    cmg_df = df[df['coarsening'] == 'cmg'].copy()
    
    if cmg_df.empty:
        print("❌ No CMG experiments found")
        return
    
    # K Parameter Analysis
    k_data = cmg_df[cmg_df['cmg_k'].notna() & (cmg_df['dataset'] == 'cora')]
    if not k_data.empty:
        print(f"\n📊 CMG K PARAMETER ANALYSIS (Filter Order):")
        print(f"{'K':<4} {'Accuracy':<9} {'Time(s)':<8} {'Compression':<11} {'Reduction Time':<13}")
        print("-" * 50)
        
        k_summary = []
        for k in sorted(k_data['cmg_k'].unique()):
            k_rows = k_data[k_data['cmg_k'] == k]
            if not k_rows.empty:
                acc = k_rows['accuracy'].mean()
                time = k_rows['total_time'].mean()
                comp = k_rows['compression_ratio'].mean()
                red_time = k_rows['reduction_time'].mean()
                
                print(f"{k:<4.0f} {acc:<9.3f} {time:<8.1f} {comp:<11.1f}x {red_time:<13.1f}")
                k_summary.append({'k': k, 'accuracy': acc, 'time': time, 'reduction_time': red_time})
        
        if k_summary:
            k_df = pd.DataFrame(k_summary)
            best_k_acc = k_df.loc[k_df['accuracy'].idxmax()]
            best_k_time = k_df.loc[k_df['time'].idxmin()]
            print(f"\n🏆 Best K for accuracy: {best_k_acc['k']:.0f} ({best_k_acc['accuracy']:.3f})")
            print(f"⚡ Best K for speed: {best_k_time['k']:.0f} ({best_k_time['time']:.1f}s)")
    
    # D Parameter Analysis (Key hypothesis test)
    d_data = cmg_df[cmg_df['cmg_d'].notna() & (cmg_df['dataset'] == 'cora')]
    if not d_data.empty:
        print(f"\n📊 CMG D PARAMETER ANALYSIS (Embedding Dimension - KEY HYPOTHESIS):")
        print(f"{'D':<4} {'Accuracy':<9} {'Time(s)':<8} {'Compression':<11} {'Efficiency':<10}")
        print("-" * 47)
        
        d_summary = []
        for d in sorted(d_data['cmg_d'].unique()):
            d_rows = d_data[d_data['cmg_d'] == d]
            if not d_rows.empty:
                acc = d_rows['accuracy'].mean()
                time = d_rows['total_time'].mean()
                comp = d_rows['compression_ratio'].mean()
                efficiency = acc / time if time > 0 else 0  # Accuracy per second
                
                print(f"{d:<4.0f} {acc:<9.3f} {time:<8.1f} {comp:<11.1f}x {efficiency:<10.4f}")
                d_summary.append({'d': d, 'accuracy': acc, 'time': time, 'efficiency': efficiency})
        
        if d_summary:
            d_df = pd.DataFrame(d_summary)
            best_d_acc = d_df.loc[d_df['accuracy'].idxmax()]
            best_d_eff = d_df.loc[d_df['efficiency'].idxmax()]
            print(f"\n🏆 Best D for accuracy: {best_d_acc['d']:.0f} ({best_d_acc['accuracy']:.3f})")
            print(f"⚡ Best D for efficiency: {best_d_eff['d']:.0f} (acc/time: {best_d_eff['efficiency']:.4f})")
            
            # Test Koutis's hypothesis: smaller d can work well
            small_d = d_df[d_df['d'] <= 15]['accuracy'].max() if len(d_df[d_df['d'] <= 15]) > 0 else 0
            large_d = d_df[d_df['d'] >= 25]['accuracy'].max() if len(d_df[d_df['d'] >= 25]) > 0 else 0
            
            print(f"\n💡 KOUTIS HYPOTHESIS TEST:")
            print(f"   Small D (≤15): Best accuracy = {small_d:.3f}")
            print(f"   Large D (≥25): Best accuracy = {large_d:.3f}")
            if small_d >= large_d - 0.01:  # Within 1% tolerance
                print(f"   ✅ HYPOTHESIS CONFIRMED: Small D performs competitively!")
            else:
                print(f"   ❌ HYPOTHESIS REJECTED: Large D still needed")

def analyze_multilevel_comparison(df):
    """Analyze multi-level performance"""
    print("\n" + "=" * 80)
    print("🏗️ MULTI-LEVEL COMPARISON ANALYSIS")
    print("=" * 80)
    print("Comparing GraphZoom-1, GraphZoom-2, GraphZoom-3 levels")
    
    # Filter multi-level experiments
    multilevel_df = df[df['level'].notna() & df['level'].isin([1, 2, 3])].copy()
    
    if multilevel_df.empty:
        print("❌ No multi-level experiments found")
        return
    
    print(f"\n📊 MULTI-LEVEL PERFORMANCE BY DATASET:")
    print(f"{'Dataset':<10} {'Method':<8} {'Level':<6} {'Accuracy':<9} {'Time(s)':<8} {'Compression':<11}")
    print("-" * 60)
    
    multilevel_summary = []
    
    for dataset in sorted(multilevel_df['dataset'].unique()):
        dataset_df = multilevel_df[multilevel_df['dataset'] == dataset]
        
        for method in ['simple', 'cmg']:
            method_df = dataset_df[dataset_df['coarsening'] == method]
            
            for level in [1, 2, 3]:
                level_df = method_df[method_df['level'] == level]
                
                if not level_df.empty:
                    acc = level_df['accuracy'].mean()
                    time = level_df['total_time'].mean()
                    comp = level_df['compression_ratio'].mean()
                    
                    print(f"{dataset:<10} {method:<8} {level:<6} {acc:<9.3f} {time:<8.1f} {comp:<11.1f}x")
                    
                    multilevel_summary.append({
                        'dataset': dataset, 'method': method, 'level': level,
                        'accuracy': acc, 'time': time, 'compression': comp
                    })
    
    # Analysis of level effects
    if multilevel_summary:
        summary_df = pd.DataFrame(multilevel_summary)
        
        print(f"\n📈 LEVEL EFFECT ANALYSIS:")
        for method in ['simple', 'cmg']:
            method_data = summary_df[summary_df['method'] == method]
            if not method_data.empty:
                print(f"\n{method.upper()} Method:")
                level_effects = method_data.groupby('level').agg({
                    'accuracy': ['mean', 'std'],
                    'time': 'mean',
                    'compression': 'mean'
                }).round(3)
                print(level_effects)

def analyze_stability(df):
    """Analyze CMG stability vs Simple coarsening"""
    print("\n" + "=" * 80)
    print("🎲 STABILITY ANALYSIS")
    print("=" * 80)
    print("Testing: 'CMG will be more stable wrt to the initial random vectors'")
    
    # Filter stability experiments
    stability_df = df[df['experiment'].str.contains('stability', na=False)].copy()
    
    if stability_df.empty:
        print("❌ No stability experiments found")
        return
    
    print(f"\n📊 STABILITY COMPARISON (Multiple Random Seeds):")
    
    stability_results = []
    
    for method in ['simple', 'cmg']:
        method_df = stability_df[stability_df['coarsening'] == method]
        
        if not method_df.empty:
            acc_mean = method_df['accuracy'].mean()
            acc_std = method_df['accuracy'].std()
            acc_var = method_df['accuracy'].var()
            time_mean = method_df['total_time'].mean()
            time_std = method_df['total_time'].std()
            count = len(method_df)
            
            # Coefficient of variation (std/mean) - lower is more stable
            acc_cv = (acc_std / acc_mean) * 100 if acc_mean > 0 else 0
            time_cv = (time_std / time_mean) * 100 if time_mean > 0 else 0
            
            stability_results.append({
                'method': method,
                'acc_mean': acc_mean,
                'acc_std': acc_std,
                'acc_cv': acc_cv,
                'time_mean': time_mean,
                'time_std': time_std,
                'time_cv': time_cv,
                'count': count
            })
            
            print(f"\n{method.upper()} Method ({count} runs):")
            print(f"   Accuracy: {acc_mean:.3f} ± {acc_std:.3f} (CV: {acc_cv:.1f}%)")
            print(f"   Time: {time_mean:.1f} ± {time_std:.1f}s (CV: {time_cv:.1f}%)")
    
    # Compare stability
    if len(stability_results) == 2:
        cmg_result = next(r for r in stability_results if r['method'] == 'cmg')
        simple_result = next(r for r in stability_results if r['method'] == 'simple')
        
        print(f"\n💡 STABILITY COMPARISON:")
        print(f"   CMG Accuracy CV: {cmg_result['acc_cv']:.1f}%")
        print(f"   Simple Accuracy CV: {simple_result['acc_cv']:.1f}%")
        
        if cmg_result['acc_cv'] < simple_result['acc_cv']:
            improvement = simple_result['acc_cv'] - cmg_result['acc_cv']
            print(f"   ✅ CMG IS MORE STABLE by {improvement:.1f}% CV")
        else:
            print(f"   ❌ Simple coarsening is more stable")
        
        print(f"\n   CMG Time CV: {cmg_result['time_cv']:.1f}%")
        print(f"   Simple Time CV: {simple_result['time_cv']:.1f}%")

def analyze_timing_breakdown(df):
    """Analyze where time is spent in the pipeline"""
    print("\n" + "=" * 80)
    print("⚡ RUNTIME BREAKDOWN ANALYSIS")
    print("=" * 80)
    print("Testing: 'whether the runtime is dominated by that of deepwalk at the smallest level'")
    
    timing_df = df[df['experiment'].str.contains('timing', na=False)].copy()
    
    if timing_df.empty:
        print("❌ No timing experiments found")
        return
    
    print(f"\n📊 RUNTIME BREAKDOWN BY COMPONENT:")
    print(f"{'Method':<8} {'Dataset':<10} {'Fusion%':<8} {'Reduction%':<11} {'Embedding%':<11} {'Refinement%':<11}")
    print("-" * 65)
    
    for _, row in timing_df.iterrows():
        total = row['total_time']
        if pd.notna(total) and total > 0:
            fusion_pct = (row['fusion_time'] / total * 100) if pd.notna(row['fusion_time']) else 0
            reduction_pct = (row['reduction_time'] / total * 100) if pd.notna(row['reduction_time']) else 0
            embedding_pct = (row['embedding_time'] / total * 100) if pd.notna(row['embedding_time']) else 0
            refinement_pct = (row['refinement_time'] / total * 100) if pd.notna(row['refinement_time']) else 0
            
            print(f"{row['coarsening']:<8} {row['dataset']:<10} {fusion_pct:<8.1f} {reduction_pct:<11.1f} {embedding_pct:<11.1f} {refinement_pct:<11.1f}")
    
    # Analyze if embedding dominates
    embedding_dominated = 0
    total_experiments = 0
    
    for _, row in timing_df.iterrows():
        total = row['total_time']
        if pd.notna(total) and total > 0 and pd.notna(row['embedding_time']):
            embedding_pct = row['embedding_time'] / total * 100
            total_experiments += 1
            
            if embedding_pct > 50:  # Embedding takes more than 50% of time
                embedding_dominated += 1
    
    if total_experiments > 0:
        print(f"\n💡 EMBEDDING DOMINANCE ANALYSIS:")
        print(f"   Experiments where embedding >50% of time: {embedding_dominated}/{total_experiments} ({embedding_dominated/total_experiments*100:.1f}%)")
        
        if embedding_dominated / total_experiments > 0.7:
            print(f"   ✅ CONFIRMED: Embedding dominates runtime (as Koutis suggested)")
        else:
            print(f"   ❌ Embedding does not dominate runtime")

def generate_koutis_summary(df):
    """Generate executive summary for Koutis"""
    print("\n" + "=" * 80)
    print("📋 EXECUTIVE SUMMARY FOR KOUTIS")
    print("=" * 80)
    
    # Overall statistics
    total_exp = len(df)
    cmg_exp = len(df[df['coarsening'] == 'cmg'])
    datasets = df['dataset'].unique()
    
    print(f"📊 EXPERIMENT OVERVIEW:")
    print(f"   Total experiments: {total_exp}")
    print(f"   CMG experiments: {cmg_exp}")
    print(f"   Datasets tested: {list(datasets)}")
    print(f"   Parameter ranges tested:")
    
    cmg_df = df[df['coarsening'] == 'cmg']
    if not cmg_df.empty:
        k_range = f"{cmg_df['cmg_k'].min():.0f}-{cmg_df['cmg_k'].max():.0f}" if cmg_df['cmg_k'].notna().any() else "N/A"
        d_range = f"{cmg_df['cmg_d'].min():.0f}-{cmg_df['cmg_d'].max():.0f}" if cmg_df['cmg_d'].notna().any() else "N/A"
        print(f"      • K (filter order): {k_range}")
        print(f"      • D (embedding dim): {d_range}")
    
    # Key findings summary
    print(f"\n🎯 KEY FINDINGS:")
    
    # Best performance
    if not df.empty:
        best_accuracy = df.loc[df['accuracy'].idxmax()]
        best_compression = df.loc[df['compression_ratio'].idxmax()] if 'compression_ratio' in df.columns else None
        
        print(f"   🏆 Best accuracy: {best_accuracy['accuracy']:.3f} ({best_accuracy['experiment']})")
        if best_compression is not None:
            print(f"   🏆 Best compression: {best_compression['compression_ratio']:.1f}x ({best_compression['experiment']})")
    
    # Parameter efficiency findings
    cmg_cora = cmg_df[(cmg_df['dataset'] == 'cora') & cmg_df['cmg_d'].notna()]
    if not cmg_cora.empty:
        small_d_best = cmg_cora[cmg_cora['cmg_d'] <= 15]['accuracy'].max() if len(cmg_cora[cmg_cora['cmg_d'] <= 15]) > 0 else 0
        large_d_best = cmg_cora[cmg_cora['cmg_d'] >= 25]['accuracy'].max() if len(cmg_cora[cmg_cora['cmg_d'] >= 25]) > 0 else 0
        
        print(f"   💡 Parameter efficiency (D≤15 vs D≥25): {small_d_best:.3f} vs {large_d_best:.3f}")
        if small_d_best >= large_d_best - 0.01:
            print(f"      ✅ Your hypothesis CONFIRMED: Small D works well!")
    
    # Stability comparison
    stability_df = df[df['experiment'].str.contains('stability', na=False)]
    if not stability_df.empty:
        cmg_stability = stability_df[stability_df['coarsening'] == 'cmg']
        simple_stability = stability_df[stability_df['coarsening'] == 'simple']
        
        if not cmg_stability.empty and not simple_stability.empty:
            cmg_cv = (cmg_stability['accuracy'].std() / cmg_stability['accuracy'].mean()) * 100
            simple_cv = (simple_stability['accuracy'].std() / simple_stability['accuracy'].mean()) * 100
            
            print(f"   🎲 Stability (CV): CMG {cmg_cv:.1f}% vs Simple {simple_cv:.1f}%")
            if cmg_cv < simple_cv:
                print(f"      ✅ Your hypothesis CONFIRMED: CMG is more stable!")
    
    # Timing insights
    timing_df = df[df['experiment'].str.contains('timing', na=False)]
    if not timing_df.empty:
        avg_embedding_pct = 0
        count = 0
        for _, row in timing_df.iterrows():
            if pd.notna(row['total_time']) and pd.notna(row['embedding_time']) and row['total_time'] > 0:
                avg_embedding_pct += (row['embedding_time'] / row['total_time']) * 100
                count += 1
        
        if count > 0:
            avg_embedding_pct /= count
            print(f"   ⚡ Average embedding time: {avg_embedding_pct:.1f}% of total")
            if avg_embedding_pct > 50:
                print(f"      ✅ Your observation CONFIRMED: Embedding dominates runtime")
    
    print(f"\n📈 RECOMMENDATIONS:")
    
    # Best parameter recommendations
    if not cmg_cora.empty:
        best_d_row = cmg_cora.loc[cmg_cora['accuracy'].idxmax()]
        print(f"   • Optimal CMG D: {best_d_row['cmg_d']:.0f} (accuracy: {best_d_row['accuracy']:.3f})")
    
    cmg_k_data = cmg_df[(cmg_df['dataset'] == 'cora') & cmg_df['cmg_k'].notna()]
    if not cmg_k_data.empty:
        best_k_row = cmg_k_data.loc[cmg_k_data['accuracy'].idxmax()]
        print(f"   • Optimal CMG K: {best_k_row['cmg_k']:.0f} (accuracy: {best_k_row['accuracy']:.3f})")
    
    print(f"   • Multi-level potential: Test natural stopping vs fixed levels")
    print(f"   • LAMG comparison: Consider Python translation for complete evaluation")

def main():
    """Main analysis function"""
    if len(sys.argv) != 2:
        print("Usage: python analyze_koutis_experiments.py <results_csv_file>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)
    
    print("🔍 KOUTIS EXPERIMENT ANALYSIS")
    print("=" * 50)
    print(f"Analyzing: {csv_path}")
    
    # Load and analyze data
    df = load_and_clean_data(csv_path)
    
    if df.empty:
        print("❌ No valid data found")
        sys.exit(1)
    
    # Run all analyses
    analyze_parameter_efficiency(df)
    analyze_multilevel_comparison(df)
    analyze_stability(df) 
    analyze_timing_breakdown(df)
    generate_koutis_summary(df)
    
    # Generate CSV summaries for easy sharing
    results_dir = Path(csv_path).parent
    
    # Parameter summary
    cmg_df = df[df['coarsening'] == 'cmg']
    if not cmg_df.empty:
        param_summary = cmg_df.groupby(['cmg_k', 'cmg_d']).agg({
            'accuracy': ['mean', 'std', 'count'],
            'total_time': 'mean',
            'compression_ratio': 'mean'
        }).round(4)
        
        param_summary.to_csv(results_dir / 'parameter_summary.csv')
        print(f"\n📁 Parameter summary saved: {results_dir / 'parameter_summary.csv'}")
    
    # Stability summary
    stability_df = df[df['experiment'].str.contains('stability', na=False)]
    if not stability_df.empty:
        stability_summary = stability_df.groupby('coarsening').agg({
            'accuracy': ['mean', 'std', 'min', 'max'],
            'total_time': ['mean', 'std']
        }).round(4)
        
        stability_summary.to_csv(results_dir / 'stability_summary.csv')
        print(f"📁 Stability summary saved: {results_dir / 'stability_summary.csv'}")
    
    # Multi-level summary
    multilevel_df = df[df['level'].notna()]
    if not multilevel_df.empty:
        multilevel_summary = multilevel_df.groupby(['coarsening', 'dataset', 'level']).agg({
            'accuracy': 'mean',
            'total_time': 'mean',
            'compression_ratio': 'mean'
        }).round(4)
        
        multilevel_summary.to_csv(results_dir / 'multilevel_summary.csv')
        print(f"📁 Multi-level summary saved: {results_dir / 'multilevel_summary.csv'}")
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📧 Key files to share with Koutis:")
    print(f"   • Main results: {csv_path}")
    print(f"   • This analysis output (copy from terminal)")
    print(f"   • Summary CSVs in: {results_dir}")

if __name__ == "__main__":
    main()
