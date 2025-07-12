#!/usr/bin/env python3
"""
Simple analysis script for CMG-GraphZoom evaluation results
No matplotlib dependency - just pandas and basic analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_results(csv_path):
    """Load results CSV and clean data"""
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} experiments from {csv_path}")
        
        # Convert accuracy to float, handle missing values
        df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
        df['total_time'] = pd.to_numeric(df['total_time'], errors='coerce')
        
        # Remove timeout/failed experiments
        df_clean = df.dropna(subset=['accuracy'])
        print(f"📊 {len(df_clean)} successful experiments")
        
        return df_clean
    except Exception as e:
        print(f"❌ Error loading {csv_path}: {e}")
        return None

def analyze_baseline_comparison(df):
    """Compare CMG vs Simple coarsening"""
    print("\n" + "=" * 60)
    print("📊 BASELINE COMPARISON: CMG vs Simple Coarsening")
    print("=" * 60)
    
    # Filter for baseline comparison
    baseline_df = df[df['coarsening'].isin(['simple', 'cmg'])]
    
    if baseline_df.empty:
        print("❌ No baseline comparison data found")
        return None
    
    # Group by dataset and embedding method
    results = []
    
    for dataset in baseline_df['dataset'].unique():
        for embed in baseline_df['embedding'].unique():
            simple_rows = baseline_df[
                (baseline_df['dataset'] == dataset) & 
                (baseline_df['embedding'] == embed) & 
                (baseline_df['coarsening'] == 'simple')
            ]
            
            cmg_rows = baseline_df[
                (baseline_df['dataset'] == dataset) & 
                (baseline_df['embedding'] == embed) & 
                (baseline_df['coarsening'] == 'cmg')
            ]
            
            if not simple_rows.empty and not cmg_rows.empty:
                simple_acc = simple_rows['accuracy'].mean()
                cmg_acc = cmg_rows['accuracy'].mean()
                simple_time = simple_rows['total_time'].mean()
                cmg_time = cmg_rows['total_time'].mean()
                
                improvement = cmg_acc - simple_acc
                speedup = simple_time / cmg_time if cmg_time > 0 else np.nan
                
                results.append({
                    'dataset': dataset,
                    'embedding': embed,
                    'simple_accuracy': simple_acc,
                    'cmg_accuracy': cmg_acc,
                    'improvement': improvement,
                    'simple_time': simple_time,
                    'cmg_time': cmg_time,
                    'speedup': speedup
                })
    
    if not results:
        print("❌ No matching pairs found for comparison")
        return None
        
    results_df = pd.DataFrame(results)
    
    print("\n📈 ACCURACY COMPARISON:")
    print(f"{'Dataset':<10} {'Embedding':<10} {'Simple':<8} {'CMG':<8} {'Improve':<8}")
    print("-" * 55)
    for _, row in results_df.iterrows():
        print(f"{row['dataset']:<10} {row['embedding']:<10} {row['simple_accuracy']:<8.3f} {row['cmg_accuracy']:<8.3f} {row['improvement']:<+8.3f}")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Average Improvement: {results_df['improvement'].mean():+.4f}")
    print(f"   Best Improvement: {results_df['improvement'].max():+.4f}")
    print(f"   Improvements > 0: {(results_df['improvement'] > 0).sum()}/{len(results_df)}")
    print(f"   Win Rate: {(results_df['improvement'] > 0).mean()*100:.1f}%")
    
    print("\n⚡ SPEEDUP COMPARISON:")
    print(f"{'Dataset':<10} {'Embedding':<10} {'Simple(s)':<10} {'CMG(s)':<10} {'Speedup':<8}")
    print("-" * 60)
    for _, row in results_df.iterrows():
        print(f"{row['dataset']:<10} {row['embedding']:<10} {row['simple_time']:<10.1f} {row['cmg_time']:<10.1f} {row['speedup']:<8.1f}x")
    
    print(f"\n🚀 SPEEDUP SUMMARY:")
    print(f"   Average Speedup: {results_df['speedup'].mean():.1f}x")
    print(f"   Best Speedup: {results_df['speedup'].max():.1f}x")
    
    return results_df

def analyze_parameter_study(df):
    """Analyze CMG parameter sensitivity"""
    print("\n" + "=" * 60)
    print("🔬 CMG PARAMETER STUDY")
    print("=" * 60)
    
    cmg_df = df[df['coarsening'] == 'cmg'].copy()
    
    if cmg_df.empty:
        print("❌ No CMG experiments found")
        return None
    
    # Extract parameters from experiment names
    def extract_param_value(exp_name, param_name):
        import re
        exp_str = str(exp_name)
        
        if f'{param_name}' in exp_str:
            pattern = f'{param_name}(\\d+(?:\\.\\d+)?)'
            match = re.search(pattern, exp_str)
            if match:
                return float(match.group(1))
        return None
    
    # Analyze each parameter
    params_analysis = {}
    
    # K parameter analysis
    k_data = []
    for _, row in cmg_df.iterrows():
        k_val = extract_param_value(row['experiment'], 'k')
        if k_val and row['dataset'] == 'cora':  # Focus on Cora for parameter study
            k_data.append({'k': k_val, 'accuracy': row['accuracy'], 'time': row['total_time']})
    
    if k_data:
        k_df = pd.DataFrame(k_data)
        k_summary = k_df.groupby('k').agg({'accuracy': ['mean', 'count'], 'time': 'mean'}).round(4)
        print(f"\n📊 K Parameter Analysis (Cora dataset):")
        print(f"{'K Value':<8} {'Accuracy':<10} {'Count':<6} {'Time(s)':<8}")
        print("-" * 35)
        for k in sorted(k_df['k'].unique()):
            k_rows = k_df[k_df['k'] == k]
            acc_mean = k_rows['accuracy'].mean()
            time_mean = k_rows['time'].mean()
            count = len(k_rows)
            print(f"{k:<8.0f} {acc_mean:<10.3f} {count:<6} {time_mean:<8.1f}")
        
        best_k = k_df.loc[k_df['accuracy'].idxmax()]
        print(f"🏆 Best K: {best_k['k']:.0f} (accuracy: {best_k['accuracy']:.3f})")
    
    # D parameter analysis
    d_data = []
    for _, row in cmg_df.iterrows():
        d_val = extract_param_value(row['experiment'], 'd')
        if d_val and row['dataset'] == 'cora':
            d_data.append({'d': d_val, 'accuracy': row['accuracy'], 'time': row['total_time']})
    
    if d_data:
        d_df = pd.DataFrame(d_data)
        print(f"\n📊 D Parameter Analysis (Cora dataset):")
        print(f"{'D Value':<8} {'Accuracy':<10} {'Count':<6} {'Time(s)':<8}")
        print("-" * 35)
        for d in sorted(d_df['d'].unique()):
            d_rows = d_df[d_df['d'] == d]
            acc_mean = d_rows['accuracy'].mean()
            time_mean = d_rows['time'].mean()
            count = len(d_rows)
            print(f"{d:<8.0f} {acc_mean:<10.3f} {count:<6} {time_mean:<8.1f}")
        
        best_d = d_df.loc[d_df['accuracy'].idxmax()]
        print(f"🏆 Best D: {best_d['d']:.0f} (accuracy: {best_d['accuracy']:.3f})")
    
    # Threshold parameter analysis
    thresh_data = []
    for _, row in cmg_df.iterrows():
        thresh_val = extract_param_value(row['experiment'], 'thresh')
        if thresh_val and row['dataset'] == 'cora':
            thresh_data.append({'threshold': thresh_val, 'accuracy': row['accuracy'], 'time': row['total_time']})
    
    if thresh_data:
        thresh_df = pd.DataFrame(thresh_data)
        print(f"\n📊 Threshold Parameter Analysis (Cora dataset):")
        print(f"{'Threshold':<10} {'Accuracy':<10} {'Count':<6} {'Time(s)':<8}")
        print("-" * 37)
        for thresh in sorted(thresh_df['threshold'].unique()):
            thresh_rows = thresh_df[thresh_df['threshold'] == thresh]
            acc_mean = thresh_rows['accuracy'].mean()
            time_mean = thresh_rows['time'].mean()
            count = len(thresh_rows)
            print(f"{thresh:<10.2f} {acc_mean:<10.3f} {count:<6} {time_mean:<8.1f}")
        
        best_thresh = thresh_df.loc[thresh_df['accuracy'].idxmax()]
        print(f"🏆 Best Threshold: {best_thresh['threshold']:.2f} (accuracy: {best_thresh['accuracy']:.3f})")

def generate_summary_table(df):
    """Generate summary table for paper/report"""
    print("\n" + "=" * 60)
    print("📋 OVERALL SUMMARY")
    print("=" * 60)
    
    # Best results per dataset-method combination
    print("\n🏆 BEST RESULTS BY DATASET:")
    print(f"{'Dataset':<10} {'Method':<8} {'Embedding':<10} {'Accuracy':<10} {'Time':<8}")
    print("-" * 55)
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        best_row = dataset_df.loc[dataset_df['accuracy'].idxmax()]
        print(f"{best_row['dataset']:<10} {best_row['coarsening']:<8} {best_row['embedding']:<10} {best_row['accuracy']:<10.3f} {best_row['total_time']:<8.1f}")
    
    # Method comparison summary
    print(f"\n📊 METHOD PERFORMANCE SUMMARY:")
    method_summary = df.groupby('coarsening').agg({
        'accuracy': ['mean', 'std', 'max'],
        'total_time': ['mean', 'std'],
        'experiment': 'count'
    }).round(4)
    
    print(f"{'Method':<8} {'Avg Acc':<8} {'Std':<6} {'Max Acc':<8} {'Avg Time':<8} {'Count':<6}")
    print("-" * 50)
    for method in method_summary.index:
        avg_acc = method_summary.loc[method, ('accuracy', 'mean')]
        std_acc = method_summary.loc[method, ('accuracy', 'std')]
        max_acc = method_summary.loc[method, ('accuracy', 'max')]
        avg_time = method_summary.loc[method, ('total_time', 'mean')]
        count = method_summary.loc[method, ('experiment', 'count')]
        print(f"{method:<8} {avg_acc:<8.3f} {std_acc:<6.3f} {max_acc:<8.3f} {avg_time:<8.1f} {count:<6.0f}")

def main():
    """Main analysis function"""
    print("🔍 CMG-GraphZoom Results Analysis")
    print("=" * 50)
    
    # Try to load results
    results_files = [
        'results/quick_results.csv',
        'results/all_results.csv'
    ]
    
    df = None
    for file_path in results_files:
        if Path(file_path).exists():
            df = load_results(file_path)
            if df is not None:
                break
    
    if df is None:
        print("❌ No results file found. Run the evaluation script first!")
        return
    
    print(f"\n📈 LOADED DATA OVERVIEW:")
    print(f"   Methods tested: {list(df['coarsening'].unique())}")
    print(f"   Datasets: {list(df['dataset'].unique())}")
    print(f"   Embeddings: {list(df['embedding'].unique())}")
    print(f"   Total experiments: {len(df)}")
    
    # Run analyses
    baseline_results = analyze_baseline_comparison(df)
    analyze_parameter_study(df)
    generate_summary_table(df)
    
    print(f"\n✅ Analysis complete!")
    print(f"💡 Key takeaways:")
    if baseline_results is not None and not baseline_results.empty:
        avg_improvement = baseline_results['improvement'].mean()
        avg_speedup = baseline_results['speedup'].mean()
        print(f"   • Average accuracy improvement: {avg_improvement:+.3f}")
        print(f"   • Average speedup: {avg_speedup:.1f}x")
        print(f"   • CMG wins in {(baseline_results['improvement'] > 0).sum()}/{len(baseline_results)} comparisons")

if __name__ == "__main__":
    main()
