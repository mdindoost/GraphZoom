#!/usr/bin/env python3
"""
Analysis script for CMG-GraphZoom evaluation results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(csv_path):
    """Load results CSV and clean data"""
    df = pd.read_csv(csv_path)
    
    # Convert accuracy to float, handle missing values
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df['total_time'] = pd.to_numeric(df['total_time'], errors='coerce')
    
    # Remove timeout/failed experiments
    df = df.dropna(subset=['accuracy'])
    
    return df

def analyze_baseline_comparison(df):
    """Compare CMG vs Simple coarsening"""
    print("=" * 60)
    print("📊 BASELINE COMPARISON: CMG vs Simple Coarsening")
    print("=" * 60)
    
    # Filter for baseline comparison
    baseline_df = df[df['coarsening'].isin(['simple', 'cmg'])]
    
    # Group by dataset and embedding method
    results = []
    
    for dataset in baseline_df['dataset'].unique():
        for embed in baseline_df['embedding'].unique():
            simple_row = baseline_df[
                (baseline_df['dataset'] == dataset) & 
                (baseline_df['embedding'] == embed) & 
                (baseline_df['coarsening'] == 'simple')
            ]
            
            cmg_row = baseline_df[
                (baseline_df['dataset'] == dataset) & 
                (baseline_df['embedding'] == embed) & 
                (baseline_df['coarsening'] == 'cmg')
            ]
            
            if not simple_row.empty and not cmg_row.empty:
                simple_acc = simple_row['accuracy'].iloc[0]
                cmg_acc = cmg_row['accuracy'].iloc[0]
                simple_time = simple_row['total_time'].iloc[0]
                cmg_time = cmg_row['total_time'].iloc[0]
                
                improvement = cmg_acc - simple_acc
                time_ratio = cmg_time / simple_time if simple_time > 0 else np.nan
                
                results.append({
                    'dataset': dataset,
                    'embedding': embed,
                    'simple_accuracy': simple_acc,
                    'cmg_accuracy': cmg_acc,
                    'improvement': improvement,
                    'simple_time': simple_time,
                    'cmg_time': cmg_time,
                    'time_ratio': time_ratio
                })
    
    results_df = pd.DataFrame(results)
    
    print("\n📈 Accuracy Comparison:")
    print(results_df[['dataset', 'embedding', 'simple_accuracy', 'cmg_accuracy', 'improvement']].round(4))
    
    print(f"\n🎯 Average Improvement: {results_df['improvement'].mean():.4f}")
    print(f"🏆 Best Improvement: {results_df['improvement'].max():.4f}")
    print(f"📊 Improvements > 0: {(results_df['improvement'] > 0).sum()}/{len(results_df)}")
    
    print("\n⏱️ Time Comparison:")
    print(results_df[['dataset', 'embedding', 'simple_time', 'cmg_time', 'time_ratio']].round(2))
    
    return results_df

def analyze_parameter_study(df):
    """Analyze CMG parameter sensitivity"""
    print("\n" + "=" * 60)
    print("🔬 CMG PARAMETER STUDY")
    print("=" * 60)
    
    cmg_df = df[df['coarsening'] == 'cmg'].copy()
    
    # Extract parameters from the parameters column or experiment name
    def extract_param_value(row, param_name):
        params_str = str(row.get('parameters', ''))
        exp_name = str(row.get('experiment', ''))
        
        # Try to extract from parameters column
        if param_name in params_str:
            import re
            pattern = f'--cmg_{param_name}\\s+(\\S+)'
            match = re.search(pattern, params_str)
            if match:
                return float(match.group(1))
        
        # Try to extract from experiment name
        if f'{param_name}' in exp_name:
            import re
            pattern = f'{param_name}(\\d+(?:\\.\\d+)?)'
            match = re.search(pattern, exp_name)
            if match:
                return float(match.group(1))
        
        return None
    
    cmg_df['k_value'] = cmg_df.apply(lambda row: extract_param_value(row, 'k'), axis=1)
    cmg_df['d_value'] = cmg_df.apply(lambda row: extract_param_value(row, 'd'), axis=1)
    cmg_df['threshold_value'] = cmg_df.apply(lambda row: extract_param_value(row, 'thresh'), axis=1)
    
    # Analyze k parameter
    k_results = cmg_df[cmg_df['k_value'].notna()].groupby('k_value')['accuracy'].agg(['mean', 'std', 'count'])
    if not k_results.empty:
        print("\n📊 K Parameter Analysis:")
        print(k_results.round(4))
        best_k = k_results['mean'].idxmax()
        print(f"🏆 Best K value: {best_k} (accuracy: {k_results.loc[best_k, 'mean']:.4f})")
    
    # Analyze d parameter  
    d_results = cmg_df[cmg_df['d_value'].notna()].groupby('d_value')['accuracy'].agg(['mean', 'std', 'count'])
    if not d_results.empty:
        print("\n📊 D Parameter Analysis:")
        print(d_results.round(4))
        best_d = d_results['mean'].idxmax()
        print(f"🏆 Best D value: {best_d} (accuracy: {d_results.loc[best_d, 'mean']:.4f})")
    
    # Analyze threshold parameter
    thresh_results = cmg_df[cmg_df['threshold_value'].notna()].groupby('threshold_value')['accuracy'].agg(['mean', 'std', 'count'])
    if not thresh_results.empty:
        print("\n📊 Threshold Parameter Analysis:")
        print(thresh_results.round(4))
        best_thresh = thresh_results['mean'].idxmax()
        print(f"🏆 Best Threshold: {best_thresh} (accuracy: {thresh_results.loc[best_thresh, 'mean']:.4f})")
    
    return k_results, d_results, thresh_results

def create_plots(df, output_dir="results/plots"):
    """Create visualization plots"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    
    # 1. Baseline comparison plot
    baseline_df = df[df['coarsening'].isin(['simple', 'cmg'])]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    sns.barplot(data=baseline_df, x='dataset', y='accuracy', hue='coarsening', ax=ax1)
    ax1.set_title('Accuracy Comparison: CMG vs Simple Coarsening')
    ax1.set_ylabel('Test Accuracy')
    
    # Time comparison
    sns.barplot(data=baseline_df, x='dataset', y='total_time', hue='coarsening', ax=ax2)
    ax2.set_title('Runtime Comparison: CMG vs Simple Coarsening')
    ax2.set_ylabel('Total Time (seconds)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/baseline_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Parameter study plots
    cmg_df = df[df['coarsening'] == 'cmg'].copy()
    
    if not cmg_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract parameter values (simplified)
        def extract_param(exp_name, param):
            import re
            pattern = f'{param}(\\d+(?:\\.\\d+)?)'
            match = re.search(pattern, str(exp_name))
            return float(match.group(1)) if match else None
        
        # K parameter plot
        k_data = []
        for _, row in cmg_df.iterrows():
            k_val = extract_param(row['experiment'], 'k')
            if k_val:
                k_data.append({'k': k_val, 'accuracy': row['accuracy']})
        
        if k_data:
            k_df = pd.DataFrame(k_data)
            sns.lineplot(data=k_df, x='k', y='accuracy', marker='o', ax=axes[0,0])
            axes[0,0].set_title('CMG Performance vs K Parameter')
            axes[0,0].set_xlabel('K (Filter Order)')
            axes[0,0].set_ylabel('Test Accuracy')
        
        # D parameter plot
        d_data = []
        for _, row in cmg_df.iterrows():
            d_val = extract_param(row['experiment'], 'd')
            if d_val:
                d_data.append({'d': d_val, 'accuracy': row['accuracy']})
        
        if d_data:
            d_df = pd.DataFrame(d_data)
            sns.lineplot(data=d_df, x='d', y='accuracy', marker='o', ax=axes[0,1])
            axes[0,1].set_title('CMG Performance vs D Parameter')
            axes[0,1].set_xlabel('D (Embedding Dimension)')
            axes[0,1].set_ylabel('Test Accuracy')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/parameter_study.png", dpi=300, bbox_inches='tight')
        plt.close()

def generate_summary_table(df):
    """Generate summary table for paper/report"""
    print("\n" + "=" * 60)
    print("📋 SUMMARY TABLE")
    print("=" * 60)
    
    # Group by method and dataset
    summary = df.groupby(['dataset', 'coarsening', 'embedding'])['accuracy'].agg(['mean', 'std']).round(4)
    
    print("\n📊 Mean ± Std Accuracy by Method:")
    print(summary)
    
    # Best results per dataset
    print("\n🏆 Best Results per Dataset:")
    best_results = df.loc[df.groupby('dataset')['accuracy'].idxmax()]
    print(best_results[['dataset', 'coarsening', 'embedding', 'accuracy', 'total_time']].round(4))

def main():
    """Main analysis function"""
    # Try to load results
    results_files = [
        'results/all_results.csv',
        'results/quick_results.csv'
    ]
    
    df = None
    for file_path in results_files:
        if Path(file_path).exists():
            print(f"📁 Loading results from {file_path}")
            df = load_results(file_path)
            break
    
    if df is None:
        print("❌ No results file found. Run the evaluation script first!")
        return
    
    print(f"📊 Loaded {len(df)} experiments")
    print(f"🔬 Methods tested: {df['coarsening'].unique()}")
    print(f"📚 Datasets: {df['dataset'].unique()}")
    print(f"🚀 Embeddings: {df['embedding'].unique()}")
    
    # Run analyses
    baseline_results = analyze_baseline_comparison(df)
    analyze_parameter_study(df)
    generate_summary_table(df)
    
    # Create plots
    try:
        create_plots(df)
        print("\n📈 Plots saved to results/plots/")
    except Exception as e:
        print(f"⚠️ Could not create plots: {e}")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()