#!/usr/bin/env python3
import os
import re
import sys
import pandas as pd
import glob
import numpy as np

def extract_metrics_from_log(log_file):
    """Extract metrics from a log file"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract experiment info from filename
        filename = os.path.basename(log_file).replace('.log', '')
        parts = filename.split('_')
        
        # Parse config name and dataset
        if len(parts) >= 3:
            method_type = parts[0]  # simple, lamg, or cmg
            level_or_ratio = parts[1]  # l1, l2, l3, r2, r3, r6
            dataset = parts[2]
        else:
            method_type = parts[0]
            level_or_ratio = ""
            dataset = parts[1] if len(parts) > 1 else "unknown"
        
        config_name = f"{method_type}_{level_or_ratio}"
        
        # Extract metrics using regex
        accuracy_match = re.search(r'Test Accuracy:\s+([\d.]+)', content)
        total_time_match = re.search(r'Total Time.*?=\s*([\d.]+)', content)
        fusion_time_match = re.search(r'Graph Fusion\s+Time:\s+([\d.]+)', content)
        reduction_time_match = re.search(r'Graph Reduction\s+Time:\s+([\d.]+)', content)
        embedding_time_match = re.search(r'Graph Embedding\s+Time:\s+([\d.]+)', content)
        refinement_time_match = re.search(r'Graph Refinement\s+Time:\s+([\d.]+)', content)
        
        # Extract coarsening info if available
        coarsening_info = ""
        if "CMG" in content or "cmg" in config_name:
            nodes_match = re.search(r'Final graph:\s+(\d+)\s+nodes', content)
            if nodes_match:
                coarsening_info = f"Final: {nodes_match.group(1)} nodes"
        
        # Check if experiment completed successfully
        if not accuracy_match or not total_time_match:
            return None
        
        return {
            'config': config_name,
            'method': method_type,
            'level_ratio': level_or_ratio,
            'dataset': dataset,
            'accuracy': float(accuracy_match.group(1)),
            'total_time': float(total_time_match.group(1)),
            'fusion_time': float(fusion_time_match.group(1)) if fusion_time_match else 0.0,
            'reduction_time': float(reduction_time_match.group(1)) if reduction_time_match else 0.0,
            'embedding_time': float(embedding_time_match.group(1)) if embedding_time_match else 0.0,
            'refinement_time': float(refinement_time_match.group(1)) if refinement_time_match else 0.0,
            'coarsening_info': coarsening_info,
            'status': 'completed'
        }
    except Exception as e:
        print(f"Error processing {log_file}: {e}")
        return None

def calculate_efficiency(df, baseline_config='simple_l2'):
    """Calculate efficiency scores relative to baseline"""
    df = df.copy()
    
    # Calculate efficiency for each dataset separately
    for dataset in df['dataset'].unique():
        dataset_mask = df['dataset'] == dataset
        baseline_mask = dataset_mask & (df['config'] == baseline_config)
        
        if baseline_mask.sum() == 0:
            print(f"Warning: No baseline {baseline_config} found for {dataset}")
            # Fallback to any simple method as baseline
            fallback_baseline = dataset_mask & df['config'].str.startswith('simple')
            if fallback_baseline.sum() > 0:
                baseline_mask = fallback_baseline.iloc[:1]
                baseline_config_actual = df.loc[baseline_mask, 'config'].iloc[0]
                print(f"Using {baseline_config_actual} as baseline instead")
            else:
                continue
        
        baseline_acc = df.loc[baseline_mask, 'accuracy'].iloc[0]
        baseline_time = df.loc[baseline_mask, 'total_time'].iloc[0]
        
        # Calculate efficiency score for this dataset
        dataset_df = df[dataset_mask].copy()
        efficiency_scores = []
        
        for _, row in dataset_df.iterrows():
            acc_ratio = row['accuracy'] / baseline_acc
            time_ratio = row['total_time'] / baseline_time
            efficiency = acc_ratio / time_ratio if time_ratio > 0 else 0
            efficiency_scores.append(efficiency)
        
        df.loc[dataset_mask, 'efficiency_score'] = efficiency_scores
    
    return df

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    logs_dir = os.path.join(results_dir, 'logs')
    
    # Process all log files
    results = []
    log_files = glob.glob(os.path.join(logs_dir, '*.log'))
    
    print(f"Processing {len(log_files)} log files...")
    
    for log_file in log_files:
        result = extract_metrics_from_log(log_file)
        if result:
            results.append(result)
        else:
            filename = os.path.basename(log_file)
            print(f"⚠️  Failed to extract metrics from {filename}")
    
    if not results:
        print("❌ No valid results found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate efficiency scores
    df = calculate_efficiency(df)
    
    # Save detailed results
    detailed_file = os.path.join(results_dir, 'detailed_results.csv')
    df.to_csv(detailed_file, index=False)
    print(f"✅ Detailed results saved to: {detailed_file}")
    
    # Create summary
    summary_file = os.path.join(results_dir, 'summary_results.csv')
    summary_cols = ['config', 'method', 'level_ratio', 'dataset', 'accuracy', 'total_time', 'efficiency_score', 'coarsening_info']
    summary_df = df[summary_cols].copy()
    summary_df = summary_df.sort_values(['dataset', 'efficiency_score'], ascending=[True, False])
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Summary results saved to: {summary_file}")
    
    # Print summary to console
    print("\n" + "="*90)
    print("PURE COARSENING BENCHMARK RESULTS")
    print("="*90)
    
    for dataset in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset].copy()
        dataset_df = dataset_df.sort_values('efficiency_score', ascending=False)
        
        print(f"\n📊 Dataset: {dataset.upper()}")
        print("-" * 80)
        print(f"{'Rank':<4} {'Config':<12} {'Method':<8} {'Level':<6} {'Accuracy':<9} {'Time(s)':<8} {'Efficiency':<10} {'Info':<15}")
        print("-" * 80)
        
        for rank, (_, row) in enumerate(dataset_df.iterrows(), 1):
            print(f"{rank:<4} {row['config']:<12} {row['method']:<8} {row['level_ratio']:<6} {row['accuracy']:<9.3f} {row['total_time']:<8.0f} {row['efficiency_score']:<10.3f} {row['coarsening_info']:<15}")
    
    # Method comparison
    print(f"\n🏆 METHOD COMPARISON (Average Efficiency)")
    print("-" * 50)
    method_avg = df.groupby('method')['efficiency_score'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    for method, stats in method_avg.iterrows():
        print(f"{method:<8} Mean: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
    
    # Best configurations per method
    print(f"\n🎯 BEST CONFIGURATION PER METHOD")
    print("-" * 60)
    for method in sorted(df['method'].unique()):
        method_df = df[df['method'] == method]
        best_idx = method_df['efficiency_score'].idxmax()
        best_row = method_df.loc[best_idx]
        print(f"{method:<8} Best: {best_row['config']:<12} Efficiency: {best_row['efficiency_score']:.3f} (Acc: {best_row['accuracy']:.3f}, Time: {best_row['total_time']:.0f}s)")
    
    # Overall ranking
    print(f"\n🥇 TOP 5 OVERALL CONFIGURATIONS")
    print("-" * 60)
    top_configs = df.nlargest(5, 'efficiency_score')
    for i, (_, row) in enumerate(top_configs.iterrows(), 1):
        print(f"{i}. {row['config']:<12} Efficiency: {row['efficiency_score']:.3f} (Acc: {row['accuracy']:.3f}, Time: {row['total_time']:.0f}s)")
    
    print(f"\n✅ Analysis complete! Check {results_dir}/ for detailed results.")

if __name__ == "__main__":
    main()
