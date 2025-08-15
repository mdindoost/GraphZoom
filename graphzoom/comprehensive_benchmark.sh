#!/bin/bash

# Comprehensive GraphZoom Benchmark Script - Updated Version
# Tests Simple, LAMG, and CMG coarsening with different levels/ratios
# Focus: Pure coarsening comparison (NO FUSION)

# Configuration
RESULTS_DIR="coarsening_benchmark_$(date +%Y%m%d_%H%M%S)"
MCR_DIR="~/matlab/R2018a"
SEED=42
EMBED_METHOD="graphsage"

# Datasets to test
DATASETS=("cora")

# Coarsening configurations to test
declare -A CONFIGS=(
    # Simple coarsening with different levels
    ["simple_l1"]="--coarse simple --level 1 -f"
    ["simple_l2"]="--coarse simple --level 2 -f"
    ["simple_l3"]="--coarse simple --level 3 -f"
    
    # LAMG coarsening with different reduction ratios  
    ["lamg_r2"]="--coarse lamg --reduce_ratio 2 --mcr_dir $MCR_DIR -f"
    ["lamg_r3"]="--coarse lamg --reduce_ratio 3 --mcr_dir $MCR_DIR -f"
    ["lamg_r6"]="--coarse lamg --reduce_ratio 6 --mcr_dir $MCR_DIR -f"
    
    # CMG coarsening with different levels (optimal parameters k=10, d=15)
    ["cmg_l1"]="--coarse cmg --level 1 --cmg_k 10 --cmg_d 15 -f"
    ["cmg_l2"]="--coarse cmg --level 2 --cmg_k 10 --cmg_d 15 -f"
    ["cmg_l3"]="--coarse cmg --level 3 --cmg_k 10 --cmg_d 15 -f"
)

# Setup
echo "=========================================="
echo "GraphZoom Pure Coarsening Benchmark"
echo "=========================================="
echo "Focus: Coarsening methods without fusion"
echo "Results will be saved in: $RESULTS_DIR"
echo "Start time: $(date)"
echo ""

# Check dependencies
if ! command -v bc &> /dev/null; then
    echo "⚠️  Warning: 'bc' command not found. Wall clock timing may not work."
    echo "   Install with: sudo apt-get install bc (Ubuntu/Debian)"
fi

# Create results directory
mkdir -p "$RESULTS_DIR/logs"
mkdir -p "$RESULTS_DIR/summaries"

# Set up environment
export PYTHONPATH="${PYTHONPATH}:/home/mohammad/cmg-x"
echo "✅ CMG path added to PYTHONPATH"

# Function to run a single experiment
run_experiment() {
    local config_name=$1
    local dataset=$2
    local config_params=${CONFIGS[$config_name]}
    
    # Create experiment identifier
    local exp_id="${config_name}_${dataset}"
    local log_file="$RESULTS_DIR/logs/${exp_id}.log"
    
    echo "🔬 Running: $config_name on $dataset"
    echo "   Config: $config_params"
    echo "   Log: $log_file"
    
    # Build full command
    local cmd="python graphzoom_timed.py --embed_method $EMBED_METHOD --dataset $dataset --seed $SEED $config_params"
    
    # Record wall clock start time
    local wall_start_time=$(date +%s.%N)
    
    # Run experiment and capture output
    echo "Command: $cmd" > "$log_file"
    echo "Configuration: $config_name" >> "$log_file"
    echo "Wall clock start time: $(date)" >> "$log_file"
    echo "Wall clock start timestamp: $wall_start_time" >> "$log_file"
    echo "========================================" >> "$log_file"
    
    timeout 7200 $cmd >> "$log_file" 2>&1  # 2 hour timeout
    local exit_code=$?
    
    # Record wall clock end time
    local wall_end_time=$(date +%s.%N)
    local wall_duration=$(echo "$wall_end_time - $wall_start_time" | bc -l)
    
    echo "========================================" >> "$log_file"
    echo "Wall clock end time: $(date)" >> "$log_file"
    echo "Wall clock end timestamp: $wall_end_time" >> "$log_file"
    echo "Wall clock duration: ${wall_duration}s" >> "$log_file"
    echo "Exit code: $exit_code" >> "$log_file"
    
    if [ $exit_code -eq 0 ]; then
        echo "   ✅ Success (Wall time: ${wall_duration}s)"
    elif [ $exit_code -eq 124 ]; then
        echo "   ⏰ Timeout (2 hours)"
    else
        echo "   ❌ Failed (exit code: $exit_code)"
    fi
    
    echo ""
}

# Function to extract metrics from log file
extract_metrics() {
    local log_file=$1
    local accuracy=$(grep "Test Accuracy:" "$log_file" | tail -1 | awk '{print $3}')
    local total_time=$(grep "Total Time" "$log_file" | tail -1 | awk -F'= ' '{print $2}')
    local fusion_time=$(grep "Graph Fusion.*Time:" "$log_file" | tail -1 | awk '{print $4}')
    local reduction_time=$(grep "Graph Reduction.*Time:" "$log_file" | tail -1 | awk '{print $4}')
    local embedding_time=$(grep "Graph Embedding.*Time:" "$log_file" | tail -1 | awk '{print $4}')
    local refinement_time=$(grep "Graph Refinement.*Time:" "$log_file" | tail -1 | awk '{print $4}')
    
    echo "$accuracy,$total_time,$fusion_time,$reduction_time,$embedding_time,$refinement_time"
}

# Run all experiments
echo "Starting experiments..."
echo ""

# Calculate total experiments
total_experiments=0
for dataset in "${DATASETS[@]}"; do
    for config_name in "${!CONFIGS[@]}"; do
        total_experiments=$((total_experiments + 1))
    done
done

echo "Total experiments to run: $total_experiments"
echo "Configurations to test:"
for config_name in $(printf '%s\n' "${!CONFIGS[@]}" | sort); do
    echo "  - $config_name: ${CONFIGS[$config_name]}"
done
echo ""

completed_experiments=0

for dataset in "${DATASETS[@]}"; do
    echo "📊 Dataset: $dataset"
    echo "----------------------------------------"
    
    # Sort config names for consistent ordering
    for config_name in $(printf '%s\n' "${!CONFIGS[@]}" | sort); do
        run_experiment "$config_name" "$dataset"
        completed_experiments=$((completed_experiments + 1))
        echo "Progress: $completed_experiments/$total_experiments"
        echo ""
    done
    echo ""
done

echo "🔍 Processing results..."

# Create enhanced Python script to analyze results
cat > "$RESULTS_DIR/analyze_results.py" << 'EOF'
#!/usr/bin/env python3
import os
import re
import sys
import pandas as pd
import glob
import numpy as np

def extract_metrics_from_log(log_file):
    """Extract metrics from a log file with corrected node count extraction"""
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
        
        # Extract wall clock time
        wall_duration_match = re.search(r'Wall clock duration:\s+([\d.]+)s', content)
        wall_time = float(wall_duration_match.group(1)) if wall_duration_match else None
        
        # Extract coarsening info based on actual log patterns
        coarsening_info = ""
        
        if "cmg" in config_name.lower():
            # CMG pattern: "Final graph: 254 nodes"
            nodes_match = re.search(r'Final graph:\s+(\d+)\s+nodes', content)
            if nodes_match:
                coarsening_info = f"Final: {nodes_match.group(1)} nodes"
                
        elif "lamg" in config_name.lower():
            # LAMG pattern: "531 train nodes" 
            nodes_match = re.search(r'(\d+)\s+train\s+nodes', content)
            if nodes_match:
                coarsening_info = f"Final: {nodes_match.group(1)} nodes"
                
        elif "simple" in config_name.lower():
            # Simple pattern: Find LAST occurrence of "Num of nodes: XXXX"
            nodes_matches = re.findall(r'Num of nodes:\s+(\d+)', content)
            if nodes_matches:
                # Take the last match (final result after all coarsening levels)
                final_nodes = nodes_matches[-1]
                coarsening_info = f"Final: {final_nodes} nodes"
        
        # Check if experiment completed successfully
        if not accuracy_match or not total_time_match:
            return None
        
        return {
            'config': config_name,
            'method': method_type,
            'level_ratio': level_or_ratio,
            'dataset': dataset,
            'accuracy': float(accuracy_match.group(1)),
            'cpu_time': float(total_time_match.group(1)),
            'wall_time': wall_time,
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
EOF

# Run the analysis
echo "📈 Analyzing results..."
python "$RESULTS_DIR/analyze_results.py" "$RESULTS_DIR"

echo ""
echo "=========================================="
echo "Pure Coarsening Benchmark Complete!"
echo "=========================================="
echo "Results directory: $RESULTS_DIR"
echo "End time: $(date)"
echo ""
echo "📁 Files created:"
echo "  - logs/: Individual experiment logs (with wall clock timing)"
echo "  - detailed_results.csv: All metrics (CPU + Wall clock)"
echo "  - summary_results.csv: Key metrics only"
echo "  - analyze_results.py: Analysis script"
echo ""
echo "🔍 To view results:"
echo "  cat $RESULTS_DIR/summary_results.csv"
echo "  python $RESULTS_DIR/analyze_results.py $RESULTS_DIR"
echo ""
echo "📊 Key configurations tested:"
echo "  Simple: levels 1, 2, 3"
echo "  LAMG: reduction ratios 2, 3, 6"
echo "  CMG: levels 1, 2, 3 (k=10, d=15)"
echo "  All without fusion (-f flag)"
echo ""
echo "⏱️  Timing metrics:"
echo "  CPU Time: Time spent on actual computation"
echo "  Wall Clock: Real elapsed time (includes I/O, system delays)"
echo "  Both metrics used for efficiency calculations"