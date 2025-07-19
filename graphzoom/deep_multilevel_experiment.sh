#!/bin/bash
# Deep Multilevel Experiment: Test Koutis's Hypothesis
# Goal: Show GraphZoom's peaks are "incidental" while CMG++ is "principled"
# Test levels 1-6 to see where each method breaks down

echo "🔬 DEEP MULTILEVEL ANALYSIS: CMG++ vs LAMG vs Simple"
echo "===================================================="
echo "Testing Koutis's hypothesis: GraphZoom peaks are incidental"
echo "CMG++ should show principled behavior at deep levels"
echo ""

# Configuration
MATLAB_MCR_ROOT="${MATLAB_MCR_ROOT:-/home/mohammad/matlab/R2018a}"
RESULTS_DIR="deep_multilevel_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create results directory
mkdir -p $RESULTS_DIR
mkdir -p $RESULTS_DIR/logs

# Results CSV
RESULTS_CSV="$RESULTS_DIR/deep_multilevel_${TIMESTAMP}.csv"
echo "method,dataset,level,accuracy,total_time,fusion_time,reduction_time,embedding_time,refinement_time,original_nodes,final_nodes,compression_ratio,notes" > $RESULTS_CSV

# Function to run single test
run_deep_test() {
    local method=$1
    local dataset=$2
    local level=$3
    local run_id=$4
    
    local test_name="${method}_${dataset}_level${level}_run${run_id}"
    local log_file="$RESULTS_DIR/logs/${test_name}.log"
    
    echo "🔄 Running: $test_name"
    
    # Build command based on method
    local cmd=""
    if [ "$method" = "cmg" ]; then
        cmd="python graphzoom_timed.py --dataset $dataset --coarse cmg --level $level --cmg_k 10 --cmg_d 10 --embed_method deepwalk --seed 42"
    elif [ "$method" = "lamg" ]; then
        cmd="python graphzoom_timed.py --dataset $dataset --coarse lamg --level $level --reduce_ratio 2 --search_ratio 12 --mcr_dir $MATLAB_MCR_ROOT --embed_method deepwalk --seed 42"
    elif [ "$method" = "simple" ]; then
        cmd="python graphzoom_timed.py --dataset $dataset --coarse simple --level $level --embed_method deepwalk --seed 42"
    fi
    
    echo "   Command: $cmd"
    
    # Run with timeout (10 minutes for deep levels)
    timeout 600 $cmd > $log_file 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # Extract metrics
        local accuracy=$(grep "Test Accuracy:" $log_file | awk '{print $3}')
        local total_time=$(grep "Total Time.*=" $log_file | awk '{print $NF}')
        local fusion_time=$(grep "Graph Fusion.*Time:" $log_file | awk '{print $4}')
        local reduction_time=$(grep "Graph Reduction.*Time:" $log_file | awk '{print $4}')
        local embedding_time=$(grep "Graph Embedding.*Time:" $log_file | awk '{print $4}')
        local refinement_time=$(grep "Graph Refinement.*Time:" $log_file | awk '{print $4}')
        
        # Dataset-specific original node counts
        local original_nodes="unknown"
        case $dataset in
            "cora") original_nodes=2708 ;;
            "citeseer") original_nodes=3327 ;;
            "pubmed") original_nodes=19717 ;;
        esac
        
        # Extract final node count
        local final_nodes="unknown"
        if [ "$method" = "cmg" ]; then
            final_nodes=$(grep "Final graph:" $log_file | tail -1 | awk '{print $4}')
        elif [ "$method" = "lamg" ]; then
            if [ -f "reduction_results/Gs.mtx" ]; then
                final_nodes=$(head -1 reduction_results/Gs.mtx | awk '{print $1}')
            fi
        elif [ "$method" = "simple" ]; then
            final_nodes=$(grep "Num of nodes:" $log_file | tail -1 | awk '{print $4}')
        fi
        
        # Calculate compression ratio
        local compression_ratio="unknown"
        if [ "$final_nodes" != "unknown" ] && [ "$final_nodes" -gt 0 ] 2>/dev/null; then
            compression_ratio=$(echo "scale=2; $original_nodes / $final_nodes" | bc -l 2>/dev/null || echo "calc_error")
        fi
        
        # Save results
        echo "$method,$dataset,$level,$accuracy,$total_time,$fusion_time,$reduction_time,$embedding_time,$refinement_time,$original_nodes,$final_nodes,$compression_ratio,completed" >> $RESULTS_CSV
        
        echo "✅ $test_name: Acc=$accuracy, Nodes: $original_nodes→$final_nodes (${compression_ratio}x), Time=${total_time}s"
        return 0
    else
        echo "❌ $test_name: FAILED (exit code: $exit_code)"
        echo "$method,$dataset,$level,FAILED,TIMEOUT,FAILED,FAILED,FAILED,FAILED,$original_nodes,unknown,unknown,failed" >> $RESULTS_CSV
        return 1
    fi
}

# Main experiment function
run_deep_multilevel_experiment() {
    echo ""
    echo "🎯 DEEP MULTILEVEL EXPERIMENT: Levels 1-6"
    echo "=========================================="
    echo "Goal: Test where each method breaks down"
    echo ""
    
    # Test parameters
    local datasets=("cora")  # Start with Cora, can expand to citeseer pubmed
    local methods=("simple" "cmg" "lamg")
    local levels=(1 2 3 4 5 6)
    local runs=1  # Single run first, can increase for statistical validation
    
    echo "📊 Experiment Matrix:"
    echo "   Datasets: ${datasets[@]}"
    echo "   Methods: ${methods[@]}"
    echo "   Levels: ${levels[@]}"
    echo "   Runs per config: $runs"
    echo "   Total tests: $((${#datasets[@]} * ${#methods[@]} * ${#levels[@]} * runs))"
    echo ""
    
    local total_tests=0
    local successful_tests=0
    local failed_tests=0
    
    # Run experiments
    for dataset in "${datasets[@]}"; do
        echo ""
        echo "📁 Testing Dataset: $dataset"
        echo "------------------------"
        
        for level in "${levels[@]}"; do
            echo ""
            echo "📊 Level $level Testing:"
            
            for method in "${methods[@]}"; do
                for run in $(seq 1 $runs); do
                    total_tests=$((total_tests + 1))
                    
                    if run_deep_test $method $dataset $level $run; then
                        successful_tests=$((successful_tests + 1))
                    else
                        failed_tests=$((failed_tests + 1))
                    fi
                done
            done
            
            # Quick level summary
            echo "   Level $level complete: $(grep "level,$level," $RESULTS_CSV | grep -c "completed") successes"
        done
    done
    
    echo ""
    echo "✅ DEEP MULTILEVEL EXPERIMENT COMPLETE!"
    echo "======================================"
    echo "📊 Results Summary:"
    echo "   Total tests: $total_tests"
    echo "   Successful: $successful_tests"
    echo "   Failed: $failed_tests"
    echo "   Success rate: $(echo "scale=1; $successful_tests * 100 / $total_tests" | bc -l)%"
    echo ""
    echo "📁 Results saved to: $RESULTS_CSV"
    echo "📁 Logs saved to: $RESULTS_DIR/logs/"
}

# Analysis function (run after experiments)
generate_analysis() {
    echo ""
    echo "🔍 GENERATING ANALYSIS..."
    echo "========================"
    
    # Create analysis script
    cat > $RESULTS_DIR/analyze_deep_multilevel.py << 'EOF'
#!/usr/bin/env python3
"""
Analysis of Deep Multilevel Experiment Results
Focus: Testing Koutis's hypothesis about incidental vs principled behavior
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def analyze_deep_multilevel(csv_path):
    """Analyze deep multilevel experiment results"""
    
    df = pd.read_csv(csv_path)
    df = df[df['accuracy'] != 'FAILED'].copy()
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df['compression_ratio'] = pd.to_numeric(df['compression_ratio'], errors='coerce')
    
    print("🔬 DEEP MULTILEVEL ANALYSIS")
    print("===========================")
    print(f"Successful tests: {len(df)}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Levels: {sorted(df['level'].unique())}")
    print()
    
    # Koutis's Hypothesis Test: Look for accuracy trajectory patterns
    print("🎯 TESTING KOUTIS'S HYPOTHESIS:")
    print("GraphZoom peaks should be 'incidental', CMG++ should be 'principled'")
    print()
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        print(f"📊 {dataset.upper()} RESULTS:")
        print("-" * 40)
        
        # Show accuracy trajectory for each method
        for method in ['simple', 'cmg', 'lamg']:
            method_df = dataset_df[dataset_df['method'] == method].sort_values('level')
            if not method_df.empty:
                accuracies = method_df['accuracy'].values
                levels = method_df['level'].values
                compressions = method_df['compression_ratio'].values
                
                print(f"\n{method.upper()} Trajectory:")
                for i, (lvl, acc, comp) in enumerate(zip(levels, accuracies, compressions)):
                    comp_str = f"{comp:.1f}x" if not pd.isna(comp) else "N/A"
                    print(f"  Level {lvl}: {acc:.3f} accuracy ({comp_str} compression)")
                
                # Analyze trajectory pattern
                if len(accuracies) >= 3:
                    peak_level = levels[np.argmax(accuracies)]
                    peak_accuracy = np.max(accuracies)
                    final_accuracy = accuracies[-1]
                    
                    print(f"  → Peak at Level {peak_level}: {peak_accuracy:.3f}")
                    print(f"  → Final (Level {levels[-1]}): {final_accuracy:.3f}")
                    print(f"  → Pattern: {'Incidental Peak' if peak_level in [2,3] else 'Other'}")
        print()
    
    # Compression efficiency analysis
    print("📈 COMPRESSION EFFICIENCY:")
    print("-" * 30)
    for method in ['simple', 'cmg', 'lamg']:
        method_df = df[df['method'] == method]
        if not method_df.empty:
            max_compression = method_df['compression_ratio'].max()
            max_level = method_df['level'].max()
            print(f"{method.upper()}: Max {max_compression:.1f}x compression at level {max_level}")
    print()
    
    # Time efficiency
    print("⚡ TIME EFFICIENCY:")
    print("-" * 20)
    for method in ['simple', 'cmg', 'lamg']:
        method_df = df[df['method'] == method]
        if not method_df.empty:
            avg_time = method_df['total_time'].mean()
            print(f"{method.upper()}: {avg_time:.1f}s average total time")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_deep_multilevel.py <results.csv>")
        sys.exit(1)
    
    analyze_deep_multilevel(sys.argv[1])
EOF
    
    # Run analysis
    python $RESULTS_DIR/analyze_deep_multilevel.py $RESULTS_CSV
}

# Main execution
main() {
    echo "Starting deep multilevel experiment..."
    echo "Results will be saved to: $RESULTS_CSV"
    echo ""
    
    # Check LAMG setup
    if [ -z "$MATLAB_MCR_ROOT" ]; then
        echo "❌ MATLAB_MCR_ROOT not set. Please set it first."
        exit 1
    fi
    
    if [ ! -d "$MATLAB_MCR_ROOT" ]; then
        echo "❌ MATLAB MCR directory not found: $MATLAB_MCR_ROOT"
        exit 1
    fi
    
    # Run experiment
    run_deep_multilevel_experiment
    
    # Generate analysis
    generate_analysis
    
    echo ""
    echo "🎯 EXPERIMENT COMPLETE!"
    echo "======================"
    echo "📁 All results in: $RESULTS_DIR/"
    echo "🔬 This tests Koutis's hypothesis about GraphZoom's incidental peaks"
    echo "📊 Next: Analyze trajectory patterns to see which method is principled"
}

# Run main function
main "$@"
