#!/bin/bash
# Corrected Deep Multilevel Experiment with Working LAMG
# Now we can do proper CMG++ vs LAMG comparison!

echo "🔬 CORRECTED DEEP MULTILEVEL ANALYSIS"
echo "======================================"
echo "CMG++: levels 1-6, LAMG: reduce_ratio 2-6, Simple: levels 1-6"
echo ""

MATLAB_MCR_ROOT="/home/mohammad/matlab/R2018a"
RESULTS_DIR="corrected_deep_multilevel_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p $RESULTS_DIR/logs
RESULTS_CSV="$RESULTS_DIR/corrected_deep_multilevel_${TIMESTAMP}.csv"
echo "method,parameter,dataset,accuracy,total_time,final_nodes,compression_ratio,notes" > $RESULTS_CSV

run_test() {
    local method=$1
    local param=$2
    local dataset=$3
    
    echo "🔄 Testing: $method with $param on $dataset"
    
    local cmd=""
    local param_name=""
    
    if [ "$method" = "cmg" ]; then
        cmd="python graphzoom_timed.py --dataset $dataset --coarse cmg --level $param --cmg_k 10 --cmg_d 10 --embed_method deepwalk --seed 42"
        param_name="level_$param"
    elif [ "$method" = "lamg" ]; then
        cmd="python graphzoom_timed.py --dataset $dataset --coarse lamg --reduce_ratio $param --mcr_dir $MATLAB_MCR_ROOT --embed_method deepwalk --seed 42"
        param_name="ratio_$param"
    elif [ "$method" = "simple" ]; then
        cmd="python graphzoom_timed.py --dataset $dataset --coarse simple --level $param --embed_method deepwalk --seed 42"
        param_name="level_$param"
    fi
    
    local log_file="$RESULTS_DIR/logs/${method}_${param_name}_${dataset}.log"
    
    timeout 600 $cmd > $log_file 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        local accuracy=$(grep "Test Accuracy:" $log_file | awk '{print $3}')
        local total_time=$(grep "Total Time.*=" $log_file | awk '{print $NF}')
        
        # Extract final nodes
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
        
        # Calculate compression
        local original_nodes=2708  # Cora
        local compression_ratio="unknown"
        if [ "$final_nodes" != "unknown" ] && [ "$final_nodes" -gt 0 ] 2>/dev/null; then
            compression_ratio=$(echo "scale=2; $original_nodes / $final_nodes" | bc -l)
        fi
        
        echo "$method,$param_name,$dataset,$accuracy,$total_time,$final_nodes,$compression_ratio,completed" >> $RESULTS_CSV
        echo "✅ $method $param_name: Acc=$accuracy, Nodes=$final_nodes (${compression_ratio}x), Time=${total_time}s"
        
    else
        echo "❌ $method $param_name: FAILED"
        echo "$method,$param_name,$dataset,FAILED,TIMEOUT,unknown,unknown,failed" >> $RESULTS_CSV
    fi
}

main() {
    echo "🎯 Running Corrected Deep Multilevel Comparison"
    echo ""
    
    # CMG++ levels 1-6
    echo "Testing CMG++ levels 1-6..."
    for level in 1 2 3 4 5 6; do
        run_test cmg $level cora
    done
    
    # LAMG reduce_ratio 2-6  
    echo "Testing LAMG reduce_ratio 2-6..."
    for ratio in 2 3 4 5 6; do
        run_test lamg $ratio cora
    done
    
    # Simple levels 1-6
    echo "Testing Simple levels 1-6..."
    for level in 1 2 3 4 5 6; do
        run_test simple $level cora
    done
    
    echo ""
    echo "✅ CORRECTED EXPERIMENT COMPLETE!"
    echo "Results: $RESULTS_CSV"
    
    # Quick analysis
    echo ""
    echo "📊 QUICK ANALYSIS:"
    echo "=================="
    
    echo "CMG++ Results:"
    grep "cmg," $RESULTS_CSV | grep completed | while IFS=, read method param dataset accuracy time nodes compression notes; do
        echo "  $param: ${accuracy} accuracy, ${nodes} nodes, ${compression}x compression"
    done
    
    echo ""
    echo "LAMG Results:"
    grep "lamg," $RESULTS_CSV | grep completed | while IFS=, read method param dataset accuracy time nodes compression notes; do
        echo "  $param: ${accuracy} accuracy, ${nodes} nodes, ${compression}x compression" 
    done
    
    echo ""
    echo "Simple Results:"
    grep "simple," $RESULTS_CSV | grep completed | while IFS=, read method param dataset accuracy time nodes compression notes; do
        echo "  $param: ${accuracy} accuracy, ${nodes} nodes, ${compression}x compression"
    done
}

main "$@"
