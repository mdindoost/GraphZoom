#!/bin/bash
# LAMG Behavior Investigation
# Test all possible parameter combinations to understand LAMG behavior

echo "🔬 LAMG BEHAVIOR INVESTIGATION"
echo "=============================="
echo "Testing all possible LAMG parameter combinations to understand:"
echo "1. Is LAMG controlled by --level or --reduce_ratio?"
echo "2. What parameters actually affect final graph size?"
echo "3. How do parameters map to GraphZoom-1/2/3 results?"
echo ""

MATLAB_MCR_ROOT="/home/mohammad/matlab/R2018a"
RESULTS_DIR="lamg_investigation_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p $RESULTS_DIR/logs
RESULTS_CSV="$RESULTS_DIR/lamg_investigation_${TIMESTAMP}.csv"

echo "test_type,param_name,param_value,final_nodes,accuracy,total_time,compression_ratio,reduce_ratio,level,search_ratio,notes" > $RESULTS_CSV

# Function to run LAMG test and extract results
run_lamg_test() {
    local test_type=$1
    local param_name=$2
    local param_value=$3
    local reduce_ratio=$4
    local level=$5
    local search_ratio=$6
    local extra_note=$7
    
    echo "🧪 Testing: $test_type | $param_name=$param_value"
    
    # Build command
    local cmd="python graphzoom_timed.py --dataset cora --coarse lamg --mcr_dir $MATLAB_MCR_ROOT --embed_method deepwalk --seed 42"
    
    # Add parameters
    if [ "$reduce_ratio" != "default" ]; then
        cmd="$cmd --reduce_ratio $reduce_ratio"
    fi
    if [ "$level" != "default" ]; then
        cmd="$cmd --level $level"
    fi
    if [ "$search_ratio" != "default" ]; then
        cmd="$cmd --search_ratio $search_ratio"
    fi
    
    local log_name="${test_type}_${param_name}_${param_value}"
    local log_file="$RESULTS_DIR/logs/${log_name}.log"
    
    echo "Command: $cmd" > $log_file
    echo "======================" >> $log_file
    
    # Run with timeout
    timeout 300 $cmd >> $log_file 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        local accuracy=$(grep "Test Accuracy:" $log_file | awk '{print $3}')
        local total_time=$(grep "Total Time.*=" $log_file | awk '{print $NF}')
        
        # Extract final nodes from LAMG output
        local final_nodes="unknown"
        if [ -f "reduction_results/Gs.mtx" ]; then
            final_nodes=$(head -1 reduction_results/Gs.mtx | awk '{print $1}')
        else
            # Try to extract from log
            final_nodes=$(grep -i "coarsened.*nodes" $log_file | tail -1 | grep -o '[0-9]\+' | tail -1)
        fi
        
        # Calculate compression
        local compression_ratio="unknown"
        if [ "$final_nodes" != "unknown" ] && [ "$final_nodes" -gt 0 ] 2>/dev/null; then
            compression_ratio=$(echo "scale=2; 2708 / $final_nodes" | bc -l)
        fi
        
        echo "$test_type,$param_name,$param_value,$final_nodes,$accuracy,$total_time,$compression_ratio,$reduce_ratio,$level,$search_ratio,\"$extra_note\"" >> $RESULTS_CSV
        
        echo "   ✅ Result: ${final_nodes} nodes, ${compression_ratio}x compression, ${accuracy} accuracy"
    else
        echo "$test_type,$param_name,$param_value,FAILED,FAILED,TIMEOUT,FAILED,$reduce_ratio,$level,$search_ratio,\"$extra_note - FAILED\"" >> $RESULTS_CSV
        echo "   ❌ FAILED"
    fi
    
    # Clean up intermediate files
    rm -rf reduction_results/* 2>/dev/null
    echo ""
}

echo "📋 INVESTIGATION PLAN:"
echo "====================="
echo "Test 1: Reduce Ratio Sweep (default level)"
echo "Test 2: Level Sweep (default reduce_ratio)" 
echo "Test 3: Cross Product (reduce_ratio × level)"
echo "Test 4: Search Ratio Impact"
echo "Test 5: GraphZoom Paper Reproduction"
echo ""

# Test 1: Reduce Ratio Sweep (keeping level default)
echo "🔍 TEST 1: REDUCE_RATIO SWEEP"
echo "============================="
echo "Testing reduce_ratio from 2 to 20 with default level"
echo ""

for reduce_ratio in 2 3 4 5 6 7 8 9 10 12 15 20; do
    run_lamg_test "reduce_ratio_sweep" "reduce_ratio" "$reduce_ratio" "$reduce_ratio" "default" "12" "level=default"
done

# Test 2: Level Sweep (keeping reduce_ratio default)  
echo "🔍 TEST 2: LEVEL SWEEP"
echo "======================"
echo "Testing level from 1 to 10 with default reduce_ratio"
echo ""

for level in 1 2 3 4 5 6 7 8 9 10; do
    run_lamg_test "level_sweep" "level" "$level" "default" "$level" "12" "reduce_ratio=default"
done

# Test 3: Cross Product - Key Combinations
echo "🔍 TEST 3: CROSS PRODUCT (reduce_ratio × level)"
echo "==============================================="
echo "Testing key combinations to see parameter interaction"
echo ""

# Test combinations that might match GraphZoom-1/2/3 results
combinations=(
    "2,1,GraphZoom-1_candidate_1"
    "2,2,GraphZoom-2_candidate_1" 
    "2,3,GraphZoom-3_candidate_1"
    "3,1,GraphZoom-1_candidate_2"
    "3,2,GraphZoom-2_candidate_2"
    "3,3,GraphZoom-3_candidate_2"
    "4,1,GraphZoom-1_candidate_3"
    "4,2,GraphZoom-2_candidate_3"
    "4,3,GraphZoom-3_candidate_3"
    "6,1,reduce_6_level_1"
    "6,2,reduce_6_level_2"
    "6,3,reduce_6_level_3"
)

for combo in "${combinations[@]}"; do
    IFS=',' read -r reduce_ratio level note <<< "$combo"
    run_lamg_test "cross_product" "reduce_ratio_${reduce_ratio}_level_${level}" "${reduce_ratio}_${level}" "$reduce_ratio" "$level" "12" "$note"
done

# Test 4: Search Ratio Impact
echo "🔍 TEST 4: SEARCH RATIO IMPACT"  
echo "==============================="
echo "Testing if search_ratio affects coarsening (keeping reduce_ratio=3, level=2)"
echo ""

for search_ratio in 4 8 12 16 20 24; do
    run_lamg_test "search_ratio_impact" "search_ratio" "$search_ratio" "3" "2" "$search_ratio" "reduce_ratio=3,level=2"
done

# Test 5: Try to Reproduce GraphZoom Paper Results Exactly
echo "🔍 TEST 5: GRAPHZOOM PAPER REPRODUCTION ATTEMPT"
echo "==============================================="
echo "Trying to match exact GraphZoom paper results:"
echo "GraphZoom-1: 1169 nodes, GraphZoom-2: 519 nodes, GraphZoom-3: 218 nodes"
echo ""

# Try different parameter combinations that might reproduce paper results
reproduction_tests=(
    "default,1,12,attempt_graphzoom_1_default_params"
    "default,2,12,attempt_graphzoom_2_default_params"
    "default,3,12,attempt_graphzoom_3_default_params"
    "2,default,12,attempt_graphzoom_reduce_2"
    "3,default,12,attempt_graphzoom_reduce_3"
    "4,default,12,attempt_graphzoom_reduce_4"
)

for test in "${reproduction_tests[@]}"; do
    IFS=',' read -r reduce_ratio level search_ratio note <<< "$test"
    run_lamg_test "reproduction_attempt" "paper_reproduction" "$note" "$reduce_ratio" "$level" "$search_ratio" "$note"
done

# Analysis and Summary
echo "✅ LAMG INVESTIGATION COMPLETE!"
echo "==============================="
echo ""
echo "📊 ANALYSIS:"
echo "============"

echo ""
echo "🎯 REDUCE_RATIO RESULTS:"
echo "------------------------"
echo "Reduce_Ratio | Final_Nodes | Compression | Accuracy"
echo "-------------|-------------|-------------|----------"
grep "reduce_ratio_sweep" $RESULTS_CSV | while IFS=, read -r test_type param_name param_value final_nodes accuracy total_time compression_ratio reduce_ratio level search_ratio notes; do
    printf "%-12s | %-11s | %-11s | %s\n" "$param_value" "$final_nodes" "$compression_ratio" "$accuracy"
done

echo ""
echo "🎯 LEVEL RESULTS:"
echo "----------------"
echo "Level | Final_Nodes | Compression | Accuracy"
echo "------|-------------|-------------|----------"
grep "level_sweep" $RESULTS_CSV | while IFS=, read -r test_type param_name param_value final_nodes accuracy total_time compression_ratio reduce_ratio level search_ratio notes; do
    printf "%-5s | %-11s | %-11s | %s\n" "$param_value" "$final_nodes" "$compression_ratio" "$accuracy"
done

echo ""
echo "🎯 CROSS PRODUCT KEY RESULTS:"
echo "----------------------------"
echo "Reduce_Ratio × Level | Final_Nodes | Compression | Note"
echo "--------------------|-------------|-------------|------"
grep "cross_product" $RESULTS_CSV | while IFS=, read -r test_type param_name param_value final_nodes accuracy total_time compression_ratio reduce_ratio level search_ratio notes; do
    printf "%-19s | %-11s | %-11s | %s\n" "${reduce_ratio}×${level}" "$final_nodes" "$compression_ratio" "$notes"
done

echo ""
echo "🔬 KEY FINDINGS:"
echo "==============="
echo "1. Parameter that controls coarsening: [To be determined from results]"
echo "2. GraphZoom-X mapping: [To be determined from results]"
echo "3. Why LAMG gets 'stuck': [To be determined from results]"
echo "4. Reproduction of paper results: [To be determined from results]"
echo ""
echo "📁 Detailed results saved to: $RESULTS_CSV"
echo "📁 Logs saved to: $RESULTS_DIR/logs/"

# Generate summary report
echo ""
echo "📋 SUMMARY REPORT FOR KOUTIS:"
echo "============================="
echo "This investigation tested all possible LAMG parameter combinations to understand:"
echo "- Whether LAMG is controlled by --level or --reduce_ratio"
echo "- How parameters map to GraphZoom-1/2/3 paper results"  
echo "- Why LAMG appears to 'get stuck' at certain node counts"
echo ""
echo "Results will clarify the 'weird' LAMG behavior and validate our experimental setup."
