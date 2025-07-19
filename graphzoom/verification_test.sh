#!/bin/bash
# Verification Test: Clean Comparison for Koutis Email
# Test identical conditions to verify our accuracy claims

echo "🔬 VERIFICATION TEST FOR KOUTIS EMAIL"
echo "====================================="
echo "Testing all methods under identical conditions to verify claims:"
echo "1. CMG++ vs Simple vs LAMG at similar compression ratios"
echo "2. Our LAMG vs GraphZoom paper LAMG results"
echo "3. Fair comparison as suggested by Koutis"
echo ""

MATLAB_MCR_ROOT="/home/mohammad/matlab/R2018a"
RESULTS_DIR="verification_test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p $RESULTS_DIR/logs
RESULTS_CSV="$RESULTS_DIR/verification_test_${TIMESTAMP}.csv"

echo "method,config,final_nodes,accuracy,total_time,compression_ratio,speedup,notes" > $RESULTS_CSV

# Fixed seed and conditions for all tests - matching GraphZoom paper exactly
SEED=42
DATASET="cora"
EMBED_METHOD="deepwalk"

# Add note about evaluation methodology
echo "📝 EVALUATION METHODOLOGY:"
echo "========================="
echo "Using GraphZoom's standard evaluation approach:"
echo "1. Train embeddings on (coarsened) graph"
echo "2. Train Logistic Regression classifier on embeddings"
echo "3. Test on held-out test set (1000 nodes for Cora)"
echo "4. Report classification accuracy = (correct predictions) / (total predictions)"
echo "5. Same train/test split as Kipf & Welling (2016)"
echo ""

run_verification_test() {
    local method=$1
    local config=$2
    local cmd_params=$3
    local expected_nodes=$4
    local notes=$5
    
    echo "🧪 Testing: $method ($config)"
    echo "   Expected: ~$expected_nodes nodes"
    echo "   Using GraphZoom's exact evaluation: embeddings → LR classifier → test accuracy"
    
    # Build base command with GraphZoom's exact parameters
    local cmd="python graphzoom_timed.py --dataset $DATASET --embed_method $EMBED_METHOD --seed $SEED"
    
    # Add DeepWalk parameters matching GraphZoom paper (Table 2 setup)
    if [ "$EMBED_METHOD" = "deepwalk" ]; then
        # GraphZoom paper uses these DeepWalk parameters
        cmd="$cmd --num_walks 10 --walk_length 80 --window_size 10 --embedding_dim 128"
    fi
    
    # Add method-specific parameters
    if [ "$method" = "cmg" ]; then
        cmd="$cmd --coarse cmg $cmd_params"
    elif [ "$method" = "lamg" ]; then
        cmd="$cmd --coarse lamg --mcr_dir $MATLAB_MCR_ROOT $cmd_params"
    elif [ "$method" = "simple" ]; then
        cmd="$cmd --coarse simple $cmd_params"
    fi
    
    local log_name="${method}_${config//[[:space:]]/_}"
    local log_file="$RESULTS_DIR/logs/${log_name}.log"
    
    echo "Command: $cmd" > $log_file
    echo "==============================" >> $log_file
    
    # Run test
    timeout 300 $cmd >> $log_file 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # Extract results
        local accuracy=$(grep "Test Accuracy:" $log_file | awk '{print $3}')
        local total_time=$(grep "Total Time.*=" $log_file | awk '{print $NF}')
        
        # Extract final nodes based on method
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
        
        # Calculate compression and speedup
        local compression_ratio="unknown"
        local speedup="unknown"
        if [ "$final_nodes" != "unknown" ] && [ "$final_nodes" -gt 0 ] 2>/dev/null; then
            compression_ratio=$(echo "scale=2; 2708 / $final_nodes" | bc -l)
        fi
        if [ "$total_time" != "unknown" ] && [ $(echo "$total_time > 0" | bc -l) -eq 1 ] 2>/dev/null; then
            # Use DeepWalk baseline time for speedup calculation
            speedup=$(echo "scale=1; 97.8 / $total_time" | bc -l)
        fi
        
        echo "$method,$config,$final_nodes,$accuracy,$total_time,$compression_ratio,$speedup,\"$notes\"" >> $RESULTS_CSV
        echo "   ✅ Result: ${final_nodes} nodes (${compression_ratio}x), ${accuracy} accuracy, ${total_time}s"
        
        # Verify expectation
        if [ "$final_nodes" != "unknown" ] && [ "$expected_nodes" != "any" ]; then
            local diff=$(echo "scale=0; ($final_nodes - $expected_nodes)^2" | bc -l)
            if [ $(echo "$diff < 100" | bc -l) -eq 1 ]; then
                echo "   ✅ Matches expectation (~$expected_nodes nodes)"
            else
                echo "   ⚠️  Differs from expectation ($expected_nodes nodes)"
            fi
        fi
    else
        echo "$method,$config,FAILED,FAILED,TIMEOUT,FAILED,FAILED,\"$notes - FAILED\"" >> $RESULTS_CSV
        echo "   ❌ FAILED"
    fi
    
    # Clean up
    rm -rf reduction_results/* 2>/dev/null
    echo ""
}

echo "📋 VERIFICATION PLAN:"
echo "===================="
echo "1. Test GraphZoom paper's exact configurations"
echo "2. Test our CMG++ at various levels"
echo "3. Test Simple coarsening at various levels"
echo "4. Create fair compression-ratio comparison"
echo ""

# DeepWalk baseline first
echo "🔍 BASELINE: DeepWalk (No Coarsening)"
echo "====================================="
run_verification_test "baseline" "deepwalk_only" "--coarse simple --level 0" "2708" "baseline_no_coarsening"

# GraphZoom Paper LAMG Results (what they claim)
echo "🔍 LAMG: GraphZoom Paper Configurations"
echo "======================================="
run_verification_test "lamg" "graphzoom_1" "--reduce_ratio 2" "1169" "paper_graphzoom_1"
run_verification_test "lamg" "graphzoom_2" "--reduce_ratio 3" "519" "paper_graphzoom_2" 
run_verification_test "lamg" "graphzoom_3" "--reduce_ratio 6" "218" "paper_graphzoom_3"

# CMG++ at various levels
echo "🔍 CMG++: Various Levels"
echo "========================"
run_verification_test "cmg" "level_1" "--level 1 --cmg_k 10 --cmg_d 10" "any" "cmg_level_1"
run_verification_test "cmg" "level_2" "--level 2 --cmg_k 10 --cmg_d 10" "any" "cmg_level_2"
run_verification_test "cmg" "level_3" "--level 3 --cmg_k 10 --cmg_d 10" "any" "cmg_level_3"
run_verification_test "cmg" "level_4" "--level 4 --cmg_k 10 --cmg_d 10" "any" "cmg_level_4"
run_verification_test "cmg" "level_5" "--level 5 --cmg_k 10 --cmg_d 10" "any" "cmg_level_5"
run_verification_test "cmg" "level_6" "--level 6 --cmg_k 10 --cmg_d 10" "any" "cmg_level_6"

# Simple coarsening at various levels
echo "🔍 Simple: Various Levels"
echo "========================="
run_verification_test "simple" "level_1" "--level 1" "any" "simple_level_1"
run_verification_test "simple" "level_2" "--level 2" "any" "simple_level_2"
run_verification_test "simple" "level_3" "--level 3" "any" "simple_level_3"
run_verification_test "simple" "level_4" "--level 4" "any" "simple_level_4"
run_verification_test "simple" "level_5" "--level 5" "any" "simple_level_5"
run_verification_test "simple" "level_6" "--level 6" "any" "simple_level_6"

# Analysis and Summary
echo "✅ VERIFICATION TEST COMPLETE!"
echo "============================="
echo ""

# Generate comparison tables
echo "📊 RESULTS SUMMARY:"
echo "==================="
echo ""

echo "🎯 LAMG vs GraphZoom Paper Claims:"
echo "----------------------------------"
echo "Config    | Our Nodes | Our Accuracy | Paper Nodes | Paper Accuracy | Difference"
echo "----------|-----------|--------------|-------------|----------------|------------"

# LAMG results
lamg_1=$(grep "lamg,graphzoom_1" $RESULTS_CSV | cut -d',' -f3,4)
lamg_2=$(grep "lamg,graphzoom_2" $RESULTS_CSV | cut -d',' -f3,4)
lamg_3=$(grep "lamg,graphzoom_3" $RESULTS_CSV | cut -d',' -f3,4)

if [ ! -z "$lamg_1" ]; then
    nodes_1=$(echo $lamg_1 | cut -d',' -f1)
    acc_1=$(echo $lamg_1 | cut -d',' -f2)
    diff_1=$(echo "scale=1; $acc_1 - 76.9" | bc -l 2>/dev/null || echo "N/A")
    printf "GraphZoom-1 | %-9s | %-12s | %-11s | %-14s | %s\n" "$nodes_1" "$acc_1" "1169" "76.9%" "$diff_1"
fi

if [ ! -z "$lamg_2" ]; then
    nodes_2=$(echo $lamg_2 | cut -d',' -f1)
    acc_2=$(echo $lamg_2 | cut -d',' -f2)
    diff_2=$(echo "scale=1; $acc_2 - 77.3" | bc -l 2>/dev/null || echo "N/A")
    printf "GraphZoom-2 | %-9s | %-12s | %-11s | %-14s | %s\n" "$nodes_2" "$acc_2" "519" "77.3%" "$diff_2"
fi

if [ ! -z "$lamg_3" ]; then
    nodes_3=$(echo $lamg_3 | cut -d',' -f1)
    acc_3=$(echo $lamg_3 | cut -d',' -f2)
    diff_3=$(echo "scale=1; $acc_3 - 75.1" | bc -l 2>/dev/null || echo "N/A")
    printf "GraphZoom-3 | %-9s | %-12s | %-11s | %-14s | %s\n" "$nodes_3" "$acc_3" "218" "75.1%" "$diff_3"
fi

echo ""
echo "🎯 Compression-Based Fair Comparison:"
echo "------------------------------------"
echo "Method      | Level | Nodes | Compression | Accuracy | Time   | Notes"
echo "------------|-------|-------|-------------|----------|--------|-------"

# Sort by compression ratio for fair comparison
grep -v "method,config" $RESULTS_CSV | sort -t',' -k6 -n | while IFS=, read -r method config nodes accuracy time compression speedup notes; do
    if [ "$nodes" != "FAILED" ] && [ "$nodes" != "unknown" ]; then
        printf "%-11s | %-5s | %-5s | %-11s | %-8s | %-6s | %s\n" "$method" "$config" "$nodes" "${compression}x" "$accuracy" "$time" "$notes"
    fi
done

echo ""
echo "🔬 KEY VERIFICATION QUESTIONS:"
echo "=============================="
echo "1. Do our LAMG results match GraphZoom paper? [Check table above]"
echo "2. At similar compression ratios, how do methods compare?"
echo "3. Does CMG++ achieve higher max compression than LAMG?"
echo "4. Are our accuracy claims to Koutis accurate?"
echo ""
echo "📧 Based on these results, we can confidently write to Koutis!"
echo ""
echo "📁 Results saved to: $RESULTS_CSV"
echo "📁 Logs saved to: $RESULTS_DIR/logs/"
