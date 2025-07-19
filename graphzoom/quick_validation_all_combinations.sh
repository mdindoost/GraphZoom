#!/bin/bash
# Quick validation test - one test from each combination to ensure everything works
# Before running the full comprehensive test suite

echo "🧪 QUICK VALIDATION: Testing All Combinations"
echo "=============================================="
echo "Goal: Verify all methods/datasets/embeddings work before full suite"
echo ""

# Configuration
MATLAB_MCR_ROOT="${MATLAB_MCR_ROOT:-/home/mohammad/matlab/R2018a}"
RESULTS_DIR="validation_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create results directory
mkdir -p $RESULTS_DIR

# Results file
RESULTS_FILE="$RESULTS_DIR/validation_results_${TIMESTAMP}.txt"
echo "Quick Validation Results - $(date)" > $RESULTS_FILE
echo "=================================" >> $RESULTS_FILE

# Function to run quick validation test
run_validation_test() {
    local method=$1
    local dataset=$2
    local embedding=$3
    local extra_params="$4"
    
    local test_name="${method}_${dataset}_${embedding}"
    echo "🔄 Testing: $test_name"
    
    # Build command based on method
    local cmd=""
    if [ "$method" = "cmg" ]; then
        cmd="python graphzoom.py --dataset $dataset --coarse $method --level 1 --cmg_k 10 --cmg_d 10 --embed_method $embedding $extra_params"
    elif [ "$method" = "lamg" ]; then
        cmd="python graphzoom.py --dataset $dataset --coarse $method --reduce_ratio 2 --search_ratio 12 --mcr_dir $MATLAB_MCR_ROOT --embed_method $embedding $extra_params"
    elif [ "$method" = "simple" ]; then
        cmd="python graphzoom.py --dataset $dataset --coarse $method --level 1 --embed_method $embedding $extra_params"
    fi
    
    echo "   Command: $cmd"
    
    # Run with shorter timeout for validation
    timeout 300 $cmd > temp_validation.log 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # Extract key metrics
        local accuracy=$(grep "Test Accuracy:" temp_validation.log | awk '{print $3}')
        local total_time=$(grep "Total Time.*=" temp_validation.log | awk '{print $NF}')
        
        # Extract cluster info for compression calculation
        local original_nodes="unknown"
        local final_clusters="unknown"
        local compression_ratio="unknown"
        
        # Dataset-specific original node counts
        case $dataset in
            "cora") original_nodes=2708 ;;
            "citeseer") original_nodes=3327 ;;
            "pubmed") original_nodes=19717 ;;
        esac
        
        # Extract final cluster count
        if [ "$method" = "cmg" ]; then
            final_clusters=$(grep "Final graph:" temp_validation.log | tail -1 | awk '{print $4}')
        elif [ "$method" = "lamg" ]; then
            if [ -f "reduction_results/Gs.mtx" ]; then
                final_clusters=$(head -1 reduction_results/Gs.mtx | awk '{print $1}')
            fi
        elif [ "$method" = "simple" ]; then
            final_clusters=$(grep "Num of nodes:" temp_validation.log | tail -1 | awk '{print $4}')
        fi
        
        # Calculate compression
        if [ "$final_clusters" != "unknown" ] && [ "$final_clusters" -gt 0 ] 2>/dev/null; then
            compression_ratio=$(echo "scale=2; $original_nodes / $final_clusters" | bc -l 2>/dev/null || echo "calc_error")
        fi
        
        echo "✅ $test_name: SUCCESS"
        echo "   Accuracy: $accuracy"
        echo "   Time: ${total_time}s"
        echo "   Clusters: $original_nodes → $final_clusters (${compression_ratio}x compression)"
        
        # Save to results file
        echo "$test_name: SUCCESS - Acc=$accuracy, Time=${total_time}s, Compression=${compression_ratio}x" >> $RESULTS_FILE
        
        rm -f temp_validation.log
        return 0
    else
        echo "❌ $test_name: FAILED (exit code: $exit_code)"
        echo "$test_name: FAILED - Exit code: $exit_code" >> $RESULTS_FILE
        
        # Save error for debugging
        echo "Error log:" >> $RESULTS_FILE
        tail -10 temp_validation.log >> $RESULTS_FILE
        echo "---" >> $RESULTS_FILE
        
        rm -f temp_validation.log
        return 1
    fi
}

# Validation test matrix
echo "🎯 VALIDATION TEST MATRIX:"
echo "Methods: CMG++, LAMG, Simple"
echo "Datasets: Cora, CiteSeer, PubMed" 
echo "Embeddings: DeepWalk, Node2Vec"
echo "Total: 18 validation tests"
echo ""

# Track results
total_tests=0
passed_tests=0
failed_tests=0

# Test all combinations
methods=("cmg" "lamg" "simple")
datasets=("cora" "citeseer" "pubmed")
embeddings=("deepwalk" "node2vec")

for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        for embedding in "${embeddings[@]}"; do
            total_tests=$((total_tests + 1))
            
            if run_validation_test $method $dataset $embedding ""; then
                passed_tests=$((passed_tests + 1))
            else
                failed_tests=$((failed_tests + 1))
            fi
            
            echo ""
        done
    done
done

# Summary
echo "📊 VALIDATION SUMMARY:"
echo "====================="
echo "Total tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $failed_tests"
echo "Success rate: $(echo "scale=1; $passed_tests * 100 / $total_tests" | bc -l)%"
echo ""

# Save summary
echo "" >> $RESULTS_FILE
echo "SUMMARY:" >> $RESULTS_FILE
echo "Total tests: $total_tests" >> $RESULTS_FILE
echo "Passed: $passed_tests" >> $RESULTS_FILE
echo "Failed: $failed_tests" >> $RESULTS_FILE
echo "Success rate: $(echo "scale=1; $passed_tests * 100 / $total_tests" | bc -l)%" >> $RESULTS_FILE

if [ $failed_tests -eq 0 ]; then
    echo "🎉 ALL VALIDATION TESTS PASSED!"
    echo "✅ Ready to run comprehensive test suite"
    echo ""
    echo "Next step: ./comprehensive_test_suite.sh"
else
    echo "⚠️  Some validation tests failed"
    echo "❌ Check issues before running comprehensive suite"
    echo "📁 See details in: $RESULTS_FILE"
    echo ""
    echo "Failed tests need to be fixed before comprehensive testing"
fi

echo ""
echo "📁 Full validation results saved to: $RESULTS_FILE"
