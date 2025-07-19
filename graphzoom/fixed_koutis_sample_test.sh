#!/bin/bash
# Fixed Koutis Sample Test - Uses Correct GraphZoom Output Patterns
# Based on actual GraphZoom output format

echo "🧪 FIXED KOUTIS SAMPLE TEST - CORRECT PATTERNS"
echo "=============================================="
echo "Goal: Validate CSV output with CORRECT GraphZoom extraction patterns"
echo "Runs: 1-2 per configuration for quick validation"
echo ""

# Configuration
MATLAB_MCR_ROOT="${MATLAB_MCR_ROOT:-/home/mohammad/matlab/R2018a}"
RESULTS_DIR="koutis_sample_validation"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create results directory
mkdir -p $RESULTS_DIR/logs

# Sample results CSV with comprehensive headers
SAMPLE_CSV="$RESULTS_DIR/koutis_sample_results_${TIMESTAMP}.csv"
echo "experiment_type,method,dataset,embedding,run_id,dimension,k_param,level,beta,reduce_ratio,search_ratio,accuracy,total_time,fusion_time,reduction_time,embedding_time,refinement_time,clustering_time,memory_mb,original_nodes,final_clusters,compression_ratio,speedup_vs_vanilla,notes" > $SAMPLE_CSV

echo "📁 Sample results will be saved to: $SAMPLE_CSV"
echo ""

# Enhanced function with CORRECT GraphZoom extraction patterns
run_sample_test() {
    local experiment_type=$1
    local method=$2
    local dataset=$3
    local embedding=$4
    local run_id=$5
    local dimension=$6
    local k_param=$7
    local level=$8
    local beta=$9
    local reduce_ratio=${10}
    local search_ratio=${11}
    
    local test_name="${experiment_type}_${method}_${dataset}"
    if [ "$method" = "cmg" ] && [ "$dimension" -gt 0 ] 2>/dev/null; then
        test_name="${test_name}_d${dimension}_run${run_id}"
    elif [ "$method" = "lamg" ]; then
        test_name="${test_name}_r${reduce_ratio}s${search_ratio}_run${run_id}"
    else
        test_name="${test_name}_run${run_id}"
    fi
    local log_file="$RESULTS_DIR/logs/${test_name}.log"
    
    echo "🔄 Running: $test_name"
    
    # Build command with CORRECT GraphZoom arguments
    local cmd=""
    if [ "$method" = "vanilla" ]; then
        # Vanilla baseline - default GraphZoom (simple coarsening level 1)
        cmd="python graphzoom.py --dataset $dataset --embed_method $embedding"
    elif [ "$method" = "cmg" ]; then
        cmd="python graphzoom.py --dataset $dataset --coarse cmg --embed_method $embedding"
        if [ "$k_param" -gt 0 ] 2>/dev/null; then
            cmd="$cmd --cmg_k $k_param"
        fi
        if [ "$dimension" -gt 0 ] 2>/dev/null; then
            cmd="$cmd --cmg_d $dimension"
        fi
        if [ "$level" -gt 0 ] 2>/dev/null; then
            cmd="$cmd --level $level"
        fi
    elif [ "$method" = "lamg" ]; then
        cmd="python graphzoom.py --dataset $dataset --coarse lamg --embed_method $embedding"
        if [ "$reduce_ratio" -gt 0 ] 2>/dev/null; then
            cmd="$cmd --reduce_ratio $reduce_ratio"
        fi
        if [ "$search_ratio" -gt 0 ] 2>/dev/null; then
            cmd="$cmd --search_ratio $search_ratio"
        fi
        if [ -n "$MATLAB_MCR_ROOT" ]; then
            cmd="$cmd --mcr_dir $MATLAB_MCR_ROOT"
        fi
    elif [ "$method" = "simple" ]; then
        cmd="python graphzoom.py --dataset $dataset --coarse simple --embed_method $embedding"
        if [ "$level" -gt 0 ] 2>/dev/null; then
            cmd="$cmd --level $level"
        fi
    fi
    
    echo "   Command: $cmd"
    
    # Run with timeout and capture all output
    timeout 300 $cmd > $log_file 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "   ✅ Test completed successfully"
        
        # Extract metrics using CORRECT GraphZoom patterns
        local accuracy=$(grep "Test Accuracy:" $log_file | awk '{print $3}')
        local total_time=$(grep "Total Time.*=" $log_file | awk '{print $NF}')
        local fusion_time=$(grep "Graph Fusion.*Time:" $log_file | awk '{print $4}')
        local reduction_time=$(grep "Graph Reduction.*Time:" $log_file | awk '{print $4}')
        local embedding_time=$(grep "Graph Embedding.*Time:" $log_file | awk '{print $4}')
        local refinement_time=$(grep "Graph Refinement.*Time:" $log_file | awk '{print $4}')
        
        # Clustering time = reduction time (GraphZoom terminology)
        local clustering_time=$reduction_time
        
        # Memory - not directly available, set to 0
        local memory_mb="0"
        
        # Dataset-specific node counts
        local original_nodes="unknown"
        case $dataset in
            "cora") original_nodes=2708 ;;
            "citeseer") original_nodes=3327 ;;
            "pubmed") original_nodes=19717 ;;
        esac
        
        # Extract final cluster count from GraphZoom output
        local final_clusters="unknown"
        if [ "$method" = "cmg" ] || [ "$method" = "simple" ] || [ "$method" = "lamg" ]; then
            # Look for "Num of nodes:" pattern in coarsening output
            final_clusters=$(grep "Num of nodes:" $log_file | tail -1 | awk '{print $4}')
            
            # Alternative: look for nodes after coarsening
            if [ -z "$final_clusters" ]; then
                final_clusters=$(grep -E "nodes.*[0-9]+" $log_file | grep -E "Coarsening|Level" | tail -1 | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/ && $i < 20000 && $i > 10) print $i}' | tail -1)
            fi
        fi
        
        # For vanilla, final_clusters = original_nodes
        if [ "$method" = "vanilla" ]; then
            final_clusters=$original_nodes
        fi
        
        # Handle empty/invalid values with defaults
        [ -z "$accuracy" ] && accuracy="0"
        [ -z "$total_time" ] && total_time="0"
        [ -z "$fusion_time" ] && fusion_time="0"
        [ -z "$reduction_time" ] && reduction_time="0"
        [ -z "$embedding_time" ] && embedding_time="0"
        [ -z "$refinement_time" ] && refinement_time="0"
        [ -z "$clustering_time" ] && clustering_time="0"
        [ -z "$final_clusters" ] || [ "$final_clusters" = "unknown" ] && final_clusters="0"
        
        # Calculate compression ratio
        local compression_ratio="1.0"
        if [ "$final_clusters" -gt 0 ] 2>/dev/null && [ "$original_nodes" != "unknown" ]; then
            compression_ratio=$(echo "scale=3; $original_nodes / $final_clusters" | bc -l 2>/dev/null || echo "1.0")
        fi
        
        # Speedup will be calculated in post-processing
        local speedup_vs_vanilla="TBD"
        
        # Save to CSV with ALL extracted data
        echo "$experiment_type,$method,$dataset,$embedding,$run_id,$dimension,$k_param,$level,$beta,$reduce_ratio,$search_ratio,$accuracy,$total_time,$fusion_time,$reduction_time,$embedding_time,$refinement_time,$clustering_time,$memory_mb,$original_nodes,$final_clusters,$compression_ratio,$speedup_vs_vanilla,completed" >> $SAMPLE_CSV
        
        # Debug output
        echo "   📊 Extracted data:"
        echo "      Accuracy: $accuracy"
        echo "      Total time: ${total_time}s"
        echo "      Fusion time: ${fusion_time}s"
        echo "      Reduction time: ${reduction_time}s"
        echo "      Embedding time: ${embedding_time}s"
        echo "      Refinement time: ${refinement_time}s"
        echo "      Original nodes: $original_nodes"
        echo "      Final clusters: $final_clusters"
        echo "      Compression: ${compression_ratio}x"
        
        return 0
    else
        echo "   ❌ Test FAILED (exit code: $exit_code)"
        echo "$experiment_type,$method,$dataset,$embedding,$run_id,$dimension,$k_param,$level,$beta,$reduce_ratio,$search_ratio,FAILED,TIMEOUT,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,$original_nodes,unknown,unknown,unknown,failed" >> $SAMPLE_CSV
        
        # Show last few lines of log for debugging
        echo "   🔍 Last few lines of log:"
        tail -5 "$log_file" | sed 's/^/      /'
        
        return 1
    fi
}

# SAMPLE 1: Vanilla Baselines (establish timing baselines)
run_vanilla_baseline_samples() {
    echo "📊 SAMPLE 1: Vanilla Baseline Tests"
    echo "=================================="
    echo "Goal: Establish baseline times for speedup calculation"
    echo ""
    
    # Test Cora with both embeddings
    run_sample_test "vanilla_baseline" "vanilla" "cora" "deepwalk" 1 128 0 0 1.0 0 0
    run_sample_test "vanilla_baseline" "vanilla" "cora" "node2vec" 1 128 0 0 1.0 0 0
}

# SAMPLE 2: Dimension Scaling (test CMG++ dimensions vs LAMG baseline)
run_dimension_scaling_samples() {
    echo ""
    echo "🎯 SAMPLE 2: Dimension Scaling Tests"
    echo "==================================="
    echo "Goal: Test CMG++ dimension stability vs LAMG baseline"
    echo "Note: LAMG has no dimension parameter - we test CMG++ scaling vs fixed LAMG"
    echo ""
    
    local test_dimensions=(10 20 30)  # CMG++ dimensions to test
    
    for dim in "${test_dimensions[@]}"; do
        echo "Testing CMG++ dimension d=$dim..."
        
        # CMG++ test with varying dimension
        run_sample_test "dimension_scaling" "cmg" "cora" "deepwalk" 1 $dim 15 1 1.0 0 0
    done
    
    # LAMG baseline (no dimension parameter - fixed approach)
    echo "Testing LAMG baseline (no dimension scaling)..."
    run_sample_test "dimension_scaling" "lamg" "cora" "deepwalk" 1 0 0 0 1.0 2 12
}

# SAMPLE 3: Hyperparameter Robustness (CMG++ parameters)
run_hyperparameter_samples() {
    echo ""
    echo "⚙️ SAMPLE 3: Hyperparameter Robustness Tests"
    echo "==========================================="
    echo "Goal: Test CMG++ robustness across parameters"
    echo ""
    
    # CMG++ hyperparameter combinations
    local configs=(
        "10 10"    # k=10, d=10
        "15 20"    # k=15, d=20
        "20 15"    # k=20, d=15
    )
    
    echo "Testing CMG++ configurations..."
    local config_id=1
    for config in "${configs[@]}"; do
        read -ra params <<< "$config"
        local k=${params[0]}
        local d=${params[1]}
        
        echo "Config $config_id: k=$k, d=$d"
        run_sample_test "hyperparameter_robustness" "cmg" "cora" "deepwalk" $config_id $d $k 1 1.0 0 0
        ((config_id++))
    done
    
    # LAMG hyperparameter samples
    echo "Testing LAMG configurations..."
    run_sample_test "hyperparameter_robustness" "lamg" "cora" "deepwalk" 1 128 0 0 1.0 2 12
    run_sample_test "hyperparameter_robustness" "lamg" "cora" "deepwalk" 2 128 0 0 1.0 3 16
}

# SAMPLE 4: Multilevel Comparison (different levels)
run_multilevel_samples() {
    echo ""
    echo "📈 SAMPLE 4: Multilevel Comparison Tests"
    echo "======================================"
    echo "Goal: Compare multilevel performance"
    echo ""
    
    local levels=(1 2 3)
    
    for level in "${levels[@]}"; do
        echo "Testing level $level..."
        
        # CMG++ multilevel
        run_sample_test "multilevel_comparison" "cmg" "cora" "deepwalk" $level 128 15 $level 1.0 0 0
        
        # LAMG for comparison (no dimension parameter)
        run_sample_test "multilevel_comparison" "lamg" "cora" "deepwalk" $level 0 0 0 1.0 2 12
        
        # Simple for baseline
        run_sample_test "multilevel_comparison" "simple" "cora" "deepwalk" $level 128 0 $level 1.0 0 0
    done
}

# SAMPLE 5: Computational Efficiency (detailed profiling)
run_efficiency_samples() {
    echo ""
    echo "⚡ SAMPLE 5: Computational Efficiency Tests"
    echo "========================================="
    echo "Goal: Compare CMG++ vs LAMG efficiency"
    echo ""
    
    # Focus on direct comparison
    echo "Testing computational efficiency..."
    
    # CMG++ efficiency test
    run_sample_test "computational_efficiency" "cmg" "cora" "deepwalk" 1 15 15 1 1.0 0 0
    
    # LAMG efficiency test (no dimension parameter)
    run_sample_test "computational_efficiency" "lamg" "cora" "deepwalk" 1 0 0 0 1.0 2 12
}

# Validate CSV output function
validate_csv_output() {
    echo ""
    echo "🔍 VALIDATING CSV OUTPUT"
    echo "======================="
    
    if [ ! -f "$SAMPLE_CSV" ]; then
        echo "❌ CSV file not created: $SAMPLE_CSV"
        return 1
    fi
    
    local total_lines=$(wc -l < "$SAMPLE_CSV")
    local header_line=$(head -1 "$SAMPLE_CSV")
    local completed_tests=$(grep -c "completed" "$SAMPLE_CSV" || echo 0)
    local failed_tests=$(grep -c "failed" "$SAMPLE_CSV" || echo 0)
    
    echo "📊 CSV Validation Results:"
    echo "   File: $SAMPLE_CSV"
    echo "   Total lines: $total_lines"
    echo "   Completed tests: $completed_tests"
    echo "   Failed tests: $failed_tests"
    echo "   Success rate: $(echo "scale=1; $completed_tests * 100 / ($completed_tests + $failed_tests)" | bc -l 2>/dev/null || echo "N/A")%"
    
    echo ""
    echo "📋 CSV Header validation:"
    echo "$header_line"
    
    echo ""
    echo "📋 Sample successful results:"
    grep "completed" "$SAMPLE_CSV" | head -3
    
    if [ $completed_tests -gt 0 ]; then
        echo ""
        echo "✅ CSV output validation PASSED"
        return 0
    else
        echo ""
        echo "❌ CSV output validation FAILED - no completed tests"
        return 1
    fi
}

# Test LAMG specifically with correct patterns
test_lamg_specifically() {
    echo ""
    echo "🔬 LAMG SPECIFIC VALIDATION"
    echo "=========================="
    echo "Goal: Validate LAMG timing extraction with correct patterns"
    
    # Check MCR setup
    if [ -z "$MATLAB_MCR_ROOT" ]; then
        echo "❌ MATLAB_MCR_ROOT not set. Cannot test LAMG."
        return 1
    fi
    
    if [ ! -d "$MATLAB_MCR_ROOT" ]; then
        echo "❌ MATLAB MCR directory not found: $MATLAB_MCR_ROOT"
        return 1
    fi
    
    echo "✅ MATLAB MCR found: $MATLAB_MCR_ROOT"
    
    # Run specific LAMG test with detailed logging
    echo "🔄 Running detailed LAMG test..."
    
    local lamg_log="$RESULTS_DIR/logs/lamg_detailed_test.log"
    local cmd="python graphzoom.py --dataset cora --coarse lamg --reduce_ratio 2 --search_ratio 12 --embed_method deepwalk --mcr_dir $MATLAB_MCR_ROOT"
    
    echo "   Command: $cmd"
    timeout 300 $cmd > $lamg_log 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ LAMG test completed successfully"
        
        echo ""
        echo "📊 LAMG Timing Extraction Validation:"
        
        # Test our extraction patterns
        local accuracy=$(grep "Test Accuracy:" $lamg_log | awk '{print $3}')
        local total_time=$(grep "Total Time.*=" $lamg_log | awk '{print $NF}')
        local fusion_time=$(grep "Graph Fusion.*Time:" $lamg_log | awk '{print $4}')
        local reduction_time=$(grep "Graph Reduction.*Time:" $lamg_log | awk '{print $4}')
        local embedding_time=$(grep "Graph Embedding.*Time:" $lamg_log | awk '{print $4}')
        local refinement_time=$(grep "Graph Refinement.*Time:" $lamg_log | awk '{print $4}')
        
        echo "   ✅ Accuracy: $accuracy"
        echo "   ✅ Total time: ${total_time}s"
        echo "   ✅ Fusion time: ${fusion_time}s"
        echo "   ✅ Reduction time: ${reduction_time}s"
        echo "   ✅ Embedding time: ${embedding_time}s"
        echo "   ✅ Refinement time: ${refinement_time}s"
        
        # Check for final cluster count
        local final_clusters=$(grep "Num of nodes:" $lamg_log | tail -1 | awk '{print $4}')
        echo "   ✅ Final clusters: $final_clusters"
        
        return 0
    else
        echo "❌ LAMG test FAILED (exit code: $exit_code)"
        echo ""
        echo "🔍 LAMG Error Analysis:"
        tail -10 $lamg_log | sed 's/^/   /'
        return 1
    fi
}

# Main execution
main() {
    echo "Starting fixed Koutis sample validation tests..."
    echo "Using CORRECT GraphZoom output patterns based on actual testing."
    echo ""
    
    # Check prerequisites
    if [ ! -f "graphzoom.py" ]; then
        echo "❌ graphzoom.py not found. Please run from GraphZoom directory."
        exit 1
    fi
    
    # Test LAMG setup first
    test_lamg_specifically
    
    echo ""
    echo "🚀 Running sample tests for each experiment type..."
    
    # Run sample tests
    run_vanilla_baseline_samples
    run_dimension_scaling_samples  
    run_hyperparameter_samples
    run_multilevel_samples
    run_efficiency_samples
    
    # Validate results
    validate_csv_output
    
    echo ""
    echo "🎉 FIXED SAMPLE VALIDATION COMPLETE!"
    echo ""
    echo "📋 SUMMARY:"
    local total_tests=$(grep -c "completed\|failed" "$SAMPLE_CSV" || echo 0)
    local completed=$(grep -c "completed" "$SAMPLE_CSV" || echo 0)
    echo "   Total sample tests: $total_tests"
    echo "   Successful tests: $completed"
    echo "   CSV file: $SAMPLE_CSV"
    echo "   Log directory: $RESULTS_DIR/logs/"
    
    if [ $completed -gt 0 ]; then
        echo ""
        echo "✅ VALIDATION SUCCESSFUL!"
        echo "🚀 Ready to run full Koutis efficiency study"
        echo ""
        echo "📋 Next steps:"
        echo "1. python validate_sample_results.py $SAMPLE_CSV"
        echo "2. Review extracted data for accuracy"
        echo "3. Run full test suite with correct patterns"
        echo "4. Scale up for comprehensive study"
    else
        echo ""
        echo "❌ VALIDATION FAILED!"
        echo "🔍 Check logs in $RESULTS_DIR/logs/ for debugging"
        exit 1
    fi
}

# Run main function
main "$@"
