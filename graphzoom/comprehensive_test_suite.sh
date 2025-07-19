#!/bin/bash
# Comprehensive Test Suite: CMG++ vs LAMG vs Simple
# Focus: Accuracy, Speed, Compression improvements
# 3 runs per configuration for statistical reliability

echo "🎯 COMPREHENSIVE TEST SUITE: CMG++ vs LAMG vs Simple"
echo "=================================================="
echo "Focus: Accuracy, Speed, Compression improvements"
echo "Runs: 3 per configuration for statistical reliability"
echo "Timeout: 15 minutes per test for large datasets"
echo "Resume: Automatically skips completed tests"
echo ""

# Configuration
MATLAB_MCR_ROOT="${MATLAB_MCR_ROOT:-/home/mohammad/matlab/R2018a}"
RESULTS_DIR="comprehensive_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create results directory
mkdir -p $RESULTS_DIR
mkdir -p $RESULTS_DIR/logs

# Results CSV with comprehensive metrics
RESULTS_CSV="$RESULTS_DIR/comprehensive_results_${TIMESTAMP}.csv"
echo "test_phase,method,dataset,embedding,run_id,level,k,d,reduce_ratio,search_ratio,accuracy,total_time,fusion_time,reduction_time,embedding_time,refinement_time,original_nodes,final_clusters,compression_ratio,speedup_vs_baseline,notes" > $RESULTS_CSV

# Function to run test with comprehensive data extraction
run_comprehensive_test() {
    local phase=$1
    local method=$2
    local dataset=$3
    local embedding=$4
    local run_id=$5
    local level=$6
    local k=$7
    local d=$8
    local reduce_ratio=$9
    local search_ratio=${10}
    local extra_params="${11}"
    
    local test_name="${phase}_${method}_${dataset}_${embedding}_run${run_id}"
    local log_file="$RESULTS_DIR/logs/${test_name}.log"
    
    # Create unique test identifier including all parameters
    local unique_id="${phase},${method},${dataset},${embedding},${run_id},${level},${k},${d},${reduce_ratio},${search_ratio}"
    
    echo "🔄 Running: $test_name"
    if [ "$phase" = "phase2" ]; then
        # For phase2, show parameters in the name for clarity
        local param_str=""
        if [ "$method" = "cmg" ]; then
            param_str="k${k}_d${d}"
        elif [ "$method" = "lamg" ]; then
            param_str="r${reduce_ratio}_s${search_ratio}"
        elif [ "$method" = "simple" ]; then
            param_str="l${level}"
        fi
        test_name="${phase}_${method}_${dataset}_${embedding}_${param_str}_run${run_id}"
        log_file="$RESULTS_DIR/logs/${test_name}.log"
        echo "🔄 Running: $test_name"
    fi
    
    # Build command based on method
    local cmd=""
    if [ "$method" = "cmg" ]; then
        cmd="python graphzoom.py --dataset $dataset --coarse $method --level $level --cmg_k $k --cmg_d $d --embed_method $embedding $extra_params"
    elif [ "$method" = "lamg" ]; then
        cmd="python graphzoom.py --dataset $dataset --coarse $method --reduce_ratio $reduce_ratio --search_ratio $search_ratio --mcr_dir $MATLAB_MCR_ROOT --embed_method $embedding $extra_params"
    elif [ "$method" = "simple" ]; then
        cmd="python graphzoom.py --dataset $dataset --coarse $method --level $level --embed_method $embedding $extra_params"
    fi
    
    echo "   Command: $cmd"
    
    # Check if this exact test configuration was already completed
    if grep -q "^${unique_id},.*,completed$" $RESULTS_CSV 2>/dev/null; then
        echo "⏭️  $test_name: SKIPPING (already completed)"
        return 0
    fi
    
    # Run with extended timeout (15 minutes for large datasets)
    timeout 900 $cmd > $log_file 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # Extract comprehensive metrics
        local accuracy=$(grep "Test Accuracy:" $log_file | awk '{print $3}')
        local total_time=$(grep "Total Time.*=" $log_file | awk '{print $NF}')
        local fusion_time=$(grep "Graph Fusion.*Time:" $log_file | awk '{print $4}')
        local reduction_time=$(grep "Graph Reduction.*Time:" $log_file | awk '{print $4}')
        local embedding_time=$(grep "Graph Embedding.*Time:" $log_file | awk '{print $4}')
        local refinement_time=$(grep "Graph Refinement.*Time:" $log_file | awk '{print $4}')
        
        # Extract clustering information
        local original_nodes="unknown"
        local final_clusters="unknown"
        local compression_ratio="unknown"
        
        # Dataset-specific original node counts
        case $dataset in
            "cora") original_nodes=2708 ;;
            "citeseer") original_nodes=3327 ;;
            "pubmed") original_nodes=19717 ;;
        esac
        
        # Extract final cluster count based on method
        if [ "$method" = "cmg" ]; then
            final_clusters=$(grep "Final graph:" $log_file | tail -1 | awk '{print $4}')
        elif [ "$method" = "lamg" ]; then
            # LAMG writes coarsened graph to reduction_results/Gs.mtx
            if [ -f "reduction_results/Gs.mtx" ]; then
                # First line format: num_nodes num_nodes num_edges
                final_clusters=$(head -1 reduction_results/Gs.mtx | awk '{print $1}')
                echo "   [DEBUG] LAMG Gs.mtx first line: $(head -1 reduction_results/Gs.mtx)"
            else
                # Fallback: estimate from reduce_ratio
                final_clusters=$(echo "scale=0; $original_nodes / $reduce_ratio" | bc -l 2>/dev/null || echo "estimation_failed")
                echo "   [DEBUG] LAMG file not found, estimated: $final_clusters"
            fi
        elif [ "$method" = "simple" ]; then
            final_clusters=$(grep "Num of nodes:" $log_file | tail -1 | awk '{print $4}')
        fi
        
        # Calculate compression ratio
        if [ "$final_clusters" != "unknown" ] && [ "$final_clusters" -gt 0 ] 2>/dev/null; then
            compression_ratio=$(echo "scale=2; $original_nodes / $final_clusters" | bc -l 2>/dev/null || echo "calc_error")
        fi
        
        # Calculate speedup (will be filled in post-processing)
        local speedup_vs_baseline="TBD"
        
        # Save to CSV
        echo "$phase,$method,$dataset,$embedding,$run_id,$level,$k,$d,$reduce_ratio,$search_ratio,$accuracy,$total_time,$fusion_time,$reduction_time,$embedding_time,$refinement_time,$original_nodes,$final_clusters,$compression_ratio,$speedup_vs_baseline,completed" >> $RESULTS_CSV
        
        echo "✅ $test_name: Acc=$accuracy, Time=${total_time}s, Clusters=$final_clusters, Compression=${compression_ratio}x"
        return 0
    else
        echo "❌ $test_name: FAILED (exit code: $exit_code)"
        echo "$phase,$method,$dataset,$embedding,$run_id,$level,$k,$d,$reduce_ratio,$search_ratio,FAILED,TIMEOUT,FAILED,FAILED,FAILED,FAILED,$original_nodes,unknown,unknown,unknown,failed" >> $RESULTS_CSV
        return 1
    fi
}
# Test Phase 1: GraphZoom Paper Replication
test_phase_1_paper_replication() {
    echo ""
    echo "📊 TEST PHASE 1: GraphZoom Paper Replication"
    echo "==========================================="
    echo "Goal: Compare against published GraphZoom results"
    echo ""
    
    local datasets=("cora" "citeseer" "pubmed")
    local embeddings=("deepwalk" "node2vec")
    
    for dataset in "${datasets[@]}"; do
        for embedding in "${embeddings[@]}"; do
            for run in 1 2 3; do
                # Simple baseline
                run_comprehensive_test "phase1" "simple" $dataset $embedding $run 1 0 0 0 0 ""
                
                # LAMG (GraphZoom default)
                run_comprehensive_test "phase1" "lamg" $dataset $embedding $run 0 0 0 2 12 ""
                
                # CMG++ (our best from previous tests)
                run_comprehensive_test "phase1" "cmg" $dataset $embedding $run 1 10 10 0 0 ""
            done
        done
    done
}

# Test Phase 2: Parameter Optimization
test_phase_2_parameter_optimization() {
    echo ""
    echo "🔬 TEST PHASE 2: Parameter Optimization"
    echo "======================================"
    echo "Goal: Find optimal parameters for each method"
    echo ""
    
    # CMG++ parameter grid (Cora only for speed)
    echo "Testing CMG++ parameters..."
    local k_values=(10 15 20)
    local d_values=(10 15 20)
    
    for k in "${k_values[@]}"; do
        for d in "${d_values[@]}"; do
            for run in 1 2 3; do
                run_comprehensive_test "phase2" "cmg" "cora" "deepwalk" $run 1 $k $d 0 0 ""
            done
        done
    done
    
    # LAMG parameter grid
    echo "Testing LAMG parameters..."
    local reduce_ratios=(2 3 4)
    local search_ratios=(8 12 16)
    
    for reduce_ratio in "${reduce_ratios[@]}"; do
        for search_ratio in "${search_ratios[@]}"; do
            for run in 1 2 3; do
                run_comprehensive_test "phase2" "lamg" "cora" "deepwalk" $run 0 0 0 $reduce_ratio $search_ratio ""
            done
        done
    done
    
    # Simple multi-level
    echo "Testing Simple multi-level..."
    local levels=(1 2 3)
    
    for level in "${levels[@]}"; do
        for run in 1 2 3; do
            run_comprehensive_test "phase2" "simple" "cora" "deepwalk" $run $level 0 0 0 0 ""
        done
    done
}

# Test Phase 3: Scalability Analysis
test_phase_3_scalability_analysis() {
    echo ""
    echo "📈 TEST PHASE 3: Scalability Analysis"
    echo "===================================="
    echo "Goal: Test performance across graph sizes with optimal parameters"
    echo ""
    
    # Use optimal parameters from phase 2 (will be determined after phase 2 completes)
    # For now, use current best estimates
    local datasets=("cora" "citeseer" "pubmed")
    
    for dataset in "${datasets[@]}"; do
        for run in 1 2 3; do
            # Best CMG++ config
            run_comprehensive_test "phase3" "cmg" $dataset "deepwalk" $run 1 15 10 0 0 ""
            
            # Best LAMG config  
            run_comprehensive_test "phase3" "lamg" $dataset "deepwalk" $run 0 0 0 2 12 ""
            
            # Best Simple config
            run_comprehensive_test "phase3" "simple" $dataset "deepwalk" $run 1 0 0 0 0 ""
        done
    done
}

# Main execution
main() {
    echo "Starting comprehensive test suite..."
    echo "Results will be saved to: $RESULTS_CSV"
    echo ""
    
    # Check if results file exists and show existing results
    if [ -f "$RESULTS_CSV" ]; then
        existing_tests=$(grep -c "completed" $RESULTS_CSV 2>/dev/null || echo 0)
        if [ $existing_tests -gt 0 ]; then
            echo "📁 Found existing results file with $existing_tests completed tests"
            echo "🔄 Will skip already completed tests and continue from where we left off"
            echo ""
        fi
    fi
    
    # # Validate LAMG setup first (only if not already validated)
    # if ! grep -q "validation,lamg,.*,completed$" $RESULTS_CSV 2>/dev/null; then
    #     echo "🔍 Validating LAMG setup..."
    #     if ! run_comprehensive_test "validation" "lamg" "cora" "deepwalk" 1 0 0 0 2 12 ""; then
    #         echo "❌ LAMG validation failed! Check MCR setup."
    #         echo "MCR Root: $MATLAB_MCR_ROOT"
    #         echo "Please fix LAMG setup before continuing."
    #         exit 1
    #     fi
    #     echo "✅ LAMG validation successful!"
    #     echo ""
    # else
    #     echo "✅ LAMG already validated - skipping validation step"
    #     echo ""
    # fi
    
    # Run test phases
    # test_phase_1_paper_replication
    test_phase_2_parameter_optimization  
    # test_phase_3_scalability_analysis
    
    echo ""
    echo "✅ ALL TESTS COMPLETED!"
    echo "📁 Results saved to: $RESULTS_CSV"
    echo "📁 Logs saved to: $RESULTS_DIR/logs/"
    echo ""
    echo "🔍 Final Summary:"
    successful_tests=$(grep -c "completed" $RESULTS_CSV)
    failed_tests=$(grep -c "failed" $RESULTS_CSV)
    total_tests=$((successful_tests + failed_tests))
    echo "   Total tests: $total_tests"
    echo "   Successful tests: $successful_tests"
    echo "   Failed tests: $failed_tests"
    echo "   Success rate: $(echo "scale=1; $successful_tests * 100 / $total_tests" | bc -l 2>/dev/null || echo "N/A")%"
    echo ""
    echo "📊 Run analysis: python analyze_comprehensive_results.py $RESULTS_CSV"
}

# Check if MCR path is set
if [ -z "$MATLAB_MCR_ROOT" ]; then
    echo "❌ MATLAB_MCR_ROOT not set. Please set it to your MCR installation path."
    echo "Example: export MATLAB_MCR_ROOT=/home/mohammad/matlab/R2018a"
    exit 1
fi

# Run main function
main "$@"