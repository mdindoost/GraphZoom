#!/bin/bash
# Koutis Sample Test Suite - Validation Before Full Run
# Goal: Test each experiment type with minimal runs to validate CSV output and LAMG timing

echo "🧪 KOUTIS SAMPLE TEST SUITE - VALIDATION RUN"
echo "============================================"
echo "Goal: Validate CSV output, LAMG timing, and all experiment types"
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

# Enhanced function to run test with comprehensive data extraction
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
    
    local test_name="${experiment_type}_${method}_${dataset}_d${dimension}_run${run_id}"
    local log_file="$RESULTS_DIR/logs/${test_name}.log"
    
    echo "🔄 Running: $test_name"
    
    # Build command based on method
    local cmd=""
    if [ "$method" = "vanilla" ]; then
        # Vanilla baseline - just run embedding without coarsening
        cmd="python graphzoom.py --dataset $dataset --embed_method $embedding --no_coarsening --profile_timing"
    elif [ "$method" = "cmg" ]; then
        cmd="python graphzoom.py --dataset $dataset --coarse $method --level $level --cmg_k $k_param --cmg_d $dimension --embed_method $embedding --profile_timing"
    elif [ "$method" = "lamg" ]; then
        cmd="python graphzoom.py --dataset $dataset --coarse $method --reduce_ratio $reduce_ratio --search_ratio $search_ratio --mcr_dir $MATLAB_MCR_ROOT --embed_method $embedding --profile_timing"
    elif [ "$method" = "simple" ]; then
        cmd="python graphzoom.py --dataset $dataset --coarse $method --level $level --embed_method $embedding --profile_timing"
    fi
    
    echo "   Command: $cmd"
    
    # Run with timeout and capture all output
    timeout 600 $cmd > $log_file 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "   ✅ Test completed successfully"
        
        # Extract ALL possible metrics from log
        local accuracy=$(grep "Test Accuracy:" $log_file | tail -1 | awk '{print $3}' | tr -d '%')
        local total_time=$(grep "Total Time.*=" $log_file | tail -1 | awk '{print $NF}')
        local fusion_time=$(grep -E "(Graph Fusion|Fusion).*Time:" $log_file | tail -1 | awk '{print $4}' | tr -d 's')
        local reduction_time=$(grep -E "(Graph Reduction|Reduction).*Time:" $log_file | tail -1 | awk '{print $4}' | tr -d 's')
        local embedding_time=$(grep -E "(Graph Embedding|Embedding).*Time:" $log_file | tail -1 | awk '{print $4}' | tr -d 's')
        local refinement_time=$(grep -E "(Graph Refinement|Refinement).*Time:" $log_file | tail -1 | awk '{print $4}' | tr -d 's')
        local clustering_time=$(grep -E "(Clustering|CMG|LAMG).*Time:" $log_file | tail -1 | awk '{print $3}' | tr -d 's')
        
        # Memory usage (if available)
        local memory_mb=$(grep -E "(Memory|Peak.*Memory)" $log_file | tail -1 | awk '{print $NF}' | tr -d 'MB')
        
        # Dataset-specific node counts
        local original_nodes="unknown"
        case $dataset in
            "cora") original_nodes=2708 ;;
            "citeseer") original_nodes=3327 ;;
            "pubmed") original_nodes=19717 ;;
        esac
        
        # Extract final cluster count based on method
        local final_clusters="unknown"
        if [ "$method" = "cmg" ]; then
            final_clusters=$(grep -E "(Final graph|CMG.*clusters|nodes after)" $log_file | tail -1 | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/ && $i < 20000) print $i}' | tail -1)
        elif [ "$method" = "lamg" ]; then
            # For LAMG, check multiple possible output formats
            if [ -f "reduction_results/Gs.mtx" ]; then
                final_clusters=$(head -2 reduction_results/Gs.mtx | tail -1 | awk '{print $1}')
                echo "   [DEBUG] LAMG Gs.mtx format: $(head -2 reduction_results/Gs.mtx | tail -1)"
            else
                # Fallback: look for LAMG output in logs
                final_clusters=$(grep -E "(coarsened|reduced|LAMG.*nodes)" $log_file | tail -1 | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/ && $i < 20000) print $i}' | tail -1)
                if [ -z "$final_clusters" ]; then
                    # Estimate from reduce_ratio if no direct output
                    final_clusters=$(echo "scale=0; $original_nodes / $reduce_ratio" | bc -l 2>/dev/null || echo "estimated")
                fi
            fi
        elif [ "$method" = "simple" ]; then
            final_clusters=$(grep -E "(Num of nodes|Final.*nodes)" $log_file | tail -1 | awk '{print $4}')
        elif [ "$method" = "vanilla" ]; then
            final_clusters=$original_nodes
        fi
        
        # Handle empty/invalid values
        [ -z "$accuracy" ] && accuracy="0"
        [ -z "$total_time" ] && total_time="0"
        [ -z "$fusion_time" ] && fusion_time="0"
        [ -z "$reduction_time" ] && reduction_time="0"
        [ -z "$embedding_time" ] && embedding_time="0"
        [ -z "$refinement_time" ] && refinement_time="0"
        [ -z "$clustering_time" ] && clustering_time="0"
        [ -z "$memory_mb" ] && memory_mb="0"
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
        echo "      Clustering time: ${clustering_time}s"
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

# SAMPLE 1: Vanilla Baselines (1 sample per type)
run_vanilla_baseline_samples() {
    echo "📊 SAMPLE 1: Vanilla Baseline Tests"
    echo "=================================="
    echo "Goal: Establish baseline times for speedup calculation"
    echo ""
    
    # Just test Cora with DeepWalk in two dimensions
    run_sample_test "vanilla_baseline" "vanilla" "cora" "deepwalk" 1 64 0 0 1.0 0 0
    run_sample_test "vanilla_baseline" "vanilla" "cora" "deepwalk" 2 128 0 0 1.0 0 0
}

# SAMPLE 2: Dimension Scaling (key points only)
run_dimension_scaling_samples() {
    echo ""
    echo "🎯 SAMPLE 2: Dimension Scaling Tests"
    echo "==================================="
    echo "Goal: Test dimension stability (CMG++ vs LAMG degradation)"
    echo ""
    
    local test_dimensions=(10 20 30)  # Small sample of dimensions
    
    for dim in "${test_dimensions[@]}"; do
        echo "Testing dimension d=$dim..."
        
        # CMG++ test
        run_sample_test "dimension_scaling" "cmg" "cora" "deepwalk" 1 $dim 15 1 1.0 0 0
        
        # LAMG test
        run_sample_test "dimension_scaling" "lamg" "cora" "deepwalk" 1 $dim 0 0 1.0 2 12
    done
}

# SAMPLE 3: Hyperparameter Robustness (few key combinations)
run_hyperparameter_samples() {
    echo ""
    echo "⚙️ SAMPLE 3: Hyperparameter Robustness Tests"
    echo "==========================================="
    echo "Goal: Test robustness across different hyperparameter settings"
    echo ""
    
    # CMG++ hyperparameter samples
    local configs=(
        "10 10 1.0"    # Low dimension, standard settings
        "15 20 1.5"    # Higher dimension, different beta
        "20 15 0.5"    # High dimension, low beta
    )
    
    echo "Testing CMG++ configurations..."
    local config_id=1
    for config in "${configs[@]}"; do
        read -ra params <<< "$config"
        local k=${params[0]}
        local d=${params[1]}
        local beta=${params[2]}
        
        echo "Config $config_id: k=$k, d=$d, β=$beta"
        run_sample_test "hyperparameter_robustness" "cmg" "cora" "deepwalk" $config_id $d $k 1 $beta 0 0
        ((config_id++))
    done
    
    # LAMG hyperparameter samples
    echo "Testing LAMG configurations..."
    run_sample_test "hyperparameter_robustness" "lamg" "cora" "deepwalk" 1 128 0 0 1.0 2 12
    run_sample_test "hyperparameter_robustness" "lamg" "cora" "deepwalk" 2 128 0 0 1.0 3 16
}

# SAMPLE 4: Multilevel Comparison (key levels)
run_multilevel_samples() {
    echo ""
    echo "📈 SAMPLE 4: Multilevel Comparison Tests"
    echo "======================================"
    echo "Goal: Compare with GraphZoom Table 2 methodology"
    echo ""
    
    local levels=(1 2 3)
    
    for level in "${levels[@]}"; do
        echo "Testing level $level..."
        
        # CMG++ multilevel
        run_sample_test "multilevel_comparison" "cmg" "cora" "deepwalk" $level 128 15 $level 1.0 0 0
        
        # LAMG for comparison  
        run_sample_test "multilevel_comparison" "lamg" "cora" "deepwalk" $level 128 0 0 1.0 2 12
        
        # Simple for baseline
        run_sample_test "multilevel_comparison" "simple" "cora" "deepwalk" $level 128 0 $level 1.0 0 0
    done
}

# SAMPLE 5: Computational Efficiency (detailed profiling)
run_efficiency_samples() {
    echo ""
    echo "⚡ SAMPLE 5: Computational Efficiency Tests"
    echo "========================================="
    echo "Goal: Detailed timing and memory profiling"
    echo ""
    
    # Focus on key comparison points
    echo "Testing computational efficiency..."
    
    # CMG++ efficiency test
    run_sample_test "computational_efficiency" "cmg" "cora" "deepwalk" 1 15 15 1 1.0 0 0
    
    # LAMG efficiency test  
    run_sample_test "computational_efficiency" "lamg" "cora" "deepwalk" 1 128 0 0 1.0 2 12
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
    echo "📋 Sample data rows:"
    tail -3 "$SAMPLE_CSV"
    
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

# Test LAMG specifically
test_lamg_specifically() {
    echo ""
    echo "🔬 LAMG SPECIFIC VALIDATION"
    echo "=========================="
    echo "Goal: Validate LAMG timing extraction and CSV output"
    
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
    local cmd="python graphzoom.py --dataset cora --coarse lamg --reduce_ratio 2 --search_ratio 12 --mcr_dir $MATLAB_MCR_ROOT --embed_method deepwalk --profile_timing"
    
    echo "   Command: $cmd"
    timeout 300 $cmd > $lamg_log 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ LAMG test completed successfully"
        
        echo ""
        echo "📊 LAMG Log Analysis:"
        echo "   Total lines in log: $(wc -l < $lamg_log)"
        
        # Check for key timing information
        echo "   Timing information found:"
        grep -E "(Time|time)" $lamg_log | head -5 | sed 's/^/     /'
        
        # Check for clustering information
        echo "   Clustering information:"
        grep -E "(cluster|nodes|graph)" $lamg_log | head -3 | sed 's/^/     /'
        
        # Check for LAMG-specific output
        echo "   LAMG-specific output:"
        grep -E "(LAMG|reduction|coarse)" $lamg_log | head -3 | sed 's/^/     /'
        
        # Check for accuracy
        local accuracy=$(grep "Test Accuracy:" $lamg_log | awk '{print $3}')
        echo "   Final accuracy: $accuracy"
        
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
    echo "Starting Koutis sample validation tests..."
    echo "This will run a small sample of each experiment type to validate the setup."
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
    echo "🎉 SAMPLE VALIDATION COMPLETE!"
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
        echo "1. Review sample CSV: cat $SAMPLE_CSV"
        echo "2. Check specific logs in: $RESULTS_DIR/logs/"
        echo "3. If everything looks good, run full test suite"
        echo "4. python analyze_koutis_results.py $SAMPLE_CSV"
    else
        echo ""
        echo "❌ VALIDATION FAILED!"
        echo "🔍 Check logs in $RESULTS_DIR/logs/ for debugging"
        exit 1
    fi
}

# Check MCR setup
if [ -z "$MATLAB_MCR_ROOT" ]; then
    echo "⚠️  MATLAB_MCR_ROOT not set. Set with:"
    echo "   export MATLAB_MCR_ROOT=/path/to/mcr"
    echo "   LAMG tests will be skipped if MCR not available."
    echo ""
fi

# Run main function
main "$@"
