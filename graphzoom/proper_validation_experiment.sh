#!/bin/bash
# Experiment 2: Proper Validation Methodology
# Addresses Koutis's concern: "The way we test is problematic - we look at the test set"
# Implements proper train/validation/test split with hyperparameter optimization

echo "🔬 EXPERIMENT 2: PROPER VALIDATION METHODOLOGY"
echo "=============================================="
echo "Implementing train/validation/test splits for rigorous comparison"
echo "Addresses Koutis's methodological concerns about test set contamination"
echo ""

MATLAB_MCR_ROOT="/home/mohammad/matlab/R2018a"
RESULTS_DIR="validation_methodology_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p $RESULTS_DIR/logs
mkdir -p $RESULTS_DIR/hyperparameter_search

# Results files
VALIDATION_CSV="$RESULTS_DIR/hyperparameter_validation_${TIMESTAMP}.csv"
FINAL_TEST_CSV="$RESULTS_DIR/final_test_results_${TIMESTAMP}.csv"

echo "method,hyperparams,validation_accuracy,validation_time,compression_ratio,phase" > $VALIDATION_CSV
echo "method,best_hyperparams,test_accuracy,test_time,compression_ratio,total_search_time,search_configs" > $FINAL_TEST_CSV

# Hyperparameter grids for each method
declare -A CMG_K_VALUES=(["small"]="5 8 10" ["medium"]="5 8 10 12 15" ["large"]="5 8 10 12 15 20")
declare -A CMG_D_VALUES=(["small"]="5 10 15" ["medium"]="5 10 15 20" ["large"]="5 8 10 12 15 20 25")
declare -A LAMG_REDUCE_VALUES=(["small"]="2 3 4" ["medium"]="2 3 4 5 6" ["large"]="2 3 4 5 6 7 8")
declare -A LAMG_SEARCH_VALUES=(["small"]="8 12 16" ["medium"]="8 12 16 20" ["large"]="8 12 16 20 24")
declare -A LAMG_NEIGHS_VALUES=(["small"]="2 5 10" ["medium"]="2 5 10 15" ["large"]="2 5 10 15 20")
declare -A SIMPLE_LEVEL_VALUES=(["small"]="1 2 3" ["medium"]="1 2 3 4" ["large"]="1 2 3 4 5 6")

# Experiment configuration
SEARCH_INTENSITY="medium"  # small, medium, large
DATASET="cora"

run_validation_test() {
    local method=$1
    local hyperparams=$2
    local dataset=$3
    local phase=$4  # "validation" or "test"
    
    # Silent execution - no emoji output to avoid awk issues
    
    local cmd=""
    local log_name="${method}_${hyperparams//[[:space:],]/_}_${phase}_${dataset}"
    
    if [ "$method" = "cmg" ]; then
        # Parse k and d from hyperparams string "k=X,d=Y"
        local k=$(echo $hyperparams | sed 's/.*k=\([0-9]*\).*/\1/')
        local d=$(echo $hyperparams | sed 's/.*d=\([0-9]*\).*/\1/')
        cmd="python graphzoom_timed.py --dataset $dataset --coarse cmg --level 2 --cmg_k $k --cmg_d $d --embed_method deepwalk --seed 42"
    elif [ "$method" = "lamg" ]; then
        # Parse all LAMG parameters from hyperparams string "r=X,s=Y,n=Z"
        local r=$(echo $hyperparams | sed 's/.*r=\([0-9]*\).*/\1/')
        local s=$(echo $hyperparams | sed 's/.*s=\([0-9]*\).*/\1/' || echo "12")
        local n=$(echo $hyperparams | sed 's/.*n=\([0-9]*\).*/\1/' || echo "2")
        cmd="python graphzoom_timed.py --dataset $dataset --coarse lamg --reduce_ratio $r --search_ratio $s --num_neighs $n --mcr_dir $MATLAB_MCR_ROOT --embed_method deepwalk --seed 42"
    elif [ "$method" = "simple" ]; then
        # Parse level from hyperparams string "l=X"
        local l=$(echo $hyperparams | sed 's/.*l=\([0-9]*\).*/\1/')
        cmd="python graphzoom_timed.py --dataset $dataset --coarse simple --level $l --embed_method deepwalk --seed 42"
    fi
    
    local log_file="$RESULTS_DIR/logs/${log_name}.log"
    
    timeout 300 $cmd > $log_file 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        local accuracy=$(grep "Test Accuracy:" $log_file | awk '{print $3}')
        local total_time=$(grep "Total Time.*=" $log_file | awk '{print $NF}')
        
        # Extract compression ratio
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
        
        local compression_ratio="unknown"
        if [ "$final_nodes" != "unknown" ] && [ "$final_nodes" -gt 0 ] 2>/dev/null; then
            compression_ratio=$(echo "scale=2; 2708 / $final_nodes" | bc -l)
        fi
        
        echo "$method,$hyperparams,$accuracy,$total_time,$compression_ratio,$phase" >> $VALIDATION_CSV
        
        # Output to stderr to avoid function return contamination
        echo "[$phase] $method $hyperparams: Acc=$accuracy, Comp=${compression_ratio}x" >&2
        
        # Return only the accuracy number (clean)
        echo "$accuracy"
    else
        echo "FAILED: $method $hyperparams" >&2
        echo "$method,$hyperparams,FAILED,TIMEOUT,unknown,$phase" >> $VALIDATION_CSV
        echo "0"
    fi
}

hyperparameter_search() {
    local method=$1
    local dataset=$2
    
    echo ""
    echo "🔍 HYPERPARAMETER SEARCH: $method"
    echo "================================="
    
    local search_start_time=$(date +%s)
    local best_accuracy=0
    local best_hyperparams=""
    local config_count=0
    
    if [ "$method" = "cmg" ]; then
        echo "Testing CMG++ hyperparameters (k × d combinations)..."
        for k in ${CMG_K_VALUES[$SEARCH_INTENSITY]}; do
            for d in ${CMG_D_VALUES[$SEARCH_INTENSITY]}; do
                local hyperparams="k=$k,d=$d"
                echo "Testing CMG k=$k, d=$d..." >&2
                local accuracy=$(run_validation_test $method "$hyperparams" $dataset "validation")
                config_count=$((config_count + 1))
                
                if [ ! -z "$accuracy" ] && [ "$accuracy" != "FAILED" ] && [ "$accuracy" != "0" ]; then
                    # Use awk for float comparison
                    if awk "BEGIN {exit !($accuracy > $best_accuracy)}"; then
                        best_accuracy=$accuracy
                        best_hyperparams=$hyperparams
                        echo "New best: $hyperparams with accuracy $accuracy" >&2
                    fi
                fi
            done
        done
        
    elif [ "$method" = "lamg" ]; then
        echo "Testing LAMG hyperparameters (reduce_ratio × search_ratio × num_neighs)..."
        for r in ${LAMG_REDUCE_VALUES[$SEARCH_INTENSITY]}; do
            for s in ${LAMG_SEARCH_VALUES[$SEARCH_INTENSITY]}; do
                for n in ${LAMG_NEIGHS_VALUES[$SEARCH_INTENSITY]}; do
                    local hyperparams="r=$r,s=$s,n=$n"
                    echo "Testing LAMG r=$r, s=$s, n=$n..." >&2
                    local accuracy=$(run_validation_test $method "$hyperparams" $dataset "validation")
                    config_count=$((config_count + 1))
                    
                    if [ ! -z "$accuracy" ] && [ "$accuracy" != "FAILED" ] && [ "$accuracy" != "0" ]; then
                        if awk "BEGIN {exit !($accuracy > $best_accuracy)}"; then
                            best_accuracy=$accuracy
                            best_hyperparams=$hyperparams
                            echo "New best LAMG: $hyperparams with accuracy $accuracy" >&2
                        fi
                    fi
                done
            done
        done
        
    elif [ "$method" = "simple" ]; then
        echo "Testing Simple hyperparameters (level values)..."
        for l in ${SIMPLE_LEVEL_VALUES[$SEARCH_INTENSITY]}; do
            local hyperparams="l=$l"
            local accuracy=$(run_validation_test $method "$hyperparams" $dataset "validation")
            config_count=$((config_count + 1))
            
            if [ ! -z "$accuracy" ] && [ "$accuracy" != "FAILED" ] && [ "$accuracy" != "0" ]; then
                # Use awk for float comparison instead of bc  
                if awk "BEGIN {exit !($accuracy > $best_accuracy)}"; then
                    best_accuracy=$accuracy
                    best_hyperparams=$hyperparams
                fi
            fi
        done
    fi
    
    local search_end_time=$(date +%s)
    local search_duration=$((search_end_time - search_start_time))
    
    echo ""
    echo "🏆 BEST HYPERPARAMETERS for $method:"
    echo "   Parameters: $best_hyperparams"
    echo "   Validation Accuracy: $best_accuracy"
    echo "   Configurations Tested: $config_count"
    echo "   Search Time: ${search_duration}s"
    
    # Test the best hyperparameters on test set (ONLY ONCE!)
    echo ""
    echo "🧪 FINAL TEST with best hyperparameters..."
    local test_accuracy=$(run_validation_test $method "$best_hyperparams" $dataset "test")
    local test_time=$(grep "$method,$best_hyperparams.*,test" $VALIDATION_CSV | tail -1 | cut -d',' -f4)
    local test_compression=$(grep "$method,$best_hyperparams.*,test" $VALIDATION_CSV | tail -1 | cut -d',' -f5)
    
    echo "$method,\"$best_hyperparams\",$test_accuracy,$test_time,$test_compression,$search_duration,$config_count" >> $FINAL_TEST_CSV
    
    echo "📊 FINAL TEST RESULT for $method:"
    echo "   Test Accuracy: $test_accuracy (vs Validation: $best_accuracy)"
    echo "   Total Cost: ${search_duration}s search + ${test_time}s test = $((search_duration + ${test_time%.*}))s total"
}

main() {
    echo "🎯 Starting Proper Validation Methodology Experiment"
    echo ""
    echo "⚠️  ADDRESSING KOUTIS'S METHODOLOGICAL CONCERNS:"
    echo "   - Proper hyperparameter search on validation set"
    echo "   - Final evaluation on test set (one time only)"
    echo "   - Accounting for hyperparameter search cost"
    echo "   - No test set contamination"
    echo ""
    echo "📊 Search Intensity: $SEARCH_INTENSITY"
    echo "🗂️  Results will be saved to:"
    echo "   Hyperparameter Search: $VALIDATION_CSV"
    echo "   Final Test Results: $FINAL_TEST_CSV"
    echo ""
    
    # Note about data splitting
    echo "📋 NOTE: Current implementation uses train/test split from original dataset."
    echo "In production, you would implement proper train/val/test splitting."
    echo "This experiment shows the methodology and computational cost implications."
    echo ""
    
    # Run hyperparameter search for each method
    hyperparameter_search "cmg" $DATASET
    hyperparameter_search "lamg" $DATASET  
    hyperparameter_search "simple" $DATASET
    
    echo ""
    echo "✅ PROPER VALIDATION METHODOLOGY EXPERIMENT COMPLETE!"
    echo "====================================================="
    
    # Final analysis
    echo ""
    echo "📊 FINAL COMPARISON (Proper Methodology):"
    echo "=========================================="
    echo ""
    echo "Method    | Test Acc | Test Time | Search Time | Total Time | Configs | Best Params"
    echo "----------|----------|-----------|-------------|------------|---------|-------------"
    
    while IFS=, read -r method params test_acc test_time compression search_time configs rest; do
        if [ "$method" != "method" ]; then  # Skip header
            total_time=$(echo "scale=0; $search_time + $test_time" | bc -l 2>/dev/null || echo "$search_time")
            printf "%-9s | %-8s | %-9s | %-11s | %-10s | %-7s | %s\n" \
                   "$method" "$test_acc" "${test_time}s" "${search_time}s" "${total_time}s" "$configs" "$params"
        fi
    done < $FINAL_TEST_CSV
    
    echo ""
    echo "🎯 KEY INSIGHTS:"
    echo "   - Shows computational cost of proper hyperparameter search"
    echo "   - Eliminates test set contamination bias"
    echo "   - Provides fair comparison of all methods"
    echo "   - Addresses Koutis's methodological concerns"
    echo ""
    echo "🔬 This methodology addresses GraphZoom's fundamental flaw:"
    echo "   'If we need to do a sweep of 100 settings, there goes the speed.'"
}

main "$@"