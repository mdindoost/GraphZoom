#!/bin/bash
# Simple GraphZoom Test - Basic Validation
# Tests with standard GraphZoom arguments only

echo "🧪 SIMPLE GRAPHZOOM TEST - BASIC VALIDATION"
echo "==========================================="
echo "Goal: Test with standard GraphZoom arguments to understand output format"
echo ""

# Create results directory
mkdir -p simple_test_results

# Simple CSV for basic tests
SIMPLE_CSV="simple_test_results/basic_graphzoom_test.csv"
echo "method,dataset,embedding,accuracy,total_time,status,notes" > $SIMPLE_CSV

# Function to run basic GraphZoom test
run_basic_test() {
    local method=$1
    local dataset=$2
    local embedding=$3
    local extra_args="$4"
    
    local test_name="${method}_${dataset}_${embedding}"
    local log_file="simple_test_results/${test_name}.log"
    
    echo "🔄 Testing: $test_name"
    
    # Build basic command
    local cmd="python graphzoom.py --dataset $dataset --embed_method $embedding"
    if [ -n "$extra_args" ]; then
        cmd="$cmd $extra_args"
    fi
    
    echo "   Command: $cmd"
    
    # Run with short timeout
    timeout 180 $cmd > $log_file 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "   ✅ SUCCESS"
        
        # Extract basic info
        local accuracy=$(grep -i "accuracy" $log_file | tail -1 | awk '{print $NF}' | tr -d '%,:')
        local total_time=$(grep -E "(time|Time)" $log_file | grep -E "[0-9]+\.[0-9]" | tail -1 | awk '{print $NF}' | tr -d 's')
        
        # Handle empty values
        [ -z "$accuracy" ] && accuracy="unknown"
        [ -z "$total_time" ] && total_time="unknown"
        
        echo "$method,$dataset,$embedding,$accuracy,$total_time,success,completed" >> $SIMPLE_CSV
        
        echo "   📊 Accuracy: $accuracy"
        echo "   ⏱️  Time: $total_time"
        
        # Show sample output for debugging
        echo "   🔍 Sample output:"
        tail -5 $log_file | sed 's/^/      /'
        
    else
        echo "   ❌ FAILED (exit code: $exit_code)"
        echo "$method,$dataset,$embedding,failed,failed,failed,failed" >> $SIMPLE_CSV
        
        echo "   🔍 Error output:"
        tail -3 $log_file | sed 's/^/      /'
    fi
    
    echo ""
}

# Test 1: Basic GraphZoom (no coarsening)
echo "📊 TEST 1: Basic GraphZoom (default settings)"
echo "============================================="
run_basic_test "default" "cora" "deepwalk" ""

# Test 2: Simple coarsening (if supported)
echo "📊 TEST 2: Simple Coarsening (if supported)"
echo "==========================================="
if python graphzoom.py --help 2>&1 | grep -q "coarse"; then
    run_basic_test "simple" "cora" "deepwalk" "--coarse simple"
else
    echo "⚠️  Coarsening not supported in this GraphZoom version"
fi

# Test 3: LAMG (if supported and MCR available)
echo "📊 TEST 3: LAMG (if supported)"
echo "============================="
if python graphzoom.py --help 2>&1 | grep -q "lamg" && [ -n "$MATLAB_MCR_ROOT" ]; then
    run_basic_test "lamg" "cora" "deepwalk" "--coarse lamg --reduce_ratio 2 --search_ratio 12 --mcr_dir $MATLAB_MCR_ROOT"
else
    echo "⚠️  LAMG not supported or MCR not available"
fi

# Test 4: Different embedding
echo "📊 TEST 4: Different Embedding"
echo "=============================="
run_basic_test "default" "cora" "node2vec" ""

# Analyze results
echo "📋 BASIC TEST RESULTS"
echo "===================="
echo "CSV file: $SIMPLE_CSV"
echo ""
echo "📊 Results summary:"
cat $SIMPLE_CSV

echo ""
echo "🔍 GraphZoom Output Analysis:"
echo "=============================="

# Find a successful log to analyze
SUCCESS_LOG=$(find simple_test_results -name "*.log" -exec grep -l "accuracy\|Accuracy" {} \; | head -1)

if [ -n "$SUCCESS_LOG" ]; then
    echo "📄 Analyzing successful run: $SUCCESS_LOG"
    echo ""
    
    echo "🎯 Accuracy patterns found:"
    grep -i "accuracy" "$SUCCESS_LOG"
    
    echo ""
    echo "⏱️  Timing patterns found:"
    grep -i "time" "$SUCCESS_LOG"
    
    echo ""
    echo "📊 Node/cluster patterns found:"
    grep -E "(node|cluster|coarse)" "$SUCCESS_LOG"
    
    echo ""
    echo "📋 Full output sample (last 10 lines):"
    tail -10 "$SUCCESS_LOG"
    
else
    echo "❌ No successful runs found to analyze"
fi

echo ""
echo "✅ BASIC GRAPHZOOM TEST COMPLETE!"
echo ""
echo "💡 Next steps:"
echo "1. Review output patterns above"
echo "2. Update extraction patterns in sample test script"
echo "3. Run full validation with correct patterns"
