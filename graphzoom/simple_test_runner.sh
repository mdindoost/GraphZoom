#!/bin/bash
# Simple Test Runner - Minimal Debug Version
# Tests one method at a time to isolate issues

echo "🧪 SIMPLE TEST RUNNER - MINIMAL DEBUG"
echo "===================================="
echo "Goal: Test one method at a time to find the issue"
echo ""

# Create simple results directory
mkdir -p simple_debug_test

# Simple CSV
SIMPLE_CSV="simple_debug_test/debug_results.csv"
echo "test_name,command,exit_code,accuracy,total_time,status" > $SIMPLE_CSV

# Function to run one test with detailed debugging
run_debug_test() {
    local test_name=$1
    local cmd="$2"
    local log_file="simple_debug_test/${test_name}.log"
    
    echo "🔄 Testing: $test_name"
    echo "   Command: $cmd"
    
    # Run with short timeout
    timeout 120 $cmd > $log_file 2>&1
    local exit_code=$?
    
    echo "   Exit code: $exit_code"
    
    if [ $exit_code -eq 0 ]; then
        echo "   ✅ SUCCESS"
        
        # Try to extract data
        local accuracy=$(grep "Test Accuracy:" $log_file | awk '{print $3}' || echo "not_found")
        local total_time=$(grep "Total Time.*=" $log_file | awk '{print $NF}' || echo "not_found")
        
        echo "   📊 Accuracy: $accuracy"
        echo "   ⏱️  Time: $total_time"
        
        echo "$test_name,$cmd,$exit_code,$accuracy,$total_time,success" >> $SIMPLE_CSV
        
    else
        echo "   ❌ FAILED"
        echo "   🔍 Error output (last 5 lines):"
        tail -5 $log_file | sed 's/^/      /'
        
        echo "$test_name,$cmd,$exit_code,failed,failed,failed" >> $SIMPLE_CSV
    fi
    
    echo ""
}

echo "📋 TESTING BASIC COMMANDS"
echo "========================="

# Test 1: Basic GraphZoom (should work from previous tests)
run_debug_test "basic_default" "python graphzoom.py --dataset cora --embed_method deepwalk"

# Test 2: Simple coarsening
run_debug_test "simple_coarsening" "python graphzoom.py --dataset cora --coarse simple --embed_method deepwalk"

# Test 3: CMG coarsening  
run_debug_test "cmg_basic" "python graphzoom.py --dataset cora --coarse cmg --embed_method deepwalk"

# Test 4: CMG with parameters
run_debug_test "cmg_with_params" "python graphzoom.py --dataset cora --coarse cmg --cmg_k 15 --cmg_d 20 --embed_method deepwalk"

# Test 5: LAMG (if MCR available)
if [ -n "$MATLAB_MCR_ROOT" ] && [ -d "$MATLAB_MCR_ROOT" ]; then
    run_debug_test "lamg_basic" "python graphzoom.py --dataset cora --coarse lamg --reduce_ratio 2 --search_ratio 12 --mcr_dir $MATLAB_MCR_ROOT --embed_method deepwalk"
else
    echo "⚠️  Skipping LAMG test - MCR not available"
fi

echo "📋 DEBUG TEST RESULTS"
echo "===================="
echo "CSV file: $SIMPLE_CSV"
echo ""
cat $SIMPLE_CSV

echo ""
echo "🔍 ANALYSIS"
echo "==========="

# Count successes
SUCCESS_COUNT=$(grep -c "success" $SIMPLE_CSV || echo 0)
TOTAL_COUNT=$(tail -n +2 $SIMPLE_CSV | wc -l)

echo "Success rate: $SUCCESS_COUNT/$TOTAL_COUNT"

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "✅ Some tests passed - issue may be with specific configurations"
    echo ""
    echo "📋 Successful tests:"
    grep "success" $SIMPLE_CSV
    echo ""
    echo "📋 Failed tests:"
    grep "failed" $SIMPLE_CSV
else
    echo "❌ All tests failed - fundamental issue with GraphZoom setup"
    echo ""
    echo "🔍 Check these potential issues:"
    echo "1. Python environment compatibility"
    echo "2. Missing dependencies"
    echo "3. Dataset files missing"
    echo "4. GraphZoom script issues"
fi

echo ""
echo "💡 Next steps:"
echo "1. Review individual log files in simple_debug_test/"
echo "2. Fix the identified issues"
echo "3. Re-run the simple tests until they pass"
echo "4. Then proceed with full validation"
