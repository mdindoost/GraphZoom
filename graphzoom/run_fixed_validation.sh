#!/bin/bash
# Run Fixed Sample Validation - Uses Correct GraphZoom Patterns

echo "🎯 FIXED KOUTIS SAMPLE VALIDATION"
echo "================================="
echo "Using CORRECT GraphZoom extraction patterns from your actual output"
echo ""

# Set environment
export MATLAB_MCR_ROOT="/home/mohammad/matlab/R2018a"

echo "🔧 Environment:"
echo "   MATLAB MCR: $MATLAB_MCR_ROOT"
echo "   Current directory: $(pwd)"
echo ""

# Check files
if [ ! -f "fixed_koutis_sample_test.sh" ]; then
    echo "❌ Please save the 'fixed_koutis_sample_test' artifact as fixed_koutis_sample_test.sh"
    exit 1
fi

if [ ! -f "validate_sample_results.py" ]; then
    echo "❌ Please save the 'validate_sample_results' artifact as validate_sample_results.py"
    exit 1
fi

# Make executable
chmod +x fixed_koutis_sample_test.sh
chmod +x validate_sample_results.py

echo "📋 FIXED SAMPLE TEST PLAN (based on your GraphZoom output):"
echo "==========================================================="
echo "✅ Correct extraction patterns:"
echo "   - Accuracy: 'Test Accuracy:' + awk '{print \$3}'"
echo "   - Total time: 'Total Time.*=' + awk '{print \$NF}'"
echo "   - Fusion time: 'Graph Fusion.*Time:' + awk '{print \$4}'"
echo "   - Reduction time: 'Graph Reduction.*Time:' + awk '{print \$4}'"
echo "   - Embedding time: 'Graph Embedding.*Time:' + awk '{print \$4}'"
echo "   - Refinement time: 'Graph Refinement.*Time:' + awk '{print \$4}'"
echo "   - Final clusters: 'Num of nodes:' + awk '{print \$4}'"
echo ""
echo "🧪 Test categories:"
echo "1. Vanilla baselines (2 tests) - for speedup calculation"
echo "2. Dimension scaling (6 tests) - CMG++ d=[10,20,30] vs LAMG"  
echo "3. Hyperparameter robustness (5 tests) - CMG++ k,d combinations"
echo "4. Multilevel comparison (9 tests) - levels 1,2,3 comparison"
echo "5. Computational efficiency (2 tests) - direct CMG++ vs LAMG"
echo "Total: ~24 tests, estimated time: 15-20 minutes"
echo ""

read -p "🚀 Start fixed validation with correct patterns? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user."
    exit 1
fi

echo "🏃 Running fixed sample tests..."
./fixed_koutis_sample_test.sh

# Find the results file
RESULTS_DIR="koutis_sample_validation"
SAMPLE_CSV=$(find $RESULTS_DIR -name "koutis_sample_results_*.csv" | head -1)

if [ -f "$SAMPLE_CSV" ]; then
    echo ""
    echo "✅ Fixed sample tests completed!"
    echo "📁 Results: $SAMPLE_CSV"
    echo ""
    
    # Quick validation check
    echo "📊 QUICK RESULTS CHECK:"
    echo "======================"
    echo "Total lines in CSV: $(wc -l < "$SAMPLE_CSV")"
    echo "Completed tests: $(grep -c "completed" "$SAMPLE_CSV" || echo 0)"
    echo "Failed tests: $(grep -c "failed" "$SAMPLE_CSV" || echo 0)"
    
    # Show sample of extracted data
    echo ""
    echo "📋 Sample extracted data (first 3 successful tests):"
    echo "method,dataset,embedding,accuracy,total_time,fusion_time,reduction_time,final_clusters,compression_ratio"
    grep "completed" "$SAMPLE_CSV" | head -3 | cut -d',' -f2,3,4,12,13,14,15,21,22
    
    echo ""
    echo "🔍 Running detailed validation analysis..."
    python3 validate_sample_results.py "$SAMPLE_CSV"
    
else
    echo "❌ No results file found!"
    echo "🔍 Check logs in: $RESULTS_DIR/logs/"
    
    # Show any error logs
    echo ""
    echo "📄 Recent log files:"
    find $RESULTS_DIR/logs -name "*.log" -exec ls -la {} \; | head -5
    
    exit 1
fi
