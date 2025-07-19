#!/bin/bash
# Run Sample Validation - Quick Test
# Validates CSV output, LAMG timing, and all experiment types with minimal runs

echo "🧪 KOUTIS SAMPLE VALIDATION - QUICK TEST"
echo "========================================"
echo "Goal: Validate setup before running full study"
echo "Time: ~10-15 minutes"
echo ""

# Check prerequisites
check_prerequisites() {
    echo "🔍 Checking prerequisites..."
    
    # Check GraphZoom
    if [ ! -f "graphzoom.py" ]; then
        echo "❌ graphzoom.py not found. Please run from GraphZoom directory."
        exit 1
    fi
    echo "✅ GraphZoom found"
    
    # Check Python
    python3 -c "import pandas, numpy" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Python dependencies available"
    else
        echo "❌ Missing Python dependencies. Run: pip install pandas numpy"
        exit 1
    fi
    
    # Check MATLAB MCR
    if [ -n "$MATLAB_MCR_ROOT" ] && [ -d "$MATLAB_MCR_ROOT" ]; then
        echo "✅ MATLAB MCR found: $MATLAB_MCR_ROOT"
    else
        echo "⚠️  MATLAB MCR not set. LAMG tests will be limited."
        echo "   Set with: export MATLAB_MCR_ROOT=/path/to/mcr"
    fi
    
    echo ""
}

# Create the sample test scripts if they don't exist
create_scripts_if_needed() {
    # The sample test script should already be created from the artifact above
    if [ ! -f "koutis_sample_test.sh" ]; then
        echo "📝 Creating sample test script..."
        # Script content would be embedded here, but since we have artifacts, 
        # user should save the koutis_sample_test artifact as koutis_sample_test.sh
        echo "❌ Please save the koutis_sample_test artifact as koutis_sample_test.sh first"
        exit 1
    fi
    
    if [ ! -f "validate_sample_results.py" ]; then
        echo "❌ Please save the validate_sample_results artifact as validate_sample_results.py first"
        exit 1
    fi
    
    # Make executable
    chmod +x koutis_sample_test.sh
    chmod +x validate_sample_results.py
}

# Run the validation
main() {
    echo "🚀 Starting sample validation..."
    
    check_prerequisites
    create_scripts_if_needed
    
    echo "📋 SAMPLE TEST PLAN:"
    echo "1. Vanilla baseline (2 tests)"
    echo "2. Dimension scaling (6 tests)"  
    echo "3. Hyperparameter robustness (5 tests)"
    echo "4. Multilevel comparison (9 tests)"
    echo "5. Computational efficiency (2 tests)"
    echo "Total: ~24 tests, estimated time: 10-15 minutes"
    echo ""
    
    read -p "🚀 Start sample validation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted by user."
        exit 1
    fi
    
    # Run sample tests
    echo "🏃 Running sample tests..."
    ./koutis_sample_test.sh
    
    # Find the results file
    RESULTS_DIR="koutis_sample_validation"
    SAMPLE_CSV=$(find $RESULTS_DIR -name "koutis_sample_results_*.csv" | head -1)
    
    if [ -f "$SAMPLE_CSV" ]; then
        echo ""
        echo "✅ Sample tests completed!"
        echo "📁 Results: $SAMPLE_CSV"
        echo ""
        echo "🔍 Running validation analysis..."
        python3 validate_sample_results.py "$SAMPLE_CSV"
        
        # Quick manual check
        echo ""
        echo "📊 QUICK MANUAL CHECK:"
        echo "Total lines in CSV: $(wc -l < "$SAMPLE_CSV")"
        echo "Completed tests: $(grep -c "completed" "$SAMPLE_CSV" || echo 0)"
        echo "Failed tests: $(grep -c "failed" "$SAMPLE_CSV" || echo 0)"
        
        # Show sample of successful results
        echo ""
        echo "📋 Sample successful results:"
        grep "completed" "$SAMPLE_CSV" | head -3 | cut -d',' -f1-5,12,13,20,21
        
    else
        echo "❌ No results file found!"
        echo "🔍 Check logs in: $RESULTS_DIR/logs/"
        exit 1
    fi
}

main "$@"
