#!/bin/bash
# Quick launcher for Koutis efficiency study

echo "🚀 LAUNCHING KOUTIS EFFICIENCY STUDY"
echo "===================================="

# Check if scripts exist
if [ ! -f "koutis_test_suite.sh" ]; then
    echo "❌ koutis_test_suite.sh not found. Please create it first."
    exit 1
fi

if [ ! -f "analyze_koutis_results.py" ]; then
    echo "❌ analyze_koutis_results.py not found. Please create it first."
    exit 1
fi

# Make scripts executable
chmod +x koutis_test_suite.sh
chmod +x analyze_koutis_results.py

echo "🔧 Setting up environment..."

# Set MATLAB MCR if available
if [ -d "/home/mohammad/matlab/R2018a" ]; then
    export MATLAB_MCR_ROOT="/home/mohammad/matlab/R2018a"
    echo "✅ MATLAB MCR found: $MATLAB_MCR_ROOT"
else
    echo "⚠️  MATLAB MCR not found. LAMG tests will be skipped."
fi

# Check Python dependencies
python3 -c "import pandas, numpy, matplotlib, seaborn, scipy" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Python dependencies available"
else
    echo "❌ Missing Python dependencies. Installing..."
    pip install pandas numpy matplotlib seaborn scipy
fi

echo ""
echo "🎯 KOUTIS STUDY OBJECTIVES:"
echo "1. Prove 2x computational efficiency vs GraphZoom"
echo "2. Show dimension stability (CMG++ vs GraphZoom degradation)"  
echo "3. Demonstrate hyperparameter robustness"
echo "4. Provide statistical rigor (missing in GraphZoom paper)"
echo "5. Match/exceed GraphZoom Table 2 speedups"
echo ""

read -p "🚀 Ready to start? This will take several hours. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user."
    exit 1
fi

echo "🏃 Starting Koutis efficiency study..."

# Run the test suite
./koutis_test_suite.sh

# Check if results were generated
RESULTS_DIR="koutis_efficiency_study"
LATEST_RESULTS=$(find $RESULTS_DIR -name "koutis_efficiency_results_*.csv" | head -1)

if [ -f "$LATEST_RESULTS" ]; then
    echo ""
    echo "✅ Tests completed! Running analysis..."
    echo "📁 Results file: $LATEST_RESULTS"
    
    # Run analysis
    python3 analyze_koutis_results.py "$LATEST_RESULTS"
    
    echo ""
    echo "🎉 KOUTIS EFFICIENCY STUDY COMPLETE!"
    echo "📊 Check the generated plots and analysis above"
    echo "📁 Detailed logs: $RESULTS_DIR/logs/"
    echo ""
    echo "📋 NEXT STEPS:"
    echo "1. Review dimension scaling plot: koutis_dimension_scaling_analysis.png"
    echo "2. Prepare publication figures from analysis results"
    echo "3. Write up efficiency claims validation"
    echo "4. Submit to top-tier venue with strong empirical evidence"
    
else
    echo "❌ No results file found. Check logs in $RESULTS_DIR/logs/"
    exit 1
fi
