#!/bin/bash
# Debug Spectral Analysis - Find where LAMG outputs files

echo "🔍 DEBUGGING SPECTRAL ANALYSIS"
echo "=============================="

MATLAB_MCR_ROOT="/home/mohammad/matlab/R2018a"

# Function to debug a single LAMG run
debug_lamg_run() {
    local reduce_ratio=$1
    
    echo "🧪 Debugging LAMG reduce_ratio=$reduce_ratio"
    echo "============================================"
    
    # Clean up first
    rm -rf reduction_results/* 2>/dev/null
    
    # Show current directory structure
    echo "📁 Current directory: $(pwd)"
    echo "📁 Contents before run:"
    ls -la
    
    if [ -d "reduction_results" ]; then
        echo "📁 reduction_results/ contents before:"
        ls -la reduction_results/
    fi
    
    # Run LAMG
    echo "🚀 Running LAMG with reduce_ratio=$reduce_ratio..."
    python graphzoom_timed.py --dataset cora --embed_method deepwalk --seed 42 \
        --coarse lamg --mcr_dir $MATLAB_MCR_ROOT --reduce_ratio $reduce_ratio \
        > debug_lamg_${reduce_ratio}.log 2>&1
    
    local exit_code=$?
    echo "📊 Exit code: $exit_code"
    
    # Check what files were created
    echo "📁 Contents after run:"
    ls -la
    
    echo "📁 reduction_results/ contents after:"
    if [ -d "reduction_results" ]; then
        ls -la reduction_results/
    else
        echo "❌ reduction_results/ directory not found!"
    fi
    
    # Check for any MTX files anywhere
    echo "🔍 Searching for MTX files..."
    find . -name "*.mtx" -type f 2>/dev/null | head -10
    
    # Check log file for clues
    echo "📄 Last 20 lines of log file:"
    tail -20 debug_lamg_${reduce_ratio}.log
    
    echo "📄 Searching log for 'mtx' references:"
    grep -i "mtx" debug_lamg_${reduce_ratio}.log || echo "No MTX references found"
    
    echo "📄 Searching log for file creation:"
    grep -i "writ\|creat\|sav\|output" debug_lamg_${reduce_ratio}.log || echo "No file creation messages found"
    
    echo "📄 Searching log for errors:"
    grep -i "error\|fail\|exception" debug_lamg_${reduce_ratio}.log || echo "No obvious errors found"
    
    echo ""
}

# Debug multiple reduce_ratio values
debug_lamg_run 2
debug_lamg_run 3

echo "🔍 ADDITIONAL DEBUGGING"
echo "======================="

# Check if the original GraphZoom script creates the files correctly
echo "📊 Testing if files are created during normal run..."

# Run a simple test
python graphzoom_timed.py --dataset cora --embed_method deepwalk --seed 42 \
    --coarse lamg --mcr_dir $MATLAB_MCR_ROOT --reduce_ratio 2 \
    > test_run.log 2>&1

# Immediately check for files
echo "📁 Files immediately after run:"
ls -la reduction_results/ 2>/dev/null || echo "No reduction_results directory"

# Check if the files exist but have different names
find . -name "*.mtx" -type f -newer test_run.log 2>/dev/null | head -5

echo "📄 Checking what GraphZoom actually creates:"
grep -i "Writing\|Created\|Saved" test_run.log || echo "No file creation messages"

echo "📄 Checking for MATLAB-related output:"
grep -i "matlab\|mcr\|lamg" test_run.log || echo "No MATLAB/LAMG messages"

echo ""
echo "🎯 DEBUGGING COMPLETE"
echo "===================="
echo "Check the output above to understand:"
echo "1. Are MTX files being created?"
echo "2. Where are they being created?"
echo "3. What are they named?"
echo "4. Are there any errors preventing creation?"
