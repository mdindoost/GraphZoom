#!/bin/bash
# Single test to verify cluster extraction fix

echo "🔧 VERIFYING CLUSTER EXTRACTION FIX"
echo "===================================="

# Test the extraction on a single CMG run
echo "🔄 Running single CMG test..."

timeout 300 python graphzoom.py --dataset cora --coarse cmg --level 1 --cmg_k 10 --cmg_d 20 --num_neighs 2 --embed_method deepwalk > single_test_log.txt 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Test completed, extracting info..."
    
    # Extract using the fixed logic
    accuracy=$(grep "Test Accuracy:" single_test_log.txt | awk '{print $3}')
    time=$(grep "Total Time" single_test_log.txt | awk '{print $NF}')
    
    # Extract final cluster count (fixed version)
    final_nodes=$(grep "Final graph:" single_test_log.txt | tail -1 | awk '{print $4}')
    if [ ! -z "$final_nodes" ]; then
        clusters_found=$final_nodes
    else
        coarsened_nodes=$(grep "Coarsened to" single_test_log.txt | tail -1 | awk '{print $4}')
        if [ ! -z "$coarsened_nodes" ]; then
            clusters_found=$coarsened_nodes
        else
            clusters_found="extraction_failed"
        fi
    fi
    
    # Calculate compression
    if [ "$clusters_found" != "extraction_failed" ] && [ "$clusters_found" -gt 0 ]; then
        compression_ratio=$(echo "scale=2; 2708 / $clusters_found" | bc -l 2>/dev/null || echo "calc_error")
    else
        compression_ratio="unknown"
    fi
    
    echo ""
    echo "📊 EXTRACTION RESULTS:"
    echo "----------------------"
    echo "Accuracy: $accuracy"
    echo "Time: ${time}s"
    echo "Final Clusters: $clusters_found"
    echo "Compression: ${compression_ratio}x"
    echo ""
    
    # Show the relevant log lines for verification
    echo "📋 RELEVANT LOG LINES:"
    echo "----------------------"
    echo "CMG cluster info:"
    grep "CMG found" single_test_log.txt | tail -2
    echo "Final graph info:"
    grep "Final graph:" single_test_log.txt
    echo "Coarsening info:"
    grep "Coarsened to" single_test_log.txt
    
    if [ "$clusters_found" != "extraction_failed" ] && [ "$compression_ratio" != "unknown" ]; then
        echo ""
        echo "✅ EXTRACTION FIX SUCCESSFUL!"
        echo "   Clusters: $clusters_found"
        echo "   Compression: ${compression_ratio}x"
    else
        echo ""
        echo "❌ EXTRACTION STILL HAS ISSUES"
    fi
else
    echo "❌ Test failed"
fi

rm -f single_test_log.txt
