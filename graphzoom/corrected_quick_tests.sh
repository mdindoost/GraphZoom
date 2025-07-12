#!/bin/bash
# Corrected quick tests focusing on k-NN, clustering, and compression analysis

echo "🔬 CORRECTED QUICK TESTS - K-NN, CLUSTERING & COMPRESSION"
echo "========================================================"

# Create results file with cluster analysis
echo "test_name,dataset,method,level,cmg_k,cmg_d,knn_neighbors,accuracy,time,clusters_found,compression_ratio,notes" > corrected_test_results.csv

run_test_with_analysis() {
    local name=$1
    local dataset=$2
    local method=$3
    local level=$4
    local cmg_k=$5
    local cmg_d=$6
    local knn_neighbors=$7
    local extra_params=$8
    
    echo "🔄 Running: $name"
    
    # Build command
    if [ "$method" = "cmg" ]; then
        cmd="python graphzoom.py --dataset $dataset --coarse $method --level $level --cmg_k $cmg_k --cmg_d $cmg_d --num_neighs $knn_neighbors --embed_method deepwalk $extra_params"
    else
        cmd="python graphzoom.py --dataset $dataset --coarse $method --level $level --num_neighs $knn_neighbors --embed_method deepwalk $extra_params"
    fi
    
    echo "   Command: $cmd"
    
    # Run with timeout
    timeout 300 $cmd > temp_log.txt 2>&1
    
    if [ $? -eq 0 ]; then
        # Extract standard metrics
        accuracy=$(grep "Test Accuracy:" temp_log.txt | awk '{print $3}')
        time=$(grep "Total Time" temp_log.txt | awk '{print $NF}')
        
        # Extract cluster/compression information
        clusters_found="unknown"
        compression_ratio="unknown"
        
        # For CMG: look for final cluster count (use Final graph nodes as definitive)
        if [ "$method" = "cmg" ]; then
            final_nodes=$(grep "Final graph:" temp_log.txt | tail -1 | awk '{print $4}')
            if [ ! -z "$final_nodes" ]; then
                clusters_found=$final_nodes
            else
                # Fallback to coarsened nodes  
                coarsened_nodes=$(grep "Coarsened to" temp_log.txt | tail -1 | awk '{print $4}')
                if [ ! -z "$coarsened_nodes" ]; then
                    clusters_found=$coarsened_nodes
                fi
            fi
        fi
        
        # For Simple: look for coarsening info
        if [ "$method" = "simple" ]; then
            coarse_nodes=$(grep "Num of nodes:" temp_log.txt | tail -1 | awk '{print $4}')
            if [ ! -z "$coarse_nodes" ]; then
                clusters_found=$coarse_nodes
            fi
        fi
        
        # Calculate compression ratio (for known dataset sizes)
        if [ "$clusters_found" != "unknown" ] && [ "$clusters_found" -gt 0 ]; then
            if [ "$dataset" = "cora" ]; then
                compression_ratio=$(echo "scale=2; 2708 / $clusters_found" | bc -l 2>/dev/null || echo "calc_error")
            elif [ "$dataset" = "citeseer" ]; then
                compression_ratio=$(echo "scale=2; 3327 / $clusters_found" | bc -l 2>/dev/null || echo "calc_error")
            elif [ "$dataset" = "pubmed" ]; then
                compression_ratio=$(echo "scale=2; 19717 / $clusters_found" | bc -l 2>/dev/null || echo "calc_error")
            fi
        fi
        
        echo "$name,$dataset,$method,$level,$cmg_k,$cmg_d,$knn_neighbors,$accuracy,$time,$clusters_found,$compression_ratio,completed" >> corrected_test_results.csv
        echo "✅ $name: Acc=$accuracy, Time=${time}s, Clusters=$clusters_found, Compression=${compression_ratio}x"
    else
        echo "$name,$dataset,$method,$level,$cmg_k,$cmg_d,$knn_neighbors,FAILED,TIMEOUT,unknown,unknown,failed" >> corrected_test_results.csv
        echo "❌ $name: FAILED/TIMEOUT"
    fi
    
    rm -f temp_log.txt
}

echo ""
echo "📊 TEST 1: K-NN Effect in GraphZoom Fusion (Koutis's actual suggestion)"
echo "---------------------------------------------------------------------"
run_test_with_analysis "knn_default" "cora" "cmg" 1 10 20 2 ""
run_test_with_analysis "knn_aggressive1" "cora" "cmg" 1 10 20 5 ""
run_test_with_analysis "knn_aggressive2" "cora" "cmg" 1 10 20 10 ""
run_test_with_analysis "knn_very_aggressive" "cora" "cmg" 1 10 20 15 ""

echo ""
echo "🎯 TEST 2: CMG Parameter Efficiency (smaller d as Koutis suggested)"
echo "------------------------------------------------------------------"
run_test_with_analysis "current_best" "cora" "cmg" 1 10 20 2 ""
run_test_with_analysis "smaller_d1" "cora" "cmg" 1 10 15 2 ""
run_test_with_analysis "smaller_d2" "cora" "cmg" 1 10 10 2 ""
run_test_with_analysis "smaller_d3" "cora" "cmg" 1 10 5 2 ""

echo ""
echo "⚖️ TEST 3: CMG vs Simple Clustering Comparison"
echo "----------------------------------------------"
run_test_with_analysis "simple_baseline" "cora" "simple" 1 0 0 2 ""
run_test_with_analysis "simple_knn5" "cora" "simple" 1 0 0 5 ""
run_test_with_analysis "cmg_baseline" "cora" "cmg" 1 10 20 2 ""
run_test_with_analysis "cmg_knn5" "cora" "cmg" 1 10 20 5 ""

echo ""
echo "🔄 TEST 4: Multi-level Compression Analysis"
echo "------------------------------------------"
run_test_with_analysis "level1_compression" "cora" "cmg" 1 10 20 2 ""
run_test_with_analysis "level2_compression" "cora" "cmg" 2 10 20 2 ""
run_test_with_analysis "level3_compression" "cora" "cmg" 3 10 20 2 ""

echo ""
echo "✅ All corrected tests completed!"
echo "📁 Results saved in: corrected_test_results.csv"
echo ""

# Quick preview of clustering results
echo "📊 CLUSTERING & COMPRESSION PREVIEW:"
echo "-----------------------------------"
if [ -f corrected_test_results.csv ]; then
    echo "Test Name | Method | Clusters | Compression | Accuracy"
    echo "----------|---------|----------|-------------|----------"
    tail -n +2 corrected_test_results.csv | while IFS=',' read -r test_name dataset method level cmg_k cmg_d knn_neighbors accuracy time clusters_found compression_ratio notes; do
        printf "%-15s | %-6s | %-8s | %-11s | %s\n" \
            "$(echo $test_name | cut -c1-15)" \
            "$method" \
            "$clusters_found" \
            "$compression_ratio" \
            "$accuracy"
    done
fi