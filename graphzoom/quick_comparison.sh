#!/bin/bash
# Quick comparison script - essential experiments only
# Estimated time: 30-45 minutes

mkdir -p results/quick_logs
echo "experiment,dataset,coarsening,embedding,accuracy,total_time,notes" > results/quick_results.csv

run_quick() {
    local name=$1
    local dataset=$2
    local coarse=$3
    local embed=$4
    local params=$5
    
    echo "🔄 Running: $name"
    
    if [ "$coarse" = "cmg" ]; then
        timeout 600 python graphzoom.py --dataset $dataset --coarse $coarse --embed_method $embed $params > results/quick_logs/${name}.log 2>&1
    else
        timeout 600 python graphzoom.py --dataset $dataset --coarse $coarse --embed_method $embed > results/quick_logs/${name}.log 2>&1
    fi
    
    if [ $? -eq 124 ]; then
        echo "$name,$dataset,$coarse,$embed,TIMEOUT,TIMEOUT,experiment_timeout" >> results/quick_results.csv
        echo "⏰ TIMEOUT: $name"
    else
        accuracy=$(grep "Test Accuracy:" results/quick_logs/${name}.log | awk '{print $3}')
        total_time=$(grep "Total Time" results/quick_logs/${name}.log | awk '{print $NF}')
        echo "$name,$dataset,$coarse,$embed,$accuracy,$total_time,completed" >> results/quick_results.csv
        echo "✅ $name: Accuracy=$accuracy, Time=${total_time}s"
    fi
}

echo "🚀 Quick CMG vs GraphZoom Comparison"

# Core comparison on Cora
echo "📊 Core Comparison (Cora dataset)"
run_quick "cora_simple_deepwalk" "cora" "simple" "deepwalk" ""
run_quick "cora_cmg_default" "cora" "cmg" "deepwalk" "--cmg_k 10 --cmg_d 20 --cmg_threshold 0.1"

# Parameter study (quick)
echo "🔬 Quick Parameter Study"
run_quick "cora_cmg_k5" "cora" "cmg" "deepwalk" "--cmg_k 5 --cmg_d 20 --cmg_threshold 0.1"
run_quick "cora_cmg_k15" "cora" "cmg" "deepwalk" "--cmg_k 15 --cmg_d 20 --cmg_threshold 0.1"
run_quick "cora_cmg_d30" "cora" "cmg" "deepwalk" "--cmg_k 10 --cmg_d 30 --cmg_threshold 0.1"
run_quick "cora_cmg_thresh05" "cora" "cmg" "deepwalk" "--cmg_k 10 --cmg_d 20 --cmg_threshold 0.05"

# Other datasets with default CMG
echo "🌐 Other Datasets"
run_quick "citeseer_simple" "citeseer" "simple" "deepwalk" ""
run_quick "citeseer_cmg" "citeseer" "cmg" "deepwalk" "--cmg_k 10 --cmg_d 20 --cmg_threshold 0.1"

echo "✅ Quick comparison completed! Check results/quick_results.csv"
