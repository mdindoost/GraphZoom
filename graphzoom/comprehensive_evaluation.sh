#!/bin/bash
# Comprehensive Evaluation of CMG-GraphZoom vs Baselines
# Run this script from GraphZoom/graphzoom/ directory

# Create results directory
mkdir -p results
cd results
mkdir -p logs accuracy_results timing_results
cd ..

# Function to run experiment and capture results
run_experiment() {
    local dataset=$1
    local coarse_method=$2
    local embed_method=$3
    local extra_params=$4
    local experiment_name=$5
    
    echo "=========================================="
    echo "Running: $experiment_name"
    echo "Dataset: $dataset, Coarsening: $coarse_method, Embedding: $embed_method"
    echo "Parameters: $extra_params"
    echo "=========================================="
    
    # Run the experiment and capture output
    if [ "$coarse_method" = "cmg" ]; then
        python graphzoom.py --dataset $dataset --coarse $coarse_method --embed_method $embed_method $extra_params > results/logs/${experiment_name}.log 2>&1
    else
        python graphzoom.py --dataset $dataset --coarse $coarse_method --embed_method $embed_method > results/logs/${experiment_name}.log 2>&1
    fi
    
    # Extract accuracy and timing from log
    accuracy=$(grep "Test Accuracy:" results/logs/${experiment_name}.log | awk '{print $3}')
    total_time=$(grep "Total Time" results/logs/${experiment_name}.log | awk '{print $NF}')
    fusion_time=$(grep "Graph Fusion.*Time:" results/logs/${experiment_name}.log | awk '{print $4}')
    reduction_time=$(grep "Graph Reduction.*Time:" results/logs/${experiment_name}.log | awk '{print $4}')
    embedding_time=$(grep "Graph Embedding.*Time:" results/logs/${experiment_name}.log | awk '{print $4}')
    refinement_time=$(grep "Graph Refinement.*Time:" results/logs/${experiment_name}.log | awk '{print $4}')
    
    # Save structured results
    echo "$experiment_name,$dataset,$coarse_method,$embed_method,$accuracy,$total_time,$fusion_time,$reduction_time,$embedding_time,$refinement_time,$extra_params" >> results/accuracy_results/all_results.csv
    
    echo "✅ Completed: $experiment_name - Accuracy: $accuracy"
    echo ""
}

# Initialize results CSV
echo "experiment,dataset,coarsening,embedding,accuracy,total_time,fusion_time,reduction_time,embedding_time,refinement_time,parameters" > results/accuracy_results/all_results.csv

echo "🚀 Starting Comprehensive CMG-GraphZoom Evaluation"
echo "📊 This will run multiple experiments - estimated time: 2-3 hours"
echo ""

# ================================================================
# PART 1: BASELINE COMPARISONS
# ================================================================
echo "🔍 PART 1: Baseline Comparisons"

datasets=("cora" "citeseer" "pubmed")
embed_methods=("deepwalk" "node2vec")

for dataset in "${datasets[@]}"; do
    for embed_method in "${embed_methods[@]}"; do
        # Simple coarsening baseline
        run_experiment $dataset "simple" $embed_method "" "${dataset}_simple_${embed_method}"
        
        # CMG coarsening (default parameters)
        run_experiment $dataset "cmg" $embed_method "--cmg_k 10 --cmg_d 20 --cmg_threshold 0.1" "${dataset}_cmg_default_${embed_method}"
    done
done

# ================================================================
# PART 2: CMG PARAMETER STUDY
# ================================================================
echo "🔬 PART 2: CMG Parameter Study"

# Test different k values (filter order)
k_values=(5 10 15 20 25)
echo "Testing different k values: ${k_values[@]}"

for k in "${k_values[@]}"; do
    run_experiment "cora" "cmg" "deepwalk" "--cmg_k $k --cmg_d 20 --cmg_threshold 0.1" "cora_cmg_k${k}_deepwalk"
done

# Test different d values (embedding dimension)
d_values=(10 20 30 40 50)
echo "Testing different d values: ${d_values[@]}"

for d in "${d_values[@]}"; do
    run_experiment "cora" "cmg" "deepwalk" "--cmg_k 10 --cmg_d $d --cmg_threshold 0.1" "cora_cmg_d${d}_deepwalk"
done

# Test different threshold values
thresholds=(0.05 0.1 0.15 0.2 0.25)
echo "Testing different threshold values: ${thresholds[@]}"

for threshold in "${thresholds[@]}"; do
    run_experiment "cora" "cmg" "deepwalk" "--cmg_k 10 --cmg_d 20 --cmg_threshold $threshold" "cora_cmg_thresh${threshold}_deepwalk"
done

# ================================================================
# PART 3: BEST PARAMETERS ON ALL DATASETS
# ================================================================
echo "🎯 PART 3: Apply Best Parameters to All Datasets"

# You can modify these based on best results from parameter study
best_k=10
best_d=20  
best_threshold=0.1

for dataset in "${datasets[@]}"; do
    for embed_method in "${embed_methods[@]}"; do
        run_experiment $dataset "cmg" $embed_method "--cmg_k $best_k --cmg_d $best_d --cmg_threshold $best_threshold" "${dataset}_cmg_best_${embed_method}"
    done
done

echo "✅ All experiments completed!"
echo "📁 Results saved in results/ directory"
echo "📊 Run the analysis script to generate summary tables and plots"
