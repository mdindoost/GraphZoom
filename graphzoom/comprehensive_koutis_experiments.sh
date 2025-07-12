#!/bin/bash
# Comprehensive Experiments Based on Koutis's Feedback
# Focus: Parameter efficiency, multi-level comparison, stability analysis, detailed timing

# Create results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results_koutis_${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}/{logs,timing,parameters,multilevel,stability}

echo "🚀 COMPREHENSIVE KOUTIS EXPERIMENTS - $(date)"
echo "Results will be saved in: ${RESULTS_DIR}"
echo "=================================================================="

# Initialize CSV files with headers
echo "experiment,dataset,coarsening,embedding,level,cmg_k,cmg_d,cmg_threshold,knn_neighbors,accuracy,total_time,fusion_time,reduction_time,embedding_time,refinement_time,original_nodes,coarsened_nodes,compression_ratio,seed" > ${RESULTS_DIR}/all_experiments.csv

echo "experiment,step,time_seconds,notes" > ${RESULTS_DIR}/detailed_timing.csv

# Function to run experiment with detailed timing and logging
run_experiment_detailed() {
    local exp_name=$1
    local dataset=$2
    local coarse_method=$3
    local embed_method=$4
    local level=$5
    local extra_params=$6
    local seed=${7:-42}
    
    echo ""
    echo "🔄 Running: $exp_name"
    echo "   Dataset: $dataset | Method: $coarse_method | Embedding: $embed_method | Level: $level"
    echo "   Parameters: $extra_params | Seed: $seed"
    echo "   Time: $(date)"
    
    # Set random seed for reproducibility
    export PYTHONHASHSEED=$seed
    
    # Run experiment with timing
    start_time=$(date +%s.%N)
    
    if [ "$coarse_method" = "cmg" ]; then
        timeout 1800 python graphzoom.py \
            --dataset $dataset \
            --coarse $coarse_method \
            --embed_method $embed_method \
            --level $level \
            $extra_params \
            --seed $seed \
            > ${RESULTS_DIR}/logs/${exp_name}.log 2>&1
    else
        timeout 1800 python graphzoom.py \
            --dataset $dataset \
            --coarse $coarse_method \
            --embed_method $embed_method \
            --level $level \
            --seed $seed \
            > ${RESULTS_DIR}/logs/${exp_name}.log 2>&1
    fi
    
    exit_code=$?
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    # Extract results from log
    if [ $exit_code -eq 0 ]; then
        accuracy=$(grep "Test Accuracy:" ${RESULTS_DIR}/logs/${exp_name}.log | awk '{print $3}')
        total_time=$(grep "Total Time" ${RESULTS_DIR}/logs/${exp_name}.log | awk '{print $NF}')
        fusion_time=$(grep "Graph Fusion.*Time:" ${RESULTS_DIR}/logs/${exp_name}.log | awk '{print $4}')
        reduction_time=$(grep "Graph Reduction.*Time:" ${RESULTS_DIR}/logs/${exp_name}.log | awk '{print $4}')
        embedding_time=$(grep "Graph Embedding.*Time:" ${RESULTS_DIR}/logs/${exp_name}.log | awk '{print $4}')
        refinement_time=$(grep "Graph Refinement.*Time:" ${RESULTS_DIR}/logs/${exp_name}.log | awk '{print $4}')
        
        # Extract CMG-specific info
        coarsened_nodes=$(grep "Final graph:" ${RESULTS_DIR}/logs/${exp_name}.log | awk '{print $3}')
        cmg_clusters=$(grep "CMG found" ${RESULTS_DIR}/logs/${exp_name}.log | awk '{print $3}')
        
        # Calculate compression ratio
        case $dataset in
            "cora") original_nodes=2708 ;;
            "citeseer") original_nodes=3327 ;;
            "pubmed") original_nodes=19717 ;;
            *) original_nodes="unknown" ;;
        esac
        
        if [[ "$coarsened_nodes" =~ ^[0-9]+$ ]] && [[ "$original_nodes" =~ ^[0-9]+$ ]]; then
            compression_ratio=$(echo "scale=2; $original_nodes / $coarsened_nodes" | bc)
        else
            compression_ratio="N/A"
        fi
        
        # Extract parameters
        cmg_k=$(echo "$extra_params" | grep -o 'cmg_k [0-9]*' | awk '{print $2}')
        cmg_d=$(echo "$extra_params" | grep -o 'cmg_d [0-9]*' | awk '{print $2}')
        cmg_threshold=$(echo "$extra_params" | grep -o 'cmg_threshold [0-9.]*' | awk '{print $2}')
        knn_neighbors=$(echo "$extra_params" | grep -o 'num_neighs [0-9]*' | awk '{print $2}')
        
        # Save to CSV
        echo "$exp_name,$dataset,$coarse_method,$embed_method,$level,$cmg_k,$cmg_d,$cmg_threshold,$knn_neighbors,$accuracy,$total_time,$fusion_time,$reduction_time,$embedding_time,$refinement_time,$original_nodes,$coarsened_nodes,$compression_ratio,$seed" >> ${RESULTS_DIR}/all_experiments.csv
        
        echo "✅ $exp_name: Accuracy=$accuracy, Time=${total_time}s, Compression=${compression_ratio}x"
    elif [ $exit_code -eq 124 ]; then
        echo "⏰ TIMEOUT: $exp_name (30 min limit)"
        echo "$exp_name,$dataset,$coarse_method,$embed_method,$level,$cmg_k,$cmg_d,$cmg_threshold,$knn_neighbors,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,$original_nodes,TIMEOUT,TIMEOUT,$seed" >> ${RESULTS_DIR}/all_experiments.csv
    else
        echo "❌ FAILED: $exp_name (exit code: $exit_code)"
        echo "$exp_name,$dataset,$coarse_method,$embed_method,$level,$cmg_k,$cmg_d,$cmg_threshold,$knn_neighbors,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,$original_nodes,FAILED,FAILED,$seed" >> ${RESULTS_DIR}/all_experiments.csv
    fi
    
    # Log timing details
    echo "$exp_name,total_experiment,$duration,wall_clock_time" >> ${RESULTS_DIR}/detailed_timing.csv
}

echo ""
echo "=================================================================="
echo "🎯 PART 1: PARAMETER EFFICIENCY STUDY (Koutis's Main Suggestion)"
echo "=================================================================="
echo "Testing: 'CMG may do better than Zoom even for a somewhat smaller d'"

# CMG K parameter range: 5 to 35 (as requested)
k_values=(5 10 15 20 25 30 35)

# CMG D parameter wide range: 5 to 50  
d_values=(5 10 15 20 25 30 40 50)

# CMG threshold values
threshold_values=(0.05 0.1 0.15 0.2)

# GraphZoom KNN range for comparison
knn_values=(2 5 10 15 20)

echo "📊 Testing CMG K parameter (filter order): ${k_values[@]}"
for k in "${k_values[@]}"; do
    run_experiment_detailed "cora_cmg_k${k}_deepwalk" "cora" "cmg" "deepwalk" 1 "--cmg_k $k --cmg_d 20 --cmg_threshold 0.1"
done

echo "📊 Testing CMG D parameter (embedding dimension): ${d_values[@]}"
for d in "${d_values[@]}"; do
    run_experiment_detailed "cora_cmg_d${d}_deepwalk" "cora" "cmg" "deepwalk" 1 "--cmg_k 10 --cmg_d $d --cmg_threshold 0.1"
done

echo "📊 Testing CMG Threshold parameter: ${threshold_values[@]}"
for thresh in "${threshold_values[@]}"; do
    run_experiment_detailed "cora_cmg_thresh${thresh}_deepwalk" "cora" "cmg" "deepwalk" 1 "--cmg_k 10 --cmg_d 20 --cmg_threshold $thresh"
done

echo "📊 Testing GraphZoom KNN parameter: ${knn_values[@]}"
for knn in "${knn_values[@]}"; do
    run_experiment_detailed "cora_simple_knn${knn}_deepwalk" "cora" "simple" "deepwalk" 1 "--num_neighs $knn"
done

echo ""
echo "=================================================================="
echo "🏗️ PART 2: MULTI-LEVEL COMPARISON (GraphZoom levels 1,2,3)"
echo "=================================================================="
echo "Testing levels mentioned by Koutis: GraphZoom-1, GraphZoom-2, GraphZoom-3"

datasets=("cora" "citeseer" "pubmed")
embed_methods=("deepwalk" "node2vec")
levels=(1 2 3)

for dataset in "${datasets[@]}"; do
    for embed in "${embed_methods[@]}"; do
        for level in "${levels[@]}"; do
            # Simple coarsening multi-level
            run_experiment_detailed "${dataset}_simple_level${level}_${embed}" "$dataset" "simple" "$embed" $level ""
            
            # CMG multi-level with best parameters found
            run_experiment_detailed "${dataset}_cmg_level${level}_${embed}" "$dataset" "cmg" "$embed" $level "--cmg_k 10 --cmg_d 20 --cmg_threshold 0.1"
        done
    done
done

echo ""
echo "=================================================================="
echo "🎲 PART 3: STABILITY ANALYSIS (Multiple Random Seeds)"
echo "=================================================================="
echo "Testing: 'CMG will be more stable wrt to the initial random vectors'"

seeds=(1 2 3 4 5 42 123 456 789 999)

echo "📊 CMG Stability Test (10 different seeds)"
for seed in "${seeds[@]}"; do
    run_experiment_detailed "cora_cmg_stability_seed${seed}" "cora" "cmg" "deepwalk" 1 "--cmg_k 10 --cmg_d 20 --cmg_threshold 0.1" $seed
done

echo "📊 Simple Coarsening Stability Test (10 different seeds)"
for seed in "${seeds[@]}"; do
    run_experiment_detailed "cora_simple_stability_seed${seed}" "cora" "simple" "deepwalk" 1 "" $seed
done

echo ""
echo "=================================================================="
echo "⚡ PART 4: RUNTIME BREAKDOWN ANALYSIS"
echo "=================================================================="
echo "Testing: 'whether the runtime is dominated by that of deepwalk at the smallest level'"

# Best CMG configuration vs Simple on all datasets with timing focus
for dataset in "${datasets[@]}"; do
    for embed in "${embed_methods[@]}"; do
        # Baseline
        run_experiment_detailed "${dataset}_baseline_${embed}_timing" "$dataset" "simple" "$embed" 1 ""
        
        # Best CMG (from parameter study)
        run_experiment_detailed "${dataset}_cmg_best_${embed}_timing" "$dataset" "cmg" "$embed" 1 "--cmg_k 10 --cmg_d 10 --cmg_threshold 0.1"
    done
done

echo ""
echo "=================================================================="
echo "🔍 PART 5: OPTIMAL PARAMETER COMBINATIONS"
echo "=================================================================="
echo "Testing combinations of best individual parameters"

# Test promising combinations based on individual parameter results
optimal_combinations=(
    "--cmg_k 10 --cmg_d 10 --cmg_threshold 0.1"
    "--cmg_k 15 --cmg_d 10 --cmg_threshold 0.1" 
    "--cmg_k 10 --cmg_d 15 --cmg_threshold 0.1"
    "--cmg_k 20 --cmg_d 10 --cmg_threshold 0.1"
    "--cmg_k 10 --cmg_d 5 --cmg_threshold 0.1"
    "--cmg_k 25 --cmg_d 10 --cmg_threshold 0.15"
)

combo_num=1
for combo in "${optimal_combinations[@]}"; do
    for dataset in "${datasets[@]}"; do
        run_experiment_detailed "${dataset}_cmg_combo${combo_num}_deepwalk" "$dataset" "cmg" "deepwalk" 1 "$combo"
    done
    ((combo_num++))
done

echo ""
echo "=================================================================="
echo "✅ EXPERIMENTS COMPLETED - $(date)"
echo "=================================================================="

# Generate summary statistics
echo "📊 EXPERIMENT SUMMARY:"
total_experiments=$(wc -l < ${RESULTS_DIR}/all_experiments.csv)
echo "   Total experiments run: $((total_experiments - 1))"  # Subtract header

successful_experiments=$(grep -v "TIMEOUT\|FAILED" ${RESULTS_DIR}/all_experiments.csv | wc -l)
echo "   Successful experiments: $((successful_experiments - 1))"  # Subtract header

echo "   Results directory: ${RESULTS_DIR}"
echo "   Main results file: ${RESULTS_DIR}/all_experiments.csv"
echo "   Timing details: ${RESULTS_DIR}/detailed_timing.csv"
echo "   Individual logs: ${RESULTS_DIR}/logs/"

echo ""
echo "🔍 To analyze results, run:"
echo "   python analyze_koutis_experiments.py ${RESULTS_DIR}/all_experiments.csv"
echo ""
echo "📈 Key files for Koutis:"
echo "   • Parameter study: ${RESULTS_DIR}/all_experiments.csv"
echo "   • Timing breakdown: ${RESULTS_DIR}/detailed_timing.csv"
echo "   • Multi-level comparison: grep 'level[123]' ${RESULTS_DIR}/all_experiments.csv"
echo "   • Stability analysis: grep 'stability' ${RESULTS_DIR}/all_experiments.csv"
