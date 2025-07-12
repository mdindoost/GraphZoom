#!/bin/bash
# Comprehensive Experiments Based on Koutis's Feedback
# Addresses: parameter efficiency, multi-level, stability, detailed timing

# Create enhanced results structure
mkdir -p results/{logs,accuracy_results,timing_results,stability_results,parameter_studies}

# Enhanced experiment function with detailed timing and parameter extraction
run_enhanced_experiment() {
    local dataset=$1
    local coarse_method=$2
    local embed_method=$3
    local level=$4
    local extra_params=$5
    local experiment_name=$6
    local seed=$7
    
    echo "=========================================="
    echo "🔬 Running: $experiment_name"
    echo "📊 Dataset: $dataset | Coarsening: $coarse_method | Embedding: $embed_method | Level: $level | Seed: $seed"
    echo "⚙️  Parameters: $extra_params"
    echo "=========================================="
    
    # Set random seed for reproducibility
    export PYTHONHASHSEED=$seed
    
    # Run experiment with timing
    start_time=$(date +%s.%N)
    
    if [ "$coarse_method" = "cmg" ]; then
        timeout 1800 python graphzoom.py --dataset $dataset --coarse $coarse_method --embed_method $embed_method --level $level $extra_params --seed $seed > results/logs/${experiment_name}_seed${seed}.log 2>&1
    else
        timeout 1800 python graphzoom.py --dataset $dataset --coarse $coarse_method --embed_method $embed_method --level $level --seed $seed > results/logs/${experiment_name}_seed${seed}.log 2>&1
    fi
    
    exit_code=$?
    end_time=$(date +%s.%N)
    wall_time=$(echo "$end_time - $start_time" | bc)
    
    if [ $exit_code -eq 124 ]; then
        echo "⏰ TIMEOUT: $experiment_name (seed $seed)"
        echo "$experiment_name,$dataset,$coarse_method,$embed_method,$level,$seed,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,$extra_params,$wall_time" >> results/accuracy_results/comprehensive_results.csv
        return
    fi
    
    log_file="results/logs/${experiment_name}_seed${seed}.log"
    
    # Extract comprehensive metrics
    accuracy=$(grep "Test Accuracy:" $log_file | awk '{print $3}')
    total_time=$(grep "Total Time" $log_file | tail -1 | awk '{print $NF}')
    fusion_time=$(grep "Graph Fusion.*Time:" $log_file | awk '{print $4}')
    reduction_time=$(grep "Graph Reduction.*Time:" $log_file | awk '{print $4}')
    embedding_time=$(grep "Graph Embedding.*Time:" $log_file | awk '{print $4}')
    refinement_time=$(grep "Graph Refinement.*Time:" $log_file | awk '{print $4}')
    
    # Extract CMG-specific metrics
    original_nodes=$(grep "Original graph:" $log_file | awk '{print $4}')
    coarsened_nodes=$(grep "CMG found" $log_file | awk '{print $3}')
    lambda_critical=$(grep "λ_critical" $log_file | awk -F'≈' '{print $2}' | awk '{print $1}')
    conductance=$(grep "Average unweighted conductance" $log_file | awk '{print $NF}')
    reweighted_edges=$(grep "Reweighted adjacency matrix has" $log_file | awk '{print $5}')
    
    # Calculate compression ratio
    if [[ -n "$original_nodes" && -n "$coarsened_nodes" && "$coarsened_nodes" != "0" ]]; then
        compression_ratio=$(echo "scale=2; $original_nodes / $coarsened_nodes" | bc)
    else
        compression_ratio="N/A"
    fi
    
    # Parse CMG parameters from extra_params
    cmg_k=$(echo "$extra_params" | grep -o -- '--cmg_k [0-9]*' | awk '{print $2}')
    cmg_d=$(echo "$extra_params" | grep -o -- '--cmg_d [0-9]*' | awk '{print $2}')
    cmg_threshold=$(echo "$extra_params" | grep -o -- '--cmg_threshold [0-9.]*' | awk '{print $2}')
    num_neighs=$(echo "$extra_params" | grep -o -- '--num_neighs [0-9]*' | awk '{print $2}')
    
    # Set defaults if not found
    cmg_k=${cmg_k:-"N/A"}
    cmg_d=${cmg_d:-"N/A"}
    cmg_threshold=${cmg_threshold:-"N/A"}
    num_neighs=${num_neighs:-"N/A"}
    
    # Save comprehensive results
    echo "$experiment_name,$dataset,$coarse_method,$embed_method,$level,$seed,$accuracy,$total_time,$fusion_time,$reduction_time,$embedding_time,$refinement_time,$wall_time,$cmg_k,$cmg_d,$cmg_threshold,$num_neighs,$original_nodes,$coarsened_nodes,$compression_ratio,$lambda_critical,$conductance,$reweighted_edges,$extra_params" >> results/accuracy_results/comprehensive_results.csv
    
    echo "✅ Completed: $experiment_name (seed $seed) - Accuracy: $accuracy, Time: ${total_time}s"
}

# Initialize comprehensive results CSV
echo "experiment,dataset,coarsening,embedding,level,seed,accuracy,total_time,fusion_time,reduction_time,embedding_time,refinement_time,wall_time,cmg_k,cmg_d,cmg_threshold,num_neighs,original_nodes,coarsened_nodes,compression_ratio,lambda_critical,conductance,reweighted_edges,parameters" > results/accuracy_results/comprehensive_results.csv

echo "🚀 KOUTIS COMPREHENSIVE EVALUATION SUITE"
echo "📋 Addresses: Parameter efficiency, Multi-level, Stability, Detailed timing"
echo ""

# =================================================================
# PART 1: KOUTIS PARAMETER EFFICIENCY STUDY
# =================================================================
echo "🔬 PART 1: Koutis Parameter Efficiency Study"
echo "Testing: CMG with smaller d, higher k vs GraphZoom baseline"

# datasets=("cora" "citeseer" "pubmed")
datasets=("cora")
embed_methods=("deepwalk" "node2vec")

# Baseline GraphZoom with different knn values
echo "📊 Testing GraphZoom knn parameter sensitivity..."
knn_values=(3 5 10)

for dataset in "${datasets[@]}"; do
    for embed_method in "${embed_methods[@]}"; do
        for knn in "${knn_values[@]}"; do
            run_enhanced_experiment $dataset "simple" $embed_method 1 "--num_neighs $knn" "${dataset}_simple_${embed_method}_knn${knn}" 42
        done
    done
done

# CMG with Koutis's suggested parameter ranges
echo "🎯 Testing CMG parameter efficiency (smaller d, higher k)..."

# Wide k range: 20 to 35 (as requested)
k_values=(20 25 30 35)

# Wide d range with emphasis on smaller values (Koutis hypothesis)
d_values=(5 10 15)

# Test k parameter sweep
for dataset in "cora"; do  # Start with Cora for parameter optimization
    for k in "${k_values[@]}"; do
        run_enhanced_experiment $dataset "cmg" "deepwalk" 1 "--cmg_k $k --cmg_d 20 --cmg_threshold 0.1" "cora_cmg_k${k}_d20" 42
    done
done

# Test d parameter sweep 
for d in "${d_values[@]}"; do
    run_enhanced_experiment "cora" "cmg" "deepwalk" 1 "--cmg_k 10 --cmg_d $d --cmg_threshold 0.1" "cora_cmg_k10_d${d}" 42
done

# Test promising combinations of high k with low d (Koutis hypothesis)
echo "🔍 Testing high k with low d combinations..."
high_k_low_d_combinations=(
    "25 5"
    "25 10"
    "30 5"
    "30 10"
    "35 5"
    "35 10"
)

for combo in "${high_k_low_d_combinations[@]}"; do
    k=$(echo $combo | awk '{print $1}')
    d=$(echo $combo | awk '{print $2}')
    run_enhanced_experiment "cora" "cmg" "deepwalk" 1 "--cmg_k $k --cmg_d $d --cmg_threshold 0.1" "cora_cmg_k${k}_d${d}_combo" 42
done

# =================================================================
# PART 2: MULTI-LEVEL ANALYSIS (GraphZoom style)
# =================================================================
echo "🏗️ PART 2: Multi-level Analysis"
echo "Testing levels 1, 2, 3 with detailed per-level timing"

levels=(1 2 3)

for dataset in "${datasets[@]}"; do
    for embed_method in "${embed_methods[@]}"; do
        for level in "${levels[@]}"; do
            # Simple coarsening multi-level
            run_enhanced_experiment $dataset "simple" $embed_method $level "" "${dataset}_simple_${embed_method}_level${level}" 42
            
            # CMG multi-level with default parameters
            run_enhanced_experiment $dataset "cmg" $embed_method $level "--cmg_k 10 --cmg_d 20 --cmg_threshold 0.1" "${dataset}_cmg_${embed_method}_level${level}" 42
        done
    done
done

# Test best parameters at different levels
echo "🎯 Testing best parameters across multiple levels..."
best_params=(
    "--cmg_k 25 --cmg_d 5 --cmg_threshold 0.1"   # High k, low d
    "--cmg_k 30 --cmg_d 10 --cmg_threshold 0.1"  # Very high k, low d  
    "--cmg_k 10 --cmg_d 5 --cmg_threshold 0.1"   # Standard k, very low d
)

for params in "${best_params[@]}"; do
    param_name=$(echo "$params" | sed 's/--cmg_//g' | sed 's/ /_/g')
    for level in "${levels[@]}"; do
        run_enhanced_experiment "cora" "cmg" "deepwalk" $level "$params" "cora_cmg_${param_name}_level${level}" 42
    done
done

# =================================================================
# PART 3: STABILITY/VARIANCE ANALYSIS (Koutis mentioned this)
# =================================================================
echo "📊 PART 3: Stability & Variance Analysis"
echo "Testing CMG stability vs GraphZoom with multiple random seeds"

seeds=(42 123 456)

echo "🎲 Running stability tests with ${#seeds[@]} different seeds..."

# Test baseline stability
for seed in "${seeds[@]}"; do
    run_enhanced_experiment "cora" "simple" "deepwalk" 1 "" "cora_simple_deepwalk_stability" $seed
    run_enhanced_experiment "cora" "cmg" "deepwalk" 1 "--cmg_k 10 --cmg_d 20 --cmg_threshold 0.1" "cora_cmg_default_stability" $seed
done

# Test best parameter combinations stability
for seed in "${seeds[@]}"; do
    run_enhanced_experiment "cora" "cmg" "deepwalk" 1 "--cmg_k 25 --cmg_d 8 --cmg_threshold 0.1" "cora_cmg_best_stability" $seed
done

# # Test on other datasets with fewer seeds
# stability_seeds=(42 123 456)
# for dataset in "citeseer" "pubmed"; do
#     for seed in "${stability_seeds[@]}"; do
#         run_enhanced_experiment $dataset "simple" "deepwalk" 1 "" "${dataset}_simple_deepwalk_stability" $seed
#         run_enhanced_experiment $dataset "cmg" "deepwalk" 1 "--cmg_k 10 --cmg_d 20 --cmg_threshold 0.1" "${dataset}_cmg_default_stability" $seed
#     done
# done

# =================================================================
# PART 4: THRESHOLD PARAMETER STUDY
# =================================================================
echo "🎛️ PART 4: Threshold Parameter Study"

thresholds=(0.05 0.1 0.15)

for threshold in "${thresholds[@]}"; do
    run_enhanced_experiment "cora" "cmg" "deepwalk" 1 "--cmg_k 10 --cmg_d 20 --cmg_threshold $threshold" "cora_cmg_thresh${threshold}" 42
done

# =================================================================
# PART 5: FINAL VALIDATION ON ALL DATASETS
# =================================================================
echo "🎯 PART 5: Final Validation - Apply Best Parameters to All Datasets"

# These will be determined from parameter study, but for now use promising ones
final_test_params=(
    "--cmg_k 25 --cmg_d 5 --cmg_threshold 0.1"    # High k, low d (Koutis hypothesis)
    "--cmg_k 30 --cmg_d 5 --cmg_threshold 0.1"   # Very high k, low d
)

for dataset in "${datasets[@]}"; do
    for embed_method in "${embed_methods[@]}"; do
        for params in "${final_test_params[@]}"; do
            param_id=$(echo "$params" | sed 's/--cmg_k \([0-9]*\) --cmg_d \([0-9]*\).*/k\1_d\2/')
            run_enhanced_experiment $dataset "cmg" $embed_method 1 "$params" "${dataset}_cmg_${embed_method}_${param_id}" 42
        done
    done
done

echo ""
echo "✅ KOUTIS COMPREHENSIVE EVALUATION COMPLETED!"
echo "📁 Results structure:"
echo "   📊 results/accuracy_results/comprehensive_results.csv - All experimental data"
echo "   📝 results/logs/ - Detailed logs for each run"
echo "   🔬 Ready for statistical analysis"
echo ""
echo "🎯 Key Questions Addressed:"
echo "   1. ✅ Parameter efficiency (smaller d, higher k)"
echo "   2. ✅ Multi-level performance (1, 2, 3 levels)"
echo "   3. ✅ Stability across random seeds"
echo "   4. ✅ Detailed timing breakdown"
echo "   5. ✅ GraphZoom knn parameter study"
echo ""
echo "📈 Next: Run enhanced analysis script to get insights!"
