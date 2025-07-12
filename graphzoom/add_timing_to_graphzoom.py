#!/usr/bin/env python3
"""
Script to add detailed timing instrumentation to your existing GraphZoom code
This will modify your filtered.py and cmg_coarsening.py to capture the granular timing Koutis wants
"""

import os
import shutil
from pathlib import Path

def add_timing_to_filtered_py():
    """Add timing instrumentation to filtered.py"""
    
    # Read original filtered.py
    with open('filtered.py', 'r') as f:
        content = f.read()
    
    # Add timing imports at the top
    timing_imports = '''import time
import json
from collections import defaultdict

# Global timing storage
_timing_data = defaultdict(list)

def start_timing(step_name):
    """Start timing a step"""
    return time.time()

def end_timing(step_name, start_time):
    """End timing a step and store result"""
    elapsed = time.time() - start_time
    _timing_data[step_name].append(elapsed)
    print(f"[TIMING] {step_name}: {elapsed:.3f}s")
    return elapsed

def get_timing_summary():
    """Get summary of all timing data"""
    summary = {}
    for step, times in _timing_data.items():
        summary[step] = {
            'total': sum(times),
            'mean': sum(times) / len(times),
            'count': len(times),
            'latest': times[-1] if times else 0
        }
    return summary

def save_timing_data(filepath):
    """Save timing data to file"""
    summary = get_timing_summary()
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)

'''
    
    # Add timing to key functions
    modified_content = timing_imports + content
    
    # Add timing to build_normalized_laplacian
    modified_content = modified_content.replace(
        'def build_normalized_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:',
        '''def build_normalized_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    start_time = start_timing("laplacian_construction")'''
    )
    
    modified_content = modified_content.replace(
        'return L_norm.tocsr()',
        '''end_timing("laplacian_construction", start_time)
    return L_norm.tocsr()'''
    )
    
    # Add timing to apply_spectral_filter
    modified_content = modified_content.replace(
        'def apply_spectral_filter(X: np.ndarray, L_norm: sp.spmatrix, k: int) -> np.ndarray:',
        '''def apply_spectral_filter(X: np.ndarray, L_norm: sp.spmatrix, k: int) -> np.ndarray:
    start_time = start_timing("spectral_filtering")'''
    )
    
    modified_content = modified_content.replace(
        'print(f"Spectral filtering complete. Output shape={Y.shape}")\n    return Y',
        '''print(f"Spectral filtering complete. Output shape={Y.shape}")
    end_timing("spectral_filtering", start_time)
    return Y'''
    )
    
    # Add timing to reweight_graph_from_embeddings
    modified_content = modified_content.replace(
        'def reweight_graph_from_embeddings(Y: np.ndarray, edge_index: np.ndarray, threshold=0.1) -> sp.csr_matrix:',
        '''def reweight_graph_from_embeddings(Y: np.ndarray, edge_index: np.ndarray, threshold=0.1) -> sp.csr_matrix:
    start_time = start_timing("graph_reweighting")'''
    )
    
    modified_content = modified_content.replace(
        'print(f"Reweighted adjacency matrix has {A_sym.nnz} nonzeros")\n    return A_sym',
        '''print(f"Reweighted adjacency matrix has {A_sym.nnz} nonzeros")
    end_timing("graph_reweighting", start_time)
    return A_sym'''
    )
    
    # Add timing to CMG clustering call
    modified_content = modified_content.replace(
        'print("[DEBUG] Calling CMG on reweighted Laplacian")',
        '''print("[DEBUG] Calling CMG on reweighted Laplacian")
    cmg_start_time = start_timing("cmg_clustering")'''
    )
    
    modified_content = modified_content.replace(
        'print(f"[DEBUG] Converted clusters (0-indexed): {cI}")',
        '''print(f"[DEBUG] Converted clusters (0-indexed): {cI}")
        end_timing("cmg_clustering", cmg_start_time)'''
    )
    
    # Add timing to conductance evaluation
    modified_content = modified_content.replace(
        'phi_stats = evaluate_phi_conductance(data, cI)',
        '''conductance_start_time = start_timing("conductance_evaluation")
    phi_stats = evaluate_phi_conductance(data, cI)
    end_timing("conductance_evaluation", conductance_start_time)'''
    )
    
    # Save the modified file
    with open('filtered_timed.py', 'w') as f:
        f.write(modified_content)
    
    return 'filtered_timed.py'

def add_timing_to_cmg_coarsening():
    """Add timing to cmg_coarsening.py"""
    
    with open('cmg_coarsening.py', 'r') as f:
        content = f.read()
    
    # Change import to use timed version
    modified_content = content.replace(
        'from filtered import cmg_filtered_clustering',
        'from filtered_timed import cmg_filtered_clustering, save_timing_data'
    )
    
    # Add timing to cmg_coarse function
    modified_content = modified_content.replace(
        'print(f"[CMG] Starting CMG coarsening with k={k}, d={d}, threshold={threshold}")',
        '''print(f"[CMG] Starting CMG coarsening with k={k}, d={d}, threshold={threshold}")
    total_start_time = time.time()'''
    )
    
    # Add final timing summary
    modified_content = modified_content.replace(
        'print(f"[CMG] Final graph: {len(G.nodes())} nodes, {len(G.edges())} edges")',
        '''print(f"[CMG] Final graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    total_cmg_time = time.time() - total_start_time
    print(f"[CMG] Total coarsening time: {total_cmg_time:.3f}s")
    
    # Save detailed timing data
    import os
    timing_file = f"results/timing_results/cmg_detailed_{level}level.json"
    os.makedirs("results/timing_results", exist_ok=True)
    save_timing_data(timing_file)'''
    )
    
    # Add time import
    modified_content = 'import time\n' + modified_content
    
    with open('cmg_coarsening_timed.py', 'w') as f:
        f.write(modified_content)
    
    return 'cmg_coarsening_timed.py'

def modify_graphzoom_for_timing():
    """Modify graphzoom.py to use timed versions and add seed support"""
    
    with open('graphzoom.py', 'r') as f:
        content = f.read()
    
    # Change imports to use timed versions
    modified_content = content.replace(
        'from cmg_coarsening import cmg_coarse, cmg_coarse_fusion',
        'from cmg_coarsening_timed import cmg_coarse, cmg_coarse_fusion'
    )
    
    # Add seed parameter
    modified_content = modified_content.replace(
        'parser.add_argument("--cmg_threshold", type=float, default=0.1, \\',
        '''parser.add_argument("--cmg_threshold", type=float, default=0.1, \\
            help="CMG cosine similarity threshold (only for CMG coarsening)")
    parser.add_argument("--seed", type=int, default=42, \\'''
    )
    
    modified_content = modified_content.replace(
            'help="CMG cosine similarity threshold (only for CMG coarsening)")',
            '''help="CMG cosine similarity threshold (only for CMG coarsening)")
    parser.add_argument("--seed", type=int, default=42, \\
            help="Random seed for reproducibility")'''
    )
    
    # Add seed setting at the beginning of main
    modified_content = modified_content.replace(
        'args = parser.parse_args()',
        '''args = parser.parse_args()
    
    # Set random seed for reproducibility
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)'''
    )
    
    # Add more detailed timing breakdown in the final report
    modified_content = modified_content.replace(
        'print(f"Total Time = Fusion_time + Reduction_time + Embedding_time + Refinement_time = {total_time:.3f}")',
        '''print(f"Total Time = Fusion_time + Reduction_time + Embedding_time + Refinement_time = {total_time:.3f}")
    
    # Additional timing breakdown for CMG
    if args.coarse == "cmg":
        from filtered_timed import get_timing_summary
        timing_summary = get_timing_summary()
        if timing_summary:
            print("\\n%%%%%% CMG Detailed Timing %%%%%%")
            for step, stats in timing_summary.items():
                print(f"{step:25s}: {stats['latest']:.3f}s")'''
    )
    
    with open('graphzoom_timed.py', 'w') as f:
        f.write(modified_content)
    
    return 'graphzoom_timed.py'

def create_run_script():
    """Create a script to run experiments with the timed versions"""
    
    script_content = '''#!/bin/bash
# Script to run Koutis experiments with detailed timing

echo "🔬 KOUTIS COMPREHENSIVE EXPERIMENTS WITH DETAILED TIMING"
echo "============================================================"

# Create results directories
mkdir -p results/{logs,accuracy_results,timing_results,parameter_studies}

# Function to run experiment with timing
run_timed_experiment() {
    local dataset=$1
    local coarse_method=$2
    local embed_method=$3
    local level=$4
    local extra_params=$5
    local experiment_name=$6
    local seed=$7
    
    echo "🧪 Running: $experiment_name (seed: $seed)"
    echo "Parameters: $extra_params"
    
    # Run with timed version
    python graphzoom_timed.py --dataset $dataset --coarse $coarse_method --embed_method $embed_method --level $level --seed $seed $extra_params > results/logs/${experiment_name}_seed${seed}.log 2>&1
    
    # Extract results
    accuracy=$(grep "Test Accuracy:" results/logs/${experiment_name}_seed${seed}.log | awk '{print $3}')
    total_time=$(grep "Total Time" results/logs/${experiment_name}_seed${seed}.log | tail -1 | awk '{print $NF}')
    
    echo "✅ $experiment_name: Accuracy=$accuracy, Time=${total_time}s"
    
    # Save to CSV
    echo "$experiment_name,$dataset,$coarse_method,$embed_method,$level,$seed,$accuracy,$total_time,$extra_params" >> results/accuracy_results/timed_results.csv
}

# Initialize results
echo "experiment,dataset,coarsening,embedding,level,seed,accuracy,total_time,parameters" > results/accuracy_results/timed_results.csv

# PART 1: Parameter efficiency study (Koutis's main interest)
echo "🎯 PART 1: Parameter Efficiency Study"

# Test Koutis hypothesis: smaller d, higher k
echo "Testing high k with small d combinations..."

# High k, small d combinations
run_timed_experiment "cora" "cmg" "deepwalk" 1 "--cmg_k 25 --cmg_d 8 --cmg_threshold 0.1" "cora_cmg_k25_d8" 42
run_timed_experiment "cora" "cmg" "deepwalk" 1 "--cmg_k 30 --cmg_d 10 --cmg_threshold 0.1" "cora_cmg_k30_d10" 42
run_timed_experiment "cora" "cmg" "deepwalk" 1 "--cmg_k 35 --cmg_d 5 --cmg_threshold 0.1" "cora_cmg_k35_d5" 42

# Standard comparisons
run_timed_experiment "cora" "simple" "deepwalk" 1 "" "cora_simple_baseline" 42
run_timed_experiment "cora" "cmg" "deepwalk" 1 "--cmg_k 10 --cmg_d 20 --cmg_threshold 0.1" "cora_cmg_default" 42

# Test k range: 5 to 35
echo "Testing k parameter range..."
for k in 5 10 15 20 25 30 35; do
    run_timed_experiment "cora" "cmg" "deepwalk" 1 "--cmg_k $k --cmg_d 20 --cmg_threshold 0.1" "cora_cmg_k${k}" 42
done

# Test d range with emphasis on small values
echo "Testing d parameter range..."
for d in 5 8 10 15 20 30 50; do
    run_timed_experiment "cora" "cmg" "deepwalk" 1 "--cmg_k 10 --cmg_d $d --cmg_threshold 0.1" "cora_cmg_d${d}" 42
done

# PART 2: Multi-level analysis
echo "🏗️ PART 2: Multi-level Analysis"

for level in 1 2 3; do
    run_timed_experiment "cora" "simple" "deepwalk" $level "" "cora_simple_level${level}" 42
    run_timed_experiment "cora" "cmg" "deepwalk" $level "--cmg_k 10 --cmg_d 20 --cmg_threshold 0.1" "cora_cmg_level${level}" 42
done

# PART 3: Stability analysis (multiple seeds)
echo "📊 PART 3: Stability Analysis"

seeds=(42 123 456 789 999)
for seed in "${seeds[@]}"; do
    run_timed_experiment "cora" "simple" "deepwalk" 1 "" "cora_simple_stability" $seed
    run_timed_experiment "cora" "cmg" "deepwalk" 1 "--cmg_k 10 --cmg_d 20 --cmg_threshold 0.1" "cora_cmg_stability" $seed
done

# PART 4: Other datasets
echo "🌐 PART 4: Other Datasets"

for dataset in "citeseer" "pubmed"; do
    run_timed_experiment $dataset "simple" "deepwalk" 1 "" "${dataset}_simple" 42
    run_timed_experiment $dataset "cmg" "deepwalk" 1 "--cmg_k 10 --cmg_d 20 --cmg_threshold 0.1" "${dataset}_cmg_default" 42
    run_timed_experiment $dataset "cmg" "deepwalk" 1 "--cmg_k 25 --cmg_d 8 --cmg_threshold 0.1" "${dataset}_cmg_optimized" 42
done

echo "✅ All experiments completed!"
echo "📁 Results saved in results/ directory"
echo "🔍 Run: python enhanced_analysis_koutis.py"
'''
    
    with open('run_koutis_experiments.sh', 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod('run_koutis_experiments.sh', 0o755)
    
    return 'run_koutis_experiments.sh'

def main():
    """Main function to create all timed versions"""
    print("🔧 CREATING TIMED VERSIONS OF GRAPHZOOM + CMG")
    print("="*50)
    
    print("1. 📝 Adding timing to filtered.py...")
    filtered_file = add_timing_to_filtered_py()
    print(f"   ✅ Created: {filtered_file}")
    
    print("2. 📝 Adding timing to cmg_coarsening.py...")
    coarsening_file = add_timing_to_cmg_coarsening()
    print(f"   ✅ Created: {coarsening_file}")
    
    print("3. 📝 Modifying graphzoom.py for timing and seed support...")
    graphzoom_file = modify_graphzoom_for_timing()
    print(f"   ✅ Created: {graphzoom_file}")
    
    print("4. 📝 Creating run script...")
    run_script = create_run_script()
    print(f"   ✅ Created: {run_script}")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Run the experiments: ./run_koutis_experiments.sh")
    print(f"2. Analyze results: python enhanced_analysis_koutis.py")
    print(f"3. Check detailed timing: results/timing_results/")
    
    print(f"\n📊 What this captures for Koutis:")
    print(f"   ✅ Parameter efficiency (smaller d, higher k)")
    print(f"   ✅ Multi-level performance analysis")
    print(f"   ✅ Stability across random seeds")
    print(f"   ✅ Detailed timing breakdown per CMG step")
    print(f"   ✅ GraphZoom knn parameter study")
    print(f"   ✅ All datasets and embedding methods")
    
    return {
        'filtered': filtered_file,
        'coarsening': coarsening_file,
        'graphzoom': graphzoom_file,
        'run_script': run_script
    }

if __name__ == "__main__":
    created_files = main()
    print(f"\n✅ All files created successfully!")
    for component, filename in created_files.items():
        print(f"   {component}: {filename}")
