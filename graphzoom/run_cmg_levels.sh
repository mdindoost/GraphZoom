#!/bin/bash

# CMG++ Hierarchical Matrix Generation Script
# Generates coarsened matrices in hierarchy: Level 1 → Level 2 → Level 3

echo "🚀 CMG++ HIERARCHICAL MATRIX GENERATION PIPELINE"
echo "=" | tr '\n' '=' | head -c 50; echo

# Configuration
DATASET_PATH="dataset/cora"
LEVELS=(1 2 3)
OUTPUT_DIR="cmg_matrices"
k=10
d=15

# Create output directory
mkdir -p $OUTPUT_DIR

# Function to run CMG++ hierarchical coarsening
run_cmg_hierarchical() {
    echo
    echo "📊 CMG++ Hierarchical Coarsening (k=$k, d=$d)"
    echo "----------------------------------------"
    
    # Create Python script for hierarchical coarsening
    cat > temp_cmg_hierarchical.py << EOF
#!/usr/bin/env python3
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import json
import torch
from torch_geometric.data import Data
import pickle
import sys
import os

def load_cora_original():
    """Load original Cora dataset."""
    with open("$DATASET_PATH/cora-G.json", 'r') as f:
        data_json = json.load(f)
    
    edges = data_json['links']
    n_nodes = len(data_json['nodes'])
    
    print(f"Original Cora: {n_nodes} nodes, {len(edges)} edges")
    
    # Build adjacency matrix
    A = sp.lil_matrix((n_nodes, n_nodes))
    for edge in edges:
        i, j = edge['source'], edge['target']
        A[i, j] = 1
        A[j, i] = 1
    A = A.tocsr()
    
    return A, n_nodes

def coarsen_graph(A_input, level, k=$k, d=$d):
    """Coarsen a graph using CMG++."""
    from filtered import cmg_filtered_clustering
    
    n_nodes = A_input.shape[0]
    print(f"Level {level}: Coarsening {n_nodes} nodes...")
    
    # Convert adjacency to edge_index for PyG
    coo = A_input.tocoo()
    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    
    # Create features (identity for graph structure focus)
    x = torch.eye(n_nodes, dtype=torch.float)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, num_nodes=n_nodes)
    
    # Set random seeds for reproducibility
    np.random.seed(42 + level)  # Different seed per level
    torch.manual_seed(42 + level)
    
    # Run CMG clustering
    clusters, n_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
        data, k=k, d=d, threshold=0.1
    )
    
    print(f"   CMG result: {n_clusters} clusters, λ_critical={lambda_crit:.6f}")
    
    # Build assignment matrix
    P = sp.lil_matrix((n_nodes, n_clusters))
    for i, cluster_id in enumerate(clusters):
        P[i, cluster_id] = 1.0
    P = P.tocsr()
    
    # Coarsen graph: A_coarse = P^T * A_input * P
    A_coarse = P.T @ A_input @ P
    A_coarse.eliminate_zeros()
    
    # Build Laplacian
    degrees = np.array(A_coarse.sum(axis=1)).flatten()
    L_coarse = sp.diags(degrees) - A_coarse
    
    print(f"   Result: {A_coarse.shape[0]} nodes, {A_coarse.nnz//2} edges")
    print(f"   Reduction: {n_nodes/A_coarse.shape[0]:.2f}x from previous level")
    
    return A_coarse, L_coarse, P, {
        'level': level,
        'input_nodes': n_nodes,
        'output_nodes': n_clusters,
        'lambda_critical': lambda_crit,
        'phi_stats': phi_stats,
        'clusters': clusters
    }

def run_hierarchical_coarsening():
    """Run complete hierarchical coarsening."""
    try:
        # Start with original Cora
        A_current, original_nodes = load_cora_original()
        
        results = {}
        
        # Hierarchical coarsening
        for level in [1, 2, 3]:
            A_coarse, L_coarse, P, info = coarsen_graph(A_current, level)
            
            # Calculate cumulative reduction from original
            cumulative_reduction = original_nodes / A_coarse.shape[0]
            
            # Store results
            results[f'level_{level}'] = {
                'level': level,
                'parameters': {'k': $k, 'd': $d},
                'nodes': A_coarse.shape[0],
                'edges': A_coarse.nnz // 2,
                'reduction_from_previous': info['input_nodes'] / A_coarse.shape[0],
                'cumulative_reduction': cumulative_reduction,
                'lambda_critical': info['lambda_critical'],
                'phi_stats': info['phi_stats'],
                'clusters': info['clusters'],
                'adjacency': A_coarse,
                'laplacian': L_coarse,
                'assignment': P
            }
            
            # Save individual level results
            with open(f'$OUTPUT_DIR/cmg_level{level}_results.pkl', 'wb') as f:
                pickle.dump(results[f'level_{level}'], f)
            
            # Save matrices
            sio.mmwrite(f'$OUTPUT_DIR/cmg_level{level}_adjacency.mtx', A_coarse)
            sio.mmwrite(f'$OUTPUT_DIR/cmg_level{level}_laplacian.mtx', L_coarse)
            sio.mmwrite(f'$OUTPUT_DIR/cmg_level{level}_assignment.mtx', P)
            
            print(f"   ✅ Saved Level {level} results")
            
            # Log success
            with open('$OUTPUT_DIR/generation_log.txt', 'a') as f:
                f.write(f"level_{level}:success:{A_coarse.shape[0]}:{cumulative_reduction:.2f}:{$k}:{$d}\\n")
            
            # Use coarsened graph as input for next level
            A_current = A_coarse
        
        # Save complete hierarchy
        with open('$OUTPUT_DIR/complete_hierarchy.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\\n✅ Hierarchical coarsening complete!")
        print(f"   Level 1: {results['level_1']['nodes']} nodes")
        print(f"   Level 2: {results['level_2']['nodes']} nodes")
        print(f"   Level 3: {results['level_3']['nodes']} nodes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in hierarchical coarsening: {e}")
        import traceback
        traceback.print_exc()
        
        # Log failure
        with open('$OUTPUT_DIR/generation_log.txt', 'a') as f:
            f.write(f"hierarchical:failed:{str(e).replace(':','_')}\\n")
        
        return False

if __name__ == "__main__":
    success = run_hierarchical_coarsening()
    sys.exit(0 if success else 1)
EOF
    
    # Run the hierarchical Python script
    python temp_cmg_hierarchical.py
    local exit_code=$?
    
    # Clean up temporary script
    rm -f temp_cmg_hierarchical.py
    
    return $exit_code
}

# Main execution
echo "Configuration:"
echo "  Dataset: $DATASET_PATH"
echo "  Levels: ${LEVELS[*]} (hierarchical)"
echo "  Parameters: k=$k, d=$d (consistent)"
echo "  Output: $OUTPUT_DIR/"
echo

# Check prerequisites
if [ ! -f "$DATASET_PATH/cora-G.json" ]; then
    echo "❌ Dataset file not found: $DATASET_PATH/cora-G.json"
    exit 1
fi

if [ ! -f "filtered.py" ]; then
    echo "❌ CMG implementation not found: filtered.py"
    exit 1
fi

echo "✅ Prerequisites checked"

# Initialize log
echo "# CMG++ Hierarchical Generation Log - $(date)" > $OUTPUT_DIR/generation_log.txt

# Run hierarchical coarsening
if run_cmg_hierarchical; then
    echo
    echo "📊 CMG++ HIERARCHICAL GENERATION SUMMARY"
    echo "=" | tr '\n' '=' | head -c 50; echo
    
    # Parse results from log
    echo "Results:"
    while IFS=':' read -r level_name status nodes reduction k d; do
        if [[ $status == "success" ]]; then
            echo "✅ $level_name: $nodes nodes (${reduction}x cumulative reduction)"
        else
            echo "❌ $level_name: $status"
        fi
    done < $OUTPUT_DIR/generation_log.txt
    
    echo
    echo "📈 Hierarchical coarsening successful!"
    echo "📁 Results saved to: $OUTPUT_DIR/"
    
    # List generated files
    echo
    echo "📄 Generated files:"
    ls -la $OUTPUT_DIR/ 2>/dev/null || echo "   No files generated"
    
    echo
    echo "🎯 Ready for analysis! Run: python analyze_matrices.py"
else
    echo
    echo "❌ Hierarchical coarsening failed"
fi

echo "=" | tr '\n' '=' | head -c 50; echo
