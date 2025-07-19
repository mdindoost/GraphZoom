#!/usr/bin/env python3
"""
CMG Implementation Adapter
Adapts the matrix extractor to work with your specific CMG code files.
"""

import numpy as np
import scipy.sparse as sp
import json
import torch
from torch_geometric.data import Data
import sys
from pathlib import Path

def load_cora_dataset(dataset_path="dataset/cora"):
    """Load Cora dataset in the format your code expects."""
    print(f"Loading Cora dataset from {dataset_path}...")
    
    # Load the JSON file
    with open(f"{dataset_path}/cora.json", 'r') as f:
        data = json.load(f)
    
    # Extract graph structure
    edges = data['links']
    nodes = data['nodes']
    n_nodes = len(nodes)
    
    print(f"Found {n_nodes} nodes, {len(edges)} edges")
    
    # Build edge_index for PyTorch Geometric
    edge_list = [(edge['source'], edge['target']) for edge in edges]
    
    # Make undirected (add reverse edges)
    edge_list_undirected = []
    for src, tgt in edge_list:
        edge_list_undirected.append((src, tgt))
        edge_list_undirected.append((tgt, src))
    
    edge_index = torch.tensor(edge_list_undirected, dtype=torch.long).t().contiguous()
    
    # Load features if available
    try:
        features = np.load(f"{dataset_path}/cora-feats.npy")
        x = torch.tensor(features, dtype=torch.float)
        print(f"Loaded features: {x.shape}")
    except FileNotFoundError:
        print("Features not found, using identity matrix")
        x = torch.eye(n_nodes, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, num_nodes=n_nodes)
    
    return data

def run_cmg_with_your_code(data, level=2, k=10, d=10, threshold=0.1):
    """Run CMG using your specific implementation."""
    
    print(f"Running CMG with parameters: level={level}, k={k}, d={d}, threshold={threshold}")
    
    try:
        # Import your CMG modules - adjust these imports based on your file structure
        from filtered import cmg_filtered_clustering
        
        print("✅ Successfully imported CMG modules")
        
        # Run the clustering
        clusters, n_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
            data, k=k, d=d, threshold=threshold
        )
        
        print(f"✅ CMG clustering complete: {n_clusters} clusters")
        print(f"   λ_critical = {lambda_crit:.6f}")
        
        return {
            'clusters': clusters,
            'n_clusters': n_clusters,
            'phi_stats': phi_stats,
            'lambda_crit': lambda_crit,
            'success': True
        }
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Available files in current directory:")
        for f in Path(".").glob("*.py"):
            print(f"   - {f}")
        return {'success': False, 'error': str(e)}
    
    except Exception as e:
        print(f"❌ CMG execution error: {e}")
        return {'success': False, 'error': str(e)}

def extract_adjacency_from_clusters(data, clusters, n_clusters):
    """Extract coarsened adjacency matrix from cluster assignments."""
    
    n_nodes = data.num_nodes
    
    # Build assignment matrix P: (n_nodes, n_clusters)
    P = sp.lil_matrix((n_nodes, n_clusters))
    for node_id, cluster_id in enumerate(clusters):
        P[node_id, cluster_id] = 1.0
    P = P.tocsr()
    
    # Build original adjacency matrix
    edge_index = data.edge_index.cpu().numpy()
    rows, cols = edge_index[0], edge_index[1]
    weights = np.ones(len(rows))
    A_orig = sp.coo_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))
    A_orig = A_orig.tocsr()
    
    # Coarsen: A_coarse = P^T * A_orig * P
    A_coarse = P.T @ A_orig @ P
    A_coarse.eliminate_zeros()
    
    # Build coarsened Laplacian
    degrees = np.array(A_coarse.sum(axis=1)).flatten()
    L_coarse = sp.diags(degrees) - A_coarse
    
    print(f"Coarsened graph: {n_clusters} nodes, {A_coarse.nnz//2} edges")
    
    return A_coarse, L_coarse, P

def test_cmg_integration():
    """Test integration with your CMG code."""
    
    print("=== Testing CMG Integration ===\n")
    
    # Load dataset
    try:
        data = load_cora_dataset()
        print(f"✅ Dataset loaded: {data.num_nodes} nodes")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False
    
    # Test CMG execution
    cmg_result = run_cmg_with_your_code(data, level=2, k=10, d=10)
    
    if not cmg_result['success']:
        print(f"❌ CMG execution failed: {cmg_result['error']}")
        return False
    
    # Extract matrices
    try:
        A_coarse, L_coarse, P = extract_adjacency_from_clusters(
            data, cmg_result['clusters'], cmg_result['n_clusters']
        )
        print(f"✅ Matrix extraction successful")
        print(f"   Coarsened adjacency: {A_coarse.shape}")
        print(f"   Coarsened Laplacian: {L_coarse.shape}")
        print(f"   Assignment matrix: {P.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Matrix extraction failed: {e}")
        return False

def create_adapted_extractor():
    """Create an adapted version of the matrix extractor for your setup."""
    
    adapted_code = '''
def extract_cmg_matrices_adapted(self, dataset_path="dataset/cora", level=2, k=10, d=10):
    """Adapted CMG matrix extraction for your specific implementation."""
    print(f"Running adapted CMG extraction (level={level}, k={k}, d={d})...")
    
    try:
        # Load dataset
        with open(f"{dataset_path}/cora.json", 'r') as f:
            data_json = json.load(f)
        
        edges = data_json['links']
        n_nodes = len(data_json['nodes'])
        
        # Build edge_index
        edge_list = [(edge['source'], edge['target']) for edge in edges]
        edge_list_undirected = []
        for src, tgt in edge_list:
            edge_list_undirected.append((src, tgt))
            edge_list_undirected.append((tgt, src))
        
        edge_index = torch.tensor(edge_list_undirected, dtype=torch.long).t().contiguous()
        
        # Load features or use identity
        try:
            features = np.load(f"{dataset_path}/cora-feats.npy")
            x = torch.tensor(features, dtype=torch.float)
        except FileNotFoundError:
            x = torch.eye(n_nodes, dtype=torch.float)
        
        # Create Data object
        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index, num_nodes=n_nodes)
        
        # Run your CMG code
        from filtered import cmg_filtered_clustering
        clusters, n_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
            data, k=k, d=d, threshold=0.1
        )
        
        # Extract matrices (same as before)
        # ... rest of extraction code ...
        
    except Exception as e:
        print(f"Error in adapted CMG extraction: {e}")
        return {}
'''
    
    print("Adapted extractor code template:")
    print(adapted_code)

if __name__ == "__main__":
    
    # Test the integration
    success = test_cmg_integration()
    
    if success:
        print("\n🎉 Integration test successful!")
        print("You can now run the full analysis pipeline:")
        print("   1. python matrix_extractor.py")
        print("   2. python spectral_analyzer.py")
    else:
        print("\n⚠️  Integration test failed.")
        print("Please check:")
        print("   1. CMG code files are in the current directory")
        print("   2. Import paths in matrix_extractor.py are correct") 
        print("   3. Dataset files are available")
        
        print("\nCreating adaptation template...")
        create_adapted_extractor()
