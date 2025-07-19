#!/usr/bin/env python3
"""
Quick test to verify CMG integration with correct file paths
"""

import numpy as np
import scipy.sparse as sp
import json
import torch
from torch_geometric.data import Data

def test_cora_loading():
    """Test loading Cora dataset with the correct file name."""
    
    print("=== Testing Cora Dataset Loading ===")
    
    dataset_path = "dataset/cora"
    
    # Load cora-G.json
    try:
        with open(f"{dataset_path}/cora-G.json", 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded cora-G.json")
        print(f"   Keys: {list(data.keys())}")
        print(f"   Nodes: {len(data['nodes'])}")
        print(f"   Links: {len(data['links'])}")
        
        # Show structure
        print(f"\n   Sample node: {data['nodes'][0]}")
        print(f"   Sample link: {data['links'][0]}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading cora-G.json: {e}")
        return None

def create_pyg_data(data):
    """Convert loaded JSON data to PyTorch Geometric format."""
    
    print("\n=== Converting to PyG Format ===")
    
    try:
        # Extract graph structure
        edges = data['links']
        nodes = data['nodes']
        n_nodes = len(nodes)
        
        print(f"   Original nodes: {n_nodes}")
        print(f"   Original edges: {len(edges)}")
        
        # Build edge list
        edge_list = []
        for edge in edges:
            src = edge['source']
            tgt = edge['target']
            edge_list.append((src, tgt))
            edge_list.append((tgt, src))  # Make undirected
        
        # Convert to tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        print(f"   Undirected edges: {edge_index.shape[1]}")
        
        # Load features
        try:
            features = np.load("dataset/cora/cora-feats.npy")
            x = torch.tensor(features, dtype=torch.float)
            print(f"   Features shape: {x.shape}")
        except FileNotFoundError:
            x = torch.eye(n_nodes, dtype=torch.float)
            print(f"   Using identity features: {x.shape}")
        
        # Create PyG Data object
        pyg_data = Data(x=x, edge_index=edge_index, num_nodes=n_nodes)
        
        print(f"✅ PyG Data created successfully")
        print(f"   Nodes: {pyg_data.num_nodes}")
        print(f"   Edges: {pyg_data.num_edges}")
        print(f"   Features: {pyg_data.x.shape}")
        
        return pyg_data
        
    except Exception as e:
        print(f"❌ Error creating PyG data: {e}")
        return None

def test_cmg_import():
    """Test importing and running CMG functions."""
    
    print("\n=== Testing CMG Import ===")
    
    try:
        from filtered import cmg_filtered_clustering
        print("✅ Successfully imported cmg_filtered_clustering")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Import warning: {e}")
        return False

def run_cmg_test(pyg_data):
    """Run a quick CMG clustering test."""
    
    print("\n=== Testing CMG Execution ===")
    
    try:
        from filtered import cmg_filtered_clustering
        
        # Test with small parameters first
        k, d = 5, 10
        print(f"   Running CMG with k={k}, d={d}...")
        
        clusters, n_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
            pyg_data, k=k, d=d, threshold=0.1
        )
        
        print(f"✅ CMG execution successful!")
        print(f"   Clusters found: {n_clusters}")
        print(f"   Lambda critical: {lambda_crit:.6f}")
        print(f"   Cluster range: {clusters.min()} to {clusters.max()}")
        
        return {
            'clusters': clusters,
            'n_clusters': n_clusters,
            'phi_stats': phi_stats,
            'lambda_crit': lambda_crit
        }
        
    except Exception as e:
        print(f"❌ CMG execution error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run complete integration test."""
    
    print("=== CMG Integration Test ===\n")
    
    # Step 1: Load dataset
    data = test_cora_loading()
    if data is None:
        return False
    
    # Step 2: Convert to PyG format
    pyg_data = create_pyg_data(data)
    if pyg_data is None:
        return False
    
    # Step 3: Test CMG import
    if not test_cmg_import():
        return False
    
    # Step 4: Run CMG
    cmg_result = run_cmg_test(pyg_data)
    if cmg_result is None:
        return False
    
    print("\n🎉 Integration test successful!")
    print("   Ready to run full matrix extraction and spectral analysis")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n📋 Next steps:")
        print("   1. python matrix_extractor.py")
        print("   2. python spectral_analyzer.py")
    else:
        print("\n⚠️ Integration test failed - check errors above")
