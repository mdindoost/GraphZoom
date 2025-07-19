#!/usr/bin/env python3
"""
CMG++ Graph Extractor
Extract coarsened graphs from CMG++ coarsening for spectral analysis
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.io import mmwrite
import sys
import os

# Add the GraphZoom directory to path to import CMG functions
sys.path.append('.')

def extract_cmg_graph_from_run(dataset='cora', level=2, cmg_k=10, cmg_d=10, threshold=0.1):
    """
    Run CMG++ coarsening and extract the coarsened graph
    """
    print(f"🔍 Extracting CMG++ graph: level={level}, k={cmg_k}, d={cmg_d}")
    
    # Import necessary modules
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import to_scipy_sparse_matrix
    from cmg_coarsening_timed import cmg_coarse
    
    # Load dataset
    if dataset == 'cora':
        from utils import json2mtx
        # Load the same way GraphZoom does
        laplacian = json2mtx(dataset)
        print(f"Original graph: {laplacian.shape[0]} nodes, {laplacian.nnz} edges")
    else:
        raise ValueError(f"Dataset {dataset} not supported yet")
    
    # Run CMG coarsening
    try:
        G, projections, laplacians, actual_level = cmg_coarse(
            laplacian, level=level, k=cmg_k, d=cmg_d, threshold=threshold
        )
        
        print(f"✅ CMG coarsening successful:")
        print(f"   Final graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
        print(f"   Levels completed: {actual_level}")
        
        # Extract adjacency matrix from NetworkX graph
        adjacency = nx.adjacency_matrix(G)
        
        # Convert to Laplacian (since that's what LAMG outputs)
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        laplacian_matrix = sp.diags(degrees) - adjacency
        
        return laplacian_matrix, adjacency, G, projections, laplacians
        
    except Exception as e:
        print(f"❌ CMG coarsening failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def save_cmg_graph_as_mtx(matrix, filename):
    """Save matrix in MTX format compatible with our spectral analysis"""
    try:
        # Make sure it's in COO format for MTX export
        matrix_coo = matrix.tocoo()
        
        # Save in MTX format
        mmwrite(filename, matrix_coo)
        print(f"✅ Saved CMG graph to: {filename}")
        
        # Verify the file format
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        print(f"   File format check:")
        print(f"   - Lines: {len(lines)}")
        print(f"   - Header: {lines[0].strip()}")
        print(f"   - First data line: {lines[1].strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save MTX file: {e}")
        return False

def extract_cmg_graphs_for_spectral_analysis():
    """Extract CMG++ graphs for spectral analysis"""
    print("🔬 CMG++ GRAPH EXTRACTION FOR SPECTRAL ANALYSIS")
    print("=" * 60)
    
    # Create output directory
    output_dir = "cmg_extracted_graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test configurations matching our spectral analysis
    configs = [
        {'level': 1, 'cmg_k': 10, 'cmg_d': 10, 'name': 'cmg_level_1'},
        {'level': 2, 'cmg_k': 10, 'cmg_d': 10, 'name': 'cmg_level_2'},
        {'level': 3, 'cmg_k': 10, 'cmg_d': 10, 'name': 'cmg_level_3'},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n🧪 Processing {config['name']}...")
        
        # Extract CMG graph
        laplacian, adjacency, G, projections, laplacians = extract_cmg_graph_from_run(
            level=config['level'], 
            cmg_k=config['cmg_k'], 
            cmg_d=config['cmg_d']
        )
        
        if laplacian is not None:
            # Save as MTX file (Laplacian format like LAMG)
            mtx_filename = f"{output_dir}/{config['name']}.mtx"
            if save_cmg_graph_as_mtx(laplacian, mtx_filename):
                results[config['name']] = {
                    'nodes': laplacian.shape[0],
                    'edges': len(G.edges()),
                    'mtx_file': mtx_filename,
                    'networkx_graph': G,
                    'projections': projections,
                    'laplacians': laplacians
                }
                
                print(f"✅ Successfully extracted {config['name']}:")
                print(f"   Nodes: {laplacian.shape[0]}")
                print(f"   Edges: {len(G.edges())}")
                print(f"   MTX file: {mtx_filename}")
        else:
            print(f"❌ Failed to extract {config['name']}")
    
    return results

def analyze_cmg_spectral_properties(results):
    """Analyze spectral properties of extracted CMG graphs"""
    print(f"\n🔍 ANALYZING CMG++ SPECTRAL PROPERTIES")
    print("=" * 60)
    
    # Import our spectral analysis function
    sys.path.append('complete_spectral_results')
    
    for name, data in results.items():
        print(f"\n📊 Analyzing {name}...")
        
        mtx_file = data['mtx_file']
        if os.path.exists(mtx_file):
            # Use our existing spectral analysis
            cmd = f"cd complete_spectral_results && python analyze_spectrum.py ../{mtx_file} {name}"
            os.system(cmd)
        else:
            print(f"❌ MTX file not found: {mtx_file}")

if __name__ == "__main__":
    # Extract CMG graphs
    results = extract_cmg_graphs_for_spectral_analysis()
    
    if results:
        print(f"\n✅ EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Extracted {len(results)} CMG++ graphs:")
        
        for name, data in results.items():
            print(f"   {name}: {data['nodes']} nodes, {data['edges']} edges")
        
        # Analyze spectral properties
        analyze_cmg_spectral_properties(results)
        
        print(f"\n📁 CMG++ graphs saved to: cmg_extracted_graphs/")
        print(f"📊 Now you can run spectral analysis on these MTX files!")
        
    else:
        print(f"\n❌ No CMG++ graphs extracted successfully")
        print("Check the error messages above for debugging info")
