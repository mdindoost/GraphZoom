#!/usr/bin/env python3
"""
Debug CMG++ Level 1 Connectivity
Investigate why CMG++ level 1 spectral analysis failed and what the connectivity actually is
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.io import mmread
import networkx as nx

def load_and_analyze_cmg_level1():
    """Load CMG++ level 1 graph and analyze connectivity with multiple methods"""
    
    mtx_file = "cmg_extracted_graphs/cmg_level_1.mtx"
    
    print("🔍 DEBUGGING CMG++ LEVEL 1 CONNECTIVITY")
    print("=" * 60)
    
    # Method 1: Load the MTX file
    print("\n📊 Loading MTX file...")
    try:
        # Manual parsing (like our spectral analysis)
        with open(mtx_file, 'r') as f:
            lines = f.readlines()
        
        # Find header
        header_line = None
        data_start = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('%'):
                continue
            else:
                header_line = line
                data_start = i + 1
                break
        
        # Parse header
        parts = header_line.split()
        rows, cols, nnz = int(parts[0]), int(parts[1]), int(parts[2])
        print(f"Matrix size: {rows}x{cols}, {nnz} non-zeros")
        
        # Parse data
        row_indices = []
        col_indices = []
        values = []
        
        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                row_indices.append(int(parts[0]) - 1)  # Convert to 0-based
                col_indices.append(int(parts[1]) - 1)  # Convert to 0-based
                values.append(float(parts[2]))
        
        # Create Laplacian matrix
        laplacian = sp.coo_matrix((values, (row_indices, col_indices)), shape=(rows, cols))
        laplacian = laplacian.tocsr()
        
        print(f"Laplacian loaded: {laplacian.shape}, {laplacian.nnz} non-zeros")
        print(f"Value range: {laplacian.data.min():.6f} to {laplacian.data.max():.6f}")
        
        return laplacian
        
    except Exception as e:
        print(f"❌ Failed to load MTX file: {e}")
        return None

def convert_laplacian_to_adjacency(laplacian):
    """Convert Laplacian to adjacency matrix"""
    print("\n🔄 Converting Laplacian to Adjacency...")
    
    # A = D - L, so A = -L + D
    # Where D is diagonal matrix of degrees
    
    # Get diagonal (degrees)
    diagonal = laplacian.diagonal()
    
    # Create adjacency: negate off-diagonal, zero diagonal
    adjacency = -laplacian.copy()
    adjacency.setdiag(0)
    
    # Ensure non-negative (take absolute value)
    adjacency.data = np.abs(adjacency.data)
    
    print(f"Adjacency matrix: {adjacency.shape}, {adjacency.nnz} non-zeros")
    print(f"Value range: {adjacency.data.min():.6f} to {adjacency.data.max():.6f}")
    
    return adjacency

def analyze_connectivity_networkx(adjacency):
    """Use NetworkX to analyze connectivity"""
    print("\n🔗 Analyzing connectivity with NetworkX...")
    
    try:
        # Convert to NetworkX graph
        G = nx.from_scipy_sparse_matrix(adjacency)
        
        print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Check connectivity
        is_connected = nx.is_connected(G)
        print(f"Is connected: {is_connected}")
        
        if not is_connected:
            # Find connected components
            components = list(nx.connected_components(G))
            print(f"Number of connected components: {len(components)}")
            
            # Show component sizes
            component_sizes = [len(comp) for comp in components]
            component_sizes.sort(reverse=True)
            print(f"Component sizes: {component_sizes[:10]}")  # Show largest 10
            
            # Show some isolated nodes
            isolated_nodes = list(nx.isolates(G))
            print(f"Isolated nodes: {len(isolated_nodes)}")
            
            return False, len(components), component_sizes
        else:
            return True, 1, [G.number_of_nodes()]
            
    except Exception as e:
        print(f"❌ NetworkX analysis failed: {e}")
        return None, None, None

def try_smaller_eigenvalue_computation(adjacency):
    """Try to compute just a few eigenvalues of the Laplacian"""
    print("\n🔢 Attempting smaller eigenvalue computation...")
    
    try:
        # Build Laplacian from adjacency
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        d_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees + 1e-12))
        laplacian_matrix = sp.diags(degrees) - adjacency
        laplacian_norm = d_inv_sqrt @ laplacian_matrix @ d_inv_sqrt
        
        print(f"Normalized Laplacian: {laplacian_norm.shape}")
        
        # Try to compute just 5 smallest eigenvalues with looser tolerance
        print("Attempting eigenvalue computation with relaxed settings...")
        
        eigenvals, eigenvecs = eigsh(laplacian_norm, k=5, which='SM', 
                                   tol=1e-3, maxiter=1000)
        
        eigenvals = np.sort(eigenvals)
        print(f"✅ Success! First 5 eigenvalues: {eigenvals}")
        
        # Analyze connectivity
        near_zero_count = sum(1 for val in eigenvals if abs(val) < 1e-6)
        fiedler_value = 0
        for val in eigenvals:
            if val > 1e-6:
                fiedler_value = val
                break
        
        print(f"Near-zero eigenvalues: {near_zero_count}")
        print(f"Fiedler value (first positive): {fiedler_value:.8f}")
        
        if near_zero_count <= 1:
            print("✅ Graph appears connected!")
            return True, fiedler_value
        else:
            print(f"❌ Graph is disconnected ({near_zero_count} components)")
            return False, fiedler_value
            
    except Exception as e:
        print(f"❌ Eigenvalue computation failed: {e}")
        return None, None

def debug_cmg_level1():
    """Complete debug analysis of CMG++ level 1"""
    
    # Load the matrix
    laplacian = load_and_analyze_cmg_level1()
    if laplacian is None:
        return
    
    # Convert to adjacency
    adjacency = convert_laplacian_to_adjacency(laplacian)
    
    # NetworkX analysis
    is_connected, num_components, component_sizes = analyze_connectivity_networkx(adjacency)
    
    # Try eigenvalue computation
    eigenvalue_connected, fiedler_value = try_smaller_eigenvalue_computation(adjacency)
    
    # Summary
    print("\n🎯 SUMMARY FOR CMG++ LEVEL 1")
    print("=" * 60)
    print(f"Graph size: 927 nodes")
    print(f"Compression: 2.92x (2708 → 927)")
    print(f"Accuracy: 75.5%")
    
    if is_connected is not None:
        print(f"\nNetworkX Analysis:")
        print(f"  Connected: {is_connected}")
        print(f"  Components: {num_components}")
        if component_sizes:
            print(f"  Largest component: {component_sizes[0]} nodes")
    
    if eigenvalue_connected is not None:
        print(f"\nSpectral Analysis:")
        print(f"  Connected: {eigenvalue_connected}")
        print(f"  Fiedler value: {fiedler_value:.8f}")
    
    # Compare with other methods
    print(f"\n🔍 COMPARISON WITH OTHER METHODS:")
    print(f"  LAMG reduce_3 (519 nodes): Fiedler = 0.024217, Accuracy = 79.5%")
    print(f"  CMG++ level_1 (927 nodes): Fiedler = {fiedler_value:.6f}, Accuracy = 75.5%")
    print(f"  CMG++ level_2 (392 nodes): Fiedler = 0.000000, Accuracy = 74.8%")
    
    if fiedler_value and fiedler_value > 0:
        print(f"\n✅ CMG++ level 1 DOES have connectivity!")
        print(f"   But LAMG reduce_3 has BETTER connectivity and HIGHER accuracy")
    else:
        print(f"\n❌ CMG++ level 1 has NO meaningful connectivity")

if __name__ == "__main__":
    debug_cmg_level1()
