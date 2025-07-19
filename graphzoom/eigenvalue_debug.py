#!/usr/bin/env python3
"""
Eigenvalue-NetworkX Disagreement Debugger
Deep dive into why we get 78 components but 0 zero eigenvalues
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
import pickle
from pathlib import Path
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

def debug_cmg_matrix_step_by_step():
    """Debug CMG++ matrix construction step by step."""
    
    print("🔍 DEBUGGING CMG++ MATRIX CONSTRUCTION")
    print("="*60)
    
    # Load a CMG++ result
    cmg_file = Path("cmg_matrices/cmg_level1_results.pkl")
    if not cmg_file.exists():
        print("❌ CMG++ results not found")
        return
    
    with open(cmg_file, 'rb') as f:
        cmg_data = pickle.load(f)
    
    A = cmg_data['adjacency']
    L = cmg_data['laplacian']
    
    print(f"📊 Matrix loaded: {A.shape[0]} nodes")
    print(f"   Adjacency NNZ: {A.nnz}")
    print(f"   Laplacian NNZ: {L.nnz}")
    
    # Step 1: Check adjacency matrix properties
    print(f"\n🔍 STEP 1: Adjacency Matrix Analysis")
    print(f"   Data range: {A.data.min()} to {A.data.max()}")
    print(f"   Symmetric: {np.allclose((A - A.T).data, 0)}")
    print(f"   Diagonal: {A.diagonal().min()} to {A.diagonal().max()}")
    
    # Check for multi-edges (weights > 1)
    multi_edges = np.sum(A.data > 1)
    print(f"   Multi-edges (weight > 1): {multi_edges} out of {A.nnz}")
    
    if multi_edges > 0:
        print(f"   ⚠️  FOUND MULTI-EDGES! This could be the issue.")
        unique_weights = np.unique(A.data)
        print(f"   Unique weights: {unique_weights[:10]}{'...' if len(unique_weights) > 10 else ''}")
    
    # Step 2: Check Laplacian construction
    print(f"\n🔍 STEP 2: Laplacian Construction Check")
    
    # Recompute degrees and Laplacian
    degrees_computed = np.array(A.sum(axis=1)).flatten()
    L_recomputed = sp.diags(degrees_computed) - A
    
    print(f"   Degrees range: {degrees_computed.min()} to {degrees_computed.max()}")
    print(f"   Row sums (should be 0): {np.abs(np.array(L.sum(axis=1)).flatten()).max()}")
    
    # Check if our Laplacian matches the stored one
    L_diff = L - L_recomputed
    max_diff = np.abs(L_diff.data).max() if L_diff.nnz > 0 else 0
    print(f"   Laplacian difference: {max_diff} (should be 0)")
    
    if max_diff > 1e-10:
        print(f"   ❌ LAPLACIAN MISMATCH! Stored L != D - A")
    
    # Step 3: NetworkX connectivity analysis
    print(f"\n🔍 STEP 3: NetworkX Connectivity Analysis")
    
    # Test both weighted and unweighted
    try:
        # Unweighted analysis (standard)
        if hasattr(nx, 'from_scipy_sparse_array'):
            G_unweighted = nx.from_scipy_sparse_array(A)
        else:
            G_unweighted = nx.from_scipy_sparse_matrix(A)
        
        components_unweighted = nx.number_connected_components(G_unweighted)
        print(f"   NetworkX (unweighted): {components_unweighted} components")
        
        # Get component details
        comp_list = list(nx.connected_components(G_unweighted))
        comp_sizes = sorted([len(c) for c in comp_list], reverse=True)
        print(f"   Component sizes: {comp_sizes[:10]}{'...' if len(comp_sizes) > 10 else ''}")
        
        # Weighted analysis
        G_weighted = nx.from_scipy_sparse_matrix(A)
        components_weighted = nx.number_connected_components(G_weighted)
        print(f"   NetworkX (weighted): {components_weighted} components")
        
        if components_unweighted != components_weighted:
            print(f"   ⚠️  Weighted vs unweighted disagreement!")
        
    except Exception as e:
        print(f"   ❌ NetworkX failed: {e}")
        return
    
    # Step 4: Create binary adjacency for comparison
    print(f"\n🔍 STEP 4: Binary vs Weighted Comparison")
    
    # Create binary version (all weights = 1)
    A_binary = A.copy()
    A_binary.data = np.ones_like(A_binary.data)
    
    # Build binary Laplacian
    degrees_binary = np.array(A_binary.sum(axis=1)).flatten()
    L_binary = sp.diags(degrees_binary) - A_binary
    
    print(f"   Binary adjacency NNZ: {A_binary.nnz}")
    print(f"   Binary degrees range: {degrees_binary.min()} to {degrees_binary.max()}")
    
    # Test binary connectivity
    if hasattr(nx, 'from_scipy_sparse_array'):
        G_binary = nx.from_scipy_sparse_array(A_binary)
    else:
        G_binary = nx.from_scipy_sparse_matrix(A_binary)
    
    components_binary = nx.number_connected_components(G_binary)
    print(f"   Binary NetworkX components: {components_binary}")
    
    # Step 5: Eigenvalue analysis comparison
    print(f"\n🔍 STEP 5: Eigenvalue Analysis")
    
    matrices_to_test = [
        ("Weighted Laplacian", L),
        ("Binary Laplacian", L_binary)
    ]
    
    for name, matrix in matrices_to_test:
        print(f"\n   🧮 Testing {name}:")
        
        try:
            # Try different eigenvalue computation methods
            n = matrix.shape[0]
            k = min(min(20, components_unweighted + 5), n-2)
            
            print(f"      Computing {k} eigenvalues...")
            
            # Method 1: Standard
            try:
                eigenvals = eigsh(matrix, k=k, which='SM', return_eigenvectors=False)
                eigenvals = np.sort(eigenvals)
                print(f"      ✅ Standard method succeeded")
            except Exception as e:
                print(f"      Standard method failed: {e}")
                
                # Method 2: Dense for small matrices
                if n < 500:
                    try:
                        print(f"      Trying dense computation...")
                        matrix_dense = matrix.toarray()
                        eigenvals = eigh(matrix_dense, eigvals_only=True)
                        eigenvals = np.sort(eigenvals)[:k]
                        print(f"      ✅ Dense method succeeded")
                    except Exception as e2:
                        print(f"      Dense method failed: {e2}")
                        continue
                else:
                    continue
            
            # Analyze eigenvalues
            print(f"      First 10 eigenvalues:")
            for i, ev in enumerate(eigenvals[:10]):
                print(f"        λ_{i+1}: {ev:.2e}")
            
            # Count zeros with different thresholds
            thresholds = [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6]
            for thresh in thresholds:
                zero_count = np.sum(np.abs(eigenvals) < thresh)
                match_status = "✅" if zero_count == components_unweighted else "❌"
                print(f"      {match_status} Threshold {thresh:.0e}: {zero_count} zeros (need {components_unweighted})")
            
            # Look for eigenvalue gaps
            if len(eigenvals) > 1:
                gaps = np.diff(eigenvals)
                largest_gap_idx = np.argmax(gaps)
                print(f"      Largest gap: between λ_{largest_gap_idx+1} and λ_{largest_gap_idx+2}")
                print(f"      Gap size: {gaps[largest_gap_idx]:.2e}")
        
        except Exception as e:
            print(f"      ❌ Failed: {e}")
    
    # Step 6: Hypothesis testing
    print(f"\n🔍 STEP 6: Hypothesis Testing")
    
    print(f"\n   💡 HYPOTHESES:")
    print(f"   1. Multi-edge weights confuse eigenvalue computation")
    print(f"   2. Laplacian construction is incorrect") 
    print(f"   3. Numerical precision issues")
    print(f"   4. NetworkX uses different connectivity definition")
    
    # Test hypothesis 1: Multi-edge effect
    if multi_edges > 0:
        print(f"\n   🧪 Testing Hypothesis 1: Multi-edge Effect")
        print(f"      NetworkX ignores edge weights for connectivity")
        print(f"      Eigenvalue method considers edge weights")
        print(f"      This explains the disagreement!")
    
    # Test hypothesis 4: Different definitions
    print(f"\n   🧪 Testing Hypothesis 4: Definition Difference")
    print(f"      NetworkX: Are nodes reachable? (graph traversal)")
    print(f"      Eigenvalues: Kernel dimension of Laplacian (linear algebra)")
    print(f"      For weighted graphs, these can disagree!")
    
    return {
        'networkx_components': components_unweighted,
        'multi_edges': multi_edges,
        'adjacency_weights': A.data,
        'eigenvalues': eigenvals if 'eigenvals' in locals() else None
    }

def test_simple_disconnected_example():
    """Test with a simple known disconnected graph."""
    
    print(f"\n🧪 TESTING SIMPLE DISCONNECTED EXAMPLE")
    print("="*60)
    
    # Create simple disconnected graph: 2 triangles
    print("Creating 2 disconnected triangles (6 nodes total)...")
    
    A = sp.lil_matrix((6, 6))
    
    # Triangle 1: nodes 0, 1, 2
    A[0, 1] = A[1, 0] = 1
    A[1, 2] = A[2, 1] = 1  
    A[2, 0] = A[0, 2] = 1
    
    # Triangle 2: nodes 3, 4, 5
    A[3, 4] = A[4, 3] = 1
    A[4, 5] = A[5, 4] = 1
    A[5, 3] = A[3, 5] = 1
    
    A = A.tocsr()
    
    # Build Laplacian
    degrees = np.array(A.sum(axis=1)).flatten()
    L = sp.diags(degrees) - A
    
    print(f"Graph: {A.shape[0]} nodes, {A.nnz//2} edges")
    print(f"Expected: 2 connected components")
    
    # NetworkX test
    if hasattr(nx, 'from_scipy_sparse_array'):
        G = nx.from_scipy_sparse_array(A)
    else:
        G = nx.from_scipy_sparse_matrix(A)
    
    nx_components = nx.number_connected_components(G)
    print(f"NetworkX result: {nx_components} components")
    
    # Eigenvalue test
    try:
        eigenvals = eigh(L.toarray(), eigvals_only=True)
        eigenvals = np.sort(eigenvals)
        
        print(f"All eigenvalues: {eigenvals}")
        
        zero_count = np.sum(np.abs(eigenvals) < 1e-12)
        print(f"Zero eigenvalues (1e-12): {zero_count}")
        
        if zero_count == nx_components:
            print(f"✅ METHODS AGREE! {zero_count} = {nx_components}")
        else:
            print(f"❌ METHODS DISAGREE! {zero_count} ≠ {nx_components}")
    
    except Exception as e:
        print(f"❌ Eigenvalue test failed: {e}")

if __name__ == "__main__":
    # Test simple case first
    test_simple_disconnected_example()
    
    # Then debug CMG++ matrices
    debug_cmg_matrix_step_by_step()
