#!/usr/bin/env python3
"""
Quick diagnostic and fix for the spectral analysis issues
"""

import numpy as np
import scipy.sparse as sp
import pickle
from pathlib import Path
from scipy.sparse.linalg import eigsh

def diagnose_matrix_issues():
    """Diagnose why we're getting negative eigenvalues and disconnected graphs."""
    
    print("=== Matrix Diagnostics ===\n")
    
    matrices_dir = Path("extracted_matrices")
    
    # Check each Laplacian matrix
    for laplacian_file in matrices_dir.glob("*_laplacian.pkl"):
        name = laplacian_file.stem
        print(f"🔍 Analyzing {name}:")
        
        with open(laplacian_file, 'rb') as f:
            L = pickle.load(f)
        
        print(f"   Shape: {L.shape}")
        print(f"   NNZ: {L.nnz}")
        
        # Check diagonal
        diag = L.diagonal()
        print(f"   Diagonal range: {diag.min():.6f} to {diag.max():.6f}")
        
        # Check if symmetric
        diff = L - L.T
        max_diff = np.abs(diff.data).max() if diff.nnz > 0 else 0
        print(f"   Symmetry error: {max_diff:.10f}")
        
        # Check row sums (should be ~0 for Laplacian)
        row_sums = np.array(L.sum(axis=1)).flatten()
        print(f"   Row sum range: {row_sums.min():.6f} to {row_sums.max():.6f}")
        
        # Check if positive semidefinite
        try:
            # Get smallest eigenvalue
            lambda_min = eigsh(L, k=1, which='SM', return_eigenvectors=False)[0]
            print(f"   Smallest eigenvalue: {lambda_min:.8f}")
            
            if lambda_min < -1e-10:
                print(f"   ❌ Matrix is NOT positive semidefinite!")
                print(f"   This suggests it's not a proper Laplacian matrix")
            else:
                print(f"   ✅ Matrix is positive semidefinite")
            
        except Exception as e:
            print(f"   ❌ Could not compute smallest eigenvalue: {e}")
        
        # Check corresponding adjacency matrix
        adj_file = matrices_dir / f"{name.replace('_laplacian', '_adjacency')}.pkl"
        if adj_file.exists():
            print(f"   Checking corresponding adjacency matrix...")
            with open(adj_file, 'rb') as f:
                A = pickle.load(f)
            
            # Verify Laplacian construction: L = D - A
            degrees = np.array(A.sum(axis=1)).flatten()
            D = sp.diags(degrees)
            L_expected = D - A
            
            # Compare with actual L
            diff_matrix = L - L_expected
            max_diff = np.abs(diff_matrix.data).max() if diff_matrix.nnz > 0 else 0
            print(f"   L vs D-A difference: {max_diff:.10f}")
            
            if max_diff > 1e-10:
                print(f"   ⚠️  Laplacian doesn't match D-A construction!")
        
        print()

def fix_laplacian_matrices():
    """Fix Laplacian matrices by reconstructing from adjacency matrices."""
    
    print("=== Fixing Laplacian Matrices ===\n")
    
    matrices_dir = Path("extracted_matrices")
    
    for adj_file in matrices_dir.glob("*_adjacency.pkl"):
        base_name = adj_file.stem.replace('_adjacency', '')
        laplacian_file = matrices_dir / f"{base_name}_laplacian.pkl"
        
        print(f"🔧 Fixing {base_name}...")
        
        # Load adjacency matrix
        with open(adj_file, 'rb') as f:
            A = pickle.load(f)
        
        print(f"   Adjacency: {A.shape[0]} nodes, {A.nnz} edges")
        
        # Ensure adjacency is proper
        A = A.tocsr()
        A.data = np.abs(A.data)  # Ensure non-negative
        A = (A + A.T) / 2  # Ensure symmetric
        A.eliminate_zeros()
        
        # Check for negative values
        if np.any(A.data < 0):
            print(f"   ⚠️  Found negative values in adjacency, fixing...")
            A.data = np.maximum(A.data, 0)
        
        # Rebuild Laplacian: L = D - A
        degrees = np.array(A.sum(axis=1)).flatten()
        L = sp.diags(degrees) - A
        
        print(f"   Degree range: {degrees.min():.1f} to {degrees.max():.1f}")
        
        # Verify properties
        row_sums = np.array(L.sum(axis=1)).flatten()
        max_row_sum = np.abs(row_sums).max()
        print(f"   Row sum error: {max_row_sum:.10f}")
        
        # Save fixed Laplacian
        with open(laplacian_file, 'wb') as f:
            pickle.dump(L, f)
        
        # Quick eigenvalue check
        try:
            lambda_min = eigsh(L, k=1, which='SM', return_eigenvectors=False)[0]
            print(f"   Fixed smallest eigenvalue: {lambda_min:.8f}")
            
            if abs(lambda_min) < 1e-8:
                print(f"   ✅ Matrix is now properly constructed")
            else:
                print(f"   ⚠️  Still has issues: λ_min = {lambda_min:.8f}")
        except:
            print(f"   ❌ Still cannot compute eigenvalues")
        
        print()

def compute_connectivity_correctly():
    """Compute connectivity properly for all matrices."""
    
    print("=== Computing Connectivity Correctly ===\n")
    
    matrices_dir = Path("extracted_matrices")
    results = {}
    
    for laplacian_file in matrices_dir.glob("*_laplacian.pkl"):
        name = laplacian_file.stem.replace('_laplacian', '')
        print(f"📊 {name}:")
        
        with open(laplacian_file, 'rb') as f:
            L = pickle.load(f)
        
        try:
            # Compute first few eigenvalues
            eigenvals = eigsh(L, k=min(10, L.shape[0]-2), which='SM', return_eigenvectors=False)
            eigenvals = np.sort(eigenvals)
            
            # Count zero eigenvalues (connected components)
            zero_eigenvals = np.sum(np.abs(eigenvals) < 1e-8)
            
            # Fiedler value (second smallest)
            fiedler = eigenvals[1] if len(eigenvals) > 1 else 0.0
            
            print(f"   Nodes: {L.shape[0]}")
            print(f"   Components: {zero_eigenvals}")
            print(f"   Connected: {'Yes' if zero_eigenvals == 1 else 'No'}")
            print(f"   Fiedler value: {fiedler:.8f}")
            print(f"   First 5 eigenvalues: {eigenvals[:5]}")
            
            results[name] = {
                'nodes': L.shape[0],
                'components': zero_eigenvals,
                'connected': zero_eigenvals == 1,
                'fiedler': fiedler,
                'eigenvalues': eigenvals.tolist()
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[name] = {'error': str(e)}
        
        print()
    
    return results

if __name__ == "__main__":
    # Step 1: Diagnose issues
    diagnose_matrix_issues()
    
    # Step 2: Fix matrices
    fix_laplacian_matrices()
    
    # Step 3: Compute connectivity correctly
    results = compute_connectivity_correctly()
    
    # Summary
    print("=== SUMMARY ===")
    for name, data in results.items():
        if 'error' not in data:
            status = "Connected" if data['connected'] else f"Disconnected ({data['components']} components)"
            print(f"{name:20s}: {data['nodes']:4d} nodes, {status}, Fiedler = {data['fiedler']:.6f}")
