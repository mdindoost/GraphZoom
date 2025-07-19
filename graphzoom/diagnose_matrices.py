#!/usr/bin/env python3
"""
Matrix Diagnostics - Check what matrices were extracted and their properties
"""

import numpy as np
import scipy.sparse as sp
import json
import pickle
from pathlib import Path

def diagnose_extracted_matrices():
    """Check what matrices were actually extracted."""
    
    matrices_dir = Path("extracted_matrices")
    
    print("=== Matrix Extraction Diagnostics ===\n")
    
    if not matrices_dir.exists():
        print("❌ extracted_matrices directory not found")
        return
    
    # List all files
    print("📁 Files in extracted_matrices:")
    for file in sorted(matrices_dir.iterdir()):
        if file.is_file():
            size_mb = file.stat().st_size / (1024*1024)
            print(f"   {file.name} ({size_mb:.2f} MB)")
    
    print()
    
    # Check summary
    summary_file = matrices_dir / "extraction_summary.json"
    if summary_file.exists():
        print("📋 Extraction Summary:")
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        for method, info in summary['methods'].items():
            print(f"\n   {method.upper()}:")
            for key, value in info.items():
                print(f"      {key}: {value}")
    
    # Check individual matrices
    print("\n🔍 Matrix Analysis:")
    
    pkl_files = list(matrices_dir.glob("*.pkl"))
    for pkl_file in pkl_files:
        print(f"\n   📄 {pkl_file.name}")
        
        try:
            with open(pkl_file, 'rb') as f:
                matrix = pickle.load(f)
            
            print(f"      Shape: {matrix.shape}")
            print(f"      NNZ: {matrix.nnz}")
            print(f"      Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6f}")
            print(f"      Format: {type(matrix)}")
            
            # Check if symmetric
            if matrix.shape[0] == matrix.shape[1]:
                try:
                    diff = matrix - matrix.T
                    is_symmetric = np.allclose(diff.data, 0, atol=1e-10)
                    print(f"      Symmetric: {is_symmetric}")
                except:
                    print(f"      Symmetric: Could not check")
            
            # Check connectivity (quick)
            if 'laplacian' in pkl_file.name and matrix.shape[0] < 5000:
                print("      Checking connectivity...")
                try:
                    # Check if matrix is positive semidefinite (all eigenvalues >= 0)
                    from scipy.sparse.linalg import eigsh
                    
                    # Get smallest eigenvalue
                    lambda_min = eigsh(matrix, k=1, which='SM', return_eigenvectors=False)[0]
                    print(f"      Smallest eigenvalue: {lambda_min:.8f}")
                    
                    if abs(lambda_min) < 1e-8:
                        print(f"      Status: Connected (λ_min ≈ 0)")
                    else:
                        print(f"      Status: Possible issues (λ_min = {lambda_min:.8f})")
                    
                    # Try to get second smallest (Fiedler)
                    if matrix.shape[0] > 2:
                        try:
                            eigenvals = eigsh(matrix, k=2, which='SM', return_eigenvectors=False)
                            fiedler = eigenvals[1]
                            print(f"      Fiedler value (λ₂): {fiedler:.8f}")
                            
                            if fiedler > 1e-8:
                                print(f"      Connectivity: Connected")
                            else:
                                print(f"      Connectivity: Disconnected")
                        except:
                            print(f"      Fiedler value: Could not compute")
                
                except Exception as e:
                    print(f"      Connectivity check failed: {e}")
        
        except Exception as e:
            print(f"      ❌ Error loading: {e}")
    
    print("\n=== Diagnosis Complete ===")

def check_lamg_files():
    """Check what LAMG files were generated."""
    
    print("\n=== LAMG Files Check ===")
    
    reduction_dir = Path("reduction_results")
    if not reduction_dir.exists():
        print("❌ reduction_results directory not found")
        return
    
    print("📁 LAMG files in reduction_results:")
    mtx_files = list(reduction_dir.glob("**/*.mtx"))
    
    for mtx_file in mtx_files:
        size_kb = mtx_file.stat().st_size / 1024
        print(f"   📄 {mtx_file.relative_to(reduction_dir)} ({size_kb:.1f} KB)")
        
        # Try to load and check
        try:
            import scipy.io as sio
            matrix = sio.mmread(str(mtx_file))
            print(f"      Shape: {matrix.shape}")
            print(f"      NNZ: {matrix.nnz}")
            print(f"      Type: {type(matrix)}")
        except Exception as e:
            print(f"      ❌ Load error: {e}")

def suggest_fixes():
    """Suggest fixes based on diagnosis."""
    
    print("\n=== Suggested Fixes ===")
    
    matrices_dir = Path("extracted_matrices")
    
    # Check for common issues
    laplacian_files = list(matrices_dir.glob("*laplacian*.pkl"))
    
    if not laplacian_files:
        print("❌ No Laplacian matrices found")
        print("   → Check matrix_extractor.py ran successfully")
    
    for lap_file in laplacian_files:
        try:
            with open(lap_file, 'rb') as f:
                matrix = pickle.load(f)
            
            # Check diagonal
            diag_vals = matrix.diagonal()
            if np.any(diag_vals < 0):
                print(f"⚠️  {lap_file.name} has negative diagonal entries")
                print("   → This suggests it's not a proper Laplacian")
            
            # Check if it's actually an adjacency matrix
            if np.all(matrix.data >= 0) and np.all(diag_vals == 0):
                print(f"⚠️  {lap_file.name} looks like an adjacency matrix, not Laplacian")
                print("   → May need to rebuild Laplacian: L = D - A")
        
        except Exception as e:
            print(f"❌ Error checking {lap_file.name}: {e}")
    
    print("\n💡 Recommendations:")
    print("1. Check that Laplacian matrices are properly constructed (L = D - A)")
    print("2. Handle disconnected graphs using shift-invert mode in eigsh")
    print("3. Use regularization for singular matrices")
    print("4. Check that LAMG matrix names match the expected pattern")

if __name__ == "__main__":
    diagnose_extracted_matrices()
    check_lamg_files() 
    suggest_fixes()
