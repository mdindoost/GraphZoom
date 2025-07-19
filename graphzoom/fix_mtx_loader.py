#!/usr/bin/env python3
"""
Robust MTX File Loader - Handle different MTX formats from LAMG
"""

import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from pathlib import Path

def inspect_mtx_file(filepath):
    """Inspect MTX file format and content."""
    
    print(f"\n🔍 Inspecting: {filepath}")
    
    # Read first few lines to understand format
    with open(filepath, 'r') as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(line.strip())
            if i > 10:  # Read first 11 lines
                break
    
    print("📄 First few lines:")
    for i, line in enumerate(lines):
        print(f"   {i+1:2d}: {line}")
    
    # Try different loading methods
    print("\n🔧 Testing different loading methods:")
    
    # Method 1: Standard scipy.io.mmread
    try:
        matrix = sio.mmread(str(filepath))
        print(f"   ✅ scipy.io.mmread: Success")
        print(f"      Shape: {matrix.shape}")
        print(f"      Type: {type(matrix)}")
        print(f"      NNZ: {matrix.nnz}")
        return matrix, "scipy"
    except Exception as e:
        print(f"   ❌ scipy.io.mmread: {e}")
    
    # Method 2: Manual parsing
    try:
        matrix = load_mtx_manual(filepath)
        print(f"   ✅ Manual parsing: Success") 
        print(f"      Shape: {matrix.shape}")
        print(f"      Type: {type(matrix)}")
        print(f"      NNZ: {matrix.nnz}")
        return matrix, "manual"
    except Exception as e:
        print(f"   ❌ Manual parsing: {e}")
    
    # Method 3: Try as dense matrix
    try:
        data = np.loadtxt(filepath, skiprows=get_header_lines(filepath))
        if data.ndim == 1:
            # Vector - convert to diagonal matrix
            matrix = sp.diags(data)
            print(f"   ✅ As vector/diagonal: Success")
            print(f"      Shape: {matrix.shape}")
            print(f"      Type: {type(matrix)}")
            return matrix, "vector"
        elif data.ndim == 2:
            if data.shape[1] == 2:
                # Edge list format (no weights)
                n = int(max(data.max(), data.shape[0]))
                matrix = sp.coo_matrix((np.ones(len(data)), (data[:, 0]-1, data[:, 1]-1)), shape=(n, n))
                print(f"   ✅ As edge list: Success")
                print(f"      Shape: {matrix.shape}")
                return matrix.tocsr(), "edge_list"
            elif data.shape[1] == 3:
                # COO format (row, col, value)
                n = int(max(data[:, 0].max(), data[:, 1].max()))
                matrix = sp.coo_matrix((data[:, 2], (data[:, 0]-1, data[:, 1]-1)), shape=(n, n))
                print(f"   ✅ As COO format: Success")
                print(f"      Shape: {matrix.shape}")
                return matrix.tocsr(), "coo"
            else:
                # Dense matrix
                matrix = sp.csr_matrix(data)
                print(f"   ✅ As dense matrix: Success")
                print(f"      Shape: {matrix.shape}")
                return matrix, "dense"
    except Exception as e:
        print(f"   ❌ As dense/vector: {e}")
    
    print(f"   ❌ All methods failed")
    return None, None

def get_header_lines(filepath):
    """Count header lines in MTX file."""
    
    header_lines = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('%') or line.startswith('#') or len(line) == 0:
                header_lines += 1
            else:
                # First non-comment line might be dimensions
                try:
                    parts = line.split()
                    if len(parts) in [2, 3] and all(part.replace('.', '').isdigit() for part in parts):
                        # Looks like dimensions or first data line
                        break
                    else:
                        header_lines += 1
                except:
                    header_lines += 1
                break
    
    return header_lines

def load_mtx_manual(filepath):
    """Manually parse MTX file."""
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header
    data_start = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line.startswith('%') and not line.startswith('#') and len(line) > 0:
            data_start = i
            break
    
    # Parse data
    data_lines = [line.strip() for line in lines[data_start:] if len(line.strip()) > 0]
    
    if len(data_lines) == 0:
        raise ValueError("No data found")
    
    # Try to determine format from first line
    first_line = data_lines[0].split()
    
    if len(first_line) == 2:
        # Could be dimensions or edge list
        try:
            rows, cols = int(first_line[0]), int(first_line[1])
            if len(data_lines) == 1:
                # Just dimensions - empty matrix
                return sp.csr_matrix((rows, cols))
            else:
                # Edge list format
                edges = []
                for line in data_lines[1:]:
                    parts = line.split()
                    if len(parts) >= 2:
                        edges.append([int(parts[0])-1, int(parts[1])-1])  # Convert to 0-indexed
                
                edges = np.array(edges)
                n = max(rows, cols, edges.max() + 1)
                matrix = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
                return matrix.tocsr()
        except:
            pass
    
    elif len(first_line) == 3:
        # COO format or dimensions + nnz
        try:
            val1, val2, val3 = int(first_line[0]), int(first_line[1]), int(first_line[2])
            
            if len(data_lines) == 1:
                # Dimensions line: rows, cols, nnz
                return sp.csr_matrix((val1, val2))
            else:
                # Check if first line is dimensions
                if val3 == len(data_lines) - 1:
                    # First line is dimensions
                    rows, cols, nnz = val1, val2, val3
                    
                    # Parse COO data
                    coo_data = []
                    for line in data_lines[1:]:
                        parts = line.split()
                        if len(parts) >= 2:
                            row, col = int(parts[0])-1, int(parts[1])-1  # Convert to 0-indexed
                            value = float(parts[2]) if len(parts) > 2 else 1.0
                            coo_data.append([row, col, value])
                    
                    if coo_data:
                        coo_data = np.array(coo_data)
                        matrix = sp.coo_matrix((coo_data[:, 2], (coo_data[:, 0], coo_data[:, 1])), 
                                             shape=(rows, cols))
                        return matrix.tocsr()
        except:
            pass
    
    # Try as simple data matrix
    try:
        data = np.array([[float(x) for x in line.split()] for line in data_lines])
        return sp.csr_matrix(data)
    except:
        pass
    
    raise ValueError("Could not parse MTX file format")

def load_all_lamg_matrices():
    """Load all LAMG matrices with robust parsing."""
    
    print("=== Loading LAMG Matrices ===")
    
    reduction_dir = Path("reduction_results")
    mtx_files = list(reduction_dir.glob("*.mtx"))
    
    loaded_matrices = {}
    
    for mtx_file in mtx_files:
        matrix, method = inspect_mtx_file(mtx_file)
        
        if matrix is not None:
            loaded_matrices[mtx_file.stem] = {
                'matrix': matrix,
                'method': method,
                'file': str(mtx_file)
            }
            
            print(f"   ✅ Successfully loaded {mtx_file.name}")
        else:
            print(f"   ❌ Failed to load {mtx_file.name}")
    
    return loaded_matrices

def analyze_lamg_matrix(matrix, name):
    """Analyze a loaded LAMG matrix."""
    
    print(f"\n📊 Analyzing {name}:")
    print(f"   Shape: {matrix.shape}")
    print(f"   NNZ: {matrix.nnz}")
    print(f"   Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6f}")
    
    # Check properties
    if matrix.shape[0] == matrix.shape[1]:
        # Square matrix
        diag = matrix.diagonal()
        print(f"   Diagonal range: [{diag.min():.3f}, {diag.max():.3f}]")
        
        # Check if symmetric
        try:
            diff = matrix - matrix.T
            is_symmetric = np.allclose(diff.data, 0, atol=1e-10)
            print(f"   Symmetric: {is_symmetric}")
        except:
            print(f"   Symmetric: Could not check")
        
        # Check if it looks like a Laplacian
        off_diag = matrix.data[matrix.diagonal() == 0] if hasattr(matrix, 'data') else []
        if len(off_diag) > 0:
            print(f"   Off-diagonal range: [{off_diag.min():.3f}, {off_diag.max():.3f}]")
            
            # Laplacian should have non-positive off-diagonals
            if np.all(off_diag <= 0):
                print(f"   Matrix type: Likely Laplacian (off-diag ≤ 0)")
            elif np.all(off_diag >= 0):
                print(f"   Matrix type: Likely Adjacency (off-diag ≥ 0)")
            else:
                print(f"   Matrix type: Mixed signs")
    
    # Try basic spectral analysis if small enough
    if matrix.shape[0] < 5000 and matrix.shape[0] == matrix.shape[1]:
        try:
            from scipy.sparse.linalg import eigsh
            
            # Get smallest eigenvalue
            lambda_min = eigsh(matrix, k=1, which='SM', return_eigenvectors=False, sigma=1e-6)[0]
            print(f"   Smallest eigenvalue: {lambda_min:.8f}")
            
            # Get largest eigenvalue  
            try:
                lambda_max = eigsh(matrix, k=1, which='LM', return_eigenvectors=False)[0]
                print(f"   Largest eigenvalue: {lambda_max:.8f}")
            except:
                print(f"   Largest eigenvalue: Could not compute")
            
        except Exception as e:
            print(f"   Eigenvalue analysis failed: {e}")

if __name__ == "__main__":
    
    # Load all LAMG matrices
    matrices = load_all_lamg_matrices()
    
    print(f"\n=== Analysis Summary ===")
    print(f"Successfully loaded {len(matrices)} matrices:")
    
    for name, info in matrices.items():
        analyze_lamg_matrix(info['matrix'], name)
    
    # Suggest which one is the coarsened graph
    print(f"\n💡 Identification:")
    
    for name, info in matrices.items():
        matrix = info['matrix']
        if matrix.shape[0] == matrix.shape[1]:
            if 400 < matrix.shape[0] < 800:
                print(f"   🎯 {name}: Likely the coarsened graph ({matrix.shape[0]} nodes)")
            elif matrix.shape[0] > 2000:
                print(f"   📊 {name}: Likely original/large graph ({matrix.shape[0]} nodes)")
            else:
                print(f"   ❓ {name}: Unknown purpose ({matrix.shape[0]} nodes)")
