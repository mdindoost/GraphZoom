import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.io import mmread
import sys
import os

def load_mtx_file(filepath):
    """Load matrix from MTX file with better error handling"""
    try:
        print(f"Attempting to load: {filepath}")
        if not os.path.exists(filepath):
            print(f"File does not exist: {filepath}")
            return None
        
        # Get file size
        file_size = os.path.getsize(filepath)
        print(f"File size: {file_size} bytes")
        
        if file_size == 0:
            print("File is empty")
            return None
        
        # Manual parsing for GraphZoom MTX format
        print("Using manual parsing for GraphZoom MTX format...")
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        print(f"File has {len(lines)} lines")
        
        # Find header (first non-comment line)
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
        
        if header_line is None:
            print("Could not find header line")
            return None
        
        # Parse header
        parts = header_line.split()
        if len(parts) < 3:
            print(f"Invalid header format: {header_line}")
            return None
        
        rows, cols, nnz = int(parts[0]), int(parts[1]), int(parts[2])
        print(f"Matrix: {rows}x{cols}, {nnz} non-zeros")
        
        # Parse data entries
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
        
        print(f"Parsed {len(values)} entries")
        
        # Create sparse matrix
        matrix = sp.coo_matrix((values, (row_indices, col_indices)), 
                             shape=(rows, cols))
        matrix = matrix.tocsr()
        
        print(f"Created matrix: {matrix.shape}, {matrix.nnz} non-zeros")
        
        # Check if this is a Laplacian matrix (has negative off-diagonal entries)
        if matrix.nnz > 0:
            min_val = matrix.data.min()
            max_val = matrix.data.max()
            print(f"Value range: {min_val:.6f} to {max_val:.6f}")
            
            if min_val < 0:
                print("⚠️  Matrix contains negative values - this appears to be a Laplacian matrix")
                print("Converting Laplacian to Adjacency matrix for spectral analysis...")
                
                # Convert Laplacian to Adjacency: A = D - L
                # where D is the diagonal (degree) matrix
                diagonal = matrix.diagonal()
                
                # Create adjacency matrix by negating off-diagonal entries
                adjacency = -matrix.copy()
                adjacency.setdiag(0)  # Remove diagonal entries
                
                # Ensure non-negative values
                adjacency.data = np.abs(adjacency.data)
                
                matrix = adjacency
                print(f"Converted to adjacency matrix: {matrix.shape}, {matrix.nnz} non-zeros")
        
        # Make symmetric if needed
        if matrix.shape[0] == matrix.shape[1]:
            matrix = matrix.maximum(matrix.T)
        
        return matrix
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None

def compute_spectrum(adjacency, k=10):
    """Compute graph Laplacian spectrum"""
    print(f"Computing spectrum for {adjacency.shape[0]} nodes, {adjacency.nnz} edges...")
    
    # Build normalized Laplacian
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    degrees_safe = degrees + 1e-12
    
    # Check for isolated nodes
    isolated_nodes = np.sum(degrees == 0)
    if isolated_nodes > 0:
        print(f"Warning: {isolated_nodes} isolated nodes found")
    
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees_safe))
    laplacian = sp.diags(degrees) - adjacency
    laplacian_norm = d_inv_sqrt @ laplacian @ d_inv_sqrt
    
    # Compute smallest eigenvalues
    try:
        n_nodes = adjacency.shape[0]
        k_actual = min(k, n_nodes - 2)
        
        if k_actual < 2:
            print(f"Graph too small for eigenvalue computation: {n_nodes} nodes")
            return np.array([0.0, 0.0]), None
            
        eigenvals, eigenvecs = eigsh(laplacian_norm, k=k_actual, which='SM', tol=1e-6)
        eigenvals = np.sort(eigenvals)
        
        # Basic sanity checks
        if eigenvals[0] < -1e-10:
            print(f"Warning: First eigenvalue is negative: {eigenvals[0]}")
        if eigenvals[0] > 1e-6:
            print(f"Warning: First eigenvalue is not close to zero: {eigenvals[0]}")
        
        return eigenvals, eigenvecs
        
    except Exception as e:
        print(f"Error computing spectrum: {e}")
        return None, None

def analyze_graph_file(filepath, method_name):
    """Analyze spectral properties of a graph file"""
    print(f"\n📊 Analyzing {method_name}: {filepath}")
    
    adjacency = load_mtx_file(filepath)
    if adjacency is None:
        print(f"❌ Could not load {filepath}")
        return None
    
    eigenvals, eigenvecs = compute_spectrum(adjacency)
    if eigenvals is None:
        print(f"❌ Could not compute spectrum for {filepath}")
        return None
    
    # Key spectral properties
    results = {
        'method': method_name,
        'nodes': adjacency.shape[0],
        'edges': adjacency.nnz // 2,  # Undirected
        'eigenvalues': eigenvals,
        'fiedler_value': eigenvals[1] if len(eigenvals) > 1 else 0,
        'spectral_gap': eigenvals[2] - eigenvals[1] if len(eigenvals) > 2 else 0,
        'algebraic_connectivity': eigenvals[1] if len(eigenvals) > 1 else 0,
    }
    
    print(f"✅ {method_name}:")
    print(f"   Nodes: {results['nodes']}")
    print(f"   Edges: {results['edges']}")
    print(f"   Fiedler value (λ₂): {results['fiedler_value']:.8f}")
    print(f"   Spectral gap (λ₃-λ₂): {results['spectral_gap']:.8f}")
    print(f"   First 5 eigenvalues: {eigenvals[:5]}")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python analyze_spectrum.py <graph_file> <method_name>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    method_name = sys.argv[2]
    
    results = analyze_graph_file(filepath, method_name)
    
    # Save results
    output_file = f"spectral_results_{method_name.lower()}.txt"
    with open(output_file, 'w') as f:
        if results:
            f.write(f"Method: {results['method']}\n")
            f.write(f"Nodes: {results['nodes']}\n")
            f.write(f"Edges: {results['edges']}\n")
            f.write(f"Fiedler_value: {results['fiedler_value']:.8f}\n")
            f.write(f"Spectral_gap: {results['spectral_gap']:.8f}\n")
            f.write(f"Eigenvalues: {','.join(map(str, results['eigenvalues']))}\n")
        else:
            f.write("Analysis failed\n")
    
    print(f"\n📁 Results saved to: {output_file}")
