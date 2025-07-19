import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import os

def load_mtx_file(filepath):
    """Load matrix from MTX file"""
    try:
        return sp.io.mmread(filepath).tocsr()
    except:
        return None

def compute_spectrum(adjacency, k=20):
    """Compute graph Laplacian spectrum"""
    print(f"Computing spectrum for {adjacency.shape[0]} nodes, {adjacency.nnz} edges...")
    
    # Build normalized Laplacian
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    degrees_safe = degrees + 1e-12
    
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees_safe))
    laplacian = sp.diags(degrees) - adjacency
    laplacian_norm = d_inv_sqrt @ laplacian @ d_inv_sqrt
    
    # Compute smallest eigenvalues
    try:
        k_actual = min(k, adjacency.shape[0] - 2)
        eigenvals, eigenvecs = eigsh(laplacian_norm, k=k_actual, which='SM')
        eigenvals = np.sort(eigenvals)
        
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
    
    # Make symmetric (in case it's not)
    adjacency = adjacency.maximum(adjacency.T)
    
    eigenvals, eigenvecs = compute_spectrum(adjacency)
    if eigenvals is None:
        return None
    
    # Key spectral properties
    results = {
        'method': method_name,
        'nodes': adjacency.shape[0],
        'edges': adjacency.nnz // 2,  # Undirected
        'eigenvalues': eigenvals,
        'fiedler_value': eigenvals[1] if len(eigenvals) > 1 else 0,
        'spectral_gap': eigenvals[2] - eigenvals[1] if len(eigenvals) > 2 else 0,
        'algebraic_connectivity': eigenvals[1],
        'clustering_coefficient': None  # Would need NetworkX for this
    }
    
    print(f"✅ {method_name}:")
    print(f"   Nodes: {results['nodes']}")
    print(f"   Edges: {results['edges']}")
    print(f"   Fiedler value (λ₂): {results['fiedler_value']:.6f}")
    print(f"   Spectral gap (λ₃-λ₂): {results['spectral_gap']:.6f}")
    print(f"   First 10 eigenvalues: {eigenvals[:10]}")
    
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
