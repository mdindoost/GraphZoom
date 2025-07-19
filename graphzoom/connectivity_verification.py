#!/usr/bin/env python3
"""
Multi-Method Connectivity Verification
Verify graph connectivity using multiple independent approaches
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.io import mmread
import networkx as nx
from pathlib import Path

def load_mtx_as_adjacency(filepath):
    """Load MTX file and convert to adjacency matrix"""
    try:
        print(f"Loading: {filepath}")
        
        # Manual parsing (like our spectral analysis)
        with open(filepath, 'r') as f:
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
        
        if header_line is None:
            return None, "No header found"
        
        # Parse header
        parts = header_line.split()
        rows, cols, nnz = int(parts[0]), int(parts[1]), int(parts[2])
        
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
                col_indices.append(int(parts[1]) - 1)
                values.append(float(parts[2]))
        
        # Create matrix
        matrix = sp.coo_matrix((values, (row_indices, col_indices)), shape=(rows, cols))
        matrix = matrix.tocsr()
        
        print(f"  Matrix: {matrix.shape}, {matrix.nnz} non-zeros")
        print(f"  Value range: [{matrix.data.min():.6f}, {matrix.data.max():.6f}]")
        
        # Convert Laplacian to Adjacency if needed
        if matrix.data.min() < 0:
            print("  Converting Laplacian to Adjacency...")
            # A = |off-diagonal entries of -L|
            adjacency = -matrix.copy()
            adjacency.setdiag(0)  # Remove diagonal
            adjacency.data = np.abs(adjacency.data)
            adjacency.eliminate_zeros()
            
            print(f"  Adjacency: {adjacency.shape}, {adjacency.nnz} non-zeros")
            return adjacency, "Converted from Laplacian"
        else:
            return matrix, "Direct adjacency matrix"
            
    except Exception as e:
        return None, f"Error: {e}"

def analyze_connectivity_scipy(adjacency):
    """Analyze connectivity using scipy.sparse.csgraph"""
    try:
        # Make symmetric
        adjacency_sym = adjacency.maximum(adjacency.T)
        
        # Find connected components
        n_components, labels = connected_components(
            csgraph=adjacency_sym, 
            directed=False, 
            return_labels=True
        )
        
        # Component sizes
        component_sizes = np.bincount(labels)
        largest_component_size = component_sizes.max()
        
        return {
            'method': 'scipy.csgraph',
            'n_components': n_components,
            'largest_component': largest_component_size,
            'component_sizes': sorted(component_sizes, reverse=True),
            'is_connected': n_components == 1,
            'labels': labels
        }
        
    except Exception as e:
        return {'method': 'scipy.csgraph', 'error': str(e)}

def analyze_connectivity_networkx(adjacency):
    """Analyze connectivity using NetworkX"""
    try:
        # Convert to NetworkX graph
        G = nx.from_scipy_sparse_matrix(adjacency)
        
        # Check connectivity
        is_connected = nx.is_connected(G)
        n_components = nx.number_connected_components(G)
        
        # Get component sizes
        components = list(nx.connected_components(G))
        component_sizes = [len(comp) for comp in components]
        component_sizes.sort(reverse=True)
        
        return {
            'method': 'networkx',
            'n_components': n_components,
            'largest_component': component_sizes[0] if component_sizes else 0,
            'component_sizes': component_sizes[:10],  # Top 10 components
            'is_connected': is_connected,
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges()
        }
        
    except Exception as e:
        return {'method': 'networkx', 'error': str(e)}

def analyze_connectivity_manual(adjacency):
    """Manual connectivity analysis using BFS/DFS"""
    try:
        n_nodes = adjacency.shape[0]
        visited = np.zeros(n_nodes, dtype=bool)
        components = []
        
        for start_node in range(n_nodes):
            if visited[start_node]:
                continue
            
            # BFS from this node
            component = []
            queue = [start_node]
            visited[start_node] = True
            
            while queue:
                node = queue.pop(0)
                component.append(node)
                
                # Find neighbors
                neighbors = adjacency[node].nonzero()[1]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            components.append(component)
        
        component_sizes = [len(comp) for comp in components]
        component_sizes.sort(reverse=True)
        
        return {
            'method': 'manual_bfs',
            'n_components': len(components),
            'largest_component': component_sizes[0] if component_sizes else 0,
            'component_sizes': component_sizes[:10],
            'is_connected': len(components) == 1
        }
        
    except Exception as e:
        return {'method': 'manual_bfs', 'error': str(e)}

def comprehensive_connectivity_analysis(filepath, method_name):
    """Run all connectivity analysis methods"""
    print(f"\n🔍 COMPREHENSIVE CONNECTIVITY ANALYSIS: {method_name}")
    print("=" * 60)
    
    # Load matrix
    adjacency, load_info = load_mtx_as_adjacency(filepath)
    print(f"Load status: {load_info}")
    
    if adjacency is None:
        print("❌ Could not load matrix")
        return None
    
    # Run all analysis methods
    results = {}
    
    print(f"\n📊 Method 1: scipy.sparse.csgraph")
    results['scipy'] = analyze_connectivity_scipy(adjacency)
    if 'error' not in results['scipy']:
        print(f"  Connected: {results['scipy']['is_connected']}")
        print(f"  Components: {results['scipy']['n_components']}")
        print(f"  Largest component: {results['scipy']['largest_component']} nodes")
        print(f"  Component sizes: {results['scipy']['component_sizes'][:5]}")
    else:
        print(f"  Error: {results['scipy']['error']}")
    
    print(f"\n📊 Method 2: NetworkX")
    results['networkx'] = analyze_connectivity_networkx(adjacency)
    if 'error' not in results['networkx']:
        print(f"  Connected: {results['networkx']['is_connected']}")
        print(f"  Components: {results['networkx']['n_components']}")
        print(f"  Largest component: {results['networkx']['largest_component']} nodes")
        print(f"  Component sizes: {results['networkx']['component_sizes']}")
        print(f"  Total nodes: {results['networkx']['total_nodes']}")
        print(f"  Total edges: {results['networkx']['total_edges']}")
    else:
        print(f"  Error: {results['networkx']['error']}")
    
    print(f"\n📊 Method 3: Manual BFS")
    results['manual'] = analyze_connectivity_manual(adjacency)
    if 'error' not in results['manual']:
        print(f"  Connected: {results['manual']['is_connected']}")
        print(f"  Components: {results['manual']['n_components']}")
        print(f"  Largest component: {results['manual']['largest_component']} nodes")
        print(f"  Component sizes: {results['manual']['component_sizes']}")
    else:
        print(f"  Error: {results['manual']['error']}")
    
    # Consistency check
    print(f"\n🔍 CONSISTENCY CHECK:")
    methods_agree = True
    
    connected_results = []
    component_counts = []
    
    for method, result in results.items():
        if 'error' not in result:
            connected_results.append(result['is_connected'])
            component_counts.append(result['n_components'])
    
    if len(set(connected_results)) > 1:
        print(f"  ⚠️  Methods disagree on connectivity!")
        methods_agree = False
    else:
        print(f"  ✅ All methods agree: Connected = {connected_results[0]}")
    
    if len(set(component_counts)) > 1:
        print(f"  ⚠️  Methods disagree on component count!")
        print(f"     Counts: {component_counts}")
        methods_agree = False
    else:
        print(f"  ✅ All methods agree: {component_counts[0]} components")
    
    return {
        'method_name': method_name,
        'filepath': filepath,
        'adjacency_shape': adjacency.shape,
        'adjacency_nnz': adjacency.nnz,
        'results': results,
        'consensus': {
            'methods_agree': methods_agree,
            'is_connected': connected_results[0] if connected_results else None,
            'n_components': component_counts[0] if component_counts else None
        }
    }

def main():
    """Main connectivity verification"""
    print("🔬 MULTI-METHOD CONNECTIVITY VERIFICATION")
    print("=" * 80)
    print("Testing graph connectivity using multiple independent methods:")
    print("1. scipy.sparse.csgraph.connected_components")
    print("2. NetworkX connectivity analysis")
    print("3. Manual BFS traversal")
    print()
    
    # Test files to analyze
    test_files = [
        ('fixed_spectral_results/graphs/lamg_reduce_2.mtx', 'LAMG reduce_2'),
        ('fixed_spectral_results/graphs/lamg_reduce_3.mtx', 'LAMG reduce_3'),
        ('fixed_spectral_results/graphs/lamg_reduce_6.mtx', 'LAMG reduce_6'),
        ('cmg_extracted_graphs/cmg_level_2.mtx', 'CMG++ level_2'),
        ('cmg_extracted_graphs/cmg_level_3.mtx', 'CMG++ level_3'),
    ]
    
    all_results = {}
    
    for filepath, method_name in test_files:
        if Path(filepath).exists():
            result = comprehensive_connectivity_analysis(filepath, method_name)
            if result:
                all_results[method_name] = result
        else:
            print(f"\n⚠️  File not found: {filepath}")
    
    # Summary comparison
    print(f"\n🎯 SUMMARY COMPARISON")
    print("=" * 80)
    print(f"{'Method':<20} {'Connected?':<12} {'Components':<12} {'Largest Comp':<15} {'Consensus':<10}")
    print("-" * 80)
    
    for method_name, result in all_results.items():
        if result['consensus']['methods_agree']:
            connected = "✅ Yes" if result['consensus']['is_connected'] else "❌ No"
            components = result['consensus']['n_components']
            largest = result['results']['networkx'].get('largest_component', 'N/A')
            consensus = "✅ Yes"
        else:
            connected = "⚠️  Disagree"
            components = "Varies"
            largest = "Varies"
            consensus = "❌ No"
        
        print(f"{method_name:<20} {connected:<12} {components:<12} {largest:<15} {consensus:<10}")
    
    # Key findings
    print(f"\n🔍 KEY FINDINGS:")
    print("=" * 80)
    
    connected_methods = []
    disconnected_methods = []
    
    for method_name, result in all_results.items():
        if result['consensus']['methods_agree']:
            if result['consensus']['is_connected']:
                connected_methods.append(method_name)
            else:
                disconnected_methods.append(method_name)
    
    if connected_methods:
        print(f"✅ CONNECTED graphs: {', '.join(connected_methods)}")
    if disconnected_methods:
        print(f"❌ DISCONNECTED graphs: {', '.join(disconnected_methods)}")
    
    print(f"\n🎯 VERDICT:")
    if 'LAMG reduce_3' in connected_methods:
        print("✅ LAMG reduce_3 IS connected - explains high accuracy!")
    elif 'LAMG reduce_3' in disconnected_methods:
        print("❌ LAMG reduce_3 is NOT connected - need to investigate further")
    else:
        print("⚠️  Could not analyze LAMG reduce_3")
    
    if 'CMG++ level_2' in disconnected_methods:
        print("✅ CMG++ level_2 IS disconnected - explains lower accuracy!")
    elif 'CMG++ level_2' in connected_methods:
        print("❌ CMG++ level_2 is connected - contradicts our hypothesis")
    else:
        print("⚠️  Could not analyze CMG++ level_2")

if __name__ == "__main__":
    main()
