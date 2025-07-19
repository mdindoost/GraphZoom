#!/usr/bin/env python3
"""
Connected Components Analysis and Validation
Validate eigenvalue-based connectivity analysis using NetworkX graph analysis
Test hypothesis: Number of connected components across multilevel hierarchy
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def load_mtx_as_adjacency(filepath):
    """Load MTX file and convert Laplacian to adjacency matrix"""
    try:
        print(f"🔍 Loading: {filepath}")
        
        if not Path(filepath).exists():
            print(f"❌ File not found: {filepath}")
            return None
        
        # Manual parsing for GraphZoom MTX format
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
            print(f"❌ No header found in {filepath}")
            return None
        
        # Parse header
        parts = header_line.split()
        if len(parts) < 3:
            print(f"❌ Invalid header: {header_line}")
            return None
        
        rows, cols, nnz = int(parts[0]), int(parts[1]), int(parts[2])
        print(f"   Matrix: {rows}x{cols}, {nnz} entries")
        
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
        
        # Create sparse matrix (this is a Laplacian)
        laplacian = sp.coo_matrix((values, (row_indices, col_indices)), 
                                 shape=(rows, cols)).tocsr()
        
        print(f"   Loaded Laplacian: {laplacian.shape}, {laplacian.nnz} non-zeros")
        print(f"   Value range: {laplacian.data.min():.6f} to {laplacian.data.max():.6f}")
        
        # Convert Laplacian to Adjacency: A = -L + diag(L)
        # For Laplacian L = D - A, so A = D - L = diag(L) - L
        adjacency = -laplacian.copy()
        adjacency.setdiag(0)  # Remove diagonal (no self-loops)
        
        # Ensure non-negative values
        adjacency.data = np.abs(adjacency.data)
        
        # Make symmetric
        adjacency = adjacency.maximum(adjacency.T)
        
        print(f"   Converted to adjacency: {adjacency.nnz} edges")
        
        return adjacency
        
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_connected_components_networkx(adjacency_matrix):
    """Analyze connected components using NetworkX"""
    try:
        # Convert to NetworkX graph (handle version compatibility)
        try:
            G = nx.from_scipy_sparse_array(adjacency_matrix)
        except AttributeError:
            # Fallback for older NetworkX versions
            G = nx.from_scipy_sparse_matrix(adjacency_matrix)
        
        # Find connected components
        components = list(nx.connected_components(G))
        num_components = len(components)
        
        # Component sizes
        component_sizes = [len(comp) for comp in components]
        component_sizes.sort(reverse=True)  # Largest first
        
        # Graph statistics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        # Largest component statistics
        if components:
            largest_component = max(components, key=len)
            largest_size = len(largest_component)
            largest_fraction = largest_size / num_nodes if num_nodes > 0 else 0
            
            # Algebraic connectivity (Fiedler eigenvalue) for connected graphs
            if num_components == 1:
                try:
                    fiedler_value = nx.algebraic_connectivity(G, method='lobpcg')
                except:
                    fiedler_value = None
            else:
                fiedler_value = 0.0  # Disconnected graph has zero algebraic connectivity
        else:
            largest_size = 0
            largest_fraction = 0
            fiedler_value = None
        
        return {
            'num_components': num_components,
            'component_sizes': component_sizes,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'largest_component_size': largest_size,
            'largest_component_fraction': largest_fraction,
            'fiedler_value': fiedler_value,
            'graph': G,
            'components': components
        }
        
    except Exception as e:
        print(f"❌ NetworkX analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def count_zero_eigenvalues(eigenvalues, tolerance=1e-10):
    """Count near-zero eigenvalues with appropriate tolerance"""
    if not eigenvalues:
        return 0
    
    zero_count = sum(1 for val in eigenvalues if abs(val) < tolerance)
    return zero_count

def load_eigenvalue_results():
    """Load previously computed eigenvalue results"""
    eigenvalue_files = {
        'LAMG reduce_2': 'fixed_spectral_results/spectral_results_lamg_reduce_2.txt',
        'LAMG reduce_3': 'fixed_spectral_results/spectral_results_lamg_reduce_3.txt', 
        'LAMG reduce_6': 'fixed_spectral_results/spectral_results_lamg_reduce_6.txt',
        'CMG++ level_1': 'complete_spectral_results/spectral_results_cmg_level_1.txt',
        'CMG++ level_2': 'complete_spectral_results/spectral_results_cmg_level_2.txt',
        'CMG++ level_3': 'complete_spectral_results/spectral_results_cmg_level_3.txt',
    }
    
    eigenvalue_results = {}
    
    for method, filepath in eigenvalue_files.items():
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('Eigenvalues:'):
                        eigenvalues_str = line.split(':', 1)[1].strip()
                        if 'Analysis failed' in eigenvalues_str or 'MTX file not available' in eigenvalues_str:
                            eigenvalue_results[method] = []
                        else:
                            try:
                                eigenvalue_results[method] = [float(x) for x in eigenvalues_str.split(',')]
                            except:
                                eigenvalue_results[method] = []
                        break
        else:
            print(f"⚠️ Eigenvalue file not found: {filepath}")
            eigenvalue_results[method] = []
    
    return eigenvalue_results

def analyze_multilevel_hierarchy():
    """Analyze component preservation across multilevel hierarchies"""
    print("\n🔬 MULTILEVEL HIERARCHY ANALYSIS")
    print("=" * 80)
    print("Testing hypothesis: Connected components behavior across levels")
    
    # LAMG hierarchy (from Cora original 2708 nodes)
    lamg_hierarchy = [
        ('LAMG reduce_2', 'fixed_spectral_results/graphs/lamg_reduce_2.mtx', 2708, 1169),
        ('LAMG reduce_3', 'fixed_spectral_results/graphs/lamg_reduce_3.mtx', 1169, 519),
        ('LAMG reduce_6', 'fixed_spectral_results/graphs/lamg_reduce_6.mtx', 519, 218),
    ]
    
    # CMG++ hierarchy  
    cmg_hierarchy = [
        ('CMG++ level_1', 'cmg_extracted_graphs/cmg_level_1.mtx', 2708, 927),
        ('CMG++ level_2', 'cmg_extracted_graphs/cmg_level_2.mtx', 927, 392),
        ('CMG++ level_3', 'cmg_extracted_graphs/cmg_level_3.mtx', 392, 236),
    ]
    
    print("\n📊 LAMG HIERARCHY:")
    print("-" * 40)
    lamg_results = []
    for method, filepath, prev_nodes, curr_nodes in lamg_hierarchy:
        adjacency = load_mtx_as_adjacency(filepath)
        if adjacency is not None:
            nx_result = analyze_connected_components_networkx(adjacency)
            if nx_result:
                num_comp = nx_result['num_components']
                largest = nx_result['largest_component_size']
                fiedler = nx_result['fiedler_value']
                lamg_results.append({
                    'method': method,
                    'components': num_comp,
                    'largest': largest,
                    'nodes': curr_nodes,
                    'fiedler': fiedler
                })
                fiedler_str = f", Fiedler={fiedler:.6f}" if fiedler is not None else ""
                print(f"{method}: {num_comp} components, largest={largest}/{curr_nodes} nodes{fiedler_str}")
            else:
                lamg_results.append(None)
        else:
            lamg_results.append(None)
    
    print("\n📊 CMG++ HIERARCHY:")
    print("-" * 40)
    cmg_results = []
    for method, filepath, prev_nodes, curr_nodes in cmg_hierarchy:
        adjacency = load_mtx_as_adjacency(filepath)
        if adjacency is not None:
            nx_result = analyze_connected_components_networkx(adjacency)
            if nx_result:
                num_comp = nx_result['num_components']
                largest = nx_result['largest_component_size']
                fiedler = nx_result['fiedler_value']
                cmg_results.append({
                    'method': method,
                    'components': num_comp,
                    'largest': largest,
                    'nodes': curr_nodes,
                    'fiedler': fiedler
                })
                fiedler_str = f", Fiedler={fiedler:.6f}" if fiedler is not None else ""
                print(f"{method}: {num_comp} components, largest={largest}/{curr_nodes} nodes{fiedler_str}")
            else:
                cmg_results.append(None)
        else:
            cmg_results.append(None)
    
    # Analyze hierarchy patterns
    print("\n🎯 HIERARCHY PATTERNS:")
    print("-" * 40)
    
    # LAMG pattern
    lamg_components = [r['components'] for r in lamg_results if r is not None]
    if len(lamg_components) > 1:
        lamg_increasing = all(lamg_components[i] <= lamg_components[i+1] for i in range(len(lamg_components)-1))
        lamg_constant = len(set(lamg_components)) == 1
        print(f"LAMG components: {lamg_components}")
        print(f"  Pattern: {'Constant' if lamg_constant else 'Increasing' if lamg_increasing else 'Mixed'}")
        
        # Check if LAMG maintains connectivity
        lamg_connected = [r for r in lamg_results if r is not None and r['components'] == 1]
        print(f"  Connected levels: {len(lamg_connected)}/{len([r for r in lamg_results if r is not None])}")
    
    # CMG++ pattern
    cmg_components = [r['components'] for r in cmg_results if r is not None]
    if len(cmg_components) > 1:
        cmg_increasing = all(cmg_components[i] <= cmg_components[i+1] for i in range(len(cmg_components)-1))
        cmg_constant = len(set(cmg_components)) == 1
        print(f"CMG++ components: {cmg_components}")
        print(f"  Pattern: {'Constant' if cmg_constant else 'Increasing' if cmg_increasing else 'Mixed'}")
        
        # Check if CMG++ maintains connectivity
        cmg_connected = [r for r in cmg_results if r is not None and r['components'] == 1]
        print(f"  Connected levels: {len(cmg_connected)}/{len([r for r in cmg_results if r is not None])}")
    
    return lamg_results, cmg_results

def generate_comprehensive_validation_report():
    """Generate comprehensive validation report"""
    print("🔬 CONNECTED COMPONENTS VALIDATION REPORT")
    print("=" * 80)
    
    # File locations
    mtx_files = {
        'LAMG reduce_2': 'fixed_spectral_results/graphs/lamg_reduce_2.mtx',
        'LAMG reduce_3': 'fixed_spectral_results/graphs/lamg_reduce_3.mtx',
        'LAMG reduce_6': 'fixed_spectral_results/graphs/lamg_reduce_6.mtx',
        'CMG++ level_1': 'cmg_extracted_graphs/cmg_level_1.mtx',
        'CMG++ level_2': 'cmg_extracted_graphs/cmg_level_2.mtx',
        'CMG++ level_3': 'cmg_extracted_graphs/cmg_level_3.mtx',
    }
    
    # Load eigenvalue results
    eigenvalue_results = load_eigenvalue_results()
    
    # Analyze each method
    results = []
    
    print("\n📊 DETAILED ANALYSIS FOR EACH METHOD:")
    print("=" * 80)
    
    for method, filepath in mtx_files.items():
        print(f"\n🔍 {method}")
        print("-" * 50)
        
        # NetworkX analysis
        adjacency = load_mtx_as_adjacency(filepath)
        if adjacency is not None:
            nx_result = analyze_connected_components_networkx(adjacency)
            if nx_result:
                print(f"   NetworkX Analysis:")
                print(f"     Nodes: {nx_result['num_nodes']}")
                print(f"     Edges: {nx_result['num_edges']}")
                print(f"     Connected Components: {nx_result['num_components']}")
                print(f"     Component sizes: {nx_result['component_sizes'][:10]}")  # Show first 10
                print(f"     Largest component: {nx_result['largest_component_size']} nodes ({nx_result['largest_component_fraction']:.1%})")
                if nx_result['fiedler_value'] is not None:
                    print(f"     Fiedler eigenvalue: {nx_result['fiedler_value']:.8f}")
        else:
            nx_result = None
            print(f"   ❌ Could not load graph")
        
        # Eigenvalue analysis
        eigenvalues = eigenvalue_results.get(method, [])
        if eigenvalues:
            zero_eigenvals = count_zero_eigenvalues(eigenvalues)
            fiedler_from_spectrum = eigenvalues[1] if len(eigenvalues) > 1 else None
            
            print(f"   Eigenvalue Analysis:")
            print(f"     Total eigenvalues: {len(eigenvalues)}")
            print(f"     Near-zero eigenvalues: {zero_eigenvals}")
            print(f"     First 5 eigenvalues: {eigenvalues[:5]}")
            if fiedler_from_spectrum is not None:
                print(f"     Fiedler (λ₂) from spectrum: {fiedler_from_spectrum:.8f}")
            
            # Validation
            if nx_result:
                components_match = nx_result['num_components'] == zero_eigenvals
                
                # Fiedler value comparison
                fiedler_match = None
                if nx_result['fiedler_value'] is not None and fiedler_from_spectrum is not None:
                    fiedler_diff = abs(nx_result['fiedler_value'] - fiedler_from_spectrum)
                    fiedler_match = fiedler_diff < 1e-4
                
                print(f"   🎯 VALIDATION:")
                print(f"     NetworkX components: {nx_result['num_components']}")
                print(f"     Zero eigenvalues: {zero_eigenvals}")
                print(f"     Components match: {'✅ YES' if components_match else '❌ NO'}")
                
                if fiedler_match is not None:
                    print(f"     Fiedler values match: {'✅ YES' if fiedler_match else '❌ NO'}")
                    if not fiedler_match:
                        print(f"     Fiedler difference: {fiedler_diff:.8f}")
                
                if not components_match:
                    diff = abs(nx_result['num_components'] - zero_eigenvals)
                    print(f"     Component difference: {diff}")
            else:
                components_match = None
                fiedler_match = None
        else:
            zero_eigenvals = None
            components_match = None
            fiedler_match = None
            fiedler_from_spectrum = None
            print(f"   ❌ No eigenvalue data available")
        
        # Store results
        results.append({
            'Method': method,
            'Nodes': nx_result['num_nodes'] if nx_result else 0,
            'Edges': nx_result['num_edges'] if nx_result else 0,
            'NetworkX_Components': nx_result['num_components'] if nx_result else None,
            'Zero_Eigenvalues': zero_eigenvals,
            'Components_Match': components_match,
            'NetworkX_Fiedler': nx_result['fiedler_value'] if nx_result else None,
            'Spectrum_Fiedler': fiedler_from_spectrum,
            'Fiedler_Match': fiedler_match,
            'Largest_Component_Size': nx_result['largest_component_size'] if nx_result else None,
            'Largest_Component_Fraction': nx_result['largest_component_fraction'] if nx_result else None,
            'Component_Sizes': str(nx_result['component_sizes'][:5]) if nx_result else None,
        })
    
    # Create summary table
    print(f"\n📊 VALIDATION SUMMARY TABLE:")
    print("=" * 100)
    
    df = pd.DataFrame(results)
    
    # Custom formatted table
    print(f"{'Method':<18} {'Nodes':<7} {'Edges':<7} {'NX_Comp':<8} {'Zero_λ':<8} {'C_Match':<8} {'NX_Fiedler':<12} {'Sp_Fiedler':<12}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        nx_comp = row['NetworkX_Components']
        zero_eig = row['Zero_Eigenvalues'] 
        c_match = row['Components_Match']
        nx_fiedler = row['NetworkX_Fiedler']
        sp_fiedler = row['Spectrum_Fiedler']
        
        nx_comp_str = str(nx_comp) if nx_comp is not None else "N/A"
        zero_eig_str = str(zero_eig) if zero_eig is not None else "N/A"
        c_match_str = "✅" if c_match else "❌" if c_match is not None else "N/A"
        nx_fiedler_str = f"{nx_fiedler:.6f}" if nx_fiedler is not None else "N/A"
        sp_fiedler_str = f"{sp_fiedler:.6f}" if sp_fiedler is not None else "N/A"
        
        print(f"{row['Method']:<18} {row['Nodes']:<7} {row['Edges']:<7} {nx_comp_str:<8} {zero_eig_str:<8} {c_match_str:<8} {nx_fiedler_str:<12} {sp_fiedler_str:<12}")
    
    # Validation statistics
    valid_matches = [r for r in results if r['Components_Match'] is not None]
    correct_matches = [r for r in valid_matches if r['Components_Match']]
    
    print(f"\n🎯 VALIDATION STATISTICS:")
    print(f"   Total methods analyzed: {len(results)}")
    print(f"   Methods with eigenvalue data: {len(valid_matches)}")
    print(f"   Correct eigenvalue predictions: {len(correct_matches)}")
    print(f"   Validation accuracy: {len(correct_matches)/len(valid_matches)*100:.1f}%" if valid_matches else "N/A")
    
    # Key findings for Koutis
    print(f"\n🚨 KEY FINDINGS FOR KOUTIS:")
    print("-" * 40)
    
    # Find the smoking gun cases
    lamg_reduce_3 = next((r for r in results if 'LAMG reduce_3' in r['Method']), None)
    cmg_level_2 = next((r for r in results if 'CMG++ level_2' in r['Method']), None)
    
    if lamg_reduce_3 and cmg_level_2:
        print(f"🎯 THE SMOKING GUN:")
        print(f"   LAMG reduce_3: {lamg_reduce_3['NetworkX_Components']} component(s)")
        print(f"   CMG++ level_2: {cmg_level_2['NetworkX_Components']} component(s)")
        
        if lamg_reduce_3['NetworkX_Components'] == 1 and cmg_level_2['NetworkX_Components'] > 1:
            print(f"   ✅ HYPOTHESIS CONFIRMED: LAMG preserves connectivity, CMG++ fragments!")
            print(f"   This explains the accuracy difference: 79.5% vs 74.8%")
        else:
            print(f"   ❌ Hypothesis not confirmed by this data")
    
    # Save detailed results
    df.to_csv('connected_components_validation.csv', index=False)
    print(f"\n📁 Detailed results saved to: connected_components_validation.csv")
    
    return results

if __name__ == "__main__":
    print("🔬 STARTING CONNECTED COMPONENTS ANALYSIS")
    print("=" * 80)
    
    # Generate comprehensive validation report
    results = generate_comprehensive_validation_report()
    
    # Analyze multilevel hierarchies
    lamg_results, cmg_results = analyze_multilevel_hierarchy()
    
    print("\n✅ CONNECTED COMPONENTS ANALYSIS COMPLETE!")
    print("=" * 80)
    print("📁 Results saved to: connected_components_validation.csv")
    print("\n🎯 KEY FINDINGS:")
    print("   1. Validation of eigenvalue-based connectivity analysis")
    print("   2. Connected components count for each method")
    print("   3. Multilevel hierarchy component preservation patterns")
    print("   4. Component size distributions")
    print("   5. Fiedler eigenvalue cross-validation")