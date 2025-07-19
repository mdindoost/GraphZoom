#!/usr/bin/env python3
"""
Connectivity vs Reduction Summary - FIXED VERSION
Shows the key relationship: Reduction Level → Connected Components → Eigenvalues
"""

import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import networkx as nx
import pandas as pd
import pickle
from pathlib import Path
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

class ConnectivitySummary:
    def __init__(self, lamg_dir="lamg_matrices", cmg_dir="cmg_matrices"):
        self.lamg_dir = Path(lamg_dir)
        self.cmg_dir = Path(cmg_dir)
    
    def load_lamg_mtx_custom(self, mtx_file):
        """Load LAMG MTX file in custom format."""
        with open(mtx_file, 'r') as f:
            lines = f.readlines()
        
        first_line = lines[0].strip().split()
        n_rows, n_cols = int(first_line[0]), int(first_line[1])
        
        rows, cols, vals = [], [], []
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                row = int(parts[0]) - 1
                col = int(parts[1]) - 1
                val = float(parts[2])
                rows.append(row)
                cols.append(col)
                vals.append(val)
        
        matrix = sp.coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
        return matrix.tocsr()
    
    def compute_eigenvalues_full(self, L, k=15):
        """Compute up to 15 eigenvalues with multiple methods."""
        try:
            n = L.shape[0]
            k = min(k, n-2)
            
            if k <= 0:
                return np.array([])
            
            print(f"        Computing {k} eigenvalues for {n}x{n} matrix...")
            
            # Method 1: Standard sparse
            try:
                eigenvals = eigsh(L, k=k, which='SM', return_eigenvectors=False)
                eigenvals = np.sort(eigenvals)
                print(f"        ✅ Standard method: {len(eigenvals)} eigenvalues")
                return eigenvals
            except Exception as e:
                print(f"        Standard method failed: {e}")
            
            # Method 2: Shift-invert
            try:
                eigenvals = eigsh(L, k=k, sigma=1e-12, which='LM', return_eigenvectors=False)
                eigenvals = np.sort(eigenvals)
                print(f"        ✅ Shift-invert method: {len(eigenvals)} eigenvalues")
                return eigenvals
            except Exception as e:
                print(f"        Shift-invert method failed: {e}")
            
            # Method 3: Dense computation for smaller matrices
            if n < 1000:
                try:
                    print(f"        Trying dense computation...")
                    L_dense = L.toarray()
                    eigenvals = eigh(L_dense, eigvals_only=True)
                    eigenvals = np.sort(eigenvals)[:k]
                    print(f"        ✅ Dense method: {len(eigenvals)} eigenvalues")
                    return eigenvals
                except Exception as e:
                    print(f"        Dense method failed: {e}")
            
            print(f"        ❌ All methods failed")
            return np.array([])
            
        except Exception as e:
            print(f"        ❌ Error: {e}")
            return np.array([])
    
    def analyze_connectivity_detailed(self, adjacency_matrix, eigenvalues=None):
        """Detailed connectivity analysis with NetworkX confirmation."""
        print(f"        Analyzing connectivity...")
        
        # NetworkX connectivity (ground truth)
        try:
            if hasattr(nx, 'from_scipy_sparse_array'):
                G = nx.from_scipy_sparse_array(adjacency_matrix)
            else:
                G = nx.from_scipy_sparse_matrix(adjacency_matrix)
            
            nx_components = nx.number_connected_components(G)
            
            # Get detailed component information
            components_list = list(nx.connected_components(G))
            component_sizes = sorted([len(c) for c in components_list], reverse=True)
            
            print(f"        NetworkX: {nx_components} components, sizes: {component_sizes[:5]}{'...' if len(component_sizes) > 5 else ''}")
            
        except Exception as e:
            print(f"        NetworkX failed: {e}")
            nx_components = -1
            component_sizes = []
        
        # Eigenvalue-based connectivity with 1e-16 threshold
        eigen_components = -1
        zero_eigenvals = []
        
        if eigenvalues is not None and len(eigenvalues) > 0:
            # Count eigenvalues below 1e-16 threshold
            zero_mask = np.abs(eigenvalues) < 1e-16
            eigen_components = np.sum(zero_mask)
            zero_eigenvals = eigenvalues[zero_mask]
            
            print(f"        Eigenvalue method (threshold 1e-16): {eigen_components} zero eigenvalues")
            print(f"        Zero eigenvalues: {[f'{ev:.2e}' for ev in zero_eigenvals[:5]]}")
            
            # Show agreement
            if nx_components > 0:
                if eigen_components == nx_components:
                    print(f"        ✅ Methods AGREE: {nx_components} components")
                else:
                    print(f"        ❌ Methods DISAGREE: NetworkX={nx_components}, Eigenvalues={eigen_components}")
        
        return {
            'networkx_components': nx_components,
            'eigenvalue_components': eigen_components,
            'component_sizes': component_sizes,
            'largest_component': component_sizes[0] if component_sizes else 0,
            'methods_agree': eigen_components == nx_components if eigen_components != -1 and nx_components != -1 else False,
            'zero_eigenvalues': zero_eigenvals.tolist() if len(zero_eigenvals) > 0 else []
        }
    
    def analyze_lamg_summary(self):
        """Analyze LAMG with detailed eigenvalue and connectivity analysis."""
        print("🔍 LAMG ANALYSIS: Detailed Eigenvalues & Connectivity")
        print("="*70)
        
        lamg_summary = []
        
        # Find LAMG files
        mtx_files = sorted(list(self.lamg_dir.glob("lamg_ratio*_Gs.mtx")))
        
        for mtx_file in mtx_files:
            ratio = int(mtx_file.stem.split('_')[1].replace('ratio', ''))
            print(f"\n📊 LAMG Ratio {ratio}:")
            
            try:
                # Load and process matrix
                L_coarse = self.load_lamg_mtx_custom(mtx_file)
                print(f"      Loaded matrix: {L_coarse.shape}")
                
                # Convert to adjacency
                diagonal = L_coarse.diagonal()
                diagonal = np.maximum(diagonal, 0)
                D = sp.diags(diagonal)
                A_coarse = D - L_coarse
                A_coarse.data = np.maximum(A_coarse.data, 0)
                A_coarse = (A_coarse + A_coarse.T) / 2
                A_coarse.eliminate_zeros()
                
                # Rebuild clean Laplacian
                degrees = np.array(A_coarse.sum(axis=1)).flatten()
                L_clean = sp.diags(degrees) - A_coarse
                
                # Compute up to 15 eigenvalues
                eigenvals = self.compute_eigenvalues_full(L_clean, k=15)
                
                # Detailed connectivity analysis
                connectivity = self.analyze_connectivity_detailed(A_coarse, eigenvals)
                
                # Calculate metrics
                nodes = A_coarse.shape[0]
                edges = A_coarse.nnz // 2
                reduction = 2708 / nodes
                
                # Create eigenvalue summary
                eigenval_summary = {}
                if len(eigenvals) > 0:
                    for i, ev in enumerate(eigenvals):
                        eigenval_summary[f'λ_{i+1}'] = f'{ev:.2e}'
                
                # Store summary
                summary = {
                    'Method': 'LAMG',
                    'Config': f'Ratio {ratio}',
                    'Nodes': nodes,
                    'Edges': edges,
                    'Reduction': f'{reduction:.1f}x',
                    'NetworkX_Components': connectivity['networkx_components'],
                    'Eigenvalue_Components': connectivity['eigenvalue_components'],
                    'Methods_Agree': connectivity['methods_agree'],
                    'Largest_Comp': connectivity['largest_component'],
                    'Largest_Comp_Pct': f'{(connectivity["largest_component"]/nodes*100):.1f}%' if nodes > 0 else 'N/A',
                    'Lambda_Critical': 'N/A',
                    'All_Eigenvalues': eigenvals.tolist() if len(eigenvals) > 0 else [],
                    **eigenval_summary
                }
                
                lamg_summary.append(summary)
                
                # Print detailed summary
                print(f"      ✅ {nodes} nodes, {edges} edges ({reduction:.1f}x reduction)")
                print(f"      🔗 Connectivity: {connectivity['networkx_components']} components (NetworkX)")
                print(f"      🔢 Zero eigenvalues: {connectivity['eigenvalue_components']} (threshold 1e-16)")
                print(f"      📊 Agreement: {'✅ YES' if connectivity['methods_agree'] else '❌ NO'}")
                
                if len(eigenvals) > 0:
                    print(f"      📈 First 5 eigenvalues: {[f'{ev:.2e}' for ev in eigenvals[:5]]}")
                    if len(eigenvals) > 5:
                        print(f"      📈 Next 5 eigenvalues: {[f'{ev:.2e}' for ev in eigenvals[5:10]]}")
                    if len(eigenvals) > 10:
                        print(f"      📈 Last eigenvalues: {[f'{ev:.2e}' for ev in eigenvals[10:]]}")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        return lamg_summary
    
    def analyze_cmg_summary(self):
        """Analyze CMG++ with detailed eigenvalue and connectivity analysis."""
        print("\n🔍 CMG++ ANALYSIS: Detailed Eigenvalues & Connectivity")
        print("="*70)
        
        cmg_summary = []
        
        # Find CMG files
        pkl_files = sorted(list(self.cmg_dir.glob("cmg_level*_results.pkl")))
        
        for pkl_file in pkl_files:
            level = int(pkl_file.stem.split('_')[1].replace('level', ''))
            print(f"\n📊 CMG++ Level {level}:")
            
            try:
                # Load CMG results
                with open(pkl_file, 'rb') as f:
                    cmg_data = pickle.load(f)
                
                A_coarse = cmg_data['adjacency']
                L_coarse = cmg_data['laplacian']
                
                print(f"      Loaded matrix: {A_coarse.shape}")
                
                # Compute up to 15 eigenvalues
                eigenvals = self.compute_eigenvalues_full(L_coarse, k=15)
                
                # Detailed connectivity analysis
                connectivity = self.analyze_connectivity_detailed(A_coarse, eigenvals)
                
                # Calculate metrics
                nodes = cmg_data['nodes']
                edges = cmg_data['edges']
                reduction = cmg_data.get('cumulative_reduction', 2708/nodes)
                
                # Create eigenvalue summary
                eigenval_summary = {}
                if len(eigenvals) > 0:
                    for i, ev in enumerate(eigenvals):
                        eigenval_summary[f'λ_{i+1}'] = f'{ev:.2e}'
                
                # Store summary
                summary = {
                    'Method': 'CMG++',
                    'Config': f'Level {level}',
                    'Nodes': nodes,
                    'Edges': edges,
                    'Reduction': f'{reduction:.1f}x',
                    'NetworkX_Components': connectivity['networkx_components'],
                    'Eigenvalue_Components': connectivity['eigenvalue_components'],
                    'Methods_Agree': connectivity['methods_agree'],
                    'Largest_Comp': connectivity['largest_component'],
                    'Largest_Comp_Pct': f'{(connectivity["largest_component"]/nodes*100):.1f}%' if nodes > 0 else 'N/A',
                    'Lambda_Critical': cmg_data['lambda_critical'],
                    'All_Eigenvalues': eigenvals.tolist() if len(eigenvals) > 0 else [],
                    **eigenval_summary
                }
                
                cmg_summary.append(summary)
                
                # Print detailed summary
                print(f"      ✅ {nodes} nodes, {edges} edges ({reduction:.1f}x reduction)")
                print(f"      🔗 Connectivity: {connectivity['networkx_components']} components (NetworkX)")
                print(f"      🔢 Zero eigenvalues: {connectivity['eigenvalue_components']} (threshold 1e-16)")
                print(f"      📊 Agreement: {'✅ YES' if connectivity['methods_agree'] else '❌ NO'}")
                print(f"      🎯 λ_critical: {cmg_data['lambda_critical']:.6f}")
                
                if len(eigenvals) > 0:
                    print(f"      📈 First 5 eigenvalues: {[f'{ev:.2e}' for ev in eigenvals[:5]]}")
                    if len(eigenvals) > 5:
                        print(f"      📈 Next 5 eigenvalues: {[f'{ev:.2e}' for ev in eigenvals[5:10]]}")
                    if len(eigenvals) > 10:
                        print(f"      📈 Last eigenvalues: {[f'{ev:.2e}' for ev in eigenvals[10:]]}")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        return cmg_summary
    
    def create_summary_table(self, lamg_summary, cmg_summary):
        """Create combined summary table - FIXED VERSION."""
        print("\n📊 COMBINED SUMMARY TABLE")
        print("="*70)
        
        all_data = lamg_summary + cmg_summary
        df = pd.DataFrame(all_data)
        
        if not df.empty:
            # Sort by method and reduction level
            df['Reduction_Val'] = df['Reduction'].str.replace('x', '').astype(float)
            df = df.sort_values(['Method', 'Reduction_Val'])
            
            # Create display table with CORRECT column names
            display_cols = ['Method', 'Config', 'Nodes', 'Reduction', 'NetworkX_Components', 'Largest_Comp_Pct', 'Lambda_Critical']
            
            print(df[display_cols].to_string(index=False))
        
        return df
    
    def analyze_trends(self, df):
        """Analyze trends in the data - FIXED VERSION."""
        print("\n🔍 TREND ANALYSIS")
        print("="*70)
        
        if df.empty:
            print("No data to analyze")
            return
        
        # LAMG trends - FIXED: Use NetworkX_Components
        lamg_data = df[df['Method'] == 'LAMG']
        if not lamg_data.empty:
            print("\n🟢 LAMG Trends:")
            print(f"   • Reduction range: {lamg_data['Reduction_Val'].min():.1f}x → {lamg_data['Reduction_Val'].max():.1f}x")
            print(f"   • Component range: {lamg_data['NetworkX_Components'].min()} → {lamg_data['NetworkX_Components'].max()} components")
            print(f"   • Node range: {lamg_data['Nodes'].max()} → {lamg_data['Nodes'].min()} nodes")
            
            # Best connectivity
            best_lamg = lamg_data.loc[lamg_data['NetworkX_Components'].idxmin()]
            print(f"   • Best connectivity: {best_lamg['Config']} ({best_lamg['NetworkX_Components']} components)")
        
        # CMG++ trends - FIXED: Use NetworkX_Components
        cmg_data = df[df['Method'] == 'CMG++']
        if not cmg_data.empty:
            print("\n🔴 CMG++ Trends:")
            print(f"   • Reduction range: {cmg_data['Reduction_Val'].min():.1f}x → {cmg_data['Reduction_Val'].max():.1f}x")
            print(f"   • Component range: {cmg_data['NetworkX_Components'].min()} → {cmg_data['NetworkX_Components'].max()} components")
            print(f"   • Node range: {cmg_data['Nodes'].max()} → {cmg_data['Nodes'].min()} nodes")
            
            # Best connectivity
            best_cmg = cmg_data.loc[cmg_data['NetworkX_Components'].idxmin()]
            print(f"   • Best connectivity: {best_cmg['Config']} ({best_cmg['NetworkX_Components']} components)")
        
        # Comparison
        if not lamg_data.empty and not cmg_data.empty:
            print("\n⚖️  COMPARISON:")
            avg_comp_lamg = lamg_data['NetworkX_Components'].mean()
            avg_comp_cmg = cmg_data['NetworkX_Components'].mean()
            
            if avg_comp_lamg < avg_comp_cmg:
                print(f"   🏆 LAMG has better average connectivity: {avg_comp_lamg:.1f} vs {avg_comp_cmg:.1f} components")
                connectivity_improvement = avg_comp_cmg / avg_comp_lamg
                print(f"   📊 LAMG is {connectivity_improvement:.1f}x better at preserving connectivity!")
            else:
                print(f"   🏆 CMG++ has better average connectivity: {avg_comp_cmg:.1f} vs {avg_comp_lamg:.1f} components")
                connectivity_improvement = avg_comp_lamg / avg_comp_cmg
                print(f"   📊 CMG++ is {connectivity_improvement:.1f}x better at preserving connectivity!")
            
            print(f"   📈 Connectivity difference: {abs(avg_comp_lamg - avg_comp_cmg):.1f} components")
            
            # Additional insights
            print(f"\n💡 KEY INSIGHTS:")
            worst_lamg = lamg_data['NetworkX_Components'].max()
            best_cmg = cmg_data['NetworkX_Components'].min()
            
            if worst_lamg < best_cmg:
                print(f"   🎯 Even LAMG's worst case ({worst_lamg} components) beats CMG++'s best case ({best_cmg} components)!")
            
            # Largest component analysis
            lamg_largest_avg = lamg_data['Largest_Comp'].mean()
            cmg_largest_avg = cmg_data['Largest_Comp'].mean()
            
            print(f"   🏗️  Average largest component: LAMG={lamg_largest_avg:.0f} nodes, CMG++={cmg_largest_avg:.0f} nodes")
    
    def run_summary(self):
        """Run complete connectivity summary analysis."""
        print("🎯 CONNECTIVITY vs REDUCTION ANALYSIS")
        print("="*70)
        print("Original Cora: 2708 nodes")
        print("Analyzing: Reduction Level → Connected Components → Eigenvalues")
        print()
        
        # Analyze both methods
        lamg_summary = self.analyze_lamg_summary()
        cmg_summary = self.analyze_cmg_summary()
        
        # Create summary table
        df = self.create_summary_table(lamg_summary, cmg_summary)
        
        # Analyze trends
        self.analyze_trends(df)
        
        # Save results
        if not df.empty:
            df.to_csv('connectivity_summary.csv', index=False)
            print(f"\n💾 Summary saved to: connectivity_summary.csv")
        
        print("\n" + "="*70)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*70)
        
        # Final summary
        if not df.empty:
            lamg_data = df[df['Method'] == 'LAMG']
            cmg_data = df[df['Method'] == 'CMG++']
            
            if not lamg_data.empty and not cmg_data.empty:
                lamg_avg = lamg_data['NetworkX_Components'].mean()
                cmg_avg = cmg_data['NetworkX_Components'].mean()
                
                print("🏆 FINAL VERDICT:")
                if lamg_avg < cmg_avg:
                    improvement = cmg_avg / lamg_avg
                    print(f"   LAMG WINS: {improvement:.1f}x better connectivity preservation!")
                    print(f"   LAMG: {lamg_avg:.1f} avg components vs CMG++: {cmg_avg:.1f} avg components")
                    print("   🎯 This explains why LAMG achieves higher GNN accuracy!")
                else:
                    improvement = lamg_avg / cmg_avg
                    print(f"   CMG++ WINS: {improvement:.1f}x better connectivity preservation!")
                    print(f"   CMG++: {cmg_avg:.1f} avg components vs LAMG: {lamg_avg:.1f} avg components")
        
        return df

if __name__ == "__main__":
    analyzer = ConnectivitySummary()
    df = analyzer.run_summary()
