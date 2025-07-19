#!/usr/bin/env python3
"""
Matrix Analysis Script
Analyzes all generated matrices from LAMG and CMG++ for spectral properties
"""

import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
from pathlib import Path
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

class MatrixAnalyzer:
    def __init__(self, lamg_dir="lamg_matrices", cmg_dir="cmg_matrices", output_dir="analysis_results"):
        self.lamg_dir = Path(lamg_dir)
        self.cmg_dir = Path(cmg_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def load_lamg_mtx_custom(self, mtx_file):
        """Load LAMG MTX file in custom format."""
        with open(mtx_file, 'r') as f:
            lines = f.readlines()
        
        # First line has dimensions
        first_line = lines[0].strip().split()
        n_rows, n_cols = int(first_line[0]), int(first_line[1])
        
        # Read coordinate data
        rows, cols, vals = [], [], []
        
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                row = int(parts[0]) - 1  # Convert to 0-indexed
                col = int(parts[1]) - 1  # Convert to 0-indexed
                val = float(parts[2])
                rows.append(row)
                cols.append(col)
                vals.append(val)
        
        # Create sparse matrix
        matrix = sp.coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
        return matrix.tocsr()
    
    def compute_eigenvalues_robust(self, L, k=12):
        """Compute eigenvalues with multiple fallback methods."""
        try:
            n = L.shape[0]
            k = min(k, n-2)
            
            if k <= 0:
                return np.array([])
            
            # Method 1: Standard sparse
            try:
                eigenvals = eigsh(L, k=k, which='SM', return_eigenvectors=False)
                return np.sort(eigenvals)
            except:
                pass
            
            # Method 2: Shift-invert
            try:
                eigenvals = eigsh(L, k=k, sigma=1e-10, which='LM', return_eigenvectors=False)
                return np.sort(eigenvals)
            except:
                pass
            
            # Method 3: Dense computation for small matrices
            if n < 1000:
                try:
                    L_dense = L.toarray()
                    eigenvals = eigh(L_dense, eigvals_only=True)
                    return np.sort(eigenvals)[:k]
                except:
                    pass
            
            return np.array([])
            
        except Exception as e:
            print(f"      ❌ Eigenvalue computation failed: {e}")
            return np.array([])
    
    def analyze_connectivity(self, adjacency_matrix, eigenvalues=None):
        """Analyze graph connectivity using multiple methods."""
        
        # NetworkX connectivity (ground truth)
        try:
            if hasattr(nx, 'from_scipy_sparse_array'):
                G = nx.from_scipy_sparse_array(adjacency_matrix)
            else:
                G = nx.from_scipy_sparse_matrix(adjacency_matrix)
            
            nx_components = nx.number_connected_components(G)
            
            # Get component sizes
            components = list(nx.connected_components(G))
            component_sizes = sorted([len(c) for c in components], reverse=True)
            
        except Exception as e:
            print(f"      NetworkX connectivity failed: {e}")
            nx_components = -1
            component_sizes = []
        
        # Eigenvalue-based connectivity
        eigen_components = -1
        if eigenvalues is not None and len(eigenvalues) > 0:
            # Try different thresholds
            for thresh in [1e-16, 1e-14, 1e-12, 1e-10]:
                zero_count = np.sum(np.abs(eigenvalues) < thresh)
                if nx_components > 0 and zero_count == nx_components:
                    eigen_components = zero_count
                    break
            
            if eigen_components == -1:
                eigen_components = np.sum(np.abs(eigenvalues) < 1e-12)
        
        return {
            'networkx_components': nx_components,
            'eigenvalue_components': eigen_components,
            'component_sizes': component_sizes,
            'largest_component': component_sizes[0] if component_sizes else 0,
            'methods_agree': eigen_components == nx_components if eigen_components != -1 else False
        }
    
    def analyze_lamg_matrices(self):
        """Analyze all LAMG matrices."""
        print("🔍 Analyzing LAMG matrices...")
        
        lamg_results = {}
        
        # Look for LAMG MTX files
        mtx_files = list(self.lamg_dir.glob("lamg_ratio*_Gs.mtx"))
        
        if not mtx_files:
            print("   ❌ No LAMG MTX files found")
            return lamg_results
        
        for mtx_file in sorted(mtx_files):
            # Extract ratio from filename
            ratio = mtx_file.stem.split('_')[1].replace('ratio', '')
            
            print(f"\n   📊 Analyzing LAMG Ratio {ratio}:")
            
            try:
                # Load LAMG matrix
                L_coarse = self.load_lamg_mtx_custom(mtx_file)
                print(f"      Loaded matrix: {L_coarse.shape}")
                
                # Convert Laplacian to adjacency
                diagonal = L_coarse.diagonal()
                
                if np.all(diagonal >= -1e-10):
                    # Proper Laplacian: A = D - L
                    diagonal = np.maximum(diagonal, 0)
                    D = sp.diags(diagonal)
                    A_coarse = D - L_coarse
                    A_coarse.data = np.maximum(A_coarse.data, 0)
                    A_coarse = (A_coarse + A_coarse.T) / 2
                    A_coarse.eliminate_zeros()
                    
                    # Rebuild Laplacian
                    degrees_clean = np.array(A_coarse.sum(axis=1)).flatten()
                    L_clean = sp.diags(degrees_clean) - A_coarse
                else:
                    print(f"      ⚠️  Unusual matrix format")
                    A_coarse = L_coarse.copy()
                    A_coarse.data = np.abs(A_coarse.data)
                    degrees = np.array(A_coarse.sum(axis=1)).flatten()
                    L_clean = sp.diags(degrees) - A_coarse
                
                # Compute eigenvalues
                eigenvals = self.compute_eigenvalues_robust(L_clean, k=12)
                
                # Analyze connectivity
                connectivity = self.analyze_connectivity(A_coarse, eigenvals)
                
                # Store results
                lamg_results[f'ratio_{ratio}'] = {
                    'method': 'LAMG',
                    'configuration': f'Ratio {ratio}',
                    'reduce_ratio': int(ratio),
                    'nodes': A_coarse.shape[0],
                    'edges': A_coarse.nnz // 2,
                    'reduction_ratio': 2708 / A_coarse.shape[0],
                    'eigenvalues': eigenvals.tolist() if len(eigenvals) > 0 else [],
                    'connectivity': connectivity,
                    'matrix_file': str(mtx_file)
                }
                
                print(f"      ✅ {A_coarse.shape[0]} nodes, {A_coarse.nnz//2} edges")
                print(f"      📊 {connectivity['networkx_components']} components")
                print(f"      🔍 Eigenvalues computed: {len(eigenvals)}")
                
            except Exception as e:
                print(f"      ❌ Error analyzing {mtx_file}: {e}")
        
        return lamg_results
    
    def analyze_cmg_matrices(self):
        """Analyze all CMG++ matrices."""
        print("\n🔍 Analyzing CMG++ matrices...")
        
        cmg_results = {}
        
        # Look for CMG pickle files
        pkl_files = list(self.cmg_dir.glob("cmg_level*_results.pkl"))
        
        if not pkl_files:
            print("   ❌ No CMG++ pickle files found")
            return cmg_results
        
        for pkl_file in sorted(pkl_files):
            # Extract level from filename
            level = pkl_file.stem.split('_')[1].replace('level', '')
            
            print(f"\n   📊 Analyzing CMG++ Level {level}:")
            
            try:
                # Load CMG results
                with open(pkl_file, 'rb') as f:
                    cmg_data = pickle.load(f)
                
                A_coarse = cmg_data['adjacency']
                L_coarse = cmg_data['laplacian']
                
                print(f"      Loaded matrix: {A_coarse.shape}")
                
                # Compute eigenvalues
                eigenvals = self.compute_eigenvalues_robust(L_coarse, k=12)
                
                # Analyze connectivity
                connectivity = self.analyze_connectivity(A_coarse, eigenvals)
                
                # Store results
                cmg_results[f'level_{level}'] = {
                    'method': 'CMG++',
                    'configuration': f"Level {level} (k={cmg_data['parameters']['k']}, d={cmg_data['parameters']['d']})",
                    'level': int(level),
                    'nodes': cmg_data['nodes'],
                    'edges': cmg_data['edges'],
                    'reduction_ratio': cmg_data.get('cumulative_reduction', cmg_data.get('reduction_ratio', 2708/cmg_data['nodes'])),
                    'lambda_critical': cmg_data['lambda_critical'],
                    'eigenvalues': eigenvals.tolist() if len(eigenvals) > 0 else [],
                    'connectivity': connectivity,
                    'parameters': cmg_data['parameters'],
                    'matrix_file': str(pkl_file)
                }
                
                print(f"      ✅ {cmg_data['nodes']} nodes, {cmg_data['edges']} edges")
                print(f"      📊 {connectivity['networkx_components']} components")
                print(f"      🔍 Eigenvalues computed: {len(eigenvals)}")
                print(f"      λ_critical: {cmg_data['lambda_critical']:.6f}")
                
            except Exception as e:
                print(f"      ❌ Error analyzing {pkl_file}: {e}")
        
        return cmg_results
    
    def create_comparison_table(self, lamg_results, cmg_results):
        """Create comprehensive comparison table."""
        print("\n📊 Creating comparison table...")
        
        all_results = []
        
        # Add LAMG results
        for key, result in lamg_results.items():
            conn = result['connectivity']
            all_results.append({
                'Method': result['method'],
                'Configuration': result['configuration'],
                'Nodes': result['nodes'],
                'Edges': result['edges'],
                'Reduction_Ratio': f"{result['reduction_ratio']:.2f}x",
                'Components_NetworkX': conn['networkx_components'],
                'Components_Eigenvalue': conn['eigenvalue_components'],
                'Methods_Agree': conn['methods_agree'],
                'Largest_Component': conn['largest_component'],
                'Eigenvalues_Count': len(result['eigenvalues']),
                'First_3_Eigenvalues': result['eigenvalues'][:3] if result['eigenvalues'] else [],
                'Lambda_Critical': 'N/A'
            })
        
        # Add CMG++ results
        for key, result in cmg_results.items():
            conn = result['connectivity']
            all_results.append({
                'Method': result['method'],
                'Configuration': result['configuration'],
                'Nodes': result['nodes'],
                'Edges': result['edges'],
                'Reduction_Ratio': f"{result['reduction_ratio']:.2f}x",
                'Components_NetworkX': conn['networkx_components'],
                'Components_Eigenvalue': conn['eigenvalue_components'],
                'Methods_Agree': conn['methods_agree'],
                'Largest_Component': conn['largest_component'],
                'Eigenvalues_Count': len(result['eigenvalues']),
                'First_3_Eigenvalues': result['eigenvalues'][:3] if result['eigenvalues'] else [],
                'Lambda_Critical': result.get('lambda_critical', 'N/A')
            })
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Save to CSV
        csv_file = self.output_dir / "complete_comparison.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"✅ Comparison table saved: {csv_file}")
        
        return df
    
    def create_visualizations(self, df):
        """Create comprehensive visualizations."""
        print("\n📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('LAMG vs CMG++ Comprehensive Analysis', fontsize=16, fontweight='bold')
        
        # Separate data by method
        lamg_data = df[df['Method'] == 'LAMG']
        cmg_data = df[df['Method'] == 'CMG++']
        
        # Plot 1: Nodes vs Components
        ax1 = axes[0, 0]
        if not lamg_data.empty:
            ax1.scatter(lamg_data['Nodes'], lamg_data['Components_NetworkX'], 
                       c='green', label='LAMG', s=100, alpha=0.7, marker='s')
        if not cmg_data.empty:
            ax1.scatter(cmg_data['Nodes'], cmg_data['Components_NetworkX'], 
                       c='red', label='CMG++', s=100, alpha=0.7, marker='o')
        
        ax1.set_xlabel('Number of Nodes')
        ax1.set_ylabel('Connected Components')
        ax1.set_title('Graph Size vs Connectivity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Reduction Ratio vs Components
        ax2 = axes[0, 1]
        if not lamg_data.empty:
            ratios = [float(r.replace('x', '')) for r in lamg_data['Reduction_Ratio']]
            ax2.scatter(ratios, lamg_data['Components_NetworkX'], 
                       c='green', label='LAMG', s=100, alpha=0.7, marker='s')
        if not cmg_data.empty:
            ratios = [float(r.replace('x', '')) for r in cmg_data['Reduction_Ratio']]
            ax2.scatter(ratios, cmg_data['Components_NetworkX'], 
                       c='red', label='CMG++', s=100, alpha=0.7, marker='o')
        
        ax2.set_xlabel('Reduction Ratio')
        ax2.set_ylabel('Connected Components')
        ax2.set_title('Reduction vs Connectivity Trade-off')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Method Agreement Check
        ax3 = axes[0, 2]
        agreement_counts = df['Methods_Agree'].value_counts()
        colors = ['green' if x else 'red' for x in agreement_counts.index]
        bars = ax3.bar(['Agree', 'Disagree'], agreement_counts.values, color=colors, alpha=0.7)
        ax3.set_title('Eigenvalue vs NetworkX Agreement')
        ax3.set_ylabel('Count')
        
        for bar, count in zip(bars, agreement_counts.values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Component Size Distribution
        ax4 = axes[1, 0]
        methods = df['Method'].unique()
        components_data = [df[df['Method'] == method]['Components_NetworkX'].tolist() for method in methods]
        bp = ax4.boxplot(components_data, labels=methods, patch_artist=True)
        
        colors = ['green' if method == 'LAMG' else 'red' for method in methods]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('Connected Components')
        ax4.set_title('Components Distribution by Method')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Largest Component Size
        ax5 = axes[1, 1]
        if not lamg_data.empty:
            ax5.scatter(lamg_data['Nodes'], lamg_data['Largest_Component'], 
                       c='green', label='LAMG', s=100, alpha=0.7, marker='s')
        if not cmg_data.empty:
            ax5.scatter(cmg_data['Nodes'], cmg_data['Largest_Component'], 
                       c='red', label='CMG++', s=100, alpha=0.7, marker='o')
        
        ax5.set_xlabel('Total Nodes')
        ax5.set_ylabel('Largest Component Size')
        ax5.set_title('Largest Component vs Graph Size')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Summary Table
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = "📊 ANALYSIS SUMMARY\n\n"
        
        if not lamg_data.empty:
            avg_components_lamg = lamg_data['Components_NetworkX'].mean()
            avg_nodes_lamg = lamg_data['Nodes'].mean()
            summary_text += f"🟢 LAMG Results:\n"
            summary_text += f"   • Configs: {len(lamg_data)}\n"
            summary_text += f"   • Avg nodes: {avg_nodes_lamg:.0f}\n"
            summary_text += f"   • Avg components: {avg_components_lamg:.1f}\n\n"
        
        if not cmg_data.empty:
            avg_components_cmg = cmg_data['Components_NetworkX'].mean()
            avg_nodes_cmg = cmg_data['Nodes'].mean()
            summary_text += f"🔴 CMG++ Results:\n"
            summary_text += f"   • Configs: {len(cmg_data)}\n"
            summary_text += f"   • Avg nodes: {avg_nodes_cmg:.0f}\n"
            summary_text += f"   • Avg components: {avg_components_cmg:.1f}\n\n"
        
        total_agree = df['Methods_Agree'].sum()
        total_cases = len(df)
        summary_text += f"✅ Verification:\n"
        summary_text += f"   • Agreement: {total_agree}/{total_cases}\n"
        summary_text += f"   • Rate: {100*total_agree/total_cases:.1f}%\n\n"
        
        summary_text += "💡 Key Finding:\n"
        if not lamg_data.empty and not cmg_data.empty:
            if avg_components_lamg < avg_components_cmg:
                summary_text += "   LAMG maintains better\n   connectivity than CMG++"
            else:
                summary_text += "   CMG++ maintains better\n   connectivity than LAMG"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        # Save plots
        plot_file = self.output_dir / "comprehensive_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizations saved: {plot_file}")
        
        return fig
    
    def generate_report(self, df, lamg_results, cmg_results):
        """Generate comprehensive analysis report."""
        print("\n📄 Generating analysis report...")
        
        report = []
        report.append("# Comprehensive LAMG vs CMG++ Analysis Report")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        report.append("## Executive Summary")
        report.append("")
        
        lamg_data = df[df['Method'] == 'LAMG']
        cmg_data = df[df['Method'] == 'CMG++']
        
        if not lamg_data.empty and not cmg_data.empty:
            avg_comp_lamg = lamg_data['Components_NetworkX'].mean()
            avg_comp_cmg = cmg_data['Components_NetworkX'].mean()
            
            if avg_comp_lamg < avg_comp_cmg:
                winner = "LAMG"
                report.append(f"**Key Finding**: {winner} maintains better connectivity ({avg_comp_lamg:.1f} vs {avg_comp_cmg:.1f} avg components)")
            else:
                winner = "CMG++"
                report.append(f"**Key Finding**: {winner} maintains better connectivity ({avg_comp_cmg:.1f} vs {avg_comp_lamg:.1f} avg components)")
        
        report.append("")
        
        # Detailed Results
        report.append("## Detailed Results")
        report.append("")
        report.append("| Method | Config | Nodes | Edges | Reduction | Components | Largest Component |")
        report.append("|--------|--------|-------|-------|-----------|------------|-------------------|")
        
        for _, row in df.iterrows():
            report.append(f"| {row['Method']} | {row['Configuration']} | {row['Nodes']} | {row['Edges']} | {row['Reduction_Ratio']} | {row['Components_NetworkX']} | {row['Largest_Component']} |")
        
        report.append("")
        
        # Connectivity Analysis
        report.append("## Connectivity Analysis")
        report.append("")
        
        if not lamg_data.empty:
            report.append("### LAMG Results")
            for _, row in lamg_data.iterrows():
                comp_ratio = row['Largest_Component'] / row['Nodes'] * 100
                report.append(f"- **{row['Configuration']}**: {row['Components_NetworkX']} components, largest = {comp_ratio:.1f}% of nodes")
            report.append("")
        
        if not cmg_data.empty:
            report.append("### CMG++ Results")
            for _, row in cmg_data.iterrows():
                comp_ratio = row['Largest_Component'] / row['Nodes'] * 100
                report.append(f"- **{row['Configuration']}**: {row['Components_NetworkX']} components, largest = {comp_ratio:.1f}% of nodes")
            report.append("")
        
        # Implications
        report.append("## Implications for Graph Neural Networks")
        report.append("")
        report.append("- **Connected graphs** enable global information flow")
        report.append("- **Disconnected components** limit message passing scope")
        report.append("- **Fewer components** generally lead to better GNN performance")
        report.append("- **Large dominant components** preserve most graph structure")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        report_file = self.output_dir / "analysis_report.md"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Report saved: {report_file}")
        
        return report_text
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("🚀 COMPREHENSIVE MATRIX ANALYSIS")
        print("=" * 60)
        
        # Analyze LAMG matrices
        lamg_results = self.analyze_lamg_matrices()
        
        # Analyze CMG++ matrices
        cmg_results = self.analyze_cmg_matrices()
        
        if not lamg_results and not cmg_results:
            print("\n❌ No matrices found to analyze!")
            print("Make sure to run:")
            print("  1. ./run_lamg_ratios.sh")
            print("  2. ./run_cmg_levels.sh")
            return
        
        # Create comparison table
        df = self.create_comparison_table(lamg_results, cmg_results)
        
        # Create visualizations
        self.create_visualizations(df)
        
        # Generate report
        self.generate_report(df, lamg_results, cmg_results)
        
        # Save detailed results
        detailed_results = {
            'lamg_results': lamg_results,
            'cmg_results': cmg_results,
            'analysis_timestamp': str(np.datetime64('now'))
        }
        
        with open(self.output_dir / "detailed_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"\n✅ Analysis complete!")
        print(f"📁 Results saved to: {self.output_dir}/")
        print("📊 Files generated:")
        print("  • complete_comparison.csv")
        print("  • comprehensive_analysis.png")
        print("  • analysis_report.md")
        print("  • detailed_results.json")
        
        return df, lamg_results, cmg_results

if __name__ == "__main__":
    analyzer = MatrixAnalyzer()
    df, lamg_results, cmg_results = analyzer.run_complete_analysis()
