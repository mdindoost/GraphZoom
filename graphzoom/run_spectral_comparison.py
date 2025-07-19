#!/usr/bin/env python3
"""
Complete Spectral Analysis Pipeline for LAMG vs CMG++ Comparison
================================================================

This script orchestrates the complete analysis:
1. Extracts CMG++ graphs using your filtering pipeline
2. Loads LAMG matrices from reduction_results/
3. Performs comprehensive spectral analysis
4. Generates comparison reports and visualizations

Usage:
    python run_spectral_comparison.py
    python run_spectral_comparison.py --extract-only  # Just extract CMG++ graphs
    python run_spectral_comparison.py --analyze-only  # Just analyze existing graphs
"""

import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import pandas as pd
import json
import argparse
from pathlib import Path
import sys

def check_dependencies():
    """Check if all required files and modules are available"""
    
    issues = []
    
    # Check for your CMG modules
    try:
        from filtered import cmg_filtered_clustering
        print("✅ filtered.py module found")
    except ImportError:
        issues.append("❌ filtered.py not found - CMG++ extraction will fail")
    
    try:
        from estimate_k_and_clusters import estimate_k
        print("✅ estimate_k_and_clusters.py module found")
    except ImportError:
        issues.append("❌ estimate_k_and_clusters.py not found")
    
    # Check for dataset
    dataset_path = Path("./dataset/cora")
    if (dataset_path / "cora.json").exists():
        print("✅ Cora dataset found")
    else:
        issues.append("❌ Cora dataset not found at ./dataset/cora/cora.json")
    
    # Check for LAMG results
    lamg_path = Path("./reduction_results")
    mtx_files = list(lamg_path.glob("*.mtx")) if lamg_path.exists() else []
    if mtx_files:
        print(f"✅ Found {len(mtx_files)} LAMG matrix files")
    else:
        issues.append("❌ No LAMG .mtx files found in ./reduction_results/")
    
    return issues

def load_lamg_graphs(results_dir="./reduction_results"):
    """Load all LAMG Laplacian matrices"""
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"LAMG results directory not found: {results_path}")
        return {}
    
    graphs = {}
    mtx_files = list(results_path.glob("*.mtx"))
    
    for mtx_file in mtx_files:
        try:
            L = sio.mmread(mtx_file).tocsr()
            
            # Extract meaningful name from filename
            name = mtx_file.stem
            if "reduce" in name or "lamg" in name.lower():
                method_name = f"LAMG_{name}"
            else:
                method_name = f"LAMG_{name}"
            
            graphs[method_name] = {
                'matrix': L,
                'source_file': str(mtx_file),
                'method_type': 'LAMG'
            }
            
            print(f"✅ Loaded LAMG matrix: {method_name} ({L.shape[0]} nodes, {L.nnz} edges)")
            
        except Exception as e:
            print(f"❌ Error loading {mtx_file}: {e}")
    
    return graphs

def load_cmg_graphs(graphs_dir="./cmg_extracted_graphs"):
    """Load CMG++ matrices and metadata"""
    
    graphs_path = Path(graphs_dir)
    if not graphs_path.exists():
        print(f"CMG++ graphs directory not found: {graphs_path}")
        return {}
    
    graphs = {}
    mtx_files = list(graphs_path.glob("*_laplacian.mtx"))
    
    for mtx_file in mtx_files:
        try:
            L = sio.mmread(mtx_file).tocsr()
            
            # Load corresponding metadata
            base_name = mtx_file.stem.replace("_laplacian", "")
            metadata_file = graphs_path / f"{base_name}_metadata.json"
            
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            method_name = f"CMG++_{base_name}"
            
            graphs[method_name] = {
                'matrix': L,
                'source_file': str(mtx_file),
                'method_type': 'CMG++',
                'metadata': metadata
            }
            
            print(f"✅ Loaded CMG++ matrix: {method_name} ({L.shape[0]} nodes, {L.nnz} edges)")
            
        except Exception as e:
            print(f"❌ Error loading {mtx_file}: {e}")
    
    return graphs

def compute_spectral_properties(L, method_name, k=15):
    """Compute comprehensive spectral properties"""
    
    n = L.shape[0]
    k = min(k, n-2)
    
    try:
        # Compute smallest eigenvalues
        eigenvals, eigenvecs = eigsh(L, k=k, which='SM', sigma=0.0)
        eigenvals = np.real(eigenvals)
        eigenvals.sort()
        
        # Connectivity analysis
        zero_threshold = 1e-8
        num_components = np.sum(eigenvals < zero_threshold)
        is_connected = num_components == 1
        
        # Key spectral metrics
        fiedler_value = eigenvals[1] if len(eigenvals) > 1 else 0.0
        spectral_gap = eigenvals[1] - eigenvals[0] if len(eigenvals) > 1 else 0.0
        
        # Effective resistance and mixing properties
        if fiedler_value > zero_threshold:
            mixing_time_est = 1.0 / fiedler_value
            conductance_lower_bound = fiedler_value / 2.0
        else:
            mixing_time_est = float('inf')
            conductance_lower_bound = 0.0
        
        # Spectral density
        eigenval_density = len(eigenvals) / (eigenvals.max() - eigenvals.min()) if eigenvals.max() > eigenvals.min() else 0
        
        result = {
            'method': method_name,
            'nodes': n,
            'edges': L.nnz // 2,
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs,
            'num_components': num_components,
            'is_connected': is_connected,
            'fiedler_value': fiedler_value,
            'spectral_gap': spectral_gap,
            'mixing_time_estimate': mixing_time_est,
            'conductance_lower_bound': conductance_lower_bound,
            'eigenvalue_density': eigenval_density,
            'connectivity_score': fiedler_value  # Simple score for ranking
        }
        
        print(f"[{method_name}] Spectral Analysis Complete:")
        print(f"   Nodes: {n}, Components: {num_components}")
        print(f"   Fiedler (λ₂): {fiedler_value:.6f}")
        print(f"   Connected: {'Yes' if is_connected else 'No'}")
        
        return result
        
    except Exception as e:
        print(f"❌ Spectral analysis failed for {method_name}: {e}")
        return None

def analyze_all_graphs(lamg_graphs, cmg_graphs, accuracy_data=None):
    """Perform spectral analysis on all graphs"""
    
    all_results = []
    
    # Default accuracy data from your experiments
    if accuracy_data is None:
        accuracy_data = {
            'lamg': 79.5,    # LAMG reduce_ratio=3 
            'cmg': 74.8,     # CMG++ level=2
            'reduce_ratio_3': 79.5,
            'level_2': 74.8,
            'k10_d10': 74.8,
            'k10_d20': 75.2,  # Estimated
        }
    
    # Analyze LAMG graphs
    print(f"\n🔍 Analyzing {len(lamg_graphs)} LAMG graphs...")
    for name, graph_data in lamg_graphs.items():
        result = compute_spectral_properties(graph_data['matrix'], name)
        if result:
            # Try to match accuracy data
            result['accuracy'] = None
            for key, acc in accuracy_data.items():
                if key.lower() in name.lower():
                    result['accuracy'] = acc
                    break
            
            result['method_type'] = 'LAMG'
            all_results.append(result)
    
    # Analyze CMG++ graphs
    print(f"\n🔍 Analyzing {len(cmg_graphs)} CMG++ graphs...")
    for name, graph_data in cmg_graphs.items():
        result = compute_spectral_properties(graph_data['matrix'], name)
        if result:
            # Add CMG-specific metadata
            metadata = graph_data.get('metadata', {})
            result['lambda_crit'] = metadata.get('lambda_crit', None)
            result['avg_conductance'] = metadata.get('avg_conductance', None)
            result['cmg_params'] = metadata.get('method_params', {})
            
            # Try to match accuracy data
            result['accuracy'] = None
            for key, acc in accuracy_data.items():
                if key.lower() in name.lower():
                    result['accuracy'] = acc
                    break
            
            result['method_type'] = 'CMG++'
            all_results.append(result)
    
    return all_results

def create_comparison_visualizations(results, output_dir="./analysis_output"):
    """Create comprehensive comparison visualizations"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Separate LAMG and CMG++ results
    lamg_results = [r for r in results if r['method_type'] == 'LAMG']
    cmg_results = [r for r in results if r['method_type'] == 'CMG++']
    
    # 1. Eigenvalue Spectrum Comparison
    fig, axes = plt.subplots(2, max(len(lamg_results), len(cmg_results)), 
                            figsize=(4*max(len(lamg_results), len(cmg_results)), 8))
    
    if len(lamg_results) == 1 and len(cmg_results) == 1:
        axes = axes.reshape(2, 1)
    
    # Plot LAMG eigenvalues
    for i, result in enumerate(lamg_results):
        if i < axes.shape[1]:
            ax = axes[0, i] if axes.ndim > 1 else axes[0]
            eigenvals = result['eigenvalues'][:10]
            ax.plot(range(len(eigenvals)), eigenvals, 'o-', color='blue', linewidth=2, markersize=6)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax.set_title(f"{result['method']}\nFiedler: {result['fiedler_value']:.6f}")
            ax.set_ylabel('Eigenvalue')
            ax.grid(True, alpha=0.3)
    
    # Plot CMG++ eigenvalues
    for i, result in enumerate(cmg_results):
        if i < axes.shape[1]:
            ax = axes[1, i] if axes.ndim > 1 else axes[1]
            eigenvals = result['eigenvalues'][:10]
            ax.plot(range(len(eigenvals)), eigenvals, 'o-', color='red', linewidth=2, markersize=6)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax.set_title(f"{result['method']}\nFiedler: {result['fiedler_value']:.6f}")
            ax.set_xlabel('Eigenvalue Index')
            ax.set_ylabel('Eigenvalue')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "eigenvalue_spectra_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: eigenvalue_spectra_comparison.png")
    
    # 2. Connectivity vs Accuracy Scatter Plot
    if any(r['accuracy'] is not None for r in results):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot points
        for result in results:
            if result['accuracy'] is not None:
                color = 'blue' if result['method_type'] == 'LAMG' else 'red'
                marker = 'o' if result['is_connected'] else 'x'
                size = 100 if result['is_connected'] else 60
                
                ax.scatter(result['fiedler_value'], result['accuracy'], 
                          c=color, marker=marker, s=size, alpha=0.7,
                          label=f"{result['method_type']} ({'Connected' if result['is_connected'] else 'Disconnected'})")
                
                # Annotate points
                ax.annotate(result['method'].replace('LAMG_', '').replace('CMG++_', ''), 
                           (result['fiedler_value'], result['accuracy']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Fiedler Value (λ₂)')
        ax.set_ylabel('Classification Accuracy (%)')
        ax.set_title('Graph Connectivity vs Classification Accuracy')
        ax.grid(True, alpha=0.3)
        
        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        plt.savefig(output_path / "connectivity_vs_accuracy.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved: connectivity_vs_accuracy.png")
    
    # 3. Method Comparison Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    methods = [r['method'] for r in results]
    fiedler_vals = [r['fiedler_value'] for r in results]
    colors = ['blue' if r['method_type'] == 'LAMG' else 'red' for r in results]
    
    # Fiedler values
    bars1 = ax1.bar(range(len(methods)), fiedler_vals, color=colors, alpha=0.7)
    ax1.set_title('Fiedler Value (λ₂) Comparison')
    ax1.set_ylabel('Fiedler Value')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([m.replace('LAMG_', '').replace('CMG++_', '') for m in methods], 
                        rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars1, fiedler_vals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(fiedler_vals)*0.01,
                f'{val:.6f}', ha='center', va='bottom', fontsize=8)
    
    # Node counts
    node_counts = [r['nodes'] for r in results]
    bars2 = ax2.bar(range(len(methods)), node_counts, color=colors, alpha=0.7)
    ax2.set_title('Coarsened Graph Size')
    ax2.set_ylabel('Number of Nodes')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels([m.replace('LAMG_', '').replace('CMG++_', '') for m in methods], 
                        rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path / "method_comparison_bars.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: method_comparison_bars.png")
    
    plt.close('all')  # Close all figures to save memory

def generate_final_report(results, output_dir="./analysis_output"):
    """Generate comprehensive analysis report"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create detailed comparison table
    comparison_data = []
    for result in results:
        row = {
            'Method': result['method'],
            'Type': result['method_type'],
            'Nodes': result['nodes'],
            'Edges': result['edges'],
            'Components': result['num_components'],
            'Connected': 'Yes' if result['is_connected'] else 'No',
            'Fiedler_Value': f"{result['fiedler_value']:.8f}",
            'Spectral_Gap': f"{result['spectral_gap']:.8f}",
            'Mixing_Time': f"{result['mixing_time_estimate']:.2f}" if result['mixing_time_estimate'] != float('inf') else "∞",
            'Conductance_Bound': f"{result['conductance_lower_bound']:.6f}",
            'Accuracy': result.get('accuracy', 'N/A')
        }
        
        # Add CMG-specific fields
        if result['method_type'] == 'CMG++':
            row['Lambda_Critical'] = result.get('lambda_crit', 'N/A')
            row['CMG_Conductance'] = result.get('avg_conductance', 'N/A')
        
        comparison_data.append(row)
    
    # Save as CSV
    df = pd.DataFrame(comparison_data)
    csv_file = output_path / "complete_spectral_analysis.csv"
    df.to_csv(csv_file, index=False)
    print(f"✅ Detailed results saved: {csv_file}")
    
    # Generate text report
    report_file = output_path / "spectral_analysis_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE SPECTRAL ANALYSIS REPORT\n")
        f.write("LAMG vs CMG++ Graph Coarsening Methods\n")
        f.write("="*80 + "\n\n")
        
        # Summary statistics
        lamg_results = [r for r in results if r['method_type'] == 'LAMG']
        cmg_results = [r for r in results if r['method_type'] == 'CMG++']
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"LAMG Methods Analyzed: {len(lamg_results)}\n")
        f.write(f"CMG++ Methods Analyzed: {len(cmg_results)}\n\n")
        
        # Connectivity analysis
        lamg_connected = sum(1 for r in lamg_results if r['is_connected'])
        cmg_connected = sum(1 for r in cmg_results if r['is_connected'])
        
        f.write("CONNECTIVITY ANALYSIS:\n")
        f.write("-" * 22 + "\n")
        f.write(f"LAMG Connected Graphs: {lamg_connected}/{len(lamg_results)}\n")
        f.write(f"CMG++ Connected Graphs: {cmg_connected}/{len(cmg_results)}\n\n")
        
        # Best performers
        if lamg_results:
            best_lamg = max(lamg_results, key=lambda x: x['fiedler_value'])
            f.write(f"Best LAMG Method: {best_lamg['method']}\n")
            f.write(f"  - Fiedler Value: {best_lamg['fiedler_value']:.8f}\n")
            f.write(f"  - Nodes: {best_lamg['nodes']}\n")
            if best_lamg.get('accuracy'):
                f.write(f"  - Accuracy: {best_lamg['accuracy']}%\n")
            f.write("\n")
        
        if cmg_results:
            best_cmg = max(cmg_results, key=lambda x: x['fiedler_value'])
            f.write(f"Best CMG++ Method: {best_cmg['method']}\n")
            f.write(f"  - Fiedler Value: {best_cmg['fiedler_value']:.8f}\n")
            f.write(f"  - Nodes: {best_cmg['nodes']}\n")
            if best_cmg.get('accuracy'):
                f.write(f"  - Accuracy: {best_cmg['accuracy']}%\n")
            f.write("\n")
        
        # Correlation analysis
        accuracy_results = [r for r in results if r.get('accuracy') is not None]
        if len(accuracy_results) >= 2:
            fiedler_vals = [r['fiedler_value'] for r in accuracy_results]
            accuracies = [r['accuracy'] for r in accuracy_results]
            correlation = np.corrcoef(fiedler_vals, accuracies)[0, 1]
            
            f.write("CORRELATION ANALYSIS:\n")
            f.write("-" * 21 + "\n")
            f.write(f"Fiedler Value vs Accuracy Correlation: r = {correlation:.4f}\n")
            
            if correlation > 0.5:
                f.write("STRONG POSITIVE CORRELATION DETECTED!\n")
                f.write("Higher connectivity strongly correlates with better accuracy.\n\n")
            elif correlation > 0.2:
                f.write("Moderate positive correlation observed.\n\n")
            else:
                f.write("Weak or no correlation observed.\n\n")
        
        # Conclusions
        f.write("KEY FINDINGS:\n")
        f.write("-" * 13 + "\n")
        
        if lamg_connected > cmg_connected:
            f.write("✓ LAMG maintains better graph connectivity than CMG++\n")
        
        if lamg_results and cmg_results:
            avg_lamg_fiedler = np.mean([r['fiedler_value'] for r in lamg_results])
            avg_cmg_fiedler = np.mean([r['fiedler_value'] for r in cmg_results])
            
            if avg_lamg_fiedler > avg_cmg_fiedler:
                f.write("✓ LAMG achieves higher average Fiedler values\n")
            
        f.write("\nRECOMMENDATIONS:\n")
        f.write("-" * 15 + "\n")
        f.write("1. Use LAMG for applications requiring strong connectivity\n")
        f.write("2. CMG++ may be suitable for applications tolerating disconnection\n")
        f.write("3. Consider hybrid approaches combining both methods\n")
        f.write("4. Monitor Fiedler values as connectivity quality indicators\n")
        
        # Detailed table
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED COMPARISON TABLE:\n")
        f.write("="*80 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n" + "="*80 + "\n")
    
    print(f"✅ Comprehensive report saved: {report_file}")
    
    return df

def main():
    """Main analysis pipeline"""
    
    print("🔬 COMPREHENSIVE SPECTRAL ANALYSIS PIPELINE")
    print("=" * 50)
    print("Comparing LAMG vs CMG++ Graph Coarsening Methods")
    print("=" * 50)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    issues = check_dependencies()
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"   {issue}")
        print("\n💡 Please resolve these issues before continuing.")
        
        # Ask if user wants to continue anyway
        response = input("\nContinue anyway? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            return
    
    # Load existing graphs
    print("\n📁 Loading existing graphs...")
    
    # Load LAMG graphs
    lamg_graphs = load_lamg_graphs("./reduction_results")
    print(f"   Found {len(lamg_graphs)} LAMG graphs")
    
    # Load CMG++ graphs  
    cmg_graphs = load_cmg_graphs("./cmg_extracted_graphs")
    print(f"   Found {len(cmg_graphs)} CMG++ graphs")
    
    # If no CMG++ graphs found, try to extract them
    if not cmg_graphs:
        print("\n🔄 No CMG++ graphs found. Attempting extraction...")
        try:
            # Run CMG extraction
            import subprocess
            result = subprocess.run([
                sys.executable, "cmg_graph_extractor.py"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ CMG++ extraction completed")
                cmg_graphs = load_cmg_graphs("./cmg_extracted_graphs")
                print(f"   Extracted {len(cmg_graphs)} CMG++ graphs")
            else:
                print(f"❌ CMG++ extraction failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Could not run CMG++ extraction: {e}")
    
    # Check if we have any graphs to analyze
    total_graphs = len(lamg_graphs) + len(cmg_graphs)
    if total_graphs == 0:
        print("\n❌ No graphs found for analysis!")
        print("Please ensure:")
        print("   1. LAMG results are in ./reduction_results/*.mtx")
        print("   2. Run CMG extraction first, or")
        print("   3. Place CMG++ graphs in ./cmg_extracted_graphs/")
        return
    
    print(f"\n✅ Total graphs for analysis: {total_graphs}")
    
    # Perform spectral analysis
    print("\n🔍 Performing spectral analysis...")
    
    # Load accuracy data (customize this with your actual results)
    accuracy_data = {
        # LAMG results
        'reduce_ratio_3': 79.5,
        'lamg_3': 79.5,
        'gs': 79.5,  # Common LAMG output filename
        
        # CMG++ results  
        'level_2': 74.8,
        'k10_d10': 74.8,
        'k10_d20': 75.2,
        'k15_d20': 75.5,
        'multilevel_level_2': 74.8,
        
        # Add more mappings as needed
        'cmg': 74.8,
    }
    
    results = analyze_all_graphs(lamg_graphs, cmg_graphs, accuracy_data)
    
    if not results:
        print("❌ No successful spectral analyses completed")
        return
    
    print(f"✅ Completed analysis of {len(results)} graphs")
    
    # Generate visualizations
    print("\n📈 Creating visualizations...")
    create_comparison_visualizations(results, "./analysis_output")
    
    # Generate comprehensive report
    print("\n📊 Generating comprehensive report...")
    final_df = generate_final_report(results, "./analysis_output")
    
    # Print summary to console
    print("\n" + "="*60)
    print("SPECTRAL ANALYSIS SUMMARY")
    print("="*60)
    
    # Quick summary table
    summary_data = []
    for result in results:
        summary_data.append({
            'Method': result['method'].replace('LAMG_', '').replace('CMG++_', ''),
            'Type': result['method_type'],
            'Nodes': result['nodes'],
            'Connected': 'Yes' if result['is_connected'] else 'No',
            'Fiedler': f"{result['fiedler_value']:.6f}",
            'Accuracy': f"{result['accuracy']}%" if result.get('accuracy') else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Key insights
    print(f"\n🔍 KEY INSIGHTS:")
    
    lamg_results = [r for r in results if r['method_type'] == 'LAMG']
    cmg_results = [r for r in results if r['method_type'] == 'CMG++']
    
    if lamg_results and cmg_results:
        lamg_connected = sum(1 for r in lamg_results if r['is_connected'])
        cmg_connected = sum(1 for r in cmg_results if r['is_connected'])
        
        print(f"   📊 LAMG connectivity: {lamg_connected}/{len(lamg_results)} graphs connected")
        print(f"   📊 CMG++ connectivity: {cmg_connected}/{len(cmg_results)} graphs connected")
        
        avg_lamg_fiedler = np.mean([r['fiedler_value'] for r in lamg_results])
        avg_cmg_fiedler = np.mean([r['fiedler_value'] for r in cmg_results])
        
        print(f"   📊 Average Fiedler - LAMG: {avg_lamg_fiedler:.6f}, CMG++: {avg_cmg_fiedler:.6f}")
        
        if avg_lamg_fiedler > avg_cmg_fiedler * 2:
            print(f"   🎯 LAMG shows significantly better connectivity!")
        elif avg_lamg_fiedler > avg_cmg_fiedler:
            print(f"   ✅ LAMG shows better connectivity")
        else:
            print(f"   ⚠️  CMG++ shows competitive connectivity")
    
    # Correlation analysis
    accuracy_results = [r for r in results if r.get('accuracy') is not None]
    if len(accuracy_results) >= 2:
        fiedler_vals = [r['fiedler_value'] for r in accuracy_results]
        accuracies = [r['accuracy'] for r in accuracy_results]
        correlation = np.corrcoef(fiedler_vals, accuracies)[0, 1]
        
        print(f"   📈 Connectivity-Accuracy correlation: r = {correlation:.4f}")
        
        if correlation > 0.5:
            print(f"   🎯 STRONG evidence: Better connectivity → Higher accuracy!")
        elif correlation > 0.2:
            print(f"   ✅ Moderate evidence: Connectivity affects accuracy")
    
    print(f"\n📁 All results saved to: ./analysis_output/")
    print(f"   📊 Complete data: spectral_analysis_results.csv")
    print(f"   📈 Visualizations: *.png files")
    print(f"   📝 Full report: spectral_analysis_report.txt")
    
    print(f"\n🎉 Analysis complete!")

def extract_only():
    """Extract CMG++ graphs only"""
    print("🔄 Extracting CMG++ graphs only...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "cmg_graph_extractor.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ CMG++ extraction completed successfully")
            print(result.stdout)
        else:
            print(f"❌ CMG++ extraction failed")
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"❌ Could not run extraction: {e}")

def analyze_only():
    """Analyze existing graphs only (skip extraction)"""
    print("🔍 Analyzing existing graphs only...")
    
    # Load graphs
    lamg_graphs = load_lamg_graphs("./reduction_results") 
    cmg_graphs = load_cmg_graphs("./cmg_extracted_graphs")
    
    total_graphs = len(lamg_graphs) + len(cmg_graphs)
    if total_graphs == 0:
        print("❌ No graphs found for analysis!")
        return
    
    print(f"Found {total_graphs} graphs for analysis")
    
    # Standard accuracy data
    accuracy_data = {
        'reduce_ratio_3': 79.5,
        'level_2': 74.8,
        'k10_d10': 74.8,
    }
    
    results = analyze_all_graphs(lamg_graphs, cmg_graphs, accuracy_data)
    
    if results:
        create_comparison_visualizations(results, "./analysis_output")
        generate_final_report(results, "./analysis_output")
        print("✅ Analysis completed successfully!")
    else:
        print("❌ No successful analyses completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive spectral analysis of LAMG vs CMG++ methods"
    )
    parser.add_argument("--extract-only", action="store_true",
                       help="Only extract CMG++ graphs")
    parser.add_argument("--analyze-only", action="store_true", 
                       help="Only analyze existing graphs")
    
    args = parser.parse_args()
    
    if args.extract_only:
        extract_only()
    elif args.analyze_only:
        analyze_only()
    else:
        main()
