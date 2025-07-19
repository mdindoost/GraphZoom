#!/bin/bash
# Real Spectral Analysis of GraphZoom Results
# Analyze actual spectral properties from GraphZoom coarsening

echo "🔬 REAL SPECTRAL ANALYSIS"
echo "=========================="
echo "Testing actual GraphZoom coarsening methods and analyzing spectral properties"
echo ""

MATLAB_MCR_ROOT="/home/mohammad/matlab/R2018a"
RESULTS_DIR="spectral_analysis_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p $RESULTS_DIR/graphs
mkdir -p $RESULTS_DIR/logs

# Python script to analyze graph spectral properties
cat > $RESULTS_DIR/analyze_spectrum.py << 'EOF'
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
EOF

# Function to run GraphZoom and analyze spectral properties
run_spectral_test() {
    local method=$1
    local config=$2
    local cmd_params=$3
    local expected_nodes=$4
    
    echo "🧪 Testing spectral properties: $method ($config)"
    
    # Build command
    local cmd="python graphzoom_timed.py --dataset cora --embed_method deepwalk --seed 42"
    
    if [ "$method" = "lamg" ]; then
        cmd="$cmd --coarse lamg --mcr_dir $MATLAB_MCR_ROOT $cmd_params"
    elif [ "$method" = "cmg" ]; then
        cmd="$cmd --coarse cmg $cmd_params"
    elif [ "$method" = "simple" ]; then
        cmd="$cmd --coarse simple $cmd_params"
    fi
    
    local log_file="$RESULTS_DIR/logs/${method}_${config}.log"
    
    # Run GraphZoom
    echo "Running: $cmd"
    $cmd > $log_file 2>&1
    
    # Check for output files
    local graph_file=""
    
    if [ "$method" = "lamg" ]; then
        if [ -f "reduction_results/Gs.mtx" ]; then
            graph_file="reduction_results/Gs.mtx"
        fi
    elif [ "$method" = "original" ]; then
        if [ -f "dataset/cora/cora.mtx" ]; then
            graph_file="dataset/cora/cora.mtx"
        fi
    fi
    
    if [ -f "$graph_file" ]; then
        # Copy graph file for analysis
        cp "$graph_file" "$RESULTS_DIR/graphs/${method}_${config}.mtx"
        
        # Analyze spectral properties
        cd $RESULTS_DIR
        python analyze_spectrum.py "graphs/${method}_${config}.mtx" "${method}_${config}"
        cd ..
        
        echo "✅ Spectral analysis complete for $method ($config)"
    else
        echo "❌ No graph file found for $method ($config)"
        echo "   Expected: $graph_file"
        echo "   Log file: $log_file"
    fi
    
    # Clean up
    rm -rf reduction_results/* 2>/dev/null
    echo ""
}

echo "📋 SPECTRAL ANALYSIS PLAN:"
echo "========================="
echo "1. Analyze original Cora graph spectral properties"
echo "2. Test LAMG coarsening at different levels"
echo "3. Test CMG++ coarsening at different levels"
echo "4. Compare spectral preservation quality"
echo ""

# Test original graph (if available)
echo "🔍 ORIGINAL GRAPH ANALYSIS"
echo "========================="
# Note: This would require the original Cora graph in MTX format
# run_spectral_test "original" "cora" "" "2708"

# Test LAMG at different reduction ratios
echo "🔍 LAMG SPECTRAL ANALYSIS"
echo "========================="
run_spectral_test "lamg" "reduce_2" "--reduce_ratio 2" "1169"
run_spectral_test "lamg" "reduce_3" "--reduce_ratio 3" "519"
run_spectral_test "lamg" "reduce_6" "--reduce_ratio 6" "218"

# Test CMG++ at different levels
echo "🔍 CMG++ SPECTRAL ANALYSIS"
echo "=========================="
run_spectral_test "cmg" "level_1" "--level 1 --cmg_k 10 --cmg_d 10" "900"
run_spectral_test "cmg" "level_2" "--level 2 --cmg_k 10 --cmg_d 10" "400"
run_spectral_test "cmg" "level_3" "--level 3 --cmg_k 10 --cmg_d 10" "250"

# Generate comparison report
echo "📊 GENERATING SPECTRAL COMPARISON REPORT"
echo "======================================="

cd $RESULTS_DIR

cat > spectral_comparison.py << 'EOF'
import glob
import numpy as np
import matplotlib.pyplot as plt

def load_results():
    """Load all spectral analysis results"""
    results = {}
    
    for file in glob.glob("spectral_results_*.txt"):
        method_name = file.replace("spectral_results_", "").replace(".txt", "")
        
        data = {}
        with open(file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    if key in ['Nodes', 'Edges']:
                        data[key] = int(value)
                    elif key in ['Fiedler_value', 'Spectral_gap']:
                        data[key] = float(value)
                    elif key == 'Eigenvalues':
                        data[key] = [float(x) for x in value.split(',')]
                    else:
                        data[key] = value.strip()
        
        results[method_name] = data
    
    return results

def plot_spectral_comparison(results):
    """Create spectral comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Fiedler values
    ax1 = axes[0, 0]
    methods = list(results.keys())
    fiedler_values = [results[m].get('Fiedler_value', 0) for m in methods]
    
    bars = ax1.bar(methods, fiedler_values, alpha=0.7)
    ax1.set_ylabel('Fiedler Value (λ₂)')
    ax1.set_title('Algebraic Connectivity Comparison')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, fiedler_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom')
    
    # Plot 2: Spectral gaps
    ax2 = axes[0, 1]
    spectral_gaps = [results[m].get('Spectral_gap', 0) for m in methods]
    
    bars = ax2.bar(methods, spectral_gaps, alpha=0.7)
    ax2.set_ylabel('Spectral Gap (λ₃ - λ₂)')
    ax2.set_title('Cluster Separation Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, spectral_gaps):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom')
    
    # Plot 3: Node count vs Fiedler value
    ax3 = axes[1, 0]
    node_counts = [results[m].get('Nodes', 0) for m in methods]
    
    ax3.scatter(node_counts, fiedler_values, alpha=0.7, s=100)
    for i, method in enumerate(methods):
        ax3.annotate(method, (node_counts[i], fiedler_values[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax3.set_xlabel('Number of Nodes')
    ax3.set_ylabel('Fiedler Value (λ₂)')
    ax3.set_title('Compression vs Connectivity')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Eigenvalue spectra comparison
    ax4 = axes[1, 1]
    for method in methods:
        eigenvals = results[method].get('Eigenvalues', [])
        if len(eigenvals) > 0:
            ax4.plot(eigenvals[:10], 'o-', label=method, alpha=0.7)
    
    ax4.set_xlabel('Eigenvalue Index')
    ax4.set_ylabel('Eigenvalue')
    ax4.set_title('Eigenvalue Spectra Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spectral_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_report(results):
    """Generate text report"""
    print("🔬 SPECTRAL ANALYSIS REPORT")
    print("=" * 50)
    
    # Sort by number of nodes
    sorted_methods = sorted(results.keys(), 
                           key=lambda x: results[x].get('Nodes', 0), 
                           reverse=True)
    
    print(f"\n{'Method':<15} {'Nodes':<8} {'Edges':<8} {'Fiedler':<12} {'Spectral Gap':<12} {'Compression':<12}")
    print("-" * 80)
    
    original_nodes = max(results[m].get('Nodes', 0) for m in results.keys())
    
    for method in sorted_methods:
        data = results[method]
        nodes = data.get('Nodes', 0)
        edges = data.get('Edges', 0)
        fiedler = data.get('Fiedler_value', 0)
        gap = data.get('Spectral_gap', 0)
        compression = original_nodes / nodes if nodes > 0 else 0
        
        print(f"{method:<15} {nodes:<8} {edges:<8} {fiedler:<12.6f} {gap:<12.6f} {compression:<12.2f}x")
    
    # Analysis
    print("\n🎯 KEY SPECTRAL INSIGHTS:")
    print("=" * 50)
    
    # Find method with best Fiedler preservation
    lamg_methods = [m for m in results.keys() if 'lamg' in m.lower()]
    cmg_methods = [m for m in results.keys() if 'cmg' in m.lower()]
    
    if lamg_methods and cmg_methods:
        # Compare similar compression levels
        print("\n📊 SPECTRAL PRESERVATION COMPARISON:")
        print("(Higher Fiedler value = Better connectivity preservation)")
        
        for lamg_method in lamg_methods:
            lamg_nodes = results[lamg_method].get('Nodes', 0)
            lamg_fiedler = results[lamg_method].get('Fiedler_value', 0)
            
            # Find CMG method with similar node count
            closest_cmg = None
            min_diff = float('inf')
            
            for cmg_method in cmg_methods:
                cmg_nodes = results[cmg_method].get('Nodes', 0)
                diff = abs(lamg_nodes - cmg_nodes)
                if diff < min_diff:
                    min_diff = diff
                    closest_cmg = cmg_method
            
            if closest_cmg:
                cmg_fiedler = results[closest_cmg].get('Fiedler_value', 0)
                cmg_nodes = results[closest_cmg].get('Nodes', 0)
                
                print(f"\n{lamg_method} ({lamg_nodes} nodes): λ₂ = {lamg_fiedler:.6f}")
                print(f"{closest_cmg} ({cmg_nodes} nodes): λ₂ = {cmg_fiedler:.6f}")
                
                if lamg_fiedler > cmg_fiedler:
                    improvement = ((lamg_fiedler - cmg_fiedler) / cmg_fiedler) * 100
                    print(f"✅ LAMG preserves connectivity {improvement:.1f}% better")
                else:
                    improvement = ((cmg_fiedler - lamg_fiedler) / lamg_fiedler) * 100
                    print(f"✅ CMG++ preserves connectivity {improvement:.1f}% better")
    
    print("\n💡 INTERPRETATION:")
    print("- Fiedler value (λ₂): Graph connectivity strength")
    print("- Spectral gap (λ₃-λ₂): Cluster separation quality")
    print("- Higher values = Better structural preservation")
    print("- Better preservation → Better embeddings → Higher accuracy")

if __name__ == "__main__":
    results = load_results()
    
    if results:
        generate_report(results)
        plot_spectral_comparison(results)
        print(f"\n📁 Spectral comparison plot saved to: spectral_comparison.png")
    else:
        print("❌ No spectral analysis results found")
EOF

# Run the comparison analysis
echo "Running spectral comparison analysis..."
python spectral_comparison.py

cd ..

echo ""
echo "✅ SPECTRAL ANALYSIS COMPLETE!"
echo "=============================="
echo ""
echo "📁 Results saved to: $RESULTS_DIR/"
echo "📊 Key files:"
echo "   - spectral_comparison.png: Visual comparison"
echo "   - spectral_results_*.txt: Individual method results"
echo "   - graphs/*.mtx: Coarsened graph files"
echo ""
echo "🎯 This analysis reveals:"
echo "   1. How well each method preserves graph connectivity (Fiedler value)"
echo "   2. How well each method preserves cluster structure (spectral gap)"
echo "   3. Why LAMG might achieve better accuracy despite similar compression"
echo "   4. The spectral reason behind CMG++'s accuracy degradation"
