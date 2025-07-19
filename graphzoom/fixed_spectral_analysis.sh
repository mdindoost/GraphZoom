#!/bin/bash
# Fixed Spectral Analysis - Proper file handling

echo "🔬 FIXED SPECTRAL ANALYSIS"
echo "=========================="

MATLAB_MCR_ROOT="/home/mohammad/matlab/R2018a"
RESULTS_DIR="fixed_spectral_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p $RESULTS_DIR/graphs
mkdir -p $RESULTS_DIR/logs

# Updated Python analysis script with better error handling
cat > $RESULTS_DIR/analyze_spectrum.py << 'EOF'
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
EOF

# Fixed function to run GraphZoom and analyze spectral properties
run_fixed_spectral_test() {
    local method=$1
    local config=$2
    local cmd_params=$3
    local expected_nodes=$4
    
    echo "🧪 Testing spectral properties: $method ($config)"
    
    # Clean up first
    rm -rf reduction_results/* 2>/dev/null
    
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
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "❌ GraphZoom failed with exit code $exit_code"
        return 1
    fi
    
    # Check for output files IMMEDIATELY after run
    echo "📁 Checking for output files..."
    ls -la reduction_results/ 2>/dev/null || echo "No reduction_results directory"
    
    # Find the coarsened graph file
    local graph_file=""
    
    if [ "$method" = "lamg" ]; then
        if [ -f "reduction_results/Gs.mtx" ]; then
            graph_file="reduction_results/Gs.mtx"
            echo "✅ Found LAMG output: $graph_file"
        else
            echo "❌ LAMG output file not found"
            echo "Available files:"
            ls -la reduction_results/ 2>/dev/null || echo "No files found"
            return 1
        fi
    fi
    
    # Copy file IMMEDIATELY before it gets cleaned up
    if [ -f "$graph_file" ]; then
        local dest_file="$RESULTS_DIR/graphs/${method}_${config}.mtx"
        cp "$graph_file" "$dest_file"
        echo "✅ Copied $graph_file to $dest_file"
        
        # Verify copy worked
        if [ -f "$dest_file" ]; then
            echo "✅ File copy successful: $(ls -lh $dest_file)"
        else
            echo "❌ File copy failed"
            return 1
        fi
        
        # Analyze spectral properties
        echo "🔍 Analyzing spectral properties..."
        cd $RESULTS_DIR
        python analyze_spectrum.py "graphs/${method}_${config}.mtx" "${method}_${config}"
        local analysis_exit=$?
        cd ..
        
        if [ $analysis_exit -eq 0 ]; then
            echo "✅ Spectral analysis complete for $method ($config)"
        else
            echo "❌ Spectral analysis failed for $method ($config)"
        fi
    else
        echo "❌ No graph file found for $method ($config)"
        echo "   Expected: $graph_file"
        echo "   Available files:"
        ls -la reduction_results/ 2>/dev/null || echo "No files found"
    fi
    
    echo ""
}

echo "📋 FIXED SPECTRAL ANALYSIS PLAN:"
echo "================================"
echo "1. Run GraphZoom with different coarsening methods"
echo "2. IMMEDIATELY copy output files before cleanup"
echo "3. Analyze spectral properties of each coarsened graph"
echo "4. Compare spectral preservation quality"
echo ""

# Test LAMG at different reduction ratios
echo "🔍 LAMG SPECTRAL ANALYSIS"
echo "========================="
run_fixed_spectral_test "lamg" "reduce_2" "--reduce_ratio 2" "1169"
run_fixed_spectral_test "lamg" "reduce_3" "--reduce_ratio 3" "519"
run_fixed_spectral_test "lamg" "reduce_6" "--reduce_ratio 6" "218"

# Generate comparison report
echo "📊 GENERATING SPECTRAL COMPARISON REPORT"
echo "======================================="

cd $RESULTS_DIR

# Create comparison script
cat > compare_spectral_results.py << 'EOF'
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
                        try:
                            data[key] = [float(x) for x in value.split(',')]
                        except:
                            data[key] = []
                    else:
                        data[key] = value.strip()
        
        results[method_name] = data
    
    return results

def generate_report(results):
    """Generate spectral comparison report"""
    print("🔬 SPECTRAL ANALYSIS REPORT")
    print("=" * 50)
    
    if not results:
        print("❌ No results found")
        return
    
    # Create DataFrame for easy analysis
    df_data = []
    for method, data in results.items():
        df_data.append({
            'Method': method,
            'Nodes': data.get('Nodes', 0),
            'Edges': data.get('Edges', 0),
            'Fiedler_Value': data.get('Fiedler_value', 0),
            'Spectral_Gap': data.get('Spectral_gap', 0),
            'Compression': 2708 / data.get('Nodes', 1) if data.get('Nodes', 0) > 0 else 0
        })
    
    df = pd.DataFrame(df_data)
    df = df.sort_values('Nodes', ascending=False)
    
    print(f"\n{'Method':<15} {'Nodes':<8} {'Edges':<8} {'Compression':<12} {'Fiedler':<12} {'Spectral Gap':<12}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{row['Method']:<15} {row['Nodes']:<8} {row['Edges']:<8} {row['Compression']:<12.2f}x {row['Fiedler_Value']:<12.6f} {row['Spectral_Gap']:<12.6f}")
    
    # Key insights
    print("\n🎯 KEY SPECTRAL INSIGHTS:")
    print("=" * 50)
    
    # Find best spectral preservation
    best_fiedler = df.loc[df['Fiedler_Value'].idxmax()]
    best_gap = df.loc[df['Spectral_Gap'].idxmax()]
    
    print(f"✅ Best Fiedler value preservation: {best_fiedler['Method']} ({best_fiedler['Fiedler_Value']:.6f})")
    print(f"✅ Best spectral gap preservation: {best_gap['Method']} ({best_gap['Spectral_Gap']:.6f})")
    
    # Compare LAMG methods
    lamg_methods = df[df['Method'].str.contains('lamg', case=False, na=False)]
    if len(lamg_methods) > 1:
        print(f"\n📊 LAMG COMPRESSION vs SPECTRAL QUALITY:")
        print("=" * 50)
        for _, row in lamg_methods.iterrows():
            print(f"{row['Method']}: {row['Compression']:.1f}x compression, λ₂={row['Fiedler_Value']:.6f}")
        
        # Check if there's a trade-off
        fiedler_trend = lamg_methods['Fiedler_Value'].corr(lamg_methods['Compression'])
        print(f"\n💡 Fiedler-Compression correlation: {fiedler_trend:.3f}")
        if fiedler_trend < -0.5:
            print("   Strong negative correlation: Higher compression → Lower connectivity")
        elif fiedler_trend > 0.5:
            print("   Strong positive correlation: Higher compression → Higher connectivity (unusual)")
        else:
            print("   Weak correlation: LAMG maintains connectivity across compression levels")
    
    return df

if __name__ == "__main__":
    results = load_results()
    
    if results:
        df = generate_report(results)
        
        # Save summary
        if not df.empty:
            df.to_csv('spectral_summary.csv', index=False)
            print(f"\n📁 Summary saved to: spectral_summary.csv")
    else:
        print("❌ No spectral analysis results found")
EOF

# Run the comparison
echo "Running spectral comparison analysis..."
python compare_spectral_results.py

cd ..

echo ""
echo "✅ FIXED SPECTRAL ANALYSIS COMPLETE!"
echo "===================================="
echo ""
echo "📁 Results saved to: $RESULTS_DIR/"
echo "📊 Key files:"
echo "   - spectral_summary.csv: Tabular results"
echo "   - spectral_results_*.txt: Individual method results"
echo "   - graphs/*.mtx: Coarsened graph files"
echo ""
echo "🎯 This analysis reveals:"
echo "   1. How LAMG's spectral properties change with compression"
echo "   2. Whether LAMG maintains connectivity across compression levels"
echo "   3. The spectral reason behind LAMG's consistent accuracy"
echo "   4. Why LAMG stops at certain compression levels"
