"""
Quick Setup and Execution Script for CMG Community Detection Experiments
Run this script to get started immediately
"""

import os
import subprocess
import sys

def install_dependencies():
    """Install required packages"""
    print("Installing required dependencies...")
    
    packages = [
        'numpy',
        'scipy',
        'networkx',
        'scikit-learn',
        'pandas',
        'matplotlib',
        'seaborn',
        'torch',
        'torch-geometric'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed")
        except:
            print(f"❌ Failed to install {package}")

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        'results',
        'alignment_validation_results', 
        'community_detection_results',
        'datasets'
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"📁 Created directory: {dir_name}")

def download_sample_datasets():
    """Download or create sample datasets"""
    import networkx as nx
    import pickle
    
    print("Creating sample datasets...")
    
    datasets = {}
    
    # 1. Karate Club (built-in)
    karate = nx.karate_club_graph()
    karate_labels = [karate.nodes[i]['club'] == 'Mr. Hi' for i in karate.nodes()]
    datasets['karate'] = (karate, [int(x) for x in karate_labels])
    
    # 2. Dolphins (synthetic version)
    dolphins = nx.planted_partition_graph(2, 31, 0.7, 0.1, seed=42)
    dolphins_labels = [0] * 31 + [1] * 31
    datasets['dolphins'] = (dolphins, dolphins_labels)
    
    # 3. Football conferences (synthetic)
    football = nx.planted_partition_graph(12, 10, 0.8, 0.1, seed=42)
    football_labels = []
    for i in range(12):
        football_labels.extend([i] * 10)
    datasets['football'] = (football, football_labels)
    
    # 4. Large synthetic community graph
    large_graph = nx.planted_partition_graph(5, 200, 0.6, 0.1, seed=42)
    large_labels = []
    for i in range(5):
        large_labels.extend([i] * 200)
    datasets['large_synthetic'] = (large_graph, large_labels)
    
    # Save datasets
    with open('datasets/community_datasets.pkl', 'wb') as f:
        pickle.dump(datasets, f)
    
    print(f"✅ Created {len(datasets)} sample datasets")
    
    # Print dataset statistics
    for name, (graph, labels) in datasets.items():
        print(f"  {name}: {len(graph.nodes())} nodes, {len(graph.edges())} edges, {len(set(labels))} communities")

def create_minimal_cmg_implementation():
    """Create a minimal CMG implementation for testing"""
    minimal_cmg = '''
"""
Minimal CMG Implementation for Testing
Replace this with your actual CMG implementation
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering

def cmg_filtered_clustering(data, k=10, d=20, threshold=0.1):
    """
    Minimal CMG implementation for testing
    Replace with your actual implementation
    """
    # Convert PyG data to NetworkX
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(edge_index.T)
    
    # Simple spectral clustering as placeholder
    try:
        # Estimate number of clusters
        n_clusters = max(2, min(10, int(np.sqrt(data.num_nodes))))
        
        # Use spectral clustering
        clustering = SpectralClustering(n_clusters=n_clusters, random_state=42)
        adj_matrix = nx.adjacency_matrix(G).toarray()
        
        if adj_matrix.shape[0] > n_clusters:
            cluster_assignments = clustering.fit_predict(adj_matrix)
        else:
            cluster_assignments = np.arange(adj_matrix.shape[0])
        
        # Compute simple conductance
        conductance = nx.average_clustering(G) if len(G.edges()) > 0 else 0.0
        
        return cluster_assignments, n_clusters, {'avg_phi': conductance}, 2.0
        
    except Exception as e:
        print(f"CMG clustering failed: {e}")
        # Fallback to random clustering
        n_clusters = max(2, data.num_nodes // 4)
        cluster_assignments = np.random.randint(0, n_clusters, data.num_nodes)
        return cluster_assignments, n_clusters, {'avg_phi': 0.5}, 2.0

def start_timing(step_name):
    """Dummy timing function"""
    import time
    return time.time()

def end_timing(step_name, start_time):
    """Dummy timing function"""
    import time
    return time.time() - start_time

def save_timing_data(filepath):
    """Dummy save function"""
    pass
'''
    
    # Create minimal implementations if they don't exist
    if not os.path.exists('filtered_timed.py'):
        with open('filtered_timed.py', 'w') as f:
            f.write(minimal_cmg)
        print("✅ Created minimal filtered_timed.py")
    
    if not os.path.exists('cmg_coarsening_timed.py'):
        minimal_coarsening = '''
from filtered_timed import cmg_filtered_clustering, start_timing, end_timing, save_timing_data

# Import all functions from the main implementation
# This is a placeholder - replace with your actual implementation
'''
        with open('cmg_coarsening_timed.py', 'w') as f:
            f.write(minimal_coarsening)
        print("✅ Created minimal cmg_coarsening_timed.py")

def run_quick_test():
    """Run a quick test to verify everything works"""
    print("\\n=== RUNNING QUICK TEST ===")
    
    try:
        from experiment_runner import AlignmentValidationExperiment
        import pickle
        
        # Load test datasets
        with open('datasets/community_datasets.pkl', 'rb') as f:
            datasets = pickle.load(f)
        
        # Run small test
        test_datasets = {'karate': datasets['karate']}
        
        experiment = AlignmentValidationExperiment()
        results_df = experiment.run_alignment_test(test_datasets, n_trials=2)
        
        print("✅ Quick test completed successfully!")
        print(f"Test results: {len(results_df)} experiments completed")
        
        # Show sample results
        if len(results_df) > 0:
            successful = results_df[results_df['success'] == True]
            print(f"Success rate: {len(successful)}/{len(results_df)} = {len(successful)/len(results_df):.1%}")
            
            if len(successful) > 0:
                for method in successful['method'].unique():
                    method_data = successful[successful['method'] == method]
                    avg_modularity = method_data['modularity'].mean()
                    print(f"  {method}: avg modularity = {avg_modularity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        print("Please check your CMG implementation and dependencies")
        return False

def main():
    """Main setup function"""
    print("=== CMG EXPERIMENT SETUP ===\\n")
    
    # Step 1: Install dependencies
    install_dependencies()
    print()
    
    # Step 2: Create directories
    create_directory_structure()
    print()
    
    # Step 3: Create sample datasets
    download_sample_datasets()
    print()
    
    # Step 4: Create minimal implementations
    create_minimal_cmg_implementation()
    print()
    
    # Step 5: Run quick test
    if run_quick_test():
        print("\\n🎉 Setup completed successfully!")
        print("\\n=== NEXT STEPS ===")
        print("1. Replace the minimal CMG implementation with your actual code")
        print("2. Run: python experiment_runner.py")
        print("3. Check results in alignment_validation_results/")
        print("\\n=== FULL EXPERIMENT COMMANDS ===")
        print("# Quick test (recommended first):")
        print("python quick_setup.py")
        print("\\n# Full alignment validation:")
        print("python experiment_runner.py")
        print("\\n# Complete community detection experiments:")
        print("python community_detection_experiments.py")
    else:
        print("\\n⚠️  Setup encountered issues. Please check error messages above.")

if __name__ == "__main__":
    main()
