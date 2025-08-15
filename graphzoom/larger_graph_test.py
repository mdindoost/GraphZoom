#!/usr/bin/env python3
"""
Test MP-aware coarsening on larger, more complex graphs where benefits should be visible
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix, diags
import time
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def create_larger_test_graph(n_nodes=50, connection_prob=0.1):
    """Create a larger, more complex test graph"""
    print(f"Creating larger test graph with {n_nodes} nodes...")
    
    # Create Erdős–Rényi random graph
    G = nx.erdos_renyi_graph(n_nodes, connection_prob, seed=42)
    
    # Add some structure: create communities
    community_size = n_nodes // 4
    
    # Add dense connections within communities
    for community in range(4):
        start = community * community_size
        end = min((community + 1) * community_size, n_nodes)
        
        for i in range(start, end):
            for j in range(i+1, end):
                if np.random.random() < 0.3:  # 30% chance of intra-community edge
                    G.add_edge(i, j)
    
    # Add sparse connections between communities
    for i in range(n_nodes):
        for j in range(i + community_size, n_nodes):
            if np.random.random() < 0.02:  # 2% chance of inter-community edge
                G.add_edge(i, j)
    
    print(f"Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def create_complex_node_features(G, dim=32):
    """Create more complex node features based on graph structure"""
    n_nodes = G.number_of_nodes()
    features = np.zeros((n_nodes, dim))
    
    # Structural features
    degrees = dict(G.degree())
    clustering = nx.clustering(G)
    
    try:
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
    except:
        betweenness = {i: 0.1 for i in G.nodes()}
        closeness = {i: 0.1 for i in G.nodes()}
    
    for i, node in enumerate(G.nodes()):
        # Basic structural features
        features[i, 0] = degrees[node] / max(degrees.values())
        features[i, 1] = clustering[node]
        features[i, 2] = betweenness[node]
        features[i, 3] = closeness[node]
        
        # Community-based features (assuming 4 communities)
        community = node // (n_nodes // 4)
        features[i, 4 + community] = 1.0
        
        # Random features with some correlation to structure
        base_pattern = np.random.randn(dim - 8)
        noise = 0.1 * np.random.randn(dim - 8)
        features[i, 8:] = base_pattern + noise
        
        # Add community-specific patterns
        if community < 4:
            community_pattern = np.random.randn(dim - 8)
            features[i, 8:] += 0.3 * community_pattern
    
    return features

def create_challenging_labels(G, features):
    """Create challenging labels that require both structure and features"""
    n_nodes = G.number_of_nodes()
    degrees = dict(G.degree())
    
    labels = np.zeros(n_nodes, dtype=int)
    
    for i, node in enumerate(G.nodes()):
        # Complex labeling rule combining multiple factors
        degree = degrees[node]
        community = node // (n_nodes // 4)
        feature_sum = np.sum(features[i, 8:12])  # Sum of some features
        
        # Multi-factor decision
        if degree > np.percentile(list(degrees.values()), 75):
            if feature_sum > 0:
                labels[i] = 2  # High degree, positive features
            else:
                labels[i] = 1  # High degree, negative features
        elif degree < np.percentile(list(degrees.values()), 25):
            labels[i] = 0  # Low degree
        else:
            if community % 2 == 0:
                labels[i] = 1  # Medium degree, even community
            else:
                labels[i] = 2  # Medium degree, odd community
    
    print(f"Label distribution: {np.bincount(labels)}")
    return labels

def test_on_larger_graph():
    """Test MP-aware coarsening on a larger graph where benefits should be visible"""
    print("TESTING ON LARGER GRAPH (50 nodes)")
    print("="*60)
    
    # Create larger graph
    G = create_larger_test_graph(n_nodes=50)
    
    # Create complex features and labels
    features = create_complex_node_features(G)
    labels = create_challenging_labels(G, features)
    
    # Import the testing function from the previous script
    # (This assumes you have the previous script available)
    from mp_aware_comprehensive_test import run_pipeline_test
    
    all_results = []
    
    # Test configurations
    coarsening_methods = ['simple', 'cmg']
    propagation_methods = ['naive', 'mp_aware']
    gnn_types = ['gcn', 'graphsage']
    
    # Baseline: Full graph
    print("\nBASELINE: FULL GRAPH")
    print("-" * 30)
    
    A_full = nx.adjacency_matrix(G).tocsr()
    
    # Test all combinations
    for coarsening_method in coarsening_methods:
        for propagation_method in propagation_methods:
            for gnn_type in gnn_types:
                try:
                    result = run_pipeline_test(
                        coarsening_method, propagation_method, gnn_type, G, labels
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"Error in {coarsening_method}+{propagation_method}+{gnn_type}: {e}")
    
    # Analyze results
    print(f"\n{'='*60}")
    print("LARGER GRAPH RESULTS")
    print(f"{'='*60}")
    
    print(f"{'Method':<25} {'Accuracy':<10} {'Time':<8} {'Compression'}")
    print("-" * 55)
    
    for result in all_results:
        method = result['method']
        accuracy = result['accuracy']
        time_taken = result['total_time']
        compression = result['compression_ratio']
        
        print(f"{method:<25} {accuracy:<10.3f} {time_taken:<8.3f} {compression:<10.1f}x")
    
    return all_results

def create_real_world_like_graph(n_nodes=100):
    """Create a graph that mimics real-world network properties"""
    print(f"Creating real-world-like graph with {n_nodes} nodes...")
    
    # Use Barabási-Albert model for scale-free properties
    G = nx.barabasi_albert_graph(n_nodes, 3, seed=42)
    
    # Add some small-world properties
    # Randomly rewire some edges
    edges_to_rewire = int(0.1 * G.number_of_edges())
    edges = list(G.edges())
    
    for _ in range(edges_to_rewire):
        if edges:
            old_edge = edges.pop(np.random.randint(len(edges)))
            G.remove_edge(*old_edge)
            
            # Add new random edge
            u = np.random.randint(n_nodes)
            v = np.random.randint(n_nodes)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
    
    print(f"Created real-world-like graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Average clustering: {nx.average_clustering(G):.3f}")
    
    return G

def main():
    """Run tests on progressively larger graphs"""
    print("PROGRESSIVE SCALE TESTING FOR MP-AWARE BENEFITS")
    print("="*80)
    
    # Test sizes
    test_sizes = [25, 50, 100]
    
    for size in test_sizes:
        print(f"\n{'#'*80}")
        print(f"TESTING GRAPH SIZE: {size} NODES")
        print(f"{'#'*80}")
        
        try:
            if size <= 50:
                G = create_larger_test_graph(n_nodes=size, connection_prob=0.1)
            else:
                G = create_real_world_like_graph(n_nodes=size)
            
            features = create_complex_node_features(G)
            labels = create_challenging_labels(G, features)
            
            # Quick test of just the key methods
            methods_to_test = [
                ('simple', 'naive', 'gcn'),
                ('simple', 'mp_aware', 'gcn'),
                ('cmg', 'naive', 'gcn'),
                ('cmg', 'mp_aware', 'gcn'),
            ]
            
            results = []
            
            for coarsening, propagation, gnn_type in methods_to_test:
                try:
                    from mp_aware_comprehensive_test import run_pipeline_test
                    result = run_pipeline_test(coarsening, propagation, gnn_type, G, labels)
                    results.append(result)
                except Exception as e:
                    print(f"Error testing {coarsening}+{propagation}+{gnn_type}: {e}")
            
            # Quick comparison
            print(f"\nRESULTS FOR {size} NODES:")
            print("-" * 40)
            
            if results:
                best_accuracy = max([r['accuracy'] for r in results])
                
                for result in results:
                    method = result['method']
                    accuracy = result['accuracy']
                    improvement = "BEST" if accuracy == best_accuracy else f"{accuracy/best_accuracy:.2f}"
                    print(f"{method:<20}: {accuracy:.3f} ({improvement})")
            
        except Exception as e:
            print(f"Error testing size {size}: {e}")
    
    print(f"\n{'='*80}")
    print("CONCLUSIONS")
    print(f"{'='*80}")
    print("Look for:")
    print("1. MP-aware methods should show better accuracy on larger graphs")
    print("2. CMG should outperform simple coarsening on complex graphs")
    print("3. Benefits should increase with graph size and complexity")

if __name__ == "__main__":
    main()
