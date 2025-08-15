#!/usr/bin/env python3
"""
Verification script to compare cluster extraction methods
"""

import numpy as np
import scipy.sparse as sp
from collections import defaultdict

def extract_clusters_from_projection_matrix(projection_matrix):
    """
    CURRENT METHOD: Extract clusters from projection matrix (what you're doing now)
    """
    clusters = []
    n_nodes, n_clusters = projection_matrix.shape
    
    print(f"[CURRENT] Projection matrix shape: {projection_matrix.shape}")
    print(f"[CURRENT] Processing {n_clusters} clusters for {n_nodes} nodes")
    
    for cluster_id in range(n_clusters):
        cluster = []
        for node_id in range(n_nodes):
            if projection_matrix[node_id, cluster_id] > 0:
                cluster.append(node_id)
        if cluster:  # Only add non-empty clusters
            clusters.append(cluster)
            
    print(f"[CURRENT] Found {len(clusters)} non-empty clusters")
    return clusters

def extract_clusters_from_assignments(cluster_assignments):
    """
    PROPOSED METHOD: Extract clusters from original assignments array
    """
    print(f"[PROPOSED] Cluster assignments shape: {len(cluster_assignments)}")
    print(f"[PROPOSED] Assignments: {cluster_assignments}")
    
    clusters = []
    num_clusters = max(cluster_assignments) + 1
    
    print(f"[PROPOSED] Expected {num_clusters} clusters")
    
    for cluster_id in range(num_clusters):
        cluster = [node_id for node_id, cid in enumerate(cluster_assignments) if cid == cluster_id]
        if cluster:  # Only add non-empty clusters
            clusters.append(cluster)
    
    print(f"[PROPOSED] Found {len(clusters)} non-empty clusters")
    return clusters

def create_projection_matrix_from_assignments(cluster_assignments, num_nodes):
    """
    Create projection matrix from cluster assignments (how CMG should do it)
    """
    num_clusters = max(cluster_assignments) + 1
    projection = sp.lil_matrix((num_nodes, num_clusters))
    
    for node_id, cluster_id in enumerate(cluster_assignments):
        projection[node_id, cluster_id] = 1.0
    
    return projection.tocsr()

def compare_cluster_extraction_methods(cluster_assignments, num_nodes):
    """
    Compare both methods and verify they give identical results
    """
    print("="*60)
    print("CLUSTER EXTRACTION METHOD COMPARISON")
    print("="*60)
    
    # Create projection matrix from assignments (simulate CMG output)
    projection_matrix = create_projection_matrix_from_assignments(cluster_assignments, num_nodes)
    
    print(f"Test input:")
    print(f"  Cluster assignments: {cluster_assignments}")
    print(f"  Number of nodes: {num_nodes}")
    print(f"  Generated projection matrix shape: {projection_matrix.shape}")
    print(f"  Projection matrix nnz: {projection_matrix.nnz}")
    
    # Method 1: Current method (from projection matrix)
    print(f"\n--- METHOD 1: FROM PROJECTION MATRIX ---")
    clusters_from_projection = extract_clusters_from_projection_matrix(projection_matrix)
    
    # Method 2: Proposed method (from assignments)
    print(f"\n--- METHOD 2: FROM ASSIGNMENTS ---")
    clusters_from_assignments = extract_clusters_from_assignments(cluster_assignments)
    
    # Compare results
    print(f"\n--- COMPARISON ---")
    print(f"Method 1 clusters: {clusters_from_projection}")
    print(f"Method 2 clusters: {clusters_from_assignments}")
    
    # Detailed comparison
    methods_match = True
    
    if len(clusters_from_projection) != len(clusters_from_assignments):
        print(f"❌ MISMATCH: Different number of clusters!")
        print(f"   Method 1: {len(clusters_from_projection)} clusters")
        print(f"   Method 2: {len(clusters_from_assignments)} clusters")
        methods_match = False
    else:
        print(f"✅ Same number of clusters: {len(clusters_from_projection)}")
    
    # Compare each cluster (need to sort since order might differ)
    for i in range(min(len(clusters_from_projection), len(clusters_from_assignments))):
        cluster1 = sorted(clusters_from_projection[i])
        cluster2 = sorted(clusters_from_assignments[i])
        
        if cluster1 != cluster2:
            print(f"❌ MISMATCH in cluster {i}:")
            print(f"   Method 1: {cluster1}")
            print(f"   Method 2: {cluster2}")
            methods_match = False
        else:
            print(f"✅ Cluster {i} matches: {cluster1}")
    
    # Final verdict
    print(f"\n--- FINAL VERDICT ---")
    if methods_match:
        print("🎉 SUCCESS: Both methods produce IDENTICAL results!")
        print("✅ Safe to replace current method with proposed method")
    else:
        print("⚠️  WARNING: Methods produce DIFFERENT results!")
        print("❌ Need to investigate discrepancies before changing")
    
    return methods_match, clusters_from_projection, clusters_from_assignments

def test_edge_cases():
    """
    Test edge cases to ensure robustness
    """
    print(f"\n" + "="*60)
    print("EDGE CASE TESTING")
    print("="*60)
    
    test_cases = [
        # (cluster_assignments, num_nodes, description)
        ([0, 0, 1, 1, 2, 2], 6, "Normal case: 3 clusters, 6 nodes"),
        ([0, 1, 0, 1], 4, "Alternating clusters"),
        ([0, 0, 0, 0], 4, "Single cluster"),
        ([0, 1, 2, 3], 4, "Each node in different cluster"),
        ([1, 1, 3, 3], 4, "Missing cluster 0 and 2"),
        ([0, 0, 1, 1, 1], 5, "Unbalanced clusters"),
    ]
    
    all_passed = True
    
    for assignments, num_nodes, description in test_cases:
        print(f"\nTest: {description}")
        try:
            match, _, _ = compare_cluster_extraction_methods(assignments, num_nodes)
            if match:
                print(f"✅ PASSED: {description}")
            else:
                print(f"❌ FAILED: {description}")
                all_passed = False
        except Exception as e:
            print(f"❌ ERROR in {description}: {e}")
            all_passed = False
    
    print(f"\n--- EDGE CASE SUMMARY ---")
    if all_passed:
        print("🎉 ALL EDGE CASES PASSED!")
    else:
        print("⚠️  Some edge cases failed - need investigation")
    
    return all_passed

def test_with_real_cmg_data():
    """
    Test with realistic CMG-style data
    """
    print(f"\n" + "="*60)
    print("REALISTIC CMG DATA TESTING")
    print("="*60)
    
    # Simulate typical CMG clustering results
    realistic_cases = [
        # Small graph like your synthetic test
        ([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], 10, "Small graph: 3 clusters"),
        
        # Medium graph with unbalanced clusters (common in CMG)
        ([0] * 5 + [1] * 8 + [2] * 3 + [3] * 7, 23, "Medium graph: unbalanced clusters"),
        
        # Large-ish simulation
        ([i // 10 for i in range(100)], 100, "Large simulation: 10 clusters of 10 nodes"),
    ]
    
    all_passed = True
    
    for assignments, num_nodes, description in realistic_cases:
        print(f"\nRealistic test: {description}")
        try:
            match, clusters1, clusters2 = compare_cluster_extraction_methods(assignments, num_nodes)
            
            # Additional checks for realistic data
            total_nodes_1 = sum(len(cluster) for cluster in clusters1)
            total_nodes_2 = sum(len(cluster) for cluster in clusters2)
            
            print(f"  Method 1 covers {total_nodes_1}/{num_nodes} nodes")
            print(f"  Method 2 covers {total_nodes_2}/{num_nodes} nodes")
            
            if total_nodes_1 != num_nodes or total_nodes_2 != num_nodes:
                print(f"❌ Coverage issue: not all nodes covered!")
                all_passed = False
                match = False
            
            if match:
                print(f"✅ PASSED: {description}")
            else:
                print(f"❌ FAILED: {description}")
                all_passed = False
                
        except Exception as e:
            print(f"❌ ERROR in {description}: {e}")
            all_passed = False
    
    return all_passed

def main():
    """
    Run comprehensive verification
    """
    print("CLUSTER EXTRACTION VERIFICATION SUITE")
    print("="*70)
    
    # Test 1: Basic functionality
    print("TEST 1: Basic functionality")
    basic_passed, _, _ = compare_cluster_extraction_methods([0, 0, 1, 1, 2, 2], 6)
    
    # Test 2: Edge cases
    print(f"\nTEST 2: Edge cases")
    edge_passed = test_edge_cases()
    
    # Test 3: Realistic CMG data
    print(f"\nTEST 3: Realistic CMG data")
    realistic_passed = test_with_real_cmg_data()
    
    # Final summary
    print(f"\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    tests = [
        ("Basic functionality", basic_passed),
        ("Edge cases", edge_passed), 
        ("Realistic CMG data", realistic_passed)
    ]
    
    all_tests_passed = True
    for test_name, passed in tests:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
        all_tests_passed = all_tests_passed and passed
    
    print(f"\n--- FINAL VERDICT ---")
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ SAFE TO REPLACE: Current method with proposed method")
        print("💡 Recommendation: Make the change - it's more efficient and equivalent")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("❌ DO NOT CHANGE: Investigate failures first")
        print("💡 Recommendation: Debug the issues before making changes")
    
    return all_tests_passed

if __name__ == "__main__":
    main()
