#!/usr/bin/env python3
"""
Script to automatically apply efficiency patches to eliminate redundant cluster extraction
"""

import os
import shutil
import re

def backup_files():
    """Create backups of original files"""
    files_to_backup = [
        'cmg_coarsening_timed.py',
        'graphzoom_timed_mpaware.py'
    ]
    
    for file in files_to_backup:
        if os.path.exists(file):
            backup_name = f"{file}.backup"
            shutil.copy2(file, backup_name)
            print(f"✅ Backed up {file} to {backup_name}")
        else:
            print(f"⚠️  Warning: {file} not found")

def patch_cmg_coarsening_timed():
    """Apply patches to cmg_coarsening_timed.py"""
    file_path = 'cmg_coarsening_timed.py'
    
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Patch 1: Add all_cluster_assignments initialization
    old_pattern1 = r'(projections = \[\]\s+laplacians = \[\]\s+current_laplacian = laplacian\.copy\(\))'
    new_pattern1 = r'projections = []\n    laplacians = []\n    all_cluster_assignments = []  # NEW: Store cluster assignments for each level\n    current_laplacian = laplacian.copy()'
    content = re.sub(old_pattern1, new_pattern1, content)
    
    # Patch 2: Store cluster assignments after CMG clustering
    old_pattern2 = r'(clusters, nc, phi_stats, lambda_crit = cmg_filtered_clustering\(\s+data, k=k, d=d, threshold=threshold\s+\)\s+print\(f"\[CMG\] Found {nc} clusters, λ_critical ≈ {lambda_crit:.4f}"\))'
    new_pattern2 = r'clusters, nc, phi_stats, lambda_crit = cmg_filtered_clustering(\n                data, k=k, d=d, threshold=threshold\n            )\n            print(f"[CMG] Found {nc} clusters, λ_critical ≈ {lambda_crit:.4f}")\n            \n            # NEW: Store the cluster assignments for this level\n            all_cluster_assignments.append(clusters)'
    content = re.sub(old_pattern2, new_pattern2, content, flags=re.MULTILINE)
    
    # Patch 3: Handle fallback case
    old_pattern3 = r'(from utils import smooth_filter, spec_coarsen\s+filter_ = smooth_filter\(current_laplacian, 0\.1\)\s+current_laplacian, mapping = spec_coarsen\(filter_, current_laplacian\)\s+projections\.append\(mapping\)\s+continue)'
    new_pattern3 = r'from utils import smooth_filter, spec_coarsen\n            filter_ = smooth_filter(current_laplacian, 0.1)\n            current_laplacian, mapping = spec_coarsen(filter_, current_laplacian)\n            projections.append(mapping)\n            \n            # NEW: For fallback, create dummy cluster assignments\n            # Extract from the mapping matrix\n            n_nodes, n_clusters = mapping.shape\n            fallback_assignments = [-1] * n_nodes\n            for node_id in range(n_nodes):\n                for cluster_id in range(n_clusters):\n                    if mapping[node_id, cluster_id] > 0:\n                        fallback_assignments[node_id] = cluster_id\n                        break\n            all_cluster_assignments.append(fallback_assignments)\n            continue'
    content = re.sub(old_pattern3, new_pattern3, content, flags=re.MULTILINE)
    
    # Patch 4: Update return statement
    old_pattern4 = r'return G, projections, laplacians, level'
    new_pattern4 = r'return G, projections, laplacians, level, all_cluster_assignments'
    content = re.sub(old_pattern4, new_pattern4, content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Applied patches to {file_path}")
    return True

def patch_graphzoom_timed_mpaware():
    """Apply patches to graphzoom_timed_mpaware.py"""
    file_path = 'graphzoom_timed_mpaware.py'
    
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Patch 1: Add helper function before main()
    helper_function = '''
def cluster_assignments_to_clusters(cluster_assignments):
    """
    Convert cluster assignments array to list of cluster node lists.
    More efficient than scanning projection matrix.
    
    Args:
        cluster_assignments: Array like [0, 0, 1, 1, 2, 2] indicating cluster for each node
        
    Returns:
        clusters: List of lists like [[0, 1], [2, 3], [4, 5]]
    """
    if len(cluster_assignments) == 0:
        return []
        
    clusters = []
    num_clusters = max(cluster_assignments) + 1
    
    for cluster_id in range(num_clusters):
        cluster = [node_id for node_id, cid in enumerate(cluster_assignments) if cid == cluster_id]
        if cluster:  # Only add non-empty clusters
            clusters.append(cluster)
    
    return clusters

'''
    
    # Insert helper function before main()
    main_pattern = r'(def main\(\):)'
    content = re.sub(main_pattern, helper_function + r'\1', content)
    
    # Patch 2: Update CMG coarsening call
    old_pattern2 = r'G, projections, laplacians, level = cmg_coarse\(\s+laplacian, args\.level, args\.cmg_k, args\.cmg_d, args\.cmg_threshold\s+\)'
    new_pattern2 = r'G, projections, laplacians, level, all_cluster_assignments = cmg_coarse(\n            laplacian, args.level, args.cmg_k, args.cmg_d, args.cmg_threshold\n        )'
    content = re.sub(old_pattern2, new_pattern2, content, flags=re.MULTILINE)
    
    # Patch 3: Replace cluster extraction in true_coarsened_graphsage section
    # This is more complex, so we'll do a simpler replacement
    old_cluster_extraction = r'# Extract clusters from CMG projections \(using ORIGINAL indices\)\s+print\("\\n📊 EXTRACTING CLUSTERS FROM CMG\.\.\."\)\s+clusters = \[\]\s+n_nodes, n_clusters = projections\[0\]\.shape\s+\s+for cluster_id in range\(n_clusters\):\s+cluster = \[\]\s+for node_id in range\(n_nodes\):\s+if projections\[0\]\[node_id, cluster_id\] > 0:\s+cluster\.append\(node_id\)\s+if cluster:\s+clusters\.append\(cluster\)\s+\s+print\(f"✅ Extracted {len\(clusters\)} clusters from CMG"\)'
    
    new_cluster_extraction = '''# EFFICIENT: Extract clusters from CMG assignments directly
        print("\\n📊 EXTRACTING CLUSTERS FROM CMG...")
        
        if args.coarse == "cmg":
            # NEW: Use cluster assignments directly (EFFICIENT!)
            print("✅ Using CMG cluster assignments directly (efficient method)")
            clusters = cluster_assignments_to_clusters(all_cluster_assignments[0])  # Use level 0
            print(f"✅ Extracted {len(clusters)} clusters directly from CMG assignments")
            
        else:
            # FALLBACK: For non-CMG methods, use projection matrix method
            print("⚠️  Using projection matrix method (fallback for non-CMG)")
            clusters = []
            n_nodes, n_clusters = projections[0].shape
            
            for cluster_id in range(n_clusters):
                cluster = []
                for node_id in range(n_nodes):
                    if projections[0][node_id, cluster_id] > 0:
                        cluster.append(node_id)
                if cluster:
                    clusters.append(cluster)
            print(f"✅ Extracted {len(clusters)} clusters from projection matrix")'''
    
    content = re.sub(old_cluster_extraction, new_cluster_extraction, content, flags=re.MULTILINE | re.DOTALL)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Applied patches to {file_path}")
    return True

def verify_patches():
    """Verify that patches were applied correctly"""
    print("\n🔍 VERIFYING PATCHES...")
    
    # Check cmg_coarsening_timed.py
    with open('cmg_coarsening_timed.py', 'r') as f:
        cmg_content = f.read()
    
    cmg_checks = [
        'all_cluster_assignments = []' in cmg_content,
        'all_cluster_assignments.append(clusters)' in cmg_content,
        'return G, projections, laplacians, level, all_cluster_assignments' in cmg_content
    ]
    
    # Check graphzoom_timed_mpaware.py
    with open('graphzoom_timed_mpaware.py', 'r') as f:
        gz_content = f.read()
    
    gz_checks = [
        'def cluster_assignments_to_clusters' in gz_content,
        'all_cluster_assignments = cmg_coarse' in gz_content,
        'cluster_assignments_to_clusters(all_cluster_assignments[0])' in gz_content
    ]
    
    all_checks = cmg_checks + gz_checks
    passed_checks = sum(all_checks)
    total_checks = len(all_checks)
    
    print(f"CMG file checks: {sum(cmg_checks)}/3 passed")
    print(f"GraphZoom file checks: {sum(gz_checks)}/3 passed")
    print(f"Total: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("✅ ALL PATCHES APPLIED SUCCESSFULLY!")
        return True
    else:
        print("⚠️  Some patches may not have applied correctly")
        return False

def main():
    print("APPLYING EFFICIENCY PATCHES")
    print("="*50)
    
    # Check if we're in the right directory
    if not os.path.exists('cmg_coarsening_timed.py'):
        print("❌ Error: Not in GraphZoom directory. Please run from /home/mohammad/GraphZoom/graphzoom/")
        return
    
    # Backup files
    print("Step 1: Creating backups...")
    backup_files()
    
    # Apply patches
    print("\nStep 2: Applying patches...")
    success1 = patch_cmg_coarsening_timed()
    success2 = patch_graphzoom_timed_mpaware()
    
    if success1 and success2:
        print("\nStep 3: Verifying patches...")
        if verify_patches():
            print("\n🎉 SUCCESS: All patches applied successfully!")
            print("\nTo test the changes, run:")
            print("python graphzoom_timed_mpaware.py --coarse cmg --embed_method true_coarsened_graphsage --dataset cora")
            print("\nYou should see: '✅ Using CMG cluster assignments directly (efficient method)'")
        else:
            print("\n⚠️  Warning: Verification failed. Check the files manually.")
    else:
        print("\n❌ Error: Some patches failed to apply")

if __name__ == "__main__":
    main()
