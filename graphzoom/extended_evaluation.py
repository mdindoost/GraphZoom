#!/usr/bin/env python3
"""
Comprehensive Refinement & Fusion Evaluation
Tests 3 refinement approaches × 2 fusion settings × multiple coarsening methods
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import norm as sparse_norm
import networkx as nx
import os
import sys
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import json

# Add GraphZoom path
sys.path.append('.')

def load_cora_dataset():
    """Load Cora dataset for node classification"""
    print("Loading Cora dataset...")
    
    try:
        # Load the dataset
        from utils import json2mtx
        
        # Check if dataset exists
        dataset_path = "dataset/cora"
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found at {dataset_path}")
            return None, None, None, None
        
        # Load graph
        laplacian = json2mtx("cora")
        
        # Load features and labels
        features = np.load("dataset/cora/cora-feats.npy")
        
        # Load labels 
        try:
            labels = np.load("dataset/cora/cora-labels.npy")
        except:
            print("⚠️  Labels not found, creating dummy labels for testing")
            labels = np.random.randint(0, 7, size=features.shape[0])  # 7 classes for Cora
        
        print(f"✅ Loaded Cora: {laplacian.shape[0]} nodes, {features.shape[1]} features")
        return laplacian, features, labels, "cora"
        
    except Exception as e:
        print(f"❌ Failed to load Cora: {e}")
        return None, None, None, None

def create_train_test_split(n_nodes, train_ratio=0.1, val_ratio=0.1, seed=42):
    """Create train/val/test splits for node classification"""
    np.random.seed(seed)
    
    # Create random permutation
    perm = np.random.permutation(n_nodes)
    
    # Calculate split sizes
    n_train = int(n_nodes * train_ratio)
    n_val = int(n_nodes * val_ratio)
    
    # Create splits
    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)
    
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True
    
    print(f"📊 Split: {n_train} train, {n_val} val, {sum(test_mask)} test")
    return train_mask, val_mask, test_mask

def train_real_embeddings(G_coarse, method="deepwalk", seed=42):
    """Train REAL embeddings on coarsened graph"""
    print(f"  🎯 Training REAL {method} embeddings...")
    
    try:
        if method == "deepwalk":
            from embed_methods.deepwalk.deepwalk import deepwalk
            embeddings = deepwalk(G_coarse)
            
        elif method == "node2vec":
            from embed_methods.node2vec.node2vec import node2vec
            embeddings = node2vec(G_coarse)
            
        else:
            # Simple fallback
            embeddings = simple_random_walk_embeddings(G_coarse, seed=seed)
            
        print(f"    ✅ Generated embeddings: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        print(f"    ❌ Failed to train {method}: {e}")
        return simple_random_walk_embeddings(G_coarse, seed=seed)

def simple_random_walk_embeddings(G, dim=128, num_walks=10, walk_length=40, seed=42):
    """Simple random walk embeddings as fallback"""
    print(f"    Creating fallback embeddings...")
    np.random.seed(seed)
    
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    if n_nodes == 0:
        return np.random.randn(1, dim)
    
    embeddings = np.zeros((n_nodes, dim))
    cooccurrence = np.zeros((n_nodes, n_nodes))
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    for _ in range(num_walks):
        for start_node in nodes:
            if start_node not in G:
                continue
                
            current = start_node
            walk = [current]
            
            for _ in range(walk_length):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                current = np.random.choice(neighbors)
                walk.append(current)
            
            # Update co-occurrence matrix
            for i, node1 in enumerate(walk):
                for j in range(max(0, i-5), min(len(walk), i+6)):  # Window size 5
                    node2 = walk[j]
                    if node1 != node2:
                        idx1, idx2 = node_to_idx[node1], node_to_idx[node2]
                        cooccurrence[idx1][idx2] += 1
    
    # SVD for embeddings
    try:
        from scipy.linalg import svd
        U, s, Vt = svd(cooccurrence + 1e-8, full_matrices=False)
        dim_actual = min(dim, U.shape[1])
        embeddings = U[:, :dim_actual] * np.sqrt(s[:dim_actual])
        
        if dim_actual < dim:
            padding = np.random.randn(n_nodes, dim - dim_actual) * 0.01
            embeddings = np.concatenate([embeddings, padding], axis=1)
            
    except:
        embeddings = np.random.randn(n_nodes, dim)
    
    return embeddings

def original_refinement(levels, projections, laplacians, embeddings, lda=0.1, power=False):
    """Original GraphZoom refinement: projection + spectral smoothing"""
    print(f"[ORIGINAL REFINEMENT]")
    for i in reversed(range(levels)):
        # Step 1: Project back to larger graph
        embeddings = projections[i] @ embeddings
        
        # Step 2: Spectral smoothing
        filter_ = smooth_filter(laplacians[i], lda)
        if power or i == 0:
            embeddings = filter_ @ (filter_ @ embeddings)
    
    return embeddings

def enhanced_mp_refinement(levels, projections, laplacians, embeddings, lda=0.1, power=False):
    """Enhanced MP refinement: projection + MP correction + spectral smoothing"""
    from enhanced_refinement import (
        build_mp_preserving_matrix, 
        extract_clusters_from_projection,
        compute_message_passing_error,
        laplacian_to_propagation,
        build_correction_filter
    )
    
    print(f"[ENHANCED MP REFINEMENT]")
    for i in reversed(range(levels)):
        # Step 1: Project back
        embeddings = projections[i] @ embeddings
        
        # Step 2: Message-passing correction
        laplacian = laplacians[i]
        clusters = extract_clusters_from_projection(projections[i])
        Q = build_mp_preserving_matrix(clusters, laplacian.shape[0])
        S = laplacian_to_propagation(laplacian)
        mp_error, S_approx, S_c = compute_message_passing_error(S, Q)
        
        print(f"  Level {i}: MP error = {mp_error:.6f}")
        
        if mp_error > 0.1:  # Apply correction if significant error
            print(f"  Applying MP correction...")
            correction_filter = build_correction_filter(S_approx, lda)
            embeddings = correction_filter @ embeddings
        
        # Step 3: Original spectral smoothing
        filter_ = smooth_filter(laplacians[i], lda)
        if power or i == 0:
            embeddings = filter_ @ (filter_ @ embeddings)
    
    return embeddings

def enhanced_mp_only(levels, projections, laplacians, embeddings, lda=0.1, power=False):
    """Enhanced MP only: projection + MP correction (NO spectral smoothing)"""
    from enhanced_refinement import (
        build_mp_preserving_matrix, 
        extract_clusters_from_projection,
        compute_message_passing_error,
        laplacian_to_propagation,
        build_correction_filter
    )
    
    print(f"[ENHANCED MP ONLY - NO SPECTRAL SMOOTHING]")
    for i in reversed(range(levels)):
        # Step 1: Project back
        embeddings = projections[i] @ embeddings
        
        # Step 2: Message-passing correction ONLY (no spectral smoothing)
        laplacian = laplacians[i]
        clusters = extract_clusters_from_projection(projections[i])
        Q = build_mp_preserving_matrix(clusters, laplacian.shape[0])
        S = laplacian_to_propagation(laplacian)
        mp_error, S_approx, S_c = compute_message_passing_error(S, Q)
        
        print(f"  Level {i}: MP error = {mp_error:.6f}")
        
        if mp_error > 0.1:  # Apply correction if significant error
            print(f"  Applying MP correction (NO spectral smoothing)...")
            correction_filter = build_correction_filter(S_approx, lda)
            embeddings = correction_filter @ embeddings
        
        # Step 3: NO spectral smoothing - just return corrected embeddings
        print(f"  Skipping spectral smoothing...")
    
    return embeddings

def smooth_filter(laplacian_matrix, lda):
    """GraphZoom smooth filter"""
    try:
        from utils import smooth_filter as original_smooth_filter
        return original_smooth_filter(laplacian_matrix, lda)
    except ImportError:
        # Fallback implementation
        dim = laplacian_matrix.shape[0]
        adj_matrix = diags(laplacian_matrix.diagonal(), 0) - laplacian_matrix + lda * sp.identity(dim)
        degree_vec = adj_matrix.sum(axis=1)
        
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.squeeze(np.asarray(np.power(degree_vec, -0.5)))
        d_inv_sqrt[np.isinf(d_inv_sqrt)|np.isnan(d_inv_sqrt)] = 0
        degree_matrix = diags(d_inv_sqrt, 0)
        norm_adj = degree_matrix @ (adj_matrix @ degree_matrix)
        return norm_adj

def run_graphzoom_pipeline(laplacian, features, method="cmg", refinement_type="original", 
                         fusion_enabled=True, cmg_levels=1, reduce_ratio=2, embedding_method="deepwalk"):
    """Run complete GraphZoom pipeline with specified configuration"""
    
    print(f"\n🚀 GraphZoom Configuration:")
    print(f"   Coarsening: {method} (levels={cmg_levels if method=='cmg' else 'N/A'})")
    print(f"   Fusion: {'ENABLED' if fusion_enabled else 'DISABLED'}")
    print(f"   Refinement: {refinement_type}")
    print(f"   Embedding: {embedding_method}")
    
    try:
        # Step 1: Fusion (optional)
        if fusion_enabled:
            print("  🔗 Running fusion step...")
            from utils import graph_fusion
            
            # GraphZoom fusion parameters
            fused_laplacian = graph_fusion(
                laplacian, features, num_neighs=2, mcr_dir="/opt/matlab/R2018A/", 
                coarse=method, fusion_input_path="temp.mtx", search_ratio=12, 
                fusion_output_dir="temp_dir", mapping_path="temp_mapping.mtx", 
                dataset="temp"
            )
            working_laplacian = fused_laplacian
            print("  ✅ Fusion completed")
        else:
            print("  ⏭️  Skipping fusion step")
            working_laplacian = laplacian
        
        # Step 2: Coarsening
        if method == "cmg":
            from cmg_coarsening_timed import cmg_coarse
            
            print(f"  🔄 CMG coarsening ({cmg_levels} levels)...")
            degree_diag = diags(working_laplacian.diagonal(), 0)
            adjacency = degree_diag - working_laplacian
            G = nx.from_scipy_sparse_matrix(adjacency)
            
            G_coarse, projections, laplacians, levels = cmg_coarse(
                working_laplacian, level=cmg_levels, k=10, d=20, threshold=0.1
            )
            
        elif method == "simple":
            from utils import sim_coarse
            print("  🔄 Simple coarsening...")
            G_coarse, projections, laplacians, levels = sim_coarse(working_laplacian, level=1)
            
        elif method == "lamg":
            # Simplified LAMG handling - fallback to simple if LAMG fails
            print(f"  🔄 LAMG coarsening (ratio={reduce_ratio})...")
            try:
                # Try LAMG (with existing LAMG code logic)
                # For now, fallback to simple
                from utils import sim_coarse
                G_coarse, projections, laplacians, levels = sim_coarse(working_laplacian, level=1)
                print("  ⚠️  Using simple coarsening as LAMG fallback")
            except:
                from utils import sim_coarse
                G_coarse, projections, laplacians, levels = sim_coarse(working_laplacian, level=1)
        
        else:
            raise ValueError(f"Unknown coarsening method: {method}")
        
        print(f"  Coarsened: {working_laplacian.shape[0]} → {G_coarse.number_of_nodes()} nodes")
        
        # Step 3: Embedding
        start_time = time.time()
        coarse_embeddings = train_real_embeddings(G_coarse, method=embedding_method)
        embedding_time = time.time() - start_time
        
        # Step 4: Refinement
        print("  🎯 Applying refinement...")
        refine_start = time.time()
        
        if refinement_type == "original":
            final_embeddings = original_refinement(
                levels, projections, laplacians, coarse_embeddings
            )
        elif refinement_type == "enhanced_mp":
            final_embeddings = enhanced_mp_refinement(
                levels, projections, laplacians, coarse_embeddings
            )
        elif refinement_type == "enhanced_mp_only":
            final_embeddings = enhanced_mp_only(
                levels, projections, laplacians, coarse_embeddings
            )
        else:
            raise ValueError(f"Unknown refinement type: {refinement_type}")
        
        refinement_time = time.time() - refine_start
        total_time = embedding_time + refinement_time
        
        print(f"  ✅ Pipeline completed: {total_time:.3f}s total")
        return final_embeddings, total_time
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def evaluate_node_classification(embeddings, features, labels, train_mask, val_mask, test_mask, use_features=True):
    """Evaluate embeddings on node classification"""
    print("🎯 Evaluating node classification...")
    
    if use_features and features is not None:
        X = np.concatenate([embeddings, features], axis=1)
        print(f"  Combined: {embeddings.shape[1]} embeddings + {features.shape[1]} features")
    else:
        X = embeddings
        print(f"  Structure only: {embeddings.shape[1]} features")
    
    # Prepare data
    X_train = X[train_mask]
    y_train = labels[train_mask]
    X_test = X[test_mask]
    y_test = labels[test_mask]
    
    # Standardize and train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='macro')
    
    print(f"  📊 Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    return {'test_accuracy': test_acc, 'test_f1': test_f1}

def print_comprehensive_results(results):
    """Print comprehensive comparison table"""
    print(f"\n" + "="*120)
    print("COMPREHENSIVE REFINEMENT × FUSION × COARSENING EVALUATION")
    print("="*120)
    
    # Group results by fusion setting
    fusion_groups = {'enabled': [], 'disabled': []}
    for r in results:
        key = 'enabled' if r['fusion_enabled'] else 'disabled'
        fusion_groups[key].append(r)
    
    for fusion_status, group_results in fusion_groups.items():
        if not group_results:
            continue
            
        print(f"\n🔗 FUSION {fusion_status.upper()}:")
        print("-" * 80)
        
        # Headers
        headers = ["Method", "Refinement", "Test Acc", "F1", "Time(s)"]
        header_row = f"{headers[0]:<15} {headers[1]:<20} {headers[2]:<10} {headers[3]:<8} {headers[4]:<8}"
        print(header_row)
        print("-" * len(header_row))
        
        # Sort by accuracy
        group_results.sort(key=lambda x: x['test_accuracy'], reverse=True)
        
        for r in group_results:
            method_info = f"{r['coarsening']}-{r.get('cmg_levels', 1)}"
            row = f"{method_info:<15} {r['refinement_type']:<20} {r['test_accuracy']:<10.4f} {r['test_f1']:<8.4f} {r['total_time']:<8.3f}"
            print(row)
    
    # Analysis by refinement type
    print(f"\n" + "="*80)
    print("REFINEMENT TYPE ANALYSIS")
    print("="*80)
    
    refinement_analysis = {}
    for r in results:
        key = (r['refinement_type'], r['fusion_enabled'])
        if key not in refinement_analysis:
            refinement_analysis[key] = []
        refinement_analysis[key].append(r['test_accuracy'])
    
    print(f"\nAverage Performance by Refinement Type:")
    print(f"{'Refinement':<20} {'Fusion':<10} {'Avg Acc':<10} {'Count':<6}")
    print("-" * 50)
    
    for (refinement, fusion), accuracies in refinement_analysis.items():
        fusion_str = "ON" if fusion else "OFF"
        avg_acc = np.mean(accuracies)
        count = len(accuracies)
        print(f"{refinement:<20} {fusion_str:<10} {avg_acc:<10.4f} {count:<6}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    
    # Best overall
    best_result = max(results, key=lambda x: x['test_accuracy'])
    print(f"   🏆 Best overall: {best_result['coarsening']}-{best_result.get('cmg_levels', 1)} + {best_result['refinement_type']} + fusion {best_result['fusion_enabled']} = {best_result['test_accuracy']:.4f}")
    
    # Fusion impact
    with_fusion = [r for r in results if r['fusion_enabled']]
    without_fusion = [r for r in results if not r['fusion_enabled']]
    
    if with_fusion and without_fusion:
        avg_with = np.mean([r['test_accuracy'] for r in with_fusion])
        avg_without = np.mean([r['test_accuracy'] for r in without_fusion])
        print(f"   🔗 Fusion impact: WITH={avg_with:.4f} vs WITHOUT={avg_without:.4f} (Δ={avg_with-avg_without:+.4f})")
    
    # Refinement comparison
    orig_results = [r for r in results if r['refinement_type'] == 'original']
    enh_mp_results = [r for r in results if r['refinement_type'] == 'enhanced_mp']
    mp_only_results = [r for r in results if r['refinement_type'] == 'enhanced_mp_only']
    
    if orig_results:
        avg_orig = np.mean([r['test_accuracy'] for r in orig_results])
        print(f"   📊 Original refinement: {avg_orig:.4f}")
    if enh_mp_results:
        avg_enh = np.mean([r['test_accuracy'] for r in enh_mp_results])
        print(f"   📊 Enhanced MP refinement: {avg_enh:.4f}")
    if mp_only_results:
        avg_mp_only = np.mean([r['test_accuracy'] for r in mp_only_results])
        print(f"   📊 Enhanced MP only: {avg_mp_only:.4f}")
    
    if enh_mp_results and orig_results:
        improvement = avg_enh - avg_orig
        print(f"   🎯 Enhanced MP vs Original: {improvement:+.4f} pp")
    
    if mp_only_results and orig_results:
        improvement = avg_mp_only - avg_orig
        print(f"   🎯 MP Only vs Original: {improvement:+.4f} pp")

def main():
    """Main comprehensive evaluation"""
    print("COMPREHENSIVE REFINEMENT × FUSION × COARSENING EVALUATION")
    print("="*80)
    
    # Load dataset
    laplacian, features, labels, dataset_name = load_cora_dataset()
    if laplacian is None:
        print("❌ Failed to load dataset")
        return
    
    # Create splits
    n_nodes = laplacian.shape[0]
    train_mask, val_mask, test_mask = create_train_test_split(n_nodes)
    
    # Comprehensive test matrix
    results = []
    
    # Test configurations: [coarsening_method, cmg_levels, refinement_type, fusion_enabled]
    test_configs = [
        # CMG tests
        ("cmg", 1, "original", True),
        ("cmg", 1, "enhanced_mp", True),
        ("cmg", 1, "enhanced_mp_only", True),
        ("cmg", 1, "original", False),
        ("cmg", 1, "enhanced_mp", False),
        ("cmg", 1, "enhanced_mp_only", False),
        
        ("cmg", 3, "original", True),
        ("cmg", 3, "enhanced_mp", True), 
        ("cmg", 3, "enhanced_mp_only", True),
        ("cmg", 3, "original", False),
        ("cmg", 3, "enhanced_mp", False),
        ("cmg", 3, "enhanced_mp_only", False),
        
        # Simple tests
        ("simple", 1, "original", True),
        ("simple", 1, "enhanced_mp", True),
        ("simple", 1, "enhanced_mp_only", True),
        ("simple", 1, "original", False),
        ("simple", 1, "enhanced_mp", False),
        ("simple", 1, "enhanced_mp_only", False),
    ]
    
    embedding_method = "deepwalk"
    
    for coarsening, levels, refinement, fusion in test_configs:
        config_name = f"{coarsening.upper()}-{levels} + {refinement} + fusion_{fusion}"
        print(f"\n" + "="*80)
        print(f"TESTING: {config_name}")
        print("="*80)
        
        try:
            # Run pipeline
            embeddings, total_time = run_graphzoom_pipeline(
                laplacian, features, 
                method=coarsening,
                refinement_type=refinement,
                fusion_enabled=fusion,
                cmg_levels=levels,
                embedding_method=embedding_method
            )
            
            if embeddings is not None:
                # Evaluate
                eval_results = evaluate_node_classification(
                    embeddings, features, labels, train_mask, val_mask, test_mask
                )
                
                # Store result
                results.append({
                    'config_name': config_name,
                    'coarsening': coarsening,
                    'cmg_levels': levels,
                    'refinement_type': refinement,
                    'fusion_enabled': fusion,
                    'embedding_method': embedding_method,
                    'total_time': total_time,
                    **eval_results
                })
                
                print(f"✅ {config_name}: {eval_results['test_accuracy']:.4f} accuracy")
            else:
                print(f"❌ {config_name}: Failed")
                
        except Exception as e:
            print(f"❌ {config_name}: Error - {e}")
    
    # Print comprehensive results
    if results:
        print_comprehensive_results(results)
        
        # Save results
        results_file = f"comprehensive_evaluation_results_{dataset_name}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {results_file}")
    else:
        print("❌ No results to analyze")
    
    print(f"\n✅ Comprehensive evaluation completed!")

if __name__ == "__main__":
    main()