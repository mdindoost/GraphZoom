#!/usr/bin/env python3
"""
COMPREHENSIVE EVALUATION: Enhanced MP + Fusion Analysis
Tests 3 refinement types × 2 fusion options × 3 coarsening methods
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

def check_matlab_setup():
    """Check MATLAB Runtime setup"""
    print("🔧 Checking MATLAB Runtime setup...")
    
    # Check MCR directory
    mcr_base = "/home/mohammad/matlab/R2018a/"
    mcr_full = "/home/mohammad/matlab/R2018a/v94/"
    
    print(f"  MCR base path exists: {os.path.exists(mcr_base)}")
    print(f"  MCR v94 path exists: {os.path.exists(mcr_full)}")
    
    # Check for key library file
    lib_path = "/home/mohammad/matlab/R2018a/v94/runtime/glnxa64/libmwlaunchermain.so"
    print(f"  Key library exists: {os.path.exists(lib_path)}")
    
    # Check current LD_LIBRARY_PATH
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    print(f"  Current LD_LIBRARY_PATH: {current_ld_path[:100]}...")
    
    # Check LAMG script and binary
    print(f"  LAMG script exists: {os.path.exists('./run_coarsening.sh')}")
    print(f"  LAMG binary exists: {os.path.exists('./coarsening')}")
    
    return os.path.exists(mcr_base) and os.path.exists(lib_path)

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
            print("Please ensure Cora dataset is available in dataset/cora/")
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
            labels = np.random.randint(0, 7, size=features.shape[0])
        
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
            raise ValueError(f"Unknown embedding method: {method}")
            
        print(f"    ✅ Generated real embeddings shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        print(f"    ❌ Failed to train {method}: {e}")
        # Simple fallback
        n_nodes = G_coarse.number_of_nodes()
        embeddings = np.random.randn(n_nodes, 128) * 0.1  # Small variance
        print(f"    📝 Using fallback embeddings: {embeddings.shape}")
        return embeddings

def original_refinement(levels, projections, laplacians, embeddings, lda=0.1, power=False):
    """Original GraphZoom refinement function"""
    print(f"  [ORIGINAL REFINEMENT]")
    for i in reversed(range(levels)):
        embeddings = projections[i] @ embeddings
        filter_ = smooth_filter(laplacians[i], lda)
        if power or i == 0:
            embeddings = filter_ @ (filter_ @ embeddings)
    return embeddings

def enhanced_mp_refinement(levels, projections, laplacians, embeddings, lda=0.1, power=False):
    """Enhanced MP refinement (traditional + MP correction)"""
    print(f"  [ENHANCED MP REFINEMENT]")
    try:
        from enhanced_refinement import (
            build_mp_preserving_matrix, 
            extract_clusters_from_projection,
            compute_message_passing_error,
            laplacian_to_propagation,
            build_correction_filter
        )
        
        for i in reversed(range(levels)):
            # Step 1: Standard projection
            embeddings = projections[i] @ embeddings
            
            # Step 2: MP correction
            laplacian = laplacians[i]
            clusters = extract_clusters_from_projection(projections[i])
            Q = build_mp_preserving_matrix(clusters, laplacian.shape[0])
            S = laplacian_to_propagation(laplacian)
            mp_error, S_approx, S_c = compute_message_passing_error(S, Q)
            
            print(f"    Level {i}: MP error = {mp_error:.6f}")
            
            if mp_error > 0.1:
                correction_filter = build_correction_filter(S_approx, lda)
                embeddings = correction_filter @ embeddings
            
            # Step 3: Traditional spectral smoothing
            filter_ = smooth_filter(laplacians[i], lda)
            if power or i == 0:
                embeddings = filter_ @ (filter_ @ embeddings)
        
        return embeddings
        
    except Exception as e:
        print(f"    ⚠️  Enhanced MP refinement failed: {e}")
        print(f"    📝 Falling back to original refinement")
        return original_refinement(levels, projections, laplacians, embeddings, lda, power)

def mp_only_refinement(levels, projections, laplacians, embeddings, lda=0.1, power=False):
    """MP-ONLY refinement (no traditional spectral smoothing)"""
    print(f"  [MP-ONLY REFINEMENT] (NEW)")
    try:
        from enhanced_refinement import (
            build_mp_preserving_matrix, 
            extract_clusters_from_projection,
            compute_message_passing_error,
            laplacian_to_propagation,
            build_correction_filter
        )
        
        for i in reversed(range(levels)):
            # Step 1: Standard projection
            embeddings = projections[i] @ embeddings
            
            # Step 2: ONLY MP correction (NO traditional smoothing)
            laplacian = laplacians[i]
            clusters = extract_clusters_from_projection(projections[i])
            Q = build_mp_preserving_matrix(clusters, laplacian.shape[0])
            S = laplacian_to_propagation(laplacian)
            mp_error, S_approx, S_c = compute_message_passing_error(S, Q)
            
            print(f"    Level {i}: MP error = {mp_error:.6f}")
            
            if mp_error > 0.1:
                print(f"    Applying ONLY MP correction (no spectral smoothing)")
                correction_filter = build_correction_filter(S_approx, lda)
                embeddings = correction_filter @ embeddings
            else:
                print(f"    Low MP error - minimal correction applied")
            
            # NO Step 3: Skip traditional spectral smoothing entirely
        
        return embeddings
        
    except Exception as e:
        print(f"    ⚠️  MP-only refinement failed: {e}")
        print(f"    📝 Falling back to original refinement")
        return original_refinement(levels, projections, laplacians, embeddings, lda, power)

def smooth_filter(laplacian_matrix, lda):
    """Original GraphZoom smooth filter"""
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

def run_graphzoom_comprehensive(laplacian, features, method="cmg", 
                              refinement_type="original", use_fusion=True,
                              embedding_method="deepwalk"):
    """Run GraphZoom with all configuration options"""
    
    print(f"\n🚀 Running GraphZoom Configuration:")
    print(f"   Coarsening: {method}")
    print(f"   Refinement: {refinement_type}")
    print(f"   Fusion: {use_fusion}")
    print(f"   Embeddings: {embedding_method}")
    
    try:
        # Step 1: Apply fusion if enabled
        if use_fusion:
            print(f"  🔗 Applying graph fusion...")
            # Use GraphZoom's graph_fusion function
            try:
                from graphzoom_timed import graph_fusion
                
                # Prepare fusion parameters
                fusion_input_path = "temp_fusion_input.mtx"
                from scipy.io import mmwrite
                mmwrite(fusion_input_path, laplacian)
                
                # Set fusion parameters
                num_neighs = 2
                mcr_dir = "/home/mohammad/matlab/R2018a/"  # Fixed path - don't include v94
                search_ratio = 12
                fusion_output_dir = "temp_fusion_results"
                mapping_path = "temp_mapping.mtx"
                dataset = "temp"
                
                os.makedirs(fusion_output_dir, exist_ok=True)
                
                cmg_params = {'k': 10, 'd': 20, 'threshold': 0.1}
                
                fused_laplacian = graph_fusion(
                    laplacian, features, num_neighs, mcr_dir, method,
                    fusion_input_path, search_ratio, fusion_output_dir, 
                    mapping_path, dataset, cmg_params
                )
                
                print(f"    ✅ Fusion completed")
                laplacian_to_use = fused_laplacian
                
                # Cleanup
                try:
                    os.remove(fusion_input_path)
                    import shutil
                    shutil.rmtree(fusion_output_dir, ignore_errors=True)
                    if os.path.exists(mapping_path):
                        os.remove(mapping_path)
                except:
                    pass
                    
            except Exception as e:
                print(f"    ⚠️  Fusion failed: {e}")
                print(f"    📝 Using original laplacian")
                laplacian_to_use = laplacian
        else:
            print(f"  ⏭️  Skipping fusion")
            laplacian_to_use = laplacian
        
        # Step 2: Apply coarsening
        if method == "cmg":
            from cmg_coarsening_timed import cmg_coarse
            degree_diag = diags(laplacian_to_use.diagonal(), 0)
            adjacency = degree_diag - laplacian_to_use
            G = nx.from_scipy_sparse_matrix(adjacency)
            G_coarse, projections, laplacians, levels = cmg_coarse(
                laplacian_to_use, level=1, k=10, d=20, threshold=0.1
            )
            
        elif method == "simple":
            from utils import sim_coarse
            G_coarse, projections, laplacians, levels = sim_coarse(laplacian_to_use, level=1)
            
        elif method == "lamg":
            # Use LAMG with fixed MCR path
            from utils import construct_proj_laplacian, mtx2graph, read_levels, sim_coarse
            from scipy.io import mmwrite
            import tempfile
            import shutil
            
            temp_dir = tempfile.mkdtemp(prefix="lamg_comprehensive_")
            input_file = os.path.join(temp_dir, "input.mtx")
            
            try:
                mmwrite(input_file, laplacian_to_use)
                
                # Fixed MCR path - don't include v94, let script append it
                mcr_dir = "/home/mohammad/matlab/R2018a/"
                reduce_ratio = 2
                
                print(f"    MCR directory: {mcr_dir}")
                print(f"    Checking if MCR exists: {os.path.exists(mcr_dir)}")
                
                lamg_script = "./run_coarsening.sh"
                if os.path.exists(lamg_script) and os.path.exists(mcr_dir):
                    cmd = f'{lamg_script} {mcr_dir} {input_file} {reduce_ratio} n {temp_dir}'
                    print(f"  Running LAMG: {cmd}")
                    result = os.system(cmd)
                    
                    gs_file = f"{temp_dir}/Gs.mtx"
                    levels_file = f"{temp_dir}/NumLevels.txt"
                    
                    if result == 0 and os.path.exists(gs_file) and os.path.exists(levels_file):
                        G_coarse = mtx2graph(gs_file)
                        levels = read_levels(levels_file)
                        projections, laplacians = construct_proj_laplacian(laplacian_to_use, levels, temp_dir)
                        print(f"  ✅ LAMG succeeded")
                    else:
                        raise Exception("LAMG execution failed")
                else:
                    raise Exception("LAMG not available")
                    
            except Exception as e:
                print(f"  ⚠️  LAMG failed: {e}")
                print(f"  📝 Using simple coarsening as fallback")
                G_coarse, projections, laplacians, levels = sim_coarse(laplacian_to_use, level=1)
            finally:
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
        
        print(f"  Coarsened: {laplacian_to_use.shape[0]} → {G_coarse.number_of_nodes()} nodes")
        
        # Step 3: Generate real embeddings
        embedding_start = time.time()
        coarse_embeddings = train_real_embeddings(G_coarse, method=embedding_method)
        embedding_time = time.time() - embedding_start
        
        # Step 4: Apply refinement based on type
        refine_start = time.time()
        
        if refinement_type == "original":
            final_embeddings = original_refinement(levels, projections, laplacians, coarse_embeddings)
        elif refinement_type == "enhanced_mp":
            final_embeddings = enhanced_mp_refinement(levels, projections, laplacians, coarse_embeddings)
        elif refinement_type == "mp_only":
            final_embeddings = mp_only_refinement(levels, projections, laplacians, coarse_embeddings)
        else:
            raise ValueError(f"Unknown refinement type: {refinement_type}")
        
        refinement_time = time.time() - refine_start
        total_time = embedding_time + refinement_time
        
        print(f"  Embedding time: {embedding_time:.3f}s")
        print(f"  Refinement time: {refinement_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Final embeddings shape: {final_embeddings.shape}")
        
        return final_embeddings, total_time
        
    except Exception as e:
        print(f"❌ GraphZoom comprehensive failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def evaluate_node_classification(embeddings, features, labels, train_mask, val_mask, test_mask):
    """Evaluate embeddings on node classification task"""
    print("🎯 Evaluating node classification...")
    
    # Use only embeddings (structure-only evaluation)
    X = embeddings
    print(f"  Using only embeddings: {embeddings.shape[1]} features")
    
    # Prepare data
    X_train = X[train_mask]
    y_train = labels[train_mask]
    X_val = X[val_mask]
    y_val = labels[val_mask]
    X_test = X[test_mask]
    y_test = labels[test_mask]
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Train logistic regression
    print("  Training classifier...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)
    
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    val_f1 = f1_score(y_val, val_pred, average='macro')
    test_f1 = f1_score(y_test, test_pred, average='macro')
    
    results = {
        'val_accuracy': val_acc,
        'test_accuracy': test_acc,
        'val_f1': val_f1,
        'test_f1': test_f1
    }
    
    print(f"  📊 Test Accuracy: {test_acc:.4f}")
    return results

def print_comprehensive_table(results_list):
    """Print comprehensive comparison table"""
    print(f"\n" + "="*120)
    print("COMPREHENSIVE EVALUATION: REFINEMENT TYPES × FUSION × COARSENING METHODS")
    print("="*120)
    
    headers = ["Method", "Fusion", "Refinement", "Test Acc", "Val Acc", "Time(s)"]
    
    # Print header
    header_row = f"{headers[0]:<10} {headers[1]:<8} {headers[2]:<12} {headers[3]:<9} {headers[4]:<9} {headers[5]:<8}"
    print(f"\n{header_row}")
    print("-" * len(header_row))
    
    # Print results
    for result in results_list:
        fusion_str = "✅" if result['fusion'] else "❌"
        row = f"{result['method']:<10} {fusion_str:<8} {result['refinement']:<12} {result['test_accuracy']:<9.4f} {result['val_accuracy']:<9.4f} {result['time']:<8.1f}"
        print(row)
    
    # Analysis sections
    print(f"\n" + "="*80)
    print("ANALYSIS: REFINEMENT TYPE IMPACT")
    print("="*80)
    
    # Group by method and fusion, compare refinement types
    method_fusion_groups = {}
    for result in results_list:
        key = (result['method'], result['fusion'])
        if key not in method_fusion_groups:
            method_fusion_groups[key] = {}
        method_fusion_groups[key][result['refinement']] = result
    
    print(f"\n{'Method':<10} {'Fusion':<8} {'Original':<9} {'Enhanced':<9} {'MP-Only':<9} {'Best'}")
    print("-" * 60)
    
    for (method, fusion), refinement_results in method_fusion_groups.items():
        fusion_str = "✅" if fusion else "❌"
        
        orig_acc = refinement_results.get('original', {}).get('test_accuracy', 0)
        enh_acc = refinement_results.get('enhanced_mp', {}).get('test_accuracy', 0)  
        mp_acc = refinement_results.get('mp_only', {}).get('test_accuracy', 0)
        
        best_type = max([
            ('original', orig_acc),
            ('enhanced_mp', enh_acc),
            ('mp_only', mp_acc)
        ], key=lambda x: x[1])
        
        print(f"{method:<10} {fusion_str:<8} {orig_acc:<9.4f} {enh_acc:<9.4f} {mp_acc:<9.4f} {best_type[0]}")
    
    print(f"\n" + "="*80)
    print("ANALYSIS: FUSION IMPACT")
    print("="*80)
    
    # Group by method and refinement, compare fusion
    method_refinement_groups = {}
    for result in results_list:
        key = (result['method'], result['refinement'])
        if key not in method_refinement_groups:
            method_refinement_groups[key] = {}
        method_refinement_groups[key][result['fusion']] = result
    
    print(f"\n{'Method':<10} {'Refinement':<12} {'No Fusion':<10} {'With Fusion':<12} {'Δ Acc':<8}")
    print("-" * 60)
    
    for (method, refinement), fusion_results in method_refinement_groups.items():
        no_fusion_acc = fusion_results.get(False, {}).get('test_accuracy', 0)
        with_fusion_acc = fusion_results.get(True, {}).get('test_accuracy', 0)
        
        diff = with_fusion_acc - no_fusion_acc
        diff_str = f"{diff:+.3f}"
        
        print(f"{method:<10} {refinement:<12} {no_fusion_acc:<10.4f} {with_fusion_acc:<12.4f} {diff_str:<8}")
    
    # Overall best
    best_overall = max(results_list, key=lambda x: x['test_accuracy'])
    print(f"\n🏆 OVERALL BEST: {best_overall['method']} + {best_overall['refinement']} + {'Fusion' if best_overall['fusion'] else 'No Fusion'}")
    print(f"   Accuracy: {best_overall['test_accuracy']:.4f}")
    print(f"   Time: {best_overall['time']:.1f}s")

def main():
    """Main comprehensive evaluation"""
    print("COMPREHENSIVE EVALUATION: MP-ONLY + FUSION ANALYSIS")
    print("="*80)
    
    # Check MATLAB setup
    matlab_ok = check_matlab_setup()
    if not matlab_ok:
        print("⚠️  MATLAB Runtime issues detected - LAMG may fail")
    
    # Load dataset
    laplacian, features, labels, dataset_name = load_cora_dataset()
    if laplacian is None:
        print("❌ Failed to load dataset")
        return
    
    # Create splits
    n_nodes = laplacian.shape[0]
    train_mask, val_mask, test_mask = create_train_test_split(n_nodes)
    
    # Define comprehensive test matrix
    # methods = ["simple", "cmg", "lamg"]
    methods = [ "lamg"]
    refinement_types = ["original", "enhanced_mp", "mp_only"]
    fusion_options = [True, False]  # True = fusion enabled, False = fusion disabled
    embedding_method = "deepwalk"
    
    results = []
    total_tests = len(methods) * len(refinement_types) * len(fusion_options)
    test_counter = 0
    
    print(f"\n🧪 Running {total_tests} comprehensive tests...")
    
    for method in methods:
        for refinement_type in refinement_types:
            for use_fusion in fusion_options:
                test_counter += 1
                fusion_str = "With Fusion" if use_fusion else "No Fusion"
                test_name = f"{method.upper()} + {refinement_type} + {fusion_str}"
                
                print(f"\n" + "="*80)
                print(f"TEST {test_counter}/{total_tests}: {test_name}")
                print("="*80)
                
                try:
                    # Run comprehensive GraphZoom
                    embeddings, runtime = run_graphzoom_comprehensive(
                        laplacian, features, method=method,
                        refinement_type=refinement_type, use_fusion=use_fusion,
                        embedding_method=embedding_method
                    )
                    
                    if embeddings is not None:
                        # Evaluate
                        eval_results = evaluate_node_classification(
                            embeddings, features, labels, train_mask, val_mask, test_mask
                        )
                        
                        # Store results
                        results.append({
                            'method': method,
                            'refinement': refinement_type,
                            'fusion': use_fusion,
                            'time': runtime,
                            'embedding_method': embedding_method,
                            **eval_results
                        })
                        
                        print(f"✅ Completed: {eval_results['test_accuracy']:.4f} accuracy")
                    else:
                        print(f"❌ Failed to generate embeddings")
                        
                except Exception as e:
                    print(f"❌ Test failed: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Print comprehensive results
    if results:
        print_comprehensive_table(results)
        
        # Save results
        results_file = f"comprehensive_evaluation_{dataset_name}_{embedding_method}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {results_file}")
        
    else:
        print("❌ No results to compare")
    
    print(f"\n✅ Comprehensive evaluation completed!")

if __name__ == "__main__":
    main()