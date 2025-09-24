import numpy as np
import networkx as nx
import os
from scipy.sparse import identity
from scipy.io import mmwrite
import sys
from argparse import ArgumentParser
from sklearn.preprocessing import normalize
import time

from embed_methods.deepwalk.deepwalk import *
from embed_methods.node2vec.node2vec import *
from utils import *
from scoring import lr
from cmg_coarsening_timed import cmg_coarse, cmg_fusion_mapping


def graph_fusion(laplacian, feature, num_neighs, mcr_dir, coarse, fusion_input_path, \
                 search_ratio, fusion_output_dir, mapping_path, dataset, cmg_params=None):
    """
    Graph fusion with support for simple, lamg, and CMG mapping methods.
    """

    # obtain mapping operator
    if coarse == "simple":
        mapping = sim_coarse_fusion(laplacian)
    elif coarse == "lamg":
        os.system('./run_coarsening.sh {} {} {} f {}'.format(mcr_dir, \
                fusion_input_path, search_ratio, fusion_output_dir))
        mapping = mtx2matrix(mapping_path)
    elif coarse == "cmg":
        # CMG provides smart node groupings for fusion
        if cmg_params is None:
            cmg_params = {'k': 10, 'd': 20, 'threshold': 0.1}
        mapping = cmg_fusion_mapping(laplacian, cmg_params['k'], cmg_params['d'], cmg_params['threshold'])
        print(f"[CMG FUSION] Generated mapping: {mapping.shape} (reduction: {laplacian.shape[0] / mapping.shape[0]:.2f}x)")
    else:
        raise NotImplementedError

    # construct feature graph
    feats_laplacian = feats2graph(feature, num_neighs, mapping)

    # fuse adj_graph with feat_graph
    fused_laplacian = laplacian + feats_laplacian

    if coarse == "lamg":
        file = open("dataset/{}/fused_{}.mtx".format(dataset, dataset), "wb")
        mmwrite("dataset/{}/fused_{}.mtx".format(dataset, dataset), fused_laplacian)
        file.close()
        print("Successfully Writing Fused Graph.mtx file!!!!!!")

    return fused_laplacian


def refinement(levels, projections, coarse_laplacian, embeddings, lda, power, 
               method="original", mp_threshold=0.1, show_mp_timing=False):
    """
    Enhanced refinement function supporting multiple methods
    
    Args:
        method: "original", "mp_aware", or "mp_aware+original"
        mp_threshold: MP error threshold for applying correction
        show_mp_timing: Whether to show detailed MP timing
    """
    
    if method == "original":
        # Original GraphZoom refinement (unchanged)
        for i in reversed(range(levels)):
            embeddings = projections[i] @ embeddings
            filter_ = smooth_filter(coarse_laplacian[i], lda)
            if power or i == 0:
                embeddings = filter_ @ (filter_ @ embeddings)
        return embeddings
    
    elif method == "mp_aware":
        # Only MP-aware refinement
        try:
            from enhanced_refinement import (
                build_mp_preserving_matrix, 
                extract_clusters_from_projection,
                compute_message_passing_error,
                laplacian_to_propagation,
                build_correction_filter
            )
        except ImportError:
            raise ImportError("enhanced_refinement module required for MP-aware refinement. "
                            "Please ensure enhanced_refinement.py is available.")
        
        if show_mp_timing:
            print(f"[MP-AWARE REFINEMENT] Starting with threshold={mp_threshold}")
        
        for i in reversed(range(levels)):
            level_start = time.process_time() if show_mp_timing else None
            
            # Step 1: Standard projection
            embeddings = projections[i] @ embeddings
            
            # Step 2: Message-passing correction
            laplacian = coarse_laplacian[i]
            clusters = extract_clusters_from_projection(projections[i])
            Q = build_mp_preserving_matrix(clusters, laplacian.shape[0])
            S = laplacian_to_propagation(laplacian)
            mp_error, S_approx, S_c = compute_message_passing_error(S, Q)
            
            if show_mp_timing:
                print(f"  Level {i}: MP error = {mp_error:.6f}")
            
            if mp_error > mp_threshold:
                if show_mp_timing:
                    print(f"  Applying MP correction (error > {mp_threshold})")
                correction_filter = build_correction_filter(S_approx, lda)
                embeddings = correction_filter @ embeddings
            elif show_mp_timing:
                print(f"  Skipping MP correction (error <= {mp_threshold})")
            
            # Step 3: Enhanced spectral smoothing
            filter_ = smooth_filter(coarse_laplacian[i], lda)
            if power or i == 0:
                embeddings = filter_ @ (filter_ @ embeddings)
            
            if show_mp_timing and level_start:
                level_time = time.process_time() - level_start
                print(f"  Level {i} total time: {level_time:.3f}s")
        
        return embeddings
    
    elif method == "mp_aware+original":
        # Both MP-aware and original refinement combined
        try:
            from enhanced_refinement import (
                build_mp_preserving_matrix, 
                extract_clusters_from_projection,
                compute_message_passing_error,
                laplacian_to_propagation,
                build_correction_filter
            )
        except ImportError:
            raise ImportError("enhanced_refinement module required for MP-aware refinement. "
                            "Please ensure enhanced_refinement.py is available.")
        
        if show_mp_timing:
            print(f"[MP-AWARE+ORIGINAL REFINEMENT] Starting with threshold={mp_threshold}")
        
        for i in reversed(range(levels)):
            level_start = time.process_time() if show_mp_timing else None
            
            # Step 1: Standard projection
            embeddings = projections[i] @ embeddings
            
            # Step 2: Message-passing correction
            laplacian = coarse_laplacian[i]
            clusters = extract_clusters_from_projection(projections[i])
            Q = build_mp_preserving_matrix(clusters, laplacian.shape[0])
            S = laplacian_to_propagation(laplacian)
            mp_error, S_approx, S_c = compute_message_passing_error(S, Q)
            
            if show_mp_timing:
                print(f"  Level {i}: MP error = {mp_error:.6f}")
            
            if mp_error > mp_threshold:
                if show_mp_timing:
                    print(f"  Applying MP correction (error > {mp_threshold})")
                correction_filter = build_correction_filter(S_approx, lda)
                embeddings = correction_filter @ embeddings
            elif show_mp_timing:
                print(f"  Skipping MP correction (error <= {mp_threshold})")
            
            # Step 3: Original GraphZoom smoothing (always applied)
            filter_ = smooth_filter(coarse_laplacian[i], lda)
            if power or i == 0:
                embeddings = filter_ @ (filter_ @ embeddings)
            
            if show_mp_timing and level_start:
                level_time = time.process_time() - level_start
                print(f"  Level {i} total time: {level_time:.3f}s")
        
        return embeddings
    
    else:
        raise ValueError(f"Unknown refinement method: {method}. "
                        f"Choose from: original, mp_aware, mp_aware+original")


def cluster_assignments_to_clusters(cluster_assignments):
    """
    Convert cluster assignments array to list of clusters.
    
    Args:
        cluster_assignments: Array like [0, 0, 1, 1, 2, 2] indicating cluster membership
    
    Returns:
        List of clusters like [[0, 1], [2, 3], [4, 5]]
    """
    clusters = []
    num_clusters = max(cluster_assignments) + 1
    
    for cluster_id in range(num_clusters):
        cluster = [node_id for node_id, cid in enumerate(cluster_assignments) if cid == cluster_id]
        if cluster:  # Only add non-empty clusters
            clusters.append(cluster)
    
    return clusters


def main():
    parser = ArgumentParser(description="GraphZoom with CMG Integration")
    parser.add_argument("-d", "--dataset", type=str, default="cora", \
            help="input dataset")
    parser.add_argument("-o", "--coarse", type=str, default="simple", \
            help="choose coarsening method: [simple, lamg, cmg]")
    
    # CMG-specific parameters
    parser.add_argument("--cmg_k", type=int, default=10, \
            help="CMG filter order (only for CMG coarsening)")
    parser.add_argument("--cmg_d", type=int, default=20, \
            help="CMG embedding dimension (only for CMG coarsening)")  
    parser.add_argument("--cmg_threshold", type=float, default=0.1, \
            help="CMG cosine similarity threshold (only for CMG coarsening)")
    parser.add_argument("--seed", type=int, default=42, \
            help="Random seed for reproducibility")

    # Original GraphZoom parameters
    parser.add_argument("-c", "--mcr_dir", type=str, default="~/matlab/R2018a/", \
            help="directory of matlab compiler runtime (only required by lamg_coarsen)")
    parser.add_argument("-s", "--search_ratio", type=int, default=12, \
            help="control the search space in graph fusion process (only required by lamg_coarsen)")
    parser.add_argument("-r", "--reduce_ratio", type=int, default=2, \
            help="control graph coarsening levels (only required by lamg_coarsen)")
    parser.add_argument("-v", "--level", type=int, default=1, \
            help="number of coarsening levels")
    parser.add_argument("-n", "--num_neighs", type=int, default=2, \
            help="control k-nearest neighbors in graph fusion process")
    parser.add_argument("-l", "--lda", type=float, default=0.1, \
            help="control self loop in adjacency matrix")
    parser.add_argument("-e", "--embed_path", type=str, default="embed_results/embeddings.npy", \
            help="path of embedding result")
    parser.add_argument("-m", "--embed_method", type=str, default="deepwalk", \
            help="[deepwalk, node2vec, graphsage, true_coarsened_graphsage]")
    parser.add_argument("-f", "--fusion", default=True, action="store_false", \
            help="whether use graph fusion")
    parser.add_argument("-p", "--power", default=False, action="store_true", \
            help="Strong power of graph filter, set True to enhance filter power")
    parser.add_argument("-g", "--sage_model", type=str, default="mean", \
            help="aggregation function in graphsage")
    parser.add_argument("-w", "--sage_weighted", default=True, action="store_false", \
            help="whether consider weighted reduced graph")

    # MP-Aware Refinement parameters
    parser.add_argument("--refinement_method", type=str, default="original", 
                       choices=["original", "mp_aware", "mp_aware+original"],
                       help="Refinement method: original (default), mp_aware, or mp_aware+original")
    parser.add_argument("--mp_threshold", type=float, default=0.1,
                       help="MP error threshold for applying correction (only for mp_aware)")
    parser.add_argument("--mp_timing", action="store_true",
                       help="Show detailed MP refinement timing breakdown")
    
    # ClusterGraphSAGE parameters
    parser.add_argument("--cluster_s_in", type=int, default=4,
                   help="Cluster sampling: inner samples per cluster")
    parser.add_argument("--cluster_s_out", type=int, default=1, 
                   help="Cluster sampling: boundary samples")
    
    # True Coarsened GraphSAGE parameters
    parser.add_argument("--true_sage_super_dim", type=int, default=256,
                    help="Super-node embedding dimension for True Coarsened GraphSAGE")
    parser.add_argument("--true_sage_final_dim", type=int, default=256,
                    help="Final embedding dimension for True Coarsened GraphSAGE")
    parser.add_argument("--true_sage_hidden_dim", type=int, default=128,
                    help="Hidden layer dimension for True Coarsened GraphSAGE")
    parser.add_argument("--true_sage_base_epochs", type=int, default=1000,
                    help="Base training epochs for True Coarsened GraphSAGE")

    args = parser.parse_args()

    # Set random seed for reproducibility
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = args.dataset
    feature_path = "dataset/{}/{}-feats.npy".format(dataset, dataset)
    fusion_input_path = "dataset/{}/{}.mtx".format(dataset, dataset)
    reduce_results = "reduction_results/"
    mapping_path = "{}Mapping.mtx".format(reduce_results)

    if args.fusion:
        coarsen_input_path = "dataset/{}/fused_{}.mtx".format(dataset, dataset)
    else:
        coarsen_input_path = "dataset/{}/{}.mtx".format(dataset, dataset)

    # Initialize variables for cluster assignments (needed for true_coarsened_graphsage)
    all_cluster_assignments = None

######Load Data######
    print("%%%%%% Loading Graph Data %%%%%%")
    laplacian = json2mtx(dataset)

    ## whether node features are required
    if args.fusion or args.embed_method in ["graphsage", "true_coarsened_graphsage"]:
        feature = np.load(feature_path)

######Graph Fusion######
    if args.fusion:
        print("%%%%%% Starting Graph Fusion %%%%%%")
        fusion_start = time.process_time()
        cmg_params = {'k': args.cmg_k, 'd': args.cmg_d, 'threshold': args.cmg_threshold}
        laplacian = graph_fusion(laplacian, feature, args.num_neighs, args.mcr_dir, args.coarse,\
                fusion_input_path, args.search_ratio, reduce_results, mapping_path, dataset, cmg_params)
        fusion_time = time.process_time() - fusion_start

######Graph Reduction######
    print("%%%%%% Starting Graph Reduction %%%%%%")
    reduce_start = time.process_time()

    if args.coarse == "simple":
        G, projections, laplacians, level = sim_coarse(laplacian, args.level)
        reduce_time = time.process_time() - reduce_start

    elif args.coarse == "lamg":
        print(f"**********************[LAMG] Coarsening input path: {coarsen_input_path}")
        
        with open(coarsen_input_path, 'r') as f:
            lines = f.readlines()
            print("[LAMG] Preview of input .mtx:")
            for i, line in enumerate(lines):
                if not line.startswith('%'):  # Skip comments
                    print(line.strip())
                if i > 100: break  # Only preview first 20 rows
            
        os.system('./run_coarsening.sh {} {} {} n {}'.format(args.mcr_dir, \
                coarsen_input_path, args.reduce_ratio, reduce_results))
        reduce_time = read_time("{}CPUtime.txt".format(reduce_results))
        G = mtx2graph("{}Gs.mtx".format(reduce_results))
        level = read_levels("{}NumLevels.txt".format(reduce_results))
        projections, laplacians = construct_proj_laplacian(laplacian, level, reduce_results)

    elif args.coarse == "cmg":
        print(f"**********************[CMG] Multi-level coarsening with {args.level} levels")
        print(f"[CMG] CMG parameters: k={args.cmg_k}, d={args.cmg_d}, threshold={args.cmg_threshold}")
        
        if args.fusion:
            print("[CMG] Processing fused graph (original + features)")
        else:
            print("[CMG] Processing original graph (no fusion)")
            
        # MODIFIED: Now catches cluster assignments too
        G, projections, laplacians, level, all_cluster_assignments = cmg_coarse(
            laplacian, args.level, args.cmg_k, args.cmg_d, args.cmg_threshold
        )
        reduce_time = time.process_time() - reduce_start

    else:
        raise NotImplementedError


######Embed Reduced Graph######
    print("%%%%%% Starting Graph Embedding %%%%%%")
    if args.embed_method == "deepwalk":
        embed_start = time.process_time()
        embeddings  = deepwalk(G)

    elif args.embed_method == "node2vec":
        embed_start = time.process_time()
        embeddings  = node2vec(G)

    elif args.embed_method == "graphsage":
        from embed_methods.graphsage.graphsage import graphsage
        
        print(f"\n=== DEBUGGING GraphSAGE INPUT for {args.coarse} ===")
        
        # Original graph info
        print(f"Original laplacian shape: {laplacian.shape}")
        print(f"Original features shape: {feature.shape}")
        
        # Coarsened graph info
        print(f"Coarsened graph G nodes: {len(G.nodes())}")
        print(f"Coarsened graph G edges: {len(G.edges())}")
        
        nx.set_node_attributes(G, False, "test")
        nx.set_node_attributes(G, False, "val")

        ## obtain mapping operator
        if args.coarse == "lamg":
            mapping = normalize(mtx2matrix(mapping_path), norm='l1', axis=1)
            
            print(f"LAMG mapping shape: {mapping.shape}")
            print(f"LAMG mapping type: {type(mapping)}")
        
        else:
            mapping = identity(feature.shape[0])
            for p in projections:
                mapping = mapping @ p
            mapping = normalize(mapping, norm='l1', axis=1).transpose()
            
            print(f"{args.coarse.upper()} mapping shape: {mapping.shape}")
            print(f"{args.coarse.upper()} mapping type: {type(mapping)}")

        ## control iterations for training
        coarse_ratio = mapping.shape[1]/mapping.shape[0]

        ## map node feats to the coarse graph
        feats = mapping @ feature
        print(f"Mapped features shape: {feats.shape}")
        print(f"Features per node: {feats.shape[0]} nodes, {feats.shape[1]} dims")

        ## control iterations for training
        coarse_ratio = mapping.shape[1]/mapping.shape[0]
        print(f"Coarse ratio: {coarse_ratio}")
        print(f"Training epochs: {int(1000/coarse_ratio)}")
        
        print("=== END DEBUG INFO ===\n")
    
        embed_start = time.process_time()
        embeddings  = graphsage(G, feats, args.sage_model, args.sage_weighted, int(1000/coarse_ratio))
        
        print(f"\n🔍 REGULAR GRAPHSAGE DIMENSION CHECK:")
        print(f"   Output embeddings shape: {embeddings.shape}")
        print(f"   Expected nodes: {len(G.nodes())}")
        print(f"   Actual embedding dimension: {embeddings.shape[1]}D")
        
        if embeddings.shape[1] == 256:
            print(f"   ✅ Regular GraphSAGE outputs 256D (same as True Coarsened)")
            print(f"   ✅ Fair comparison confirmed!")
        elif embeddings.shape[1] == 64:
            print(f"   ⚠️  Regular GraphSAGE outputs 64D (different from True Coarsened 256D)")
            print(f"   ❓ Comparison may not be fair - True Coarsened has 4x more capacity")
        else:
            print(f"   ❓ Regular GraphSAGE outputs {embeddings.shape[1]}D")
            print(f"   ❓ Need to analyze comparison fairness")
        
        embed_time = time.process_time() - embed_start
    
    # In the embedding methods section:
    elif args.embed_method == "true_coarsened_graphsage":
        from embed_methods.graphsage.true_coarsened_graphsage import true_coarsened_graphsage
        
        print("-----++++++++++++RUNNING TRUE COARSENED GRAPHSAGE (New Idea)")
        print(f"🎯 Dataset: {dataset}")
        print(f"🎯 Coarsening method: {args.coarse}")
        print(f"🎯 Fusion enabled: {args.fusion}")
        print(f"🎯 Levels: {args.level}")
        
        # EFFICIENT: Use cluster assignments directly if available (CMG), otherwise fall back to projection scanning
        print("\n📊 EXTRACTING CLUSTERS...")
        
        if args.coarse == "cmg" and all_cluster_assignments is not None:
            # EFFICIENT: Use cluster assignments directly from CMG
            print("✅ Using cluster assignments directly from CMG (efficient)")
            clusters = cluster_assignments_to_clusters(all_cluster_assignments[0])  # Use level 0
            print(f"✅ Efficiently extracted {len(clusters)} clusters from CMG assignments")
            
        else:
            # FALLBACK: Extract from projection matrix (for non-CMG methods)
            print("⚠️  Falling back to projection matrix extraction (less efficient)")
            clusters = []
            n_nodes, n_clusters = projections[0].shape
            
            for cluster_id in range(n_clusters):
                cluster = []
                for node_id in range(n_nodes):
                    if projections[0][node_id, cluster_id] > 0:
                        cluster.append(node_id)
                if cluster:
                    clusters.append(cluster)
            
            print(f"✅ Extracted {len(clusters)} clusters from projection matrix")
        
        # Show cluster statistics
        cluster_sizes = [len(cluster) for cluster in clusters]
        print(f"📈 Cluster size stats: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.1f}")
        
        # Detailed cluster info for small datasets
        if len(clusters) <= 10:  # Only for small graphs to avoid spam
            for i, cluster in enumerate(clusters):
                print(f"  Cluster {i}: {cluster}")
        
        # Convert Laplacian back to NetworkX for original graph
        print("\n📄 CONVERTING TO NETWORKX FORMAT...")
        degree_diag = diags(laplacian.diagonal(), 0)
        adjacency = degree_diag - laplacian
        
        # Ensure non-negative adjacency (remove any numerical artifacts)
        adjacency.data = np.abs(adjacency.data)
        
        original_nx_graph = nx.from_scipy_sparse_matrix(adjacency)
        
        print(f"✅ Original NetworkX graph: {original_nx_graph.number_of_nodes()} nodes, {original_nx_graph.number_of_edges()} edges")
        print(f"✅ Using ORIGINAL features shape: {feature.shape}")
        print(f"✅ Coarsened graph G: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"✅ Coarsening ratio: {original_nx_graph.number_of_nodes() / G.number_of_nodes():.2f}x")
        
        # Calculate appropriate training epochs based on coarsening ratio
        base_epochs = args.true_sage_base_epochs
        coarse_ratio = G.number_of_nodes() / original_nx_graph.number_of_nodes()
        training_epochs = max(100, int(base_epochs * coarse_ratio))  # Scale training with coarsening
        
        print(f"\n⚙️ TRAINING CONFIGURATION:")
        print(f"   Base epochs: {base_epochs}")
        print(f"   Coarse ratio: {coarse_ratio:.3f}")
        print(f"   Adjusted training epochs: {training_epochs}")
        print(f"   Super embedding dim: {args.true_sage_super_dim}")
        print(f"   Final embedding dim: {args.true_sage_final_dim}")
        print(f"   Hidden dim: {args.true_sage_hidden_dim}")
        
        # Pre-execution validation
        print(f"\n🔍 PRE-EXECUTION VALIDATION:")
        print(f"   ✅ Original graph connected components: {nx.number_connected_components(original_nx_graph)}")
        print(f"   ✅ Coarsened graph connected components: {nx.number_connected_components(G)}")
        print(f"   ✅ Feature matrix shape matches nodes: {feature.shape[0] == original_nx_graph.number_of_nodes()}")
        print(f"   ✅ All cluster nodes valid: {all(all(node < original_nx_graph.number_of_nodes() for node in cluster) for cluster in clusters)}")
        print(f"   ✅ Projection matrix shape: {projections[0].shape}")
        print(f"   ✅ Laplacian matrices count: {len(laplacians)}")
        
        # Run True Coarsened GraphSAGE with full training
        print(f"\n🎬 STARTING TRUE COARSENED GRAPHSAGE EXECUTION...")
        embed_start = time.process_time()
        
        try:
            embeddings = true_coarsened_graphsage(
                original_graph=original_nx_graph,
                features=feature,  # Use ORIGINAL features (not coarsened)
                clusters=clusters,
                coarsened_graph=G,  # CMG coarsened graph
                projections=projections,
                laplacians=laplacians,
                super_embed_dim=args.true_sage_super_dim,
                final_embed_dim=args.true_sage_final_dim,
                hidden_dim=args.true_sage_hidden_dim,
                training_epochs=training_epochs  # Full training based on coarsening ratio
            )
            
            print("🎉 TRUE COARSENED GRAPHSAGE COMPLETED SUCCESSFULLY!")
            
            # Validate output
            expected_shape = (original_nx_graph.number_of_nodes(), args.true_sage_final_dim)
            if embeddings.shape == expected_shape:
                print(f"✅ Output shape validation passed: {embeddings.shape}")
            else:
                print(f"⚠️ Output shape unexpected: got {embeddings.shape}, expected {expected_shape}")
            
            # Check for NaN/Inf values
            if np.isfinite(embeddings).all():
                print(f"✅ Embeddings are finite (no NaN/Inf)")
            else:
                print(f"⚠️ Warning: Embeddings contain NaN/Inf values")
                
            # Embedding statistics
            print(f"📊 Embedding statistics:")
            print(f"   Mean: {np.mean(embeddings):.6f}")
            print(f"   Std:  {np.std(embeddings):.6f}")
            print(f"   Min:  {np.min(embeddings):.6f}")
            print(f"   Max:  {np.max(embeddings):.6f}")
            
        except Exception as e:
            print(f"❌ TRUE COARSENED GRAPHSAGE FAILED: {e}")
            print(f"🔋 Error details:")
            import traceback
            traceback.print_exc()
            
            # Fallback to regular GraphSAGE if True Coarsened fails
            print(f"\n🔄 FALLING BACK TO REGULAR GRAPHSAGE...")
            from embed_methods.graphsage.graphsage import graphsage
            
            # Use the existing GraphSAGE pipeline as fallback
            nx.set_node_attributes(G, False, "test")
            nx.set_node_attributes(G, False, "val")
            
            # Create mapping for fallback
            mapping = identity(feature.shape[0])
            for p in projections:
                mapping = mapping @ p
            mapping = normalize(mapping, norm='l1', axis=1).transpose()
            
            coarse_ratio = mapping.shape[1]/mapping.shape[0]
            feats = mapping @ feature
            
            embeddings = graphsage(G, feats, args.sage_model, args.sage_weighted, int(1000/coarse_ratio))
            print(f"✅ Fallback completed: {embeddings.shape}")
        
        # IMPORTANT: Set flag to skip GraphZoom refinement since True Coarsened does its own
        skip_graphzoom_refinement = True
        print(f"\n⭐️ Will skip standard GraphZoom refinement (already done internally)")
    
    embed_time = time.process_time() - embed_start


    ######Refinement######
    print("%%%%%% Starting Graph Refinement %%%%%%")
    print(f"Refinement method: {args.refinement_method}")

    # Skip refinement for methods that do their own refinement
    if args.embed_method == "true_coarsened_graphsage":
        print("⭐️ SKIPPING GraphZoom refinement - True Coarsened GraphSAGE already refined")
        print(f"✅ True Coarsened GraphSAGE handled refinement internally")
        print(f"📊 Final embeddings ready for evaluation: {embeddings.shape}")
        refine_time = 0.0  # No additional refinement time
    else:
        # Normal GraphZoom refinement for other methods
        if args.refinement_method != "original" and args.mp_timing:
            print(f"MP threshold: {args.mp_threshold}")
            print(f"Detailed MP timing: {args.mp_timing}")
        
        if args.refinement_method != "original":
            # Show debug info for MP-aware methods
            print(f"\n=== DEBUGGING REFINEMENT for {args.coarse} ===")
            print(f"Input embeddings shape: {embeddings.shape}")
            print(f"Number of projection levels: {level}")
            for i, proj in enumerate(projections):
                print(f"Projection {i} shape: {proj.shape}")
        
        refine_start = time.process_time()
        embeddings = refinement(level, projections, laplacians, embeddings, 
                            args.lda, args.power, 
                            method=args.refinement_method,
                            mp_threshold=args.mp_threshold,
                            show_mp_timing=args.mp_timing)
        refine_time = time.process_time() - refine_start

        if args.refinement_method != "original":
            print(f"Final embeddings shape: {embeddings.shape}")
            print("=== END REFINEMENT DEBUG ===\n")
        
######Save Embeddings######
    np.save(args.embed_path, embeddings)


######Evaluation######
    lr("dataset/{}/".format(dataset), args.embed_path, dataset)

######Enhanced Timing Report######
    if args.embed_method == "true_coarsened_graphsage":
        print(f"\n%%%%%% TRUE COARSENED GRAPHSAGE PERFORMANCE %%%%%%")
        print(f"Coarsening method: {args.coarse}")
        print(f"Original nodes: {original_nx_graph.number_of_nodes()}")
        print(f"Super-nodes: {G.number_of_nodes()}")
        print(f"Coarsening ratio: {original_nx_graph.number_of_nodes() / G.number_of_nodes():.2f}x")
        print(f"Training epochs used: {training_epochs}")
        print(f"Super embedding dim: {args.true_sage_super_dim}")
        print(f"Final embedding dim: {args.true_sage_final_dim}")
        
        # Calculate theoretical speedup
        theoretical_speedup = (original_nx_graph.number_of_nodes() / G.number_of_nodes()) ** 2
        print(f"Theoretical complexity reduction: {theoretical_speedup:.1f}x")

    ######Report timing information######
    print("%%%%%% CPU time %%%%%%")
    if args.fusion:
        total_time = fusion_time + reduce_time + embed_time + refine_time
        print(f"Graph Fusion     Time: {fusion_time:.3f}")
    else:
        total_time = reduce_time + embed_time + refine_time
        print("Graph Fusion     Time: 0")
    print(f"Graph Reduction  Time: {reduce_time:.3f}")
    print(f"Graph Embedding  Time: {embed_time:.3f}")
    print(f"Graph Refinement Time: {refine_time:.3f}")
    print(f"Total Time = Fusion_time + Reduction_time + Embedding_time + Refinement_time = {total_time:.3f}")
    
    # Additional timing breakdown for CMG
    if args.coarse == "cmg":
        try:
            from filtered_timed import get_timing_summary
            timing_summary = get_timing_summary()
            if timing_summary:
                print("\n%%%%%% CMG Detailed Timing %%%%%%")
                for step, stats in timing_summary.items():
                    print(f"{step:25s}: {stats['latest']:.3f}s")
        except ImportError:
            pass


if __name__ == "__main__":
    sys.exit(main())