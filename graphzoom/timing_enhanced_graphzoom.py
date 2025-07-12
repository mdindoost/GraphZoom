# Enhanced timing modifications for graphzoom.py
# Add these modifications to capture detailed timing as requested by Koutis

import time
import logging

# Add this timing decorator
def time_step(step_name):
    """Decorator to time individual steps"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"[TIMING] {step_name}: {duration:.3f} seconds")
            
            # Log to file for analysis
            with open("timing_log.txt", "a") as f:
                f.write(f"{step_name},{duration:.3f}\n")
            
            return result
        return wrapper
    return decorator

# Modifications to add to your CMG coarsening functions:

# In cmg_coarsening.py, enhance the cmg_coarse function:
def cmg_coarse(laplacian, level=1, k=10, d=20, threshold=0.1):
    """Enhanced with detailed timing"""
    print(f"[CMG TIMING] Starting CMG coarsening with k={k}, d={d}, threshold={threshold}")
    
    projections = []
    laplacians = []
    current_laplacian = laplacian.copy()
    
    # Track total CMG time
    total_cmg_start = time.time()
    
    for i in range(level):
        level_start = time.time()
        print(f"[CMG TIMING] Level {i+1} starting")
        
        # Store current Laplacian
        laplacians.append(current_laplacian.copy())
        
        # Step 1: Convert to PyG format
        step1_start = time.time()
        data = scipy_to_pyg_data(current_laplacian)
        step1_time = time.time() - step1_start
        print(f"[CMG TIMING]   Step 1 (scipy->pyg): {step1_time:.3f}s")
        
        # Step 2: Run CMG clustering (most important timing)
        step2_start = time.time()
        try:
            # Break down CMG internal timing
            spectral_start = time.time()
            clusters, nc, phi_stats, lambda_crit = cmg_filtered_clustering(
                data, k=k, d=d, threshold=threshold
            )
            spectral_time = time.time() - spectral_start
            
            print(f"[CMG TIMING]   Step 2a (spectral filtering): {spectral_time:.3f}s")
            print(f"[CMG TIMING]   Step 2b (found {nc} clusters)")
            
        except Exception as e:
            print(f"[CMG TIMING]   Step 2 FAILED: {e}")
            # Fallback timing
            fallback_start = time.time()
            from utils import smooth_filter, spec_coarsen
            filter_ = smooth_filter(current_laplacian, 0.1)
            current_laplacian, mapping = spec_coarsen(filter_, current_laplacian)
            fallback_time = time.time() - fallback_start
            print(f"[CMG TIMING]   Fallback coarsening: {fallback_time:.3f}s")
            projections.append(mapping)
            continue
        
        step2_time = time.time() - step2_start
        print(f"[CMG TIMING]   Step 2 (CMG clustering): {step2_time:.3f}s")
        
        # Step 3: Build projection matrix
        step3_start = time.time()
        num_nodes = current_laplacian.shape[0]
        row, col, data_vals = [], [], []
        
        for node_id in range(num_nodes):
            cluster_id = clusters[node_id]
            row.append(node_id)
            col.append(cluster_id)
            data_vals.append(1.0)
        
        mapping = csr_matrix((data_vals, (row, col)), shape=(num_nodes, nc))
        projections.append(mapping)
        step3_time = time.time() - step3_start
        print(f"[CMG TIMING]   Step 3 (build projection): {step3_time:.3f}s")
        
        # Step 4: Create coarsened Laplacian
        step4_start = time.time()
        current_laplacian = mapping.T @ current_laplacian @ mapping
        step4_time = time.time() - step4_start
        print(f"[CMG TIMING]   Step 4 (coarsen Laplacian): {step4_time:.3f}s")
        
        level_time = time.time() - level_start
        print(f"[CMG TIMING] Level {i+1} completed: {level_time:.3f}s total")
        print(f"[CMG TIMING]   Result: {nc} nodes, compression ratio: {num_nodes/nc:.1f}x")
    
    # Step 5: Final graph construction
    final_start = time.time()
    degree_diag = diags(current_laplacian.diagonal(), 0)
    adjacency = degree_diag - current_laplacian
    adjacency.data = np.abs(adjacency.data)
    G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='wgt')
    final_time = time.time() - final_start
    print(f"[CMG TIMING] Final graph construction: {final_time:.3f}s")
    
    total_cmg_time = time.time() - total_cmg_start
    print(f"[CMG TIMING] TOTAL CMG TIME: {total_cmg_time:.3f}s")
    
    return G, projections, laplacians, level

# Enhanced timing for GraphZoom main function:
def enhanced_main():
    """Add this timing enhancement to graphzoom.py main()"""
    
    # Add timing file initialization
    with open("timing_log.txt", "w") as f:
        f.write("step,time_seconds\n")
    
    # Add detailed timing around each major step:
    
    # STEP 1: Data Loading
    load_start = time.time()
    print("%%%%%% Loading Graph Data %%%%%%")
    laplacian = json2mtx(dataset)
    if args.fusion or args.embed_method == "graphsage":
        feature = np.load(feature_path)
    load_time = time.time() - load_start
    print(f"[TIMING] Data Loading: {load_time:.3f}s")
    
    # STEP 2: Graph Fusion (if enabled)
    if args.fusion:
        fusion_start = time.time()
        print("%%%%%% Starting Graph Fusion %%%%%%")
        
        # Break down fusion timing
        print(f"[FUSION TIMING] Original graph: {laplacian.shape[0]} nodes")
        
        laplacian = graph_fusion(laplacian, feature, args.num_neighs, args.mcr_dir, args.coarse,
                   fusion_input_path, args.search_ratio, reduce_results, mapping_path, dataset)
        
        fusion_time = time.time() - fusion_start
        print(f"[TIMING] Graph Fusion: {fusion_time:.3f}s")
        print(f"[FUSION TIMING] Fused graph: {laplacian.shape[0]} nodes")
    else:
        fusion_time = 0
        print("[TIMING] Graph Fusion: 0s (disabled)")
    
    # STEP 3: Graph Reduction (KEY TIMING FOR KOUTIS)
    reduction_start = time.time()
    print("%%%%%% Starting Graph Reduction %%%%%%")
    
    if args.coarse == "simple":
        simple_start = time.time()
        G, projections, laplacians, level = sim_coarse(laplacian, args.level)
        simple_time = time.time() - simple_start
        print(f"[TIMING] Simple coarsening: {simple_time:.3f}s")
        reduction_time = simple_time
        
    elif args.coarse == "lamg":
        lamg_start = time.time()
        os.system('./run_coarsening.sh {} {} {} n {}'.format(args.mcr_dir,
                coarsen_input_path, args.reduce_ratio, reduce_results))
        lamg_time = time.time() - lamg_start
        print(f"[TIMING] LAMG system call: {lamg_time:.3f}s")
        
        # Additional LAMG processing time
        post_start = time.time()
        reduction_time = read_time("{}CPUtime.txt".format(reduce_results))
        G = mtx2graph("{}Gs.mtx".format(reduce_results))
        level = read_levels("{}NumLevels.txt".format(reduce_results))
        projections, laplacians = construct_proj_laplacian(laplacian, level, reduce_results)
        post_time = time.time() - post_start
        print(f"[TIMING] LAMG post-processing: {post_time:.3f}s")
        
    elif args.coarse == "cmg":
        # CMG timing is handled inside cmg_coarse function
        G, projections, laplacians, level = cmg_coarse(
            laplacian, args.level, args.cmg_k, args.cmg_d, args.cmg_threshold
        )
        reduction_time = time.time() - reduction_start
    
    print(f"[TIMING] Graph Reduction TOTAL: {reduction_time:.3f}s")
    
    # STEP 4: Graph Embedding (KOUTIS SUSPECTS THIS DOMINATES)
    embedding_start = time.time()
    print("%%%%%% Starting Graph Embedding %%%%%%")
    
    print(f"[EMBEDDING TIMING] Coarsened graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    if args.embed_method == "deepwalk":
        dw_start = time.time()
        embeddings = deepwalk(G)
        dw_time = time.time() - dw_start
        print(f"[EMBEDDING TIMING] DeepWalk: {dw_time:.3f}s")
        
    elif args.embed_method == "node2vec":
        n2v_start = time.time()
        embeddings = node2vec(G)
        n2v_time = time.time() - n2v_start
        print(f"[EMBEDDING TIMING] Node2Vec: {n2v_time:.3f}s")
        
    # ... handle other embedding methods similarly
    
    embedding_time = time.time() - embedding_start
    print(f"[TIMING] Graph Embedding TOTAL: {embedding_time:.3f}s")
    
    # STEP 5: Refinement
    refinement_start = time.time()
    print("%%%%%% Starting Graph Refinement %%%%%%")
    embeddings = refinement(level, projections, laplacians, embeddings, args.lda, args.power)
    refinement_time = time.time() - refinement_start
    print(f"[TIMING] Graph Refinement: {refinement_time:.3f}s")
    
    # STEP 6: Evaluation
    eval_start = time.time()
    np.save(args.embed_path, embeddings)
    lr("dataset/{}/".format(dataset), args.embed_path, dataset)
    eval_time = time.time() - eval_start
    print(f"[TIMING] Evaluation: {eval_time:.3f}s")
    
    # FINAL TIMING REPORT (Enhanced for Koutis)
    total_time = fusion_time + reduction_time + embedding_time + refinement_time
    
    print("\n" + "="*60)
    print("📊 DETAILED TIMING BREAKDOWN FOR KOUTIS")
    print("="*60)
    print(f"Data Loading:      {load_time:>8.3f}s ({load_time/total_time*100:>5.1f}%)")
    print(f"Graph Fusion:      {fusion_time:>8.3f}s ({fusion_time/total_time*100:>5.1f}%)")
    print(f"Graph Reduction:   {reduction_time:>8.3f}s ({reduction_time/total_time*100:>5.1f}%)")
    print(f"Graph Embedding:   {embedding_time:>8.3f}s ({embedding_time/total_time*100:>5.1f}%)")
    print(f"Graph Refinement:  {refinement_time:>8.3f}s ({refinement_time/total_time*100:>5.1f}%)")
    print(f"Evaluation:        {eval_time:>8.3f}s ({eval_time/total_time*100:>5.1f}%)")
    print("-" * 60)
    print(f"TOTAL:             {total_time:>8.3f}s (100.0%)")
    
    # Koutis's specific questions
    if embedding_time > (fusion_time + reduction_time + refinement_time):
        print(f"\n💡 KOUTIS INSIGHT: Embedding DOMINATES runtime ({embedding_time/total_time*100:.1f}%)")
    else:
        print(f"\n💡 KOUTIS INSIGHT: Embedding does NOT dominate ({embedding_time/total_time*100:.1f}%)")
    
    print(f"🔍 Coarsening efficiency: {reduction_time:.3f}s to reduce graph size")
    print(f"⚡ Speedup potential: Focus on {'embedding' if embedding_time > reduction_time else 'coarsening'}")

# Add this to capture CMG internal timing in filtered.py:
def enhanced_cmg_filtered_clustering(data, k=10, d=20, threshold=0.1, conductance_method='both'):
    """Enhanced version with detailed timing"""
    
    timing_log = {}
    total_start = time.time()
    
    print("[CMG DETAILED TIMING] Starting CMG filtered clustering pipeline")
    
    # Step 1: Graph preparation
    step1_start = time.time()
    edge_index = data.edge_index.cpu().numpy()
    n = data.num_nodes
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=n).tocsr()
    timing_log['graph_prep'] = time.time() - step1_start
    print(f"[CMG TIMING] Graph preparation: {timing_log['graph_prep']:.3f}s")
    
    # Step 2: Build normalized Laplacian
    step2_start = time.time()
    L_norm = build_normalized_laplacian(A)
    timing_log['laplacian_build'] = time.time() - step2_start
    print(f"[CMG TIMING] Laplacian construction: {timing_log['laplacian_build']:.3f}s")
    
    # Step 3: Generate random vectors and filter
    step3_start = time.time()
    np.random.seed(42)
    X = np.random.randn(n, d)
    timing_log['random_vectors'] = time.time() - step3_start
    print(f"[CMG TIMING] Random vector generation: {timing_log['random_vectors']:.3f}s")
    
    # Step 4: Apply spectral filter (KEY STEP)
    step4_start = time.time()
    Y = apply_spectral_filter(X, L_norm, k)
    timing_log['spectral_filter'] = time.time() - step4_start
    print(f"[CMG TIMING] Spectral filtering (k={k}): {timing_log['spectral_filter']:.3f}s")
    
    # Step 5: Reweight graph
    step5_start = time.time()
    A_reweighted = reweight_graph_from_embeddings(Y, edge_index, threshold=threshold)
    timing_log['reweighting'] = time.time() - step5_start
    print(f"[CMG TIMING] Graph reweighting: {timing_log['reweighting']:.3f}s")
    
    # Step 6: Build Laplacian for CMG
    step6_start = time.time()
    degrees = np.array(A_reweighted.sum(axis=1)).flatten()
    L_reweighted = sp.diags(degrees) - A_reweighted
    timing_log['reweighted_laplacian'] = time.time() - step6_start
    print(f"[CMG TIMING] Reweighted Laplacian: {timing_log['reweighted_laplacian']:.3f}s")
    
    # Step 7: Run CMG clustering (CORE ALGORITHM)
    step7_start = time.time()
    try:
        cI_raw, nc = cmgCluster(L_reweighted.tocsc())
        cI = cI_raw - 1
        timing_log['cmg_cluster'] = time.time() - step7_start
        print(f"[CMG TIMING] Core CMG clustering: {timing_log['cmg_cluster']:.3f}s")
        print(f"[CMG TIMING] Found {nc} clusters")
    except Exception as e:
        timing_log['cmg_cluster'] = time.time() - step7_start
        print(f"[CMG TIMING] CMG clustering FAILED: {e}")
        cI = np.zeros(n, dtype=int)
        nc = 1
    
    # Step 8: Evaluate conductance
    step8_start = time.time()
    phi_stats = evaluate_phi_conductance(data, cI)
    timing_log['conductance_eval'] = time.time() - step8_start
    print(f"[CMG TIMING] Conductance evaluation: {timing_log['conductance_eval']:.3f}s")
    
    timing_log['total'] = time.time() - total_start
    
    print(f"[CMG TIMING] TOTAL CMG PIPELINE: {timing_log['total']:.3f}s")
    print(f"[CMG TIMING] Breakdown: Filter={timing_log['spectral_filter']:.1f}s, "
          f"Reweight={timing_log['reweighting']:.1f}s, "
          f"Cluster={timing_log['cmg_cluster']:.1f}s")
    
    return cI, nc, phi_stats, compute_lambda_critical(k)
