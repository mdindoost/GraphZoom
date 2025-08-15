#!/usr/bin/env python3
"""
Changes needed to integrate MP-aware refinement into graphzoom_timed.py
Only showing the parts that need modification
"""

# ==== CHANGE 1: Add to argument parsing (around line 76) ====
# Add this after the existing arguments, before args = parser.parse_args():

    # MP-Aware Refinement parameters
    parser.add_argument("--refinement_method", type=str, default="original", 
                       choices=["original", "mp_aware", "mp_aware+original"],
                       help="Refinement method: original (default), mp_aware, or mp_aware+original")
    parser.add_argument("--mp_threshold", type=float, default=0.1,
                       help="MP error threshold for applying correction (only for mp_aware)")
    parser.add_argument("--mp_timing", action="store_true",
                       help="Show detailed MP refinement timing breakdown")

# ==== CHANGE 2: Replace the refinement function (around line 45) ====
# Replace the existing refinement function with this enhanced version:

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

# ==== CHANGE 3: Update the refinement call (around line 191) ====
# Replace the existing refinement section with:

######Refinement######
    print("%%%%%% Starting Graph Refinement %%%%%%")
    print(f"Refinement method: {args.refinement_method}")
    
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

# ==== CHANGE 4: Update timing report (around line 213) ====
# In the timing report section, add this after the existing timing printout:

    # Additional timing breakdown for MP-aware refinement
    if args.refinement_method != "original" and args.mp_timing:
        print(f"\n%%%%%% MP-Aware Refinement Timing %%%%%%")
        print(f"Refinement method: {args.refinement_method}")
        print(f"MP threshold used: {args.mp_threshold}")

# ==== USAGE EXAMPLES ====
"""
# Original behavior (unchanged):
python graphzoom_timed.py --coarse cmg --dataset cora

# Use only MP-aware refinement:
python graphzoom_timed.py --coarse cmg --dataset cora --refinement_method mp_aware

# Use MP-aware + original refinement:
python graphzoom_timed.py --coarse cmg --dataset cora --refinement_method mp_aware+original

# With detailed timing:
python graphzoom_timed.py --coarse cmg --dataset cora --refinement_method mp_aware --mp_timing

# With custom MP threshold:
python graphzoom_timed.py --coarse cmg --dataset cora --refinement_method mp_aware --mp_threshold 0.05
"""