#!/usr/bin/env python3
"""
Quick fix and retest script for dimension mismatch issue
"""

def test_dimension_fix():
    """Test that the dimension fix works correctly"""
    
    print("🔧 TESTING DIMENSION FIX")
    print("="*50)
    
    import numpy as np
    import networkx as nx
    import sys
    import os
    
    # Add path for imports
    sys.path.append('.')
    sys.path.append('./embed_methods/graphsage/')
    
    try:
        # Create small test graph
        G = nx.karate_club_graph()
        n_nodes = G.number_of_nodes()
        
        # Create test features
        features = np.random.randn(n_nodes, 16)
        
        # Create simple clusters (divide into 3 groups)
        clusters = [
            list(range(0, 11)),      # First group
            list(range(11, 22)),     # Second group  
            list(range(22, n_nodes)) # Remaining nodes
        ]
        
        print(f"Test graph: {n_nodes} nodes")
        print(f"Features shape: {features.shape}")
        print(f"Clusters: {len(clusters)} clusters")
        
        # Create simple coarsened graph
        coarsened_G = nx.Graph()
        coarsened_G.add_nodes_from(range(len(clusters)))
        coarsened_G.add_edges_from([(0,1), (1,2)])
        
        # Create dummy projections and laplacians
        import scipy.sparse as sp
        projection = sp.lil_matrix((n_nodes, len(clusters)))
        for node_id in range(n_nodes):
            for cluster_id, cluster in enumerate(clusters):
                if node_id in cluster:
                    projection[node_id, cluster_id] = 1.0
        
        projections = [projection.tocsr()]
        laplacians = [nx.laplacian_matrix(G)]
        
        # Test different target dimensions
        target_dims = [8, 16, 32, 64]
        
        for target_dim in target_dims:
            print(f"\n🎯 Testing target dimension: {target_dim}")
            
            # Import enhanced version
            from enhanced_true_coarsened_graphsage import enhanced_true_coarsened_graphsage
            
            embeddings = enhanced_true_coarsened_graphsage(
                original_graph=G,
                features=features,
                clusters=clusters,
                coarsened_graph=coarsened_G,
                projections=projections,
                laplacians=laplacians,
                super_embed_dim=64,  # Can be different from final
                final_embed_dim=target_dim,  # This should control output
                hidden_dim=8,
                base_epochs=50,  # Small for testing
                computational_strategy="fair_comparison",
                training_distribution="balanced"
            )
            
            print(f"  Expected shape: ({n_nodes}, {target_dim})")
            print(f"  Actual shape:   {embeddings.shape}")
            
            # Validate
            expected_shape = (n_nodes, target_dim)
            if embeddings.shape == expected_shape:
                print(f"  ✅ DIMENSION MATCH!")
            else:
                print(f"  ❌ DIMENSION MISMATCH!")
                return False
            
            # Check finite values
            if np.isfinite(embeddings).all():
                print(f"  ✅ All finite values")
            else:
                print(f"  ❌ Contains NaN/Inf")
                return False
            
            # Check diversity
            unique_embeddings = []
            for embedding in embeddings:
                is_unique = True
                for unique_emb in unique_embeddings:
                    if np.allclose(embedding, unique_emb, atol=1e-6):
                        is_unique = False
                        break
                if is_unique:
                    unique_embeddings.append(embedding)
            
            diversity = len(unique_embeddings) / len(embeddings)
            print(f"  Diversity: {diversity:.3f}")
            
            if diversity < 0.8:
                print(f"  ⚠️ Low diversity warning")
        
        print(f"\n✅ ALL DIMENSION TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Dimension fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 QUICK DIMENSION FIX TEST")
    print("="*60)
    
    success = test_dimension_fix()
    
    if success:
        print(f"\n🎉 DIMENSION FIX SUCCESSFUL!")
        print(f"💡 The enhanced version now correctly respects final_embed_dim parameter")
        print(f"🚀 Ready to run full tests and integration!")
    else:
        print(f"\n❌ DIMENSION FIX FAILED")
        print(f"💡 Check the enhanced_true_coarsened_graphsage.py implementation")
